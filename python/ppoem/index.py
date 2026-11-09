"""
Project index (lazy, memory-bounded)
====================================
Very large repositories (vllm, pytorch/benchmark, ...) have thousands of
Python files. Parsing all of them up-front costs minutes and gigabytes, so the
index works lazily:

  * all .py files are read as text once (cheap);
  * a file is parsed with the AST package only when it is needed: it may
    contain a model-loading call, or the resolution of a variable reaches it
    (import, call site, argparse option, ...);
  * call sites, argparse options and function definitions are located by a
    fast text search first, and only the matching files are parsed;
  * parsed modules are kept in an LRU cache; when it is full, the least
    recently used modules are released at a safe point (between two model
    calls), together with every cache that refers to them.

Per parsed module the index records imports (relative imports made absolute),
assignments per scope (incl. with/for targets), top-level functions and
classes (methods, attributes set in the class body or via self.x = ...,
annotations, base classes).
"""

from __future__ import annotations

import ast
import re
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from .common import SETTINGS, Settings, is_test_path, iter_python_files


@dataclass
class ClassInfo:
    name: str
    module: "ModuleInfo"
    node: ast.ClassDef
    methods: Dict[str, ast.AST] = field(default_factory=dict)
    attrs: Dict[str, List[Tuple[ast.AST, Optional[ast.AST]]]] = field(default_factory=lambda: defaultdict(list))
    annotations: Dict[str, ast.AST] = field(default_factory=dict)


@dataclass
class ModuleInfo:
    dotted: str
    rel: str
    tree: ast.Module
    is_test: bool
    parents: Dict[ast.AST, ast.AST] = field(default_factory=dict)
    imports: Dict[str, Tuple[str, Optional[str]]] = field(default_factory=dict)
    functions: Dict[str, ast.AST] = field(default_factory=dict)
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    assigns: Dict[Tuple[object, str], list] = field(default_factory=lambda: defaultdict(list))
    calls: List[ast.Call] = field(default_factory=list)


FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def scope_key(m: ModuleInfo, node) -> object:
    """0 = module level, id(function) = function scope, ("class", id) = class body."""
    cur = node
    while cur in m.parents:
        cur = m.parents[cur]
        if isinstance(cur, FUNC_TYPES):
            return id(cur)
        if isinstance(cur, ast.ClassDef):
            return ("class", id(cur))
    return 0


def assign_pairs(node):
    """(target, value) pairs of an assignment-like statement."""
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, (ast.Tuple, ast.List)) and isinstance(node.value, (ast.Tuple, ast.List)) \
                    and len(t.elts) == len(node.value.elts):
                yield from zip(t.elts, node.value.elts)
            else:
                yield t, node.value
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        yield node.target, node.value
    elif isinstance(node, ast.NamedExpr):
        yield node.target, node.value


def _is_self_attr(node) -> bool:
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("self", "cls")


def _argparse_dests(call: ast.Call) -> List[str]:
    for k in call.keywords:
        if k.arg == "dest" and isinstance(k.value, ast.Constant):
            return [str(k.value.value)]
    flags = [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    longs = [f for f in flags if f.startswith("--")]
    chosen = longs[:1] or flags[:1]
    return [f.lstrip("-").replace("-", "_") for f in chosen]


class ProjectIndex:
    def __init__(self, root: Path, settings: Settings = SETTINGS, max_parsed: Optional[int] = None):
        self.root = Path(root)
        self.settings = settings
        self.max_parsed = max_parsed or getattr(settings, "max_parsed_modules", 600)
        self.text: Dict[str, str] = {}
        self.dotted_of: Dict[str, str] = {}
        self.rel_of: Dict[str, str] = {}
        self.parse_errors: List[str] = []
        self._cache: "OrderedDict[str, ModuleInfo]" = OrderedDict()
        self._failed: Set[str] = set()
        self._calls: Dict[str, List[Tuple[ModuleInfo, ast.Call]]] = {}
        self._argparse: Dict[str, List[Tuple[ModuleInfo, Optional[ast.AST]]]] = {}
        self._def_count: Dict[str, int] = {}
        self.on_release = []                      # callbacks run when modules are released
        self.parsed_total = 0
        self._called_in: Optional[Dict[str, Set[str]]] = None    # name -> files calling name(...)
        self._defined: Optional[Dict[str, int]] = None           # name -> number of definitions
        self._suffix_map: Optional[Dict[str, Optional[str]]] = None
        for path, rel in iter_python_files(self.root, settings):
            try:
                self.text[rel] = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            parts = list(Path(rel).parts)
            parts[-1] = parts[-1][:-3]
            if parts[-1] == "__init__":
                parts = parts[:-1]
            dotted = ".".join(parts) or "__init__"
            self.dotted_of[rel] = dotted
            self.rel_of.setdefault(dotted, rel)

    # ── parsing / cache ──────────────────────────────────────────────────────
    @property
    def n_files(self) -> int:
        return len(self.text)

    def module(self, rel: str) -> Optional[ModuleInfo]:
        m = self._cache.get(rel)
        if m is not None:
            self._cache.move_to_end(rel)
            return m
        if rel in self._failed or rel not in self.text:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(self.text[rel])
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            self._failed.add(rel)
            self.parse_errors.append(rel)
            return None
        m = ModuleInfo(self.dotted_of[rel], rel, tree, is_test_path(rel))
        self._index_module(m)
        self._cache[rel] = m
        self.parsed_total += 1
        return m

    def checkpoint(self, keep: Tuple[str, ...] = ()) -> None:
        """Release least recently used modules when the cache is too big
        (call only between two independent resolutions)."""
        if len(self._cache) <= self.max_parsed:
            return
        for rel in list(self._cache):
            if len(self._cache) <= self.max_parsed // 2:
                break
            if rel not in keep:
                del self._cache[rel]
        self._calls.clear()
        self._argparse.clear()
        for cb in self.on_release:
            cb()

    def _index_module(self, m: ModuleInfo):
        for parent in ast.walk(m.tree):
            for child in ast.iter_child_nodes(parent):
                m.parents[child] = parent
        pkg = m.dotted.split(".") if m.rel.endswith("__init__.py") else m.dotted.split(".")[:-1]
        for node in ast.walk(m.tree):
            if isinstance(node, ast.Call):
                m.calls.append(node)
                continue
            pairs = list(assign_pairs(node))
            if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                pairs.append((node.target, node.iter))
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                pairs += [(i.optional_vars, i.context_expr) for i in node.items if i.optional_vars is not None]
            names = [(t.id, v) for t, v in pairs if isinstance(t, ast.Name)]
            if names:
                key = scope_key(m, node)
                line = getattr(node, "lineno", None) or getattr(getattr(node, "iter", None), "lineno", 0)
                for n, v in names:
                    m.assigns[(key, n)].append((line, node, v))
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        m.imports[a.asname] = (a.name, None)
                    else:
                        root = a.name.split(".")[0]
                        m.imports.setdefault(root, (root, None))
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    up = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                    base = ".".join([p for p in up if p] + ([node.module] if node.module else []))
                for a in node.names:
                    if a.name != "*":
                        m.imports[a.asname or a.name] = (base, a.name)
            elif isinstance(node, ast.ClassDef) and node.name not in m.classes:
                m.classes[node.name] = self._index_class(m, node)
        for node in m.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                m.functions[node.name] = node

    def _index_class(self, m: ModuleInfo, node: ast.ClassDef) -> ClassInfo:
        ci = ClassInfo(node.name, m, node)
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ci.methods[item.name] = item
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                ci.annotations[item.target.id] = item.annotation
            for tgt, val in assign_pairs(item):
                if isinstance(tgt, ast.Name):
                    ci.attrs[tgt.id].append((val, None))
        for meth in ci.methods.values():
            for sub in ast.walk(meth):
                if isinstance(sub, ast.AnnAssign) and _is_self_attr(sub.target):
                    ci.annotations.setdefault(sub.target.attr, sub.annotation)
                for tgt, val in assign_pairs(sub):
                    if _is_self_attr(tgt):
                        ci.attrs[tgt.attr].append((val, meth))
        return ci

    # ── text search (no parsing) ─────────────────────────────────────────────
    def files_matching(self, pattern: "re.Pattern", production_only: bool = True) -> Iterator[str]:
        for rel, text in self.text.items():
            if production_only and is_test_path(rel):
                continue
            if pattern.search(text):
                yield rel

    _CALL_RX = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    _DEF_RX = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)

    def _build_word_index(self):
        self._called_in, self._defined = defaultdict(set), defaultdict(int)
        for rel, text in self.text.items():
            for name in self._DEF_RX.findall(text):
                self._defined[name] += 1
            if is_test_path(rel):
                continue
            for name in set(self._CALL_RX.findall(text)):
                self._called_in[name].add(rel)

    def calls_to(self, name: str) -> List[Tuple[ModuleInfo, ast.Call]]:
        """Call sites `name(...)` / `x.name(...)` in production code."""
        if name in self._calls:
            return self._calls[name]
        if self._called_in is None:
            self._build_word_index()
        out = []
        for rel in sorted(self._called_in.get(name, ())):
            m = self.module(rel)
            if m is None:
                continue
            for c in m.calls:
                f = c.func
                if (isinstance(f, ast.Name) and f.id == name) or (isinstance(f, ast.Attribute) and f.attr == name):
                    out.append((m, c))
        self._calls[name] = out
        return out

    def argparse_options(self, dest: str) -> List[Tuple[ModuleInfo, Optional[ast.AST]]]:
        if dest in self._argparse:
            return self._argparse[dest]
        flag = re.escape(dest).replace("_", "[-_]")
        rx = re.compile(r"add_argument\s*\([^)]*?['\"](?:--?)?" + flag + r"['\"]", re.S)
        out = []
        for rel in self.files_matching(rx):
            m = self.module(rel)
            if m is None:
                continue
            for c in m.calls:
                name = c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", None)
                if name == "add_argument" and dest in _argparse_dests(c):
                    out.append((m, next((k.value for k in c.keywords if k.arg == "default"), None)))
        self._argparse[dest] = out
        return out

    def definition_count(self, name: str) -> int:
        """How many functions / methods named `name` the project defines."""
        if self._defined is None:
            self._build_word_index()
        return self._defined.get(name, 0)

    # ── module / symbol lookup ───────────────────────────────────────────────
    def _rel_for_dotted(self, dotted: str) -> Optional[str]:
        if not dotted:
            return None
        rel = self.rel_of.get(dotted)
        if rel is not None:
            return rel
        if self._suffix_map is None:
            # source roots (src/, lib/, python/, ...): leading folders that are NOT
            # packages may be dropped, e.g. src/pkg/mod.py is importable as pkg.mod.
            # A package folder is never dropped, so vllm/.../torch.py is not "torch".
            self._suffix_map = {}
            for d, r in self.rel_of.items():
                parts = d.split(".")
                for i in range(1, len(parts)):
                    if "/".join(parts[:i]) + "/__init__.py" in self.text:
                        break
                    key = ".".join(parts[i:])
                    self._suffix_map[key] = None if key in self._suffix_map else r
        return self._suffix_map.get(dotted)

    def find_module(self, dotted: str) -> Optional[ModuleInfo]:
        rel = self._rel_for_dotted(dotted)
        return self.module(rel) if rel else None

    def is_project_module(self, dotted: str) -> bool:
        return self._rel_for_dotted(dotted) is not None or self._rel_for_dotted(dotted.split(".")[0]) is not None

    def lookup_symbol(self, module_name: str, symbol: Optional[str], _depth: int = 0):
        """What `from module_name import symbol` refers to inside the project:
        ('module', m) | ('func', m, node) | ('class', ClassInfo) | ('var', m) | None."""
        if _depth > 8:
            return None
        if symbol is None:
            m = self.find_module(module_name)
            return ("module", m) if m else None
        sub = self.find_module(f"{module_name}.{symbol}" if module_name else symbol)
        if sub:
            return ("module", sub)
        m = self.find_module(module_name)
        if not m:
            return None
        if symbol in m.functions:
            return ("func", m, m.functions[symbol])
        if symbol in m.classes:
            return ("class", m.classes[symbol])
        if symbol in m.imports:
            mod, name = m.imports[symbol]
            if (mod, name) != (module_name, symbol):
                hit = self.lookup_symbol(mod, name, _depth + 1)
                if hit:
                    return hit
        return ("var", m)

    def imported_project_modules(self, m: ModuleInfo) -> Set[str]:
        out = set()
        for mod, sym in m.imports.values():
            for cand in ((f"{mod}.{sym}" if sym else mod), mod):
                rel = self._rel_for_dotted(cand)
                if rel:
                    out.add(self.dotted_of[rel])
                    break
        return out

    def linked(self, a: ModuleInfo, b: ModuleInfo) -> bool:
        """True when one module imports the other (or they are the same)."""
        return a.rel == b.rel or b.dotted in self.imported_project_modules(a) \
            or a.dotted in self.imported_project_modules(b)
