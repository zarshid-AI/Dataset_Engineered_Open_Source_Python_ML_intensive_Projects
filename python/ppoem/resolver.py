"""
Project-wide static resolution of model arguments
=================================================
Starting from the expression passed as the model argument of a model-loading
call, the resolver follows its definition across the files of the project:

  * aliases to other variables, string concatenation, f-strings, %-formatting,
    str.format(), str methods (split()[i], replace(), strip(), ...)
  * path construction: os.path.join, pathlib Path(...) / "x", .parent, __file__
  * constants imported from other modules (also re-exports in __init__.py)
  * class attributes assigned in the class body or in any method (self.x),
    base classes, @property return values, and constructor keyword arguments of
    classes without __init__ (dataclasses / pydantic / attrs models)
  * dictionary entries with constant keys, dict.get(key, default)
  * return values of project functions
  * function parameters: the default value and the argument passed at every
    call site in production code that is *linked* to the function (imported,
    same module, self/cls, an instance of the class, ...). Calls that cannot be
    linked to the function are ignored, so no unrelated value is reported.
  * both branches of `a if c else b`, every operand of `a or b`

Resolution stops when the value only exists at run time:
  dynamic_cli     args.x, sys.argv             (argparse default reported)
  dynamic_env     os.getenv / os.environ       (default value reported)
  dynamic_config  cfg.x, config["x"]
  dynamic_file    json.load, open().read, glob, listdir
  dynamic_input   input()
  external        value returned by a call into another library
  unresolved      anything else (reason recorded)

Every result is a `Cand` (candidate) carrying the value (or None), its kind,
the file and line where the value is written and the chain of steps followed.
"""

from __future__ import annotations

import ast
import itertools
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .common import SETTINGS, Settings, src
from .index import FUNC_TYPES, ClassInfo, ModuleInfo, ProjectIndex
from .rules import library_of

CLI_ROOTS = {"args", "opt", "opts", "options", "arguments", "parsed_args", "cli_args",
             "cmd_args", "flags", "FLAGS", "argv"}
CONFIG_ROOTS = {"cfg", "config", "conf", "configs", "hparams", "hps", "hp", "params",
                "settings", "model_config", "model_cfg", "train_config", "training_args",
                "model_args", "data_args", "omegaconf", "hydra"}
FILE_READERS = {"load", "loads", "safe_load", "read", "read_text", "readline", "readlines",
                "from_file", "from_json", "from_yaml", "read_csv", "read_json", "load_config"}
OPENERS = {"open", "File", "safe_open", "gzip.open", "fsspec.open", "smart_open.open", "io.open",
           "bz2.open", "lzma.open", "tarfile.open", "zipfile.ZipFile"}
FS_LISTERS = {"glob", "iglob", "rglob", "listdir", "scandir", "walk", "iterdir"}
PATH_JOIN = {"os.path.join", "osp.join", "path.join", "posixpath.join", "ntpath.join", "os.path.normpath"}
PATH_CTORS = {"Path", "pathlib.Path", "PurePath", "pathlib.PurePath", "PosixPath", "pathlib.PosixPath"}
STR_UNARY = {"strip": str.strip, "lstrip": str.lstrip, "rstrip": str.rstrip, "lower": str.lower,
             "upper": str.upper, "casefold": str.casefold, "title": str.title}
# a dictionary key that names a model artifact (Note 4: weights['detection_weights.pt'])
MODEL_ARTIFACT_RE = re.compile(
    r"(\.(pt|pth|bin|safetensors|ckpt|onnx|h5|hdf5|keras|pb|tflite|gguf|ggml|pkl|joblib|"
    r"model|engine|plan|ubj|xgb|msgpack|params|mlmodel|torchscript|ptl)$)", re.IGNORECASE)

VALUE_KINDS = ("literal", "default_param", "env_default", "cli_default", "subscript_key")


@dataclass(frozen=True)
class Cand:
    value: Optional[str]           # resolved string or None
    kind: str                      # see module docstring
    source: str = ""               # expression where resolution stopped (value None)
    file: str = ""                 # where the value is written
    line: Optional[int] = None
    trace: Tuple[str, ...] = ()

    def step(self, s: str) -> "Cand":
        return Cand(self.value, self.kind, self.source, self.file, self.line, (s,) + self.trace)

    def with_kind(self, kind: str, note: str = "") -> "Cand":
        return Cand(self.value, kind, self.source, self.file, self.line,
                    ((note,) if note else ()) + self.trace)


@dataclass
class Ctx:
    module: ModuleInfo
    func: Optional[ast.AST] = None
    cls: Optional[ClassInfo] = None
    line: int = 10 ** 9
    # parameters of `func` bound to the arguments of the call that led here:
    # name -> (argument expression, caller context) or None (not passed: default only)
    bindings: Optional[dict] = None

    def at(self, line: int) -> "Ctx":
        return Ctx(self.module, self.func, self.cls, line, self.bindings)


_WEAKNESS = {"literal": 0, "default_param": 1, "subscript_key": 2, "cli_default": 3, "env_default": 3}


def _combine(parts: List[Cand], joiner, where: Tuple[str, int], label: str) -> Cand:
    if all(p.value is not None for p in parts):
        kind = max((p.kind for p in parts), key=lambda k: _WEAKNESS.get(k, 9))
        try:
            value = joiner([p.value for p in parts])
        except Exception:
            return Cand(None, "unresolved", label, trace=(f"{label}: cannot combine",))
        return Cand(value, kind, "", where[0], where[1], (f"{label} @ {where[0]}:{where[1]}",))
    bad = next(p for p in parts if p.value is None)
    return Cand(None, bad.kind, bad.source, trace=(f"{label} @ {where[0]}:{where[1]}",) + bad.trace)


def _dedup(cands: List[Cand]) -> List[Cand]:
    seen, out = set(), []
    for c in cands:
        k = (c.value, c.kind, c.source, c.file, c.line)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _last_name(f) -> str:
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


def _root_name(node) -> Optional[str]:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _chain_names(node) -> set:
    names = set()
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.add(node.id)
    return names


def _params(func) -> List[str]:
    a = func.args
    return [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs]


def _param_default(func, name):
    a = func.args
    pos = a.posonlyargs + a.args
    for p, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
        if p.arg == name:
            return d
    for p, d in zip(a.kwonlyargs, a.kw_defaults):
        if p.arg == name and d is not None:
            return d
    return None


def _param_annotation(func, name):
    a = func.args
    for p in a.posonlyargs + a.args + a.kwonlyargs:
        if p.arg == name:
            return p.annotation
    return None


def _is_property(func) -> bool:
    return any(src(d) in ("property", "cached_property", "functools.cached_property") for d in func.decorator_list)


class Resolver:
    def __init__(self, index: ProjectIndex, settings: Settings = SETTINGS):
        self.ix = index
        self.s = settings
        self._steps = 0
        self._qual_cache: Dict[object, object] = {}
        index.on_release.append(self._qual_cache.clear)

    # ══════════════════════════════════════════════════════════════════════════
    # public API
    # ══════════════════════════════════════════════════════════════════════════
    def context_of(self, m: ModuleInfo, node) -> Ctx:
        func, cls = self.enclosing(m, node)
        return Ctx(m, func, cls, getattr(node, "lineno", 10 ** 9))

    def resolve_expr(self, node, ctx: Ctx) -> List[Cand]:
        """Resolve one model argument (resets the work budget)."""
        self._steps = 0
        return self.resolve(node, ctx)

    # ══════════════════════════════════════════════════════════════════════════
    # scopes
    # ══════════════════════════════════════════════════════════════════════════
    def enclosing(self, m: ModuleInfo, node) -> Tuple[Optional[ast.AST], Optional[ClassInfo]]:
        func, cls, cur = None, None, node
        while cur in m.parents:
            cur = m.parents[cur]
            if func is None and isinstance(cur, FUNC_TYPES):
                func = cur
            if isinstance(cur, ast.ClassDef):
                cls = m.classes.get(cur.name)
                break
        return func, cls

    def scope_assignments(self, m: ModuleInfo, scope, name: str, before: int, use=None):
        """Assignments to `name` in `scope` (function or module) before line `before`,
        latest first, up to the first one that always runs before the use.
        With `use` (the node where the name is read), assignments in the other
        branch of an if/else that contains the use are skipped."""
        owner = scope if scope is not None else m.tree
        found = [x for x in m.assigns.get((id(scope) if scope is not None else 0, name), []) if x[0] < before]
        found.sort(key=lambda x: x[0], reverse=True)
        use_blocks = self._blocks(m, use) if use is not None else None
        out = []
        for line, stmt, val in found:
            if use_blocks is not None and self._other_branch(m, stmt, use_blocks):
                continue
            out.append((line, val, stmt))
            if use_blocks is not None:
                if self._block_of(m, stmt) in use_blocks:      # same or enclosing block: always runs
                    break
            elif m.parents.get(stmt) is owner:
                break
        return out

    _BLOCK_FIELDS = ("body", "orelse", "finalbody")

    def _block_of(self, m: ModuleInfo, stmt):
        p = m.parents.get(stmt)
        if p is None:
            return None
        for f in self._BLOCK_FIELDS:
            lst = getattr(p, f, None)
            if isinstance(lst, list) and any(x is stmt for x in lst):
                return (id(p), f)
        return (id(p), "?")

    def _blocks(self, m: ModuleInfo, node) -> set:
        key = ("blocks", id(node))
        cached = self._qual_cache.get(key)
        if cached is not None:
            return cached
        out, cur = set(), node
        while cur in m.parents:
            b = self._block_of(m, cur)
            if b is not None:
                out.add(b)
            cur = m.parents[cur]
        out.add((id(m.tree), "body"))
        self._qual_cache[key] = out
        return out

    def _other_branch(self, m: ModuleInfo, stmt, use_blocks: set) -> bool:
        cur = stmt
        while cur in m.parents:
            p = m.parents[cur]
            if isinstance(p, ast.If):
                mine = "body" if any(x is cur for x in p.body) else "orelse" if any(x is cur for x in p.orelse) else None
                other = {"body": "orelse", "orelse": "body"}.get(mine)
                if other and (id(p), other) in use_blocks:
                    return True
            cur = p
        return False

    # ══════════════════════════════════════════════════════════════════════════
    # resolution
    # ══════════════════════════════════════════════════════════════════════════
    def resolve(self, node, ctx: Ctx, depth: int = 0, seen=frozenset()) -> List[Cand]:
        if node is None:
            return [Cand(None, "unresolved", "", trace=("no expression",))]
        self._steps += 1
        if self._steps > self.s.max_steps:
            return [Cand(None, "unresolved", src(node, 120), trace=("resolution budget exhausted",))]
        if depth > self.s.max_depth:
            return [Cand(None, "unresolved", src(node, 120), trace=("max depth",))]
        key = (id(node), ctx.module.rel, id(ctx.func), ctx.line)
        if key in seen:
            return [Cand(None, "unresolved", src(node, 120), trace=("cycle",))]
        try:
            out = self._resolve(node, ctx, depth, seen | {key})
        except RecursionError:
            out = [Cand(None, "unresolved", src(node, 120), trace=("recursion",))]
        return _dedup(out)[: self.s.max_candidates]

    def _r(self, node, ctx, depth, seen):
        return self.resolve(node, ctx, depth + 1, seen)

    def _where(self, node, ctx) -> Tuple[str, int]:
        return ctx.module.rel, getattr(node, "lineno", ctx.line)

    def _resolve(self, node, ctx: Ctx, depth, seen) -> List[Cand]:
        R = lambda n, c=ctx: self._r(n, c, depth, seen)
        rel, line = self._where(node, ctx)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return [Cand(node.value, "literal", "", rel, line, (f"literal @ {rel}:{line}",))]
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return [Cand(str(node.value), "literal", "", rel, line, (f"number @ {rel}:{line}",))]
            return [Cand(None, "unresolved", src(node), trace=(f"{node.value!r} is not a string",))]

        if isinstance(node, ast.Await):
            return R(node.value)
        if isinstance(node, ast.IfExp):
            return R(node.body) + R(node.orelse)
        if isinstance(node, ast.BoolOp):
            vals = node.values if isinstance(node.op, ast.Or) else node.values[-1:]
            return [c for v in vals for c in R(v)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return [_combine([a, b], "".join, (rel, line), "concatenation")
                    for a, b in itertools.islice(itertools.product(R(node.left), R(node.right)), self.s.max_candidates)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):          # pathlib  a / "b"
            return [_combine([a, b], lambda xs: xs[0].rstrip("/") + "/" + xs[1].lstrip("/"), (rel, line), "path /")
                    for a, b in itertools.islice(itertools.product(R(node.left), R(node.right)), self.s.max_candidates)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Constant) \
                and isinstance(node.left.value, str):                                 # "%s/x" % a
            items = node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
            parts = [R(i) for i in items]
            fmt = node.left.value
            return [_combine(list(p), lambda xs: fmt % tuple(xs), (rel, line), "%-format")
                    for p in itertools.islice(itertools.product(*parts), self.s.max_candidates)]
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts.append([Cand(str(v.value), "literal", "", rel, line)])
                elif isinstance(v, ast.FormattedValue):
                    parts.append(R(v.value))
            return [_combine(list(p), "".join, (rel, line), "f-string")
                    for p in itertools.islice(itertools.product(*parts), self.s.max_candidates)]
        if isinstance(node, ast.Name):
            return self._name(node, ctx, depth, seen)
        if isinstance(node, ast.Attribute):
            return self._attribute(node, ctx, depth, seen)
        if isinstance(node, ast.Subscript):
            return self._subscript(node, ctx, depth, seen)
        if isinstance(node, ast.Call):
            return self._call(node, ctx, depth, seen)
        if isinstance(node, ast.Starred):
            return R(node.value)
        return [Cand(None, "unresolved", src(node, 120), trace=(f"{type(node).__name__} @ {rel}:{line}",))]

    # ── names ────────────────────────────────────────────────────────────────
    def _name(self, node: ast.Name, ctx: Ctx, depth, seen) -> List[Cand]:
        name, m = node.id, ctx.module
        if name == "__file__":
            return [Cand(m.rel, "literal", "", m.rel, getattr(node, "lineno", None), (f"__file__ = {m.rel}",))]
        scope = ctx.func
        line = ctx.line
        while scope is not None:
            sctx = ctx.at(line) if scope is ctx.func else Ctx(m, scope, ctx.cls, line)
            assigns = self.scope_assignments(m, scope, name, line, use=node if scope is ctx.func else None)
            if assigns:
                return self._from_assigns(name, assigns, sctx, depth, seen)
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) and name in _params(scope):
                if scope is ctx.func and ctx.bindings is not None and name in ctx.bindings:
                    return self._bound_parameter(name, scope, ctx, depth, seen)
                return self._parameter(name, scope, sctx, depth, seen)
            if isinstance(scope, ast.Lambda) and name in _params(scope):
                return [Cand(None, "unresolved", name, trace=("lambda parameter",))]
            line = getattr(scope, "lineno", line)
            scope = self.enclosing(m, scope)[0]
        assigns = self.scope_assignments(m, None, name, 10 ** 9)
        if assigns:
            return self._from_assigns(name, assigns, Ctx(m), depth, seen)
        if name in m.imports:
            mod, sym = m.imports[name]
            hit = self.ix.lookup_symbol(mod, sym)
            if hit and hit[0] == "var":
                tm = hit[1]
                assigns = self.scope_assignments(tm, None, sym, 10 ** 9)
                if assigns:
                    return [c.step(f"import {sym} from {tm.rel}")
                            for c in self._from_assigns(sym, assigns, Ctx(tm), depth, seen)]
                return [Cand(None, "unresolved", name, trace=(f"{sym} not assigned in {tm.rel}",))]
            if hit is None:
                full = f"{mod}.{sym}" if sym else mod
                return [Cand(None, "external", full, trace=(f"imported from {full} (not in project)",))]
            return [Cand(None, "unresolved", name, trace=(f"{name} is a {hit[0]}",))]
        if name in CLI_ROOTS:
            return [Cand(None, "dynamic_cli", name, trace=("command-line arguments",))]
        return [Cand(None, "unresolved", name, trace=(f"no assignment of {name} found",))]

    def _from_assigns(self, name, assigns, ctx: Ctx, depth, seen) -> List[Cand]:
        out = []
        for line, val, stmt in assigns:
            sub = ctx.at(line)
            here = f"{ctx.module.rel}:{line}"
            if isinstance(stmt, (ast.With, ast.AsyncWith)):
                if isinstance(val, ast.Call) and (src(val.func) in OPENERS or _last_name(val.func) in OPENERS) and val.args:
                    out += [c.step(f"{name} = file opened from {src(val.args[0], 60)} @ {here}")
                            for c in self._r(val.args[0], sub, depth, seen)]
                else:
                    out.append(Cand(None, "unresolved", f"with {src(val, 100)} as {name}", trace=(f"context manager @ {here}",)))
                continue
            if isinstance(stmt, (ast.For, ast.AsyncFor, ast.comprehension)):
                out += [c.step(f"for {name} in {src(val, 60)} @ {here}") for c in self._iter_values(val, sub, depth, seen)]
                continue
            out += [c.step(f"{name} = {src(val, 80)} @ {here}") for c in self._r(val, sub, depth, seen)]
        return out

    def _iter_values(self, it, ctx: Ctx, depth, seen) -> List[Cand]:
        node = it
        if isinstance(node, ast.Name):
            for scope in ([ctx.func, None] if ctx.func is not None else [None]):
                hits = self.scope_assignments(ctx.module, scope, node.id, ctx.line)
                if hits and isinstance(hits[0][1], (ast.List, ast.Tuple, ast.Set)):
                    node = hits[0][1]
                    break
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return [c for el in node.elts[: self.s.max_candidates] for c in self._r(el, ctx, depth, seen)]
        if isinstance(node, ast.Call) and _last_name(node.func) in FS_LISTERS:
            return [Cand(None, "dynamic_file", src(node, 120), trace=("files found on disk at run time",))]
        return [Cand(None, "unresolved", f"for ... in {src(node, 100)}", trace=("loop over a computed collection",))]

    # ── parameters ───────────────────────────────────────────────────────────
    def _bound_parameter(self, name, func, ctx: Ctx, depth, seen) -> List[Cand]:
        """Parameter of a function entered through a specific call: use the
        argument of that call (or the default when it was not passed)."""
        bound = ctx.bindings.get(name)
        fname = getattr(func, "name", "?")
        if bound is not None:
            arg, caller = bound
            return [c.step(f"argument {name} of this call to {fname}()") for c in self._r(arg, caller, depth, seen)]
        default = _param_default(func, name)
        if default is None or (isinstance(default, ast.Constant) and default.value is None):
            return [Cand(None, "unresolved", name, trace=(f"parameter {name} of {fname}() not passed, no default",))]
        dctx = Ctx(ctx.module, self.enclosing(ctx.module, func)[0], ctx.cls, getattr(func, "lineno", ctx.line))
        return [c.with_kind("default_param" if c.kind == "literal" else c.kind, f"default of parameter {name} in {fname}()")
                for c in self._r(default, dctx, depth, seen)]

    def bind(self, call: ast.Call, func, caller: Ctx, drop_first: bool) -> dict:
        """Map the parameters of `func` to the arguments of `call`."""
        a = func.args
        positional = [p.arg for p in a.posonlyargs + a.args]
        if drop_first:
            positional = positional[1:]
        out = {p: None for p in positional + [p.arg for p in a.kwonlyargs]}
        for i, arg in enumerate(call.args):
            if isinstance(arg, ast.Starred):
                break
            if i < len(positional):
                out[positional[i]] = (arg, caller)
        for k in call.keywords:
            if k.arg and k.arg in out:
                out[k.arg] = (k.value, caller)
        return out

    def _parameter(self, name, func, ctx: Ctx, depth, seen) -> List[Cand]:
        out = []
        m = ctx.module
        fname = getattr(func, "name", "?")
        default = _param_default(func, name)
        if default is not None and not (isinstance(default, ast.Constant) and default.value is None):
            dctx = Ctx(m, self.enclosing(m, func)[0], ctx.cls, getattr(func, "lineno", ctx.line))
            for c in self._r(default, dctx, depth, seen):
                kind = "default_param" if c.kind == "literal" else c.kind
                out.append(c.with_kind(kind, f"default of parameter {name} in {fname}()"))

        func_cls = self.enclosing(m, func)[1]
        is_method = func_cls is not None and func in func_cls.methods.values()
        params = [p.arg for p in func.args.posonlyargs + func.args.args]
        if is_method and not any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in func.decorator_list):
            params = params[1:]                                   # drop self / cls
        call_names = [func_cls.name] if (is_method and fname == "__init__") else [fname]
        n_sites = 0
        for cname in call_names:
            for cm, call in self.ix.calls_to(cname):
                if n_sites >= self.s.max_call_sites:
                    break
                if call.lineno == getattr(func, "lineno", -1) and cm.rel == m.rel:
                    continue
                if not self._call_targets(call, cm, func, m, func_cls, fname == "__init__"):
                    continue
                arg = _match_argument(call, name, params)
                if arg is None:
                    continue
                n_sites += 1
                cctx = self.context_of(cm, call)
                out += [c.step(f"argument of {src(call.func, 60)}(...) @ {cm.rel}:{call.lineno}")
                        for c in self._r(arg, cctx, depth, seen)]
        if not out:
            out.append(Cand(None, "unresolved", name,
                            trace=(f"parameter {name} of {fname}(): no default and no linked call site",)))
        return out

    def _call_targets(self, call: ast.Call, cm: ModuleInfo, func, fm: ModuleInfo,
                      fcls: Optional[ClassInfo], is_ctor: bool) -> bool:
        """Is `call` (in module cm) really a call of `func` (defined in fm)?"""
        f = call.func
        if isinstance(f, ast.Name):
            if f.id in cm.imports:
                mod, sym = cm.imports[f.id]
                hit = self.ix.lookup_symbol(mod, sym)
                if not hit:
                    return False
                if is_ctor:
                    return hit[0] == "class" and hit[1] is fcls
                return hit[0] == "func" and hit[2] is func
            if cm.rel == fm.rel:
                if is_ctor:
                    return fcls is not None and f.id == fcls.name
                if fm.functions.get(f.id) is func:
                    return True
                parent = fm.parents.get(func)                       # nested helper function
                return parent is not None and not isinstance(parent, (ast.Module, ast.ClassDef)) \
                    and self._inside(cm, call, parent)
            return False
        if not isinstance(f, ast.Attribute):
            return False
        recv = f.value
        root = _root_name(recv)
        # module attribute:  pkg.module.func(...)  /  module.Class(...)
        if isinstance(recv, (ast.Name, ast.Attribute)) and root in cm.imports:
            target = self._module_of_expr(recv, cm)
            if target is not None:
                if is_ctor:
                    return target.rel == fm.rel and f.attr == fcls.name
                return target.rel == fm.rel and fm.functions.get(f.attr) is func
        if is_ctor or fcls is None:
            return False
        # method calls
        caller_cls = self.enclosing(cm, call)[1]
        if isinstance(recv, ast.Name) and recv.id in ("self", "cls"):
            return caller_cls is not None and self._is_subclass(caller_cls, fcls)
        if isinstance(recv, ast.Call) and _last_name(recv.func) == "super":
            return caller_cls is not None and self._is_subclass(caller_cls, fcls)
        for ci in self._class_from_expr(recv, cm):            # ClassName.method(...)
            return self._is_subclass(ci, fcls)
        inferred = self._infer_project_classes(recv, self.context_of(cm, call))
        if inferred:
            return any(self._is_subclass(ci, fcls) for ci in inferred)
        # receiver unknown: accept only if the method name is unique in the project
        # and the receiver is not an object of another library
        return self.ix.definition_count(f.attr) == 1 and self.qualify(recv, self.context_of(cm, call)) is None \
            and (root not in cm.imports or self.ix.lookup_symbol(*cm.imports[root]) is not None)

    def _inside(self, m: ModuleInfo, node, scope) -> bool:
        cur = node
        while cur in m.parents:
            cur = m.parents[cur]
            if cur is scope:
                return True
        return False

    def _module_of_expr(self, expr, m: ModuleInfo) -> Optional[ModuleInfo]:
        if isinstance(expr, ast.Name) and expr.id in m.imports:
            mod, sym = m.imports[expr.id]
            hit = self.ix.lookup_symbol(mod, sym)
            return hit[1] if hit and hit[0] == "module" else None
        if isinstance(expr, ast.Attribute):
            base = self._module_of_expr(expr.value, m)
            if base is not None:
                return self.ix.find_module(f"{base.dotted}.{expr.attr}")
        return None

    def _is_subclass(self, ci: ClassInfo, target: ClassInfo, _seen=None) -> bool:
        if ci is target:
            return True
        _seen = (_seen or set()) | {id(ci)}
        for base in ci.node.bases:
            for b in self._class_from_expr(base, ci.module):
                if id(b) not in _seen and self._is_subclass(b, target, _seen):
                    return True
        return False

    # ── attributes ───────────────────────────────────────────────────────────
    def _attribute(self, node: ast.Attribute, ctx: Ctx, depth, seen) -> List[Cand]:
        m, attr, recv = ctx.module, node.attr, node.value
        rel, line = self._where(node, ctx)

        if attr in ("parent", "name", "stem", "suffix"):                     # pathlib helpers
            base = self._r(recv, ctx, depth, seen)
            if any(c.value is not None for c in base):
                fn = {"parent": lambda p: p.rstrip("/").rsplit("/", 1)[0] if "/" in p.rstrip("/") else ".",
                      "name": lambda p: p.rstrip("/").rsplit("/", 1)[-1],
                      "stem": lambda p: p.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0],
                      "suffix": lambda p: "." + p.rsplit(".", 1)[-1] if "." in p.rsplit("/", 1)[-1] else ""}[attr]
                return [Cand(fn(c.value), c.kind, c.source, rel, line, (f".{attr} @ {rel}:{line}",) + c.trace)
                        if c.value is not None else c for c in base]
        if src(node) in ("os.environ", "environ"):
            return [Cand(None, "dynamic_env", src(node), trace=("environment",))]

        if isinstance(recv, ast.Name) and recv.id in ("self", "cls") and ctx.cls is not None:
            res = self._class_attr(ctx.cls, attr, depth, seen)
            if res:
                return res
        target = self._module_of_expr(recv, m)                                  # module.CONST
        if target is not None:
            assigns = self.scope_assignments(target, None, attr, 10 ** 9)
            if assigns:
                return [c.step(f"{src(recv)}.{attr} in {target.rel}")
                        for c in self._from_assigns(attr, assigns, Ctx(target), depth, seen)]
            if attr in target.imports:
                return self._name(ast.copy_location(ast.Name(id=attr, ctx=ast.Load()), node), Ctx(target), depth, seen)
        for ci in self._class_from_expr(recv, m):                               # ClassName.ATTR
            res = self._class_attr(ci, attr, depth, seen)
            if res:
                return res
        for ci in self._infer_project_classes(recv, ctx):                        # obj.attr
            res = self._class_attr(ci, attr, depth, seen)
            if res:
                return res

        root = _root_name(node)
        if root in CLI_ROOTS or self._from_parse_args(recv, ctx):
            return self._cli_argument(node, attr, ctx, depth, seen)
        if root in CONFIG_ROOTS or _chain_names(node) & CONFIG_ROOTS:
            return [Cand(None, "dynamic_config", src(node, 120), trace=("configuration object",))]
        if root in m.imports and self.ix.lookup_symbol(*m.imports[root]) is None:
            return [Cand(None, "external", src(node, 120), trace=("attribute of another library",))]
        return [Cand(None, "unresolved", src(node, 120), trace=(f"attribute {attr} not found",))]

    def _cli_argument(self, node, attr, ctx: Ctx, depth, seen) -> List[Cand]:
        out = [Cand(None, "dynamic_cli", src(node, 120), trace=("command-line argument",))]
        options = self.ix.argparse_options(attr)
        linked = [o for o in options if o[0].rel == ctx.module.rel] or \
                 [o for o in options if self.ix.linked(o[0], ctx.module)]
        for am, default in linked:
            if default is None:
                continue
            for c in self._r(default, Ctx(am), depth, seen):
                if c.value is not None:
                    out.append(c.with_kind("cli_default", f"default of --{attr} @ {am.rel}"))
        return out

    def _from_parse_args(self, root, ctx: Ctx) -> bool:
        if not isinstance(root, ast.Name):
            return False
        for scope in ([ctx.func, None] if ctx.func is not None else [None]):
            for _, val, _ in self.scope_assignments(ctx.module, scope, root.id, ctx.line):
                if isinstance(val, ast.Call) and _last_name(val.func) in ("parse_args", "parse_known_args"):
                    return True
                if isinstance(val, ast.Subscript) and isinstance(val.value, ast.Call) \
                        and _last_name(val.value.func) == "parse_known_args":
                    return True
        return False

    def _class_attr(self, ci: ClassInfo, attr, depth, seen, _visited=None) -> List[Cand]:
        _visited = (_visited or set()) | {id(ci)}
        out = []
        for val, meth in ci.attrs.get(attr, []):
            line = getattr(val, "lineno", 10 ** 9)
            cctx = Ctx(ci.module, meth, ci, line)
            out += [c.step(f"{ci.name}.{attr} = {src(val, 80)} @ {ci.module.rel}:{line}")
                    for c in self._r(val, cctx, depth, seen)]
        meth = ci.methods.get(attr)
        if meth is not None and _is_property(meth):
            out += self._returns(meth, Ctx(ci.module, meth, ci), depth, seen)
        # dataclass / pydantic style: fields set through constructor keywords
        if (attr in ci.annotations or ci.attrs.get(attr)) and not self._has_init(ci):
            out += self._ctor_kwargs(ci, attr, depth, seen)
        if not out:
            for base in ci.node.bases:
                for b in self._class_from_expr(base, ci.module):
                    if id(b) not in _visited:
                        out += self._class_attr(b, attr, depth, seen, _visited)
        return out

    def _has_init(self, ci: ClassInfo, _seen=None) -> bool:
        if "__init__" in ci.methods:
            return True
        _seen = (_seen or set()) | {id(ci)}
        return any(self._has_init(b, _seen) for base in ci.node.bases
                   for b in self._class_from_expr(base, ci.module) if id(b) not in _seen)

    def _ctor_kwargs(self, ci: ClassInfo, attr, depth, seen) -> List[Cand]:
        out, n = [], 0
        for cm, call in self.ix.calls_to(ci.name):
            if n >= self.s.max_call_sites:
                break
            linked = any(c is ci for c in self._class_from_expr(call.func, cm))
            if not linked:
                continue
            for k in call.keywords:
                if k.arg == attr:
                    n += 1
                    out += [c.step(f"{ci.name}({attr}=...) @ {cm.rel}:{call.lineno}")
                            for c in self._r(k.value, self.context_of(cm, call), depth, seen)]
        return out

    def _class_from_expr(self, expr, m: ModuleInfo) -> List[ClassInfo]:
        if isinstance(expr, ast.Name):
            if expr.id in m.classes:
                return [m.classes[expr.id]]
            if expr.id in m.imports:
                hit = self.ix.lookup_symbol(*m.imports[expr.id])
                if hit and hit[0] == "class":
                    return [hit[1]]
            return []
        if isinstance(expr, ast.Attribute):
            target = self._module_of_expr(expr.value, m)
            if target is not None and expr.attr in target.classes:
                return [target.classes[expr.attr]]
        return []

    def _infer_project_classes(self, expr, ctx: Ctx, depth: int = 0) -> List[ClassInfo]:
        """Project classes an expression is an instance of (obj = MyClass(...), obj: MyClass)."""
        if depth > 5:
            return []
        m = ctx.module
        if isinstance(expr, ast.Call):
            return self._class_from_expr(expr.func, m)
        if isinstance(expr, ast.Name):
            scope = ctx.func
            while scope is not None:
                for _, val, _ in self.scope_assignments(m, scope, expr.id, ctx.line):
                    found = self._infer_project_classes(val, Ctx(m, scope, ctx.cls, ctx.line), depth + 1)
                    if found:
                        return found
                if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) and expr.id in _params(scope):
                    ann = _param_annotation(scope, expr.id)
                    return self._class_from_annotation(ann, m)
                scope = self.enclosing(m, scope)[0]
            for _, val, _ in self.scope_assignments(m, None, expr.id, 10 ** 9):
                found = self._infer_project_classes(val, Ctx(m), depth + 1)
                if found:
                    return found
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) \
                and expr.value.id in ("self", "cls") and ctx.cls is not None:
            for val, meth in ctx.cls.attrs.get(expr.attr, []):
                found = self._infer_project_classes(val, Ctx(m, meth, ctx.cls, getattr(val, "lineno", 10 ** 9)), depth + 1)
                if found:
                    return found
            ann = ctx.cls.annotations.get(expr.attr)
            if ann is not None:
                return self._class_from_annotation(ann, m)
        return []

    def _class_from_annotation(self, ann, m: ModuleInfo) -> List[ClassInfo]:
        if ann is None:
            return []
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):     # "MyClass"
            try:
                ann = ast.parse(ann.value, mode="eval").body
            except SyntaxError:
                return []
        if isinstance(ann, ast.Subscript):                                   # Optional[MyClass]
            inner = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
            return [c for e in inner for c in self._class_from_annotation(e, m)]
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):      # MyClass | None
            return self._class_from_annotation(ann.left, m) + self._class_from_annotation(ann.right, m)
        return self._class_from_expr(ann, m)

    # ── subscripts ───────────────────────────────────────────────────────────
    def _subscript(self, node: ast.Subscript, ctx: Ctx, depth, seen) -> List[Cand]:
        rel, line = self._where(node, ctx)
        text = src(node.value)
        if text in ("os.environ", "environ"):
            return [Cand(None, "dynamic_env", src(node, 120), trace=("environment variable",))]
        if text == "sys.argv":
            return [Cand(None, "dynamic_cli", src(node, 120), trace=("sys.argv",))]

        # "a/b/c".split("/")[-1]
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) \
                and node.value.func.attr in ("split", "rsplit") and isinstance(node.slice, (ast.Constant, ast.UnaryOp)):
            try:
                idx = ast.literal_eval(node.slice)
            except Exception:
                idx = None
            if isinstance(idx, int):
                call = node.value
                sep_c = self._r(call.args[0], ctx, depth, seen) if call.args else [Cand(None, "literal")]
                maxs = ast.literal_eval(call.args[1]) if len(call.args) > 1 and isinstance(call.args[1], ast.Constant) else -1
                out = []
                for base in self._r(call.func.value, ctx, depth, seen):
                    for sep in sep_c:
                        if base.value is None:
                            out.append(base)
                            continue
                        try:
                            parts = getattr(base.value, call.func.attr)(sep.value, maxs) if sep.value is not None \
                                else getattr(base.value, call.func.attr)()
                            out.append(Cand(parts[idx], base.kind, "", rel, line,
                                            (f".{call.func.attr}()[{idx}] @ {rel}:{line}",) + base.trace))
                        except (IndexError, TypeError, ValueError):
                            out.append(Cand(None, "unresolved", src(node, 120), trace=("split index out of range",)))
                return out

        keys = [c for c in self._r(node.slice, ctx, depth, seen) if c.value is not None]
        key_values = {c.value for c in keys}
        out = []
        for d, dctx in self._dict_literals(node.value, ctx):
            for k, v in zip(d.keys, d.values):
                if isinstance(k, ast.Constant) and str(k.value) in key_values:
                    out += [c.step(f"{text}[{k.value!r}]") for c in self._r(v, dctx, depth, seen)]
        if out:
            return out
        root = _root_name(node.value)
        if root in CONFIG_ROOTS or _chain_names(node.value) & CONFIG_ROOTS:
            stop = [Cand(None, "dynamic_config", src(node, 120), trace=("configuration lookup",))]
        else:
            stop = [Cand(None, "unresolved", src(node, 120), trace=("subscript of a computed container",))]
        # Note 4: torch.load(weights['detection_weights.pt']) -> the key names the artifact
        artifact = [Cand(c.value, "subscript_key", "", c.file, c.line, (f"key of {text}[...] @ {rel}:{line}",))
                    for c in keys if MODEL_ARTIFACT_RE.search(c.value)]
        return artifact + stop if artifact else stop

    def _dict_literals(self, expr, ctx: Ctx):
        if isinstance(expr, ast.Dict):
            return [(expr, ctx)]
        if isinstance(expr, ast.Call) and _last_name(expr.func) == "dict" and not expr.args:
            fake = ast.Dict(keys=[ast.Constant(k.arg) for k in expr.keywords if k.arg],
                            values=[k.value for k in expr.keywords if k.arg])
            return [(fake, ctx)]
        found = []
        if isinstance(expr, ast.Name):
            for scope in ([ctx.func, None] if ctx.func is not None else [None]):
                for line, val, _ in self.scope_assignments(ctx.module, scope, expr.id, ctx.line if scope else 10 ** 9):
                    if isinstance(val, (ast.Dict, ast.Call)):
                        found += self._dict_literals(val, Ctx(ctx.module, scope, ctx.cls, line))
                if found:
                    return found
            if expr.id in ctx.module.imports:
                hit = self.ix.lookup_symbol(*ctx.module.imports[expr.id])
                if hit and hit[0] == "var":
                    for line, val, _ in self.scope_assignments(hit[1], None, ctx.module.imports[expr.id][1], 10 ** 9):
                        if isinstance(val, ast.Dict):
                            return [(val, Ctx(hit[1]))]
        if isinstance(expr, ast.Attribute):
            target = self._module_of_expr(expr.value, ctx.module)
            if target is not None:
                for line, val, _ in self.scope_assignments(target, None, expr.attr, 10 ** 9):
                    if isinstance(val, ast.Dict):
                        return [(val, Ctx(target))]
            if isinstance(expr.value, ast.Name) and expr.value.id in ("self", "cls") and ctx.cls:
                for val, meth in ctx.cls.attrs.get(expr.attr, []):
                    if isinstance(val, ast.Dict):
                        found.append((val, Ctx(ctx.cls.module, meth, ctx.cls, getattr(val, "lineno", 10 ** 9))))
        return found

    # ── calls ────────────────────────────────────────────────────────────────
    def _call(self, node: ast.Call, ctx: Ctx, depth, seen) -> List[Cand]:
        f = node.func
        fname = src(f)
        last = _last_name(f)
        rel, line = self._where(node, ctx)
        R = lambda n: self._r(n, ctx, depth, seen)

        # environment variables: value at run time, default declared in code (Note 1)
        if fname in ("os.getenv", "getenv", "os.environ.get", "environ.get", "os.environ.setdefault"):
            key = src(node.args[0]) if node.args else ""
            out = [Cand(None, "dynamic_env", f"{fname}({key})", trace=("environment variable",))]
            default = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "default"), None)
            if default is not None:
                for c in R(default):
                    if c.value is not None:
                        out.append(c.with_kind("env_default", f"default of {fname}({key}) @ {rel}:{line}"))
                    else:
                        out.append(c)
            return out
        if fname in ("input", "getpass.getpass", "getpass"):
            return [Cand(None, "dynamic_input", fname + "(...)", trace=("user input",))]

        # str() and string methods
        if fname == "str" and node.args:
            return R(node.args[0])
        if isinstance(f, ast.Attribute) and f.attr in STR_UNARY and not node.args:
            op = STR_UNARY[f.attr]
            return [Cand(op(c.value), c.kind, c.source, c.file, c.line, c.trace) if c.value is not None else c
                    for c in R(f.value)]
        if isinstance(f, ast.Attribute) and f.attr in ("replace", "removeprefix", "removesuffix") and node.args:
            args = [R(a) for a in node.args[:2]]
            out = []
            for base in R(f.value):
                for combo in itertools.islice(itertools.product(*args), 4):
                    if base.value is None or any(a.value is None for a in combo):
                        out.append(base if base.value is None else Cand(None, "unresolved", src(node, 120),
                                                                        trace=(f".{f.attr}() with computed argument",)))
                        continue
                    val = getattr(base.value, f.attr)(*[a.value for a in combo])
                    out.append(Cand(val, base.kind, "", base.file, base.line, (f".{f.attr}() @ {rel}:{line}",) + base.trace))
            return out
        if isinstance(f, ast.Attribute) and f.attr == "join" and isinstance(f.value, ast.Constant) \
                and isinstance(f.value.value, str) and node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
            parts = [R(e) for e in node.args[0].elts]
            sep = f.value.value
            return [_combine(list(p), sep.join, (rel, line), "str.join")
                    for p in itertools.islice(itertools.product(*parts), self.s.max_candidates)]
        if isinstance(f, ast.Attribute) and f.attr == "format":
            base_c = R(f.value)
            pos = [R(a) for a in node.args]
            kws = [(k.arg, R(k.value)) for k in node.keywords if k.arg]
            out = []
            for b in base_c:
                if b.value is None:
                    out.append(b)
                    continue
                for combo in itertools.islice(itertools.product(*(pos + [v for _, v in kws])), self.s.max_candidates):
                    if all(c.value is not None for c in combo):
                        try:
                            val = b.value.format(*[c.value for c in combo[: len(pos)]],
                                                 **{k: c.value for (k, _), c in zip(kws, combo[len(pos):])})
                            out.append(Cand(val, b.kind, "", rel, line, (f"str.format @ {rel}:{line}",)))
                            continue
                        except (IndexError, KeyError, ValueError):
                            pass
                    bad = next((c for c in combo if c.value is None), None)
                    out.append(bad or Cand(None, "unresolved", src(node, 120), trace=("str.format",)))
            return out

        # paths
        if fname in PATH_JOIN or fname in PATH_CTORS or (isinstance(f, ast.Attribute) and f.attr == "joinpath"):
            items = list(node.args)
            if isinstance(f, ast.Attribute) and f.attr == "joinpath":
                items = [f.value] + items
            if items:
                parts = [R(a) for a in items]
                return [_combine(list(p), lambda xs: "/".join(x.rstrip("/") for x in xs[:-1]) + ("/" if len(xs) > 1 else "") + xs[-1]
                                 if len(xs) > 1 else xs[0], (rel, line), fname or "joinpath")
                        for p in itertools.islice(itertools.product(*parts), self.s.max_candidates)]
        if fname in ("os.path.dirname", "osp.dirname") and node.args:
            return [Cand(c.value.rstrip("/").rsplit("/", 1)[0] if "/" in c.value else ".", c.kind, "", rel, line, c.trace)
                    if c.value is not None else c for c in R(node.args[0])]
        if fname in ("os.path.basename", "osp.basename") and node.args:
            return [Cand(c.value.rstrip("/").rsplit("/", 1)[-1], c.kind, "", rel, line, c.trace)
                    if c.value is not None else c for c in R(node.args[0])]
        if fname in ("os.path.abspath", "os.path.realpath", "os.path.expanduser", "os.path.expandvars",
                     "osp.abspath", "osp.realpath", "osp.expanduser", "os.fspath") and node.args:
            return R(node.args[0])
        if isinstance(f, ast.Attribute) and f.attr in ("resolve", "absolute", "expanduser", "as_posix") and not node.args:
            return R(f.value)
        if isinstance(f, ast.Attribute) and f.attr == "with_suffix" and node.args:
            return [_combine([a, b], lambda xs: (xs[0].rsplit(".", 1)[0] if "." in xs[0].rsplit("/", 1)[-1] else xs[0]) + xs[1],
                             (rel, line), "with_suffix")
                    for a, b in itertools.islice(itertools.product(R(f.value), R(node.args[0])), self.s.max_candidates)]

        # dict.get(key, default)
        if isinstance(f, ast.Attribute) and f.attr == "get" and node.args:
            sub = ast.copy_location(ast.Subscript(value=f.value, slice=node.args[0], ctx=ast.Load()), node)
            out = [c for c in self._subscript(sub, ctx, depth, seen) if c.value is not None or c.kind != "unresolved"]
            if len(node.args) > 1:
                out += R(node.args[1])
            return out or [Cand(None, "unresolved", src(node, 120), trace=("dict.get on a computed container",))]

        if last in FILE_READERS or fname in OPENERS:
            return [Cand(None, "dynamic_file", src(node, 120), trace=("value read from a file",))]

        # project functions -> their return values
        for func, fctx in self._project_functions(f, ctx):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_method = fctx.cls is not None and func in fctx.cls.methods.values() \
                    and not any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in func.decorator_list)
                fctx.bindings = self.bind(node, func, ctx, drop_first=is_method)
            res = self._returns(func, fctx, depth, seen)
            if res:
                return [c.step(f"return of {getattr(func, 'name', '?')}() @ {fctx.module.rel}") for c in res]

        root = _root_name(f)
        if root in CLI_ROOTS:
            return [Cand(None, "dynamic_cli", src(node, 120), trace=("command-line argument",))]
        if root in CONFIG_ROOTS:
            return [Cand(None, "dynamic_config", src(node, 120), trace=("configuration object",))]
        return [Cand(None, "external", src(node, 120), trace=(f"value returned by {fname}()",))]

    def _project_functions(self, f, ctx: Ctx):
        m = ctx.module
        if isinstance(f, ast.Name):
            if f.id in m.functions:
                yield m.functions[f.id], Ctx(m, m.functions[f.id])
                return
            if f.id in m.imports:
                hit = self.ix.lookup_symbol(*m.imports[f.id])
                if hit and hit[0] == "func":
                    yield hit[2], Ctx(hit[1], hit[2])
                return
            scope = ctx.func
            while scope is not None:                                   # nested helper
                for sub in ast.iter_child_nodes(scope):
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == f.id:
                        yield sub, Ctx(m, sub, ctx.cls)
                        return
                scope = self.enclosing(m, scope)[0]
        elif isinstance(f, ast.Attribute):
            recv = f.value
            classes = []
            if isinstance(recv, ast.Name) and recv.id in ("self", "cls") and ctx.cls:
                classes = [ctx.cls]
            else:
                classes = self._class_from_expr(recv, m) or self._infer_project_classes(recv, ctx)
            for ci in classes:
                meth = self._find_method(ci, f.attr)
                if meth is not None:
                    owner = self.enclosing(meth[0].module, meth[1])[1] or meth[0]
                    yield meth[1], Ctx(meth[0].module, meth[1], owner)
                    return
            target = self._module_of_expr(recv, m)
            if target is not None and f.attr in target.functions:
                yield target.functions[f.attr], Ctx(target, target.functions[f.attr])

    def _find_method(self, ci: ClassInfo, name, _seen=None):
        if name in ci.methods:
            return ci, ci.methods[name]
        _seen = (_seen or set()) | {id(ci)}
        for base in ci.node.bases:
            for b in self._class_from_expr(base, ci.module):
                if id(b) not in _seen:
                    hit = self._find_method(b, name, _seen)
                    if hit:
                        return hit
        return None

    def _returns(self, func, fctx: Ctx, depth, seen) -> List[Cand]:
        out = []
        for node in ast.walk(func):
            if isinstance(node, ast.Return) and node.value is not None and self.enclosing(fctx.module, node)[0] is func:
                c = Ctx(fctx.module, func, fctx.cls or self.enclosing(fctx.module, func)[1], node.lineno,
                        fctx.bindings)
                out += self._r(node.value, c, depth, seen)
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # library qualification (used to recognise model-loading calls)
    # ══════════════════════════════════════════════════════════════════════════
    def qualify(self, expr, ctx: Ctx, depth: int = 0) -> Optional[str]:
        """Dotted name of the *library* object an expression refers to, e.g.
        `client.chat.completions.create` with client = OpenAI()  ->
        'openai.OpenAI().chat.completions.create'. None for project objects."""
        if depth == 0:
            k = id(expr)
            if k in self._qual_cache:
                return self._qual_cache[k]
            res = self._qualify(expr, ctx, 0)
            self._qual_cache[k] = res
            return res
        return self._qualify(expr, ctx, depth)

    def _qualify(self, expr, ctx: Ctx, depth: int) -> Optional[str]:
        if depth > 8 or expr is None:
            return None
        m = ctx.module
        if isinstance(expr, ast.Call):
            q = self._qualify(expr.func, ctx, depth + 1)
            return q + "()" if q else None
        if isinstance(expr, ast.Attribute):
            if isinstance(expr.value, ast.Name) and expr.value.id in ("self", "cls") and ctx.cls is not None:
                for val, meth in ctx.cls.attrs.get(expr.attr, []):
                    q = self._qualify(val, Ctx(m, meth, ctx.cls, getattr(val, "lineno", 10 ** 9)), depth + 1)
                    if q:
                        return q
                ann = ctx.cls.annotations.get(expr.attr)
                q = self._qualify_annotation(ann, ctx, depth)
                if q:
                    return q
                return None
            base = self._qualify(expr.value, ctx, depth + 1)
            return f"{base}.{expr.attr}" if base else None
        if isinstance(expr, ast.Name):
            scope = ctx.func
            while scope is not None:
                assigns = self.scope_assignments(m, scope, expr.id, ctx.line)
                if assigns:
                    return self._qualify(assigns[0][1], Ctx(m, scope, ctx.cls, assigns[0][0]), depth + 1)
                if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) and expr.id in _params(scope):
                    return self._qualify_annotation(_param_annotation(scope, expr.id), ctx, depth)
                scope = self.enclosing(m, scope)[0]
            if expr.id in m.imports:
                return self._external_origin(*m.imports[expr.id])
            assigns = self.scope_assignments(m, None, expr.id, 10 ** 9)
            if assigns:
                return self._qualify(assigns[0][1], Ctx(m, None, None, assigns[0][0]), depth + 1)
        return None

    def _qualify_annotation(self, ann, ctx, depth) -> Optional[str]:
        if ann is None:
            return None
        if isinstance(ann, ast.Subscript):
            inner = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
            for e in inner:
                q = self._qualify_annotation(e, ctx, depth)
                if q:
                    return q
            return None
        if isinstance(ann, ast.BinOp):
            return self._qualify_annotation(ann.left, ctx, depth) or self._qualify_annotation(ann.right, ctx, depth)
        q = self._qualify(ann, ctx, depth + 1)
        return q + "()" if q else None

    def _external_origin(self, mod: str, sym: Optional[str], depth: int = 0) -> Optional[str]:
        """Dotted library name of an import; follows project re-exports."""
        if depth > 6:
            return None
        full = f"{mod}.{sym}" if sym else mod
        if library_of(full):
            # a target library, also when the repository *is* that library
            # (e.g. `from vllm import LLM` in the examples of vllm itself)
            return full
        hit = self.ix.lookup_symbol(mod, sym)
        if hit is None:
            if self.ix.is_project_module(mod):
                return None
            return full
        if hit[0] == "var" and sym and sym in hit[1].imports:
            return self._external_origin(*hit[1].imports[sym], depth=depth + 1)
        return None


def _match_argument(call: ast.Call, name: str, params: List[str]) -> Optional[ast.AST]:
    for k in call.keywords:
        if k.arg == name:
            return k.value
    if name in params:
        i = params.index(name)
        if i < len(call.args) and not any(isinstance(a, ast.Starred) for a in call.args[: i + 1]):
            return call.args[i]
    return None
