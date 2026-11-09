"""
Model extraction for one cloned project
=======================================
1. index the project (ppoem.index)
2. find every model-loading call in production code (ppoem.rules), with the
   library recognised from imports, aliases and the object the call is made on
3. resolve the model argument across the project (ppoem.resolver)
4. apply the model filters (ppoem.filters)
5. return dataset entries in the PPOEM format:

   literal_string   the model name is known
     library, model, model_variable, model_file, variable_file, model_line,
     variable_line, call_expression, notes
       - direct literal:   model_variable/variable_file/variable_line = null,
                           model_file/model_line = the call
       - resolved value:   model_variable = expression passed to the call,
                           variable_file/variable_line = the call,
                           model_file/model_line = where the value is written
   variable_based   the model name depends on run-time information
     library, variable, file, variable_line, call_expression, notes, variable_source

Besides the entries, the extraction records every excluded call (with the
reason) and the resolution chain of every call, for manual checking.
"""

from __future__ import annotations

import ast
import itertools
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import filters
from .common import SETTINGS, Settings, src
from .index import ModuleInfo, ProjectIndex
from .resolver import Cand, Ctx, Resolver, _dedup
from .rules import LIBRARIES, LoadRule, build_rules, class_allowed, class_name, library_of

EXTRACTOR_VERSION = "2.0.2"

# files that can contain a model-loading call (library names, also via re-exports)
CANDIDATE_RX = re.compile(
    r"torch|tensorflow|keras|sklearn|transformers|openai|anthropic|groq|fireworks|llama_api_client|"
    r"deepseek|vllm|llama_cpp|xgboost|ollama|vertexai|genai|generativeai|diffusers|huggingface_hub|"
    r"from_pretrained|completions|snapshot_download|hf_hub_download|load_model", re.IGNORECASE)

# order used when a call on an unknown client object could belong to several
# OpenAI-compatible SDKs imported in the same file
FALLBACK_PRIORITY = ("llama_api_client", "groq", "fireworks", "deepseek", "anthropic", "openai")

KIND_TEXT = {
    "literal": "resolved from {var}",
    "default_param": "default value of parameter, resolved from {var}",
    "env_default": "default value of environment variable, resolved from {var}",
    "cli_default": "default value of command-line option, resolved from {var}",
    "subscript_key": "key of the container passed to the call",
}
STATUS_TEXT = {
    "dynamic_cli": "command-line argument",
    "dynamic_env": "environment variable",
    "dynamic_config": "configuration object",
    "dynamic_file": "file content",
    "dynamic_input": "user input",
    "external": "value returned by another library",
    "unresolved": "not statically resolvable",
}


@dataclass
class ModelCall:
    rule: LoadRule
    library: str
    module: ModuleInfo
    call: ast.Call
    qualname: str
    arg: Optional[ast.AST]          # expression holding the model (None for model_class rules)
    arg_text: str
    extra: Optional[ast.AST] = None


@dataclass
class Extraction:
    literal_string: List[dict] = field(default_factory=list)
    variable_based: List[dict] = field(default_factory=list)
    excluded: List[dict] = field(default_factory=list)
    trace: List[dict] = field(default_factory=list)
    library_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)
    libraries_detected: List[str] = field(default_factory=list)
    model_libraries: List[str] = field(default_factory=list)
    n_files: int = 0
    parse_errors: List[str] = field(default_factory=list)

    @property
    def has_models(self) -> bool:
        return bool(self.literal_string or self.variable_based)

    def to_dict(self) -> dict:
        return {
            "extractor_version": EXTRACTOR_VERSION,
            "n_files": self.n_files,
            "parse_errors": self.parse_errors,
            "library_usage": self.library_usage,
            "libraries_detected": self.libraries_detected,
            "model_libraries": self.model_libraries,
            "literal_string": self.literal_string,
            "variable_based": self.variable_based,
            "excluded": self.excluded,
            "trace": self.trace,
        }


class ProjectExtractor:
    def __init__(self, root: Path, settings: Settings = SETTINGS):
        self.s = settings
        self.index = ProjectIndex(Path(root), settings)
        self.resolver = Resolver(self.index, settings)
        self.rules = build_rules(settings.non_model_class_suffixes, settings.sklearn_model_types)
        self._fallback_rules = [r for r in self.rules if r.fallback_suffixes]
        self._rules_by_lib: Dict[str, List[LoadRule]] = defaultdict(list)
        for r in self.rules:
            self._rules_by_lib[r.library].append(r)
        self._module_libs: Dict[str, set] = {}
        # reduced budget used when a project exceeds its time budget
        from dataclasses import replace
        self._fast_settings = replace(settings, max_steps=200, max_call_sites=5)

    # ── which libraries a module imports ─────────────────────────────────────
    def imported_libraries(self, m: ModuleInfo) -> set:
        if m.rel not in self._module_libs:
            libs = set()
            for mod, sym in m.imports.values():
                origin = self.resolver._external_origin(mod, sym)
                lib = library_of(origin) if origin else None
                if lib:
                    libs.add(lib)
            self._module_libs[m.rel] = libs
        return self._module_libs[m.rel]

    # ── detection ────────────────────────────────────────────────────────────
    def _arg(self, call: ast.Call, specs) -> Optional[ast.AST]:
        for pos, kw in specs:
            if kw:
                for k in call.keywords:
                    if k.arg == kw:
                        return k.value
            if pos is not None and pos < len(call.args):
                if not any(isinstance(a, ast.Starred) for a in call.args[: pos + 1]):
                    return call.args[pos]
        return None

    def _kwargs_arg(self, call: ast.Call, specs, ctx: Ctx) -> Tuple[Optional[ast.AST], str]:
        """Model passed inside **kwargs: look the key up in a dict literal."""
        for k in call.keywords:
            if k.arg is None:
                names = [kw for _, kw in specs if kw]
                for d, _ in self.resolver._dict_literals(k.value, ctx):
                    for key, val in zip(d.keys, d.values):
                        if isinstance(key, ast.Constant) and key.value in names:
                            return val, src(val, 200)
                return k.value, "**" + src(k.value, 200)
        return None, ""

    def _match(self, m: ModuleInfo, call: ast.Call) -> Optional[ModelCall]:
        ctx = self.resolver.context_of(m, call)
        qual = self.resolver.qualify(call.func, ctx)
        rule = None
        if qual:
            for r in self._rules_by_lib.get(library_of(qual) or "", []):
                if r.matches(qual) and class_allowed(r, qual, self.s.non_model_class_suffixes):
                    rule = r
                    break
            library = rule.library if rule else None
        elif isinstance(call.func, ast.Attribute):
            chain = src(call.func)
            libs = self.imported_libraries(m)
            matches = [r for r in self._fallback_rules
                       if any(chain.endswith("." + sfx) for sfx in r.fallback_suffixes)
                       and set(r.fallback_requires) & libs]
            if matches:
                matches.sort(key=lambda r: FALLBACK_PRIORITY.index(r.library) if r.library in FALLBACK_PRIORITY else 99)
                rule = matches[0]
                qual = f"<{rule.library} client>.{chain.split('.', 1)[-1]}"
            library = rule.library if rule else None
        if rule is None:
            return None

        if rule.kind == "model_class":
            return ModelCall(rule, library, m, call, qual, None, "")
        arg = self._arg(call, rule.model_args)
        arg_text = src(arg, 200) if arg is not None else ""
        if arg is None:
            arg, arg_text = self._kwargs_arg(call, rule.model_args, ctx)
        if arg is None:
            return None                       # API called without a model (e.g. InferenceClient())
        extra = self._arg(call, rule.extra_args) if rule.extra_args else None
        return ModelCall(rule, library, m, call, qual, arg, arg_text, extra)

    def candidate_modules(self) -> List[str]:
        """Production files that can contain a model-loading call (text pre-filter)."""
        return sorted(self.index.files_matching(CANDIDATE_RX))

    def find_model_calls(self) -> List[ModelCall]:
        """All model-loading calls (used by tests; extract() streams instead)."""
        out = []
        for rel in self.candidate_modules():
            m = self.index.module(rel)
            if m is not None:
                out += [mc for mc in (self._match(m, c) for c in m.calls) if mc is not None]
        return out

    # ── library usage (Step 2 "library imported and called") ────────────────
    def _count_usage(self, m: ModuleInfo, usage: Dict[str, Dict[str, int]]):
        for node in ast.walk(m.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    ([node.module] if node.module and not node.level else [])
                for lib in {library_of(n) for n in names if n} - {None}:
                    usage[lib]["imports"] += 1
        for call in m.calls:
            q = self.resolver.qualify(call.func, self.resolver.context_of(m, call))
            lib = library_of(q) if q else None
            if lib:
                usage[lib]["calls"] += 1

    # ── resolution and entries ───────────────────────────────────────────────
    def _call_text(self, call) -> str:
        return src(call, self.s.max_call_chars)

    def _resolve_arg(self, mc: ModelCall) -> List[Cand]:
        ctx = self.resolver.context_of(mc.module, mc.call)
        cands = self.resolver.resolve_expr(mc.arg, ctx)
        if mc.extra is None:
            return cands
        extras = [c for c in self.resolver.resolve_expr(mc.extra, ctx) if c.value is not None]
        if not extras:
            return cands
        out = []
        for c, e in itertools.product(cands, extras):
            if c.value is None:
                out.append(c)
                continue
            parts = (e.value, c.value) if mc.rule.extra_before else (c.value, e.value)
            out.append(Cand(mc.rule.extra_join.join(parts), c.kind, "", c.file, c.line, c.trace))
        return _dedup(out)

    def extract(self) -> Extraction:
        ex = Extraction()
        usage = {lib: {"imports": 0, "calls": 0} for lib in LIBRARIES}
        literal_keys, variable_keys = set(), set()
        deadline = time.time() + self.s.project_time_budget
        for rel in self.candidate_modules():
            m = self.index.module(rel)
            if m is None:
                continue
            self._count_usage(m, usage)
            for call in m.calls:
                mc = self._match(m, call)
                if mc is None:
                    continue
                if mc.qualname.startswith("<"):
                    usage[mc.library]["calls"] += 1
                if time.time() > deadline:
                    self.resolver.s = self._fast_settings
                self._process(ex, mc, literal_keys, variable_keys)
                self.index.checkpoint(keep=(rel,))
            self.index.checkpoint(keep=(rel,))
        ex.n_files = self.index.n_files
        ex.parse_errors = list(self.index.parse_errors)
        ex.model_libraries = sorted({e["library"] for e in ex.literal_string + ex.variable_based})
        ex.library_usage = {k: v for k, v in usage.items() if v["imports"] or v["calls"]}
        ex.libraries_detected = sorted(lib for lib, u in ex.library_usage.items() if u["imports"] and u["calls"])
        return ex

    def _process(self, ex: Extraction, mc: ModelCall, literal_keys: set, variable_keys: set):
        rel, line = mc.module.rel, mc.call.lineno
        call_text = self._call_text(mc.call)
        rule = mc.rule
        base_excl = {"library": mc.library, "rule": rule.rule_id, "file": rel, "line": line,
                     "call_expression": call_text}

        # Note 6: Hugging Face datasets / spaces
        if rule.repo_type_arg is not None:
            rt = self._arg(mc.call, (rule.repo_type_arg,))
            if rt is not None:
                rt_vals = [c.value for c in self.resolver.resolve_expr(rt, self.resolver.context_of(mc.module, mc.call))
                           if c.value is not None]
                reason = filters.non_model_repo_type(rt_vals, self.s.hf_non_model_repo_types)
                if reason:
                    ex.excluded.append({**base_excl, "reason": reason})
                    return

        # model given by the class / function itself (sklearn, xgboost, keras applications)
        if rule.kind == "model_class":
            model = rule.fixed_model or class_name(mc.qualname)
            self._add_literal(ex, literal_keys, mc, model, None, rel, line, call_text,
                              self._class_note(mc, model))
            ex.trace.append({"file": rel, "line": line, "library": mc.library, "call_expression": call_text,
                             "status": "model_class", "values": [model], "chains": []})
            return

        # model given as a plain string literal
        if isinstance(mc.arg, ast.Constant) and isinstance(mc.arg.value, str) and mc.extra is None:
            value = mc.arg.value
            reason = filters.implausible_value_reason(value) or \
                (filters.data_file_reason(value) if rule.artifact_check else None)
            if reason:
                ex.excluded.append({**base_excl, "model": value, "reason": reason})
                return
            self._add_literal(ex, literal_keys, mc, value, None, rel, line, call_text,
                              f"Matched {rule.rule_id}: {value} (literal string)")
            ex.trace.append({"file": rel, "line": line, "library": mc.library, "call_expression": call_text,
                             "status": "literal", "values": [value], "chains": []})
            return

        # model given by a variable / expression: project-wide resolution
        cands = self._resolve_arg(mc)
        kept, dropped = [], []
        for c in cands:
            if c.value is None:
                continue
            reason = filters.implausible_value_reason(c.value) or \
                (filters.data_file_reason(c.value) if rule.artifact_check else None)
            (dropped if reason else kept).append((c, reason))
        unresolved = [c for c in cands if c.value is None]
        status = self._status(cands, kept)
        ex.trace.append({
            "file": rel, "line": line, "library": mc.library, "call_expression": call_text,
            "model_variable": mc.arg_text, "status": status,
            "values": [c.value for c, _ in kept],
            "stopped_at": sorted({c.source for c in unresolved if c.source}),
            "chains": [" -> ".join(c.trace) for c in cands[:6]],
        })

        if kept:
            other = self._runtime_sources(unresolved)
            for c, _ in kept:
                note = KIND_TEXT.get(c.kind, "resolved from {var}").format(var=mc.arg_text)
                if other:
                    note += f"; other paths depend on {other}"
                self._add_literal(ex, literal_keys, mc, c.value, c, rel, line, call_text,
                                  f"Matched {rule.rule_id}: {c.value} ({note})")
            return
        if dropped and not unresolved:
            for c, reason in dropped:
                ex.excluded.append({**base_excl, "model": c.value, "reason": reason})
            return
        if rule.artifact_check:
            reason = filters.data_variable_reason(mc.arg_text)
            if reason:
                ex.excluded.append({**base_excl, "model_variable": mc.arg_text, "reason": reason})
                return
        for c, reason in dropped:
            ex.excluded.append({**base_excl, "model": c.value, "reason": reason})
        self._add_variable(ex, variable_keys, mc, rel, line, call_text, unresolved, status)


    # ── helpers ──────────────────────────────────────────────────────────────
    def _status(self, cands, kept) -> str:
        if kept and all(c.value is not None for c in cands):
            return "resolved"
        if kept:
            return "partially_resolved"
        kinds = sorted({c.kind for c in cands if c.value is None})
        dyn = [k for k in kinds if k.startswith("dynamic")]
        return dyn[0] if dyn else ("external" if "external" in kinds else "unresolved")

    def _runtime_sources(self, unresolved: List[Cand]) -> str:
        parts = []
        for c in unresolved:
            label = STATUS_TEXT.get(c.kind, "unresolved value")
            parts.append(f"{label} ({c.source})" if c.source else label)
        return "; ".join(dict.fromkeys(parts))[:300]

    def _class_note(self, mc: ModelCall, model: str) -> str:
        if mc.library == "sklearn":
            return f"sklearn estimator instantiation: {model}"
        if mc.rule.fixed_model:
            return f"Matched {mc.rule.rule_id}: {mc.qualname} trains a {model}"
        return f"{mc.library} model instantiation: {model}"

    def _add_literal(self, ex, keys, mc: ModelCall, model, cand: Optional[Cand], rel, line, call_text, notes):
        if cand is None:
            entry = {"library": mc.library, "model": model, "model_variable": None, "model_file": rel,
                     "variable_file": None, "model_line": line, "variable_line": None,
                     "call_expression": call_text, "notes": notes}
        else:
            entry = {"library": mc.library, "model": model, "model_variable": mc.arg_text,
                     "model_file": cand.file or rel, "variable_file": rel,
                     "model_line": cand.line if cand.line is not None else line, "variable_line": line,
                     "call_expression": call_text, "notes": notes}
        key = (entry["library"], entry["model"], entry["model_file"], entry["model_line"],
               entry["variable_file"], entry["variable_line"])
        if key not in keys:
            keys.add(key)
            ex.literal_string.append(entry)

    def _add_variable(self, ex, keys, mc: ModelCall, rel, line, call_text, unresolved, status):
        sources = "; ".join(dict.fromkeys(c.source for c in unresolved if c.source))[:300]
        what = STATUS_TEXT.get(status, "not statically resolvable")
        entry = {"library": mc.library, "variable": mc.arg_text, "file": rel, "variable_line": line,
                 "call_expression": call_text,
                 "notes": f"Matched {mc.rule.rule_id}: {mc.arg_text} (variable reference; value depends on {what})",
                 "variable_source": sources or mc.arg_text}
        key = (entry["library"], entry["variable"], rel, line)
        if key not in keys:
            keys.add(key)
            ex.variable_based.append(entry)


def extract_project(root: Path, settings: Settings = SETTINGS) -> Extraction:
    return ProjectExtractor(root, settings).extract()
