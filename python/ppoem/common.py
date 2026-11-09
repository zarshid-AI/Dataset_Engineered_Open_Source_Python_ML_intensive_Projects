"""
Shared settings and helpers of the PPOEM engine.

Values come from python/config.py when it can be imported; otherwise the
defaults below are used (handy for unit tests and for reusing the engine in
another project).
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Set, Tuple

try:                                    # python/config.py
    import config as _cfg               # type: ignore
except Exception:                       # pragma: no cover - engine used stand-alone
    _cfg = None


def _c(name, default):
    return getattr(_cfg, name, default) if _cfg is not None else default


@dataclass
class Settings:
    skip_dirs: Set[str] = field(default_factory=lambda: set(_c("SCANNER_SKIP_DIRS", {
        ".git", "__pycache__", ".venv", "venv", "env", ".env", "node_modules", ".tox",
        ".eggs", "build", "dist", "site-packages"})))
    max_file_mb: float = _c("SCANNER_MAX_FILE_MB", 10)
    max_files: int = _c("SCANNER_MAX_FILES_REPO", 10000)
    max_depth: int = _c("RESOLVER_MAX_DEPTH", 15)
    max_candidates: int = _c("RESOLVER_MAX_CANDIDATES", 12)
    max_call_sites: int = _c("RESOLVER_MAX_CALL_SITES", 30)
    max_steps: int = _c("RESOLVER_MAX_STEPS", 4000)
    sklearn_model_types: Tuple[str, ...] = tuple(_c("SKLEARN_MODEL_TYPES",
                                                    ("classifier", "regressor", "cluster", "outlier_detector")))
    hf_non_model_repo_types: Tuple[str, ...] = tuple(_c("HF_NON_MODEL_REPO_TYPES", ("dataset", "space")))
    non_model_class_suffixes: Tuple[str, ...] = tuple(_c("NON_MODEL_CLASS_SUFFIXES", (
        "Tokenizer", "TokenizerFast", "Processor", "FeatureExtractor", "ImageProcessor",
        "ImageProcessorFast", "VideoProcessor", "Config", "GenerationConfig", "Scheduler")))
    max_call_chars: int = _c("DATASET_MAX_CALL_CHARS", 300)
    max_parsed_modules: int = _c("RESOLVER_MAX_PARSED_MODULES", 600)
    project_time_budget: float = _c("RESOLVER_PROJECT_TIME_BUDGET", 1200)


SETTINGS = Settings()

# ── test files: same rule as the original scanner ───────────────────────────
TEST_DIR_NAMES = {"test", "tests", "testing", "__test__", "__tests__"}
TEST_FILE_RE = re.compile(r"^test_|_test\.py$|_tests\.py$", re.IGNORECASE)


def is_test_path(rel_path: str) -> bool:
    parts = Path(rel_path.replace("\\", "/")).parts
    if any(p.lower() in TEST_DIR_NAMES for p in parts[:-1]):
        return True
    return bool(parts and TEST_FILE_RE.search(parts[-1]))


def iter_python_files(root: Path, settings: Settings = SETTINGS) -> Iterator[Tuple[Path, str]]:
    """Yield (absolute path, posix relative path) of every .py file, tests included
    (callers decide what to do with tests). On Windows the walk uses the
    extended-length prefix, so files deeper than 260 characters are read too."""
    from .repo import long_path
    base = long_path(root)
    count = 0
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in settings.skip_dirs)
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            if count >= settings.max_files:
                return
            full = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(full) > settings.max_file_mb * 1024 * 1024:
                    continue
            except OSError:
                continue
            count += 1
            yield Path(full), os.path.relpath(full, base).replace(os.sep, "/")


def src(node: Optional[ast.AST], limit: int = 0) -> str:
    """Source text of an AST node (empty string when not available)."""
    if node is None:
        return ""
    try:
        text = ast.unparse(node)
    except Exception:
        text = "<expr>"
    if limit and len(text) > limit:
        text = text[: limit - 3] + "..."
    return text
