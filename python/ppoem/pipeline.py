"""
Helpers shared by the pipeline scripts (Steps 2 and 4).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from .common import SETTINGS, Settings
from .extractor import EXTRACTOR_VERSION, extract_project
from .repo import clone, clone_dir_name, remove, safe_dir_name


def setup_logging(name: str, log_file: Optional[Path], level: str = "INFO") -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        log.addHandler(fh)
    return log


def full_name_of(row: dict) -> str:
    fn = str(row.get("full_name") or "").strip()
    if fn and fn.lower() != "nan":
        return fn
    url = str(row.get("url") or "").rstrip("/")
    return "/".join(url.split("/")[-2:])


def cache_path(details_dir: Path, full_name: str) -> Path:
    return Path(details_dir) / f"{safe_dir_name(full_name)}.json"


def load_cache(details_dir: Path, full_name: str) -> Optional[dict]:
    p = cache_path(details_dir, full_name)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if data.get("extractor_version") == EXTRACTOR_VERSION else None


def save_cache(details_dir: Path, full_name: str, data: dict) -> Path:
    p = cache_path(details_dir, full_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def analyse_repository(full_name: str, url: str, temp_dir: Path, details_dir: Path,
                       commit: Optional[str] = None, token: Optional[str] = None,
                       git_timeout: int = 300, settings: Settings = SETTINGS) -> dict:
    """Clone one repository (optionally at a pinned commit), run the extractor,
    store the per-project JSON and delete the clone. Safe to run in a worker
    process: it never raises, errors are returned in the result."""
    start = time.time()
    dest = Path(temp_dir) / clone_dir_name(full_name)
    result = {"full_name": full_name, "url": url, "commit": "", "error": "", "seconds": 0.0}
    try:
        ok, info = clone(url, dest, commit=commit, token=token, timeout=git_timeout)
        if not ok:
            result["error"] = f"clone failed: {info}"
            return result
        result["commit"] = info
        ex = extract_project(dest, settings)
        data = {"full_name": full_name, "url": url, "commit": info, **ex.to_dict()}
        save_cache(details_dir, full_name, data)
        result.update(data)
    except Exception as exc:                                  # pragma: no cover - defensive
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-2000:]
    finally:
        remove(dest)
        result["seconds"] = round(time.time() - start, 1)
    return result


def json_value(v):
    """Convert pandas / numpy values to plain JSON values (missing -> None)."""
    if v is None:
        return None
    try:
        import numpy as np
        if isinstance(v, np.generic):
            v = v.item()
    except ImportError:                                       # pragma: no cover
        pass
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        import pandas as pd
        if v is pd.NA or v is pd.NaT:
            return None
    except ImportError:                                       # pragma: no cover
        pass
    return v


def _pool(workers: int) -> ProcessPoolExecutor:
    kwargs = {"max_tasks_per_child": 1} if sys.version_info >= (3, 11) else {}
    return ProcessPoolExecutor(max_workers=max(1, workers), **kwargs)


def run_jobs(jobs, workers: int, on_result, retries: int = 1, log: Optional[logging.Logger] = None) -> None:
    """Run analyse_repository(*job) for every job, each in a separate process, so
    one huge repository cannot take the whole run down. A job whose process dies
    (e.g. out of memory) is retried alone; after `retries` failed attempts it is
    reported with an error instead. on_result(job, result) is called in this process."""
    pending = list(jobs)
    if workers > 1 and len(pending) > 1:
        crashed = []
        with _pool(workers) as pool:
            futures = {pool.submit(analyse_repository, *job): job for job in pending}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    on_result(job, fut.result())
                except BrokenProcessPool:
                    crashed.append(job)
        if crashed and log:
            log.warning(f"A worker process died; {len(crashed)} project(s) will be run again one at a time.")
        pending = crashed
    for job in pending:                                       # one project per process
        result = None
        for attempt in range(retries + 1):
            try:
                with _pool(1) as pool:
                    result = pool.submit(analyse_repository, *job).result()
                break
            except BrokenProcessPool:
                if log:
                    log.warning(f"{job[0]}: worker process died (attempt {attempt + 1}/{retries + 1})")
        if result is None:
            result = {"full_name": job[0], "url": job[1], "commit": "", "seconds": 0.0,
                      "error": "worker process died (probably out of memory); lower SCANNER_WORKERS "
                               "or RESOLVER_MAX_PARSED_MODULES and run again"}
        on_result(job, result)
