"""
Cloning and deleting repositories (Windows-safe), pinned to a commit.

Only the .py files are needed for the analysis, so a repository is fetched as
a *partial* clone (`--filter=blob:none`, depth 1) with a sparse checkout of
`*.py`. Git then downloads only the Python files: this is much faster and it
avoids most "Filename too long" errors on Windows, because the long non-Python
paths (docs, test data, templates, ...) are never written to disk.

On top of that:
  * every git command runs with core.longpaths=true (Git for Windows can then
    write paths longer than 260 characters);
  * clones go to a short temporary folder with a short folder name (see
    config.SCANNER_TEMP_DIR), so the .git internals stay short as well;
  * Python reads and deletes the clone through the Windows extended-length
    path prefix (\\\\?\\), see long_path();
  * if the partial clone is not supported (old git, unusual server), a normal
    shallow clone is used instead.

The token is sent as an HTTP header instead of being written into the clone
URL, so it never ends up in .git/config or in log files.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

IS_WINDOWS = os.name == "nt"
SPARSE_PATTERNS = "*.py\n"


def long_path(path) -> str:
    """Absolute path; on Windows with the \\\\?\\ prefix so paths > 260 chars work."""
    p = os.path.abspath(str(path))
    if IS_WINDOWS and not p.startswith("\\\\?\\"):
        p = "\\\\?\\UNC\\" + p[2:] if p.startswith("\\\\") else "\\\\?\\" + p
    return p


def clone_dir_name(full_name: str) -> str:
    """Short, unique folder name for a clone: 'repo-1a2b3c4d' (max ~30 chars)."""
    repo = re.sub(r"[^A-Za-z0-9._-]", "_", full_name.split("/")[-1])[:20]
    return f"{repo}-{hashlib.sha1(full_name.encode()).hexdigest()[:8]}"


def safe_dir_name(full_name: str) -> str:
    """File name used for the per-project JSON (readable, not used for clones)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", full_name.replace("/", "__"))[:120]


def _env() -> dict:
    return {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}


def _base(token: Optional[str]) -> List[str]:
    args = ["git", "-c", "core.longpaths=true", "-c", "core.autocrlf=false", "-c", "advice.detachedHead=false"]
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        args += ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}"]
    return args


def _run(cmd: List[str], timeout: int) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=_env())
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:500])


def _clone_python_only(url: str, dest: Path, ref: str, token: Optional[str], timeout: int) -> None:
    dest.mkdir(parents=True)
    g = _base(token) + ["-C", str(dest)]
    _run(g + ["init", "-q"], timeout)
    _run(g + ["config", "core.longpaths", "true"], timeout)
    _run(g + ["config", "core.sparseCheckout", "true"], timeout)
    _run(g + ["remote", "add", "origin", url], timeout)
    info = dest / ".git" / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "sparse-checkout").write_text(SPARSE_PATTERNS, encoding="utf-8")
    _run(g + ["fetch", "-q", "--depth", "1", "--filter=blob:none", "--no-tags", "origin", ref], timeout)
    _run(g + ["checkout", "-q", "FETCH_HEAD"], timeout)


def _clone_full(url: str, dest: Path, commit: Optional[str], token: Optional[str], timeout: int) -> None:
    if commit:
        dest.mkdir(parents=True)
        g = _base(token) + ["-C", str(dest)]
        _run(g + ["init", "-q"], timeout)
        _run(g + ["config", "core.longpaths", "true"], timeout)
        _run(g + ["remote", "add", "origin", url], timeout)
        _run(g + ["fetch", "-q", "--depth", "1", "--no-tags", "origin", commit], timeout)
        _run(g + ["checkout", "-q", "FETCH_HEAD"], timeout)
    else:
        _run(_base(token) + ["clone", "-q", "--depth", "1", "--no-tags",
                             "-c", "core.longpaths=true", url, str(dest)], timeout)


def clone(url: str, dest: Path, commit: Optional[str] = None, token: Optional[str] = None,
          timeout: int = 300, python_only: bool = True) -> Tuple[bool, str]:
    """Clone `url` into `dest` (only .py files when `python_only` and possible).
    With `commit`, check out exactly that commit. Returns (ok, sha or error)."""
    dest = Path(dest)
    remove(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for method in (("python_only", "full") if python_only else ("full",)):
        try:
            if method == "python_only":
                _clone_python_only(url, dest, commit or "HEAD", token, timeout)
            else:
                _clone_full(url, dest, commit, token, timeout)
            sha = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True,
                                 text=True, timeout=60, env=_env()).stdout.strip()
            return True, sha
        except subprocess.TimeoutExpired:
            errors.append(f"{method}: timeout after {timeout}s")
        except FileNotFoundError:
            return False, "git is not installed or not on PATH"
        except Exception as exc:
            errors.append(f"{method}: {exc}")
        remove(dest)
    return False, " | ".join(errors)[:800]


def remove(path) -> None:
    """Delete a folder, also read-only files and paths > 260 chars on Windows."""
    if not path:
        return
    lp = long_path(path)
    if not os.path.exists(lp):
        return

    def _onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    for _ in range(3):
        shutil.rmtree(lp, onerror=_onerror)
        if not os.path.exists(lp):
            return
        time.sleep(0.5)
    if IS_WINDOWS:
        subprocess.run(["cmd", "/c", "rd", "/s", "/q", lp], capture_output=True)
    else:
        subprocess.run(["rm", "-rf", lp], capture_output=True)
