"""
LLM as Judge - Evaluates GitHub projects using Ollama (gpt-oss:120b)
Usage: python llm_judge.py
Requires: pip install pandas requests gitpython
Requires: Ollama running locally with gpt-oss:120b model pulled
"""

import os
import ast
import json
import pandas as pd
import requests
import logging
from pathlib import Path
from urllib.parse import urlparse
import sys

# Load project config
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from ppoem.repo import clone as git_clone, clone_dir_name, long_path, remove as remove_dir  # long-path safe (Windows)

# ── Configuration (all values from config.py / .env) ─────────────────────────
CSV_FILE       = str(cfg.DETECTED_LIBS_CSV)
OUTPUT_JSON    = str(cfg.LLM_RESPONSE_JSON)
OLLAMA_URL     = cfg.OLLAMA_URL
MODEL_NAME     = cfg.MODEL_NAME
CLONE_BASE_DIR = str(cfg.LLM_CLONE_BASE_DIR)
LOG_LEVEL      = getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO)
# ────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize_github_url(url: str) -> str:
    """Strip trailing slashes / .git suffix and return clean HTTPS URL."""
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    if url.startswith("git@github.com:"):
        url = url.replace("git@github.com:", "https://github.com/")
    return url


def clone_repo(url: str, dest: str) -> bool:
    """
    Shallow clone (depth 1) of the whole repository (README and folder tree are
    needed here). Runs git with core.longpaths=true, so repositories with very
    long file names also clone on Windows.
    """
    log.info(f"Cloning {url} -> {dest}")
    ok, info = git_clone(url, Path(dest), token=cfg.GITHUB_SEARCH_TOKEN or None,
                         timeout=cfg.SCANNER_GIT_TIMEOUT * 4, python_only=False)
    if ok:
        log.info(f"  -> Clone complete: {dest}")
        return True
    log.warning(f"  -> Clone failed: {info}")
    return False


def has_test_folder(repo_path: str) -> bool:
    """
    Return True if repo contains any folder/file that looks like a test suite.
    Checks both via AST (Python test files) and simple folder-name heuristics.
    """
    test_dir_names = {"test", "tests", "test_suite", "testing", "spec", "specs", "__tests__"}
    repo = Path(repo_path)

    # 1. Directory name heuristic
    for item in repo.rglob("*"):
        if item.is_dir() and item.name.lower() in test_dir_names:
            return True

    # 2. Python file AST scan – look for unittest / pytest patterns
    for py_file in repo.rglob("*.py"):
        if any(part.lower().startswith("test") for part in py_file.parts):
            return True
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                # Import of unittest or pytest at top level
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = (
                        [alias.name for alias in node.names]
                        if isinstance(node, ast.Import)
                        else ([node.module] if node.module else [])
                    )
                    if any(n and ("unittest" in n or "pytest" in n) for n in names):
                        return True
                # Class inheriting from TestCase
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        base_str = ast.unparse(base) if hasattr(ast, "unparse") else ""
                        if "TestCase" in base_str:
                            return True
        except Exception:
            pass

    return False


def get_project_structure(repo_path: str, max_lines: int = 150) -> str:
    """Return a trimmed directory tree of the repo."""
    lines = []
    repo = Path(repo_path)
    for item in sorted(repo.rglob("*")):
        # Skip hidden / venv / node_modules
        parts = item.relative_to(repo).parts
        if any(p.startswith(".") or p in {"node_modules", "__pycache__", ".git", "venv", "env"} for p in parts):
            continue
        depth = len(parts) - 1
        prefix = "    " * depth + ("📁 " if item.is_dir() else "📄 ")
        lines.append(prefix + item.name)
        if len(lines) >= max_lines:
            lines.append("... (truncated)")
            break
    return "\n".join(lines)


def get_readme(repo_path: str, max_chars: int = 3000) -> str:
    """Return README content (truncated)."""
    repo = Path(repo_path)
    for name in ["README.md", "README.rst", "README.txt", "README", "readme.md"]:
        candidate = repo / name
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8", errors="ignore")
            return text[:max_chars] + ("\n... (truncated)" if len(text) > max_chars else "")
    return "(No README found)"


def ask_ollama(prompt: str, retries: int = 3, backoff_base: int = 10, timeout: int = 300) -> dict:
    """
    Send prompt to Ollama with retry + exponential backoff.
      timeout=300s : covers slow CPU inference on large prompts
      retries=3    : handles cold-starts and transient resource spikes
      backoff      : waits 10s -> 20s -> 40s between attempts
    """
    import time

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            log.info(f"  -> Ollama attempt {attempt}/{retries} (timeout={timeout}s)")
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            raw = resp.json().get("response", "")

            # Try to extract JSON block from response
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(raw[start:end])

            # Fallback: parse yes/no from raw text
            lower = raw.lower()
            verdict = "yes" if "well-built" in lower or '"yes"' in lower else "no"
            return {"well_built_project": verdict, "reasoning": raw.strip()}

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = str(e)
            wait = backoff_base * (2 ** (attempt - 1))  # 10s, 20s, 40s
            if attempt < retries:
                log.warning(f"  -> Attempt {attempt} failed (timeout/connection): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                log.error(f"  -> All {retries} attempts failed. Last error: {e}")

        except Exception as e:
            last_error = str(e)
            log.warning(f"  -> Attempt {attempt} unexpected error: {e}")
            if attempt < retries:
                wait = backoff_base * (2 ** (attempt - 1))
                time.sleep(wait)

    return {
        "well_built_project": "error",
        "reasoning": f"Failed after {retries} attempts. Last error: {last_error}"
    }

def build_prompt(url: str, structure: str, readme: str) -> str:
    return f"""You are a senior software engineer reviewing GitHub repositories.

Repository URL: {url}

Project Structure:
{structure}

README:
{readme}

Your task:
1. Classify the PROJECT TYPE from the categories below.
2. Decide if this is a serious, well-built software project or not.

PROJECT TYPE — pick exactly one:
- "library"          : Reusable code package meant to be imported by others (has setup.py / pyproject.toml, public API)
- "framework"        : Opinionated scaffold others build applications on top of
- "application"      : Complete, runnable software product (web app, desktop app, CLI tool, service)
- "training_tool"    : ML/AI model training scripts, experiments, or pipelines
- "deployment_tool"  : CI/CD, Docker, Kubernetes, infrastructure-as-code, MLOps tooling
- "homework"         : University assignment, course project, or personal learning exercise
- "personal_backup"  : Random files, notes, configs, or personal scripts with no clear public purpose
- "other"            : Does not fit any category above

Respond ONLY with a JSON object in this exact format (no extra text):
{{
  "well_built_project": "yes" or "no",
  "project_type": one of the categories above as a string,
  "confidence": "high" / "medium" / "low",
  "reasoning": "1-3 sentence explanation covering both the type classification and quality assessment"
}}

Criteria for well_built_project "yes":
- Has clear purpose and documentation
- Structured codebase (src/, tests/, config files, CI/CD)
- Not just a single script or personal notes
- Shows real software engineering practices

Criteria for well_built_project "no":
- Homework submissions, university assignments
- Personal notes or backups
- Random files with no clear purpose
- Minimal or no documentation
- Single script with no structure
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load CSV
    if not os.path.exists(CSV_FILE):
        log.error(f"CSV file not found: {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    if "url" not in df.columns:
        log.error("CSV must have a 'url' column")
        return

    # Initialize new columns if not present
    if "test_tool" not in df.columns:
        df["test_tool"] = ""
    if "well_built_project" not in df.columns:
        df["well_built_project"] = ""
    if "project_type" not in df.columns:
        df["project_type"] = ""

    # Load existing JSON responses to support resume on restart
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            llm_responses = json.load(f)
        log.info(f"Loaded {len(llm_responses)} existing responses from {OUTPUT_JSON}")
    else:
        llm_responses = []

    # Make clone base dir if not exists
    os.makedirs(CLONE_BASE_DIR, exist_ok=True)

    for idx, row in df.iterrows():
        url = str(row.get("url", "")).strip()

        # Skip already-processed rows (all three result columns filled)
        if (pd.notna(row.get("test_tool")) and str(row.get("test_tool")).strip() != "" and
                pd.notna(row.get("well_built_project")) and str(row.get("well_built_project")).strip() != "" and
                pd.notna(row.get("project_type")) and str(row.get("project_type")).strip() != ""):
            log.info(f"Row {idx}: already processed, skipping.")
            continue

        name = str(row.get("name", "")).strip()

        if not url or not url.startswith("http"):
            log.warning(f"Row {idx}: Invalid URL '{url}', skipping.")
            df.at[idx, "test_tool"] = "no"
            df.at[idx, "well_built_project"] = "error"
            df.at[idx, "project_type"] = "error"
            entry = {"name": name, "url": url, "test_tool": "no",
                     "well_built_project": "error", "project_type": "error",
                     "reasoning": "Invalid URL"}
            llm_responses.append(entry)
            # ── Save both files immediately ──
            df.to_csv(CSV_FILE, index=False)
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(llm_responses, f, indent=2, ensure_ascii=False)
            continue

        url = normalize_github_url(url)
        repo_name = urlparse(url).path.strip("/")
        clone_dir = os.path.join(CLONE_BASE_DIR, clone_dir_name(repo_name))   # short path

        # ── Step 1: Clone ────────────────────────────────────────────────────
        success = clone_repo(url, clone_dir)
        if not success:
            df.at[idx, "test_tool"] = "no"
            df.at[idx, "well_built_project"] = "error"
            df.at[idx, "project_type"] = "error"
            entry = {"name": name, "url": url, "test_tool": "no",
                     "well_built_project": "error", "project_type": "error",
                     "reasoning": "Failed to clone repository"}
            llm_responses.append(entry)
            # ── Save both files immediately ──
            df.to_csv(CSV_FILE, index=False)
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(llm_responses, f, indent=2, ensure_ascii=False)
            log.info(f"Row {idx}: clone failed → saved and moving on.")
            continue

        # ── Step 2: Check test folder ────────────────────────────────────────
        # ── Step 2: Check test folder (csv column only, does NOT skip LLM) ──
        has_tests = has_test_folder(long_path(clone_dir))
        test_tool_value = "yes" if has_tests else "no"
        df.at[idx, "test_tool"] = test_tool_value
        log.info(f"Row {idx} | {url} | has_tests={has_tests}")

        # ── Step 3: Get structure + README (always) ──────────────────────────
        structure = get_project_structure(long_path(clone_dir))
        readme    = get_readme(long_path(clone_dir))

        # ── Step 4: Ask LLM (always, regardless of test_tool value) ──────────
        prompt   = build_prompt(url, structure, readme)
        response = ask_ollama(prompt)
        log.info(f"  -> LLM response: {response}")

        verdict = response.get("well_built_project", "no").lower()
        if verdict not in ("yes", "no"):
            verdict = "no"

        proj_type = response.get("project_type", "other").lower().strip()
        valid_types = ("library", "framework", "application", "training_tool",
                       "deployment_tool", "homework", "personal_backup", "other")
        if proj_type not in valid_types:
            proj_type = "other"

        df.at[idx, "well_built_project"] = verdict
        df.at[idx, "project_type"] = proj_type
        entry = {"name": name, "url": url, "test_tool": test_tool_value,
                 "well_built_project": verdict, "project_type": proj_type,
                 **{k: v for k, v in response.items()
                    if k not in ("well_built_project", "project_type")}}

        llm_responses.append(entry)

        # ── Step 5: Save CSV + JSON immediately after this project ───────────
        df.to_csv(CSV_FILE, index=False)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(llm_responses, f, indent=2, ensure_ascii=False)
        log.info(f"  → CSV and JSON saved.")

        # ── Step 6: Delete cloned repo to free disk space ────────────────────
        remove_dir(clone_dir)
        log.info(f"  → Deleted clone: {clone_dir}")

    log.info("Done! All projects processed.")
    print("\n=== Summary ===")
    print(df[["name", "url", "test_tool", "well_built_project", "project_type"]].to_string())


if __name__ == "__main__":
    main()