"""
Step 2 - ML/AI library usage and model-loading detection
========================================================
Clones every repository collected in Step 1 and, with one AST analysis of all
its production .py files (test files skipped), detects at the same time:

  * which ML/AI libraries are imported AND called      -> libraries_detected
  * which calls load / instantiate / query a model     -> model entries
    (literal model names and variable-based references, resolved across the
    whole project, see ppoem/resolver.py)

A repository is kept when at least one model-loading call is found.
The analysed commit is recorded so Step 4 analyses exactly the same code.

Input
    data/github_python_repos.csv                                  (Step 1)
Output
    data/github_library_usage_results.csv        every project + decision
    results/detected_library_with_all_filter_github_project_python_projects.csv
                                                 kept projects (input of Step 3)
    results/github_library_usage_scanned_projects/<owner__repo>.json
                                                 library usage + model extraction
    python/scanner.log

Usage
    python github_python_repos_remain_filter.py                  # all projects, resumes
    python github_python_repos_remain_filter.py --workers 4      # parallel
    python github_python_repos_remain_filter.py --limit 10
    python github_python_repos_remain_filter.py --projects owner/repo other/repo
    python github_python_repos_remain_filter.py --restart        # ignore previous results
    python github_python_repos_remain_filter.py --retry-errors   # re-run only the projects with an error
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd  # noqa: E402

import config as cfg  # noqa: E402
from ppoem.pipeline import full_name_of, run_jobs, setup_logging  # noqa: E402
from ppoem.repo import remove  # noqa: E402

log = setup_logging("step2", cfg.SCANNER_LOG_FILE, cfg.LOG_LEVEL)

NEW_COLUMNS = ["n_python_files", "libraries_detected", "model_load_libraries", "n_literal_models", "n_variable_models",
               "analyzed_commit", "step2_decision", "error"]
JUDGE_COLUMNS = ["test_tool", "well_built_project", "project_type", "confidence", "reasoning"]


def decision(res: dict) -> str:
    if res.get("error"):
        return "error"
    if res.get("literal_string") or res.get("variable_based"):
        return "kept"
    if not res.get("n_files"):
        return "excluded: no Python files in the repository"
    if res.get("libraries_detected"):
        return "excluded: library used but no model loaded"
    return "excluded: no target library imported and called"


def result_row(row: dict, res: dict) -> dict:
    out = dict(row)
    out["n_python_files"] = res.get("n_files", 0)
    out["libraries_detected"] = "; ".join(res.get("libraries_detected", []))
    out["model_load_libraries"] = "; ".join(res.get("model_libraries", []))
    out["n_literal_models"] = len(res.get("literal_string", []))
    out["n_variable_models"] = len(res.get("variable_based", []))
    out["analyzed_commit"] = res.get("commit", "")
    out["step2_decision"] = decision(res)
    out["error"] = res.get("error", "")
    return out


def write_kept(all_rows: pd.DataFrame):
    kept = all_rows[all_rows["step2_decision"] == "kept"].copy()
    # keep Step 3 answers if the judge already ran on these projects
    if cfg.DETECTED_LIBS_CSV.exists():
        try:
            old = pd.read_csv(cfg.DETECTED_LIBS_CSV)
            cols = [c for c in JUDGE_COLUMNS if c in old.columns]
            if cols and "full_name" in old.columns:
                kept = kept.drop(columns=[c for c in cols if c in kept.columns]).merge(
                    old[["full_name"] + cols], on="full_name", how="left")
        except Exception as exc:
            log.warning(f"Could not carry over judge columns: {exc}")
    cfg.DETECTED_LIBS_CSV.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(cfg.DETECTED_LIBS_CSV, index=False, encoding=cfg.SCANNER_CSV_ENCODING)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(cfg.PROJECTS_CSV))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--projects", nargs="*", help="owner/repo names to (re)analyse")
    ap.add_argument("--workers", type=int, default=cfg.SCANNER_WORKERS)
    ap.add_argument("--restart", action="store_true", help="discard previous Step 2 results")
    ap.add_argument("--retry-errors", action="store_true",
                    help="analyse again the projects whose previous run ended with an error")
    args = ap.parse_args()

    projects = pd.read_csv(args.input)
    projects["full_name"] = [full_name_of(r) for r in projects.to_dict("records")]
    if args.projects:
        wanted = {p.lower() for p in args.projects}
        projects = projects[projects.full_name.str.lower().isin(wanted)]
    if args.limit:
        projects = projects.head(args.limit)

    out_csv = Path(cfg.SCANNER_OUTPUT_CSV)
    done = pd.DataFrame()
    if out_csv.exists() and not args.restart:
        done = pd.read_csv(out_csv)
        if args.retry_errors and len(done) and "step2_decision" in done.columns:
            n_err = int((done.step2_decision == "error").sum())
            done = done[done.step2_decision != "error"]
            log.info(f"Retrying {n_err} project(s) that ended with an error")
        if args.projects:                                  # re-analyse the listed projects
            done = done[~done.full_name.str.lower().isin({p.lower() for p in args.projects})]
    done_names = set(done.full_name) if len(done) else set()
    todo = [r for r in projects.to_dict("records") if r["full_name"] not in done_names]
    log.info(f"Step 2 | {len(projects)} project(s) selected, {len(projects) - len(todo)} already analysed, "
             f"{len(todo)} to analyse, workers={args.workers}")

    temp_dir = Path(cfg.SCANNER_TEMP_DIR) / "scan"
    temp_dir.mkdir(parents=True, exist_ok=True)
    token = cfg.GITHUB_SEARCH_TOKEN or None
    rows = done.to_dict("records") if len(done) else []
    started = datetime.now()

    def record(i, row, res):
        out = result_row(row, res)
        rows.append(out)
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding=cfg.SCANNER_CSV_ENCODING)  # checkpoint
        libs = out["libraries_detected"] or "-"
        log.info(f"[{i}/{len(todo)}] {row['full_name']}: {out['step2_decision']} | libraries: {libs} | "
                 f"models: {out['n_literal_models']} literal, {out['n_variable_models']} variable "
                 f"({res.get('seconds', 0)}s)")
        if res.get("error"):
            log.debug(res.get("traceback", res["error"]))

    by_name = {row["full_name"]: row for row in todo}
    jobs = [(row["full_name"], str(row.get("url", "")).strip(), temp_dir, cfg.PROJECT_DETAILS_DIR,
             None, token, cfg.SCANNER_GIT_TIMEOUT) for row in todo]
    counter = {"i": 0}

    def on_result(job, res):
        counter["i"] += 1
        record(counter["i"], by_name[job[0]], res)

    try:
        run_jobs(jobs, args.workers, on_result, retries=cfg.SCANNER_RETRIES, log=log)
    finally:
        for child in (temp_dir.iterdir() if temp_dir.exists() else []):
            remove(child)

    all_rows = pd.DataFrame(rows)
    if all_rows.empty:
        log.info("Nothing analysed.")
        return
    all_rows.to_csv(out_csv, index=False, encoding=cfg.SCANNER_CSV_ENCODING)
    kept = write_kept(all_rows)
    counts = all_rows.step2_decision.value_counts()
    log.info("\n" + "=" * 70)
    log.info("STEP 2 SUMMARY")
    for k, v in counts.items():
        log.info(f"  {k:<50} {v}")
    log.info(f"  kept projects -> {cfg.DETECTED_LIBS_CSV} ({len(kept)})")
    log.info(f"  all projects  -> {out_csv}")
    log.info(f"  details       -> {cfg.PROJECT_DETAILS_DIR}/")
    log.info(f"  elapsed       -> {datetime.now() - started}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
