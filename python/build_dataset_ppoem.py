"""
Step 4 - Build the PPOEM dataset
================================
Takes the projects kept in Step 2 and judged in Step 3, runs the AST model
extraction with project-wide variable resolution on each of them, and writes
the dataset in the PPOEM format:

{
  "owner/repo": [
    {
      "variable_based": [ {library, variable, file, variable_line, call_expression, notes, variable_source} ],
      "literal_string": [ {library, model, model_variable, model_file, variable_file,
                           model_line, variable_line, call_expression, notes} ],
      "owner": ..., "description": ..., ...          # repository metadata (Step 1)
      "libraries_detected": "openai; torch"
    }
  ]
}

The extraction of Step 2 is reused when it was made by the same extractor
version; otherwise (or with --reextract) the repository is cloned again at the
commit analysed in Step 2, so the dataset always matches that code.

Input
    results/detected_library_with_all_filter_github_project_python_projects.csv  (Steps 2 + 3)
    results/github_library_usage_scanned_projects/<owner__repo>.json            (Step 2 cache)
Output
    results/dataset_PPOEM.json                        the dataset
    results/dataset_PPOEM_summary.csv                 one row per project (included / dropped)
    results/dataset_PPOEM_statistics.csv              counts for the paper
    results/dataset_PPOEM_resolution_trace.json       resolution chain of every model call
    results/dataset_PPOEM_excluded_calls.json         calls removed by the model filters (with reason)

Usage
    python build_dataset_ppoem.py
    python build_dataset_ppoem.py --reextract           # clone and analyse again (pinned commit)
    python build_dataset_ppoem.py --include-not-well-built
    python build_dataset_ppoem.py --projects owner/repo
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd   

import config as cfg   
from ppoem.extractor import EXTRACTOR_VERSION  
from ppoem.pipeline import full_name_of, json_value, load_cache, run_jobs, setup_logging   

log = setup_logging("step4", cfg.DATASET_LOG_FILE, cfg.LOG_LEVEL)

STATS_CSV = cfg.RESULTS_DIR / "dataset_PPOEM_statistics.csv"
LITERAL_FIELDS = ["library", "model", "model_variable", "model_file", "variable_file",
                  "model_line", "variable_line", "call_expression", "notes"]
VARIABLE_FIELDS = ["library", "variable", "file", "variable_line", "call_expression", "notes", "variable_source"]


def nullify(v):
    if v is None and cfg.DATASET_NULL_AS_STRING:
        return "null"
    return v


def format_entry(entry: dict, fields) -> dict:
    return {f: nullify(entry.get(f)) for f in fields}


TEXT_COLUMNS: set = set()


def metadata(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k in cfg.DATASET_EXCLUDED_COLUMNS:
            continue
        v = json_value(v)
        if v is None and k in TEXT_COLUMNS:                 # missing text -> "" as in the published dataset
            v = ""
        out[k] = v
    return out


def dataset_record(row: dict, ex: dict) -> dict:
    rec = {}
    if ex.get("variable_based"):
        rec["variable_based"] = [format_entry(e, VARIABLE_FIELDS) for e in ex["variable_based"]]
    if ex.get("literal_string"):
        rec["literal_string"] = [format_entry(e, LITERAL_FIELDS) for e in ex["literal_string"]]
    rec.update(metadata(row))
    rec["libraries_detected"] = "; ".join(ex.get("libraries_detected", []))
    return rec


def statistics(dataset: dict, summary: pd.DataFrame) -> pd.DataFrame:
    lit = {k for k, v in dataset.items() if v[0].get("literal_string")}
    var = {k for k, v in dataset.items() if v[0].get("variable_based")}
    entries = [e for v in dataset.values() for e in v[0].get("literal_string", [])]
    rows = [
        ("projects in dataset", len(dataset)),
        ("projects with literal model names", len(lit)),
        ("projects with variable-based references", len(var)),
        ("projects in both", len(lit & var)),
        ("projects with literal model names only", len(lit - var)),
        ("projects with variable-based references only", len(var - lit)),
        ("literal_string entries", len(entries)),
        ("variable_based entries", sum(len(v[0].get("variable_based", [])) for v in dataset.values())),
        ("literal_string entries resolved from a variable", sum(1 for e in entries if e["model_variable"] not in (None, "null"))),
        ("distinct models", len({(e["library"], e["model"]) for e in entries})),
        ("distinct sklearn model classes", len({e["model"] for e in entries if e["library"] == "sklearn"})),
        ("distinct non-sklearn models", len({(e["library"], e["model"]) for e in entries if e["library"] != "sklearn"})),
    ]
    for status, n in summary["status"].value_counts().items():
        rows.append((f"projects: {status}", int(n)))
    for lib, n in Counter(e["library"] for e in entries).most_common():
        rows.append((f"literal_string entries: {lib}", n))
    return pd.DataFrame(rows, columns=["measure", "count"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(cfg.DETECTED_LIBS_CSV))
    ap.add_argument("--projects", nargs="*")
    ap.add_argument("--reextract", action="store_true", help="clone and analyse again even if a cache exists")
    ap.add_argument("--include-not-well-built", action="store_true")
    ap.add_argument("--workers", type=int, default=cfg.SCANNER_WORKERS)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    TEXT_COLUMNS.update(c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
                        and not pd.api.types.is_bool_dtype(df[c]))
    df["full_name"] = [full_name_of(r) for r in df.to_dict("records")]
    if args.projects:
        df = df[df.full_name.str.lower().isin({p.lower() for p in args.projects})]
    summary = []
    if "well_built_project" in df.columns and cfg.DATASET_EXCLUDE_NOT_WELL_BUILT and not args.include_not_well_built:
        bad = df[df.well_built_project.astype(str).str.strip().str.lower() == "no"]
        for r in bad.to_dict("records"):
            summary.append({"full_name": r["full_name"], "status": "dropped: not well built (LLM judge)"})
        df = df.drop(bad.index)
    elif "well_built_project" not in df.columns:
        log.warning("No LLM-judge columns found (Step 3 not run): all projects are used.")

    rows = df.to_dict("records")
    extractions, todo = {}, []
    for r in rows:
        cached = None if args.reextract else load_cache(cfg.PROJECT_DETAILS_DIR, r["full_name"])
        if cached:
            extractions[r["full_name"]] = cached
        else:
            todo.append(r)
    log.info(f"Step 4 | {len(rows)} project(s); {len(extractions)} from Step 2 cache (extractor "
             f"{EXTRACTOR_VERSION}), {len(todo)} to clone and analyse")

    temp_dir = Path(cfg.SCANNER_TEMP_DIR) / "scan"
    temp_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for r in todo:
        commit = str(r.get("analyzed_commit") or "").strip()
        commit = commit if commit and commit.lower() != "nan" else None
        jobs.append((r["full_name"], str(r["url"]).strip(), temp_dir, cfg.PROJECT_DETAILS_DIR, commit,
                     cfg.GITHUB_SEARCH_TOKEN or None, cfg.SCANNER_GIT_TIMEOUT))
    counter = {"i": 0}

    def on_result(job, res):
        counter["i"] += 1
        extractions[job[0]] = res
        log.info(f"[{counter['i']}/{len(jobs)}] {job[0]} analysed ({res.get('seconds')}s) {res.get('error', '')}")

    run_jobs(jobs, args.workers, on_result, retries=cfg.SCANNER_RETRIES, log=log)

    dataset, trace, excluded = {}, {}, {}
    for r in rows:
        name = r["full_name"]
        ex = extractions.get(name, {})
        if ex.get("error"):
            summary.append({"full_name": name, "status": f"dropped: {ex['error']}"})
            continue
        if ex.get("excluded"):
            excluded[name] = ex["excluded"]
        if ex.get("trace"):
            trace[name] = ex["trace"]
        if not (ex.get("literal_string") or ex.get("variable_based")):
            summary.append({"full_name": name, "status": "dropped: no model-loading call"})
            continue
        dataset[name] = [dataset_record(r, ex)]
        summary.append({"full_name": name, "status": "included",
                        "n_literal_string": len(ex.get("literal_string", [])),
                        "n_variable_based": len(ex.get("variable_based", [])),
                        "libraries_detected": "; ".join(ex.get("libraries_detected", [])),
                        "commit": ex.get("commit", "")})

    cfg.DATASET_JSON.parent.mkdir(parents=True, exist_ok=True)
    cfg.DATASET_JSON.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    cfg.DATASET_TRACE_JSON.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    cfg.DATASET_EXCLUDED_JSON.write_text(json.dumps(excluded, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(cfg.DATASET_SUMMARY_CSV, index=False, encoding=cfg.SCANNER_CSV_ENCODING)
    stats = statistics(dataset, summary_df)
    stats.to_csv(STATS_CSV, index=False, encoding=cfg.SCANNER_CSV_ENCODING)

    log.info("\n" + "=" * 70)
    log.info("STEP 4 SUMMARY")
    for m, n in stats.values:
        log.info(f"  {m:<55} {n}")
    log.info(f"  dataset    -> {cfg.DATASET_JSON}")
    log.info(f"  statistics -> {STATS_CSV}")
    log.info(f"  summary    -> {cfg.DATASET_SUMMARY_CSV}")
    log.info(f"  trace      -> {cfg.DATASET_TRACE_JSON}")
    log.info(f"  excluded   -> {cfg.DATASET_EXCLUDED_JSON}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
