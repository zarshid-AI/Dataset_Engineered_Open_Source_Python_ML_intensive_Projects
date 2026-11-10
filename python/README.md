# python/

Runnable pipeline scripts, the central configuration and the `ppoem` analysis engine. For the
analysis internals and file-by-file module overview, see `ppoem/README.md`.

| File | Step | Reads | Writes |
|---|---|---|---|
| `config.py` | – | `.env` (project root) | – |
| `github_scraper.py` | 1 | GitHub Search + Core API | `data/github_python_repos.csv` |
| `github_python_repos_remain_filter.py` | 2 | `data/github_python_repos.csv` | `data/github_library_usage_results.csv` (all projects + decision), `results/detected_library_with_all_filter_github_project_python_projects.csv` (kept projects), `results/github_library_usage_scanned_projects/<owner__repo>.json` |
| `llm_as_judge.py` | 3 | kept-projects CSV | same CSV (judge columns), `results/llm_response.json` |
| `build_dataset_ppoem.py` | 4 | kept-projects CSV + per-project JSON | `results/dataset_PPOEM*.json / .csv` |

The input and output schemas are documented in
[`data/README.md`](../data/README.md) and [`results/README.md`](../results/README.md).

## Step 1: repository scraping

`github_scraper.py` is the GitHub repository discovery stage. It reads values from
`config.py` and `.env`, then scans GitHub by star count and repository metadata to collect
candidate Python ML/AI projects.

Actual run:

```bash
python github_scraper.py
```

What it does:

- loads `GITHUB_SEARCH_TOKEN`, `GITHUB_CORE_TOKENS`, and the scraper thresholds from
  `config.py`
- filters by language, stars, forks, commits, recent last-commit date, and repository health
- writes the output CSV defined by `SCRAPER_OUTPUT_FILE` (typically `data/github_python_repos.csv`)
- includes metadata such as last commit date, PR counts, license, contributors and repository info

There are no CLI flags in the script; configuration is controlled in `config.py` and the
environment file.

## Step 2: library usage and model-load filtering

`github_python_repos_remain_filter.py` clones each repository collected in Step 1 and runs the
PPOEM AST analysis over its production Python files.

```bash
python github_python_repos_remain_filter.py                    # all projects, resumes automatically
python github_python_repos_remain_filter.py --workers 4      # parallel scan
python github_python_repos_remain_filter.py --limit 10        # scan only the first N rows
python github_python_repos_remain_filter.py --projects owner/repo other/repo
python github_python_repos_remain_filter.py --restart        # discard previous Step 2 results
python github_python_repos_remain_filter.py --retry-errors   # rerun only errored projects
```

Outputs:

- `data/github_library_usage_results.csv` — full Step 2 result table for all projects
- `results/detected_library_with_all_filter_github_project_python_projects.csv` — kept projects
- `results/github_library_usage_scanned_projects/<owner__repo>.json` — detailed per-project analysis

Decision per project (`step2_decision`):

- `kept`
- `excluded: no target library imported and called`
- `excluded: library used but no model loaded`
- `excluded: no Python files in the repository`
- `error` (clone failure, worker crash, etc.)

## Step 3: LLM judge

`llm_as_judge.py` takes the kept projects from Step 2, clones each repository, inspects the
structure and README, and asks Ollama to classify the project type and decide whether the repo
is a serious, well-built project.

Actual usage:

```bash
python llm_as_judge.py
```

Important behavior from the script:

- reads `CSV_FILE = cfg.DETECTED_LIBS_CSV` and writes to `OUTPUT_JSON = cfg.LLM_RESPONSE_JSON`
- reuses prior JSON responses to resume interrupted runs
- clones each candidate repo into `CLONE_BASE_DIR` (configured in `config.py`)
- checks for a test directory or Python test patterns via `has_test_folder()`
- fills `test_tool`, `well_built_project`, and `project_type` columns in the CSV
- writes both the CSV and JSON immediately after each processed project
- removes the clone after judgment to save disk space

The judge expects Ollama to be running locally and uses the model configured in `MODEL_NAME`
(from `config.py` / `.env`). It sends a structured prompt and accepts answers that map to the
project type categories (`library`, `framework`, `application`, `training_tool`, etc.) and a
`well_built_project` verdict of `yes` or `no`.

## Step 4: PPOEM dataset construction

`build_dataset_ppoem.py` builds the final dataset from the Step 2 + Step 3 outputs. It reuses
cached extraction results when possible and clones repositories again only when needed.

```bash
python build_dataset_ppoem.py                       # build the dataset from Step 2/3 outputs
python build_dataset_ppoem.py --reextract           # force re-clone and re-analyse at the recorded commit
python build_dataset_ppoem.py --include-not-well-built
python build_dataset_ppoem.py --projects owner/repo
python build_dataset_ppoem.py --workers 4
```

Outputs:

- `results/dataset_PPOEM.json` — final PPOEM dataset
- `results/dataset_PPOEM_summary.csv` — one row per project, with inclusion or drop reason
- `results/dataset_PPOEM_statistics.csv` — aggregate counts for reporting
- `results/dataset_PPOEM_resolution_trace.json` — per-call resolution trace
- `results/dataset_PPOEM_excluded_calls.json` — calls filtered out for non-model reasons

This step is the project’s final assembly stage: it keeps only well-built projects (unless
`--include-not-well-built` is passed), resolves model names across each repository, and emits the
final dataset used for downstream analysis and publication.

## Using the engine directly

```python
import sys; sys.path.insert(0, "python")
from pathlib import Path
from ppoem.extractor import extract_project

ex = extract_project(Path("path/to/cloned/repo"))
ex.literal_string     # models with known names (dataset format)
ex.variable_based     # model references that depend on run-time values
ex.excluded           # calls removed by the model filters, with the reason
ex.trace              # resolution chain of every model-loading call
```

## Adding a library or an API

Add a `LoadRule` to `ppoem/rules.py` (qualified-name pattern, where the model argument is,
kind of model) and its import root to `LIBRARIES`. Then add a test in `tests/`.

The generated regular-expression reference for these rules is
[`dataset_paper/regex_catalog.md`](../dataset_paper/regex_catalog.md).
