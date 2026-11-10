# data/

Input tables used to build the PPOEM dataset. This folder contains the raw and intermediate
metadata produced by the GitHub scraping and repository filtering stages. The final dataset
release lives in the project root `dataset_paper/` folder.

| File | Produced by | Content |
|---|---|---|
| `ML_AI_LLM_models_loading_library_python.csv` | manual | reference table of the ML/AI/LLM libraries and how they load models (documentation; the detection rules are in `python/ppoem/rules.py`) |
| `github_python_repos.csv` | Step 1 | 1,065 repositories that meet the collection criteria, with metadata |
| `github_library_usage_results.csv` | Step 2 | every repository of Step 1 + `n_python_files`, `libraries_detected`, `model_load_libraries`, `n_literal_models`, `n_variable_models`, `analyzed_commit`, `step2_decision`, `error` |

## CSV columns

The two repository tables share the GitHub metadata columns below:

| Columns | Meaning |
|---|---|
| `name`, `full_name`, `owner` | Repository name, `owner/name` identifier, and owner |
| `description`, `url` | GitHub description and repository URL |
| `stars`, `forks`, `watchers`, `commits`, `contributors` | Repository activity and contributor counts |
| `open_issues`, `pull_requests_open`, `pull_requests_merged` | Issue and pull-request counts |
| `created_at`, `updated_at`, `last_commit_date` | Repository creation, update, and latest-commit timestamps |
| `language`, `size_kb`, `default_branch`, `topics` | Primary language, size, default branch, and topic labels |
| `license_key`, `license_name`, `license_spdx_id`, `license_url` | License identifiers, display name, SPDX ID, and API URL |
| `has_issues`, `has_projects`, `has_pages`, `has_wiki`, `has_downloads` | GitHub feature flags |
| `archived`, `disabled`, `visibility` | Repository state and visibility |

`github_library_usage_results.csv` additionally records `project` (repository identifier),
`libraries_detected` (semicolon-separated imported target libraries), and the Step 2 analysis
fields: `n_python_files`, `model_load_libraries`, `n_literal_models`, `n_variable_models`,
`analyzed_commit`, `step2_decision`, and `error`. The exact columns can vary between historical
runs as the scanner adds fields.

`ML_AI_LLM_models_loading_library_python.csv` is a reference catalog. Its columns are
`language`, `library_framework`, `model_types_supported`, `library_stem`, `import_include`,
`load_run_model`, `documentation_page`, `different_ways_import`, `ways_load_model`, and
`resources_from_library_is_ML_library`; they describe supported libraries, import examples,
model-loading examples, documentation, and supporting resources.
