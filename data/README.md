# data/

Input tables used to build the PPOEM dataset. This folder contains the raw and intermediate
metadata produced by the GitHub scraping and repository filtering stages. The final dataset
release lives in the project root `dataset_paper/` folder.

| File | Produced by | Content |
|---|---|---|
| `ML_AI_LLM_models_loading_library_python.csv` | manual | reference table of the ML/AI/LLM libraries and how they load models (documentation; the detection rules are in `python/ppoem/rules.py`) |
| `github_python_repos.csv` | Step 1 | 1,065 repositories that meet the collection criteria, with metadata |
| `github_library_usage_results.csv` | Step 2 | every repository of Step 1 + `n_python_files`, `libraries_detected`, `model_load_libraries`, `n_literal_models`, `n_variable_models`, `analyzed_commit`, `step2_decision`, `error` |
