# results/

Execution artifacts generated while running the PPOEM pipeline. The current repository snapshot
contains the generated output files below; the final published dataset is exported to
`dataset_paper/dataset_PPOEM.json`.

| File / folder | Produced by | Content |
|---|---|---|
| `detected_library_with_all_filter_github_project_python_projects.csv` | Step 2 (+ Step 3 columns) | projects that load at least one model; input of the LLM judge and of Step 4 |
| `llm_response.json` | Step 3 | raw LLM-judge answers |
| `dataset_PPOEM.json` | Step 4 | pipeline-produced dataset used to build the final releases |
| `dataset_PPOEM_statistics.csv` | Step 4 | counts for the paper (projects with literal / variable models, distinct models, per library) |
| `dataset_PPOEM_summary.csv` | Step 4 | one row per project: included or dropped (and why), number of entries, commit |
| `dataset_PPOEM_resolution_trace.json` | Step 4 | for every model-loading call: status, values, where resolution stopped, chain of steps with file:line |

## CSV columns

`detected_library_with_all_filter_github_project_python_projects.csv` contains the GitHub
metadata described in [`data/README.md`](../data/README.md), plus:

| Columns | Meaning |
|---|---|
| `libraries_detected`, `model_load_libraries` | Detected libraries and libraries with model-load calls |
| `n_literal_models`, `n_variable_models` | Counts of statically resolved and variable-based model entries |
| `step2_decision` | Step 2 keep/exclude/error decision |
| `test_tool`, `well_built_project`, `project_type`, `confidence`, `reasoning` | Step 3 LLM-judge classification and explanation |
| `llm_annotation_source` | Script/model source used for the annotation |

`dataset_PPOEM_summary.csv` has `full_name`, `status`, `n_literal_string`,
`n_variable_based`, `libraries_detected`, and `commit`. `dataset_PPOEM_statistics.csv` is a
two-column aggregate table: `measure` names a reported metric and `count` gives its value.

## JSON contents

- `llm_response.json` is a list of judge responses with `name`, `url`, `test_tool`,
  `well_built_project`, `project_type`, `confidence`, and `reasoning`.
- `dataset_PPOEM.json` maps each `owner/repo` to dataset records containing model entries,
  repository metadata, `libraries_detected`, and `llm_judge` annotations. Model entries use
  `literal_string` or `variable_based`; their fields identify the library, model/value
  expression, source file and line, call expression, and explanatory notes.
- `dataset_PPOEM_resolution_trace.json` maps each repository to one record per detected call.
  Each record includes `file`, `line`, `library`, `call_expression`, `status`, resolved `values`,
  optional `model_variable`/`stopped_at`, and `chains` showing the resolution path.
