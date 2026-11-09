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

