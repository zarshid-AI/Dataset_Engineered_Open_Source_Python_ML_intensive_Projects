# PPOEM analysis engine

This package contains the static-analysis engine used by the PPOEM pipeline to detect,
resolve, and explain ML model-loading calls in Python repositories.

## What it does

The engine scans a cloned repository, indexes production Python files, matches model-loading
APIs, and resolves the model name or model artifact from project code. It can distinguish:

- literal model values such as `"gpt-4o"` or `"bert-base-uncased"`
- model references that are assigned through variables or configuration objects
- excluded calls, such as dataset loaders or optimizer checkpoints, that are not model loads
- resolution traces showing how the analyzer linked the call site back to the source value

## Core modules

| Module | Responsibility |
|---|---|
| `index.py` | lazy AST index of the repository: imports, assignments, classes, functions and call sites |
| `rules.py` | library API rules and model-loading signatures for scikit-learn, PyTorch, TensorFlow, Hugging Face, OpenAI, and others |
| `sklearn_models.py` | scikit-learn estimator families that count as model classes |
| `resolver.py` | project-wide static resolution of variable and configuration values |
| `filters.py` | rejects non-model artifacts such as datasets, training state, or implausible values |
| `extractor.py` | end-to-end extraction for one repository, returning the final dataset entries and trace |
| `pipeline.py` | clone, analyze and cache workflow for project-level scanning |
| `repo.py` | repository clone and cleanup logic |
| `common.py` | shared configuration, helper utilities and default rules |

## Typical usage

```python
import sys
from pathlib import Path

sys.path.insert(0, "python")
from ppoem.extractor import extract_project

result = extract_project(Path("path/to/cloned/repo"))
print(result.literal_string)
print(result.variable_based)
print(result.excluded)
print(result.trace)
```

## Relationship to the project

- `python/github_scraper.py` collects candidate repositories from GitHub.
- `python/github_python_repos_remain_filter.py` uses this engine to find repositories that import and call ML libraries.
- `python/build_dataset_ppoem.py` exports the final PPOEM dataset from the resolved results.

This package is therefore the analysis core behind the repository-level detection and the final
machine-learning model dataset.

## Rule reference

[`dataset_paper/regex_catalog.md`](../../dataset_paper/regex_catalog.md) is a generated,
human-readable snapshot of the import patterns and model-load patterns currently declared in
`rules.py`. Each library section lists the import regex, the qualified-call regex, the model
argument location, and any text-extraction fallback. Update `rules.py` first and regenerate the
catalog when detection rules change; the catalog is documentation, not the runtime source.
