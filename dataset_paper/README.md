# Dataset paper artifacts

This folder contains the publication-oriented PPOEM dataset bundle. It is separate from
`results/`, which contains intermediate pipeline outputs and diagnostic traces.

## Files

### `dataset_PPOEM.json`

The canonical dataset. The top-level keys are GitHub repository identifiers in the form
`owner/repo`; each value is a list of records. A record may contain:

- `literal_string`: model entries whose value was resolved statically.
- `variable_based`: model entries whose value remains runtime-dependent.
- `llm_judge`: project classification produced by Step 3.
- `libraries_detected`: semicolon-separated libraries found in the repository.

Model-entry objects identify the `library`, model or variable expression, source and call-site
files/lines, `call_expression`, and explanatory `notes`. Variable-based entries also include
`variable_source`.

### `scanned_github_systems.csv`

The inventory of GitHub repositories scanned for the study. Columns:

| Column | Meaning |
|---|---|
| `project`, `owner` | `owner/repo` identifier and owner |
| `description`, `url` | Repository description and URL |
| `stars`, `forks`, `watchers`, `commits`, `open_issues`, `pull_requests_open`, `contributors` | GitHub activity and contributor counts |
| `created_at`, `updated_at`, `last_commit_date` | Repository and commit timestamps |
| `language`, `size_kb`, `default_branch`, `topics` | Language, size, branch, and topics |
| `license_key`, `license_name`, `license_spdx_id`, `license_url` | License metadata |
| `has_issues`, `has_projects`, `has_pages`, `has_wiki`, `has_downloads` | GitHub feature flags |
| `archived`, `disabled`, `visibility` | Repository state and visibility |

### `regex_catalog.md`

A generated reference for the detector rules in `python/ppoem/rules.py`. It explains the
regular expressions used to recognize imports and model-loading calls for each supported
library, including the qualified call name, model argument position or keyword, and fallback
patterns. It is useful for reviewing coverage and debugging detection decisions; the Python
rules file remains the executable source of truth.
