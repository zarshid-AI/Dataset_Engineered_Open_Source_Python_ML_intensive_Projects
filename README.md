# PPOEM — Python Project Open-source Engineered ML-intensive

A tool that builds a curated dataset of open-source Python projects that load or use machine
learning models, and that reports **which model** each project uses and **where** in the code.

```
GitHub API ──► Step 1  github_scraper.py                     
             Engineered Python repositories
                 │
                 ▼
             Step 2  github_python_repos_remain_filter.py    
             repositories that import + call an ML
                 │ 
                 │    (clone + AST, one pass)                
                 │     library AND load / use a model
                 ▼
             Step 3  llm_as_judge.py                         
             project type, well-built, test tool
                 │
                 ▼
             Step 4  build_dataset_ppoem.py                   
             results/dataset_PPOEM.json
```

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env               # GitHub token(s), Ollama URL

cd python
python github_scraper.py                         # Step 1
python github_python_repos_remain_filter.py      # Step 2  (resumable, --workers N)
python llm_as_judge.py                           # Step 3  (needs Ollama)
python build_dataset_ppoem.py                    # Step 4

python -m pytest ../tests -q                     # unit tests of the analysis engine
```

Every step can be re-run: Step 2 continues where it stopped, Step 4 reuses the Step 2 analysis
(or re-clones the **same commit** with `--reextract`).

---

## Project structure

```
python/
├── config.py                             all settings (paths, thresholds, filters)
├── github_scraper.py                     Step 1  GitHub Search + Core API
├── github_python_repos_remain_filter.py  Step 2  library + model-load detection
├── llm_as_judge.py                       Step 3  LLM-as-a-judge (Ollama)
├── build_dataset_ppoem.py                Step 4  dataset in the PPOEM format
└── ppoem/                                analysis engine (shared by Steps 2 and 4)
    ├── index.py          lazy AST index of a repository (imports, scopes, classes, call sites)
    ├── rules.py          model-loading APIs of every library  ← add new libraries here
    ├── sklearn_models.py scikit-learn estimators that count as models
    ├── resolver.py       project-wide static resolution of model names
    ├── filters.py        datasets / training state / implausible values are not models
    ├── extractor.py      detection + resolution + dataset entries for one repository
    ├── pipeline.py       clone → analyse → cache, crash-safe parallel runner
    ├── repo.py           git clone (pinned commit, token in header) and delete
    └── common.py         settings, test-file rule, helpers
data/                     inputs and Step 1–2 tables
results/                  Step 2–4 outputs (see results/README.md)
```

---

## Dataset paper: `dataset_paper/` (main dataset)

The `dataset_paper/` folder is the project’s primary dataset release. It contains the final,
curated dataset used for analysis and reporting in the paper, along with the tracked GitHub
systems used to build it.

Current contents:

- `dataset_PPOEM.json` — the canonical PPOEM dataset, containing the resolved model-loading
  entries for the scanned repositories.
- `scanned_github_systems.csv` — the GitHub repository inventory and metadata used for the
  dataset construction workflow.

This folder is the main dataset for the project and should be treated as the authoritative
publication artifact. The `results/` folder contains intermediate and execution outputs created
while running the pipeline, while `dataset_paper/` stores the final dataset bundle used for the
study and downstream analysis.

---



---

## How a model is found

### 1. Model-loading calls (Step 2 and Step 4)

Every production `.py` file (test files and folders are skipped) is parsed with the Python
`ast` package. A call is a model-loading call when its **qualified name** matches a rule in
`ppoem/rules.py`. The qualified name is computed from the imports of the file (aliases
included) and from the object the call is made on:

| Code | Qualified name |
|---|---|
| `import torch as th; th.load(p)` | `torch.load` |
| `from transformers import AutoModel as AM; AM.from_pretrained(m)` | `transformers.AutoModel.from_pretrained` |
| `client = OpenAI(); client.chat.completions.create(model=m)` | `openai.OpenAI().chat.completions.create` |
| `self.client = Groq()` in `__init__`, `self.client.chat.completions.create(...)` | `groq.Groq().chat.completions.create` |

Libraries: scikit-learn, PyTorch, TensorFlow/Keras, transformers, diffusers, huggingface_hub,
OpenAI, Anthropic, Groq, Fireworks, Llama API, DeepSeek, vLLM, llama.cpp, XGBoost, Ollama,
Vertex AI, Google GenAI.

### 2. Which calls are models

| Rule | Example |
|---|---|
| scikit-learn: only classifiers, regressors, clustering and outlier-detection estimators | `RandomForestClassifier()` kept; `StandardScaler()`, `OneHotEncoder()`, `TfidfVectorizer()`, `SimpleImputer()` ignored |
| transformers / diffusers: tokenizers, processors, configs and schedulers are not models | `AutoTokenizer.from_pretrained(...)` ignored |
| Hugging Face Hub: datasets and spaces are not models | `snapshot_download("squad", repo_type="dataset")` excluded |
| files that hold data or training state are not models | `torch.load("train_data.pt")`, `torch.load(optimizer_path)` excluded |

Every excluded call is listed with its reason in `results/dataset_PPOEM_excluded_calls.json`.

### 3. Model name when it is a variable (project-wide resolution)

The whole repository is indexed (assignments, imports, class definitions, functions and call
sites). Starting from the expression passed to the API, the definition is followed **across
files**:

* aliases, string concatenation, f-strings, `%`, `str.format()`, `split()[i]`, `replace()`, …
* paths: `os.path.join(dir, "best.pt")`, `Path(d) / "best.pt"`, `.parent`, `__file__`
* constants imported from other modules (also re-exported in `__init__.py`)
* class attributes set in the class body or in any method (`self.model`), base classes,
  `@property`, and constructor keywords of dataclass / pydantic classes (`Agent(model_name=…)`)
* dictionary entries with constant keys, `dict.get(key, default)`
* return values of project functions — with the arguments of **that** call
* function parameters: default value and the argument passed at every **linked** call site in
  production code (same module, imported function, `self`, an instance of the class)
* every branch of `a if c else b` and `a or b`; assignments in the other branch of an
  `if/else` are never mixed up

Only definitions that are really linked to the call are followed: a method called `load` in
an unrelated class, or a call site in a test file, is never used. Resolution stops when the
value only exists at run time; the source expression is recorded, plus the default value
declared in the code:

| Source | Result |
|---|---|
| `os.environ.get("FLUX_DIR", "black-forest-labs/FLUX.1-dev")` | model `black-forest-labs/FLUX.1-dev` (default of the environment variable) |
| `args.model` with `parser.add_argument("--model", default="gpt-4o")` | model `gpt-4o` (default of the command-line option) |
| `os.environ["MODEL"]`, `cfg.model`, `json.load(f)["model"]`, `input()` | `variable_based` entry, `variable_source` = the expression |
| `weights["detection_weights.pt"]` (container unknown) | model `detection_weights.pt` (the key names the file) |

---

## Dataset format (`results/dataset_PPOEM.json`)

```json
{
  "owner/repo": [
    {
      "variable_based": [
        {
          "library": "vllm",
          "variable": "os.environ['VLLM_MODEL']",
          "file": "serve.py",
          "variable_line": 11,
          "call_expression": "LLM(model=os.environ['VLLM_MODEL'])",
          "notes": "Matched vllm.LLM: os.environ['VLLM_MODEL'] (variable reference; value depends on environment variable)",
          "variable_source": "os.environ['VLLM_MODEL']"
        }
      ],
      "literal_string": [
        {
          "library": "huggingface_hub",
          "model": "OpenVINO/whisper-tiny-fp16-ov",
          "model_variable": "model_id",
          "model_file": "whisper_live/transcriber/transcriber_openvino.py",
          "variable_file": "whisper_live/transcriber/transcriber_openvino.py",
          "model_line": 11,
          "variable_line": 21,
          "call_expression": "hf_hub.snapshot_download(model_id, local_dir=model_path)",
          "notes": "Matched huggingface_hub.download: OpenVINO/whisper-tiny-fp16-ov (default value of parameter, resolved from model_id)"
        },
        {
          "library": "sklearn",
          "model": "SVC",
          "model_variable": "null",
          "model_file": "src/squlearn/qrc/qrc_classifier.py",
          "variable_file": "null",
          "model_line": 129,
          "variable_line": "null",
          "call_expression": "SVC(**self.ml_model_options)",
          "notes": "sklearn estimator instantiation: SVC"
        }
      ],
      "owner": "...", "description": "...", "url": "...", "stars": 100, "...": "...",
      "libraries_detected": "huggingface_hub; vllm; sklearn"
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `model` | model name, model id, model file or estimator class |
| `model_variable` | expression passed to the call (`"null"` when the model is written in the call) |
| `model_file`, `model_line` | where the model name is written |
| `variable_file`, `variable_line` | the model-loading call (`"null"` when the model is written in the call) |
| `variable_source` | (`variable_based`) where resolution stopped: run-time value |

Missing values are written as the string `"null"`, like the published dataset
(`DATASET_NULL_AS_STRING = False` in `config.py` writes JSON `null`).

---

## Configuration (`python/config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `SCRAPER_MIN_STARS / FORKS / COMMITS` | 100 / 10 / 100 | Step 1 thresholds |
| `YEAR_FILTER_MIN_YEAR` | 2026 | last commit year |
| `SCANNER_WORKERS` | 2 | repositories analysed in parallel (each in its own process) |
| `SCANNER_RETRIES` | 1 | retries of a project whose process died |
| `RESOLVER_MAX_PARSED_MODULES` | 600 | parsed files kept in memory; lower it on small machines |
| `RESOLVER_PROJECT_TIME_BUDGET` | 1200 s | after that, remaining calls get a reduced budget |
| `SKLEARN_MODEL_TYPES` | classifier, regressor, cluster, outlier_detector | scikit-learn families counted as models |
| `HF_NON_MODEL_REPO_TYPES` | dataset, space | Hugging Face repo types that are not models |
| `NON_MODEL_CLASS_SUFFIXES` | Tokenizer, Processor, Config, Scheduler, … | `from_pretrained` classes that are not models |
| `DATASET_EXCLUDE_NOT_WELL_BUILT` | True | drop projects judged "no" in Step 3 |
| `DATASET_NULL_AS_STRING` | True | `"null"` vs JSON `null` |

Memory: the largest repositories in the study (e.g. vllm, 6,000 files) need about 700 MB per
worker. With 8 GB of RAM, use 4–6 workers.

