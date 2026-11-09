"""
Central Configuration for the PPOEM dataset tool
================================================
All configurable values are defined here.
Sensitive values (tokens, credentials) are loaded from .env via python-dotenv.
"""

import os
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:                      # the tool still runs without python-dotenv
    def load_dotenv(*_a, **_k):
        return False

# ── Resolve project root (one level up from this file) ──────────────────────
BASE_DIR    = Path(__file__).parent.parent.resolve()
PYTHON_DIR  = BASE_DIR / "python"
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Load .env from project root
load_dotenv(BASE_DIR / ".env")

# ── GitHub API Tokens (loaded from .env) ─────────────────────────────────────
GITHUB_SEARCH_TOKEN: str = os.getenv("GITHUB_SEARCH_TOKEN", "")
GITHUB_CORE_TOKENS: list = [
    t.strip()
    for t in os.getenv("GITHUB_CORE_TOKENS", "").split(",")
    if t.strip()
]

# ── Ollama / LLM Judge ───────────────────────────────────────────────────────
OLLAMA_URL:  str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME:  str = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")

# ══════════════════════════════════════════════════════════════════════════════
# FILES
# ══════════════════════════════════════════════════════════════════════════════
# Step 1  github_scraper.py
LIBRARIES_CSV     = DATA_DIR / "ML_AI_LLM_models_loading_library_python.csv"   # reference table
PROJECTS_CSV      = DATA_DIR / "github_python_repos.csv"
YEAR_FILTER_CSV   = PROJECTS_CSV

# Step 2  github_python_repos_remain_filter.py
SCANNER_OUTPUT_CSV  = DATA_DIR / "github_library_usage_results.csv"            # all projects + decision
DETECTED_LIBS_CSV   = RESULTS_DIR / "detected_library_with_all_filter_github_project_python_projects.csv"
PROJECT_DETAILS_DIR = RESULTS_DIR / "github_library_usage_scanned_projects"   # one JSON per project
SCANNER_LOG_FILE    = PYTHON_DIR / "scanner.log"

# Step 3  llm_as_judge.py  (reads and updates DETECTED_LIBS_CSV)
LLM_RESPONSE_JSON = RESULTS_DIR / "llm_response.json"

# Step 4  build_dataset_ppoem.py
DATASET_JSON          = RESULTS_DIR / "dataset_PPOEM.json"
DATASET_SUMMARY_CSV   = RESULTS_DIR / "dataset_PPOEM_summary.csv"
DATASET_TRACE_JSON    = RESULTS_DIR / "dataset_PPOEM_resolution_trace.json"
DATASET_EXCLUDED_JSON = RESULTS_DIR / "dataset_PPOEM_excluded_calls.json"
DATASET_LOG_FILE      = PYTHON_DIR / "dataset_builder.log"

# ── GitHub Scraper Settings ───────────────────────────────────────────────────
SCRAPER_MIN_STARS        = 100
SCRAPER_MIN_FORKS        = 10
SCRAPER_MIN_COMMITS      = 100
SCRAPER_LANGUAGE         = "python"
SCRAPER_PER_PAGE         = 100
SCRAPER_MAX_RESULTS      = 1000
SCRAPER_PARALLEL_WORKERS = 10
SCRAPER_BATCH_SIZE       = 50
SCRAPER_START_STARS      = 100
SCRAPER_END_STARS        = 500000
SCRAPER_OUTPUT_FILE      = str(PROJECTS_CSV)

# ── Year Filter ───────────────────────────────────────────────────────────────
YEAR_FILTER_MIN_YEAR = 2026

# ── Repository cloning / scanning (Steps 2 and 4) ────────────────────────────
SCANNER_GIT_TIMEOUT    = 300
SCANNER_MAX_FILE_MB    = 10
SCANNER_MAX_FILES_REPO = 10000
SCANNER_CSV_ENCODING   = "utf-8-sig"
# Repositories are cloned into a SHORT folder (Windows limits paths to 260
# characters). Default: <system temp>/ppoem, e.g. C:\Users\<you>\AppData\Local\Temp\ppoem.
# Set PPOEM_TEMP_DIR in .env for an even shorter one, e.g. C:\ppoem
SCANNER_TEMP_DIR       = Path(os.getenv("PPOEM_TEMP_DIR") or os.path.join(tempfile.gettempdir(), "ppoem"))
SCANNER_SKIP_DIRS: set = {
    ".git", "__pycache__", ".venv", "venv", "env", ".env",
    "node_modules", ".tox", ".eggs", "build", "dist",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "site-packages", ".idea", ".vscode",
}
# Number of repositories processed in parallel (each in its own process)
SCANNER_WORKERS = int(os.getenv("SCANNER_WORKERS", "2"))
# A project whose worker dies (e.g. out of memory) is retried alone this many times
SCANNER_RETRIES = 1

# ── Variable resolution limits (per model-load call) ─────────────────────────
RESOLVER_MAX_DEPTH      = 15      # length of one resolution chain
RESOLVER_MAX_CANDIDATES = 12      # candidate values kept per call
RESOLVER_MAX_CALL_SITES = 30      # call sites followed for one parameter
RESOLVER_MAX_STEPS      = 4000    # work budget per call (keeps huge repos fast)
RESOLVER_MAX_PARSED_MODULES  = 600   # parsed files kept in memory (LRU); lower it on small machines
RESOLVER_PROJECT_TIME_BUDGET = 1200  # seconds; after that, remaining calls get a reduced budget

# ── Model filters ────────────────────────────────────────────────────────────
# scikit-learn: estimator families that count as models (Note 2).
# Allowed values: "classifier", "regressor", "cluster", "outlier_detector".
SKLEARN_MODEL_TYPES = ("classifier", "regressor", "cluster", "outlier_detector")

# Hugging Face Hub repo types that are NOT models (Note 6)
HF_NON_MODEL_REPO_TYPES = ("dataset", "space")

# transformers / diffusers classes whose from_pretrained() loads an asset that
# is not a model (tokenizers, processors, configs, schedulers).
NON_MODEL_CLASS_SUFFIXES = (
    "Tokenizer", "TokenizerFast", "Processor", "FeatureExtractor",
    "ImageProcessor", "ImageProcessorFast", "VideoProcessor", "Config",
    "GenerationConfig", "Scheduler",
)

# ── Dataset format (Step 4) ──────────────────────────────────────────────────
# The published dataset stores missing values as the string "null".
# Set to False to write real JSON null instead.
DATASET_NULL_AS_STRING = True
# Columns of the project CSV that are NOT copied into the dataset metadata
DATASET_EXCLUDED_COLUMNS = (
    "name", "full_name", "well_built_project", "project_type", "test_tool",
    "confidence", "reasoning", "step2_decision", "n_literal_models",
    "n_variable_models", "model_load_libraries", "analyzed_commit", "error", "n_python_files",
)
# Drop projects judged "no" by llm_as_judge.py (Step 3)
DATASET_EXCLUDE_NOT_WELL_BUILT = True
# Max characters of call_expression stored in the dataset
DATASET_MAX_CALL_CHARS = 300

# ── LLM Judge Settings ────────────────────────────────────────────────────────
LLM_CLONE_BASE_DIR  = SCANNER_TEMP_DIR / "judge"
LLM_OLLAMA_RETRIES  = 3
LLM_OLLAMA_BACKOFF  = 10
LLM_OLLAMA_TIMEOUT  = 300
LLM_TEMPERATURE     = 0.1

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
