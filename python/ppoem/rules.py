"""
Regular expressions for import and model-load detection
=======================================================
IMPORT_REGEX    one regular expression per library: the import statements of
                that library (`import torch`, `from transformers import X`, ...).
LoadRule.regex  one regular expression per model-loading API, matched against
                the *qualified name* of a call:

    import torch as th; th.load(p)                     -> torch.load
    from transformers import AutoModel as AM
    AM.from_pretrained(m)                              -> transformers.AutoModel.from_pretrained
    client = OpenAI(); client.chat.completions.create  -> openai.OpenAI().chat.completions.create

`model_args` says where the model is passed, as (position, keyword) pairs;
the first one present in the call is used.

To support a new library or API, add its import regex and a LoadRule here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .sklearn_models import sklearn_model_classes

ArgSpec = Tuple[Optional[int], Optional[str]]

# library key -> import roots (dotted module names)
LIBRARIES: Dict[str, Tuple[str, ...]] = {
    "sklearn":          ("sklearn",),
    "torch":            ("torch",),
    "tensorflow":       ("tensorflow", "keras"),
    "transformers":     ("transformers",),
    "openai":           ("openai",),
    "anthropic":        ("anthropic",),
    "groq":             ("groq",),
    "fireworks":        ("fireworks",),
    "llama_api_client": ("llama_api_client",),
    "deepseek":         ("deepseek",),
    "vllm":             ("vllm",),
    "llama_cpp":        ("llama_cpp",),
    "xgboost":          ("xgboost",),
    "ollama":           ("ollama",),
    "vertexai":         ("vertexai",),
    "genai":            ("google.genai", "google.generativeai"),
    "diffusers":        ("diffusers",),
    "huggingface_hub":  ("huggingface_hub",),
}


def _import_regex(roots: Sequence[str]) -> str:
    r = "|".join(re.escape(x) for x in roots)
    alts = [rf"import\s+(?:{r})(?:\.\w+)*\b", rf"from\s+(?:{r})(?:\.\w+)*\s+import\s"]
    for root in roots:                       # from google import genai
        if "." in root:
            parent, last = root.rsplit(".", 1)
            alts.append(rf"from\s+{re.escape(parent)}\s+import\s+(?:\w+\s*,\s*)*{re.escape(last)}\b")
    return r"^\s*(?:" + "|".join(alts) + ")"


# one import regular expression per library, applied to the import statements
# of a file (the statements themselves are found with the AST package)
IMPORT_REGEX: Dict[str, str] = {lib: _import_regex(roots) for lib, roots in LIBRARIES.items()}
# google.genai and google.generativeai are also imported as `from google import genai`
IMPORT_REGEX["genai"] = (r"^\s*(?:import\s+google\.(?:genai|generativeai)\b"
                         r"|from\s+google(?:\.(?:genai|generativeai)\S*\s+import\s|\s+import\s+(?:genai|generativeai)\b))")
_IMPORT_RX = {lib: re.compile(rx) for lib, rx in IMPORT_REGEX.items()}


def library_of_import(statement: str) -> Optional[str]:
    """Library imported by one import statement (e.g. 'from torch.nn import Linear')."""
    for lib, rx in _IMPORT_RX.items():
        if rx.match(statement):
            return lib
    return None


def library_of(qualname: str) -> Optional[str]:
    """Library of a qualified name ('torch.load' -> 'torch')."""
    for lib, roots in LIBRARIES.items():
        for r in roots:
            if qualname == r or qualname.startswith(r + ".") or qualname.startswith(r + "("):
                return lib
    return None


@dataclass(frozen=True)
class LoadRule:
    rule_id: str
    library: str
    regex: str                                # matched against the full qualified call name
    kind: str                                 # model_id | model_file | model_class
    model_args: Tuple[ArgSpec, ...] = ()
    extra_args: Tuple[ArgSpec, ...] = ()      # joined to the model value (torch.hub repo:model)
    extra_join: str = ":"
    extra_before: bool = True
    fixed_model: Optional[str] = None         # model name when the API names no model (xgboost.train)
    exclude_class_suffixes: bool = False      # skip Tokenizer / Processor / Scheduler ...
    class_allowlist: Optional[FrozenSet[str]] = None
    artifact_check: bool = False              # skip data / optimizer files (Note 6)
    repo_type_arg: Optional[ArgSpec] = None   # Hugging Face repo_type (Note 6)
    fallback_regex: Optional[str] = None      # call chain accepted when the client object is unknown
    fallback_requires: Tuple[str, ...] = ()   # ... if the file imports one of these libraries
    value_regex: Optional[str] = None         # source text of the call -> literal model name (group "value")

    def matches(self, qualname: str) -> bool:
        return _rx(self.regex).fullmatch(qualname) is not None

    def matches_fallback(self, chain: str) -> bool:
        return bool(self.fallback_regex) and _rx(self.fallback_regex).search(chain) is not None

    def literal_value(self, call_text: str) -> Optional[str]:
        """Model name written as a string in the call, read with value_regex."""
        if not self.value_regex:
            return None
        m = _rx(self.value_regex).search(call_text)
        return m.group("value") if m else None


_COMPILED: Dict[str, "re.Pattern"] = {}


def _rx(pattern: str) -> "re.Pattern":
    rx = _COMPILED.get(pattern)
    if rx is None:
        rx = _COMPILED[pattern] = re.compile(pattern)
    return rx


# ── OpenAI-compatible SDKs (OpenAI, Groq, Fireworks, Llama API, DeepSeek): 3 regexes each
#  1. qualified call name: the library or one of its client objects, then any API path
#     ending in create / generate / parse / stream (chat.completions.create, embeddings.create, ...)
#  2. value_regex on the source text of the call: the model name written as a string
#  3. fallback on the call chain when the client object cannot be identified
_API_END = r"(?:\.\w+(?:\(\))?)*\.(?:create|acreate|generate|parse|stream|edit|create_variation)"
_CHAT_FALLBACK = r"\.(?:\w+\.)*(?:completions|embeddings|responses)\.(?:create|parse|stream)$"
_MODEL_KW: Tuple[ArgSpec, ...] = ((None, "model"), (None, "engine"), (None, "deployment_id"))


def _value_regex(names: str) -> str:
    """e.g. client.chat.completions.create(model='gpt-4', messages=[...])  ->  gpt-4"""
    return ("(?:" + names + r")\.(?:\w+\.)+(?:create|generate)\s*\([\s\S]*?"
            r"(?:model|engine|deployment_id)\s*=\s*" + """(['"])(?P<value>[^'"]+)\\1""")


def _chat(lib: str, qual_regex: str, names: str, model_args: Tuple[ArgSpec, ...] = _MODEL_KW) -> LoadRule:
    return LoadRule(f"{lib}.chat_api", lib, qual_regex, "model_id", model_args,
                    fallback_regex=_CHAT_FALLBACK, fallback_requires=(lib,), value_regex=_value_regex(names))


def build_rules(non_model_suffixes: Sequence[str], sklearn_types: Sequence[str]) -> List[LoadRule]:
    sk = frozenset(sklearn_model_classes(sklearn_types))
    return [
        # ── scikit-learn: estimator instantiation (Note 2: predictive models only)
        LoadRule("sklearn.estimator", "sklearn", r"sklearn(?:\.\w+)*\.[A-Z]\w*", "model_class",
                 class_allowlist=sk),

        # ── PyTorch
        LoadRule("torch.load", "torch", r"torch\.load", "model_file", ((0, "f"),), artifact_check=True),
        LoadRule("torch.jit.load", "torch", r"torch\.jit\.load", "model_file", ((0, "f"),)),
        LoadRule("torch.export.load", "torch", r"torch\.export\.load", "model_file", ((0, "f"),)),
        LoadRule("torch.hub.load", "torch", r"torch\.hub\.load", "model_id",
                 ((1, "model"),), extra_args=((0, "repo_or_dir"),)),
        LoadRule("torch.hub.load_state_dict_from_url", "torch",
                 r"torch\.(?:hub\.load_state_dict_from_url|utils\.model_zoo\.load_url)", "model_file", ((0, "url"),)),

        # ── TensorFlow / Keras
        LoadRule("tensorflow.load_model", "tensorflow",
                 r"(?:tensorflow\.(?:python\.)?keras|keras)\.(?:models\.|saving\.)?load_model"
                 r"|tensorflow\.(?:compat\.v1\.)?saved_model\.load"
                 r"|tensorflow\.keras\.experimental\.load_from_saved_model",
                 "model_file", ((0, "filepath"), (0, "export_dir"), (None, "export_dir"))),
        LoadRule("tensorflow.lite.Interpreter", "tensorflow", r"tensorflow\.lite\.(?:python\.interpreter\.)?Interpreter",
                 "model_file", ((None, "model_path"), (0, "model_path"))),
        LoadRule("tensorflow.keras.applications", "tensorflow",
                 r"(?:tensorflow\.)?keras\.applications(?:\.\w+)*\.[A-Z]\w*", "model_class"),

        # ── transformers (Tokenizer / Processor / Config ... excluded)
        LoadRule("transformers.from_pretrained", "transformers", r"transformers(?:\.\w+)*\.\w+\.from_pretrained",
                 "model_id", ((0, "pretrained_model_name_or_path"),), exclude_class_suffixes=True),
        LoadRule("transformers.pipeline", "transformers", r"transformers\.pipeline", "model_id",
                 ((None, "model"), (1, "model"))),

        # ── diffusers (schedulers excluded)
        LoadRule("diffusers.from_pretrained", "diffusers", r"diffusers(?:\.\w+)*\.\w+\.from_pretrained", "model_id",
                 ((0, "pretrained_model_name_or_path"),), exclude_class_suffixes=True),
        LoadRule("diffusers.from_single_file", "diffusers", r"diffusers(?:\.\w+)*\.\w+\.from_single_file",
                 "model_file", ((0, "pretrained_model_link_or_path"), (0, "pretrained_model_link_or_path_or_dict"))),
        LoadRule("diffusers.load_lora_weights", "diffusers", r"diffusers(?:\.\w+)*\(\)\.load_lora_weights",
                 "model_id", ((0, "pretrained_model_name_or_path_or_dict"),)),

        # ── Hugging Face Hub (Note 6: repo_type="dataset"/"space" excluded)
        LoadRule("huggingface_hub.download", "huggingface_hub",
                 r"huggingface_hub\.(?:HfApi\(\)\.|file_download\.|_snapshot_download\.)?"
                 r"(?:hf_hub_download|snapshot_download)",
                 "model_id", ((0, "repo_id"),), repo_type_arg=(None, "repo_type")),
        LoadRule("huggingface_hub.from_pretrained_keras", "huggingface_hub", r"huggingface_hub\.from_pretrained_keras",
                 "model_id", ((0, "repo_id"),)),
        LoadRule("huggingface_hub.InferenceClient", "huggingface_hub",
                 r"huggingface_hub\.(?:Async)?InferenceClient(?:\(\)\.\w+)?", "model_id",
                 ((None, "model"), (0, "model"))),

        # ── OpenAI-compatible SDKs
        _chat("openai", "openai" + _API_END, "client|openai"),
        _chat("groq", "groq" + _API_END, "client|groq"),
        _chat("fireworks", "fireworks" + _API_END + r"|fireworks(?:\.llm)?\.LLM", "client|fireworks",
              _MODEL_KW + ((0, "model"),)),
        _chat("llama_api_client", "llama_api_client" + _API_END, "client|llama"),
        _chat("deepseek", "deepseek" + _API_END, "client|deepseek"),

        # ── Anthropic
        LoadRule("anthropic.messages", "anthropic",
                 r"anthropic\.(?:Async)?Anthropic(?:Bedrock|Vertex)?\(\)\."
                 r"(?:messages\.(?:create|stream|count_tokens|batches\.create)|beta\.messages\.(?:create|stream)"
                 r"|completions\.create)",
                 "model_id", ((None, "model"),),
                 fallback_regex=r"\.(?:beta\.)?messages\.(?:create|stream)$", fallback_requires=("anthropic",)),

        # ── vLLM
        LoadRule("vllm.LLM", "vllm", r"vllm\.(?:entrypoints\.llm\.)?LLM", "model_id", ((0, "model"), (None, "model"))),
        LoadRule("vllm.EngineArgs", "vllm", r"vllm\.(?:engine\.arg_utils\.)?(?:Async)?EngineArgs", "model_id",
                 ((None, "model"),)),

        # ── llama.cpp
        LoadRule("llama_cpp.Llama", "llama_cpp", r"llama_cpp\.(?:llama\.)?Llama", "model_file",
                 ((0, "model_path"), (None, "model_path"))),
        LoadRule("llama_cpp.from_pretrained", "llama_cpp", r"llama_cpp\.Llama\.from_pretrained", "model_id",
                 ((0, "repo_id"),), extra_args=((1, "filename"),), extra_join="/", extra_before=False),
        LoadRule("llama_cpp.load_model_from_file", "llama_cpp",
                 r"llama_cpp\.llama_(?:load_model_from_file|model_load_from_file)", "model_file", ((0, "path_model"),)),

        # ── XGBoost
        LoadRule("xgboost.estimator", "xgboost",
                 r"xgboost\.(?:sklearn\.)?XGB(?:RF)?(?:Classifier|Regressor)|xgboost\.(?:sklearn\.)?XGBRanker"
                 r"|xgboost\.dask\.DaskXGB(?:Classifier|Regressor)", "model_class"),
        LoadRule("xgboost.train", "xgboost", r"xgboost\.(?:training\.|dask\.)?train", "model_class",
                 fixed_model="Booster"),
        LoadRule("xgboost.Booster", "xgboost", r"xgboost\.Booster", "model_file", ((None, "model_file"),)),
        LoadRule("xgboost.load_model", "xgboost", r"xgboost(?:\.\w+)*\(\)\.load_model", "model_file", ((0, "fname"),)),

        # ── Ollama
        LoadRule("ollama.api", "ollama", r"ollama\.(?:(?:Async)?Client\(\)\.)?(?:chat|generate|embeddings|embed|pull|show)",
                 "model_id", ((0, "model"), (None, "model"))),

        # ── Google Vertex AI / Gemini
        LoadRule("vertexai.GenerativeModel", "vertexai",
                 r"vertexai\.(?:preview\.)?(?:generative_models\.)?GenerativeModel", "model_id", ((0, "model_name"),)),
        LoadRule("vertexai.from_pretrained", "vertexai",
                 r"vertexai\.(?:preview\.)?(?:language|vision)_models(?:\.\w+)*\.\w+\.from_pretrained",
                 "model_id", ((0, "model_name"),)),
        LoadRule("genai.GenerativeModel", "genai", r"google\.generativeai\.GenerativeModel", "model_id",
                 ((0, "model_name"),)),
        LoadRule("genai.client", "genai",
                 r"google\.genai\.Client\(\)\.(?:aio\.)?(?:models\.(?:generate_content(?:_stream)?|embed_content"
                 r"|generate_images|generate_videos|count_tokens)|chats\.create)",
                 "model_id", ((None, "model"),)),
    ]


def class_name(qualname: str) -> str:
    """'sklearn.ensemble.RandomForestClassifier' -> 'RandomForestClassifier',
    'transformers.AutoModel.from_pretrained' -> 'AutoModel'."""
    parts = qualname.replace("()", "").split(".")
    if parts and parts[-1] in ("from_pretrained", "from_single_file"):
        parts = parts[:-1]
    return parts[-1] if parts else qualname


def class_allowed(rule: LoadRule, qualname: str, non_model_suffixes: Sequence[str]) -> bool:
    cname = class_name(qualname)
    if rule.class_allowlist is not None:
        return cname in rule.class_allowlist
    if rule.exclude_class_suffixes and any(cname.endswith(s) for s in non_model_suffixes):
        return False
    return True