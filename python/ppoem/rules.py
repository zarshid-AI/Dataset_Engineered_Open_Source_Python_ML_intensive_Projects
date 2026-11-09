"""
Model-loading APIs
==================
One `LoadRule` per way a library loads, instantiates or calls a model.
Calls are matched on their *qualified* name, computed from the imports of the
file (aliases included) and from the objects they are called on:

    import torch as th; th.load(p)                     -> torch.load
    from transformers import AutoModel as AM
    AM.from_pretrained(m)                              -> transformers.AutoModel.from_pretrained
    client = OpenAI(); client.chat.completions.create  -> openai.OpenAI().chat.completions.create

`model_args` lists where the model is passed, as (position, keyword) pairs;
the first one present in the call is used.

To support a new library or API, add a LoadRule here; nothing else changes.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .sklearn_models import sklearn_model_classes

ArgSpec = Tuple[Optional[int], Optional[str]]

# library key -> import roots
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


def library_of(qualname: str) -> Optional[str]:
    for lib, roots in LIBRARIES.items():
        for r in roots:
            if qualname == r or qualname.startswith(r + ".") or qualname.startswith(r + "("):
                return lib
    return None


@dataclass(frozen=True)
class LoadRule:
    rule_id: str
    library: str
    patterns: Tuple[str, ...]                 # fnmatch patterns on the qualified call name
    kind: str                                 # model_id | model_file | model_class
    model_args: Tuple[ArgSpec, ...] = ()
    extra_args: Tuple[ArgSpec, ...] = ()      # joined to the model value (torch.hub repo:model)
    extra_join: str = ":"
    extra_before: bool = True                 # extra value written before the model value
    fixed_model: Optional[str] = None         # model name when the API names no model (xgboost.train)
    exclude_class_suffixes: bool = False      # skip Tokenizer / Processor / Scheduler ...
    class_allowlist: Optional[FrozenSet[str]] = None
    class_is_uppercase: bool = False          # last segment must be a class (Keras applications)
    artifact_check: bool = False              # skip data / optimizer files (Note 6)
    repo_type_arg: Optional[ArgSpec] = None   # Hugging Face repo_type (Note 6)
    fallback_suffixes: Tuple[str, ...] = ()   # attribute chains accepted when the receiver is unknown
    fallback_requires: Tuple[str, ...] = ()   # ... if the file imports one of these libraries

    def matches(self, qualname: str) -> bool:
        rx = _COMPILED.get(self.patterns)
        if rx is None:
            rx = _COMPILED[self.patterns] = re.compile("|".join(f"(?:{fnmatch.translate(p)})" for p in self.patterns))
        return rx.match(qualname) is not None


_COMPILED: Dict[Tuple[str, ...], "re.Pattern"] = {}


_OPENAI_METHODS = (
    "chat.completions.create", "chat.completions.parse", "chat.completions.stream",
    "beta.chat.completions.parse", "beta.chat.completions.stream", "completions.create",
    "embeddings.create", "responses.create", "responses.parse", "responses.stream",
    "images.generate", "images.edit", "images.create_variation", "audio.transcriptions.create",
    "audio.translations.create", "audio.speech.create", "audio.speech.with_streaming_response.create",
    "moderations.create", "fine_tuning.jobs.create", "beta.assistants.create",
)
_OPENAI_LEGACY = ("ChatCompletion.create", "ChatCompletion.acreate", "Completion.create",
                  "Completion.acreate", "Embedding.create", "Embedding.acreate")
_MODEL_KW: Tuple[ArgSpec, ...] = ((None, "model"), (None, "engine"), (None, "deployment_id"))


def _openai_compatible(lib: str, clients: Sequence[str], modules: Sequence[str]) -> LoadRule:
    pats = []
    for c in clients:
        pats += [f"{c}().{m}" for m in _OPENAI_METHODS]
        pats += [f"{c}().with_options().{m}" for m in _OPENAI_METHODS]
    for mod in modules:
        pats += [f"{mod}.{m}" for m in _OPENAI_METHODS + _OPENAI_LEGACY]
    return LoadRule(f"{lib}.chat_api", lib, tuple(pats), "model_id", _MODEL_KW,
                    fallback_suffixes=_OPENAI_METHODS, fallback_requires=(lib,))


def build_rules(non_model_suffixes: Sequence[str], sklearn_types: Sequence[str]) -> List[LoadRule]:
    sk = frozenset(sklearn_model_classes(sklearn_types))
    rules = [
        # ── scikit-learn: estimator instantiation (Note 2: predictive models only)
        LoadRule("sklearn.estimator", "sklearn", ("sklearn.*",), "model_class", class_allowlist=sk),

        # ── PyTorch
        LoadRule("torch.load", "torch", ("torch.load",), "model_file", ((0, "f"),), artifact_check=True),
        LoadRule("torch.jit.load", "torch", ("torch.jit.load",), "model_file", ((0, "f"),)),
        LoadRule("torch.export.load", "torch", ("torch.export.load",), "model_file", ((0, "f"),)),
        LoadRule("torch.hub.load", "torch", ("torch.hub.load",), "model_id",
                 ((1, "model"),), extra_args=((0, "repo_or_dir"),)),
        LoadRule("torch.hub.load_state_dict_from_url", "torch",
                 ("torch.hub.load_state_dict_from_url", "torch.utils.model_zoo.load_url"), "model_file", ((0, "url"),)),

        # ── TensorFlow / Keras
        LoadRule("tensorflow.load_model", "tensorflow",
                 ("tensorflow.keras.models.load_model", "tensorflow.keras.saving.load_model",
                  "tensorflow.python.keras.models.load_model", "keras.models.load_model",
                  "keras.saving.load_model", "keras.load_model", "tensorflow.saved_model.load",
                  "tensorflow.compat.v1.saved_model.load", "tensorflow.keras.experimental.load_from_saved_model"),
                 "model_file", ((0, "filepath"), (0, "export_dir"), (None, "export_dir"))),
        LoadRule("tensorflow.lite.Interpreter", "tensorflow",
                 ("tensorflow.lite.Interpreter", "tensorflow.lite.python.interpreter.Interpreter"),
                 "model_file", ((None, "model_path"), (0, "model_path"))),
        LoadRule("tensorflow.keras.applications", "tensorflow",
                 ("tensorflow.keras.applications.*", "keras.applications.*"), "model_class",
                 class_is_uppercase=True),

        # ── transformers
        LoadRule("transformers.from_pretrained", "transformers", ("transformers.*.from_pretrained",),
                 "model_id", ((0, "pretrained_model_name_or_path"),), exclude_class_suffixes=True),
        LoadRule("transformers.pipeline", "transformers", ("transformers.pipeline",), "model_id",
                 ((None, "model"), (1, "model"))),

        # ── diffusers
        LoadRule("diffusers.from_pretrained", "diffusers", ("diffusers.*.from_pretrained",), "model_id",
                 ((0, "pretrained_model_name_or_path"),), exclude_class_suffixes=True),
        LoadRule("diffusers.from_single_file", "diffusers", ("diffusers.*.from_single_file",), "model_file",
                 ((0, "pretrained_model_link_or_path"), (0, "pretrained_model_link_or_path_or_dict"))),
        LoadRule("diffusers.load_lora_weights", "diffusers", ("diffusers.*().load_lora_weights",), "model_id",
                 ((0, "pretrained_model_name_or_path_or_dict"),)),

        # ── Hugging Face Hub (Note 6: repo_type="dataset"/"space" is excluded)
        LoadRule("huggingface_hub.download", "huggingface_hub",
                 ("huggingface_hub.hf_hub_download", "huggingface_hub.snapshot_download",
                  "huggingface_hub.HfApi().hf_hub_download", "huggingface_hub.HfApi().snapshot_download",
                  "huggingface_hub.file_download.hf_hub_download", "huggingface_hub._snapshot_download.snapshot_download"),
                 "model_id", ((0, "repo_id"),), repo_type_arg=(None, "repo_type")),
        LoadRule("huggingface_hub.from_pretrained_keras", "huggingface_hub",
                 ("huggingface_hub.from_pretrained_keras",), "model_id", ((0, "repo_id"),)),
        LoadRule("huggingface_hub.InferenceClient", "huggingface_hub",
                 ("huggingface_hub.InferenceClient", "huggingface_hub.AsyncInferenceClient",
                  "huggingface_hub.InferenceClient().*", "huggingface_hub.AsyncInferenceClient().*"),
                 "model_id", ((None, "model"), (0, "model"))),

        # ── OpenAI-compatible SDKs
        _openai_compatible("openai", ("openai.OpenAI", "openai.AsyncOpenAI", "openai.AzureOpenAI",
                                      "openai.AsyncAzureOpenAI", "openai.Client", "openai.AsyncClient"), ("openai",)),
        _openai_compatible("groq", ("groq.Groq", "groq.AsyncGroq"), ()),
        _openai_compatible("fireworks", ("fireworks.Fireworks", "fireworks.AsyncFireworks",
                                         "fireworks.client.Fireworks", "fireworks.client.AsyncFireworks"),
                           ("fireworks.client",)),
        _openai_compatible("llama_api_client", ("llama_api_client.LlamaAPIClient",
                                                "llama_api_client.AsyncLlamaAPIClient"), ()),
        _openai_compatible("deepseek", ("deepseek.DeepSeekAPI", "deepseek.DeepSeekClient"), ()),
        LoadRule("fireworks.LLM", "fireworks", ("fireworks.LLM", "fireworks.llm.LLM"), "model_id",
                 ((None, "model"), (0, "model"))),

        # ── Anthropic
        LoadRule("anthropic.messages", "anthropic",
                 tuple(f"{c}().{m}" for c in ("anthropic.Anthropic", "anthropic.AsyncAnthropic",
                                              "anthropic.AnthropicBedrock", "anthropic.AnthropicVertex",
                                              "anthropic.AsyncAnthropicBedrock", "anthropic.AsyncAnthropicVertex")
                       for m in ("messages.create", "messages.stream", "messages.count_tokens",
                                 "beta.messages.create", "beta.messages.stream", "completions.create",
                                 "messages.batches.create")),
                 "model_id", ((None, "model"),),
                 fallback_suffixes=("messages.create", "messages.stream", "beta.messages.create"),
                 fallback_requires=("anthropic",)),

        # ── vLLM
        LoadRule("vllm.LLM", "vllm", ("vllm.LLM", "vllm.entrypoints.llm.LLM"), "model_id",
                 ((0, "model"), (None, "model"))),
        LoadRule("vllm.EngineArgs", "vllm", ("vllm.EngineArgs", "vllm.AsyncEngineArgs",
                                             "vllm.engine.arg_utils.EngineArgs", "vllm.engine.arg_utils.AsyncEngineArgs"),
                 "model_id", ((None, "model"),)),

        # ── llama.cpp
        LoadRule("llama_cpp.Llama", "llama_cpp", ("llama_cpp.Llama", "llama_cpp.llama.Llama"), "model_file",
                 ((0, "model_path"), (None, "model_path"))),
        LoadRule("llama_cpp.from_pretrained", "llama_cpp", ("llama_cpp.Llama.from_pretrained",), "model_id",
                 ((0, "repo_id"),), extra_args=((1, "filename"),), extra_join="/", extra_before=False),
        LoadRule("llama_cpp.load_model_from_file", "llama_cpp",
                 ("llama_cpp.llama_load_model_from_file", "llama_cpp.llama_model_load_from_file"),
                 "model_file", ((0, "path_model"),)),

        # ── XGBoost
        LoadRule("xgboost.estimator", "xgboost",
                 tuple(f"xgboost.{c}" for c in ("XGBClassifier", "XGBRegressor", "XGBRanker", "XGBRFClassifier",
                                                "XGBRFRegressor", "sklearn.XGBClassifier", "sklearn.XGBRegressor",
                                                "sklearn.XGBRanker", "dask.DaskXGBClassifier", "dask.DaskXGBRegressor")),
                 "model_class"),
        LoadRule("xgboost.train", "xgboost", ("xgboost.train", "xgboost.training.train", "xgboost.dask.train"),
                 "model_class", fixed_model="Booster"),
        LoadRule("xgboost.Booster", "xgboost", ("xgboost.Booster",), "model_file", ((None, "model_file"),)),
        LoadRule("xgboost.load_model", "xgboost", ("xgboost.*().load_model",), "model_file", ((0, "fname"),)),

        # ── Ollama
        LoadRule("ollama.api", "ollama",
                 tuple(f"{p}.{m}" for p in ("ollama", "ollama.Client()", "ollama.AsyncClient()")
                       for m in ("chat", "generate", "embeddings", "embed", "pull", "show")),
                 "model_id", ((0, "model"), (None, "model"))),

        # ── Google Vertex AI / Gemini
        LoadRule("vertexai.GenerativeModel", "vertexai",
                 ("vertexai.generative_models.GenerativeModel", "vertexai.preview.generative_models.GenerativeModel",
                  "vertexai.GenerativeModel"), "model_id", ((0, "model_name"),)),
        LoadRule("vertexai.from_pretrained", "vertexai",
                 ("vertexai.language_models.*.from_pretrained", "vertexai.preview.language_models.*.from_pretrained",
                  "vertexai.vision_models.*.from_pretrained", "vertexai.preview.vision_models.*.from_pretrained"),
                 "model_id", ((0, "model_name"),)),
        LoadRule("genai.GenerativeModel", "genai", ("google.generativeai.GenerativeModel",), "model_id",
                 ((0, "model_name"),)),
        LoadRule("genai.client", "genai",
                 tuple(f"google.genai.Client().{a}{m}" for a in ("", "aio.")
                       for m in ("models.generate_content", "models.generate_content_stream", "models.embed_content",
                                 "models.generate_images", "models.generate_videos", "models.count_tokens",
                                 "chats.create")),
                 "model_id", ((None, "model"),)),
    ]
    return rules


def class_name(qualname: str) -> str:
    """'sklearn.ensemble.RandomForestClassifier' -> 'RandomForestClassifier',
    'transformers.AutoModel.from_pretrained' -> 'AutoModel'."""
    parts = qualname.replace("()", "").split(".")
    if parts and parts[-1] in ("from_pretrained", "from_single_file"):
        parts = parts[:-1]
    return parts[-1] if parts else qualname


_UPPER = re.compile(r"^[A-Z]")


def class_allowed(rule: LoadRule, qualname: str, non_model_suffixes: Sequence[str]) -> bool:
    cname = class_name(qualname)
    if rule.class_allowlist is not None:
        return cname in rule.class_allowlist
    if rule.class_is_uppercase and not _UPPER.match(cname):
        return False
    if rule.exclude_class_suffixes and any(cname.endswith(s) for s in non_model_suffixes):
        return False
    return True
