"""
Filters that keep only real models in the dataset
=================================================
Note 2  scikit-learn: only predictive estimators (see sklearn_models.py; applied
        through the sklearn rule's class allow-list).
Note 6  Datasets and training state are not models:
        * Hugging Face Hub downloads with repo_type="dataset" / "space"
        * files loaded with torch.load() whose name says they hold data,
          optimizer / scheduler state, embeddings, features, labels, ...
Every excluded call is written to the excluded-calls report with its reason,
so the decision can be checked by hand.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Set

# words that mark a file / variable as data or training state, not a model
DATA_TOKENS: Set[str] = {
    "data", "dataset", "datasets", "train", "training", "val", "valid", "validation", "test", "testing",
    "eval", "evaluation", "features", "feature", "feats", "embeddings", "embedding", "embeds", "emb",
    "labels", "label", "targets", "vocab", "vocabulary", "stats", "statistics", "mean", "std",
    "indices", "index", "idx", "cache", "cached", "samples", "sample", "batch", "batches",
    "optimizer", "optim", "scheduler", "sched", "lr", "rng", "random", "seed", "trainer",
    "history", "metrics", "results", "logs", "log", "tensor", "tensors", "inputs", "outputs",
    "mask", "masks", "corpus", "split", "splits", "fold", "folds", "graph", "edges", "nodes",
}
# words that mark a file / variable as a model (they win over DATA_TOKENS in a file name)
MODEL_TOKENS: Set[str] = {
    "model", "models", "weights", "weight", "ckpt", "checkpoint", "checkpoints", "pretrained",
    "best", "last", "final", "net", "network", "encoder", "decoder", "backbone", "head", "vae",
    "unet", "lora", "adapter", "classifier", "regressor", "detector", "generator", "discriminator",
    "policy", "actor", "critic", "agent", "params", "state_dict", "statedict", "pth", "pt", "bin",
    "safetensors", "onnx", "gguf", "engine", "jit", "torchscript", "scripted",
}

_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z])(?=[A-Z])")


def tokens(text: str) -> Set[str]:
    return {t.lower() for t in _SPLIT.split(text or "") if t}


def basename(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def data_file_reason(value: str) -> Optional[str]:
    """Reason why a loaded file is data / state rather than a model, or None."""
    name = basename(value)
    stem_tokens = tokens(name.rsplit(".", 1)[0] if "." in name else name)
    hit = stem_tokens & DATA_TOKENS
    if hit and not (stem_tokens & (MODEL_TOKENS - {"pt", "pth", "bin"})):
        return f"file name '{name}' refers to data/state ({', '.join(sorted(hit))})"
    return None


def data_variable_reason(variable_text: str) -> Optional[str]:
    """Same check on the variable / expression passed to the call (last identifier)."""
    ids = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", variable_text or "")
    if not ids:
        return None
    toks = tokens(ids[-1])
    hit = toks & DATA_TOKENS
    if hit and not (toks & MODEL_TOKENS):
        return f"variable '{ids[-1]}' refers to data/state ({', '.join(sorted(hit))})"
    return None


def non_model_repo_type(values: Iterable[str], non_model_types: Iterable[str]) -> Optional[str]:
    vals = {v for v in values if v}
    bad = vals & set(non_model_types)
    if vals and vals == bad:
        return f"Hugging Face repo_type={', '.join(sorted(bad))} (not a model)"
    return None


def implausible_value_reason(value: str) -> Optional[str]:
    """Values that cannot be a model name / path (protects against wrong links)."""
    v = value.strip()
    if not v:
        return "empty string"
    if "\n" in v or len(v) > 300:
        return "multi-line or very long string"
    if v.isdigit():
        return "number"
    if v.count(" ") > 4:
        return "looks like text, not a model name"
    if v in {".", "/", "./", "~"}:
        return "not a model name"
    return None
