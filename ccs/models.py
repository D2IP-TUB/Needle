"""Registry of the open-weight models used in the paper.

``qwen3-4b-2507`` is the main model; the other eight are the transfer models. Every model runs at its
own maximum context length (``long_max_len``): Qwen2.5 and Qwen3-8B are extended to 128K with YaRN
exactly as documented on their model cards, Qwen3-4B-Instruct-2507 uses its native 256K window.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelSpec:
    key: str
    hf_id: str
    native_max_len: int
    long_max_len: int
    long_overrides: Optional[dict] = None
    chat_template_kwargs: dict = field(default_factory=dict)
    tensor_parallel: int = 1
    label: str = ""

    def max_len(self, long_context: bool) -> int:
        return self.long_max_len if long_context else self.native_max_len


# vLLM 0.13 applies dict-valued hf_overrides *after* it has normalised the HF config, so the
# override must target ``rope_parameters`` (not ``rope_scaling``) and carry ``rope_theta``
# explicitly (1e6 for Qwen2.5 and Qwen3).
YARN_128K = {
    "rope_parameters": {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
    }
}

MODELS = {
    "qwen3-4b-2507": ModelSpec(
        key="qwen3-4b-2507",
        hf_id="Qwen/Qwen3-4B-Instruct-2507",
        native_max_len=131072,
        long_max_len=262144,
        label="Qwen3-4B-Instruct-2507",
    ),
    "qwen3-8b": ModelSpec(
        key="qwen3-8b",
        hf_id="Qwen/Qwen3-8B",
        native_max_len=40960,
        long_max_len=131072,
        long_overrides=YARN_128K,
        chat_template_kwargs={"enable_thinking": False},
        label="Qwen3-8B (non-thinking)",
    ),
    "qwen2.5-1.5b": ModelSpec(
        key="qwen2.5-1.5b",
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        native_max_len=32768,
        long_max_len=131072,
        long_overrides=YARN_128K,
        label="Qwen2.5-1.5B-Instruct",
    ),
    "qwen2.5-7b": ModelSpec(
        key="qwen2.5-7b",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        native_max_len=32768,
        long_max_len=131072,
        long_overrides=YARN_128K,
        label="Qwen2.5-7B-Instruct",
    ),
    "llama-3.2-3b": ModelSpec(
        key="llama-3.2-3b",
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        native_max_len=131072,
        long_max_len=131072,
        label="Llama-3.2-3B-Instruct",
    ),
    "llama-3.1-8b": ModelSpec(
        key="llama-3.1-8b",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        native_max_len=131072,
        long_max_len=131072,
        label="Llama-3.1-8B-Instruct",
    ),
    "mistral-7b-v0.3": ModelSpec(
        key="mistral-7b-v0.3",
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
        native_max_len=32768,
        long_max_len=32768,
        label="Mistral-7B-Instruct-v0.3",
    ),
    "phi-4-mini": ModelSpec(
        key="phi-4-mini",
        hf_id="microsoft/Phi-4-mini-instruct",
        native_max_len=131072,
        long_max_len=131072,
        label="Phi-4-mini-instruct",
    ),
    "granite-3.3-8b": ModelSpec(
        key="granite-3.3-8b",
        hf_id="ibm-granite/granite-3.3-8b-instruct",
        native_max_len=131072,
        long_max_len=131072,
        label="Granite-3.3-8B-Instruct",
    ),
}

MAIN_MODEL = "qwen3-4b-2507"
# order of the transfer table
TRANSFER_MODELS = ["qwen3-4b-2507", "qwen3-8b", "qwen2.5-1.5b", "qwen2.5-7b", "llama-3.2-3b", "llama-3.1-8b",
                   "mistral-7b-v0.3", "phi-4-mini", "granite-3.3-8b"]


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model '{key}'. Known: {sorted(MODELS)}")
    return MODELS[key]
