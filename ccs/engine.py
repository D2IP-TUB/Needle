"""vLLM inference wrapper.

One ``Engine`` is one loaded model on the GPUs visible to the process. Prompts are rendered through the
model's chat template once and fed to vLLM as token ids, so the token count used for the
context-window check is exactly what the model sees. Decoding is greedy (temperature 0).
"""

import time
from typing import Optional

from .models import ModelSpec, get_model


class Engine:
    def __init__(
        self,
        model_key: str,
        long_context: bool = True,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        tensor_parallel: Optional[int] = None,
        seed: int = 0,
    ):
        from vllm import LLM  # imported lazily so that data building never needs vLLM

        self.spec: ModelSpec = get_model(model_key)
        self.max_model_len = max_model_len or self.spec.max_len(long_context)
        kwargs = dict(
            model=self.spec.hf_id,
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel or self.spec.tensor_parallel,
            enable_prefix_caching=True,
            seed=seed,
        )
        if long_context and self.spec.long_overrides:
            kwargs["hf_overrides"] = self.spec.long_overrides
        t0 = time.time()
        self.llm = LLM(**kwargs)
        self.tok = self.llm.get_tokenizer()
        print(f"[engine] loaded {self.spec.hf_id} max_len={self.max_model_len} in {time.time() - t0:.0f}s", flush=True)

    def encode(self, prompt: str) -> list:
        """Token ids of ``prompt`` rendered as a single user turn of the model's chat template."""
        messages = [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           **self.spec.chat_template_kwargs)
        if hasattr(ids, "input_ids"):  # BatchEncoding in some transformers versions
            ids = ids["input_ids"]
        return list(ids)
