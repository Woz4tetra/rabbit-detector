from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_moondream(
    model_id: str = "vikhyatk/moondream2",
    revision: str = "2025-01-09",
    device: str = "cuda:0",
):
    """Load Moondream2 from HuggingFace.

    Avoids device_map= so that caching_allocator_warmup is never called —
    that codepath hits an attribute the moondream custom code doesn't define.
    Instead the model loads on CPU then moves to the target device.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device=device, dtype=torch.float16)
    model.eval()
    return model, tokenizer
