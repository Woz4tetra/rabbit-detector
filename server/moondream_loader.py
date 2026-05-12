from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_moondream(
    model_id: str = "vikhyatk/moondream2",
    revision: str = "2025-01-09",
    device: str = "cuda:0",
):
    """Load Moondream2 from HuggingFace. Pins the revision to avoid breaking API changes."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer
