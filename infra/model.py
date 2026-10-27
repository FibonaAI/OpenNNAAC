"""Load Qwen3.5 with Transformers."""

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


def load_model(model_id: str, device: str = "cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    return tokenizer, model
