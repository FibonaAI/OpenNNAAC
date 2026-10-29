"""Load a Qwen3 / Qwen3.5 checkpoint with Transformers."""

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer


def load_model(model_id: str, device: str = "cuda:0"):
    """The model class comes from the checkpoint's config: Qwen3ForCausalLM for Qwen3-32B,
    Qwen3_5ForConditionalGeneration for Qwen3.5."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_cls = getattr(transformers, AutoConfig.from_pretrained(model_id).architectures[0])
    model = (
        model_cls.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    return tokenizer, model
