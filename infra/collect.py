"""Read and write MLP neuron activations with hooks, and stream them over texts."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Protocol

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# (layer, gate [B, T, N], up [B, T, N], act [B, T, N]) -> None
Reader = Callable[[int, Tensor, Tensor, Tensor], None]
# (layer, act [B, T, N]) -> replacement act, or None to keep it
Writer = Callable[[int, Tensor], Tensor | None]


class Aggregator(Protocol):
    def __call__(
        self,
        layer: int,
        gate: Tensor,  # [B, T, N] gate_proj output
        up: Tensor,  # [B, T, N] up_proj output
        act: Tensor,  # [B, T, N] down_proj input = silu(gate) * up
        mask: Tensor,  # [B, T] bool, True for real tokens
        doc_ids: Tensor,  # [B] long, index of each row in `texts`
    ) -> None: ...


def get_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Decoder layers of the language model; multimodal models nest it one level deeper."""
    inner = model.model
    return getattr(inner, "language_model", inner).layers


@contextmanager
def mlp_hooks(
    model: PreTrainedModel, read: Reader | None = None, write: Writer | None = None
) -> Generator[None, None, None]:
    """Hook every MLP's down_proj input. Works for forward and generate.

    `write` runs first and its result replaces the activation; `read` then sees the
    activation actually used. gate/up are the unedited projections.
    """
    # Per layer, MLP(x) = down_proj(act_fn(gate_proj(x)) * up_proj(x)) runs as:
    #   1. gate_proj done -> its forward hook sets store["gate"]
    #   2. up_proj done   -> its forward hook sets store["up"]
    #   3. act = act_fn(gate) * up
    #   4. before down_proj, `pre` gets act as inp[0], applies `write`, then pops gate/up
    #      and calls read(i, gate, up, act)
    # Layers run sequentially, so one shared `store` suffices; pop frees the tensors and
    # raises KeyError if the order is ever violated.
    store: dict[str, Tensor] = {}
    hooks = []
    for i, layer in enumerate(get_layers(model)):
        mlp = layer.mlp
        if read is not None:
            hooks += [
                mlp.gate_proj.register_forward_hook(lambda mod, inp, out: store.__setitem__("gate", out)),
                mlp.up_proj.register_forward_hook(lambda mod, inp, out: store.__setitem__("up", out)),
            ]

        def pre(mod, inp, i=i):
            act = inp[0]
            if write is not None:
                edited = write(i, act)
                if edited is not None:
                    act = edited
            if read is not None:
                read(i, store.pop("gate"), store.pop("up"), act)
            return None if act is inp[0] else (act,)

        hooks.append(mlp.down_proj.register_forward_pre_hook(pre))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()
        store.clear()


def collect(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    fn: Aggregator,
    batch_size: int = 8,
    max_length: int = 512,
) -> None:
    """Token positions passed to `fn` match `tokenizer(text).input_ids` (truncated to max_length)."""
    windows = [ids[:max_length] for ids in tokenizer(texts)["input_ids"]]
    collect_ids(model, windows, fn, tokenizer.pad_token_id, batch_size)


@torch.inference_mode()
def collect_ids(
    model: PreTrainedModel,
    windows: list[list[int]],
    fn: Aggregator,
    pad_token_id: int,
    batch_size: int = 8,
    first_doc_id: int = 0,
) -> None:
    """Run token-id windows in batches, padded on the right.

    `doc_ids` index into `windows`, plus first_doc_id when `windows` is a shard of a larger list.
    """
    batch_info: dict[str, Tensor] = {}

    def read(layer: int, gate: Tensor, up: Tensor, act: Tensor) -> None:
        fn(layer, gate, up, act, batch_info["mask"], batch_info["doc_ids"])

    with mlp_hooks(model, read=read):
        for start in range(0, len(windows), batch_size):
            rows = windows[start : start + batch_size]
            width = max(len(r) for r in rows)
            input_ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(rows), width), dtype=torch.bool)
            for i, r in enumerate(rows):
                input_ids[i, : len(r)] = torch.tensor(r)
                mask[i, : len(r)] = True
            batch_info["mask"] = mask.to(model.device)
            batch_info["doc_ids"] = torch.arange(
                first_doc_id + start, first_doc_id + start + len(rows), device=model.device
            )
            # Only hooks are read; logits_to_keep=1 skips the [batch, width, vocab] lm_head output
            model(input_ids=input_ids.to(model.device), attention_mask=batch_info["mask"].long(), logits_to_keep=1)
