"""Aggregators for `collect.collect`. Each can `merge` a peer's state, for multi-GPU runs."""

from typing import Self

import torch
from torch import Tensor


def merge_top(
    keys: Tensor, fields: list[Tensor], new_keys: Tensor, new_fields: list[Tensor]
) -> tuple[Tensor, list[Tensor]]:
    """Keep the k largest keys per row. keys/fields: [N, k]; new_keys: [N, M]; new_fields: [N, M] or [M]."""
    k = keys.shape[1]
    top, idx = torch.cat([keys, new_keys], dim=1).topk(k, dim=1)
    is_old, old_idx, new_idx = idx < k, idx.clamp(max=k - 1), (idx - k).clamp(min=0)
    merged = []
    for old, new in zip(fields, new_fields, strict=True):
        picked = new.gather(1, new_idx) if new.dim() == 2 else new[new_idx]
        merged.append(torch.where(is_old, old.gather(1, old_idx), picked))
    return top, merged


class TopK:
    """Per neuron, the k tokens with the largest (pos pole) and smallest (neg pole) activation.

    Results, each [L, N, k] and sorted strongest first:
      pos_values / neg_values: float32 activations (neg_values are negative)
      pos_docs / neg_docs:     long index into `texts`
      pos_positions / neg_positions: long token position in that text
    Slots never filled hold -inf / +inf with doc and position -1.
    """

    def __init__(self, n_layers: int, n_neurons: int, k: int, device: torch.device | str):
        shape = (n_layers, n_neurons, k)
        self.k = k
        self.pos_values = torch.full(shape, -torch.inf, device=device)
        self.neg_values = torch.full(shape, torch.inf, device=device)
        self.pos_docs = torch.full(shape, -1, dtype=torch.long, device=device)
        self.neg_docs = torch.full(shape, -1, dtype=torch.long, device=device)
        self.pos_positions = torch.full(shape, -1, dtype=torch.long, device=device)
        self.neg_positions = torch.full(shape, -1, dtype=torch.long, device=device)

    def __call__(self, layer: int, gate: Tensor, up: Tensor, act: Tensor, mask: Tensor, doc_ids: Tensor) -> None:
        B, T, N = act.shape
        flat = act.float().masked_fill(~mask[..., None], torch.nan).reshape(B * T, N).T  # [N, B*T]
        docs = doc_ids.repeat_interleave(T)  # [B*T]
        positions = torch.arange(T, device=act.device).repeat(B)  # [B*T]
        for sign, values, doc_out, pos_out in self._poles():
            # Signed so that larger is always stronger; padding (nan) becomes -inf.
            self._update(layer, sign, values, doc_out, pos_out, (flat * sign).nan_to_num(-torch.inf), docs, positions)

    def merge(self, other: Self) -> None:
        for (sign, values, doc_out, pos_out), (_, o_values, o_docs, o_pos) in zip(
            self._poles(), other._poles(), strict=True
        ):
            for layer in range(len(values)):
                self._update(layer, sign, values, doc_out, pos_out, o_values[layer] * sign, o_docs[layer], o_pos[layer])

    def cpu(self) -> Self:
        for name, value in self.state_dict().items():
            setattr(self, name, value)
        return self

    def _poles(self) -> list[tuple[float, Tensor, Tensor, Tensor]]:
        return [
            (1.0, self.pos_values, self.pos_docs, self.pos_positions),
            (-1.0, self.neg_values, self.neg_docs, self.neg_positions),
        ]

    def _update(
        self,
        layer: int,
        sign: float,
        values: Tensor,
        doc_out: Tensor,
        pos_out: Tensor,
        new_keys: Tensor,
        new_docs: Tensor,
        new_positions: Tensor,
    ) -> None:
        top, (doc, pos) = merge_top(
            values[layer] * sign, [doc_out[layer], pos_out[layer]], new_keys, [new_docs, new_positions]
        )
        empty = top == -torch.inf
        doc_out[layer] = doc.masked_fill(empty, -1)
        pos_out[layer] = pos.masked_fill(empty, -1)
        values[layer] = top * sign

    def state_dict(self) -> dict[str, Tensor]:
        return {
            name: getattr(self, name).cpu()
            for name in (
                "pos_values",
                "neg_values",
                "pos_docs",
                "neg_docs",
                "pos_positions",
                "neg_positions",
            )
        }


class BandSample:
    """Per pole, a uniform sample of up to k tokens whose signed activation is in [lo, hi] x pole max.

    Sampling keeps the k largest random keys among in-band tokens. Results are [2, L, N, k]
    with index 0 = positive pole, 1 = negative pole; values are signed activations and
    empty slots have doc and position -1.

    Example: one neuron with positive pole max 8.0 and negative pole max 5.0, k=2. The bands are
    [2.4, 5.6] for the positive pole and [-3.5, -1.5] for the negative one:

        token    act    positive band   negative band
        cat      +8.0   no (too strong)
        pet      +4.0   yes, key 0.81
        fur      +3.0   yes, key 0.23
        tail     +2.5   yes, key 0.57
        the      +0.1   no (too weak)
        bank     -2.0                   yes, key 0.40
        stock    -5.0                   no (too strong)

    The positive pole keeps {pet 0.81, tail 0.57}; the negative pole keeps {bank} and one empty
    slot. If a later batch brings "paw" with key 0.92, the positive pole becomes {paw, pet}.
    Since keys are i.i.d. uniform, the final top-k is a uniform sample without replacement over
    all in-band tokens, using only k slots. Shards merge the same way, as long as their keys are
    independent.
    """

    def __init__(
        self,
        pole_max: Tensor,
        k: int,
        device: torch.device | str,
        lo: float = 0.3,
        hi: float = 0.7,
        seed: int = 0,
    ):
        self.pole_max = pole_max.to(device)  # [2, L, N], magnitudes
        shape = (*pole_max.shape, k)
        self.lo, self.hi = lo, hi
        # Shards merged later must use different seeds so their keys are independent.
        self.generator: torch.Generator | None = torch.Generator(device).manual_seed(seed)
        self.keys = torch.full(shape, -torch.inf, device=device)
        self.values = torch.full(shape, torch.nan, device=device)
        self.docs = torch.full(shape, -1, dtype=torch.long, device=device)
        self.positions = torch.full(shape, -1, dtype=torch.long, device=device)

    def __call__(self, layer: int, gate: Tensor, up: Tensor, act: Tensor, mask: Tensor, doc_ids: Tensor) -> None:
        B, T, N = act.shape
        flat = act.float().reshape(B * T, N).T  # [N, B*T]
        real = mask.reshape(B * T)
        docs = doc_ids.repeat_interleave(T)
        positions = torch.arange(T, device=act.device).repeat(B)
        for s, sign in enumerate((1.0, -1.0)):
            m = self.pole_max[s, layer][:, None]  # [N, 1]
            signed = flat * sign
            in_band = (signed >= self.lo * m) & (signed <= self.hi * m) & real & (m > 0)
            rand = torch.rand(flat.shape, generator=self.generator, device=act.device)
            new_keys = torch.where(in_band, rand, -torch.inf)
            keys, (values, d, p) = merge_top(
                self.keys[s, layer],
                [self.values[s, layer], self.docs[s, layer], self.positions[s, layer]],
                new_keys,
                [flat, docs, positions],
            )
            empty = keys == -torch.inf
            self.keys[s, layer] = keys
            self.values[s, layer] = values
            self.docs[s, layer] = d.masked_fill(empty, -1)
            self.positions[s, layer] = p.masked_fill(empty, -1)

    def merge(self, other: Self) -> None:
        for s in range(2):
            for layer in range(self.keys.shape[1]):
                keys, (values, docs, positions) = merge_top(
                    self.keys[s, layer],
                    [self.values[s, layer], self.docs[s, layer], self.positions[s, layer]],
                    other.keys[s, layer],
                    [other.values[s, layer], other.docs[s, layer], other.positions[s, layer]],
                )
                self.keys[s, layer], self.values[s, layer] = keys, values
                self.docs[s, layer], self.positions[s, layer] = docs, positions

    def cpu(self) -> Self:
        for name in ("pole_max", "keys", "values", "docs", "positions"):
            setattr(self, name, getattr(self, name).cpu())
        self.generator = None  # a CUDA generator cannot be pickled; a merged result does not sample
        return self

    def state_dict(self) -> dict[str, Tensor]:
        return {"values": self.values.cpu(), "docs": self.docs.cpu(), "positions": self.positions.cpu()}


class Projected:
    """Runs `inner` on activations projected onto fixed directions instead of on the neurons.

    dirs is [L, N, D]: unit d of layer l has activation act @ dirs[l, :, d], and `inner` sees D units
    per layer. Used for control units such as random directions and planted mixtures of two neurons.
    """

    def __init__(self, inner, dirs: Tensor, device: torch.device | str):
        self.inner = inner
        self.dirs = dirs.to(device)

    def __call__(self, layer: int, gate: Tensor, up: Tensor, act: Tensor, mask: Tensor, doc_ids: Tensor) -> None:
        self.inner(layer, gate, up, act.float() @ self.dirs[layer], mask, doc_ids)

    def merge(self, other: Self) -> None:
        self.inner.merge(other.inner)

    def cpu(self) -> Self:
        self.dirs = self.dirs.cpu()
        self.inner = self.inner.cpu()
        return self

    def state_dict(self) -> dict[str, Tensor]:
        return self.inner.state_dict()


class Gather:
    """Every real token's activation for a fixed list of neurons.

    `values[doc]` is a float32 CPU tensor [T_doc, K]; column j belongs to `neurons[j]`.
    """

    def __init__(self, neurons: list[tuple[int, int]]):
        self.neurons = neurons
        self.by_layer: dict[int, tuple[list[int], list[int]]] = {}  # layer -> (neuron ids, columns)
        for col, (layer, n) in enumerate(neurons):
            ids, cols = self.by_layer.setdefault(layer, ([], []))
            ids.append(n)
            cols.append(col)
        self.values: dict[int, Tensor] = {}

    def __call__(self, layer: int, gate: Tensor, up: Tensor, act: Tensor, mask: Tensor, doc_ids: Tensor) -> None:
        if layer not in self.by_layer:
            return
        ids, cols = self.by_layer[layer]
        picked = act[..., ids].float().cpu()  # [B, T, K_layer]
        for row, doc in enumerate(doc_ids.tolist()):
            real = picked[row][mask[row].cpu()]  # [T_doc, K_layer]
            if doc not in self.values:
                self.values[doc] = torch.full((len(real), len(self.neurons)), torch.nan)
            self.values[doc][:, cols] = real

    def merge(self, other: Self) -> None:
        self.values.update(other.values)

    def cpu(self) -> Self:
        return self
