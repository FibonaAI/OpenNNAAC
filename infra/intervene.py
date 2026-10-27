"""Edits on MLP neuron activations, used as the `write` of `collect.mlp_hooks`."""

from collections.abc import Callable

import torch
from torch import Tensor

# (selected act [B, P, K_layer], columns of these neurons in the edit's neuron list) -> new values
EditFn = Callable[[Tensor, list[int]], Tensor]


class Edit:
    """Apply `fn` to the given neurons at the given positions of the current forward input.

    positions=None edits every position, which is also what generate needs: after the
    first step each forward sees only the new token. Explicit positions are for single
    forward passes and apply to every row of the batch.
    """

    def __init__(self, neurons: list[tuple[int, int]], fn: EditFn, positions: list[int] | None = None):
        self.fn = fn
        self.positions = positions
        self.by_layer: dict[int, tuple[Tensor, list[int]]] = {}  # layer -> (neuron ids, columns)
        grouped: dict[int, tuple[list[int], list[int]]] = {}
        for col, (layer, n) in enumerate(neurons):
            ids, cols = grouped.setdefault(layer, ([], []))
            ids.append(n)
            cols.append(col)
        for layer, (ids, cols) in grouped.items():
            self.by_layer[layer] = (torch.tensor(ids), cols)

    def __call__(self, layer: int, act: Tensor) -> Tensor | None:
        if layer not in self.by_layer:
            return None
        ids, cols = self.by_layer[layer]
        ids = ids.to(act.device)
        pos = (
            torch.arange(act.shape[1], device=act.device)
            if self.positions is None
            else torch.tensor(self.positions, device=act.device)
        )
        index = (slice(None), pos[:, None], ids[None, :])  # selects [B, P, K_layer]
        act = act.clone()
        act[index] = self.fn(act[index], cols).to(act.dtype)
        return act
