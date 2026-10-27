"""Data-parallel collection: one model replica per GPU, shards merged at the end."""

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Protocol, Self

from torch import Tensor

from .collect import collect_ids
from .model import load_model


class MergeableAggregator(Protocol):
    def __call__(self, layer: int, gate: Tensor, up: Tensor, act: Tensor, mask: Tensor, doc_ids: Tensor) -> None: ...

    def merge(self, other: Self) -> None: ...

    def cpu(self) -> Self: ...


# (device, rank) -> a fresh aggregator on that device; must be picklable, e.g. a
# module-level function or functools.partial of one
AggregatorFactory = Callable[[str, int], MergeableAggregator]


def _run_shard(
    model_id: str,
    device: str,
    rank: int,
    windows: list[list[int]],
    first_doc_id: int,
    make_fn: AggregatorFactory,
    pad_token_id: int,
    batch_size: int,
) -> MergeableAggregator:
    _, model = load_model(model_id, device)
    fn = make_fn(device, rank)
    collect_ids(model, windows, fn, pad_token_id, batch_size, first_doc_id)
    return fn.cpu()


def parallel_collect(
    model_id: str,
    windows: list[list[int]],
    make_fn: AggregatorFactory,
    pad_token_id: int,
    devices: list[str],
    batch_size: int = 8,
) -> MergeableAggregator:
    """Split `windows` into contiguous shards of about equal token count, one per device.

    Each shard runs in its own spawned process; the merged aggregator is returned on CPU
    and its doc ids index into the full `windows`. The caller must not need CUDA state
    shared with the workers.
    """
    total = sum(map(len, windows))
    bounds, seen = [0], 0
    for i, w in enumerate(windows):
        seen += len(w)
        if len(bounds) < len(devices) and seen >= total * len(bounds) / len(devices):
            bounds.append(i + 1)
    bounds += [len(windows)] * (len(devices) + 1 - len(bounds))

    with ProcessPoolExecutor(len(devices), mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(
                _run_shard,
                model_id,
                device,
                rank,
                windows[bounds[rank] : bounds[rank + 1]],
                bounds[rank],
                make_fn,
                pad_token_id,
                batch_size,
            )
            for rank, device in enumerate(devices)
        ]
        results = [f.result() for f in futures]
    merged = results[0]
    for other in results[1:]:
        merged.merge(other)
    return merged
