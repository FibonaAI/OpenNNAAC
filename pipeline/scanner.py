"""Scan a corpus for every unit's strongest tokens and a sample of its mid-strength tokens.

A unit is a direction in the MLP-hidden space (the down_proj input). A neuron n is the one-hot basis
e_n; other units, such as control directions, are any other basis vectors. A pole is one firing direction
of a unit: (layer, unit, sign), sign +1 or -1.

  corpus = Corpus.from_jsonl("data/mix-55m.jsonl", tokenizer, max_tokens=50_000_000, window=1024)
  result = Scanner("./models/Qwen3-32B", devices).scan(corpus, band_tokens=10_000_000)
  result.save("runs/x/scan.pt")
"""

import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from infra import BandSample, Projected, TopK, parallel_collect

from .corpus import Corpus

Pole = tuple[int, int, int]  # (layer, unit, sign)


@dataclass
class Example:
    value: float  # signed activation
    context: str  # 48 tokens before, the peak token in [[ ]], 8 tokens after


@dataclass
class ScanResult:
    """What a scan keeps per pole, and the corpus to recover each token's context from.

    top:  TopK state, each [L, U, k], strongest first (U units per layer):
            pos_values / neg_values:       float32 signed activations (empty: -inf / +inf)
            pos_docs / neg_docs:           long window index (empty: -1)
            pos_positions / neg_positions: long offset in that window (empty: -1)
    band: BandSample state, each [2, L, U, band_k], index 0 = pos:
            values: float32 signed activations (empty: nan); docs, positions as above
    config: corpus, tokens, window, band_tokens (tokens of the band pass), k, band_k
    """

    docs: list[list[int]]
    origins: list[tuple[int, int]]
    top: dict[str, Tensor]
    band: dict[str, Tensor]
    config: dict

    @classmethod
    def load(cls, path: str | Path) -> "ScanResult":
        """A scan.pt saved by `save`."""
        return cls(**torch.load(path, weights_only=False))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(vars(self), path)

    def corpus(self) -> Corpus:
        return Corpus(self.docs, self.config["window"], self.config["corpus"])

    def pole_max(self) -> Tensor:
        """[2, L, U] largest |activation| of each pole over the corpus, index 0 = pos; 0 if it never fired."""
        return _pole_max(self.top)

    def fired(self, pole: Pole) -> bool:
        layer, unit, sign = pole
        return self.top["pos_docs" if sign > 0 else "neg_docs"][layer, unit, 0].item() >= 0

    def sample_poles(self, count: int, seed: int = 0) -> list[Pole]:
        """`count` distinct random poles that fired at least once."""
        n_layers, n_units, _ = self.top["pos_values"].shape
        rng = random.Random(seed)
        poles: list[Pole] = []
        while len(poles) < count:
            pole = (rng.randrange(n_layers), rng.randrange(n_units), rng.choice((1, -1)))
            if self.fired(pole) and pole not in poles:
                poles.append(pole)
        return poles

    def examples(
        self, pole: Pole, tokenizer: PreTrainedTokenizerBase, before: int = 48, after: int = 8
    ) -> tuple[list[Example], list[Example]]:
        """(top, band) examples of one pole: up to k strongest first, and up to band_k in-band ones.

        Empty slots are skipped. Each context is up to `before` tokens, the peak token in [[ ]], then up to
        `after` tokens, taken from the whole document, so it crosses window boundaries.

        Example: pole (12, 4096, 1) whose strongest top slot is value 7.8, window 153, position 200, with
        origins[153] = (37, 1024). The token is docs[37][1224], giving
          Example(7.8, "...the sofa and curled up. My[[ cat]] always sleeps there...")
        """
        layer, unit, sign = pole
        name, s = ("pos", 0) if sign > 0 else ("neg", 1)
        sources = {
            "top": [self.top[f"{name}_{f}"][layer, unit] for f in ("values", "docs", "positions")],
            "band": [self.band[f][s, layer, unit] for f in ("values", "docs", "positions")],
        }
        out = {}
        for kind, (values, wins, positions) in sources.items():
            out[kind] = []
            for v, w, p in zip(values.tolist(), wins.tolist(), positions.tolist(), strict=True):
                if w < 0:
                    continue
                doc, off = self.origins[w]
                ids, at = self.docs[doc], off + p  # position inside the whole document
                context = (
                    tokenizer.decode(ids[max(0, at - before) : at])
                    + "[["
                    + tokenizer.decode(ids[at : at + 1])
                    + "]]"
                    + tokenizer.decode(ids[at + 1 : at + 1 + after])
                )
                out[kind].append(Example(v, context))
        return out["top"], out["band"]


class Scanner:
    """Two passes over a corpus with one model replica per device.

    Pass 1 over the whole corpus keeps each pole's k strongest tokens, which also gives its max. Pass 2
    over the first band_tokens samples band_k tokens per pole at 30-70% of that max; it only needs a few
    in-band tokens, takes whole windows and so may overshoot by under one window.

    Example: 10M tokens, band_tokens 2M, window 1024. Pass 1 scans all ~10M tokens; pass 2 rescans only
    the first ~2M tokens (about 2000 windows).
    """

    def __init__(self, model_id: str, devices: list[str], batch_size: int = 8, k: int = 20, band_k: int = 10):
        self.model_id, self.devices, self.batch_size, self.k, self.band_k = model_id, devices, batch_size, k, band_k
        config = AutoConfig.from_pretrained(model_id).get_text_config()
        self.n_layers, self.n_neurons = config.num_hidden_layers, config.intermediate_size
        self.pad = AutoTokenizer.from_pretrained(model_id).pad_token_id

    def scan(self, corpus: Corpus, band_tokens: int, basis: Tensor | None = None) -> ScanResult:
        """Neurons by default; with basis [L, N, U], the U units per layer whose activation is act @ basis."""
        n_units = self.n_neurons if basis is None else basis.shape[2]
        top = self._collect(corpus.windows, partial(_make_topk, self.n_layers, n_units, self.k, basis))
        n_band, seen = corpus.prefix(band_tokens)
        band = self._collect(corpus.windows[:n_band], partial(_make_band, _pole_max(top), self.band_k, basis))
        config = {"corpus": corpus.source, "tokens": corpus.tokens, "window": corpus.window, "band_tokens": seen}
        return ScanResult(corpus.docs, corpus.origins, top, band, config | {"k": self.k, "band_k": self.band_k})

    def _collect(self, windows: list[list[int]], factory) -> dict[str, Tensor]:
        return parallel_collect(self.model_id, windows, factory, self.pad, self.devices, self.batch_size).state_dict()


# Aggregator factories for parallel_collect, which needs them picklable: module-level, bound with partial.


def _make_topk(n_layers: int, n_units: int, k: int, basis: Tensor | None, device: str, rank: int):
    top = TopK(n_layers, n_units, k, device)
    return top if basis is None else Projected(top, basis, device)


def _make_band(pole_max: Tensor, k: int, basis: Tensor | None, device: str, rank: int):
    band = BandSample(pole_max, k, device, seed=rank)
    return band if basis is None else Projected(band, basis, device)


def _pole_max(top: dict[str, Tensor]) -> Tensor:
    return torch.stack([top["pos_values"][..., 0], -top["neg_values"][..., 0]]).clamp(min=0)
