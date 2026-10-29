"""Control units: directions that are not neurons, scanned and labeled exactly like neurons so that their
labels can be compared. The two kinds follow IRSI's controls (https://irsi.ai/notes/already-clean/).

  R: random unit vectors in the MLP-hidden space. If neurons are no cleaner than R, the neuron basis is not
     special.
  M: planted mixtures a_i / max_i + a_j / max_j of two random neurons of a layer, which should look like two
     unrelated jobs. If they are labeled ONE, the labeler cannot see polysemanticity.

  controls = Controls.make(neurons, per_layer=4)
  controls.scan(Scanner(model_id, devices), neurons)
  controls.save("runs/x/controls.pt")
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .scanner import Pole, Scanner, ScanResult


@dataclass
class Controls:
    """Per layer, units 0..per_layer-1 are R and units per_layer..2*per_layer-1 are M.

    dirs:   [L, N, 2 * per_layer], the basis vector of every unit
    units:  one dict per unit: set (R or M), layer, unit, and for M the two neurons i and j
    result: their scan, a ScanResult over the neurons' corpus whose units are these directions
    """

    dirs: Tensor
    units: list[dict]
    per_layer: int
    result: ScanResult | None = None

    @classmethod
    def make(cls, neurons: ScanResult, per_layer: int = 4, seed: int = 0) -> "Controls":
        """Random R directions, and M mixtures of neurons that fired positively in the neurons' scan
        (max_i is the neuron's positive-pole max there)."""
        pole_max = neurons.pole_max()
        n_layers, n_neurons = pole_max.shape[1:]
        g = torch.Generator().manual_seed(seed)
        dirs = torch.zeros(n_layers, n_neurons, 2 * per_layer)
        units = []
        for layer in range(n_layers):
            r = torch.randn(n_neurons, per_layer, generator=g)
            dirs[layer, :, :per_layer] = r / r.norm(dim=0)
            units += [{"set": "R", "layer": layer, "unit": d} for d in range(per_layer)]
            alive = (pole_max[0, layer] > 0).nonzero().squeeze(1)
            for d in range(per_layer, 2 * per_layer):
                i, j = alive[torch.randperm(len(alive), generator=g)[:2]].tolist()
                dirs[layer, i, d] = 1 / pole_max[0, layer, i]
                dirs[layer, j, d] = 1 / pole_max[0, layer, j]
                units.append({"set": "M", "layer": layer, "unit": d, "i": i, "j": j})
        return cls(dirs, units, per_layer)

    def scan(self, scanner: Scanner, neurons: ScanResult) -> None:
        """Scan these units on the neurons' own corpus, with the same band pass."""
        self.result = scanner.scan(neurons.corpus(), neurons.config["band_tokens"], basis=self.dirs)

    def basis(self, pole: Pole) -> Tensor:
        """[N] basis vector of a control pole (layer, unit, sign)."""
        return self.dirs[pole[0], :, pole[1]]

    def poles(self) -> list[tuple[str, Pole]]:
        """(set, pole) of every unit to label: both poles of each R direction, whose sign is arbitrary, and only
        the positive pole of each M mixture, which is built from its neurons' positive poles."""
        return [
            (u["set"], (u["layer"], u["unit"], sign))
            for u in self.units
            for sign in ((1, -1) if u["set"] == "R" else (1,))
        ]

    def save(self, path: str | Path) -> None:
        """controls.pt; the scan's corpus is not saved again, since it is the neurons'."""
        assert self.result is not None, "scan first"
        state = {"dirs": self.dirs, "units": self.units, "per_layer": self.per_layer}
        torch.save(state | {"top": self.result.top, "band": self.result.band}, path)

    @classmethod
    def load(cls, path: str | Path, neurons: ScanResult) -> "Controls":
        """A controls.pt, with its result reading contexts from the neurons' corpus."""
        state = torch.load(path, weights_only=False)
        result = ScanResult(neurons.docs, neurons.origins, state["top"], state["band"], neurons.config)
        return cls(state["dirs"], state["units"], state["per_layer"], result)
