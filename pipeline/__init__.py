"""Building blocks of the neuron experiments, on top of infra; the entry scripts are in experiments/ and at
the project root (corpus preparation).

corpus:   DatasetHub, Corpus: download dataset files and cut documents into windows
scanner:  Scanner, ScanResult: a unit's strongest and mid-strength tokens over a corpus
controls: Controls: R (random) and M (planted mixture) control units, scanned like neurons
label:    DirectLogitEffect, Dashboard, DashboardBuilder, Labeler, LabelStore: dashboards and LLM labels
"""

from .controls import Controls
from .corpus import Corpus, DatasetHub, HuggingFaceHub, ModelScopeHub
from .label import CATEGORIES, Dashboard, DashboardBuilder, DirectLogitEffect, Label, Labeler, LabelStore, Unit
from .scanner import Example, Pole, Scanner, ScanResult

__all__ = [
    "CATEGORIES",
    "Controls",
    "Corpus",
    "Dashboard",
    "DashboardBuilder",
    "DatasetHub",
    "DirectLogitEffect",
    "Example",
    "HuggingFaceHub",
    "Label",
    "LabelStore",
    "Labeler",
    "ModelScopeHub",
    "Pole",
    "ScanResult",
    "Scanner",
    "Unit",
]
