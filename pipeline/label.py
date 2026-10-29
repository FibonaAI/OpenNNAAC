"""LLM labels for units from their dashboards.

A unit is anything with a dashboard: a neuron pole, a control direction, or a dashboard loaded from elsewhere.
Its Dashboard is what the labeler sees, sent as JSON: top_contexts, band_contexts_30_70pct, promoted_tokens,
suppressed_tokens, notes. The instruction, categories and dashboard fields follow IRSI's labeling protocol
(https://irsi.ai/notes/already-clean/), so labels are comparable with theirs.

  DirectLogitEffect: the tokens a unit's output direction raises and lowers most (promoted / suppressed).
  DashboardBuilder:  dashboards of scanned poles, as blind Units: sampled neuron poles mixed with controls.
  Labeler:           sends dashboards to an OpenAI-compatible chat API and parses the category, one dashboard
                     per request (the single-unit prompt) or `batch` per request (one prompt for several).
  LabelStore:        labels.jsonl, one record per attempt. The last record of a unit wins and failed units are
                     retried, so an interrupted run resumes.

  units = DashboardBuilder(model, tokenizer).units(neurons, count=384, controls=controls)
  labeler = Labeler("google/gemini-3.8-flash", "Qwen3-32B", reasoning_effort="high")
  labeler.label_all(units, LabelStore("runs/x/labels.jsonl"), workers=16)

The API key and base URL come from OPENAI_API_KEY and OPENAI_BASE_URL.
"""

import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from infra import get_layers

from .controls import Controls
from .scanner import Example, Pole, ScanResult

# What a unit does, as defined in INSTRUCTION:
#   ONE:             one main job, possibly with small related offshoots
#   TWO_RELATED:     two related jobs
#   MULTI_UNRELATED: two or more unrelated jobs
#   UNCLEAR:         no confident hypothesis
CATEGORIES = ("ONE", "TWO_RELATED", "MULTI_UNRELATED", "UNCLEAR")

# Instruction and answer format for one dashboard, copied verbatim from IRSI's labeling protocol
# (https://irsi.ai/notes/already-clean/, code/labeler2/label_other.py in already-clean-bundle.zip); only
# the model name is a placeholder. The text is IRSI's, not covered by this project's MIT license, and its
# bundle states no license of its own. Kept unchanged so that our labels are comparable with theirs.
INSTRUCTION = (
    "You are a careful neuroscientist of language models labeling one unit of {model} from its dashboard. Look for one shared "
    "pattern rather than a list of nouns; units often fire on punctuation or the token before where a meaning matters, so read "
    "the surrounding context; the negative side and the promoted/suppressed tokens are clues; consider a core job with small "
    "related offshoots. But do not force a story: if the contexts contain two or more unrelated jobs, say so. Write explanation "
    "(2-3 sentences, specific enough to predict new examples) and category: ONE (one main job, possibly with small related "
    "offshoots), TWO_RELATED (two related jobs), MULTI_UNRELATED (two or more unrelated jobs), UNCLEAR (no confident hypothesis); "
    "confidence in [0,1]."
)
FORMAT = (
    'Respond with only a JSON object and nothing else: {"explanation": <string>, "category": <one of ONE, TWO_RELATED, '
    'MULTI_UNRELATED, UNCLEAR>, "confidence": <number in [0,1]>}.'
)
# Answer format for several dashboards in one request: the single-unit instruction applied to each, answered
# as an array keyed by the id in each dashboard's header.
BATCH_FORMAT = (
    "Respond with only a JSON array and nothing else, one object per dashboard in the order given: "
    '[{"id": <the id in the dashboard\'s header>, "explanation": <string>, "category": <one of ONE, TWO_RELATED, '
    'MULTI_UNRELATED, UNCLEAR>, "confidence": <number in [0,1]>}, ...].'
)


NOTES = (
    "act_frac = activation / this unit's maximum over the corpus. Peak token wrapped in [[ ]]; 48 tokens left, 8 right. "
    "Tokens ranked by direct logit effect (largest first); magnitudes omitted."
)


@dataclass
class Dashboard:
    """What the labeler sees of one unit; to_dict gives its JSON, whose field names are in brackets.

    The contexts show where the pole fires (its input side), the tokens what it pushes the model to say
    next (its output side). A pole whose contexts share one pattern that matches its tokens reads as ONE.

    Fields:
      top [top_contexts]:
          The pole's strongest firings over the whole corpus, strongest first: up to k = 20 contexts
          (fewer if it fired at fewer tokens). The first one has act_frac 1.0 by definition.
      band [band_contexts_30_70pct]:
          Up to band_k = 10 contexts drawn at random from the tokens where the pole fires at 30-70% of
          its maximum, from the scan's band pass (a prefix of the corpus). They show what the pole does
          at ordinary strength, not just at its extremes.
      promoted [promoted_tokens]:
          The 10 tokens whose next-token logits the pole's output direction raises most, strongest
          first: the direct logit effect of down_proj's column (or down_proj @ basis for a control)
          through the final norm and lm_head (see DirectLogitEffect). They depend only on the weights,
          not on the corpus.
      suppressed [suppressed_tokens]:
          The 10 tokens it lowers most. For the negative pole of a neuron, promoted and suppressed
          swap with its positive pole's.
      notes [notes]:
          How to read the fields above, shown to the labeler as part of the dashboard.
      id [id]:
          Optional id written into the JSON, as its first field; None leaves it out. Dashboards built
          here leave it out, since batched prompts show Unit.id in each dashboard's header instead.

    Each context is {"text": ..., "act_frac": ...}:
      text:     up to 48 tokens before the peak token, the peak token wrapped in [[ ]], then up to 8
                tokens after it, decoded from the whole document.
      act_frac: the peak token's |activation| over the pole's maximum |activation| over the corpus,
                rounded to 3 decimals, so it reads the same for every unit whatever its scale.
    """

    top: list[dict]
    band: list[dict]
    promoted: list[str]
    suppressed: list[str]
    notes: str = NOTES
    id: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Dashboard":
        return cls(
            d["top_contexts"],
            d["band_contexts_30_70pct"],
            d["promoted_tokens"],
            d["suppressed_tokens"],
            d["notes"],
            d.get("id"),
        )

    def to_dict(self) -> dict:
        """Fields in a fixed order (id first when set), so from_dict(d).to_dict() keeps d's JSON when d has
        the same order."""
        d = {
            "top_contexts": self.top,
            "band_contexts_30_70pct": self.band,
            "promoted_tokens": self.promoted,
            "suppressed_tokens": self.suppressed,
            "notes": self.notes,
        }
        return d if self.id is None else {"id": self.id} | d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class Unit:
    """One thing to label: one pole and its dashboard, not a whole neuron.

    A neuron gives up to two Units, its positive and its negative pole, since the two fire on different
    tokens and push opposite tokens. A control direction gives one Unit per labeled pole (both for R, the
    positive one for M), and a dashboard loaded from elsewhere is one Unit with an empty meta.
    """

    id: str  # unique within a store; shown to the labeler in batched prompts, so it must not reveal the set
    set: str  # group, reported per group by LabelStore.summary: e.g. N neurons, R random directions, M mixtures
    dashboard: Dashboard
    meta: dict = field(default_factory=dict)  # stored with the label, e.g. layer, neuron, sign


# ---------------------------------------------------------------- dashboards


class DirectLogitEffect:
    """Tokens whose logits a unit's output direction raises most and lowers most, directly.

    This reads the unit's output weights, not activations: what it writes into the residual stream, taken
    straight to the logits as if no later layer changed it. Effects through later layers, which read and
    transform what it writes, are left out, so the tokens are clues to its role, not its whole effect.
    Unlike the logit lens, which unembeds the residual stream of a given input at some layer, this needs
    no input.

    A unit is a direction in the MLP-hidden space (the down_proj input, [N]) and its activation is
    act @ basis. A neuron n is the one-hot basis e_n, so by default its output direction is down_proj's
    column n. Pass `basis` for any other unit, e.g. a control: a random unit vector (R) or
    e_i / max_i + e_j / max_j (M). Its output direction is then down_proj @ basis, the neurons' columns
    weighted by basis, and the pole's unit index is ignored.

    The output direction, times sign, goes through the final RMSNorm scale g and lm_head. Both are the
    same for every unit, so g is folded into lm_head once, in float32 (3.1 GB for Qwen3-32B), instead of
    per call. The norm's division by the residual stream's rms is left out: it only rescales the logits.

      effect = DirectLogitEffect(model, tokenizer)
      promotes, suppresses = effect((12, 4096, 1))
    """

    # Shapes: D hidden size, N MLP neurons, V vocabulary (Qwen3-32B: 5120, 25600, 151936).

    @torch.inference_mode()
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, top: int = 10):
        self.model, self.tokenizer, self.top = model, tokenizer, top
        lm = getattr(model.model, "language_model", model.model)
        lm_head = model.get_output_embeddings().weight  # [V, D]
        # The final RMSNorm of a ones vector, whose rms is 1, is its scale g; read this way because Qwen3
        # stores g as the norm's weight and Qwen3.5 as 1 + weight.
        g = lm.norm(torch.ones(1, lm_head.shape[1], dtype=torch.float32, device=lm_head.device))[0]  # [D]
        self.unembed = lm_head.float() * g.float()  # [V, D], row t: how a residual direction moves token t

    @torch.inference_mode()
    def __call__(self, pole: Pole, basis: Tensor | None = None) -> tuple[list[str], list[str]]:
        """(top tokens raised, top tokens lowered), each decoded one token at a time."""
        layer, n, sign = pole
        weight = get_layers(self.model)[layer].mlp.down_proj.weight  # [D, N], column n is what neuron n writes
        # Only the neurons in the basis: one column for a neuron, two for M; R uses all N.
        idx = torch.tensor([n]) if basis is None else basis.nonzero().squeeze(1)  # [K]
        coef = torch.ones(1) if basis is None else basis[idx].float()  # [K]
        direction = weight[:, idx.to(weight.device)].float() @ coef.to(weight.device) * sign  # [D, K] @ [K] -> [D]
        logits = self.unembed @ direction  # [V, D] @ [D] -> [V]
        return self._decode(logits.topk(self.top).indices), self._decode((-logits).topk(self.top).indices)

    def _decode(self, ids: Tensor) -> list[str]:
        return [self.tokenizer.decode([i]) for i in ids.tolist()]


class DashboardBuilder:
    """Dashboards of scanned poles, and the Units to label them as."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.effect = DirectLogitEffect(model, tokenizer)

    def dashboard(self, result: ScanResult, pole: Pole, basis: Tensor | None = None) -> Dashboard:
        top, band = result.examples(pole, self.tokenizer)
        pole_max = abs(top[0].value) if top else 1.0

        def contexts(examples: list[Example]) -> list[dict]:
            return [{"text": e.context, "act_frac": round(abs(e.value) / pole_max, 3)} for e in examples]

        return Dashboard(contexts(top), contexts(band), *self.effect(pole, basis))

    def units(self, neurons: ScanResult, count: int, seed: int = 0, controls: Controls | None = None) -> list[Unit]:
        """`count` random neuron poles (set N), plus every control pole (sets R and M) when given, in a shuffled
        order that mixes the sets, so labels are blind.

        Ids are opaque (u + 10 hex digits), since batched prompts show them to the labeler; the
        set, layer, neuron (or control index) and sign are kept in each unit's meta.
        """
        poles = [("N", pole) for pole in neurons.sample_poles(count, seed)]
        poles += controls.poles() if controls else []
        random.Random(seed).shuffle(poles)
        units = []
        for set_, pole in poles:
            result, basis = (neurons, None) if set_ == "N" else (controls.result, controls.basis(pole))
            meta = {"layer": pole[0], "neuron": pole[1], "sign": pole[2]}
            uid = "u" + hashlib.sha1(json.dumps({"set": set_} | meta).encode()).hexdigest()[:10]
            units.append(Unit(uid, set_, self.dashboard(result, pole, basis), meta))
        return units


# ---------------------------------------------------------------- labels


@dataclass
class Label:
    id: str
    category: str | None  # one of CATEGORIES; None when the reply could not be parsed
    explanation: str | None = None
    confidence: float | None = None
    raw: str = ""  # the whole reply, kept for failures

    @classmethod
    def parse(cls, uid: str, answer, raw: str) -> "Label | None":
        if not isinstance(answer, dict) or answer.get("category") not in CATEGORIES:
            return None
        return cls(uid, answer["category"], answer.get("explanation"), answer.get("confidence"), raw)


class Labeler:
    """Labels units through a chat API, `batch` dashboards per request.

    `request` goes as is to chat.completions.create, e.g. reasoning_effort="high", max_tokens=32768; leave out
    what an endpoint rejects. A request is retried up to `tries` times on API errors or replies missing some
    unit's label.
    """

    def __init__(self, llm: str, model_name: str, batch: int = 1, tries: int = 3, **request):
        from openai import OpenAI

        self.client = OpenAI()
        self.llm, self.model_name, self.batch, self.tries, self.request = llm, model_name, batch, tries, request

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """The command-line options of from_args."""
        parser.add_argument("--llm", required=True, help="model name on the OpenAI-compatible endpoint")
        parser.add_argument("--batch", type=int, default=1, help="dashboards per request")
        parser.add_argument("--workers", type=int, default=16, help="requests in flight")
        parser.add_argument("--reasoning-effort", help="e.g. high; omitted from the request when not set")
        parser.add_argument("--max-tokens", type=int, help="omitted from the request when not set")

    @classmethod
    def from_args(cls, args: argparse.Namespace, model_name: str) -> "Labeler":
        request = {"reasoning_effort": args.reasoning_effort, "max_tokens": args.max_tokens}
        return cls(args.llm, model_name, args.batch, **{k: v for k, v in request.items() if v is not None})

    def prompt(self, units: list[Unit]) -> str:
        instruction = INSTRUCTION.format(model=self.model_name)
        if len(units) == 1:
            return f"{instruction}\n\nDashboard (JSON):\n{units[0].dashboard.to_json()}\n\n{FORMAT}"
        body = "\n\n".join(f"Dashboard {u.id} (JSON):\n{u.dashboard.to_json()}" for u in units)
        intro = f"Below are {len(units)} dashboards, each of a different unit; label each one on its own."
        return f"{instruction}\n\n{intro}\n\n{body}\n\n{BATCH_FORMAT}"

    def ask(self, units: list[Unit]) -> list[Label]:
        """One label per unit, failed ones with category None."""
        got: dict[str, Label] = {}
        text = ""
        for _ in range(self.tries):
            try:
                reply = self.client.chat.completions.create(
                    model=self.llm, messages=[{"role": "user", "content": self.prompt(units)}], **self.request
                )
                text = reply.choices[0].message.content or ""
            except Exception as e:  # network errors and rate limits: retry, then record the failure
                text = f"ERROR: {e}"
                continue
            got |= self._parse(text, units)
            if len(got) == len(units):
                break
        return [got.get(u.id) or Label(u.id, None, raw=text) for u in units]

    def label_all(self, units: list[Unit], store: "LabelStore", workers: int = 16) -> None:
        """Label every unit the store has no label for, `batch` per request and `workers` requests at a time.

        Batches are cut in the order of `units`; each is stored as soon as its request returns.
        """
        todo = store.pending(units)
        groups = [todo[i : i + self.batch] for i in range(0, len(todo), self.batch)]
        with ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(self.ask, group): group for group in groups}
            for future in as_completed(futures):
                store.add(futures[future], future.result())

    @staticmethod
    def _parse(text: str, units: list[Unit]) -> dict[str, Label]:
        pattern = r"\{.*\}" if len(units) == 1 else r"\[.*\]"
        match = re.search(pattern, text, re.DOTALL)
        try:
            answer = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            return {}
        if len(units) == 1:
            answers = {units[0].id: answer}
        else:
            ids = {u.id for u in units}
            answers = {a["id"]: a for a in answer or [] if isinstance(a, dict) and a.get("id") in ids}
        labels = {uid: Label.parse(uid, a, text) for uid, a in answers.items()}
        return {uid: label for uid, label in labels.items() if label is not None}


class LabelStore:
    """labels.jsonl: one record per attempt, {id, set, **meta, category, explanation, confidence, raw, dashboard}."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def records(self) -> dict[str, dict]:
        """The last record of each unit."""
        if not self.path.exists():
            return {}
        return {r["id"]: r for r in map(json.loads, self.path.read_text().splitlines())}

    def pending(self, units: list[Unit]) -> list[Unit]:
        done = self.categories()
        return [u for u in units if u.id not in done]

    def add(self, units: list[Unit], labels: list[Label]) -> None:
        with open(self.path, "a") as f:
            for unit, label in zip(units, labels, strict=True):
                record = (
                    {"id": unit.id, "set": unit.set, **unit.meta}
                    | asdict(label)
                    | {"dashboard": unit.dashboard.to_dict()}
                )
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def categories(self) -> dict[str, str]:
        """id -> category of every labeled unit."""
        return {uid: r["category"] for uid, r in self.records().items() if r["category"]}

    def summary(self) -> dict:
        """Per set, labeled and failed units and the share of each category among the labeled."""
        records = list(self.records().values())
        out = {}
        for s in sorted({r["set"] for r in records}):
            rs = [r for r in records if r["set"] == s]
            labeled = [r for r in rs if r["category"]]
            out[s] = {"labeled": len(labeled), "failed": len(rs) - len(labeled)}
            out[s] |= {c: sum(r["category"] == c for r in labeled) / max(len(labeled), 1) for c in CATEGORIES}
        return out
