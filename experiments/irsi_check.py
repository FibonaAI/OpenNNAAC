"""Check our pipeline against IRSI's released Qwen3-32B dashboards (https://irsi.ai/notes/already-clean/).

The bundle (already-clean-bundle.zip, unpacked) holds 768 blinded dashboards in results/A/dash and the key
in results/A/blinding-key/KEY.json mapping each to its set, layer, neuron, pole and pole_max. Sets: N random
neurons, G Goodfire gallery neurons, R random directions, M / M80 planted mixtures, C (see the bundle).

  acts:    re-run every context of the N and G dashboards through the model and read the neuron at the
           [[peak]] token; compare with IRSI's act_frac * pole_max (negative for the neg pole). Contexts
           keep only 48 tokens before the peak, so small differences from IRSI's full-context values are
           expected.
  effects: our top-10 promoted / suppressed tokens (DirectLogitEffect) against the N and G dashboards'.
  label:   label all 768 dashboards with pipeline.label (IRSI's own instruction) on our LLM, then compare
           the ONE rate of each set with IRSI's Claude labels. --batch 12 sends IRSI's Claude batches of 12.

Run from the project root:
  uv run python -m experiments.irsi_check acts --model ./models/Qwen3-32B
  uv run python -m experiments.irsi_check effects --model ./models/Qwen3-32B
  uv run python -m experiments.irsi_check label --llm <model> [--batch 12] [--reasoning-effort high]
Outputs go to --out (default runs/irsi-check). The label step reads OPENAI_API_KEY and OPENAI_BASE_URL.
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor

from infra import collect_ids, load_model
from pipeline import Dashboard, DirectLogitEffect, Labeler, LabelStore, Unit


def load_bundle(bundle: str) -> tuple[dict, dict]:
    """(key, dashboards), both keyed by dashboard id."""
    root = Path(bundle) / "results" / "A"
    key = json.loads((root / "blinding-key" / "KEY.json").read_text())
    dash = {Path(p).stem: json.loads(Path(p).read_text()) for p in glob.glob(str(root / "dash" / "*.json"))}
    return key, dash


def neuron_contexts(key: dict, dash: dict) -> list[dict]:
    """Every top and band context of the N and G dashboards, with the activation IRSI reports."""
    rows = []
    for uid, unit in key.items():
        if unit["set"] not in ("N", "G"):
            continue
        sign = 1.0 if unit["pole"] == "pos" else -1.0
        for kind, field in (("top", "top_contexts"), ("band", "band_contexts_30_70pct")):
            for c in dash[uid].get(field, []):
                rows.append(
                    {
                        "id": uid,
                        "set": unit["set"],
                        "layer": unit["layer"],
                        "neuron": unit["neuron"],
                        "kind": kind,
                        "text": c["text"],
                        "irsi": sign * c["act_frac"] * unit["pole_max"],
                    }
                )
    return rows


class PeakReader:
    """Activation of each row's neuron at its last real token, i.e. the [[peak]]."""

    def __init__(self, layers: list[int], neurons: list[int], device: torch.device):
        self.layers = torch.tensor(layers, device=device)
        self.neurons = torch.tensor(neurons, device=device)
        self.values = torch.full((len(layers),), torch.nan, device=device)

    def __call__(self, layer: int, gate: Tensor, up: Tensor, act: Tensor, mask: Tensor, doc_ids: Tensor) -> None:
        rows = (self.layers[doc_ids] == layer).nonzero().squeeze(1)
        if len(rows) == 0:
            return
        last = mask.sum(1) - 1
        self.values[doc_ids[rows]] = act[rows, last[rows], self.neurons[doc_ids[rows]]].float()


def summarize_acts(rows: list[dict]) -> dict:
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[f"{r['set']}_{r['kind']}"].append(r)
    for name, rs in sorted(groups.items()):
        ours = torch.tensor([r["ours"] for r in rs])
        irsi = torch.tensor([r["irsi"] for r in rs])
        rel = ((ours - irsi).abs() / irsi.abs().clamp(min=1e-6)).sort().values
        out[name] = {
            "contexts": len(rs),
            "pearson": torch.corrcoef(torch.stack([ours, irsi]))[0, 1].item(),
            "same_sign": (ours.sign() == irsi.sign()).float().mean().item(),
            "rel_err_median": rel[len(rel) // 2].item(),
            "within_10pct": (rel <= 0.10).float().mean().item(),
            "within_25pct": (rel <= 0.25).float().mean().item(),
        }
    return out


def acts(model_id: str, bundle: str, out: Path, device: str, batch_size: int) -> None:
    key, dash = load_bundle(bundle)
    rows = neuron_contexts(key, dash)
    tokenizer, model = load_model(model_id, device)
    windows = []
    for r in rows:
        prefix, _, rest = r["text"].partition("[[")
        peak = rest.partition("]]")[0]
        windows.append(tokenizer(prefix)["input_ids"] + tokenizer(peak)["input_ids"])
    reader = PeakReader([r["layer"] for r in rows], [r["neuron"] for r in rows], model.device)
    collect_ids(model, windows, reader, tokenizer.pad_token_id, batch_size)
    for r, v in zip(rows, reader.values.tolist(), strict=True):
        r["ours"] = v
    with open(out / "acts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize_acts(rows)
    (out / "acts_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def effects(model_id: str, bundle: str, out: Path, device: str) -> None:
    key, dash = load_bundle(bundle)
    tokenizer, model = load_model(model_id, device)
    effect = DirectLogitEffect(model, tokenizer)
    rows = []
    for uid, unit in key.items():
        if unit["set"] not in ("N", "G"):
            continue
        sign = 1 if unit["pole"] == "pos" else -1
        promotes, suppresses = effect((unit["layer"], unit["neuron"], sign))
        d = dash[uid]
        rows.append(
            {
                "id": uid,
                "set": unit["set"],
                "promoted_overlap": len(set(promotes) & set(d["promoted_tokens"])) / len(d["promoted_tokens"]),
                "suppressed_overlap": len(set(suppresses) & set(d["suppressed_tokens"])) / len(d["suppressed_tokens"]),
                "ours": [promotes, suppresses],
                "irsi": [d["promoted_tokens"], d["suppressed_tokens"]],
            }
        )
    with open(out / "effects.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {name: sum(r[name] for r in rows) / len(rows) for name in ("promoted_overlap", "suppressed_overlap")} | {
        "units": len(rows),
        "exact_promoted": sum(r["promoted_overlap"] == 1.0 for r in rows),
    }
    (out / "effects_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def irsi_claude_labels(bundle: str) -> dict:
    labels = {}
    for p in glob.glob(str(Path(bundle) / "results" / "A" / "labels" / "claude_batch_*.json")):
        for x in json.loads(Path(p).read_text())["labels"]:
            labels[x["id"]] = x["category"]
    return labels


def summarize_labels(key: dict, ours: dict, claude: dict) -> dict:
    out = {}
    for s in sorted({u["set"] for u in key.values()}):
        ids = [i for i, u in key.items() if u["set"] == s]
        mine = [ours[i] for i in ids if i in ours]
        theirs = [claude[i] for i in ids if i in claude]
        both = [(ours[i], claude[i]) for i in ids if i in ours and i in claude]
        out[s] = {
            "units": len(ids),
            "ours_one": sum(c == "ONE" for c in mine) / max(len(mine), 1),
            "claude_one": sum(c == "ONE" for c in theirs) / max(len(theirs), 1),
            "same_category": sum(a == b for a, b in both) / max(len(both), 1),
            "same_one_vs_not": sum((a == "ONE") == (b == "ONE") for a, b in both) / max(len(both), 1),
            "labeled": len(mine),
        }
    return out


def label(bundle: str, out: Path, labeler: Labeler, workers: int) -> None:
    """Label all 768 dashboards, in ALL_IDS.json order so that batches of 12 are IRSI's Claude batches."""
    key, dash = load_bundle(bundle)
    ids = json.loads((Path(bundle) / "results" / "A" / "labels" / "ALL_IDS.json").read_text())
    store = LabelStore(out / "labels.jsonl")
    labeler.label_all([Unit(uid, key[uid]["set"], Dashboard.from_dict(dash[uid])) for uid in ids], store, workers)
    ours = store.categories()
    summary = summarize_labels(key, ours, irsi_claude_labels(bundle)) | {"failed": len(ids) - len(ours)}
    (out / "labels_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", default="data/irsi/already-clean-bundle")
    parser.add_argument("--out", default="runs/irsi-check")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("acts", "effects"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="./models/Qwen3-32B")
        p.add_argument("--device", default="cuda:0")
        if name == "acts":
            p.add_argument("--batch-size", type=int, default=32)
    p = sub.add_parser("label")
    Labeler.add_arguments(p)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "acts":
        acts(args.model, args.bundle, out, args.device, args.batch_size)
    elif args.command == "effects":
        effects(args.model, args.bundle, out, args.device)
    else:
        labeler = Labeler.from_args(args, "Qwen3-32B")
        label(args.bundle, out, labeler, args.workers)


if __name__ == "__main__":
    main()
