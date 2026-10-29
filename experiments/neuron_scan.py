"""Experiment 1: large-corpus scan, then LLM labels for sampled neuron poles, with IRSI's controls.

  scan:     corpus -> per pole, top-k contexts and a sample from the 30-70%-of-max band (scan.pt)
  controls: R random directions and M planted mixtures per layer, scanned like neurons (controls.pt)
  label:    sampled neuron poles, plus the controls with --controls, as blind IRSI-style dashboards
            -> LLM labels (labels.jsonl) -> share of each category per set (labels_summary.json)

Run from the project root (scan and controls use every listed GPU, one model replica each):
  uv run python -m experiments.neuron_scan scan --corpus corpus.jsonl --out runs/scan --devices 0,1,2,3,4,5,6,7
  uv run python -m experiments.neuron_scan controls --run runs/scan --devices 0,1,2,3,4,5,6,7
  uv run python -m experiments.neuron_scan label --run runs/scan --llm <model> [--controls] [--batch 12] [--dry-run]
The label step reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from infra import load_model
from pipeline import Controls, Corpus, DashboardBuilder, Labeler, LabelStore, Scanner, ScanResult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="./models/Qwen3.5-4B")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--corpus", required=True, help="JSONL with a 'text' field")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=10_000_000)
    p.add_argument("--band-tokens", type=int, default=2_000_000)
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--devices", default="0", help="comma-separated GPU indices, e.g. 0,1,2,3")
    p = sub.add_parser("controls", help="scan R and M control units of an existing run")
    p.add_argument("--run", required=True)
    p.add_argument("--per-layer", type=int, default=4, help="R directions and M mixtures per layer")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--devices", default="0", help="comma-separated GPU indices, e.g. 0,1,2,3")
    p = sub.add_parser("label")
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="cuda:0", help="GPU for the promoted / suppressed tokens")
    p.add_argument("--count", type=int, default=500, help="neuron poles to sample")
    p.add_argument("--controls", action="store_true", help="also label every unit in controls.pt (run controls first)")
    p.add_argument("--dry-run", action="store_true", help="print the first prompt and stop")
    Labeler.add_arguments(p)
    args = parser.parse_args()

    if args.command in ("scan", "controls"):
        devices = [f"cuda:{i}" for i in args.devices.split(",")]
        scanner = Scanner(args.model, devices, args.batch_size)
    if args.command == "scan":
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        corpus = Corpus.from_jsonl(args.corpus, tokenizer, args.max_tokens, args.window)
        scanner.scan(corpus, args.band_tokens).save(Path(args.out) / "scan.pt")
    elif args.command == "controls":
        neurons = ScanResult.load(Path(args.run) / "scan.pt")
        controls = Controls.make(neurons, args.per_layer)
        controls.scan(scanner, neurons)
        controls.save(Path(args.run) / "controls.pt")
    else:
        run = Path(args.run)
        neurons = ScanResult.load(run / "scan.pt")
        controls = Controls.load(run / "controls.pt", neurons) if args.controls else None
        tokenizer, model = load_model(args.model, args.device)
        units = DashboardBuilder(model, tokenizer).units(neurons, args.count, controls=controls)
        labeler = Labeler.from_args(args, Path(args.model).name)
        if args.dry_run:
            print(labeler.prompt(units[: args.batch]))
            return
        store = LabelStore(run / "labels.jsonl")
        labeler.label_all(units, store, args.workers)
        summary = store.summary()
        (run / "labels_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
