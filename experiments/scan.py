"""Experiment 1: large-corpus scan, then LLM labels for sampled neuron poles.

A pole is one firing direction of a neuron: (layer, neuron, sign), sign +1 or -1.

  scan:  corpus -> per pole, top-k contexts and a sample from the 30-70%-of-max band
  label: sample poles -> contexts (48 tokens before, peak, 8 after) + promoted and
         suppressed output tokens -> OpenAI-compatible chat API -> label

Run from the project root (scan uses every listed GPU, one model replica each):
  uv run python -m experiments.scan scan --corpus corpus.jsonl --out runs/scan --devices 0,1,2,3,4,5,6,7
  uv run python -m experiments.scan label --run runs/scan --llm <model> [--dry-run]
The label step reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment.
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from infra import BandSample, TopK, get_layers, load_model, parallel_collect

Pole = tuple[int, int, int]  # (layer, neuron, sign)


# ---------------------------------------------------------------- corpus


def load_corpus(path: str, tokenizer: PreTrainedTokenizerBase, max_tokens: int, max_doc_tokens: int) -> list[list[int]]:
    """Token ids of the documents in a JSONL file with a "text" field.

    Each document keeps only its first max_doc_tokens tokens, so with max_doc_tokens = window it
    fits in one window: the model sees every token with its full preceding context, and a few
    very long documents cannot use up the budget. max_tokens caps the total length of all kept
    tokens together; only the last document is cut further to fit.
    """
    # TODO: tokens past max_doc_tokens are never scanned (about 40% of fineweb-edu tokens at
    # 1024, mostly from the ~4% of documents over 4096 tokens), so neurons that respond to late
    # parts of long documents are missed. Options: a 2048 window (~90% of documents whole, about
    # 2x compute), or overlapping windows that only count tokens in their second half.
    docs, total = [], 0
    with open(path) as f:
        for line in f:
            ids = tokenizer(json.loads(line)["text"])["input_ids"]
            docs.append(ids[: min(max_doc_tokens, max_tokens - total)])
            total += len(docs[-1])
            if total >= max_tokens:
                break
    return docs


def make_windows(docs: list[list[int]], window: int) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Split each document into consecutive windows of at most `window` tokens.

    Returns (windows, origins):
      windows[i]: token ids of window i; the last window of a document may be shorter
      origins[i]: (doc, offset), i.e. window i is docs[doc][offset:offset + window]

    Example:
      >>> make_windows([[1, 2, 3, 4, 5], [6, 7]], window=2)
      ([[1, 2], [3, 4], [5], [6, 7]], [(0, 0), (0, 2), (0, 4), (1, 0)])
    """
    windows, origins = [], []
    for d, ids in enumerate(docs):
        for off in range(0, len(ids), window):
            windows.append(ids[off : off + window])
            origins.append((d, off))
    return windows, origins


# ---------------------------------------------------------------- scan


def make_topk(n_layers: int, n_neurons: int, k: int, device: str, rank: int) -> TopK:
    return TopK(n_layers, n_neurons, k, device)


def make_band(pole_max: Tensor, k: int, device: str, rank: int) -> BandSample:
    return BandSample(pole_max, k, device, seed=rank)


def scan(
    model_id: str,
    corpus: str,
    out: str,
    devices: list[str],
    max_tokens: int = 10_000_000,
    band_tokens: int = 2_000_000,
    window: int = 1024,
    k: int = 20,
    band_k: int = 10,
    batch_size: int = 8,
) -> None:
    """Pass 1 over the corpus: top-k per pole. Pass 2 over its first band_tokens: band sample.

    max_tokens caps the corpus for pass 1; band_tokens caps pass 2, which only needs a few
    in-band tokens per pole and runs on a prefix of the same windows. Pass 2 takes whole
    windows until their total reaches band_tokens, so it may overshoot by under one window.

    Example: max_tokens=10_000_000, band_tokens=2_000_000, window=1024. Pass 1 scans all
    ~10M tokens to find each pole's max; pass 2 rescans only the first ~2M tokens (about
    2000 windows) and samples band_k in-band tokens per pole from them.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id).get_text_config()
    n_layers, n_neurons = config.num_hidden_layers, config.intermediate_size
    docs = load_corpus(corpus, tokenizer, max_tokens, max_doc_tokens=window)
    windows, origins = make_windows(docs, window)
    pad = tokenizer.pad_token_id

    top = parallel_collect(model_id, windows, partial(make_topk, n_layers, n_neurons, k), pad, devices, batch_size)

    pole_max = torch.stack([top.pos_values[..., 0], -top.neg_values[..., 0]]).clamp(min=0)
    n_band, seen = 0, 0
    while n_band < len(windows) and seen < band_tokens:
        seen += len(windows[n_band])
        n_band += 1
    band = parallel_collect(model_id, windows[:n_band], partial(make_band, pole_max, band_k), pad, devices, batch_size)

    Path(out).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "docs": docs,
            "origins": origins,
            "top": top.state_dict(),
            "band": band.state_dict(),
            "config": {
                "corpus": corpus,
                "tokens": sum(map(len, docs)),
                "window": window,
                "band_tokens": seen,
                "k": k,
                "band_k": band_k,
            },
        },
        Path(out) / "scan.pt",
    )


# ---------------------------------------------------------------- contexts


@dataclass
class Example:
    value: float  # signed activation
    context: str  # 48 tokens before, the peak token in [[ ]], 8 tokens after


def examples(
    run: dict, tokenizer: PreTrainedTokenizerBase, pole: Pole, before: int = 48, after: int = 8
) -> tuple[list[Example], list[Example]]:
    """(top examples, band examples) of one pole, strongest first for top.

    Args:
      run: the dict saved by `scan` as scan.pt. Uses:
        docs:    list[list[int]], token ids of every corpus document
        origins: list[tuple[int, int]], origins[w] = (doc, offset), where window w starts in docs
        top:     dict[str, Tensor], TopK state on CPU, each [L, N, k]:
                   pos_values / neg_values:       float32, signed activations (empty: -inf / +inf)
                   pos_docs / neg_docs:           long, window index (empty: -1)
                   pos_positions / neg_positions: long, offset in that window (empty: -1)
        band:    dict[str, Tensor], BandSample state on CPU, each [2, L, N, band_k], index 0 = pos:
                   values:    float32, signed activations (empty: nan)
                   docs:      long, window index (empty: -1)
                   positions: long, offset in that window (empty: -1)
      tokenizer: decodes token ids back to text.
      pole: (layer, neuron, sign), sign +1 for the positive pole, -1 for the negative one.
      before / after: tokens of context kept before / after the peak token.

    Returns:
      (top, band), two lists of Example. top holds up to k examples, strongest first;
      band holds up to band_k in-band samples in no particular order. Empty slots are skipped,
      so either list may be shorter or empty. Each Example has:
        value:   signed activation of the peak token (negative for the negative pole)
        context: up to `before` tokens, the peak token in [[ ]], then up to `after` tokens,
                 taken from the whole document so context crosses window boundaries

    Example: pole (12, 4096, 1) whose strongest top slot is value 7.8, window 153, position 200,
    with origins[153] = (37, 1024). The token is docs[37][1224], giving
      Example(7.8, "...the sofa and curled up. My[[ cat]] always sleeps there...")
    """
    layer, n, sign = pole
    s = 0 if sign > 0 else 1
    name = "pos" if sign > 0 else "neg"
    top = run["top"]
    sources = {
        "top": (top[f"{name}_values"][layer, n], top[f"{name}_docs"][layer, n], top[f"{name}_positions"][layer, n]),
        "band": (
            run["band"]["values"][s, layer, n],
            run["band"]["docs"][s, layer, n],
            run["band"]["positions"][s, layer, n],
        ),
    }
    out = {}
    for kind, (values, wins, positions) in sources.items():
        out[kind] = []
        for v, w, p in zip(values.tolist(), wins.tolist(), positions.tolist(), strict=True):
            if w < 0:
                continue
            doc, off = run["origins"][w]
            ids, at = run["docs"][doc], off + p  # position inside the whole document
            context = (
                tokenizer.decode(ids[max(0, at - before) : at])
                + "[["
                + tokenizer.decode(ids[at : at + 1])
                + "]]"
                + tokenizer.decode(ids[at + 1 : at + 1 + after])
            )
            out[kind].append(Example(v, context))
    return out["top"], out["band"]


@torch.inference_mode()
def output_effects(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, pole: Pole, top: int = 10
) -> tuple[list[str], list[str]]:
    """Tokens whose logits the pole's output direction raises most and lowers most (logit lens).

    Uses down_proj's column, the final RMSNorm scale (Qwen3.5 stores it as 1 + weight), and lm_head.
    """
    layer, n, sign = pole
    lm = getattr(model.model, "language_model", model.model)
    direction = get_layers(model)[layer].mlp.down_proj.weight[:, n].float() * sign
    direction = direction * (1.0 + lm.norm.weight.float())
    logits = model.get_output_embeddings().weight.float() @ direction

    def decode(ids):
        return [tokenizer.decode([i]) for i in ids.tolist()]

    return decode(logits.topk(top).indices), decode((-logits).topk(top).indices)


# ---------------------------------------------------------------- label

SYSTEM = """You are analyzing one unit inside a language model. You will see text excerpts where \
the unit fires, with the token it fires on marked as [[token]] and the activation strength in \
parentheses, plus the output tokens the unit makes more and less likely. Work out what single \
job the unit does. Be specific: prefer "the word 'bank' meaning a riverside" over "nouns"."""

ANSWER = """Answer in exactly this format:
LABEL: <a short phrase naming what the unit responds to>
EXPLANATION: <one or two sentences, including what the context needs for it to fire>"""


def build_prompt(top: list[Example], band: list[Example], promotes: list[str], suppresses: list[str]) -> str:
    lines = ["Strongest activations:"]
    lines += [f"{i}. ({e.value:+.2f}) {e.context!r}" for i, e in enumerate(top, 1)]
    if band:
        lines += ["", "Medium activations (30-70% of the strongest):"]
        lines += [f"{i}. ({e.value:+.2f}) {e.context!r}" for i, e in enumerate(band, 1)]
    lines += ["", f"Output tokens it promotes: {promotes}", f"Output tokens it suppresses: {suppresses}", "", ANSWER]
    return "\n".join(lines)


def parse_answer(text: str) -> tuple[str, str]:
    fields = {"LABEL": "", "EXPLANATION": ""}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip().upper() in fields:
            fields[key.strip().upper()] = value.strip()
    return fields["LABEL"], fields["EXPLANATION"]


def sample_poles(run: dict, count: int, seed: int = 0) -> list[Pole]:
    """Random poles that fired at least once in the scan."""
    n_layers, n_neurons, _ = run["top"]["pos_values"].shape
    rng = random.Random(seed)
    poles: list[Pole] = []
    while len(poles) < count:
        layer, n, sign = rng.randrange(n_layers), rng.randrange(n_neurons), rng.choice((1, -1))
        name = "pos" if sign > 0 else "neg"
        if run["top"][f"{name}_docs"][layer, n, 0] >= 0 and (layer, n, sign) not in poles:
            poles.append((layer, n, sign))
    return poles


def label(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    run_dir: str,
    llm: str,
    count: int = 500,
    seed: int = 0,
    workers: int = 16,
    dry_run: bool = False,
) -> None:
    """Write labels.jsonl in run_dir: one record per sampled pole with its prompt and answer."""
    from openai import OpenAI

    run = torch.load(Path(run_dir) / "scan.pt", weights_only=False)
    poles = sample_poles(run, count, seed)
    prompts = []
    for pole in poles:
        top, band = examples(run, tokenizer, pole)
        prompts.append(build_prompt(top, band, *output_effects(model, tokenizer, pole)))
    if dry_run:
        print(prompts[0])
        return

    client = OpenAI()

    def ask(prompt: str) -> str:
        reply = client.chat.completions.create(
            model=llm, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        )
        return reply.choices[0].message.content or ""

    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(ask, prompts))
    with open(Path(run_dir) / "labels.jsonl", "w") as f:
        for (layer, n, sign), prompt, answer in zip(poles, prompts, answers, strict=True):
            name, explanation = parse_answer(answer)
            f.write(
                json.dumps(
                    {
                        "layer": layer,
                        "neuron": n,
                        "sign": sign,
                        "label": name,
                        "explanation": explanation,
                        "answer": answer,
                        "prompt": prompt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="./models/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda:0", help="GPU for the label step")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--corpus", required=True, help="JSONL with a 'text' field")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=10_000_000)
    p.add_argument("--band-tokens", type=int, default=2_000_000)
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--devices", default="0", help="comma-separated GPU indices, e.g. 0,1,2,3")
    p = sub.add_parser("label")
    p.add_argument("--run", required=True)
    p.add_argument("--llm", required=True, help="model name on the OpenAI-compatible endpoint")
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--dry-run", action="store_true", help="print the first prompt and stop")
    args = parser.parse_args()

    if args.command == "scan":
        devices = [f"cuda:{i}" for i in args.devices.split(",")]
        scan(
            args.model,
            args.corpus,
            args.out,
            devices,
            args.max_tokens,
            args.band_tokens,
            args.window,
            batch_size=args.batch_size,
        )
    else:
        tokenizer, model = load_model(args.model, args.device)
        label(model, tokenizer, args.run, args.llm, args.count, workers=args.workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
