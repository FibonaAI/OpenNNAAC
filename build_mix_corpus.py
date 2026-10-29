"""Build a mixed corpus with IRSI's source proportions and export it as JSONL for scan.

IRSI's Qwen3-32B corpus (https://irsi.ai/notes/already-clean/, code/llm/build_corpus.py) is 70% FineWeb
sample-10BT, 15% Wikipedia 20231101 in 10 languages (1.5% each) and 15% codeparrot/github-code-clean.
Neither FineWeb nor github-code-clean is mirrored on ModelScope and Hugging Face is too slow from here,
so this build substitutes:
  web:  FineWeb-Edu sample-10BT (an educational-quality filtered subset of FineWeb)
  code: bigcode/the-stack-smol, an equal share for each of a dozen common languages
Wikipedia is the same dataset and dump as IRSI's. Like IRSI, texts shorter than 200 characters are skipped.

Budgets count tokens after each document is cut to --max-doc-tokens, i.e. the tokens scan will see.
Documents of all sources are shuffled together, so any prefix of the corpus (scan's band pass reads
one) keeps the mix. One random parquet file per source is read; rows are sampled from it at random.
A <out>.manifest.json records per-source files, commits, documents and tokens.

Run from the project root:
  uv run build_mix_corpus.py --out data/mix-55m.jsonl --tokens 55000000
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from pipeline.corpus import ModelScopeHub

WIKI_LANGS = ["zh", "ru", "es", "de", "ja", "ko", "ar", "hi", "bn", "fr"]
# Smallest parquet file of each language's 20231101 dump on ModelScope, to keep downloads short.
WIKI_FILES = {
    "zh": "train-00002-of-00006",
    "ru": "train-00007-of-00021",
    "es": "train-00010-of-00013",
    "de": "train-00014-of-00020",
    "ja": "train-00014-of-00015",
    "ko": "train-00002-of-00003",
    "ar": "train-00004-of-00007",
    "hi": "train-00001-of-00002",
    "bn": "train-00001-of-00002",
    "fr": "train-00011-of-00017",
}
CODE_LANGS = ["java", "javascript", "c", "php", "python", "c++", "c-sharp", "go", "typescript", "ruby", "shell", "rust"]


def sources() -> list[dict]:
    """(name, share of the budget, ModelScope dataset, file, text field) for every source."""
    out = [
        {
            "name": "web",
            "share": 0.70,
            "dataset": "HuggingFaceFW/fineweb-edu",
            "file": "sample/10BT/000_00000.parquet",
            "field": "text",
        }
    ]
    for lang in WIKI_LANGS:
        out.append(
            {
                "name": f"wiki_{lang}",
                "share": 0.15 / len(WIKI_LANGS),
                "dataset": "wikimedia/wikipedia",
                "file": f"20231101.{lang}/{WIKI_FILES[lang]}.parquet",
                "field": "text",
            }
        )
    for lang in CODE_LANGS:
        out.append(
            {
                "name": f"code_{lang}",
                "share": 0.15 / len(CODE_LANGS),
                "dataset": "bigcode/the-stack-smol",
                "file": f"data/{lang}/data.json",
                "field": "content",
            }
        )
    return out


def read_texts(path: Path, field: str) -> list[str]:
    if path.name.endswith(".parquet") or ".parquet@" in path.name:
        return pq.read_table(path, columns=[field]).column(field).to_pylist()
    with open(path) as f:
        return [json.loads(line)[field] for line in f if line.strip()]


def take(texts: list[str], tokenizer, budget: int, max_doc_tokens: int, rng: random.Random) -> list[tuple[str, int]]:
    """Random texts of at least 200 characters until their tokens, each cut to max_doc_tokens, reach budget."""
    order = list(range(len(texts)))
    rng.shuffle(order)
    picked, total, batch = [], 0, 256
    for start in range(0, len(order), batch):
        chunk = [texts[i] for i in order[start : start + batch] if texts[i] and len(texts[i]) >= 200]
        for text, ids in zip(chunk, tokenizer(chunk)["input_ids"], strict=True):
            n = min(len(ids), max_doc_tokens)
            picked.append((text, n))
            total += n
            if total >= budget:
                return picked
    raise RuntimeError(f"only {total} of {budget} tokens available")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", type=int, required=True, help="total budget, counted after the per-document cut")
    parser.add_argument("--max-doc-tokens", type=int, default=1024, help="should match scan's --window")
    parser.add_argument("--tokenizer", default="./models/Qwen3.5-4B")
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(args.seed)
    docs, manifest = [], []
    hub = ModelScopeHub(out.parent)
    for src in sources():
        path, commit = hub.fetch(src["dataset"], src["file"])
        budget = round(args.tokens * src["share"])
        picked = take(read_texts(path, src["field"]), tokenizer, budget, args.max_doc_tokens, rng)
        docs += [(src["name"], text) for text, _ in picked]
        tokens = sum(n for _, n in picked)
        manifest.append({**src, "commit": commit, "docs": len(picked), "tokens": tokens})
        print(f"{src['name']}: {len(picked)} documents, {tokens} tokens", flush=True)

    rng.shuffle(docs)
    with open(out, "w") as f:
        for i, (name, text) in enumerate(docs):
            f.write(json.dumps({"id": i, "source": name, "text": text}, ensure_ascii=False) + "\n")
    meta = {"tokenizer": args.tokenizer, "max_doc_tokens": args.max_doc_tokens, "seed": args.seed, "sources": manifest}
    out.with_name(out.name + ".manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(docs)} documents -> {out}")


if __name__ == "__main__":
    main()
