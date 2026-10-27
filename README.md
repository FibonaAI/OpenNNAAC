# OpenNNAAC

An open-source implementation of the neuron experiments in Goodfire's blog post
[Neural Networks Are Already Clean](https://already-clean.goodfire.com) (Nick Cammarata).
Find out what the MLP neurons of an LLM do, and test whether they cause the behavior they
correlate with.

## Experiments

| | Approach | Question | Status |
|---|---|---|---|
| 1 | Large-corpus scan | Bottom-up: what does each neuron respond to in ordinary text? | Implemented |
| 2 | Contrast | Top-down: which neurons separate paired prompts that differ in one property? | Planned |
| 3 | Causal intervention | Does replacing or zeroing candidate neurons change the model's behavior? | Planned |

Correlated activation is not a mechanism: candidates from experiments 1 and 2 are only
hypotheses until experiment 3 tests them.

## Setup

Requires uv and NVIDIA GPUs with BF16 support. Run every command from the project root:

```bash
uv sync --locked
```

Scripts that take `--model` accept a local model directory (the default is under `./models/`)
or a Hugging Face model id, downloaded once into the Hugging Face cache.

## Entry scripts

### `fetch_corpus.py`: prepare a text corpus

Downloads one parquet shard of a dataset and writes its first `--docs` documents as JSONL
(`{"id", "text"}` per line), the corpus format of experiment 1:

```bash
uv run fetch_corpus.py --out data/fineweb-edu.jsonl                 # ModelScope mirror
uv run fetch_corpus.py --out data/fineweb-edu.jsonl --source hf     # Hugging Face
```

- Default data: the first shard of `HuggingFaceFW/fineweb-edu` `sample/10BT` (about 2 GB,
  roughly 700M tokens), first 50,000 documents (about 50M tokens).
- `--revision` takes a branch, tag or commit of the chosen source; it is resolved to a commit,
  recorded with the document count in `<out>.manifest.json`, and the shard is cached per
  commit, so reruns do not download again.
- From mainland China, ModelScope is fastest without the proxy: add `.modelscope.cn` to
  `no_proxy` (about 20 MB/s, versus about 10 MB/s for Hugging Face through the proxy).

## Experiment 1: large-corpus scan

```bash
uv run python -m experiments.scan scan --corpus data/fineweb-edu.jsonl --out runs/scan \
    --devices 0,1,2,3,4,5,6,7
uv run python -m experiments.scan label --run runs/scan --llm <model> --dry-run
```

`scan` runs one model replica per listed GPU. For every neuron and direction it keeps the 20
strongest tokens and a sample of tokens at 30–70% of that maximum, and saves them to
`<out>/scan.pt` with enough to recover each token's context. The defaults (10M tokens,
1024-token windows, batch 8) take about 5.5 minutes on 8 H20s.

`label` shows sampled neuron directions to an LLM through an OpenAI-compatible API
(`OPENAI_API_KEY`, `OPENAI_BASE_URL`) and asks it to name them; `--dry-run` prints the first
prompt without calling the API.
