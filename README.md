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

### `build_mix_corpus.py`: prepare the corpus

Builds the corpus of experiment 1 as JSONL (`{"id", "source", "text"}` per line) with the source
proportions of IRSI's Qwen3-32B corpus: 70% web, 15% Wikipedia in 10 languages, 15% code.

```bash
uv run build_mix_corpus.py --out data/mix-55m.jsonl --tokens 55000000
```

- Sources: FineWeb-Edu `sample/10BT` for web, Wikipedia `20231101` for zh, ru, es, de, ja, ko,
  ar, hi, bn and fr, and `bigcode/the-stack-smol` for a dozen programming languages. IRSI used
  FineWeb and `codeparrot/github-code-clean`, which are not mirrored on ModelScope.
- Budgets count tokens after each document is cut to `--max-doc-tokens` (1024, scan's window),
  and documents of all sources are shuffled together, so any prefix keeps the mix. The 55M-token
  build above has 94,931 documents.
- Files come from ModelScope and are cached next to `--out` per commit; `<out>.manifest.json`
  records each source's file, commit, documents and tokens. From mainland China, ModelScope is
  fastest without the proxy: add `.modelscope.cn` to `no_proxy`.

## Code layout

- `infra/`: model loading, MLP hooks, aggregators, multi-GPU collection and edits; nothing experiment-specific.
- `pipeline/`: building blocks of the neuron experiments, without entry points.
  `corpus.py` (`DatasetHub`, `Corpus`), `scanner.py` (`Scanner`, `ScanResult`),
  `controls.py` (IRSI's R and M control units) and `label.py` (`DirectLogitEffect`, `Dashboard`,
  `DashboardBuilder`, `Labeler`, `LabelStore`).
- `experiments/`: entry scripts, one per experiment. The corpus preparation script,
  `build_mix_corpus.py`, is at the project root.

## Experiment 1: large-corpus scan

```bash
uv run python -m experiments.neuron_scan scan --corpus data/mix-55m.jsonl --out runs/scan \
    --devices 0,1,2,3,4,5,6,7
uv run python -m experiments.neuron_scan controls --run runs/scan --devices 0,1,2,3,4,5,6,7
uv run python -m experiments.neuron_scan label --run runs/scan --llm <model> --controls --dry-run
```

`scan` runs one model replica per listed GPU. For every neuron pole it keeps the 20 strongest
tokens and a sample of tokens at 30–70% of that maximum, and saves them to `<out>/scan.pt` with
enough to recover each token's context. The defaults (10M tokens, 1024-token windows, batch 8)
take about 5.5 minutes on 8 H20s.

`controls` scans IRSI's control units the same way into `controls.pt`: per layer, random
directions (R) and planted mixtures of two neurons (M).

`label` turns sampled neuron poles, and with `--controls` every control unit, into IRSI-style
dashboards in a blind shuffled order, and labels them through an OpenAI-compatible API
(`OPENAI_API_KEY`, `OPENAI_BASE_URL`) with IRSI's instruction; `--batch 12` sends 12 dashboards
per request and `--dry-run` prints the first prompt without calling the API. Labels go to
`labels.jsonl` (resumable) and the share of each category per set to `labels_summary.json`.

`experiments/irsi_check.py` checks the pipeline against IRSI's released Qwen3-32B dashboards
(activations, promoted / suppressed tokens, labels).

## Acknowledgements

Experiment 1 follows IRSI's replication on Qwen3-32B
([irsi.ai/notes/already-clean](https://irsi.ai/notes/already-clean/)): its labeling protocol,
control units and corpus mix, checked against its released dashboards.

## License

MIT, see [LICENSE](LICENSE).
