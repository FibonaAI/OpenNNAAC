"""Download one parquet shard of a text dataset and export its first documents as JSONL for scan.

  modelscope: www.modelscope.cn mirror; fast from mainland China (add .modelscope.cn to no_proxy).
              The shard is cached next to --out.
  hf:         huggingface.co, or the endpoint in HF_ENDPOINT; cached in the Hugging Face cache.

--revision is a branch, tag or commit of the chosen source (default: master on modelscope, main
on hf); the two sources use different commit ids. It is resolved to a commit before downloading,
and the shard is cached per commit, so rerunning with a larger --docs does not download again
while a moved branch does. A <out>.manifest.json records the resolved commit and document count.

Run from the project root:
  uv run fetch_corpus.py --out data/fineweb-edu.jsonl
"""

import argparse
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def modelscope_url(dataset: str, file: str, revision: str) -> str:
    return f"https://www.modelscope.cn/api/v1/datasets/{dataset}/repo?Revision={revision}&FilePath={file}"


def modelscope_commit(dataset: str, file: str, revision: str) -> str:
    """ModelScope has no public ref API; its download redirect names the commit in `revision=`."""
    try:
        urllib.request.build_opener(_NoRedirect).open(modelscope_url(dataset, file, revision))
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return parse_qs(urlsplit(e.headers["Location"]).query)["revision"][0]
        raise
    raise RuntimeError(f"ModelScope did not redirect {dataset}/{file}@{revision} to a commit")


def download(url: str, path: Path) -> None:
    """Download to path through a .part file, so an interrupted download is never mistaken for a cache hit."""
    if path.exists():
        return
    part = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    part.rename(path)


def fetch_shard(source: str, dataset: str, file: str, revision: str | None, cache_dir: Path) -> tuple[Path, str]:
    """Download the shard at revision unless cached; returns its path and the commit it resolved to."""
    if source == "hf":
        commit = HfApi().dataset_info(dataset, revision=revision).sha
        return Path(hf_hub_download(dataset, file, repo_type="dataset", revision=commit)), commit
    commit = modelscope_commit(dataset, file, revision or "master")
    path = cache_dir / f"{dataset.replace('/', '--')}--{file.replace('/', '--')}@{commit}"
    download(modelscope_url(dataset, file, commit), path)
    return path, commit


def export(shard: Path, out: Path, docs: int) -> int:
    """Write the first `docs` rows of shard as {"id", "text"} lines; returns the number written."""
    n = 0
    with open(out, "w") as f:
        for batch in pq.ParquetFile(shard).iter_batches(columns=["id", "text"]):
            for row in batch.to_pylist():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                if n == docs:
                    return n
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["modelscope", "hf"], default="modelscope")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--file", default="sample/10BT/000_00000.parquet")
    parser.add_argument(
        "--revision", help="branch, tag or commit of --source; default: master on modelscope, main on hf"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--docs", type=int, default=50_000, help="fineweb-edu averages about 1k tokens per document")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shard, commit = fetch_shard(args.source, args.dataset, args.file, args.revision, out.parent)
    n = export(shard, out, args.docs)
    manifest = {
        "source": args.source,
        "dataset": args.dataset,
        "file": args.file,
        "revision": args.revision,
        "commit": commit,
        "docs": n,
    }
    out.with_name(out.name + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{n} documents -> {out}")


if __name__ == "__main__":
    main()
