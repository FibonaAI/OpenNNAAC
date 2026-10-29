"""Text corpora: download a dataset shard, and tokenize documents into the windows a scan feeds the model.

DatasetHub: one file of a dataset at a commit, cached; ModelScopeHub and HuggingFaceHub are the two hubs.
Corpus:     token ids of a JSONL corpus, cut into windows.
"""

import json
import shutil
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from huggingface_hub import HfApi, hf_hub_download
from transformers import PreTrainedTokenizerBase

# ---------------------------------------------------------------- download


class DatasetHub(ABC):
    """Where dataset files come from. fetch resolves a revision to a commit first and downloads the file
    at that commit, cached per commit: a rerun does not download again, while a moved branch does."""

    default_revision: str

    @staticmethod
    def create(name: str, cache_dir: Path) -> "DatasetHub":
        """The hub called `name` on the command line: "modelscope" or "hf"."""
        return {"modelscope": ModelScopeHub, "hf": HuggingFaceHub}[name](cache_dir)

    def fetch(self, dataset: str, file: str, revision: str | None = None) -> tuple[Path, str]:
        """(local path, commit) of `file` in `dataset` at revision, a branch, tag or commit."""
        commit = self.commit(dataset, file, revision or self.default_revision)
        return self.download(dataset, file, commit), commit

    @abstractmethod
    def commit(self, dataset: str, file: str, revision: str) -> str:
        """The commit that revision points to; commit ids differ between hubs."""

    @abstractmethod
    def download(self, dataset: str, file: str, commit: str) -> Path:
        """The file at commit, downloaded unless already cached."""


class HuggingFaceHub(DatasetHub):
    """huggingface.co, or the endpoint in HF_ENDPOINT. Files go to the Hugging Face cache, not cache_dir."""

    default_revision = "main"

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir

    def commit(self, dataset: str, file: str, revision: str) -> str:
        return HfApi().dataset_info(dataset, revision=revision).sha

    def download(self, dataset: str, file: str, commit: str) -> Path:
        return Path(hf_hub_download(dataset, file, repo_type="dataset", revision=commit))


class ModelScopeHub(DatasetHub):
    """The www.modelscope.cn mirror, fast from mainland China (add .modelscope.cn to no_proxy). Files go to
    cache_dir as <dataset>--<file>@<commit>, with "/" replaced by "--"."""

    default_revision = "master"

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def url(dataset: str, file: str, revision: str) -> str:
        # quote: file names such as data/c++/data.json would otherwise lose their "+" to the query string
        return f"https://www.modelscope.cn/api/v1/datasets/{dataset}/repo?Revision={revision}&FilePath={quote(file)}"

    def commit(self, dataset: str, file: str, revision: str) -> str:
        """ModelScope has no public ref API; its download redirect names the commit in `revision=`.
        The API sometimes never answers, hence the timeout and retries."""
        for attempt in range(5):
            try:
                urllib.request.build_opener(_NoRedirect).open(self.url(dataset, file, revision), timeout=60)
            except urllib.error.HTTPError as e:
                if e.code in (301, 302, 303, 307, 308):
                    return parse_qs(urlsplit(e.headers["Location"]).query)["revision"][0]
                raise
            except (TimeoutError, urllib.error.URLError):
                if attempt == 4:
                    raise
                continue
            break
        raise RuntimeError(f"ModelScope did not redirect {dataset}/{file}@{revision} to a commit")

    def download(self, dataset: str, file: str, commit: str) -> Path:
        """Through a .part file, so an interrupted download is never mistaken for a cache hit."""
        path = self.cache_dir / f"{dataset.replace('/', '--')}--{file.replace('/', '--')}@{commit}"
        if path.exists():
            return path
        part = path.with_name(path.name + ".part")
        with urllib.request.urlopen(self.url(dataset, file, commit), timeout=300) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        part.rename(path)
        return path


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


# ---------------------------------------------------------------- tokenize


@dataclass
class Corpus:
    """Token ids of documents and the windows a scan feeds the model.

    windows[i] is docs[d][off:off + window] for origins[i] = (d, off); the last window of a document
    may be shorter. Example: docs [[1, 2, 3, 4, 5], [6, 7]] with window 2 give windows
    [[1, 2], [3, 4], [5], [6, 7]] and origins [(0, 0), (0, 2), (0, 4), (1, 0)].
    """

    docs: list[list[int]]
    window: int
    source: str = ""  # where the documents came from, recorded in scan results
    windows: list[list[int]] = field(init=False, repr=False)
    origins: list[tuple[int, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.windows, self.origins = [], []
        for d, ids in enumerate(self.docs):
            for off in range(0, len(ids), self.window):
                self.windows.append(ids[off : off + self.window])
                self.origins.append((d, off))

    @classmethod
    def from_jsonl(cls, path: str, tokenizer: PreTrainedTokenizerBase, max_tokens: int, window: int) -> "Corpus":
        """Documents of a JSONL file with a "text" field, each cut to its first `window` tokens.

        Cut this way, every document fits in one window, so the model sees every token with its full
        preceding context and a few very long documents cannot use up the budget. max_tokens caps the
        total of all kept tokens; only the last document is cut further to fit.
        """
        # TODO: tokens past the window are never scanned (about 40% of fineweb-edu tokens at 1024,
        # mostly from the ~4% of documents over 4096 tokens), so neurons that respond to late parts of
        # long documents are missed. Options: a 2048 window (~90% of documents whole, about 2x
        # compute), or overlapping windows that only count tokens in their second half.
        docs, total = [], 0
        with open(path) as f:
            for line in f:
                ids = tokenizer(json.loads(line)["text"])["input_ids"]
                docs.append(ids[: min(window, max_tokens - total)])
                total += len(docs[-1])
                if total >= max_tokens:
                    break
        return cls(docs, window, path)

    @property
    def tokens(self) -> int:
        return sum(map(len, self.docs))

    def prefix(self, tokens: int) -> tuple[int, int]:
        """(number of leading windows, their tokens): whole windows until they hold `tokens` tokens."""
        n, seen = 0, 0
        while n < len(self.windows) and seen < tokens:
            seen += len(self.windows[n])
            n += 1
        return n, seen
