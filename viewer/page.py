"""LabelViewer: one self-contained HTML page from one or more labels.jsonl files."""

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("template.html")
# Record fields that are not the unit's meta: the unit, its label and its dashboard.
_FIELDS = {"id", "set", "category", "explanation", "confidence", "raw", "dashboard"}


def read_labels(path: str | Path) -> dict[str, dict]:
    """The last record of each unit in a labels.jsonl: {id, set, **meta, category, explanation, confidence,
    raw, dashboard} per line, one line per attempt, so a later attempt replaces an earlier one."""
    return {r["id"]: r for r in map(json.loads, Path(path).read_text().splitlines())}


class LabelViewer:
    """Units from one or more labels.jsonl files, joined by unit id; each file is one labeler's column."""

    def __init__(self, title: str):
        self.title = title
        self.labelers: list[str] = []
        self.units: dict[str, dict] = {}  # id -> {id, set, meta, labels: {labeler: label}}
        self.dashboards: dict[str, dict] = {}  # id -> dashboard JSON, shared by every labeler
        self.references: dict[str, dict[str, dict]] = {}  # name -> id -> {title, text, notes, source}

    def add(self, labeler: str, path: str | Path) -> None:
        """Every unit's last label in the labels.jsonl at `path`, under the name `labeler`."""
        self.labelers.append(labeler)
        for uid, r in read_labels(path).items():
            meta = {k: v for k, v in r.items() if k not in _FIELDS}
            unit = self.units.setdefault(uid, {"id": uid, "set": r["set"], "meta": meta, "labels": {}})
            label = {"category": r["category"], "explanation": r["explanation"], "confidence": r["confidence"]}
            # A batched request's raw reply is shared by its whole batch; keep it only where it explains a failure.
            unit["labels"][labeler] = label | ({"raw": r["raw"]} if not r["category"] else {})
            self.dashboards.setdefault(uid, r["dashboard"])

    def add_reference(self, name: str, path: str | Path) -> None:
        """Someone else's description of units, shown beside the labels: a JSONL file with one
        {"id", "title", "text", "notes" (optional list), "source" (optional),
         "scores" (optional {name: {"value", "max", "note"}})} per line. Scores also get list columns."""
        refs = map(json.loads, Path(path).read_text().splitlines())
        self.references[name] = {r["id"]: {k: v for k, v in r.items() if k != "id"} for r in refs}

    def render(self) -> str:
        data = {
            "title": self.title,
            "labelers": self.labelers,
            "units": list(self.units.values()),
            "dashboards": self.dashboards,
            "references": self.references,
        }
        # Escape <, > and & so that nothing in the data, such as "<!--" or "<script" from web text, can end or
        # change the <script> element it is embedded in; JSON.parse reads the \uXXXX escapes back.
        payload = json.dumps(data, ensure_ascii=False)
        payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        title = self.title.replace("&", "&amp;").replace("<", "&lt;")
        return TEMPLATE.read_text().replace("__TITLE__", title).replace("__DATA__", payload)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.render())
