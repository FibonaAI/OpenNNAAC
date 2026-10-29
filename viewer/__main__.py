"""Write an HTML page for browsing labeled units: a filterable unit table and a page per unit.

Each argument is one labeler: a run directory with labels.jsonl, or a labels.jsonl file, optionally
named as name=path (default: the directory's name). Units are joined by id, so labelers of the same
units show side by side. --reference adds someone else's description of the units, such as an author's
write-up, shown beside the labels on each unit's page.

Run from the project root:
  uv run python -m viewer gemini=runs/irsi-check-gemini claude=runs/irsi-check-claude-b12 \
      --out runs/viewer/irsi-check.html
Then open the page in a browser; it needs no server.
"""

import argparse
from pathlib import Path

from .page import LabelViewer


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m viewer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("labelers", nargs="+", help="[name=]run directory or labels.jsonl")
    parser.add_argument("--out", required=True, help="the HTML file to write")
    parser.add_argument("--title", default="Labels")
    parser.add_argument(
        "--reference", action="append", default=[], help="name=reference.jsonl: someone else's description of units"
    )
    args = parser.parse_args()

    viewer = LabelViewer(args.title)
    for spec in args.labelers:
        name, _, path = spec.rpartition("=")
        path = Path(path)
        file = path / "labels.jsonl" if path.is_dir() else path
        viewer.add(name or (path.name if path.is_dir() else path.parent.name), file)
    for spec in args.reference:
        name, _, path = spec.partition("=")
        viewer.add_reference(name, path)
    viewer.save(args.out)
    print(f"{len(viewer.units)} units, {len(viewer.labelers)} labelers -> {args.out}")


if __name__ == "__main__":
    main()
