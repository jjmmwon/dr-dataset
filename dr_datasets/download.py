"""Dispatcher for `python -m dr_datasets.download <name>`.

Download scripts live in `scripts/` at the repo root (see docs/datasets/<name>.md
for what each one does). This dispatcher is a convenience for running them from
a `git clone` of this repo — it resolves `scripts/` relative to this file, so it
only works when `dr_datasets/` and `scripts/` are still siblings on disk (a repo
checkout, editable install, etc). If you `pip install`-ed this package without
the rest of the repo, run the script directly from a clone instead.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import paths, registry

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DOCS_DIR = REPO_ROOT / "docs" / "datasets"


def download(name: str) -> None:
    info = registry.DATASETS.get(name)
    if info is None:
        raise SystemExit(
            f"Unknown dataset '{name}'. Known datasets: {sorted(registry.DATASETS)}"
        )

    doc_path = DOCS_DIR / info.doc

    if info.script is None:
        print(f"'{name}' has no automated download script.")
        if doc_path.is_file():
            print(doc_path.read_text(encoding="utf-8"))
        else:
            print(f"See docs/datasets/{info.doc} for manual download instructions.")
        return

    script_path = SCRIPTS_DIR / info.script
    if not script_path.is_file():
        raise SystemExit(
            f"Can't find {script_path}. `dr_datasets.download` only works from a git "
            f"checkout of the dr-datasets repo (dr_datasets/ and scripts/ as siblings). "
            f"Clone the repo and run `python scripts/{info.script}` directly instead."
        )

    print(f"[{name}] running scripts/{info.script} -> {paths.data_home() / name}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a dr-datasets dataset into DR_DATA_HOME (default /root/dataset)."
    )
    parser.add_argument("name", choices=sorted(registry.DATASETS))
    args = parser.parse_args()
    download(args.name)


if __name__ == "__main__":
    main()
