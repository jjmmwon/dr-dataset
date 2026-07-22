"""Loader for the dr-datasets dataset directory (see `dr_datasets.paths.data_home`).

Convention: every dataset lives in `<data_home>/<name>/` and (once downloaded)
exposes an `embeddings.npy` array of shape (N, f). Everything else in that
folder (labels, raw text metadata, PCA loadings, ...) is dataset-specific and
can be read with `load_array`. Datasets aren't shipped with this package —
run `python -m dr_datasets.download <name>` (see `docs/datasets/<name>.md`) to
populate `<data_home>/<name>/` first.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import paths, registry

DATASETS_DIR = paths.data_home()


def _download_hint(name: str) -> str:
    info = registry.DATASETS.get(name)
    if info is None:
        return ""
    if info.script is None:
        return f" No download script is available — see docs/datasets/{info.doc}."
    return (
        f" Run `python -m dr_datasets.download {name}` "
        f"(or `python scripts/{info.script}`) to fetch it; see docs/datasets/{info.doc}."
    )


def list_datasets() -> List[str]:
    """Names of all datasets that have a metadata.json under datasets/."""
    if not DATASETS_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in DATASETS_DIR.iterdir()
        if p.is_dir() and (p / "metadata.json").is_file()
    )


def dataset_dir(name: str) -> Path:
    d = DATASETS_DIR / name
    if not d.is_dir():
        if name in registry.DATASETS:
            raise FileNotFoundError(
                f"'{name}' hasn't been downloaded yet (looked in {d})."
                f"{_download_hint(name)}"
            )
        raise FileNotFoundError(
            f"Unknown dataset '{name}'. Known datasets: {sorted(registry.DATASETS)}. "
            f"Locally downloaded: {list_datasets()}"
        )
    return d


def load_metadata(name: str) -> Dict[str, Any]:
    path = dataset_dir(name) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_arrays(name: str) -> List[str]:
    """Paths (relative to the dataset dir) of all .npy arrays, excluding raw/."""
    d = dataset_dir(name)
    arrays = []
    for p in sorted(d.rglob("*.npy")):
        rel = p.relative_to(d)
        if "raw" in rel.parts:
            continue
        arrays.append(str(rel))
    return arrays


def load_array(name: str, filename: str, mmap: Optional[str] = None) -> np.ndarray:
    """Load any .npy file inside a dataset dir, e.g. filename='labels/louvain_labels_int.npy'."""
    path = dataset_dir(name) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Available arrays for '{name}': {list_arrays(name)}"
        )
    return np.load(path, mmap_mode=mmap, allow_pickle=True)


def load_embeddings(name: str, mmap: Optional[str] = "r") -> np.ndarray:
    """Load the (N, f) embeddings array for a dataset. Memory-mapped by default."""
    path = dataset_dir(name) / "embeddings.npy"
    if not path.is_file():
        meta = load_metadata(name)
        status = meta.get("status", "unknown")
        raise FileNotFoundError(
            f"{path} does not exist (dataset status: {status})."
            f"{_download_hint(name)}"
        )
    return np.load(path, mmap_mode=mmap)
