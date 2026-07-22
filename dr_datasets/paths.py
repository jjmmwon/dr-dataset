"""Shared resolution for where dataset files live on disk.

Used by both `dr_datasets.loader` (reading) and `scripts/download_*.py` (writing),
so both agree on the same directory without duplicating the env-var logic.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_HOME = Path("/root/dataset")


def data_home() -> Path:
    """Root directory holding `<name>/embeddings.npy` etc. for every dataset.

    Defaults to `/root/dataset`; override with the `DR_DATA_HOME` env var for
    non-root users or non-default layouts.
    """
    return Path(os.environ.get("DR_DATA_HOME", DEFAULT_DATA_HOME))
