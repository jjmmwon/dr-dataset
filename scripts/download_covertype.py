#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UCI Covertype -> dr-datasets convention (embeddings.npy + labels.npy + metadata.json)

Fetched via sklearn.datasets.fetch_covtype() (downloads from sklearn's built-in official
mirror). All 54 features (10 continuous + 4 one-hot Wilderness_Area + 40 one-hot Soil_Type)
are already numeric, so they're saved as-is as float32 embeddings, no extra encoding needed.
The target is forest cover type, 7 classes (1-7).

Requires: pip install "dr-datasets[download]" (or just scikit-learn)

Example:
python scripts/download_covertype.py
# Default save location: $DR_DATA_HOME/covertype (/root/dataset/covertype if DR_DATA_HOME is unset)

Output files (<output-dir> = --output-dir, see default above):
- <output-dir>/embeddings.npy   : (581012, 54) float32
- <output-dir>/labels.npy        : (581012,) int32, values 1-7 (forest cover type)
- <output-dir>/metadata.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dr_datasets.paths import data_home


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(data_home() / "covertype"))
    parser.add_argument(
        "--raw-cache-dir",
        type=str,
        default=None,
        help="sklearn download cache location. Default: <output-dir>/raw",
    )
    args = parser.parse_args()

    from sklearn.datasets import fetch_covtype

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir = Path(args.raw_cache_dir) if args.raw_cache_dir else out_dir / "raw"
    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[1/3] downloading covtype via sklearn.datasets.fetch_covtype (cache: {raw_cache_dir}) ...")
    bunch = fetch_covtype(data_home=str(raw_cache_dir), download_if_missing=True)

    print("[2/3] casting to float32/int32")
    embeddings = np.asarray(bunch.data, dtype=np.float32)
    labels = np.asarray(bunch.target, dtype=np.int32)

    print("[3/3] saving outputs")
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "labels.npy", labels)

    metadata = {
        "name": "covertype",
        "description": (
            "UCI Covertype: 30x30m US forest patches, 54 features "
            "(10 continuous + 4 one-hot Wilderness_Area + 40 one-hot Soil_Type), "
            "target = 1 of 7 forest cover types. Classic UMAP/t-SNE benchmark."
        ),
        "source": "sklearn.datasets.fetch_covtype",
        "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "dtype": "float32",
        "num_classes": 7,
        "label_values": "1-7 (forest cover type, see UCI Covertype documentation)",
        "status": "ready",
        "build_script": "scripts/download_covertype.py",
        "elapsed_sec": time.time() - t0,
        "output_files": {
            "embeddings_npy": str(out_dir / "embeddings.npy"),
            "labels_npy": str(out_dir / "labels.npy"),
        },
        "raw": {
            "description": "sklearn download cache under raw/",
            "cache_dir": str(raw_cache_dir),
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("done.")
    print(f"embeddings shape = {embeddings.shape}")
    print(f"elapsed = {time.time() - t0:.2f} sec")


if __name__ == "__main__":
    main()
