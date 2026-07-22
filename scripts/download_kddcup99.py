#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KDD Cup 1999 (full, ~4.9M rows) -> dr-datasets convention (embeddings.npy + labels + metadata.json)

Fetched via sklearn.datasets.fetch_kddcup99(percent10=False). Of the original 41 features, 3
(protocol_type, service, flag) are categorical and get one-hot encoded; the remaining 38 are
cast to float and concatenated as-is. The target is saved two ways: 23 fine-grained attack-type
strings -> integer-encoded (legend.json), plus a normal/attack binary label.

Memory note: the first implementation used pandas.get_dummies(), which duplicated the whole
source array and hit a peak RSS of 124GB, getting OOM-killed on a 125GB RAM machine. It was
rewritten without pandas, using only numpy/sklearn.preprocessing.OneHotEncoder — narrowing the
source object array down to float32/one-hot column-by-column as needed, rather than all at
once (peak stays around the size of the raw object array itself, a few GB to a few dozen GB,
exactly what sklearn returns).

Requires: pip install "dr-datasets[download]" (or just scikit-learn)

Example:
python scripts/download_kddcup99.py
# Default save location: $DR_DATA_HOME/kddcup99 (/root/dataset/kddcup99 if DR_DATA_HOME is unset)

Output files (<output-dir> = --output-dir, see default above):
- <output-dir>/embeddings.npy               : (4898431, f) float32, f = 38 + one-hot(protocol_type,service,flag)
- <output-dir>/labels/attack_type_int.npy    : (4898431,) int32
- <output-dir>/labels/attack_type_legend.json
- <output-dir>/labels/binary.npy             : (4898431,) int32, 0=normal 1=attack
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


def decode_col(col: np.ndarray) -> np.ndarray:
    return np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in col],
        dtype=object,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(data_home() / "kddcup99"))
    parser.add_argument(
        "--raw-cache-dir",
        type=str,
        default=None,
        help="sklearn download cache location. Default: <output-dir>/raw",
    )
    parser.add_argument(
        "--percent10",
        action="store_true",
        help="Fetch only the 10%% subset (494,021 rows) instead of the full 4,898,431 rows.",
    )
    args = parser.parse_args()

    from sklearn.datasets import fetch_kddcup99
    from sklearn.preprocessing import OneHotEncoder

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir = Path(args.raw_cache_dir) if args.raw_cache_dir else out_dir / "raw"
    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(
        f"[1/6] downloading kddcup99 (percent10={args.percent10}) via sklearn.datasets.fetch_kddcup99 "
        f"(cache: {raw_cache_dir}) ...",
        flush=True,
    )
    bunch = fetch_kddcup99(
        percent10=args.percent10, data_home=str(raw_cache_dir), download_if_missing=True, as_frame=False
    )
    raw = bunch.data  # (N, 41) object ndarray
    feature_names = list(bunch.feature_names)
    print(f"      raw shape = {raw.shape}", flush=True)

    print("[2/6] splitting numeric vs categorical columns", flush=True)
    is_cat = [isinstance(raw[0, j], (bytes, bytearray)) for j in range(len(feature_names))]
    cat_idx = [j for j, c in enumerate(is_cat) if c]
    num_idx = [j for j, c in enumerate(is_cat) if not c]
    obj_cols = [feature_names[j] for j in cat_idx]
    numeric_cols = [feature_names[j] for j in num_idx]
    print(f"      categorical columns (one-hot): {obj_cols}", flush=True)
    print(f"      numeric columns kept as-is: {len(numeric_cols)}", flush=True)

    print("[3/6] casting numeric columns to float32", flush=True)
    numeric = np.empty((raw.shape[0], len(num_idx)), dtype=np.float32)
    for out_j, j in enumerate(num_idx):
        numeric[:, out_j] = raw[:, j].astype(np.float32)

    print("[4/6] one-hot encoding categorical columns", flush=True)
    cat_block = np.stack([decode_col(raw[:, j]) for j in cat_idx], axis=1)  # (N, 3) str
    encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
    cat_onehot = encoder.fit_transform(cat_block)
    del cat_block, raw  # free the big object array before the final concat/allocation

    embeddings = np.hstack([numeric, cat_onehot])
    del numeric, cat_onehot

    print("[5/6] encoding labels (fine-grained attack type + binary normal/attack)", flush=True)
    target = decode_col(bunch.target)
    attack_types, attack_type_int = np.unique(target, return_inverse=True)
    attack_type_int = attack_type_int.astype(np.int32)
    legend = {str(i): label for i, label in enumerate(attack_types.tolist())}
    binary = (target != "normal.").astype(np.int32)

    print("[6/6] saving outputs", flush=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(labels_dir / "attack_type_int.npy", attack_type_int)
    np.save(labels_dir / "binary.npy", binary)
    with open(labels_dir / "attack_type_legend.json", "w", encoding="utf-8") as f:
        json.dump(legend, f, ensure_ascii=False, indent=2)

    metadata = {
        "name": "kddcup99",
        "description": (
            "KDD Cup 1999 network intrusion detection. Original 41 features "
            "(3 categorical one-hot encoded + 38 numeric), 23 fine-grained attack types "
            "plus a normal/attack binary label. Discrete, well-separated clusters -- "
            "classic clustering/anomaly-detection DR benchmark."
        ),
        "source": f"sklearn.datasets.fetch_kddcup99(percent10={args.percent10})",
        "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "dtype": "float32",
        "original_numeric_columns": numeric_cols,
        "one_hot_encoded_columns": obj_cols,
        "num_attack_types": int(len(attack_types)),
        "status": "ready",
        "build_script": "scripts/download_kddcup99.py",
        "elapsed_sec": time.time() - t0,
        "output_files": {
            "embeddings_npy": str(out_dir / "embeddings.npy"),
            "attack_type_int_npy": str(labels_dir / "attack_type_int.npy"),
            "attack_type_legend_json": str(labels_dir / "attack_type_legend.json"),
            "binary_npy": str(labels_dir / "binary.npy"),
        },
        "labels": {
            "attack_type": {
                "description": "23-class fine-grained attack type, integer-encoded (see legend.json)",
                "labels_int_npy": "labels/attack_type_int.npy",
                "legend_json": "labels/attack_type_legend.json",
            },
            "binary": {
                "description": "0=normal, 1=attack",
                "labels_int_npy": "labels/binary.npy",
            },
        },
        "raw": {
            "description": "sklearn download cache under raw/",
            "cache_dir": str(raw_cache_dir),
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("done.", flush=True)
    print(f"embeddings shape = {embeddings.shape}", flush=True)
    print(f"num attack types = {len(attack_types)}", flush=True)
    print(f"elapsed = {time.time() - t0:.2f} sec", flush=True)


if __name__ == "__main__":
    main()
