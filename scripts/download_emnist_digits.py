#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMNIST digits (torchvision) -> dr-datasets convention (embeddings.npy + labels.npy + metadata.json)

Fetches train(240,000) + test(40,000) = 280,000 28x28 grayscale digit (0-9) images via
torchvision.datasets.EMNIST(split="digits"). No embedding model — same approach as
covertype/sift1m, using raw pixels directly as the feature vector (28*28=784 dims,
[0,1]-normalized). Flattening pixels to (N, 784) is a fixed permutation applied identically
to every row, so it preserves pairwise distances — no embedding model is needed for
DR (UMAP/t-SNE) benchmarking purposes.

Requires: pip install "dr-datasets[download]" (or just torch torchvision)

Example:
python scripts/download_emnist_digits.py
# Default save location: $DR_DATA_HOME/emnist_digits (/root/dataset/emnist_digits if DR_DATA_HOME is unset)

Output files (<output-dir> = --output-dir, see default above):
- <output-dir>/embeddings.npy : (280000, 784) float32, [0,1]-normalized flattened pixels
- <output-dir>/labels.npy      : (280000,) int32, values 0-9
- <output-dir>/is_train.npy    : (280000,) bool, True=train (first 240,000 rows), False=test (last 40,000 rows)
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
    parser.add_argument("--output-dir", type=str, default=str(data_home() / "emnist_digits"))
    parser.add_argument(
        "--raw-cache-dir",
        type=str,
        default=None,
        help="torchvision download cache location. Default: <output-dir>/raw",
    )
    args = parser.parse_args()

    from torchvision.datasets import EMNIST

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir = Path(args.raw_cache_dir) if args.raw_cache_dir else out_dir / "raw"
    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[1/3] downloading EMNIST(split=digits) via torchvision (cache: {raw_cache_dir}) ...")
    train_ds = EMNIST(root=str(raw_cache_dir), split="digits", train=True, download=True)
    test_ds = EMNIST(root=str(raw_cache_dir), split="digits", train=False, download=True)

    print("[2/3] flattening + concatenating train/test")
    train_images = train_ds.data.numpy()  # (240000, 28, 28) uint8
    test_images = test_ds.data.numpy()  # (40000, 28, 28) uint8
    train_labels = train_ds.targets.numpy()  # (240000,) int64, 0-9
    test_labels = test_ds.targets.numpy()  # (40000,) int64, 0-9

    images = np.concatenate([train_images, test_images], axis=0)
    embeddings = images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
    labels = np.concatenate([train_labels, test_labels], axis=0).astype(np.int32)
    is_train = np.concatenate(
        [np.ones(len(train_images), dtype=bool), np.zeros(len(test_images), dtype=bool)]
    )

    print("[3/3] saving outputs")
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "is_train.npy", is_train)

    metadata = {
        "name": "emnist_digits",
        "description": (
            "EMNIST digits split (torchvision): 28x28 grayscale handwritten digit images "
            "(0-9), train(240,000) + test(40,000) concatenated. Flattened [0,1]-normalized "
            "pixels used directly as the feature vector, no embedding model."
        ),
        "source": "torchvision.datasets.EMNIST(split='digits')",
        "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "dtype": "float32",
        "num_classes": 10,
        "label_values": "0-9 (digit)",
        "status": "ready",
        "build_script": "scripts/download_emnist_digits.py",
        "elapsed_sec": time.time() - t0,
        "output_files": {
            "embeddings_npy": str(out_dir / "embeddings.npy"),
            "labels_npy": str(out_dir / "labels.npy"),
            "is_train_npy": str(out_dir / "is_train.npy"),
        },
        "notes": (
            "Row order is train rows first (240,000), then test rows (40,000) — see "
            "is_train.npy for the boolean split mask, row-aligned with embeddings.npy/labels.npy. "
            "Pixels are the original EMNIST orientation (transposed/rotated relative to canonical "
            "digit images, a known torchvision/EMNIST quirk); left uncorrected since flattening is "
            "a fixed per-row permutation and does not affect pairwise distances."
        ),
        "raw": {
            "description": "torchvision download cache (raw EMNIST .gz/idx files), cached under raw/",
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
