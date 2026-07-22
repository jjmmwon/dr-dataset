#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SIFT1M -> dr-datasets convention (embeddings.npy + metadata.json)

Fetches the HDF5 file (sift-128-euclidean.hdf5) distributed by the ann-benchmarks project.
It already contains train(1,000,000 x 128), test(10,000 x 128) SIFT descriptor vectors, and
ground-truth top-100 neighbors/distances for the test queries. train becomes dr-datasets'
embeddings.npy; the rest are saved as reference extra files.

Requires: pip install "dr-datasets[download]" (or just h5py requests tqdm)

Example:
python scripts/download_sift1m.py
# Default save location: $DR_DATA_HOME/sift1m (/root/dataset/sift1m if DR_DATA_HOME is unset)

Output files (<output-dir> = --output-dir, see default above):
- <output-dir>/embeddings.npy   : (1000000, 128) float32  (train)
- <output-dir>/test.npy          : (10000, 128) float32    (query vectors)
- <output-dir>/neighbors.npy     : (10000, 100) int32       (ground-truth top-100 neighbor idx into embeddings.npy)
- <output-dir>/distances.npy     : (10000, 100) float32     (distance to each of those neighbors)
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

SOURCE_URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"


def download(url: str, dst: Path) -> None:
    import requests
    from tqdm import tqdm

    if dst.exists() and dst.stat().st_size > 0:
        print(f"      cache hit: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.rename(dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(data_home() / "sift1m"))
    parser.add_argument(
        "--raw-cache-dir",
        type=str,
        default=None,
        help="Downloaded raw hdf5 save location. Default: <output-dir>/raw",
    )
    parser.add_argument("--source-url", type=str, default=SOURCE_URL)
    args = parser.parse_args()

    import h5py

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir = Path(args.raw_cache_dir) if args.raw_cache_dir else out_dir / "raw"
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = raw_cache_dir / "sift-128-euclidean.hdf5"

    t0 = time.time()
    print(f"[1/3] downloading {args.source_url} -> {hdf5_path}")
    download(args.source_url, hdf5_path)

    print("[2/3] reading arrays from hdf5")
    with h5py.File(hdf5_path, "r") as f:
        print(f"      hdf5 keys: {list(f.keys())}")
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int32)
        distances = np.asarray(f["distances"], dtype=np.float32)

    print("[3/3] saving outputs")
    np.save(out_dir / "embeddings.npy", train)
    np.save(out_dir / "test.npy", test)
    np.save(out_dir / "neighbors.npy", neighbors)
    np.save(out_dir / "distances.npy", distances)

    metadata = {
        "name": "sift1m",
        "description": (
            "SIFT1M: 1M 128-d SIFT image descriptors, the standard ANN/manifold benchmark "
            "used throughout the ANN-search and DR literature."
        ),
        "source_url": args.source_url,
        "embedding_shape": [int(train.shape[0]), int(train.shape[1])],
        "dtype": "float32",
        "status": "ready",
        "build_script": "scripts/download_sift1m.py",
        "elapsed_sec": time.time() - t0,
        "output_files": {
            "embeddings_npy": str(out_dir / "embeddings.npy"),
            "test_npy": str(out_dir / "test.npy"),
            "neighbors_npy": str(out_dir / "neighbors.npy"),
            "distances_npy": str(out_dir / "distances.npy"),
        },
        "notes": (
            "test.npy are held-out query vectors (not part of embeddings.npy). "
            "neighbors.npy[i] holds the indices (into embeddings.npy) of the true top-100 "
            "nearest neighbors of test.npy[i], with distances.npy[i] the corresponding distances. "
            "Shipped directly by the ann-benchmarks hdf5 file, not gpumap-derived."
        ),
        "raw": {
            "description": "Downloaded hdf5 source file, cached under raw/",
            "cache_dir": str(raw_cache_dir),
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("done.")
    print(f"embeddings shape = {train.shape}")
    print(f"elapsed = {time.time() - t0:.2f} sec")


if __name__ == "__main__":
    main()
