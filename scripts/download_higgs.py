#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UCI HIGGS -> dr-datasets convention (embeddings.npy + labels.npy + metadata.json)

Downloads HIGGS.csv.gz (28 kinematic features + signal/background label, ~11M rows total)
distributed by UCI, randomly samples --n-samples rows, and saves in dr-datasets convention.
The sampled result is also cached to <output-dir>/features_<n>.parquet so the full CSV isn't
re-parsed on every run.

Requires: pip install "dr-datasets[download]" (or just requests pandas pyarrow tqdm)

Example:
python scripts/download_higgs.py
# Default save location: $DR_DATA_HOME/higgs (/root/dataset/higgs if DR_DATA_HOME is unset)

Output files (<output-dir> = --output-dir, see default above):
- <output-dir>/embeddings.npy        : (n_samples, 28) float32
- <output-dir>/labels.npy             : (n_samples,) int32, 1=signal 0=background
- <output-dir>/features_<n>.parquet   : cached sampled feature+label data (reused on re-runs)
- <output-dir>/metadata.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dr_datasets.paths import data_home

SOURCE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

COL_NAMES = [
    "label",
    "lepton_pT", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pT", "jet1_eta", "jet1_phi", "jet1_b_tag",
    "jet2_pT", "jet2_eta", "jet2_phi", "jet2_b_tag",
    "jet3_pT", "jet3_eta", "jet3_phi", "jet3_b_tag",
    "jet4_pT", "jet4_eta", "jet4_phi", "jet4_b_tag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb",
]


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
    parser.add_argument("--output-dir", type=str, default=str(data_home() / "higgs"))
    parser.add_argument(
        "--raw-cache-dir",
        type=str,
        default=None,
        help="Downloaded raw csv.gz save location. Default: <output-dir>/raw",
    )
    parser.add_argument("--source-url", type=str, default=SOURCE_URL)
    parser.add_argument("--n-samples", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir = Path(args.raw_cache_dir) if args.raw_cache_dir else out_dir / "raw"
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_cache_dir / "HIGGS.csv.gz"
    parquet_path = out_dir / f"features_{args.n_samples}.parquet"

    t0 = time.time()
    if parquet_path.exists():
        print(f"[1/3] cache hit: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        print(f"[1/3] downloading {args.source_url} -> {csv_path}")
        download(args.source_url, csv_path)

        print("      loading full CSV (~11M rows) and sampling")
        df_full = pd.read_csv(
            csv_path,
            header=None,
            names=COL_NAMES,
            dtype={"label": "int8", **{c: "float32" for c in COL_NAMES[1:]}},
            compression="gzip",
        )
        df = df_full.sample(n=args.n_samples, random_state=args.seed).reset_index(drop=True)
        df.to_parquet(parquet_path, index=False)

    print(f"[2/3] sample shape = {df.shape}, signal={int((df.label == 1).sum()):,}, "
          f"background={int((df.label == 0).sum()):,}")

    X = df.drop(columns="label")
    y = df["label"]
    embeddings = X.to_numpy(dtype="float32")
    labels = y.to_numpy(dtype="int32")

    print("[3/3] saving outputs")
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "labels.npy", labels)

    metadata = {
        "name": "higgs",
        "description": "UCI HIGGS particle physics dataset (28 kinematic features + signal/background label).",
        "source_url": args.source_url,
        "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "dtype": "float32",
        "feature_names": list(X.columns),
        "num_classes": 2,
        "label_values": "1=signal, 0=background",
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "status": "ready",
        "build_script": "scripts/download_higgs.py",
        "elapsed_sec": time.time() - t0,
        "output_files": {
            "embeddings_npy": str(out_dir / "embeddings.npy"),
            "labels_npy": str(out_dir / "labels.npy"),
            "features_parquet": str(parquet_path),
        },
        "raw": {
            "description": "Downloaded HIGGS csv.gz, cached under raw/",
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
