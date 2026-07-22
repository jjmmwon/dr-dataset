#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stanford GloVe 840B.300d -> dr-datasets convention (embeddings.npy + words.txt + metadata.json)

Downloads glove.840B.300d.zip (~2.03GB) distributed by Stanford NLP, extracts it, and parses
the space-separated text format (word + 300 floats) into embeddings.npy / words.txt.

Requires: pip install "dr-datasets[download]" (or just requests tqdm)

Example:
python scripts/download_glove.py
# Default save location: $DR_DATA_HOME/glove_840b_300d (/root/dataset/glove_840b_300d if DR_DATA_HOME is unset)

Output files (<output-dir> = --output-dir, see default above):
- <output-dir>/embeddings.npy   : (2196017, 300) float32
- <output-dir>/words.txt         : one token per line, row-aligned with embeddings.npy
- <output-dir>/metadata.json
"""

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dr_datasets.paths import data_home

SOURCE_URL = "http://nlp.stanford.edu/data/glove.840B.300d.zip"


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
    parser.add_argument("--output-dir", type=str, default=str(data_home() / "glove_840b_300d"))
    parser.add_argument(
        "--raw-cache-dir",
        type=str,
        default=None,
        help="Downloaded/extracted raw file save location. Default: <output-dir>/raw",
    )
    parser.add_argument("--source-url", type=str, default=SOURCE_URL)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir = Path(args.raw_cache_dir) if args.raw_cache_dir else out_dir / "raw"
    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_cache_dir / "glove.840B.300d.zip"
    txt_path = raw_cache_dir / "glove.840B.300d.txt"

    t0 = time.time()
    print(f"[1/4] downloading {args.source_url} -> {zip_path} (~2.03GB)")
    download(args.source_url, zip_path)

    if not txt_path.exists():
        print(f"[2/4] extracting {zip_path} -> {raw_cache_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_cache_dir)
    else:
        print(f"[2/4] cache hit: {txt_path}")

    print("[3/4] parsing text file (~2.2M lines)")
    words = []
    vectors = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.rstrip().split(" ")
            vector = np.asarray(vals[-300:], dtype="float32")
            word = " ".join(vals[:-300])
            words.append(word)
            vectors.append(vector)
    embeddings = np.asarray(vectors, dtype="float32")

    print("[4/4] saving outputs")
    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "words.txt", "w", encoding="utf-8") as f:
        for word in words:
            f.write(word + "\n")

    metadata = {
        "name": "glove_840b_300d",
        "description": "Stanford GloVe 840B token, 300d word vectors.",
        "source_url": args.source_url,
        "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "dtype": "float32",
        "status": "ready",
        "build_script": "scripts/download_glove.py",
        "elapsed_sec": time.time() - t0,
        "output_files": {
            "embeddings_npy": str(out_dir / "embeddings.npy"),
            "words_txt": str(out_dir / "words.txt"),
        },
        "notes": "words.txt has one token per line, row-aligned with embeddings.npy (line i <-> embeddings.npy[i]).",
        "raw": {
            "description": "Downloaded zip + extracted txt, cached under raw/",
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
