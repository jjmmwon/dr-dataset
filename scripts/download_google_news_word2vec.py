#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Google News word2vec (300d, ~3M words) -> dr-datasets convention (embeddings.npy + words.txt + metadata.json)

Fetches the pretrained word2vec-google-news-300 model via gensim.downloader, loads it as
KeyedVectors, then saves it in dr-datasets convention. gensim-data's download cache (including
the original compressed file) is kept under --raw-cache-dir (default: <output-dir>/raw/gensim-data).

Requires: pip install "dr-datasets[download]" (or just gensim)

Example:
python scripts/download_google_news_word2vec.py
# Default save location: $DR_DATA_HOME/google_news_word2vec300 (/root/dataset/google_news_word2vec300 if DR_DATA_HOME is unset)

Output files (<output-dir> = --output-dir, see default above):
- <output-dir>/embeddings.npy   : (3000000, 300) float32
- <output-dir>/words.txt          : one token per line, row-aligned with embeddings.npy
- <output-dir>/metadata.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dr_datasets.paths import data_home


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=str, default=str(data_home() / "google_news_word2vec300")
    )
    parser.add_argument(
        "--raw-cache-dir",
        type=str,
        default=None,
        help="gensim-data download cache location. Default: <output-dir>/raw/gensim-data",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_cache_dir = (
        Path(args.raw_cache_dir) if args.raw_cache_dir else out_dir / "raw" / "gensim-data"
    )
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    # gensim.downloader reads GENSIM_DATA_DIR as a module-level constant at import
    # time, so the env var must be set BEFORE importing it.
    os.environ["GENSIM_DATA_DIR"] = str(raw_cache_dir)

    import gensim.downloader as api

    t0 = time.time()
    print(f"[1/3] downloading/loading word2vec-google-news-300 via gensim.downloader "
          f"(cache: {raw_cache_dir}) ...")
    kv = api.load("word2vec-google-news-300")

    print("[2/3] extracting vectors + vocabulary")
    vectors = np.asarray(kv.vectors, dtype=np.float32)
    words = list(kv.index_to_key)
    if len(words) != vectors.shape[0]:
        raise RuntimeError(
            f"word count ({len(words)}) != vector row count ({vectors.shape[0]})"
        )

    print("[3/3] saving outputs")
    np.save(out_dir / "embeddings.npy", vectors)
    with open(out_dir / "words.txt", "w", encoding="utf-8") as f:
        for w in words:
            f.write(w + "\n")

    metadata = {
        "name": "google_news_word2vec300",
        "description": "Pretrained Google News word2vec vectors (300d, ~3M words).",
        "source": "gensim.downloader dataset 'word2vec-google-news-300'",
        "embedding_shape": [int(vectors.shape[0]), int(vectors.shape[1])],
        "dtype": "float32",
        "status": "ready",
        "build_script": "scripts/download_google_news_word2vec.py",
        "elapsed_sec": time.time() - t0,
        "output_files": {
            "embeddings_npy": str(out_dir / "embeddings.npy"),
            "words_txt": str(out_dir / "words.txt"),
        },
        "notes": "words.txt has one token per line, row-aligned with embeddings.npy (line i <-> embeddings.npy[i]).",
        "raw": {
            "description": "gensim-data download cache under raw/",
            "cache_dir": str(raw_cache_dir),
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("done.")
    print(f"embeddings shape = {vectors.shape}")
    print(f"elapsed = {time.time() - t0:.2f} sec")


if __name__ == "__main__":
    main()
