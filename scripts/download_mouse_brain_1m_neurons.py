#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10x Genomics "1.3 Million Brain Cells from E18 Mice" -> dr-datasets convention.

Unlike the other download_*.py scripts, this dataset is split into 3 steps, one of which
needs a GPU:

  1. Download the raw H5 (this script does this automatically, CPU only)
  2. HVG(5000) + PCA(50) preprocessing (calls scripts/build_mouse_brain_pca.py —
     needs a CUDA GPU + cupy + rapids_singlecell. Fails at this step without a GPU.)
  3. (optional) Align Louvain cluster labels (calls scripts/build_mouse_brain_louvain_labels.py)
     — the original CSV for these labels is derived from scanpy's "1.3M neurons" tutorial, and
     dr-datasets doesn't have a confirmed public URL it can fetch automatically. If you need the
     labels, prepare a barcode,label CSV yourself at `<output-dir>/raw/louvain.csv.gz` and step 3
     runs; otherwise it's skipped automatically (embeddings.npy/barcodes.npy etc. are still
     produced normally).

Requires: pip install "dr-datasets[download]" + a CUDA GPU/RAPIDS environment (step 2 only).
Without a GPU, you can still fetch just the h5 path via --input and run
scripts/build_mouse_brain_pca.py directly on a separate machine that has RAPIDS.

Example:
python scripts/download_mouse_brain_1m_neurons.py
# Default save location: $DR_DATA_HOME/mouse_brain_1m_neurons (/root/dataset/... if DR_DATA_HOME is unset)

Output files (<output-dir> = --output-dir):
- <output-dir>/embeddings.npy, barcodes.npy, hvg_gene_names.npy,
  pca_variance_ratio.npy, pca_loadings.npy, metadata.json  (step 2 output)
- <output-dir>/labels/louvain_labels_int.npy, labels/louvain_legend.json  (step 3, only if the CSV was provided)
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dr_datasets.paths import data_home

RAW_H5_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/"
    "1M_neurons_filtered_gene_bc_matrices_h5.h5"
)
SCRIPTS_DIR = Path(__file__).resolve().parent


def download(url: str, dst: Path) -> None:
    import requests
    from tqdm import tqdm

    if dst.exists() and dst.stat().st_size > 0:
        print(f"      cache hit: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=1200) as resp:
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
    parser.add_argument(
        "--output-dir", type=str, default=str(data_home() / "mouse_brain_1m_neurons")
    )
    parser.add_argument("--raw-h5-url", type=str, default=RAW_H5_URL)
    parser.add_argument("--n-top-hvg", type=int, default=5000)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Only download the raw H5; skip the GPU-only PCA step.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    h5_path = raw_dir / "1M_neurons_filtered_gene_bc_matrices_h5.h5"
    louvain_path = raw_dir / "louvain.csv.gz"

    print(f"[1/3] downloading raw H5 from {args.raw_h5_url}")
    print("      (~1.4GB; if this 10x Genomics URL has moved, check "
          "https://www.10xgenomics.com/datasets for '1.3 Million Brain Cells from E18 Mice' "
          "and pass --raw-h5-url)")
    download(args.raw_h5_url, h5_path)

    if args.skip_pca:
        print("[2/3] --skip-pca set, not running build_mouse_brain_pca.py")
        return

    print("[2/3] running build_mouse_brain_pca.py (requires CUDA GPU + rapids_singlecell)")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "build_mouse_brain_pca.py"),
            "--input", str(h5_path),
            "--output-dir", str(out_dir),
            "--n-top-hvg", str(args.n_top_hvg),
            "--n-pcs", str(args.n_pcs),
        ],
        check=True,
    )

    if not louvain_path.is_file():
        print(f"[3/3] skipping Louvain labels: {louvain_path} not found.")
        print(
            "      dr-datasets doesn't have a confirmed public URL for the Louvain CSV used "
            "originally (derived from scanpy's '1.3M neurons' tutorial). If you have it, "
            f"place it at {louvain_path} (barcode,label CSV, gzip-compressed) and re-run "
            "this script, or run scripts/build_mouse_brain_louvain_labels.py directly."
        )
        return

    print("[3/3] running build_mouse_brain_louvain_labels.py")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "build_mouse_brain_louvain_labels.py"),
            "--h5", str(h5_path),
            "--louvain", str(louvain_path),
            "--out-dir", str(out_dir / "labels"),
        ],
        check=True,
    )

    print("done.")


if __name__ == "__main__":
    main()
