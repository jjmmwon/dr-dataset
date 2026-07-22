#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Converts a Louvain clustering CSV into an integer label array aligned to 10x H5 barcode order.

Normally not run directly — invoked via scripts/download_mouse_brain_1m_neurons.py
(see that script's notes for why you need to obtain the --louvain CSV yourself).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dr_datasets.paths import data_home


def decode_array(arr):
    out = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)


def looks_like_10x_barcode(x: str) -> bool:
    x = str(x).strip().replace("\ufeff", "").upper()
    return re.fullmatch(r"[ACGTN]+(?:-\d+)?", x) is not None


def looks_like_html(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(4096).lstrip()

    lower = head.lower()
    return (
        head.startswith(b"<!DOCTYPE html")
        or head.startswith(b"<html")
        or (b"<html" in lower and b"github" in lower)
    )


def infer_legacy_genome_group(h5_path: str, genome: str | None) -> str:
    if genome is not None:
        return genome

    with h5py.File(h5_path, "r") as f:
        groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        if len(groups) != 1:
            raise ValueError(
                f"Could not infer genome group automatically. Found groups: {groups}. "
                f"Please pass --genome explicitly."
            )
        return groups[0]


def read_10x_barcodes(h5_path: str, genome: str | None = None) -> np.ndarray:
    """
    Supports:
      - legacy format: /<genome>/barcodes
      - new format:    /matrix/barcodes
    """
    with h5py.File(h5_path, "r") as f:
        if "matrix" in f:
            g = f["matrix"]
        else:
            genome = infer_legacy_genome_group(h5_path, genome)
            g = f[genome]

        if "barcodes" not in g:
            raise KeyError("Could not find 'barcodes' in the 10x H5 file.")

        return decode_array(g["barcodes"][:])


def read_louvain_csv(label_path: str) -> pd.Series:
    """
    Robust parser for scanpy_usage louvain.csv.gz.

    Supports:
      1) headerless CSV
         barcode,label
      2) headered CSV
         barcode,louvain
      3) pandas index CSV
         ,louvain
         barcode,label

    Returns:
      pd.Series
        index  -> barcode
        values -> original Louvain label as string
    """
    if looks_like_html(label_path):
        raise ValueError(
            f"{label_path} looks like an HTML page, not the raw clustering file. "
            f"Download the GitHub Raw file instead."
        )

    df = pd.read_csv(
        label_path,
        compression="infer",
        header=None,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    if df.shape[1] < 2:
        raise ValueError(
            f"Could not parse {label_path}. Expected at least 2 columns, got {df.shape[1]}."
        )

    # Drop header row if the first value does not look like a 10x barcode.
    first_val = str(df.iloc[0, 0]).strip().replace("\ufeff", "")
    if not looks_like_10x_barcode(first_val):
        df = df.iloc[1:].reset_index(drop=True)

    barcodes = df.iloc[:, 0].astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    labels = df.iloc[:, 1].astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    valid = (barcodes != "") & (labels != "")
    barcodes = barcodes[valid]
    labels = labels[valid]

    dup = barcodes.duplicated(keep="first")
    n_dup = int(dup.sum())
    if n_dup > 0:
        print(f"      Warning: {n_dup} duplicate barcodes in Louvain CSV; keeping first occurrence.")
        barcodes = barcodes[~dup]
        labels = labels[~dup]

    return pd.Series(labels.to_numpy(dtype=object), index=barcodes.to_numpy(dtype=object), dtype=object)


def natural_sort_labels(labels: list[str]) -> list[str]:
    def is_int_like(x: str) -> bool:
        return re.fullmatch(r"-?\d+", x) is not None

    if all(is_int_like(x) for x in labels):
        return sorted(labels, key=lambda x: int(x))
    return sorted(labels)


def align_labels_exact(
    h5_barcodes: np.ndarray,
    label_series: pd.Series,
    missing_label: str = "__MISSING__",
) -> tuple[np.ndarray, int]:
    """
    Exact barcode alignment only.
    Do not strip suffixes, because this dataset can contain barcode collisions across channels.
    """
    if label_series.index.duplicated().any():
        n_dup = int(label_series.index.duplicated().sum())
        raise ValueError(f"Louvain label index has {n_dup} duplicate barcodes.")

    aligned = label_series.reindex(h5_barcodes)
    missing_mask = aligned.isna().to_numpy()
    n_missing = int(missing_mask.sum())

    aligned = aligned.astype(object).to_numpy()

    if n_missing > 0:
        examples = h5_barcodes[missing_mask][:10].tolist()
        print(f"      Warning: {n_missing} H5 barcodes have no matched label.")
        print(f"      Examples: {examples}")
        aligned[missing_mask] = missing_label

    return aligned.astype(object), n_missing


def encode_labels(labels: np.ndarray, missing_label: str = "__MISSING__") -> tuple[np.ndarray, dict]:
    unique_raw = pd.unique(labels).tolist()
    has_missing = missing_label in unique_raw

    ordered = [x for x in unique_raw if x != missing_label]
    ordered = natural_sort_labels([str(x) for x in ordered])

    if has_missing:
        ordered.append(missing_label)

    label_to_int = {lab: i for i, lab in enumerate(ordered)}

    labels_int = np.fromiter(
        (label_to_int[str(x)] for x in labels),
        dtype=np.int32,
        count=len(labels),
    )

    counts = np.bincount(labels_int, minlength=len(ordered))

    legend = {
        str(i): {
            "original_label": ordered[i],
            "count": int(counts[i]),
        }
        for i in range(len(ordered))
    }

    return labels_int, legend


def main():
    parser = argparse.ArgumentParser(
        description="Convert scanpy Louvain CSV into integer label array aligned to 10x H5 barcode order."
    )
    parser.add_argument(
        "--h5",
        default=str(
            data_home() / "mouse_brain_1m_neurons" / "raw" / "1M_neurons_filtered_gene_bc_matrices_h5.h5"
        ),
        help="10x H5 file path",
    )
    parser.add_argument(
        "--louvain",
        default=str(data_home() / "mouse_brain_1m_neurons" / "raw" / "louvain.csv.gz"),
        help="Path to louvain.csv.gz",
    )
    parser.add_argument(
        "--out-dir",
        default=str(data_home() / "mouse_brain_1m_neurons" / "labels"),
        help="Output directory",
    )
    parser.add_argument("--genome", default=None, help="Legacy 10x genome group, e.g. mm10")
    parser.add_argument(
        "--missing-label",
        default="__MISSING__",
        help="Placeholder label used when a barcode has no matched label",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Reading barcodes from 10x H5...")
    h5_barcodes = read_10x_barcodes(args.h5, genome=args.genome)
    print(f"      H5 barcodes: {len(h5_barcodes):,}")

    print("[2/4] Reading Louvain CSV...")
    label_series = read_louvain_csv(args.louvain)
    print(f"      CSV rows after parsing: {len(label_series):,}")

    print("[3/4] Aligning labels to H5 barcode order...")
    labels_aligned, n_missing = align_labels_exact(
        h5_barcodes,
        label_series,
        missing_label=args.missing_label,
    )
    print(f"      Aligned labels: {len(labels_aligned):,}")
    print(f"      Missing labels filled as {args.missing_label}: {n_missing:,}")

    print("[4/4] Encoding labels as contiguous integers...")
    labels_int, legend = encode_labels(labels_aligned, missing_label=args.missing_label)

    np.save(out_dir / "louvain_labels_int.npy", labels_int)

    with open(out_dir / "louvain_legend.json", "w", encoding="utf-8") as f:
        json.dump(legend, f, ensure_ascii=False, indent=2)

    print("done.")
    print(f"saved: {out_dir / 'louvain_labels_int.npy'}")
    print(f"saved: {out_dir / 'louvain_legend.json'}")
    print("note: barcodes.npy is not re-saved here; it is shared with")
    print("      the embeddings barcodes.npy one level up (same H5 barcode order).")
    print(f"labels_int shape = {labels_int.shape}")
    print(f"num_classes = {len(legend)}")


if __name__ == "__main__":
    main()
