#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10x Genomics mouse brain H5 -> GPU preprocessing -> PCA(50) -> save

Requires: CUDA GPU + cupy + rapids_singlecell + scanpy (cannot run on a plain CPU environment).
Normally not run directly — invoked via scripts/download_mouse_brain_1m_neurons.py.

Example:
python scripts/build_mouse_brain_pca.py \
    --input $DR_DATA_HOME/mouse_brain_1m_neurons/raw/1M_neurons_filtered_gene_bc_matrices_h5.h5 \
    --output-dir $DR_DATA_HOME/mouse_brain_1m_neurons \
    --n-top-hvg 5000 \
    --n-pcs 50

Output files (<output-dir> = --output-dir):
- <output-dir>/embeddings.npy          : (N, 50) float32
- <output-dir>/barcodes.npy             : (N,)
- <output-dir>/hvg_gene_names.npy       : (n_top_hvg,)
- <output-dir>/pca_variance_ratio.npy   : (50,)
- <output-dir>/pca_loadings.npy         : (n_top_hvg, 50)
- <output-dir>/metadata.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import scanpy as sc
import scipy.sparse as sp


def infer_10x_genome(h5_path: str, genome: str | None) -> str | None:
    """
    For a legacy 10x h5, infer the genome (e.g. mm10) from the root group.
    For the new format (/matrix/...), return None.
    """
    if genome is not None:
        return genome

    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())

        # new-style 10x h5
        if "matrix" in keys:
            return None

        # legacy 10x h5
        groups = [k for k in keys if isinstance(f[k], h5py.Group)]
        if len(groups) == 1:
            return groups[0]

        if len(groups) > 1:
            raise ValueError(
                f"Multiple genome groups found in legacy 10x H5: {groups}. "
                f"Please pass --genome explicitly."
            )

        return None


def to_numpy(x):
    """
    Safely convert cupy / dask / numpy arrays to numpy.
    """
    if hasattr(x, "compute"):
        x = x.compute()
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="10x H5 file path")
    parser.add_argument("--output-dir", type=str, required=True, help="output directory")
    parser.add_argument("--genome", type=str, default=None, help="legacy 10x genome group, e.g. mm10")
    parser.add_argument("--target-sum", type=float, default=1e4, help="normalize_total target_sum")
    parser.add_argument("--n-top-hvg", type=int, default=5000, help="number of highly variable genes")
    parser.add_argument("--hvg-flavor", type=str, default="cell_ranger", choices=["cell_ranger", "seurat"])
    parser.add_argument("--n-pcs", type=int, default=50, help="number of PCA components")
    parser.add_argument("--scale-max", type=float, default=10.0, help="clip value for scaling")
    args = parser.parse_args()

    # GPU library imports
    import cupy as cp
    import rapids_singlecell as rsc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # 1) read the 10x H5
    genome = infer_10x_genome(args.input, args.genome)
    print(f"[1/7] reading 10x h5: {args.input}")
    print(f"      inferred genome = {genome}")

    adata = sc.read_10x_h5(args.input, genome=genome, gex_only=True)
    adata.var_names_make_unique()

    # convert to CSR since row-major access dominates
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr()
    else:
        raise TypeError("Expected sparse matrix from 10x H5, but got dense matrix.")

    print(f"      raw shape = {adata.shape} (cells, genes)")

    # original shape, kept for metadata
    original_n_cells, original_n_genes = adata.shape

    # 2-4) normalize + log1p + HVG on CPU (moving the full matrix to GPU would exceed pinned memory)
    print("[2/7] normalize_total + log1p (CPU)")
    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    sc.pp.log1p(adata)

    print(f"[3/7] selecting HVGs: top {args.n_top_hvg} ({args.hvg_flavor}) (CPU)")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args.n_top_hvg,
        flavor=args.hvg_flavor,
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"      after HVG shape = {adata.shape}")

    # 4) move to GPU (after the HVG subset — a much smaller matrix)
    print("[4/7] moving HVG-subset AnnData to GPU")
    rsc.get.anndata_to_GPU(adata)

    # 5) scaling
    # zero_center=False is more memory-friendly for large sparse/scRNA data
    print("[5/7] scaling")
    rsc.pp.scale(adata, zero_center=False, max_value=args.scale_max)

    # 6) PCA to 50 dims
    print(f"[6/7] PCA -> {args.n_pcs} dims")
    rsc.pp.pca(
        adata,
        n_comps=args.n_pcs,
        use_highly_variable=False,  # already subset to HVGs
    )

    # 7) save outputs
    print("[7/7] saving outputs")
    X_pca = to_numpy(adata.obsm["X_pca"]).astype(np.float32, copy=False)
    pca_loadings = to_numpy(adata.varm["PCs"]).astype(np.float32, copy=False)
    variance_ratio = to_numpy(adata.uns["pca"]["variance_ratio"]).astype(np.float32, copy=False)

    barcodes = adata.obs_names.to_numpy()
    hvg_gene_names = adata.var_names.to_numpy()

    np.save(out_dir / "embeddings.npy", X_pca)
    np.save(out_dir / "barcodes.npy", barcodes)
    np.save(out_dir / "hvg_gene_names.npy", hvg_gene_names)
    np.save(out_dir / "pca_variance_ratio.npy", variance_ratio)
    np.save(out_dir / "pca_loadings.npy", pca_loadings)

    metadata = {
        "input_h5": str(args.input),
        "genome": genome,
        "original_shape_cells_genes": [int(original_n_cells), int(original_n_genes)],
        "final_hvg_shape_cells_genes": [int(adata.n_obs), int(adata.n_vars)],
        "output_pca_shape": [int(X_pca.shape[0]), int(X_pca.shape[1])],
        "target_sum": float(args.target_sum),
        "n_top_hvg": int(args.n_top_hvg),
        "hvg_flavor": args.hvg_flavor,
        "n_pcs": int(args.n_pcs),
        "scale_max": float(args.scale_max),
        "dtype": "float32",
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("done.")
    print(f"saved: {out_dir / 'embeddings.npy'}")
    print(f"X_pca shape = {X_pca.shape}")
    print(f"elapsed = {time.time() - t0:.2f} sec")


if __name__ == "__main__":
    main()