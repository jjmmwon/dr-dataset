# mouse_brain_1m_neurons

10x Genomics "1.3 Million Brain Cells from E18 Mice" scRNA-seq dataset, preprocessed to
HVG(5000) + PCA(50).

- **Shape**: `embeddings.npy` (1,306,127, 50), `float32`
- **Source**: 10x Genomics public dataset ("1M_neurons")

## Download

Partially scriptable — [`scripts/download_mouse_brain_1m_neurons.py`](../../scripts/download_mouse_brain_1m_neurons.py)
orchestrates 3 steps, one of which needs a GPU:

1. **Raw H5 download** (CPU only) — fetches `1M_neurons_filtered_gene_bc_matrices_h5.h5`
   (~1.4GB) from 10x Genomics.
2. **HVG + PCA preprocessing** (needs a **CUDA GPU** with `cupy` + `rapids_singlecell` +
   `scanpy` — see [`scripts/build_mouse_brain_pca.py`](../../scripts/build_mouse_brain_pca.py)).
   This step will fail on a CPU-only machine.
3. **Louvain cluster labels** (optional) — dr-datasets does not have a confirmed public URL for
   the Louvain-clustering CSV originally used (derived from scanpy's "1.3M neurons"
   tutorial). If you have this CSV, place it at
   `<output-dir>/raw/louvain.csv.gz` (a `barcode,label` CSV, gzip-compressed) before running
   the script and step 3 runs automatically; otherwise it's skipped and everything else is
   still produced.

```bash
pip install "dr-datasets[download]"   # + a CUDA/RAPIDS environment for step 2
python scripts/download_mouse_brain_1m_neurons.py
# or: python -m dr_datasets.download mouse_brain_1m_neurons

# CPU-only machine: fetch the raw H5 only, run step 2 elsewhere
python scripts/download_mouse_brain_1m_neurons.py --skip-pca
```

Saves to `$DR_DATA_HOME/mouse_brain_1m_neurons`
(`/root/dataset/mouse_brain_1m_neurons` if `DR_DATA_HOME` is unset). If the 10x Genomics URL
in the script has moved, check <https://www.10xgenomics.com/datasets> for "1.3 Million Brain
Cells from E18 Mice" and pass `--raw-h5-url`.

## Files

| file | shape | notes |
|---|---|---|
| `embeddings.npy` | (1306127, 50) float32 | PCA(50) of HVG(5000)-subset, normalized/log1p/scaled |
| `barcodes.npy` | (1306127,) | 10x cell barcodes, row-aligned with `embeddings.npy` |
| `hvg_gene_names.npy` | (5000,) | gene-indexed, **not** cell-indexed |
| `pca_loadings.npy` | (5000, 50) | gene-indexed, **not** cell-indexed |
| `pca_variance_ratio.npy` | (50,) | explained variance ratio per PC |
| `labels/louvain_labels_int.npy` | (1306127,) int32 | Louvain cluster label, integer-encoded (only if step 3 ran) |
| `labels/louvain_legend.json` | — | int -> original Louvain label string + count |
| `metadata.json` | — | shape/dtype/source metadata |
