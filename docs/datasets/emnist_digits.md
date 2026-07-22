# emnist_digits

EMNIST digits split via `torchvision.datasets.EMNIST`, train (240,000) + test (40,000)
concatenated, 28x28 grayscale digit images flattened to 784-d and `[0,1]`-normalized. No
embedding model — raw pixels are used directly, since a fixed per-row permutation like
flatten doesn't affect pairwise distances (fine for DR/UMAP/t-SNE benchmarking).

- **Shape**: `embeddings.npy` (280,000, 784), `float32`
- **Source**: `torchvision.datasets.EMNIST(split="digits")`

## Download

Scriptable — [`scripts/download_emnist_digits.py`](../../scripts/download_emnist_digits.py).

```bash
pip install "dr-datasets[download]"   # or: pip install torch torchvision
python scripts/download_emnist_digits.py
# or: python -m dr_datasets.download emnist_digits
```

Saves to `$DR_DATA_HOME/emnist_digits` (`/root/dataset/emnist_digits` if `DR_DATA_HOME` is unset).

## Files

| file | shape | notes |
|---|---|---|
| `embeddings.npy` | (280000, 784) float32 | flattened, `[0,1]`-normalized pixels |
| `labels.npy` | (280000,) int32 | digit, values 0-9 |
| `is_train.npy` | (280000,) bool | `True` for the first 240,000 rows (train), `False` for the last 40,000 (test) |
| `metadata.json` | — | shape/dtype/source metadata |

Pixel orientation follows torchvision's EMNIST convention (transposed/rotated relative to
canonical digit images — a known torchvision/EMNIST quirk); left uncorrected since it doesn't
affect pairwise distances.
