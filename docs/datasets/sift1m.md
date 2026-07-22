# sift1m

1M 128-d SIFT image descriptors, the standard ANN-search / manifold-learning benchmark,
from [ann-benchmarks.com](http://ann-benchmarks.com/)'s `sift-128-euclidean.hdf5`.

- **Shape**: `embeddings.npy` (1,000,000, 128), `float32` (the `train` split of the hdf5)
- **Source**: `http://ann-benchmarks.com/sift-128-euclidean.hdf5`

## Download

Scriptable — [`scripts/download_sift1m.py`](../../scripts/download_sift1m.py) downloads the
hdf5 file directly and splits it into the files below.

```bash
pip install "dr-datasets[download]"   # or: pip install h5py requests tqdm
python scripts/download_sift1m.py
# or: python -m dr_datasets.download sift1m
```

Saves to `$DR_DATA_HOME/sift1m` (`/root/dataset/sift1m` if `DR_DATA_HOME` is unset).

## Files

| file | shape | notes |
|---|---|---|
| `embeddings.npy` | (1000000, 128) float32 | train vectors |
| `test.npy` | (10000, 128) float32 | held-out query vectors, not part of `embeddings.npy` |
| `neighbors.npy` | (10000, 100) int32 | ground-truth top-100 neighbor indices into `embeddings.npy`, per query |
| `distances.npy` | (10000, 100) float32 | distance to each of those neighbors |
| `metadata.json` | — | shape/dtype/source metadata |

`neighbors.npy`/`distances.npy` are shipped directly in the source hdf5, not derived — kept
as-is since they're useful ANN/DR ground truth.
