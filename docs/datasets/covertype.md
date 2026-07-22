# covertype

UCI Covertype: 30x30m US forest patches, 54 features (10 continuous + 4 one-hot
`Wilderness_Area` + 40 one-hot `Soil_Type`), target = 1 of 7 forest cover types. A classic
UMAP/t-SNE benchmark dataset — already all-numeric, discrete well-separated-ish clusters.

- **Shape**: `embeddings.npy` (581,012, 54), `float32`
- **Source**: [`sklearn.datasets.fetch_covtype`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html) (downloads from sklearn's official mirror)

## Download

Scriptable — [`scripts/download_covertype.py`](../../scripts/download_covertype.py) wraps
`sklearn.datasets.fetch_covtype`.

```bash
pip install "dr-datasets[download]"   # or: pip install scikit-learn
python scripts/download_covertype.py
# or: python -m dr_datasets.download covertype
```

Saves to `$DR_DATA_HOME/covertype` (`/root/dataset/covertype` if `DR_DATA_HOME` is unset).

## Files

| file | shape | notes |
|---|---|---|
| `embeddings.npy` | (581012, 54) float32 | the 54 features |
| `labels.npy` | (581012,) int32 | forest cover type, values 1-7 |
| `metadata.json` | — | shape/dtype/source metadata |
