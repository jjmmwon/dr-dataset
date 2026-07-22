# kddcup99

KDD Cup 1999 network intrusion detection, full (not 10%) split. Original 41 columns become
122 after one-hot encoding the 3 categorical ones (`protocol_type`, `service`, `flag`); the
other 38 numeric columns are cast to float as-is. Discrete, well-separated clusters — a
classic clustering/anomaly-detection DR benchmark.

- **Shape**: `embeddings.npy` (4,898,431, 122), `float32`
- **Source**: `sklearn.datasets.fetch_kddcup99(percent10=False)`

## Download

Scriptable — [`scripts/download_kddcup99.py`](../../scripts/download_kddcup99.py).

```bash
pip install "dr-datasets[download]"   # or: pip install scikit-learn
python scripts/download_kddcup99.py
# or: python -m dr_datasets.download kddcup99
```

Saves to `$DR_DATA_HOME/kddcup99` (`/root/dataset/kddcup99` if `DR_DATA_HOME` is unset).

> **Memory note**: the original implementation used `pandas.get_dummies()` for the one-hot
> step and OOM-killed at ~124GB RSS on a 125GB machine. The shipped script instead does the
> one-hot encoding with `sklearn.preprocessing.OneHotEncoder` directly on narrow numpy arrays
> (numeric columns cast column-by-column, categoricals isolated before encoding, the source
> object array freed before the final concat) — peak stays in the low tens of GB.

## Files

| file | shape | notes |
|---|---|---|
| `embeddings.npy` | (4898431, 122) float32 | 38 numeric + one-hot(protocol_type, service, flag) |
| `labels/attack_type_int.npy` | (4898431,) int32 | 23-class fine-grained attack type (see legend) |
| `labels/attack_type_legend.json` | — | int -> original attack-type string |
| `labels/binary.npy` | (4898431,) int32 | 0=normal, 1=attack |
| `metadata.json` | — | shape/dtype/source metadata, incl. exact numeric/categorical column split |
