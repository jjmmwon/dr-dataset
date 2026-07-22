# higgs

UCI HIGGS particle physics dataset: 28 kinematic features + a signal/background label. The
full source CSV has ~11M rows; a random 5M-row sample is used by default (matches the size
most commonly cited for this dataset in DR/ML benchmarks).

- **Shape**: `embeddings.npy` (5,000,000, 28) by default, `float32`
- **Source**: `https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz`

## Download

Scriptable — [`scripts/download_higgs.py`](../../scripts/download_higgs.py).

```bash
pip install "dr-datasets[download]"   # or: pip install pandas pyarrow requests tqdm
python scripts/download_higgs.py
# or: python -m dr_datasets.download higgs
```

Saves to `$DR_DATA_HOME/higgs` (`/root/dataset/higgs` if `DR_DATA_HOME` is unset). The full
~2.6GB csv.gz is downloaded once and cached under `<output-dir>/raw/`; the sampled
`features_<n>.parquet` is cached under `<output-dir>/` so re-running with the same
`--n-samples`/`--seed` skips both the download and the full-CSV parse.

Use `--n-samples`/`--seed` to change the sample size or draw a different sample.

## Files

| file | shape | notes |
|---|---|---|
| `embeddings.npy` | (n_samples, 28) float32 | kinematic features |
| `labels.npy` | (n_samples,) int32 | 1=signal, 0=background |
| `features_<n>.parquet` | n_samples rows | sampled features+label cache, re-used across re-runs |
| `metadata.json` | — | shape/dtype/source metadata, incl. `feature_names`, `n_samples`, `seed` |
