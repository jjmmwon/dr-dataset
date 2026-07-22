# Datasets

One row per dataset. None of this data is shipped in the repo — run the linked download
script (or follow the manual instructions) to populate `$DR_DATA_HOME/<name>/`
(`/root/dataset/<name>` if `DR_DATA_HOME` is unset). See [`README.md`](README.md) for the
general install/usage flow and [`dr_datasets/loader.py`](dr_datasets/loader.py) for loading code.

| dataset | embeddings shape (N, f) | dtype | download | docs |
|---|---|---|---|---|
| [`covertype`](docs/datasets/covertype.md) | (581,012, 54) | float32 | script | [covertype.md](docs/datasets/covertype.md) |
| [`sift1m`](docs/datasets/sift1m.md) | (1,000,000, 128) | float32 | script | [sift1m.md](docs/datasets/sift1m.md) |
| [`emnist_digits`](docs/datasets/emnist_digits.md) | (280,000, 784) | float32 | script | [emnist_digits.md](docs/datasets/emnist_digits.md) |
| [`kddcup99`](docs/datasets/kddcup99.md) | (4,898,431, 122) | float32 | script | [kddcup99.md](docs/datasets/kddcup99.md) |
| [`google_news_word2vec300`](docs/datasets/google_news_word2vec300.md) | (3,000,000, 300) | float32 | script | [google_news_word2vec300.md](docs/datasets/google_news_word2vec300.md) |
| [`glove_840b_300d`](docs/datasets/glove_840b_300d.md) | (2,196,017, 300) | float32 | script | [glove_840b_300d.md](docs/datasets/glove_840b_300d.md) |
| [`amazon_reviews_multi`](docs/datasets/amazon_reviews_multi.md) | (1,200,000, 768) | float32 | script | [amazon_reviews_multi.md](docs/datasets/amazon_reviews_multi.md) |
| [`higgs`](docs/datasets/higgs.md) | (5,000,000, 28) | float32 | script | [higgs.md](docs/datasets/higgs.md) |
| [`mouse_brain_1m_neurons`](docs/datasets/mouse_brain_1m_neurons.md) | (1,306,127, 50) | float32 | script (GPU for one step) | [mouse_brain_1m_neurons.md](docs/datasets/mouse_brain_1m_neurons.md) |
| [`svhn`](docs/datasets/svhn.md) | (630,420, 768) | float32 | manual | [svhn.md](docs/datasets/svhn.md) |

## Adding a new dataset

1. Pick a dataset name and add it to `DATASETS: Dict[str, DatasetInfo]` in
   [`dr_datasets/registry.py`](dr_datasets/registry.py) (script filename + doc filename).
2. Add `scripts/download_<name>.py` — argparse script with `--output-dir` defaulting to
   `str(dr_datasets.paths.data_home() / "<name>")`, that downloads from a public source and
   writes `embeddings.npy` (shape `(N, f)`) + `metadata.json` (follow the fields used in
   existing scripts: `embedding_shape`, `dtype`, `source`/`source_url`, `status`,
   `build_script`, `output_files`). If there's no public/scriptable source, skip this step —
   `registry.py`'s `script=None` signals a manual-download-only dataset (see `svhn.md` for
   the pattern).
3. Add `docs/datasets/<name>.md` (follow the structure of the existing pages: description,
   shape/dtype, source, Download, Files table).
4. Add a row to the table above.
