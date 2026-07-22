# dr-datasets

A catalog of embedding/feature datasets for dimensionality-reduction (DR) experiments
(UMAP, t-SNE, ...). This repo ships **code and documentation only** — no data files.
Each dataset has a download script (or, if no automated download exists, manual instructions)
that fetches it from its public source into a local directory.

## Datasets

| dataset | shape (N, f) | domain | download |
|---|---|---|---|
| [`emnist_digits`](docs/datasets/emnist_digits.md) | (280,000, 784) | handwritten digit images (raw pixels) | script |
| [`covertype`](docs/datasets/covertype.md) | (581,012, 54) | forest-cover tabular features | script |
| [`svhn`](docs/datasets/svhn.md) | (630,420, 768) | CLIP embeddings of digit images | manual |
| [`sift1m`](docs/datasets/sift1m.md) | (1,000,000, 128) | SIFT image descriptors (ANN benchmark) | script |
| [`amazon_reviews_multi`](docs/datasets/amazon_reviews_multi.md) | (1,200,000, 768) | multilingual review sentence embeddings | script |
| [`mouse_brain_1m_neurons`](docs/datasets/mouse_brain_1m_neurons.md) | (1,306,127, 50) | scRNA-seq PCA embeddings | script (GPU for one step) |
| [`glove_840b_300d`](docs/datasets/glove_840b_300d.md) | (2,196,017, 300) | pretrained GloVe word vectors | script |
| [`google_news_word2vec300`](docs/datasets/google_news_word2vec300.md) | (3,000,000, 300) | pretrained word2vec word vectors | script |
| [`kddcup99`](docs/datasets/kddcup99.md) | (4,898,431, 122) | network-intrusion tabular features | script |
| [`higgs`](docs/datasets/higgs.md) | (5,000,000, 28) | particle-physics tabular features | script |

Click a dataset name for its full description, source, and download instructions. See
[`DATASETS.md`](DATASETS.md) for the same table with dtype/docs columns, plus how to add a
new dataset.

```
scripts/download_<name>.py   # fetches a dataset from its public source
dr_datasets/                 # library: loader (read) + download dispatcher
docs/datasets/<name>.md      # per-dataset description + how to get it
DATASETS.md                  # summary table of every dataset
```

## Install

```bash
pip install "dr-datasets[download] @ git+https://github.com/jjmmwon/dr-datasets.git"
```

- Base install (`dr-datasets`) only needs `numpy` — enough to use `dr_datasets.loader` against data
  you've already downloaded (e.g. in another project that just reads embeddings).
- The `[download]` extra pulls in everything the `scripts/download_*.py` scripts need
  (`requests`, `scikit-learn`, `torchvision`, `gensim`, `sentence-transformers`, `h5py`, ...).

To run the download scripts directly (recommended — lets you read/modify them), clone the
repo instead:

```bash
git clone https://github.com/jjmmwon/dr-datasets.git
cd dr-datasets
pip install -e ".[download]"
```

## Downloading a dataset

Every dataset in [`DATASETS.md`](DATASETS.md) has a page under `docs/datasets/` describing
where it comes from and how to get it. For the datasets with an automated script:

```bash
python scripts/download_covertype.py
# or, equivalently, from anywhere once the package is installed from a clone:
python -m dr_datasets.download covertype
```

Both save to `$DR_DATA_HOME/covertype` — **`DR_DATA_HOME` defaults to `/root/dataset`**; set
the env var to change it (e.g. if you're not running as root):

```bash
export DR_DATA_HOME=/home/me/dr-datasets-cache
python scripts/download_covertype.py
```

A couple of datasets need something beyond "run a script" — see their doc page:
[`svhn`](docs/datasets/svhn.md) (no automated download; manual source URL only) and
[`mouse_brain_1m_neurons`](docs/datasets/mouse_brain_1m_neurons.md) (one preprocessing step
needs a CUDA GPU).

## Loading data

```python
from dr_datasets import loader

loader.list_datasets()                         # -> datasets already downloaded into DR_DATA_HOME
loader.load_metadata("covertype")               # -> dict from metadata.json
loader.load_embeddings("covertype")             # -> np.ndarray (581012, 54), memory-mapped by default
loader.load_array("mouse_brain_1m_neurons", "labels/louvain_labels_int.npy")
loader.list_arrays("mouse_brain_1m_neurons")    # -> all .npy files in that dataset (excluding raw/)
```

`load_embeddings` memory-maps by default (`mmap="r"`) since several of these arrays are
multiple GB; pass `mmap=None` to force a full in-memory load. If a dataset hasn't been
downloaded yet, `loader` raises `FileNotFoundError` with the exact command to run.

## Adding a new dataset

See the "Adding a new dataset" section in [`DATASETS.md`](DATASETS.md).

## Known gotchas

- `mouse_brain_1m_neurons`'s `barcodes.npy`/`hvg_gene_names.npy` are object-dtype (string)
  arrays pickled by numpy >= 2.0. Loading them with numpy < 2.0 fails with
  `ModuleNotFoundError: No module named 'numpy._core'`. Plain float32 arrays
  (`embeddings.npy`, `pca_loadings.npy`, ...) are unaffected.
- Dataset-specific build gotchas (e.g. the `kddcup99` one-hot-encoding memory note, the
  `google_news_word2vec300` gensim cache-dir env var ordering) are documented on each
  dataset's page under `docs/datasets/`, not here.
