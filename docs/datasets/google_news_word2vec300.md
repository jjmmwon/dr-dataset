# google_news_word2vec300

Pretrained Google News word2vec vectors, 300d, ~3M words, fetched via `gensim.downloader`.

- **Shape**: `embeddings.npy` (3,000,000, 300), `float32`
- **Source**: `gensim.downloader` dataset `word2vec-google-news-300`

## Download

Scriptable — [`scripts/download_google_news_word2vec.py`](../../scripts/download_google_news_word2vec.py).

```bash
pip install "dr-datasets[download]"   # or: pip install gensim
python scripts/download_google_news_word2vec.py
# or: python -m dr_datasets.download google_news_word2vec300
```

Saves to `$DR_DATA_HOME/google_news_word2vec300`
(`/root/dataset/google_news_word2vec300` if `DR_DATA_HOME` is unset).

> **Gotcha**: `gensim.downloader` reads `GENSIM_DATA_DIR` as a module-level constant at
> import time. The script already sets this env var before importing `gensim.downloader` so
> the cache lands under `<output-dir>/raw/gensim-data` instead of `~/gensim-data` — if you're
> calling gensim yourself elsewhere, remember to set the env var *before* the import.

## Files

| file | shape | notes |
|---|---|---|
| `embeddings.npy` | (3000000, 300) float32 | word vectors |
| `words.txt` | 3,000,000 lines | one token per line, row-aligned with `embeddings.npy` |
| `metadata.json` | — | shape/dtype/source metadata |
