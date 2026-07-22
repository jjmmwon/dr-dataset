# amazon_reviews_multi

`intfloat/multilingual-e5-base` sentence embeddings of `mteb/amazon_reviews_multi` (train
split, 6 languages x 200k rows = 1.2M rows). Row order is grouped by language — see
`metadata.json:language_offsets` for exact row ranges per language.

- **Shape**: `embeddings.npy` (1,200,000, 768), `float32`
- **Source**: HF dataset `mteb/amazon_reviews_multi` (fetched via the
  [datasets-server parquet API](https://huggingface.co/docs/datasets-server)), embedded with
  `sentence-transformers`

## Download

Scriptable — [`scripts/download_amazon_reviews_multi.py`](../../scripts/download_amazon_reviews_multi.py).
Downloads parquet files for each language/split, then runs the embedding model over all rows
incrementally (writes memory-mapped `.npy` outputs, so it doesn't hold everything in RAM at
once).

```bash
pip install "dr-datasets[download]"   # or: pip install pandas pyarrow requests sentence-transformers tqdm
python scripts/download_amazon_reviews_multi.py
# or: python -m dr_datasets.download amazon_reviews_multi
```

A GPU is strongly recommended (`--device cuda`) — embedding 1.2M rows on CPU is slow. Saves
to `$DR_DATA_HOME/amazon_reviews_multi`
(`/root/dataset/amazon_reviews_multi` if `DR_DATA_HOME` is unset).

## Files

| file | shape | notes |
|---|---|---|
| `embeddings.npy` | (1200000, 768) float32 | multilingual-e5-base sentence embeddings |
| `labels.npy` | (1200000,) int32 | star rating minus 1, values 0-4 |
| `stars.npy` | (1200000,) int32 | star rating, values 1-5 |
| `languages.npy` | (1200000,) `<U*` | language code per row (`de`, `en`, `es`, `fr`, `ja`, `zh`) |
| `text_metadata.parquet` | 1,200,000 rows | `row_idx`, `id`, `text`, `label`, `stars`, `label_text`, `language` |
| `metadata.json` | — | shape/dtype/source metadata, incl. `language_offsets` (start/end row per language) |
