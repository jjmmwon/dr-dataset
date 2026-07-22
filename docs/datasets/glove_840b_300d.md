# glove_840b_300d

Stanford GloVe 840B token, 300d word vectors.

- **Shape**: `embeddings.npy` (2,196,017, 300), `float32`
- **Source**: `http://nlp.stanford.edu/data/glove.840B.300d.zip` (~2.03GB zip)

## Download

Scriptable — [`scripts/download_glove.py`](../../scripts/download_glove.py) downloads and
unzips the archive, then parses the space-separated text format (word + 300 floats per line).

```bash
pip install "dr-datasets[download]"   # or: pip install requests tqdm
python scripts/download_glove.py
# or: python -m dr_datasets.download glove_840b_300d
```

Saves to `$DR_DATA_HOME/glove_840b_300d`
(`/root/dataset/glove_840b_300d` if `DR_DATA_HOME` is unset). Parsing ~2.2M lines of text
takes a while — expect this to run longer than the other download scripts.

## Files

| file | shape | notes |
|---|---|---|
| `embeddings.npy` | (2196017, 300) float32 | word vectors |
| `words.txt` | 2,196,017 lines | one token per line, row-aligned with `embeddings.npy` |
| `metadata.json` | — | shape/dtype/source metadata |
