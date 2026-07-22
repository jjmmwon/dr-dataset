# svhn

CLIP embeddings of SVHN (Street View House Numbers) digit images.

- **Shape**: `embeddings.npy` (630,420, 768), `float32`
- **Model used**: [`tanganke/clip-vit-base-patch32_svhn`](https://huggingface.co/tanganke/clip-vit-base-patch32_svhn)
  (base model `openai/clip-vit-base-patch32`)

## Download

**No automated script.** The `embeddings.npy`/`labels.npy` this repo previously shipped were
copied from another project's evaluation data, and the exact embedding pipeline (image
preprocessing, batching, etc.) that produced them was not preserved — so `dr_datasets.download`
can't reproduce that exact array.

What you can do:

- **Get the raw SVHN images yourself**, from the official source:
  <http://ufldl.stanford.edu/housenumbers/> (train/test/extra `.mat` files, or the original
  format). `python -m dr_datasets.download svhn` prints this same pointer.
- **Recompute embeddings** by running the raw images through the model above (or any
  CLIP-family model) yourself — there's no ready-made script for this in dr-datasets.

If you build a working download/embedding script for this dataset, a PR replacing this file
with a real `scripts/download_svhn.py` (and updating `registry.py`/`DATASETS.md`) would be
very welcome.

## Files (if you reproduce them yourself)

| file | shape | notes |
|---|---|---|
| `embeddings.npy` | (630420, 768) float32 | CLIP image embeddings |
| `labels.npy` | (630420,) int32 | digit, values 0-9 |
| `metadata.json` | — | shape/dtype/model metadata |
