#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dr_datasets.paths import data_home

DEFAULT_DATASET_ID = "mteb/amazon_reviews_multi"
DEFAULT_LANGUAGES = ["de", "en", "es", "fr", "ja", "zh"]
DEFAULT_SPLIT = "train"
DEFAULT_DATASETS_SERVER_URL = "https://datasets-server.huggingface.co/parquet"
REQUIRED_COLUMNS = ["id", "text", "label", "label_text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed the full train split of mteb/amazon_reviews_multi for the selected "
            "languages using an intfloat/multilingual-e5-* model, then save aligned "
            "embeddings.npy and metadata.parquet. This version downloads the auto-"
            "converted parquet files to a local cache and writes outputs incrementally "
            "to avoid holding all embeddings in RAM at once."
        )
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=DEFAULT_DATASET_ID,
        help="HF dataset id. Default: mteb/amazon_reviews_multi",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_LANGUAGES,
        help="Language configs to load. Default: de en es fr ja zh",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help="Dataset split to embed. Default: train",
    )
    parser.add_argument(
        "--star-offset",
        type=int,
        default=1,
        help=(
            "Value added to label when creating the derived 'stars' column. "
            "Default: 1, so labels 0..4 become stars 1..5."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="intfloat/multilingual-e5-base",
        help=(
            "SentenceTransformer model name. Examples: intfloat/multilingual-e5-small, "
            "intfloat/multilingual-e5-base, intfloat/multilingual-e5-large"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Embedding batch size used by SentenceTransformer.encode.",
    )
    parser.add_argument(
        "--row-batch-size",
        type=int,
        default=8192,
        help="How many dataset rows to load from parquet at a time. Default: 8192",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Embedding device, e.g. cuda, cuda:0, cpu. Default: auto.",
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="L2-normalize embeddings before saving.",
    )
    parser.add_argument(
        "--e5-prefix-mode",
        type=str,
        choices=["auto", "query", "passage", "none"],
        default="auto",
        help=(
            "Prefix mode for E5-family models. 'auto' uses 'query: ' for model names "
            "containing 'e5'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(data_home() / "amazon_reviews_multi"),
        help="Directory to save embeddings and metadata. "
        "Default: $DR_DATA_HOME/amazon_reviews_multi (or /root/dataset/... if unset).",
    )
    parser.add_argument(
        "--parquet-cache-dir",
        type=str,
        default=None,
        help=(
            "Directory to cache downloaded parquet files. "
            "Default: <output-dir>/raw/_parquet_cache"
        ),
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for Hugging Face parquet API calls and downloads.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Redownload parquet files even if they already exist in the cache.",
    )
    parser.add_argument(
        "--metadata-compression",
        type=str,
        default="zstd",
        help="Compression codec for text_metadata.parquet. Default: zstd",
    )
    return parser.parse_args()



def resolve_text_prefix(model_name: str, prefix_mode: str) -> str:
    if prefix_mode == "query":
        return "query: "
    if prefix_mode == "passage":
        return "passage: "
    if prefix_mode == "none":
        return ""
    if "e5" in model_name.lower():
        return "query: "
    return ""



def add_prefix_to_texts(texts: pd.Series, prefix: str) -> List[str]:
    clean = texts.fillna("").astype(str)
    if not prefix:
        return clean.tolist()
    return (prefix + clean).tolist()



def fetch_all_parquet_metadata(dataset_id: str, timeout: float) -> List[Dict]:
    response = requests.get(
        DEFAULT_DATASETS_SERVER_URL,
        params={"dataset": dataset_id},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    parquet_files = data.get("parquet_files", [])
    if not parquet_files:
        raise RuntimeError(f"No parquet files returned for dataset={dataset_id}")
    return parquet_files



def get_resolve_urls_for_split(
    parquet_files: List[Dict],
    dataset_id: str,
    config: str,
    split: str,
) -> List[str]:
    urls = []
    for item in parquet_files:
        if item.get("dataset") != dataset_id:
            continue
        if item.get("config") != config:
            continue
        if item.get("split") != split:
            continue
        url = item.get("url")
        if url:
            urls.append(str(url))
    if not urls:
        available = sorted(
            {
                (str(x.get("config")), str(x.get("split")))
                for x in parquet_files
                if x.get("dataset") == dataset_id
            }
        )
        raise RuntimeError(
            f"No parquet resolve URLs found for {dataset_id}/{config}/{split}. "
            f"Available (config, split) pairs include: {available[:20]}"
        )
    return urls



def safe_filename_from_url(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    basename = url.rstrip("/").split("/")[-1]
    if not basename.endswith(".parquet"):
        basename = f"{basename}.parquet"
    return f"{digest}_{basename}"



def download_to_cache(url: str, cache_dir: Path, timeout: float, force: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / safe_filename_from_url(url)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    if out_path.exists() and not force and out_path.stat().st_size > 0:
        return out_path

    with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    os.replace(tmp_path, out_path)
    return out_path



def prepare_local_parquet_paths(
    dataset_id: str,
    languages: Iterable[str],
    split: str,
    parquet_files: List[Dict],
    cache_root: Path,
    timeout: float,
    force_redownload: bool,
) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for language in languages:
        urls = get_resolve_urls_for_split(parquet_files, dataset_id, language, split)
        split_cache_dir = cache_root / language / split
        local_paths = [
            download_to_cache(
                url,
                cache_dir=split_cache_dir,
                timeout=timeout,
                force=force_redownload,
            )
            for url in urls
        ]
        out[language] = local_paths
    return out



def validate_parquet_columns(parquet_path: Path, required_columns: List[str]) -> None:
    schema_names = set(pq.ParquetFile(parquet_path).schema.names)
    missing = [c for c in required_columns if c not in schema_names]
    if missing:
        raise KeyError(
            f"Missing required columns in {parquet_path}: {missing}; "
            f"available={sorted(schema_names)}"
        )



def count_rows(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        total += int(pq.ParquetFile(path).metadata.num_rows)
    return total



def iter_parquet_batches(
    parquet_paths: Iterable[Path],
    columns: List[str],
    batch_size: int,
):
    for parquet_path in parquet_paths:
        pf = pq.ParquetFile(parquet_path)
        validate_parquet_columns(parquet_path, columns)
        for record_batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            yield parquet_path, record_batch



def encode_text_batch(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return np.asarray(embeddings, dtype=np.float32)



def build_language_lengths(total_rows_by_language: Dict[str, int]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in total_rows_by_language.items()}



def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_root = (
        Path(args.parquet_cache_dir)
        if args.parquet_cache_dir
        else output_dir / "raw" / "_parquet_cache"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_root / "parquet_manifest.json"
    if manifest_path.exists() and not args.force_redownload:
        with open(manifest_path, "r", encoding="utf-8") as f:
            parquet_files = json.load(f)
    else:
        parquet_files = fetch_all_parquet_metadata(args.dataset_id, timeout=args.http_timeout)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(parquet_files, f, ensure_ascii=False, indent=2)

    print(f"[1/5] Downloading/caching parquet files for split={args.split}")
    local_parquet = prepare_local_parquet_paths(
        dataset_id=args.dataset_id,
        languages=args.languages,
        split=args.split,
        parquet_files=parquet_files,
        cache_root=cache_root,
        timeout=args.http_timeout,
        force_redownload=args.force_redownload,
    )

    print("[2/5] Counting rows and preparing output arrays")
    rows_by_language = {
        language: count_rows(local_parquet[language])
        for language in args.languages
    }
    total_rows = int(sum(rows_by_language.values()))

    model = SentenceTransformer(args.model_name, device=args.device)
    embedding_dim = int(model.get_sentence_embedding_dimension())
    text_prefix = resolve_text_prefix(args.model_name, args.e5_prefix_mode)

    embeddings_path = output_dir / "embeddings.npy"
    labels_path = output_dir / "labels.npy"
    stars_path = output_dir / "stars.npy"
    languages_path = output_dir / "languages.npy"
    metadata_path = output_dir / "text_metadata.parquet"

    embeddings_mm = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, embedding_dim),
    )
    labels_mm = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.int32,
        shape=(total_rows,),
    )
    stars_mm = np.lib.format.open_memmap(
        stars_path,
        mode="w+",
        dtype=np.int32,
        shape=(total_rows,),
    )
    max_lang_len = max(len(x) for x in args.languages)
    languages_mm = np.lib.format.open_memmap(
        languages_path,
        mode="w+",
        dtype=f"<U{max_lang_len}",
        shape=(total_rows,),
    )

    print("[3/5] Embedding full dataset and writing outputs incrementally")
    row_cursor = 0
    metadata_writer: pq.ParquetWriter | None = None

    language_offsets: Dict[str, Dict[str, int]] = {}
    progress = tqdm(total=total_rows, desc="Embedding rows")

    try:
        for language in args.languages:
            start_offset = row_cursor
            for parquet_path, record_batch in iter_parquet_batches(
                local_parquet[language],
                columns=REQUIRED_COLUMNS,
                batch_size=args.row_batch_size,
            ):
                df = pa.Table.from_batches([record_batch]).to_pandas()
                if df.empty:
                    continue

                df["label"] = df["label"].astype(np.int32)
                df["stars"] = df["label"] + int(args.star_offset)
                df["language"] = language
                n_rows = len(df)

                texts = add_prefix_to_texts(df["text"], text_prefix)
                emb = encode_text_batch(
                    model=model,
                    texts=texts,
                    batch_size=args.batch_size,
                    normalize_embeddings=args.normalize_embeddings,
                )
                if emb.shape != (n_rows, embedding_dim):
                    raise RuntimeError(
                        f"Unexpected embedding shape {emb.shape} for batch with {n_rows} rows"
                    )

                next_cursor = row_cursor + n_rows
                embeddings_mm[row_cursor:next_cursor] = emb
                labels_mm[row_cursor:next_cursor] = df["label"].to_numpy(dtype=np.int32, copy=False)
                stars_mm[row_cursor:next_cursor] = df["stars"].to_numpy(dtype=np.int32, copy=False)
                languages_mm[row_cursor:next_cursor] = language

                metadata_df = df[["id", "text", "label", "stars", "label_text", "language"]].copy()
                metadata_df.insert(
                    0,
                    "row_idx",
                    np.arange(row_cursor, next_cursor, dtype=np.int64),
                )
                metadata_table = pa.Table.from_pandas(metadata_df, preserve_index=False)
                if metadata_writer is None:
                    metadata_writer = pq.ParquetWriter(
                        metadata_path,
                        metadata_table.schema,
                        compression=args.metadata_compression,
                    )
                metadata_writer.write_table(metadata_table)

                row_cursor = next_cursor
                progress.update(n_rows)

            language_offsets[language] = {
                "start_row": int(start_offset),
                "end_row_exclusive": int(row_cursor),
                "num_rows": int(row_cursor - start_offset),
            }
    finally:
        progress.close()
        if metadata_writer is not None:
            metadata_writer.close()
        embeddings_mm.flush()
        labels_mm.flush()
        stars_mm.flush()
        languages_mm.flush()
        del embeddings_mm
        del labels_mm
        del stars_mm
        del languages_mm

    if row_cursor != total_rows:
        raise RuntimeError(
            f"Wrote {row_cursor} rows, but expected {total_rows}. Outputs may be incomplete."
        )

    print("[4/5] Computing summary statistics")
    label_counts: Dict[str, int] = {}
    for label in range(5):
        label_counts[str(label)] = 0
    for language in args.languages:
        for parquet_path, record_batch in iter_parquet_batches(
            local_parquet[language],
            columns=["label"],
            batch_size=max(args.row_batch_size * 4, 32768),
        ):
            labels = np.asarray(record_batch.column(0))
            unique, counts = np.unique(labels.astype(np.int32, copy=False), return_counts=True)
            for u, c in zip(unique.tolist(), counts.tolist()):
                label_counts[str(int(u))] = label_counts.get(str(int(u)), 0) + int(c)

    summary = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "languages": args.languages,
        "total_rows": total_rows,
        "rows_by_language": build_language_lengths(rows_by_language),
        "language_offsets": language_offsets,
        "model_name": args.model_name,
        "device": args.device,
        "normalize_embeddings": bool(args.normalize_embeddings),
        "e5_prefix_mode": args.e5_prefix_mode,
        "applied_prefix": text_prefix,
        "embedding_shape": [int(total_rows), int(embedding_dim)],
        "label_counts": label_counts,
        "output_files": {
            "embeddings_npy": str(embeddings_path),
            "labels_npy": str(labels_path),
            "stars_npy": str(stars_path),
            "languages_npy": str(languages_path),
            "text_metadata_parquet": str(metadata_path),
            "metadata_json": str(output_dir / "metadata.json"),
            "parquet_manifest_json": str(manifest_path),
        },
    }

    print("[5/5] Writing summary")
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Embeddings saved to: {embeddings_path}")
    print(f"Metadata saved to:   {metadata_path}")
    print(f"Summary saved to:    {output_dir / 'metadata.json'}")
    print("Later sampling can be done by filtering text_metadata.parquet and indexing embeddings.npy with row_idx.")


if __name__ == "__main__":
    main()
