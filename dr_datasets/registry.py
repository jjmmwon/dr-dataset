"""Static catalog of datasets this library knows how to fetch.

Shared by `dr_datasets.loader` (for "how do I get this dataset" error messages)
and `dr_datasets.download` (dispatch table), so both agree on dataset names
without importing each other.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Optional


class DatasetInfo(NamedTuple):
    script: Optional[str]  # scripts/<script>, or None if there's no download script
    doc: str  # docs/datasets/<doc>


DATASETS: Dict[str, DatasetInfo] = {
    "covertype": DatasetInfo("download_covertype.py", "covertype.md"),
    "sift1m": DatasetInfo("download_sift1m.py", "sift1m.md"),
    "emnist_digits": DatasetInfo("download_emnist_digits.py", "emnist_digits.md"),
    "kddcup99": DatasetInfo("download_kddcup99.py", "kddcup99.md"),
    "google_news_word2vec300": DatasetInfo(
        "download_google_news_word2vec.py", "google_news_word2vec300.md"
    ),
    "glove_840b_300d": DatasetInfo("download_glove.py", "glove_840b_300d.md"),
    "amazon_reviews_multi": DatasetInfo(
        "download_amazon_reviews_multi.py", "amazon_reviews_multi.md"
    ),
    "higgs": DatasetInfo("download_higgs.py", "higgs.md"),
    "mouse_brain_1m_neurons": DatasetInfo(
        "download_mouse_brain_1m_neurons.py", "mouse_brain_1m_neurons.md"
    ),
    "svhn": DatasetInfo(None, "svhn.md"),
}
