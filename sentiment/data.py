"""Load the review datasets used for training and evaluation.

* Amazon Polarity (https://huggingface.co/datasets/fancyzhx/amazon_polarity):
  3.6M Amazon product reviews labelled negative (1-2 stars) or positive (4-5 stars).
  We only download ONE of the four training shards (~260 MB, 900k reviews) plus the
  official test file, then take a random subsample so training fits in ~15 minutes.
* IMDB (https://huggingface.co/datasets/stanfordnlp/imdb): 50k movie reviews.
  We never train on it - it is a *cross-domain* test: how well does a model trained on
  product reviews handle movie reviews?
"""

from __future__ import annotations

import re

import pandas as pd
from datasets import load_dataset

AMAZON_REPO = "fancyzhx/amazon_polarity"
AMAZON_FILES = {
    "train": "amazon_polarity/train-00000-of-00004.parquet",
    "test": "amazon_polarity/test-00000-of-00001.parquet",
}
IMDB_REPO = "stanfordnlp/imdb"
IMDB_TEST_FILE = "plain_text/test-00000-of-00001.parquet"

# Column names we look for (in this order) when a user uploads a CSV of reviews.
TEXT_COLUMN_CANDIDATES = ["review", "review_text", "text", "content", "comment", "body", "reviews"]


def clean_text(text: str | None) -> str:
    """Remove HTML line breaks (common in IMDB) and squeeze repeated whitespace."""
    text = re.sub(r"<br\s*/?>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def build_text(title: str | None, content: str | None) -> str:
    """Amazon reviews have a title and a body; the title often carries the verdict."""
    title, content = clean_text(title), clean_text(content)
    if title and content:
        # Add a full stop so the title reads as its own sentence.
        sep = " " if title[-1] in ".!?" else ". "
        return f"{title}{sep}{content}"
    return title or content


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Balanced random sample: n/2 negative + n/2 positive reviews."""
    per_class = n // 2
    parts = [g.sample(n=min(per_class, len(g)), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_amazon(n_train: int = 60_000, n_val: int = 2_000, n_test: int = 5_000, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Return {"train", "val", "test"} DataFrames with columns `text` and `label` (0=neg, 1=pos).

    train and val come from the official training shard (disjoint), test from the
    official test file, so the test reviews are never seen during training.
    """
    ds = load_dataset(AMAZON_REPO, data_files=AMAZON_FILES)
    # Shuffle once and take a slice that is comfortably bigger than what we need,
    # instead of converting all 900k rows to pandas.
    pool = ds["train"].shuffle(seed=seed).select(range(min(len(ds["train"]), 3 * (n_train + n_val)))).to_pandas()
    test_pool = ds["test"].shuffle(seed=seed).select(range(3 * n_test)).to_pandas()

    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(text=[build_text(t, c) for t, c in zip(df["title"], df["content"])])
        df = df[df["text"].str.len() > 0].drop_duplicates(subset="text")
        return df[["text", "label"]]

    pool, test_pool = prepare(pool), prepare(test_pool)
    train = _sample(pool, n_train, seed)
    val = _sample(pool[~pool["text"].isin(set(train["text"]))], n_val, seed)
    test = _sample(test_pool, n_test, seed)
    return {"train": train, "val": val, "test": test}


def load_imdb_test(n: int = 2_000, seed: int = 42) -> pd.DataFrame:
    """A balanced subsample of the IMDB test set (columns `text`, `label`)."""
    df = load_dataset(IMDB_REPO, data_files={"test": IMDB_TEST_FILE})["test"].to_pandas()
    df = df.assign(text=df["text"].map(clean_text))
    return _sample(df[["text", "label"]], n, seed)


def find_text_column(df: pd.DataFrame) -> str:
    """Guess which column of an uploaded CSV holds the review text.

    Prefer a well-known column name; otherwise pick the text column with the
    longest average length (reviews are usually the longest field).
    """
    lower = {c.lower().strip(): c for c in df.columns}
    for name in TEXT_COLUMN_CANDIDATES:
        if name in lower:
            return lower[name]
    text_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    if not text_cols:
        raise ValueError("The CSV has no text column. Add a column named 'review' or 'text'.")
    return max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())
