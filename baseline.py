"""Classic ML baseline: TF-IDF features + logistic regression.

TF-IDF turns each review into a long vector of word / word-pair weights
("great" appears, "waste of money" appears, ...). Logistic regression then learns
one weight per feature. It ignores word order beyond pairs, so it cannot really
understand "not bad at all" - which is exactly what the transformer should do better.

Trained on the same 60k Amazon reviews as the transformer. Saves the pipeline to
models/tfidf_logreg.joblib (evaluate.py scores it) and writes the most positive /
negative words to results/top_words.md.

Usage:  python baseline.py
"""

from __future__ import annotations

import time

from sentiment import BASELINE_PATH, RESULTS_DIR  # isort: skip  (sets HF_HOME first)

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from sentiment.data import load_amazon


def main() -> None:
    splits = load_amazon()
    start = time.time()
    pipeline = make_pipeline(
        # Unigrams + bigrams so "not good" and "waste money" become features.
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0),
    )
    pipeline.fit(splits["train"]["text"], splits["train"]["label"])
    print(f"Trained in {time.time() - start:.1f} s")

    for name in ("val", "test"):
        pred = pipeline.predict(splits[name]["text"])
        gold = splits[name]["label"]
        print(f"{name:5s} accuracy {accuracy_score(gold, pred):.4f}  macro-F1 {f1_score(gold, pred, average='macro'):.4f}")

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, BASELINE_PATH)
    print(f"Saved to {BASELINE_PATH}")

    # The learned weights are directly interpretable: big positive weight = positive word.
    vectorizer, clf = pipeline.named_steps["tfidfvectorizer"], pipeline.named_steps["logisticregression"]
    vocab = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    lines = [
        "# What the TF-IDF baseline learned",
        "",
        "Words / word pairs with the largest logistic-regression weights (trained on 60k Amazon reviews).",
        "",
        "| Most negative | weight | Most positive | weight |",
        "|---|---:|---|---:|",
    ]
    for neg, pos in zip(order[:20], order[::-1][:20]):
        lines.append(f"| {vocab[neg]} | {clf.coef_[0][neg]:.2f} | {vocab[pos]} | {clf.coef_[0][pos]:.2f} |")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "top_words.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
