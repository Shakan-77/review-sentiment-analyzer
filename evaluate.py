"""Compare the three models on held-out test sets and write the report to results/.

Models
  * TF-IDF + logistic regression   (classic ML baseline, baseline.py)
  * RoBERTa, off the shelf         (cardiffnlp/twitter-roberta-base-sentiment-latest, NOT trained by us)
  * DistilBERT, fine-tuned by us   (train.py)

Test sets
  * Amazon Polarity test - 5,000 product reviews from the official test split (in-domain)
  * IMDB test            - 2,000 movie reviews (cross-domain: none of our models trained on movies)

Outputs: results/results.md (tables), results/metrics.json,
         results/confusion_matrix.png, results/confusion_matrix_imdb.png

Usage:  python evaluate.py
"""

from __future__ import annotations

import json
import time

from sentiment import LABELS, OFF_THE_SHELF_MODEL, RESULTS_DIR  # isort: skip  (sets HF_HOME first)

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sentiment.data import load_amazon, load_imdb_test
from sentiment.gpu import wait_for_gpu
from sentiment.plots import plot_confusion_matrices
from sentiment.predict import BaselineClassifier, load_finetuned, load_off_the_shelf, to_binary

MODEL_NAMES = {
    "baseline": "TF-IDF + LogReg",
    "roberta": "RoBERTa (off-the-shelf)",
    "distilbert": "DistilBERT (fine-tuned)",
}


def score(clf, df) -> dict:
    """Run one model over one test set and compute the metrics."""
    texts = df["text"].tolist()
    start = time.time()
    probs = clf.predict_proba(texts)
    seconds = time.time() - start
    pred = to_binary(probs, clf.labels)
    gold = df["label"].to_numpy()
    result = {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, average="macro"),
        "reviews_per_sec": len(texts) / max(seconds, 1e-9),
        "confusion_matrix": confusion_matrix(gold, pred, labels=[0, 1]).tolist(),
    }
    if "neutral" in clf.labels:
        # How often would the 3-class model have said "neutral" (argmax) on a clearly polar review?
        result["neutral_rate"] = float(np.mean(probs.argmax(1) == clf.labels.index("neutral")))
    return result


def markdown_table(results: dict, test_set: str) -> list[str]:
    lines = ["| Model | Accuracy | Macro-F1 | Speed (reviews/s) |", "|---|---:|---:|---:|"]
    for key, name in MODEL_NAMES.items():
        r = results[key][test_set]
        lines.append(f"| {name} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['reviews_per_sec']:,.0f} |")
    return lines


def main() -> None:
    wait_for_gpu(min_free_mb=1500)
    test_sets = {"amazon": load_amazon()["test"], "imdb": load_imdb_test()}
    print({k: len(v) for k, v in test_sets.items()})

    loaders = {"baseline": BaselineClassifier, "roberta": load_off_the_shelf, "distilbert": load_finetuned}
    results: dict[str, dict] = {}
    for key, load in loaders.items():
        clf = load()
        clf.predict_proba(["warm-up"])  # first call is slow (CUDA init), keep it out of the timing
        results[key] = {name: score(clf, df) for name, df in test_sets.items()}
        for name, r in results[key].items():
            print(f"{MODEL_NAMES[key]:26s} {name:7s} acc {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f}  "
                  f"{r['reviews_per_sec']:.0f} reviews/s")
        del clf

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    for name, file, title in [
        ("amazon", "confusion_matrix.png", "Amazon product reviews - held-out test set (n=5,000)"),
        ("imdb", "confusion_matrix_imdb.png", "IMDB movie reviews - cross-domain test (n=2,000)"),
    ]:
        mats = {MODEL_NAMES[k]: np.array(results[k][name]["confusion_matrix"]) for k in MODEL_NAMES}
        fig = plot_confusion_matrices(mats, LABELS, title)
        fig.savefig(RESULTS_DIR / file, dpi=150)

    train_info = ""
    history_file = RESULTS_DIR / "train_history.json"
    if history_file.exists():
        h = json.loads(history_file.read_text(encoding="utf-8"))
        train_info = (f"DistilBERT was fine-tuned on {h['args']['n_train']:,} Amazon reviews for "
                      f"{h['args']['epochs']} epochs (batch {h['final_batch_size']}, max length "
                      f"{h['args']['max_length']} tokens, fp16) in {h['total_minutes']} minutes on an RTX 4050 laptop GPU.")

    neutral = results["roberta"]["amazon"].get("neutral_rate", 0)
    lines = [
        "# Results",
        "",
        "All numbers are on reviews **no model saw during training**. Binary task: negative vs positive.",
        "The off-the-shelf RoBERTa predicts 3 classes; it is scored by comparing P(positive) with P(negative).",
        "",
        train_info,
        "",
        "## Amazon product reviews (in-domain, n = 5,000)",
        "",
        *markdown_table(results, "amazon"),
        "",
        "![Confusion matrices - Amazon](confusion_matrix.png)",
        "",
        "## IMDB movie reviews (cross-domain, n = 2,000)",
        "",
        "None of the models were trained on movie reviews; this shows how well each one transfers.",
        "",
        *markdown_table(results, "imdb"),
        "",
        "![Confusion matrices - IMDB](confusion_matrix_imdb.png)",
        "",
        "## Notes",
        "",
        f"- Off-the-shelf model: `{OFF_THE_SHELF_MODEL}` (trained on tweets). On the Amazon test set its "
        f"top prediction was *neutral* for {neutral:.1%} of reviews, even though every review is clearly "
        "1-2 or 4-5 stars - a sign of domain mismatch (tweets vs. product reviews). It also does worst on "
        "IMDB, whose long movie reviews (often 300+ words, truncated to 256 tokens) look nothing like tweets.",
        "- Speed is measured on the shared laptop GPU (transformers, fp16, batch 32) and CPU (TF-IDF); "
        "treat it as a rough comparison.",
        "- See `top_words.md` for the words the TF-IDF baseline relies on.",
        "",
    ]
    (RESULTS_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
