"""Fast unit tests (no downloads) + slower tests that only run when trained models exist."""

import numpy as np
import pandas as pd
import pytest

from sentiment import BASELINE_PATH, LABELS, MODEL_DIR
from sentiment.data import _sample, build_text, clean_text, find_text_column
from sentiment.explain import aspect_sentiment, find_aspects, split_sentences, word_importance
from sentiment.plots import plot_confusion_matrices, plot_distribution
from sentiment.predict import BaseClassifier, to_binary


# ------------------------------------------------------------------ data
def test_clean_text_removes_html_breaks_and_extra_spaces():
    assert clean_text("Great film.<br /><br />Loved   it") == "Great film. Loved it"
    assert clean_text(None) == ""


def test_build_text_joins_title_and_body():
    assert build_text("Great buy", "Works well.") == "Great buy. Works well."
    assert build_text("Great buy!", "Works well.") == "Great buy! Works well."
    assert build_text("", "Only body") == "Only body"


def test_sample_is_balanced():
    df = pd.DataFrame({"text": [f"t{i}" for i in range(100)], "label": [0] * 70 + [1] * 30})
    sample = _sample(df, 40, seed=0)
    assert sample["label"].value_counts().to_dict() == {0: 20, 1: 20}


def test_find_text_column_prefers_known_names_then_longest():
    assert find_text_column(pd.DataFrame({"id": [1], "Review": ["nice"]})) == "Review"
    df = pd.DataFrame({"name": ["Bob"], "opinion": ["This is a long piece of feedback"], "stars": [5]})
    assert find_text_column(df) == "opinion"


# --------------------------------------------------------------- predict
def test_to_binary_ignores_neutral():
    labels = ["negative", "neutral", "positive"]
    probs = np.array([[0.1, 0.6, 0.3], [0.5, 0.4, 0.1]])
    assert to_binary(probs, labels).tolist() == [1, 0]
    assert to_binary(np.array([[0.2, 0.8]]), LABELS).tolist() == [1]


class KeywordClassifier(BaseClassifier):
    """A tiny fake model: positive if it sees 'great', negative if it sees 'awful'."""

    labels = ["negative", "positive"]

    def predict_proba(self, texts):
        rows = []
        for t in texts:
            score = 0.5 + 0.4 * ("great" in t.lower()) - 0.4 * ("awful" in t.lower())
            rows.append([1 - score, score])
        return np.array(rows)


def test_predict_returns_label_dicts():
    assert KeywordClassifier().predict(["great"])[0] == pytest.approx({"negative": 0.1, "positive": 0.9})


# --------------------------------------------------------------- explain
def test_split_sentences_splits_on_punctuation_and_but():
    assert split_sentences("Love it. The price is high but quality is great!") == [
        "Love it.", "The price is high", "but quality is great!",
    ]


def test_find_aspects():
    assert find_aspects("Shipping was slow and delivery late") == ["delivery / shipping", "performance"]
    assert find_aspects("Hello there") == []


def test_word_importance_finds_the_driving_word():
    label, pieces = word_importance(KeywordClassifier(), "the food was great honestly")
    assert label == "positive"
    best = max(pieces, key=lambda p: p[1])
    assert best[0] == "great"


def test_word_importance_chunks_long_text():
    _, pieces = word_importance(KeywordClassifier(), " ".join(["word"] * 500), max_pieces=50)
    assert len(pieces) <= 50


def test_aspect_sentiment_scores_each_aspect():
    df = aspect_sentiment(KeywordClassifier(), "The quality is great. But the shipping was awful.")
    result = dict(zip(df["aspect"], df["sentiment"]))
    assert result["quality"] == "positive"
    assert result["delivery / shipping"] == "negative"


def test_aspect_sentiment_empty_when_no_aspects():
    assert aspect_sentiment(KeywordClassifier(), "Hello.").empty


# ----------------------------------------------------------------- plots
def test_plots_build_without_error():
    plot_distribution({"positive": 3, "negative": 1})
    plot_confusion_matrices({"m": np.array([[5, 1], [2, 4]])}, LABELS, "test")


# ------------------------------------------------- trained models (slow)
@pytest.mark.skipif(not (MODEL_DIR / "config.json").exists(), reason="fine-tuned model not present")
def test_finetuned_model_gets_obvious_reviews_right():
    from sentiment.predict import load_finetuned

    clf = load_finetuned()
    preds = clf.predict([
        "Absolutely fantastic, works perfectly and I would buy it again.",
        "Broke after one day. Total waste of money, do not buy.",
    ])
    assert max(preds[0], key=preds[0].get) == "positive"
    assert max(preds[1], key=preds[1].get) == "negative"


@pytest.mark.skipif(not (MODEL_DIR / "config.json").exists(), reason="fine-tuned model not present")
def test_finetuned_model_handles_negation():
    from sentiment.predict import load_finetuned

    pred = load_finetuned().predict(["Not bad at all, I expected much worse."])[0]
    assert pred["positive"] > 0.5


@pytest.mark.skipif(not BASELINE_PATH.exists(), reason="baseline model not present")
def test_baseline_model_loads_and_predicts():
    from sentiment.predict import BaselineClassifier

    clf = BaselineClassifier()
    probs = clf.predict_proba(["great product, love it", "terrible, broke immediately"])
    assert to_binary(probs, clf.labels).tolist() == [1, 0]
