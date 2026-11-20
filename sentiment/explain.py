"""Explain *why* a review got its sentiment.

1. Word importance (occlusion): remove one word at a time and see how much the
   model's confidence in its answer drops. Words whose removal changes the answer
   a lot are the ones driving the prediction. Simple, model-agnostic, and easy to
   explain - no gradients needed.
2. Aspect-level sentiment: split the review into sentences, tag each sentence with
   the product/movie aspects it mentions (price, quality, delivery, ...) using a
   small keyword list, and score each sentence separately. "Great quality but
   shipping took forever" -> quality: positive, delivery: negative.
"""

from __future__ import annotations

import math
import re

import pandas as pd

from .predict import BaseClassifier

# Keyword stems per aspect. A word matches if it STARTS with a stem
# ("deliver" matches delivery, delivered, ...). Extend freely.
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "price / value": ["price", "cheap", "expensive", "cost", "value", "money", "worth", "overpriced", "afford", "bargain", "refund"],
    "quality": ["quality", "durab", "sturdy", "flimsy", "broke", "broken", "cheaply made", "well made", "material", "defect", "lasted", "lasting"],
    "delivery / shipping": ["deliver", "shipping", "shipped", "arriv", "package", "packag", "courier"],
    "customer service": ["service", "support", "seller", "customer care", "return", "replace", "warranty", "staff", "respon"],
    "ease of use": ["easy", "difficult", "simple", "setup", "set up", "install", "instruction", "manual", "user friendly", "intuitive", "confusing"],
    "performance": ["work", "battery", "fast", "slow", "speed", "performance", "power", "sound", "noise", "loud"],
    "design / looks": ["design", "look", "color", "colour", "size", "fit", "style", "comfort", "beautiful", "ugly"],
    "story / plot": ["story", "plot", "script", "ending", "writing", "boring", "predictable"],
    "acting / characters": ["acting", "actor", "actress", "cast", "character", "role"],
    "visuals / music": ["visual", "effects", "cinematography", "soundtrack", "music", "score", "scene"],
}

_WORD_RE = re.compile(r"\S+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+|\s+(?=but\s)", flags=re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation, new lines and before 'but' (which usually flips the opinion)."""
    return [s.strip(" ,;") for s in _SENTENCE_RE.split(text) if s and s.strip(" ,;")]


def find_aspects(sentence: str) -> list[str]:
    """Aspects whose keywords appear in the sentence (word-start match, case-insensitive)."""
    lower = sentence.lower()
    return [
        aspect for aspect, stems in ASPECT_KEYWORDS.items()
        if any(re.search(rf"\b{re.escape(stem)}", lower) for stem in stems)
    ]


def word_importance(clf: BaseClassifier, text: str, max_pieces: int = 60) -> tuple[str, list[tuple[str, float]]]:
    """Return (predicted_label, [(text_piece, importance), ...]).

    importance > 0  -> the piece pushes TOWARDS the predicted label
    importance < 0  -> the piece pushes AGAINST it
    Long reviews are cut into at most `max_pieces` chunks of consecutive words,
    so we never need more than max_pieces + 1 forward passes.
    """
    words = _WORD_RE.findall(text)
    if not words:
        return "", []
    size = max(1, math.ceil(len(words) / max_pieces))
    pieces = [" ".join(words[i : i + size]) for i in range(0, len(words), size)]

    # One batch: the full text first, then the text with each piece left out.
    variants = [text] + [" ".join(pieces[:i] + pieces[i + 1 :]) for i in range(len(pieces))]
    probs = clf.predict_proba(variants)
    target = int(probs[0].argmax())
    scores = probs[0, target] - probs[1:, target]  # confidence drop when the piece is missing
    return clf.labels[target], [(p, float(s)) for p, s in zip(pieces, scores)]


def aspect_sentiment(clf: BaseClassifier, text: str) -> pd.DataFrame:
    """One row per aspect mentioned: overall sentiment, how sure, number of mentions, evidence."""
    rows = []
    sentences = split_sentences(text)
    tagged = [(s, find_aspects(s)) for s in sentences]
    tagged = [(s, a) for s, a in tagged if a]
    if tagged:
        probs = clf.predict_proba([s for s, _ in tagged])
        pos = clf.labels.index("positive")
        neg = clf.labels.index("negative")
        for (sentence, aspects), p in zip(tagged, probs):
            # Positive share among the two polar classes (ignores "neutral" for 3-class models).
            score = p[pos] / max(p[pos] + p[neg], 1e-9)
            rows += [{"aspect": a, "score": score, "sentence": sentence} for a in aspects]
    columns = ["aspect", "sentiment", "confidence", "mentions", "evidence"]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    out = []
    for aspect, group in df.groupby("aspect", sort=False):
        score = group["score"].mean()
        out.append({
            "aspect": aspect,
            "sentiment": "positive" if score >= 0.5 else "negative",
            "confidence": round(float(max(score, 1 - score)), 3),
            "mentions": len(group),
            # The most opinionated sentence is the best evidence.
            "evidence": group.loc[(group["score"] - 0.5).abs().idxmax(), "sentence"],
        })
    return pd.DataFrame(out, columns=columns)
