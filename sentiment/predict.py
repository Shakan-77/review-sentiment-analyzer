"""One simple interface for every model we compare.

Each classifier exposes:
    labels                    -> list of class names, e.g. ["negative", "positive"]
    predict_proba(texts)      -> numpy array (n_texts, n_labels) of probabilities
    predict(texts)            -> list of {label: probability} dicts

That lets the evaluation script, the tests and the Gradio app treat the
fine-tuned DistilBERT, the off-the-shelf RoBERTa and the TF-IDF baseline the same way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import BASELINE_PATH, LABELS, MODEL_DIR, OFF_THE_SHELF_MODEL
from .gpu import pick_device


class BaseClassifier:
    labels: list[str]

    def predict_proba(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        probs = self.predict_proba(texts)
        return [dict(zip(self.labels, row.tolist())) for row in probs]


class TransformerClassifier(BaseClassifier):
    """Any Hugging Face sequence-classification model (local folder or Hub name)."""

    def __init__(self, name_or_path: str | Path, device: str | None = None, max_length: int = 256):
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(str(name_or_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(name_or_path)).to(self.device).eval()
        # The model config stores the class names, e.g. {0: "negative", 1: "neutral", 2: "positive"}.
        id2label = self.model.config.id2label
        self.labels = [id2label[i].lower() for i in range(len(id2label))]
        self.max_length = max_length

    @torch.inference_mode()
    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.labels)))
        chunks = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                list(texts[i : i + batch_size]), truncation=True, max_length=self.max_length,
                padding=True, return_tensors="pt",
            ).to(self.device)
            # fp16 on the GPU: roughly 2x faster and half the memory, same predictions in practice.
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                logits = self.model(**enc).logits
            chunks.append(logits.float().softmax(dim=-1).cpu().numpy())
        return np.concatenate(chunks)


def load_finetuned(model_dir: str | Path = MODEL_DIR, **kwargs) -> TransformerClassifier:
    """Our fine-tuned DistilBERT (negative / positive)."""
    if not (Path(model_dir) / "config.json").exists():
        raise FileNotFoundError(f"No trained model in {model_dir}. Run `python train.py` first.")
    return TransformerClassifier(model_dir, **kwargs)


def load_off_the_shelf(**kwargs) -> TransformerClassifier:
    """cardiffnlp RoBERTa, trained on ~124M tweets for negative / neutral / positive. Not fine-tuned by us."""
    return TransformerClassifier(OFF_THE_SHELF_MODEL, **kwargs)


class BaselineClassifier(BaseClassifier):
    """TF-IDF + logistic regression, saved by baseline.py as a scikit-learn Pipeline."""

    def __init__(self, path: str | Path = BASELINE_PATH):
        import joblib

        if not Path(path).exists():
            raise FileNotFoundError(f"No baseline model at {path}. Run `python baseline.py` first.")
        self.pipeline = joblib.load(path)
        self.labels = [LABELS[c] for c in self.pipeline.classes_]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))


def to_binary(probs: np.ndarray, labels: list[str]) -> np.ndarray:
    """Turn probabilities into 0/1 (negative/positive) predictions.

    For a 3-class model (with "neutral") we compare only P(positive) vs P(negative),
    because the review datasets have no neutral class. This is the usual, fair way
    to score a 3-class model on a 2-class test set.
    """
    return (probs[:, labels.index("positive")] >= probs[:, labels.index("negative")]).astype(int)
