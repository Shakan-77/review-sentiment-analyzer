"""Gradio web app for the Review Sentiment Analyzer.

Tabs
  1. Single review - sentiment from both transformers, the words that drove the
     prediction, and aspect-level sentiment (price, quality, delivery, ...).
  2. Batch (CSV)   - upload a CSV of reviews, get per-review predictions, a
     distribution chart and a downloadable CSV.
  3. Model comparison - the evaluation results written by evaluate.py.

Usage:  python app.py      then open http://127.0.0.1:7860
"""

from __future__ import annotations

from sentiment import MODEL_DIR, RESULTS_DIR  # isort: skip  (sets HF_HOME / GRADIO_TEMP_DIR first)

import re
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd

from sentiment.data import find_text_column
from sentiment.explain import aspect_sentiment, word_importance
from sentiment.plots import plot_distribution
from sentiment.predict import load_finetuned, load_off_the_shelf, to_binary

FINETUNED = "DistilBERT (fine-tuned, positive / negative)"
ROBERTA = "RoBERTa (off-the-shelf, positive / neutral / negative)"
MAX_BATCH_ROWS = 5_000  # keep the demo responsive

EXAMPLES = [
    "Absolutely love this blender. It crushes ice in seconds and cleanup is a breeze. Worth every penny!",
    "The quality is great but shipping took three weeks and customer service never answered my emails.",
    "Stopped working after two days. Cheaply made and a waste of money. Returning it.",
    "Not bad at all - I expected much worse for the price.",
    "The acting was wooden and the plot was painfully predictable, but the soundtrack is beautiful.",
]


@lru_cache(maxsize=None)
def get_model(name: str):
    """Load each model once, on first use (downloads RoBERTa on the very first run)."""
    return load_finetuned() if name == FINETUNED else load_off_the_shelf()


def available_models() -> list[str]:
    return [FINETUNED, ROBERTA] if (MODEL_DIR / "config.json").exists() else [ROBERTA]


# ---------------------------------------------------------------- single review
def analyze(text: str):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please paste a review first.")
    models = available_models()
    main = get_model(models[0])  # the fine-tuned model when it exists

    finetuned_scores = get_model(FINETUNED).predict([text])[0] if FINETUNED in models else {}
    roberta_scores = get_model(ROBERTA).predict([text])[0]

    # Occlusion importance, re-signed so that + means "pushes towards positive".
    label, pieces = word_importance(main, text)
    sign = 1 if label == "positive" else -1
    signed = [(piece, sign * score) for piece, score in pieces]
    biggest = max((abs(s) for _, s in signed), default=0)
    threshold = max(0.02, 0.25 * biggest)  # only highlight the pieces that really matter
    highlights = [(piece + " ", ("positive" if s > 0 else "negative") if abs(s) >= threshold else None)
                  for piece, s in signed]

    aspects = aspect_sentiment(main, text)
    if aspects.empty:
        aspects = pd.DataFrame([{"aspect": "(no known aspect mentioned)", "sentiment": "", "confidence": None,
                                 "mentions": 0, "evidence": ""}])
    return finetuned_scores, roberta_scores, highlights, aspects


# ------------------------------------------------------------------- batch CSV
def read_csv(path: str) -> pd.DataFrame:
    """Read an uploaded CSV, tolerating the common non-UTF-8 encodings from Excel."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise gr.Error("Could not read the file as CSV.")


def analyze_batch(file, model_name: str):
    if file is None:
        raise gr.Error("Upload a CSV file first.")
    df = read_csv(file if isinstance(file, str) else file.name)
    column = find_text_column(df)
    df = df[df[column].notna()].head(MAX_BATCH_ROWS).reset_index(drop=True)
    texts = df[column].astype(str).tolist()

    clf = get_model(model_name)
    probs = clf.predict_proba(texts)
    out = df.copy()
    out["sentiment"] = [clf.labels[i] for i in probs.argmax(1)]
    out["confidence"] = probs.max(1).round(4)
    for j, label in enumerate(clf.labels):
        out[f"p_{label}"] = probs[:, j].round(4)
    if "neutral" in clf.labels:
        out["polarity"] = ["positive" if p else "negative" for p in to_binary(probs, clf.labels)]

    counts = out["sentiment"].value_counts().to_dict()
    fig = plot_distribution(counts, title=f"Sentiment of {len(out):,} reviews")

    path = Path(tempfile.mkdtemp()) / "review_predictions.csv"
    out.to_csv(path, index=False)
    summary = (f"Read **{len(out):,}** reviews from column **`{column}`** using *{model_name}*. "
               + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    preview = out[[column, "sentiment", "confidence"]].head(200)
    return summary, fig, preview, str(path)


# ------------------------------------------------------------ model comparison
def results_markdown() -> str:
    file = RESULTS_DIR / "results.md"
    if not file.exists():
        return "No results yet - run `python evaluate.py`."
    # Images are shown separately below, so drop the markdown image lines.
    return re.sub(r"^!\[.*\]\(.*\)\s*$", "", file.read_text(encoding="utf-8"), flags=re.MULTILINE)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Review Sentiment Analyzer") as demo:
        gr.Markdown("# Review Sentiment Analyzer\nPaste a product or movie review, or upload a CSV of reviews.")
        if FINETUNED not in available_models():
            gr.Markdown("> The fine-tuned model was not found - run `python train.py`. Using the off-the-shelf model only.")

        with gr.Tab("Single review"):
            text = gr.Textbox(label="Review", lines=5, placeholder="Type or paste a review...")
            button = gr.Button("Analyze", variant="primary")
            with gr.Row():
                finetuned_out = gr.Label(label=FINETUNED, num_top_classes=2)
                roberta_out = gr.Label(label=ROBERTA, num_top_classes=3)
            highlights = gr.HighlightedText(
                label="Which words drove the prediction (blue = pushes positive, red = pushes negative)",
                color_map={"positive": "#2a78d6", "negative": "#e34948"}, show_legend=True, combine_adjacent=True,
            )
            aspects = gr.Dataframe(label="Aspect-level sentiment", wrap=True)
            gr.Examples(EXAMPLES, inputs=text)
            outputs = [finetuned_out, roberta_out, highlights, aspects]
            button.click(analyze, inputs=text, outputs=outputs)
            text.submit(analyze, inputs=text, outputs=outputs)

        with gr.Tab("Batch (CSV)"):
            gr.Markdown(f"Upload a CSV with one review per row (a column named `review` or `text` works best; "
                        f"otherwise the longest text column is used). Up to {MAX_BATCH_ROWS:,} rows. "
                        "Try `examples/sample_reviews.csv`.")
            with gr.Row():
                file = gr.File(label="CSV file", file_types=[".csv"])
                model = gr.Radio(available_models(), value=available_models()[0], label="Model")
            run = gr.Button("Analyze file", variant="primary")
            summary = gr.Markdown()
            chart = gr.Plot(label="Sentiment distribution")
            table = gr.Dataframe(label="Predictions (first 200 rows)", wrap=True)
            download = gr.File(label="Download all predictions")
            run.click(analyze_batch, inputs=[file, model], outputs=[summary, chart, table, download])

        with gr.Tab("Model comparison"):
            gr.Markdown(results_markdown())
            for name in ("confusion_matrix.png", "confusion_matrix_imdb.png"):
                if (RESULTS_DIR / name).exists():
                    gr.Image(str(RESULTS_DIR / name), show_label=False)
    return demo


if __name__ == "__main__":
    build_app().launch()
