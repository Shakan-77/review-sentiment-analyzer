"""Review Sentiment Analyzer package.

Importing this package first (before transformers / datasets / gradio) keeps
every file the project writes inside the project folder:
  * downloaded models and datasets  -> .hf_cache/
  * Gradio uploads, matplotlib cache, Python temp files -> .tmp/
All of these folders are gitignored. A value you set yourself (e.g. your own
HF_HOME environment variable) still wins thanks to `setdefault`.
"""

import os
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
_TMP_DIR = PROJECT_DIR / ".tmp"
_TMP_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".hf_cache"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(_TMP_DIR / "gradio"))
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_DIR / "matplotlib"))
tempfile.tempdir = str(_TMP_DIR)

# Where things live.
MODEL_DIR = PROJECT_DIR / "models" / "distilbert-sentiment"  # our fine-tuned transformer
BASELINE_PATH = PROJECT_DIR / "models" / "tfidf_logreg.joblib"  # classic ML baseline
RESULTS_DIR = PROJECT_DIR / "results"

# The pretrained checkpoint we fine-tune, and the off-the-shelf model we compare against.
BASE_MODEL = "distilbert/distilbert-base-uncased"
OFF_THE_SHELF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # negative / neutral / positive

# Our own models predict two classes; index = integer label in the dataset.
LABELS = ["negative", "positive"]
