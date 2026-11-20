"""Matplotlib charts used by evaluate.py and the Gradio app.

Colours: sentiment is a *polarity* (negative <-> positive), so it uses two
opposite hues with a gray in the middle for neutral. Confusion-matrix cells
show a *magnitude* (how many reviews), so they use one blue going light -> dark.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # draw to files / images, no window needed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SENTIMENT_COLORS = {"negative": "#e34948", "neutral": "#9a9993", "positive": "#2a78d6"}
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f0efec", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])


def _style(ax) -> None:
    """Quiet axes: no box, light grid, muted tick labels."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    ax.set_facecolor(SURFACE)


def plot_distribution(counts: dict[str, int], title: str = "Sentiment distribution"):
    """Horizontal bar per sentiment, labelled with count and share."""
    order = [label for label in ("positive", "neutral", "negative") if label in counts]
    values = [counts[label] for label in order]
    total = max(sum(values), 1)
    fig, ax = plt.subplots(figsize=(6.4, 0.7 + 0.6 * len(order)), facecolor=SURFACE)
    bars = ax.barh(order, values, height=0.6, color=[SENTIMENT_COLORS[label] for label in order])
    ax.invert_yaxis()  # positive on top
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{value}  ({value / total:.0%})", va="center", color=TEXT, fontsize=10)
    ax.set_xlim(0, max(values + [1]) * 1.25)
    ax.xaxis.set_visible(False)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12)
    _style(ax)
    ax.tick_params(axis="y", labelsize=11, labelcolor=TEXT)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(matrices: dict[str, np.ndarray], labels: list[str], title: str):
    """One small confusion matrix per model, side by side (rows = true label, columns = predicted)."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (name, cm) in zip(axes, matrices.items()):
        share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)  # colour = share of each true class
        ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.1%}", ha="center", va="center", fontsize=10,
                        color="white" if share[i, j] > 0.55 else TEXT)
        ax.set_xticks(range(len(labels)), labels, color=TEXT_MUTED)
        ax.set_yticks(range(len(labels)), labels, color=TEXT_MUTED, rotation=90, va="center")
        ax.set_xlabel("predicted", color=TEXT_MUTED)
        ax.set_ylabel("true", color=TEXT_MUTED)
        ax.set_title(name, color=TEXT, fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(title, color=TEXT, fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    return fig
