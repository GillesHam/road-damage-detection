"""
Evaluation utilities: turn Ultralytics validation results into the metrics,
tables and plots the project specification asks for (mAP50, mAP50-95, precision,
recall, F1, per-class AP, parameter counts, training curves, model comparison).

These helpers are thin wrappers so the training notebook stays readable: the
heavy lifting (mAP computation, confusion matrix) is done by Ultralytics; here
we just extract, tabulate and plot.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def metrics_from_results(results, model_name: str, n_params: int | None = None) -> dict:
    """Extract the headline metrics from an Ultralytics `val()` results object."""
    box = results.box
    precision = float(np.mean(box.p)) if len(box.p) else float("nan")
    recall = float(np.mean(box.r)) if len(box.r) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else float("nan"))
    return {
        "model": model_name,
        "params(M)": round(n_params / 1e6, 2) if n_params else None,
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4),
    }


def per_class_table(results) -> pd.DataFrame:
    """Per-class AP50 / AP50-95 table from a `val()` result."""
    box = results.box
    rows = []
    for i, c in enumerate(box.ap_class_index):
        code = config.ID_TO_CLASS.get(int(c), str(c))
        rows.append({
            "class": code,
            "name": config.CLASS_NAMES.get(code, code),
            "AP50": round(float(box.ap50[i]), 4),
            "AP50-95": round(float(box.ap[i]), 4),
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
        })
    return pd.DataFrame(rows)


def comparison_table(metric_dicts: list[dict]) -> pd.DataFrame:
    """Stack the per-model metric dicts into a single comparison DataFrame."""
    return pd.DataFrame(metric_dicts).set_index("model")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_training_curves(results_csv: str | Path, title: str = "Training curves"):
    """Plot loss + mAP curves from an Ultralytics `results.csv`."""
    import matplotlib.pyplot as plt
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for col, lbl in [("train/box_loss", "train box"),
                     ("val/box_loss", "val box"),
                     ("train/cls_loss", "train cls"),
                     ("val/cls_loss", "val cls")]:
        if col in df:
            axes[0].plot(df["epoch"], df[col], label=lbl)
    axes[0].set_title("Losses"); axes[0].set_xlabel("epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    for col, lbl in [("metrics/mAP50(B)", "mAP50"),
                     ("metrics/mAP50-95(B)", "mAP50-95"),
                     ("metrics/precision(B)", "precision"),
                     ("metrics/recall(B)", "recall")]:
        if col in df:
            axes[1].plot(df["epoch"], df[col], label=lbl)
    axes[1].set_title("Validation metrics"); axes[1].set_xlabel("epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_model_comparison(comp_df: pd.DataFrame):
    """Grouped bar chart of mAP / precision / recall across model scales."""
    import matplotlib.pyplot as plt
    metrics = ["mAP50", "mAP50-95", "precision", "recall"]
    metrics = [m for m in metrics if m in comp_df.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comp_df))
    width = 0.8 / len(metrics)
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, comp_df[m], width, label=m)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(comp_df.index, rotation=10)
    ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title("Model-scale comparison"); ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def speed_accuracy_frame(comp_df: pd.DataFrame, speeds_ms: dict[str, float]) -> pd.DataFrame:
    """Attach inference speed to the comparison table for the trade-off plot."""
    df = comp_df.copy()
    df["speed(ms/img)"] = [speeds_ms.get(m, np.nan) for m in df.index]
    if "speed(ms/img)" in df:
        df["FPS"] = (1000.0 / df["speed(ms/img)"]).round(1)
    return df
