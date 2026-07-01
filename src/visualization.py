"""
Visualisation helpers: draw detections and severity on images, and a few EDA
plotting utilities shared by the notebooks.

All drawing works on RGB numpy arrays so the functions are equally usable from a
notebook, a script, or the Gradio demo. Nothing here depends on a trained model;
the functions take already-computed detections / severity records.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from . import config


# --------------------------------------------------------------------------- #
# Drawing boxes
# --------------------------------------------------------------------------- #
def _draw_label(img: np.ndarray, text: str, x: int, y: int,
                color: tuple[int, int, int]) -> None:
    """Draw a filled label chip with readable text above a box corner."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    y_top = max(0, y - th - base - 4)
    cv2.rectangle(img, (x, y_top), (x + tw + 6, y_top + th + base + 4), color, -1)
    cv2.putText(img, text, (x + 3, y_top + th + 2), font, scale,
                (255, 255, 255), thick, cv2.LINE_AA)


def draw_detections(
    image: np.ndarray,
    severity_records: list[dict],
    color_by: str = "severity",
) -> np.ndarray:
    """Return a copy of `image` with boxes + labels drawn.

    `severity_records` is the list produced by `severity.score_detection`.
    `color_by` is either "severity" (traffic-light) or "class".
    """
    canvas = image.copy()
    for rec in severity_records:
        x1, y1, x2, y2 = (int(v) for v in rec["bbox"])
        if color_by == "severity":
            color = config.SEVERITY_COLORS[rec["severity"]]
        else:
            color = config.CLASS_COLORS.get(rec["class"], (0, 255, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = (f"{rec['class']} {rec['confidence']:.2f} "
                 f"| {rec['severity']} {rec['severity_score']:.2f}")
        _draw_label(canvas, label, x1, y1, color)
    return canvas


def severity_banner(image: np.ndarray, condition: dict) -> np.ndarray:
    """Stamp the overall road-condition verdict as a banner across the top."""
    canvas = image.copy()
    level = condition.get("level", "None")
    color = config.SEVERITY_COLORS.get(level, (90, 90, 90))
    text = (f"ROAD CONDITION: {level}  "
            f"(index {condition.get('index', 0):.2f}, "
            f"{condition.get('n_defects', 0)} defects)")
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, 30), color, -1)
    cv2.putText(canvas, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #
# EDA plots (matplotlib) — imported lazily so cv2-only use stays light
# --------------------------------------------------------------------------- #
def plot_class_distribution(counter: Counter, ax=None, title: str = "Class distribution"):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    codes = config.YOLO_NAMES
    counts = [counter.get(c, 0) for c in codes]
    colors = [tuple(v / 255 for v in config.CLASS_COLORS[c]) for c in codes]
    bars = ax.bar([config.CLASS_NAMES[c] for c in codes], counts, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Number of boxes")
    ax.tick_params(axis="x", rotation=20)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(c),
                ha="center", va="bottom", fontsize=9)
    return ax


def plot_boxes_per_image(annotations, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    counts = [len(a.boxes) for a in annotations]
    ax.hist(counts, bins=range(0, max(counts) + 2), color="#4C78A8",
            edgecolor="white", align="left")
    ax.set_title("Damage instances per image")
    ax.set_xlabel("boxes per image")
    ax.set_ylabel("number of images")
    return ax


def plot_bbox_size_distribution(annotations, ax=None):
    """Scatter of relative box width vs height — reveals tiny/elongated cracks."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for ann in annotations:
        for b in ann.boxes:
            color = tuple(v / 255 for v in config.CLASS_COLORS[b.cls])
            ax.scatter(b.width / ann.width, b.height / ann.height,
                       s=12, alpha=0.5, color=color)
    ax.set_xlabel("relative width")
    ax.set_ylabel("relative height")
    ax.set_title("Bounding-box shape distribution")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def make_gallery(image_paths, annotations_by_name, n: int = 6, cols: int = 3,
                 figsize=(14, 9)):
    """Grid of sample images with their VOC boxes drawn — for the EDA notebook."""
    import matplotlib.pyplot as plt
    paths = list(image_paths)[:n]
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, paths):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        ann = annotations_by_name.get(Path(p).name)
        if ann:
            for b in ann.boxes:
                color = config.CLASS_COLORS[b.cls]
                cv2.rectangle(img, (int(b.xmin), int(b.ymin)),
                              (int(b.xmax), int(b.ymax)), color, 2)
        ax.imshow(img)
        ax.set_title(Path(p).name, fontsize=8)
    fig.tight_layout()
    return fig
