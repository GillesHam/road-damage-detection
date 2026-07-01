"""
Central configuration for the Road Damage Detection project.

Everything that the notebooks and the other modules need to agree on lives here:
the class taxonomy, the colour palette used for plotting, the default training
hyper-parameters and the weights of the rule-based severity model. Keeping these
in one place means a single edit propagates to the EDA, training, inference and
demo code without any duplication.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Class taxonomy
# --------------------------------------------------------------------------- #
# The four RDD2022 damage categories we keep. The order defines the integer
# class id used in the YOLO label files (D00 -> 0, D10 -> 1, D20 -> 2, D40 -> 3).
CLASS_IDS: dict[str, int] = {
    "D00": 0,  # longitudinal crack  (runs along the driving direction)
    "D10": 1,  # transverse crack    (runs across the road)
    "D20": 2,  # alligator crack     (interconnected / fatigue cracking)
    "D40": 3,  # pothole
}
ID_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_IDS.items()}

# Human-readable names used in plots, the report and the demo UI.
CLASS_NAMES: dict[str, str] = {
    "D00": "Longitudinal crack",
    "D10": "Transverse crack",
    "D20": "Alligator crack",
    "D40": "Pothole",
}
# Ordered list of short codes, indexed by class id. This is what goes into the
# `names:` field of the Ultralytics data.yaml.
YOLO_NAMES: list[str] = [ID_TO_CLASS[i] for i in range(len(ID_TO_CLASS))]

# Distinct, colour-blind friendly RGB colours per class (for matplotlib / cv2).
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "D00": (31, 119, 180),   # blue
    "D10": (255, 127, 14),   # orange
    "D20": (148, 103, 189),  # purple
    "D40": (214, 39, 40),    # red
}

# Severity is drawn with a fixed traffic-light palette so it reads instantly.
SEVERITY_COLORS: dict[str, tuple[int, int, int]] = {
    "Low": (44, 160, 44),     # green
    "Medium": (255, 193, 7),  # amber
    "High": (214, 39, 40),    # red
}
SEVERITY_LEVELS: list[str] = ["Low", "Medium", "High"]

# --------------------------------------------------------------------------- #
# Severity model parameters
# --------------------------------------------------------------------------- #
# The severity score has no ground truth in RDD2022, so it is an engineered,
# fully transparent heuristic. Each class carries a base severity that reflects
# its structural meaning, and the geometry of the box (size / extent) scales it.
# These numbers are documented and justified in report/REPORT.md and can be
# tuned in one place.
SEVERITY_BASE: dict[str, float] = {
    "D00": 0.25,  # a single longitudinal crack is usually minor
    "D10": 0.30,  # transverse cracks slightly worse (water ingress, edges)
    "D20": 0.55,  # alligator cracking signals fatigue / sub-base failure
    "D40": 0.60,  # potholes are an immediate hazard
}
# Thresholds that turn the continuous 0-1 score into a Low/Medium/High label.
SEVERITY_THRESHOLDS: tuple[float, float] = (0.34, 0.67)

# --------------------------------------------------------------------------- #
# Training defaults (reported verbatim in the notebook and the report)
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    model_scales: list[str] = field(default_factory=lambda: ["n", "s", "m"])
    epochs: int = 100
    patience: int = 20          # early-stopping patience
    imgsz: int = 512            # native resolution of the RDD China-Drone tiles
    batch: int = 16
    optimizer: str = "AdamW"
    lr0: float = 1e-3           # initial learning rate
    lrf: float = 1e-2           # final lr factor (cosine schedule -> lr0*lrf)
    cos_lr: bool = True         # cosine learning-rate scheduler
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    seed: int = 42
    amp: bool = True            # automatic mixed precision
    # Data augmentation (Ultralytics keys). Tuned for small, top-down road tiles:
    # heavy mosaic, modest colour jitter, no vertical flip (gravity matters less
    # from a drone, but we keep it off to stay faithful to capture geometry).
    hsv_h: float = 0.015
    hsv_s: float = 0.5
    hsv_v: float = 0.4
    fliplr: float = 0.5
    flipud: float = 0.0
    mosaic: float = 1.0
    scale: float = 0.5
    translate: float = 0.1
    close_mosaic: int = 10      # disable mosaic for the last N epochs


TRAIN = TrainConfig()

# --------------------------------------------------------------------------- #
# Data-split configuration
# --------------------------------------------------------------------------- #
SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.20, 0.10)  # train / val / test
SPLIT_SEED: int = 42


# --------------------------------------------------------------------------- #
# Path discovery
# --------------------------------------------------------------------------- #
def find_project_dir() -> Path:
    """Locate the project root both locally and inside Google Colab.

    On Colab the folder lives somewhere under a mounted Drive; we glob for it so
    the notebooks do not hard-code a brittle absolute path. Locally we fall back
    to the parent of this file's `src/` directory.
    """
    patterns = [
        "/content/drive/MyDrive/**/CompVis Group Project",
        "/content/drive/Shareddrives/**/CompVis Group Project",
        "/content/drive/MyDrive/CompVis Group Project",
    ]
    for pattern in patterns:
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return Path(hits[0])
    # Local fallback: src/ -> project root
    return Path(__file__).resolve().parent.parent


def get_paths(project_dir: str | os.PathLike | None = None) -> dict[str, Path]:
    """Return the canonical set of paths used across the project."""
    root = Path(project_dir) if project_dir else find_project_dir()
    return {
        "root": root,
        "raw_images": root / "Data" / "images",
        "raw_annotations": root / "Data" / "annotations" / "xmls",
        "yolo_dataset": root / "outputs" / "yolo_dataset",
        "data_yaml": root / "outputs" / "yolo_dataset" / "data.yaml",
        "models": root / "models",
        "outputs": root / "outputs",
        "figures": root / "figures",
        "report": root / "report",
    }
