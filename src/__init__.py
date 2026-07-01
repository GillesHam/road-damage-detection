"""Road Damage Detection — reusable source package.

Modules
-------
config         : class taxonomy, colours, hyper-parameters, severity weights, paths
data_prep      : Pascal VOC (XML) -> YOLO conversion + stratified split + data.yaml
severity       : transparent rule-based severity scoring and road-condition index
visualization  : draw detections / severity and EDA plots
evaluation     : metrics extraction, comparison tables and result plots
train          : transfer-learning helpers (YOLO26 with YOLOv8 fallback)
inference      : RoadDamageDetector tying detection + severity together

The interactive dashboard lives in the repo-root `streamlit_app.py` (and the
Colab notebook `04_Demo_App.ipynb`), not in this package.
"""
from . import config  # noqa: F401

__all__ = [
    "config", "data_prep", "severity", "visualization",
    "evaluation", "train", "inference",
]
__version__ = "1.0.0"
