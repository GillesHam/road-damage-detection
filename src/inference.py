"""
Inference wrapper that ties the trained YOLO detector to the severity model.

`RoadDamageDetector` loads an Ultralytics weight file once and exposes a single
`analyze()` call that returns, for any image, the raw detections, their severity
records and the aggregated road-condition index, plus an annotated RGB image.
Both the inference notebook and the Gradio demo use this class, so detection and
severity behaviour are identical everywhere.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config, visualization
from .severity import Detection, image_condition_index


class RoadDamageDetector:
    def __init__(self, weights: str | Path, conf: float = 0.25, iou: float = 0.6):
        # Imported here so the module can be inspected without ultralytics installed.
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        # Map model's own class indices -> our canonical class codes.
        self._names = self.model.names

    # ------------------------------------------------------------------ #
    def _to_detections(self, result) -> list[Detection]:
        dets: list[Detection] = []
        if result.boxes is None:
            return dets
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), cf, ci in zip(xyxy, confs, clss):
            code = self._names.get(int(ci), config.ID_TO_CLASS.get(int(ci), str(ci)))
            dets.append(Detection(code, float(cf), float(x1), float(y1),
                                  float(x2), float(y2)))
        return dets

    # ------------------------------------------------------------------ #
    def analyze(self, image: str | Path | np.ndarray) -> dict:
        """Run detection + severity on one image.

        `image` may be a path or an RGB numpy array. Returns a dict with the
        annotated image (RGB), the per-detection severity records and the
        overall condition index.
        """
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = rgb.shape[:2]
        result = self.model.predict(rgb[:, :, ::-1], conf=self.conf, iou=self.iou,
                                    verbose=False)[0]
        dets = self._to_detections(result)
        condition = image_condition_index(dets, w, h)
        records = condition.get("detections", [])

        annotated = visualization.draw_detections(rgb, records, color_by="severity")
        annotated = visualization.severity_banner(annotated, condition)
        return {
            "image": rgb,
            "annotated": annotated,
            "condition": condition,
            "detections": records,
        }

    # ------------------------------------------------------------------ #
    def analyze_batch(self, image_paths) -> list[dict]:
        return [self.analyze(p) for p in image_paths]


def load_best_detector(models_dir: str | Path, prefer: str = "m",
                       conf: float = 0.25) -> RoadDamageDetector:
    """Find and load the best available trained weight under `models_dir`.

    Looks for `road_damage_yolo26{scale}/weights/best.pt`, preferring the
    requested scale and falling back to whatever exists.
    """
    models_dir = Path(models_dir)
    order = [prefer] + [s for s in ["m", "s", "n"] if s != prefer]
    for scale in order:
        cand = list(models_dir.glob(f"*{scale}*/weights/best.pt"))
        if cand:
            return RoadDamageDetector(cand[0], conf=conf)
    any_best = list(models_dir.glob("**/best.pt"))
    if any_best:
        return RoadDamageDetector(any_best[0], conf=conf)
    raise FileNotFoundError(f"No trained best.pt found under {models_dir}")
