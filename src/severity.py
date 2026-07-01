"""
Rule-based severity estimation for detected road damage.

RDD2022 carries no severity ground truth, so severity here is an *engineered,
fully transparent* index rather than a learned label. The design goal is that
every number can be explained to a grader in one sentence:

    severity = clip( base[class]  +  geometry_term ,  0, 1 )

* `base[class]` encodes structural meaning (a pothole starts more severe than a
  hairline longitudinal crack — see config.SEVERITY_BASE).
* `geometry_term` grows with how much of the frame the damage occupies and, for
  linear cracks, with their length and thickness.

The continuous score is then bucketed into Low / Medium / High. An image-level
"road condition index" aggregates all detections so the demo can give a single
verdict per photo. Limitations (scale ambiguity, no depth) are discussed in the
report; this module deliberately keeps the logic simple and inspectable.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass
class Detection:
    """A model detection in pixel xyxy coordinates."""
    cls: str                 # class code, e.g. "D40"
    conf: float              # detector confidence 0-1
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height


def _geometry_term(det: Detection, img_w: int, img_h: int) -> float:
    """Class-aware geometric contribution to severity, in roughly [0, 0.45].

    * Potholes / alligator cracking are area-driven: a large blown-out patch is
      far worse than a small one, so we use the square-root of the area fraction
      (sub-linear, because even a modest pothole is already a hazard).
    * Longitudinal / transverse cracks are length-and-thickness driven: a long,
      wide crack matters more than a short hairline one. We combine the relative
      length (longer side / image diagonal) with relative thickness.
    """
    img_area = float(img_w * img_h)
    area_frac = det.area / img_area if img_area else 0.0

    if det.cls in ("D40", "D20"):  # pothole, alligator -> area driven
        return min(0.45, 0.9 * (area_frac ** 0.5))

    # Linear cracks (D00, D10): length + thickness driven.
    long_side = max(det.width, det.height)
    short_side = max(1.0, min(det.width, det.height))
    diag = (img_w ** 2 + img_h ** 2) ** 0.5
    length_term = long_side / diag                      # 0-1
    thickness_term = short_side / max(img_w, img_h)      # small
    return min(0.45, 0.35 * length_term + 0.6 * thickness_term)


def severity_score(det: Detection, img_w: int, img_h: int) -> float:
    """Continuous severity in [0, 1] for a single detection."""
    base = config.SEVERITY_BASE.get(det.cls, 0.3)
    score = base + _geometry_term(det, img_w, img_h)
    return float(max(0.0, min(1.0, score)))


def severity_label(score: float) -> str:
    """Bucket a continuous score into Low / Medium / High."""
    low_t, high_t = config.SEVERITY_THRESHOLDS
    if score < low_t:
        return "Low"
    if score < high_t:
        return "Medium"
    return "High"


def score_detection(det: Detection, img_w: int, img_h: int) -> dict:
    """Full severity record for one detection (used by inference + demo)."""
    s = severity_score(det, img_w, img_h)
    return {
        "class": det.cls,
        "class_name": config.CLASS_NAMES.get(det.cls, det.cls),
        "confidence": round(float(det.conf), 3),
        "severity_score": round(s, 3),
        "severity": severity_label(s),
        "bbox": [round(det.xmin, 1), round(det.ymin, 1),
                 round(det.xmax, 1), round(det.ymax, 1)],
        "area_fraction": round(det.area / float(img_w * img_h), 4),
    }


def image_condition_index(
    detections: list[Detection], img_w: int, img_h: int
) -> dict:
    """Aggregate per-detection severities into a single road-condition verdict.

    The index is dominated by the worst single defect (a road with one deep
    pothole is unsafe regardless of how many hairline cracks it has) but is
    nudged upward when many defects coexist. Concretely:

        index = max_severity  +  0.05 * (n_defects - 1)   (capped at 1.0)

    which is then bucketed with the same Low/Medium/High thresholds.
    """
    if not detections:
        return {"index": 0.0, "level": "None", "n_defects": 0, "worst": None}

    scored = [score_detection(d, img_w, img_h) for d in detections]
    worst = max(scored, key=lambda r: r["severity_score"])
    index = min(1.0, worst["severity_score"] + 0.05 * (len(scored) - 1))
    return {
        "index": round(index, 3),
        "level": severity_label(index),
        "n_defects": len(scored),
        "worst": worst,
        "detections": scored,
    }
