# Road Damage Detection & Severity Estimation

**Computer Vision Group Project — IE School of Science & Technology (MBD)**

This submission detects **potholes** and **road cracks** in drone road imagery
with a transfer-learned **YOLO26** detector. It also estimates a transparent
**severity score** for every detected defect and an overall **road-condition
index** per image. A live **Gradio dashboard** runs the full pipeline on newly
uploaded images for the presentation demo.

---

## 1. Problem & data

| | |
|---|---|
| **Task** | Object detection (4 damage classes) + engineered severity scoring |
| **Dataset** | [RDD2022](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) — China-Drone subset, 300 images (512×512), CC BY 4.0 |
| **Annotations** | 300 Pascal VOC XML files included in `Data/annotations/xmls/` |
| **Classes** | `D00` longitudinal crack · `D10` transverse crack · `D20` alligator crack · `D40` pothole |
| **Model** | YOLO26 (Ultralytics), COCO-pretrained → fine-tuned. Scales `n/s/m` compared |
| **Transfer learning** | COCO backbone reused; detector head adapted to 4 road-damage classes |

The dataset comes **already annotated**. The submitted pipeline therefore starts
from public Pascal VOC annotations, converts them to YOLO format, and applies
transfer learning to the road-damage classes.

---

## 2. Repository structure

```
CompVis Group Project/
├── final notebooks/
│   ├── 01_EDA.ipynb                 # data exploration + VOC→YOLO conversion + split
│   ├── 02_Training.ipynb            # transfer learning (n/s/m) + evaluation
│   ├── 03_Inference_Severity.ipynb  # inference, severity, qualitative + error analysis
│   └── 04_Demo_App.ipynb            # Gradio live-demo dashboard
├── Data/                    # submitted raw data: images/ + annotations/xmls/
├── models/                  # trained weights are saved here
├── outputs/ · figures/      # generated artefacts
├── report/REPORT.md         # full university-style project report
├── Road_Damage_Detection_Presentation.pptx
├── requirements.txt
└── README.md
```

---

## 3. How to run (Google Colab)

The notebooks are designed to run **top-to-bottom on a Colab T4 GPU**. They
mount Drive, auto-locate the project folder, and write their helper modules to
`/content/src/` at runtime. This makes the submitted notebooks self-contained:
the teacher only needs this project folder, including `Data/`, `models/`,
`outputs/`, and `figures/`.

1. Upload this whole folder to Google Drive (keep the name `CompVis Group Project`).
2. Open each notebook in Colab, set **Runtime → Change runtime type → T4 GPU**.
3. Run in order:
   1. **`final notebooks/01_EDA.ipynb`** — explore the data, document class imbalance, and build the YOLO dataset.
   2. **`final notebooks/02_Training.ipynb`** — fine-tune YOLO26 `n/s/m`, evaluate on the held-out test split, and save weights to `models/`.
   3. **`final notebooks/03_Inference_Severity.ipynb`** — run inference, score severity, and perform qualitative/failure analysis.
   4. **`final notebooks/04_Demo_App.ipynb`** — launch the Gradio dashboard (`share=True` gives a public link for the live demo).

Each notebook re-derives the train/val/test split deterministically
(`seed=42`), so they stay consistent even when run in separate sessions.

---

## 4. Severity model (headline novelty)

RDD2022 has **no severity ground truth**, so severity is an **engineered,
transparent index** (not a learned label):

```
severity = clip( base[class] + geometry_term , 0, 1 )  →  Low / Medium / High
```

* `base[class]` encodes structural meaning (pothole/alligator start higher than a hairline crack).
* `geometry_term` grows with how much of the frame the damage covers; for linear cracks it also uses length and thickness.
* An image-level **condition index** is dominated by the worst defect and nudged up when many defects coexist.

Full justification, limitations and validation discussion are in
[`report/REPORT.md`](report/REPORT.md).

---

## 5. Deliverables checklist (per project brief)

EDA · preprocessing · train/val/test split · transfer learning · training ·
evaluation (mAP50, mAP50-95, precision, recall, F1, per-class AP) · confusion
matrix · training curves · inference · prediction visualisation · qualitative
analysis · at least five failure cases · severity estimation · live demo ·
limitations & future work — all covered across the four notebooks and the
report. The accompanying `Road_Damage_Detection_Presentation.pptx` summarises
the same work for the 12-minute class presentation and live demo.
