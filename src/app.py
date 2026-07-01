"""
Gradio live-demo dashboard for the presentation.

Launches an interactive web app where a grader can drop in a *new* road image
(not from the training set) and immediately see the detected potholes / cracks,
each defect's severity, and an overall road-condition verdict. In Colab,
`launch(share=True)` returns a public URL that works during the live demo.

Usage (from a notebook or script):

    from src.app import build_demo
    demo = build_demo("models/road_damage_yolo26m/weights/best.pt")
    demo.launch(share=True)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .inference import RoadDamageDetector, load_best_detector


def build_demo(weights: str | Path | None = None, models_dir: str | Path = "models",
               conf: float = 0.25):
    """Build (but do not launch) the Gradio Blocks demo."""
    import gradio as gr

    detector = (RoadDamageDetector(weights, conf=conf) if weights
                else load_best_detector(models_dir, conf=conf))

    def analyze(image, conf_thr):
        if image is None:
            return None, pd.DataFrame(), "Upload a road image to begin."
        detector.conf = float(conf_thr)
        out = detector.analyze(image)
        cond = out["condition"]
        records = out["detections"]
        if records:
            table = pd.DataFrame([{
                "Class": r["class_name"],
                "Confidence": r["confidence"],
                "Severity": r["severity"],
                "Score": r["severity_score"],
                "Area %": round(r["area_fraction"] * 100, 2),
            } for r in records])
        else:
            table = pd.DataFrame(columns=["Class", "Confidence", "Severity",
                                          "Score", "Area %"])
        verdict = (
            f"## Road condition: {cond['level']}  \n"
            f"**Condition index:** {cond['index']:.2f}  ·  "
            f"**Defects found:** {cond['n_defects']}"
        )
        if cond.get("worst"):
            w = cond["worst"]
            verdict += (f"  \n**Most severe:** {w['class_name']} "
                        f"({w['severity']}, score {w['severity_score']:.2f})")
        return out["annotated"], table, verdict

    with gr.Blocks(title="Road Damage & Severity Inspector") as demo:
        gr.Markdown(
            "# 🛣️ Road Damage & Severity Inspector\n"
            "Upload a road image. The model (YOLO26, fine-tuned on RDD2022) "
            "detects **potholes** and **cracks**, scores each defect's "
            "**severity**, and reports an overall **road-condition index**."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="numpy", label="Road image")
                conf_slider = gr.Slider(0.05, 0.9, value=conf, step=0.05,
                                        label="Confidence threshold")
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                out_img = gr.Image(type="numpy", label="Detections + severity")
                verdict_md = gr.Markdown()
        out_table = gr.Dataframe(label="Detected defects", wrap=True)

        btn.click(analyze, [inp, conf_slider], [out_img, out_table, verdict_md])
        inp.change(analyze, [inp, conf_slider], [out_img, out_table, verdict_md])

    return demo
