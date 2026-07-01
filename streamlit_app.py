"""
Streamlit Community Cloud entry point for the Road Damage & Severity Inspector.

This is the *deployment* twin of `final notebooks/04_Demo_App.ipynb`. Unlike the
Colab version, it reads the `src/` helper package and the trained weights from
the repository itself (no Google Drive, no `/content`), so Streamlit Community
Cloud can serve it as a permanent public web app.

Deploy: push this repo to GitHub, then on https://share.streamlit.io create an
app pointing at `streamlit_app.py`. See DEPLOY.md for the full checklist.

Local run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import sys

# Make the repo root importable so `from src import ...` works regardless of the
# directory Streamlit launches us from.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import streamlit as st
from PIL import Image

from src import config
from src.inference import load_best_detector

# Which trained scale to serve. YOLO26s (~19 MB) is the best mAP model and stays
# comfortably inside the free-tier RAM budget; override with MODEL_SCALE=m for
# higher recall if your tier allows it.
MODEL_SCALE = os.environ.get("MODEL_SCALE", "s")


# --------------------------------------------------------------------------- #
# Palette helpers (kept in sync with src/config.py so nothing is hard-coded)
# --------------------------------------------------------------------------- #
def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


SEVERITY_HEX = {lvl: _hex(rgb) for lvl, rgb in config.SEVERITY_COLORS.items()}
SEVERITY_HEX["None"] = "#64748b"
CLASS_HEX = {code: _hex(rgb) for code, rgb in config.CLASS_COLORS.items()}
SEVERITY_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "None": "⚪"}


def _accent_vars(level: str) -> str:
    r, g, b = config.SEVERITY_COLORS.get(level, (100, 116, 139))
    return (f"--accent:{_hex((r, g, b))};"
            f"--accent-soft:rgba({r},{g},{b},0.16);"
            f"--accent-line:rgba({r},{g},{b},0.55);")


# --------------------------------------------------------------------------- #
# Global stylesheet — a compact, self-contained dark theme
# --------------------------------------------------------------------------- #
CSS = """
<style>
:root {
  --rd-bg: #0a0f1c; --rd-panel: #121a2e; --rd-panel-soft: #0f1728;
  --rd-border: rgba(148,163,184,0.14); --rd-text: #e8eefc;
  --rd-muted: #9fb0cc; --rd-amber: #f59e0b;
}
.stApp {
  background:
    radial-gradient(1200px 520px at 12% -12%, rgba(245,158,11,0.12), transparent 60%),
    radial-gradient(1000px 620px at 100% -5%, rgba(56,189,248,0.10), transparent 55%),
    var(--rd-bg);
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }

/* bordered containers become dark cards */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--rd-panel); border: 1px solid var(--rd-border) !important;
  border-radius: 18px; box-shadow: 0 20px 42px -32px rgba(0,0,0,0.9);
}
[data-testid="stFileUploaderDropzone"] {
  background: var(--rd-panel-soft); border: 1px dashed var(--rd-border); border-radius: 14px;
}

/* ---- hero ------------------------------------------------------------- */
.rd-hero {
  position: relative; overflow: hidden; border-radius: 22px; padding: 30px 34px 34px;
  background: linear-gradient(135deg, #0f1b30 0%, #12233f 55%, #191a2f 100%);
  border: 1px solid var(--rd-border); box-shadow: 0 26px 55px -30px rgba(0,0,0,0.85);
  margin-bottom: 22px;
}
.rd-hero::after {
  content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 6px;
  background-image: repeating-linear-gradient(90deg, rgba(245,158,11,0.85) 0 40px, transparent 40px 78px);
  opacity: 0.85;
}
.rd-eyebrow { letter-spacing: .24em; text-transform: uppercase; font-size: 12px; font-weight: 800; color: var(--rd-amber); }
.rd-hero h1 { margin: 10px 0 8px; font-size: 34px; font-weight: 800; letter-spacing: -0.02em; color: var(--rd-text); }
.rd-hero p { margin: 0; max-width: 730px; font-size: 15px; color: var(--rd-muted); }
.rd-hero b { color: #ffd38a; font-weight: 700; }
.rd-chips { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
.rd-chip { padding: 6px 13px; border-radius: 999px; font-size: 12.5px; font-weight: 600; color: #cdd8ef; background: rgba(148,163,184,0.10); border: 1px solid var(--rd-border); }
.rd-chip b { color: var(--rd-amber); }

/* ---- section titles --------------------------------------------------- */
.rd-section-title { display: flex; align-items: center; gap: 8px; margin: 2px 2px 8px; font-weight: 700; font-size: 15px; color: var(--rd-text); }
.rd-section-title .n { color: var(--rd-muted); font-weight: 500; font-size: 13px; }

/* ---- verdict card ----------------------------------------------------- */
.rd-verdict { border-radius: 18px; padding: 22px 24px; margin-top: 8px; background: linear-gradient(160deg, var(--accent-soft), rgba(255,255,255,0)); border: 1px solid var(--rd-border); border-left: 6px solid var(--accent); }
.rd-v-head { display: flex; align-items: center; gap: 15px; }
.rd-badge { width: 56px; height: 56px; border-radius: 15px; display: grid; place-items: center; font-size: 27px; background: var(--accent-soft); border: 1px solid var(--accent-line); }
.rd-v-sub { font-size: 12px; letter-spacing: .16em; text-transform: uppercase; color: var(--rd-muted); }
.rd-v-level { font-size: 30px; font-weight: 800; line-height: 1.05; color: var(--rd-text); }
.rd-kpis { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 20px; }
.rd-kpi { flex: 1; min-width: 130px; padding: 14px 16px; border-radius: 14px; background: var(--rd-panel-soft); border: 1px solid var(--rd-border); }
.rd-kpi .val { font-size: 22px; font-weight: 800; color: var(--rd-text); }
.rd-kpi .cap { margin-top: 4px; font-size: 11.5px; letter-spacing: .09em; text-transform: uppercase; color: var(--rd-muted); }
.rd-gauge { position: relative; height: 14px; margin-top: 22px; border-radius: 999px; background: linear-gradient(90deg, #16a34a 0%, #f59e0b 55%, #dc2626 100%); }
.rd-gauge-marker { position: absolute; top: 50%; width: 20px; height: 20px; border-radius: 50%; background: #fff; border: 3px solid var(--accent); transform: translate(-50%, -50%); box-shadow: 0 2px 9px rgba(0,0,0,0.55); }
.rd-gauge-scale { display: flex; justify-content: space-between; margin-top: 9px; font-size: 11.5px; color: var(--rd-muted); }

/* ---- defects table ---------------------------------------------------- */
.rd-table-wrap { overflow-x: auto; }
.rd-dtable { width: 100%; border-collapse: separate; border-spacing: 0 8px; font-size: 14px; }
.rd-dtable th { padding: 0 14px 4px; text-align: left; font-size: 11.5px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--rd-muted); }
.rd-dtable td { padding: 12px 14px; color: var(--rd-text); background: var(--rd-panel-soft); border-top: 1px solid var(--rd-border); border-bottom: 1px solid var(--rd-border); }
.rd-dtable td:first-child { border-left: 1px solid var(--rd-border); border-radius: 12px 0 0 12px; }
.rd-dtable td:last-child { border-right: 1px solid var(--rd-border); border-radius: 0 12px 12px 0; }
.rd-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 9px; vertical-align: middle; }
.rd-pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 700; color: #fff; }
.rd-bar { display: inline-block; width: 88px; height: 8px; margin-right: 9px; border-radius: 999px; overflow: hidden; vertical-align: middle; background: rgba(148,163,184,0.20); }
.rd-bar > i { display: block; height: 100%; border-radius: 999px; background: #38bdf8; }
.rd-num { color: var(--rd-muted); font-variant-numeric: tabular-nums; }

/* ---- legend & empty states ------------------------------------------- */
.rd-legend { display: flex; flex-wrap: wrap; align-items: center; gap: 14px 20px; padding: 14px 18px; border-radius: 18px; background: var(--rd-panel); border: 1px solid var(--rd-border); margin-top: 8px; }
.rd-legend .lab { font-size: 12px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--rd-muted); }
.rd-legend .grp { display: flex; flex-wrap: wrap; gap: 14px; }
.rd-legend .item { display: flex; align-items: center; gap: 7px; font-size: 13px; color: #cdd8ef; }
.rd-empty { padding: 32px; text-align: center; color: var(--rd-muted); border: 1px dashed var(--rd-border); border-radius: 16px; background: var(--rd-panel-soft); }
.rd-empty .big { font-size: 34px; margin-bottom: 8px; }
.rd-empty b { color: var(--rd-text); }
.rd-footer { padding: 22px 0 4px; text-align: center; font-size: 12.5px; color: var(--rd-muted); }
</style>
"""


# --------------------------------------------------------------------------- #
# HTML fragment builders (all left-aligned, no blank lines: Streamlit-safe)
# --------------------------------------------------------------------------- #
def _hero_html() -> str:
    return (
        "<div class='rd-hero'>"
        "<div class='rd-eyebrow'>Computer Vision · IE School of Science &amp; Technology</div>"
        "<h1>🛣️ Road Damage &amp; Severity Inspector</h1>"
        "<p>Drop in any road photo — a phone snapshot, a dashcam frame, an image from the web —"
        " and the model localises <b>potholes</b> and <b>cracks</b>, grades each defect's"
        " severity, and returns a single, at-a-glance road-condition verdict.</p>"
        "<div class='rd-chips'>"
        "<span class='rd-chip'><b>YOLO26</b> · fine-tuned</span>"
        "<span class='rd-chip'><b>RDD2022</b> · 4 damage classes</span>"
        "<span class='rd-chip'><b>Transparent</b> severity index</span>"
        "<span class='rd-chip'><b>Real-time</b> inference</span>"
        "</div></div>"
    )


def _legend_html() -> str:
    cls_items = "".join(
        f"<span class='item'><span class='rd-dot' style='background:{CLASS_HEX[c]}'></span>"
        f"{config.CLASS_NAMES[c]}</span>"
        for c in config.YOLO_NAMES
    )
    sev_items = "".join(
        f"<span class='item'><span class='rd-pill' style='background:{SEVERITY_HEX[s]}'>{s}</span></span>"
        for s in config.SEVERITY_LEVELS
    )
    return (
        "<div class='rd-legend'>"
        "<span class='lab'>Damage classes</span>"
        f"<span class='grp'>{cls_items}</span>"
        "<span class='lab' style='margin-left:8px'>Severity</span>"
        f"<span class='grp'>{sev_items}</span>"
        "</div>"
    )


def _footer_html() -> str:
    return ("<div class='rd-footer'>CompVis group project · YOLO26 detector + engineered "
            "severity model · demo powered by Streamlit</div>")


def _empty_box(big: str, msg: str) -> str:
    return f"<div class='rd-empty'><div class='big'>{big}</div>{msg}</div>"


def _verdict_html(cond: dict) -> str:
    n = int(cond.get("n_defects", 0) or 0)
    if n == 0:
        return _empty_box("✅", "No damage detected above the current confidence "
                                "threshold. Lower the slider to probe for faint cracks.")
    level = cond.get("level", "None")
    index = float(cond.get("index", 0.0) or 0.0)
    pct = max(0, min(100, int(round(index * 100))))
    worst = cond.get("worst") or {}
    worst_name = worst.get("class_name", "—")
    worst_sev = worst.get("severity", "—")
    emoji = SEVERITY_EMOJI.get(level, "⚪")
    plural = "s" if n != 1 else ""
    return (
        f"<div class='rd-verdict' style='{_accent_vars(level)}'>"
        "<div class='rd-v-head'>"
        f"<div class='rd-badge'>{emoji}</div>"
        "<div><div class='rd-v-sub'>Overall road condition</div>"
        f"<div class='rd-v-level'>{level}</div></div></div>"
        "<div class='rd-kpis'>"
        f"<div class='rd-kpi'><div class='val'>{index:.2f}</div><div class='cap'>Condition index</div></div>"
        f"<div class='rd-kpi'><div class='val'>{n}</div><div class='cap'>Defect{plural} found</div></div>"
        f"<div class='rd-kpi'><div class='val'>{worst_name}</div><div class='cap'>Most severe · {worst_sev}</div></div>"
        "</div>"
        f"<div class='rd-gauge'><div class='rd-gauge-marker' style='left:{pct}%'></div></div>"
        "<div class='rd-gauge-scale'><span>0.0 · safe</span><span>0.5</span><span>1.0 · critical</span></div>"
        "</div>"
    )


def _table_html(records: list) -> str:
    if not records:
        return "<div class='rd-empty' style='padding:22px'>No defects to list yet.</div>"
    records = sorted(records, key=lambda r: r.get("severity_score", 0), reverse=True)
    rows = []
    for r in records:
        cls_color = CLASS_HEX.get(r["class"], "#64748b")
        sev_color = SEVERITY_HEX.get(r["severity"], "#64748b")
        conf = float(r["confidence"])
        rows.append(
            "<tr>"
            f"<td><span class='rd-dot' style='background:{cls_color}'></span>{r['class_name']}"
            f" <span class='rd-num'>· {r['class']}</span></td>"
            f"<td><span class='rd-bar'><i style='width:{int(round(conf * 100))}%'></i></span>"
            f"<span class='rd-num'>{conf:.2f}</span></td>"
            f"<td><span class='rd-pill' style='background:{sev_color}'>{r['severity']}</span></td>"
            f"<td class='rd-num'>{float(r['severity_score']):.2f}</td>"
            f"<td class='rd-num'>{float(r['area_fraction']) * 100:.1f}%</td>"
            "</tr>"
        )
    return (
        "<div class='rd-table-wrap'><table class='rd-dtable'><thead><tr>"
        "<th>Damage type</th><th>Confidence</th><th>Severity</th><th>Score</th><th>Coverage</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


# --------------------------------------------------------------------------- #
# Detector (loaded once, cached across reruns)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading YOLO26 detector…")
def get_detector():
    paths = config.get_paths()  # no Drive on a server -> resolves to repo root
    return load_best_detector(paths["models"], prefer=MODEL_SCALE, conf=0.25)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Road Damage & Severity Inspector",
                       page_icon="🛣️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(_hero_html(), unsafe_allow_html=True)

    try:
        detector = get_detector()
    except FileNotFoundError:
        st.error("No trained weight found under `models/`. Commit a "
                 "`road_damage_yolo26*/weights/best.pt` file to the repo "
                 "(the 19 MB YOLO26s is a good default) and reboot the app.")
        st.stop()

    left, right = st.columns([5, 7], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("<div class='rd-section-title'>📤 Upload road image"
                        "<span class='n'>· any phone photo or web frame</span></div>",
                        unsafe_allow_html=True)
            uploaded = st.file_uploader("Road image", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                        label_visibility="collapsed")
            conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
            st.caption("Analysis runs automatically when you upload or move the slider.")

    result = None
    if uploaded is not None:
        image = np.array(Image.open(uploaded).convert("RGB"))
        detector.conf = float(conf)
        with st.spinner("Analyzing road…"):
            result = detector.analyze(image)

    with right:
        with st.container(border=True):
            st.markdown("<div class='rd-section-title'>🔍 Detections &amp; severity"
                        "<span class='n'>· boxes coloured by severity</span></div>",
                        unsafe_allow_html=True)
            if result is not None:
                st.image(result["annotated"], use_container_width=True)
            else:
                st.markdown(_empty_box("🖼️", "The annotated result will appear here "
                                             "once you upload an image."),
                            unsafe_allow_html=True)

    if result is not None:
        st.markdown(_verdict_html(result["condition"]), unsafe_allow_html=True)
    else:
        st.markdown(_empty_box("🛣️", "Upload a road image to get an instant condition verdict."),
                    unsafe_allow_html=True)

    st.markdown("<div class='rd-section-title' style='margin-top:14px'>📋 Detected defects"
                "<span class='n'>· ranked by severity</span></div>", unsafe_allow_html=True)
    st.markdown(_table_html(result["detections"] if result else []), unsafe_allow_html=True)

    st.markdown(_legend_html(), unsafe_allow_html=True)
    st.markdown(_footer_html(), unsafe_allow_html=True)


main()
