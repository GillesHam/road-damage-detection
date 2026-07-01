# Deploying the dashboard to Streamlit Community Cloud

The live demo (`final notebooks/04_Demo_App.ipynb`) is for the presentation and
runs on Colab. This document covers hosting the **same dashboard** as a
permanent public web app on the free **Streamlit Community Cloud**.

## What was added for deployment

| File | Purpose |
|---|---|
| `streamlit_app.py` | Repo-relative entry point (no Colab / Drive). Streamlit Cloud runs this. |
| `requirements.txt` | Python deps (already in the repo; `streamlit` replaces `gradio`). |
| `packages.txt` | Debian system libs so OpenCV loads (`libgl1`, `libglib2.0-0`). |
| `.streamlit/config.toml` | Dark theme + upload-size cap. |

The app serves the **YOLO26s** weight (`models/road_damage_yolo26s/weights/best.pt`,
~19 MB) by default — small enough for the free tier and the best model by mAP.
Set the `MODEL_SCALE` environment variable to `m` or `n` to serve a different one.

## Steps

1. **Push this project to a GitHub repo.**
   ```bash
   cd "CompVis Group Project"
   git init
   git add .
   git commit -m "Road damage dashboard + Streamlit deployment"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   The `models/` weights are 19–42 MB each — under GitHub's 100 MB limit, so a
   normal `git push` works (no Git LFS needed). Make sure `best.pt` is committed
   and **not** excluded by a `.gitignore`.

2. **Create the app.** Go to <https://share.streamlit.io> → *Create app* →
   *Deploy from GitHub*. Select your repo/branch and set:
   - **Main file path:** `streamlit_app.py`
   - **Python version:** 3.11 or 3.12

3. **Deploy.** The first build installs PyTorch + Ultralytics, so it takes a few
   minutes. When it finishes you get a permanent `https://<app>.streamlit.app`
   URL. Loading the model happens on the first request (cached afterwards).

## Notes & troubleshooting

- **`ImportError: libGL.so.1`** — means `packages.txt` wasn't picked up; confirm
  it is at the repo root and reboot the app from *Manage app → Reboot*.
- **Out of memory / app restarts** — the free tier is RAM-limited. Stay on the
  `s` (or `n`, ~5 MB) weight; avoid `m`. Inference is CPU-only here, ~1–3 s per
  image, which is fine for a demo.
- **Model file too big for GitHub (>100 MB)** — not an issue for these weights,
  but if you ever hit it, use Git LFS or download the weight at startup from a
  GitHub Release and load it by path.
- **Local smoke test before pushing:**
  ```bash
  pip install -r requirements.txt
  streamlit run streamlit_app.py
  ```

## Other hosts

The same `streamlit_app.py` runs unchanged on Hugging Face Spaces (Streamlit
SDK, more RAM) or in a container on Cloud Run / Render / Fly.io — only the
requirements/packages wiring differs.
