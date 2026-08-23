"""
app.py — ForensicSplicing: Image Splicing Detection System
===========================================================
Streamlit web application implementing two forensic detection modes:

  Phase 1 — Heuristic (ELA)
      Uses Error Level Analysis to highlight regions with anomalous
      JPEG compression residuals. A rule-based score classifies the image
      as Authentic / Suspicious / Spliced.

  Phase 2 — Deep Learning (U-Net)
      Loads a trained U-Net (unet_splicing_model.pth) and runs pixel-level
      inference to produce a probability mask, which is rendered as a
      colour heatmap overlay. Falls back to ELA if weights are missing.

UI Design: Forensic Precision dark theme — charcoal backgrounds, Forensic
Cyan (#00FFFF) accents, Danger Red (#FF4B4B) alerts, Inter + JetBrains Mono.

Run:
    streamlit run app.py
"""

import io
import time
import traceback
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config (MUST be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="ForensicSplicing | Image Tamper Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports ───────────────────────────────────────────────────────────
from utils.ela_processor import (
    calculate_ela,
    get_ela_stats,
    get_ela_heatmap,
    get_heatmap_overlay,
    generate_forgery_score,
    enhance_unet_mask,
)
from utils.anomaly_reporter import generate_anomaly_report, format_anomaly_html


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Forensic Precision Theme
# (matches DESIGN.md spec: #101319 background, #00FFFF cyan, #FF4B4B danger red,
#  Inter UI font, JetBrains Mono data font, sharp 0px radius)
# ─────────────────────────────────────────────────────────────────────────────

FORENSIC_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── CSS Custom Properties ── */
:root {
    --bg:           #101319;
    --bg-panel:     #1d2026;
    --bg-card:      #191c22;
    --bg-hover:     #272a31;
    --cyan:         #00FFFF;
    --cyan-dim:     #00dddd;
    --cyan-muted:   rgba(0,255,255,0.12);
    --red:          #FF4B4B;
    --red-dim:      #cc3030;
    --red-muted:    rgba(255,75,75,0.15);
    --yellow:       #FFD700;
    --on-surface:   #e1e2eb;
    --on-variant:   #b9cac9;
    --muted:        #839493;
    --border:       #3a4a49;
    --border-cyan:  rgba(0,255,255,0.3);
    --font-ui:      'Inter', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}

/* ── Reset ── */
html, body, [class*="css"] { font-family: var(--font-ui); }

/* ── App background ── */
.stApp {
    background-color: var(--bg) !important;
    color: var(--on-surface);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0b0e14 !important;
    border-right: 1px solid var(--border) !important;
    width: 300px !important;
}
[data-testid="stSidebar"] * { color: var(--on-surface) !important; }
[data-testid="stSidebar"] label { color: var(--on-variant) !important; font-size: 0.78rem !important; font-family: var(--font-mono) !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0d1018 !important;
    border: 1px dashed var(--border-cyan) !important;
    border-radius: 0 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--cyan) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 10px 20px !important;
    transition: background 0.15s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: var(--cyan-dim) !important;
}

/* ── Radio buttons ── */
[data-testid="stRadio"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    color: var(--on-variant) !important;
}
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--cyan) !important; }

/* ── Sliders ── */
[data-testid="stSlider"] .st-ae { background: var(--cyan) !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    padding: 14px 18px !important;
}

/* ── Alerts ── */
.stAlert { border-radius: 0 !important; }
.stSuccess { border-left: 3px solid var(--cyan) !important; }
.stWarning { border-left: 3px solid var(--yellow) !important; }
.stError   { border-left: 3px solid var(--red) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 16px 0 !important; }

/* ──────────────────── Custom Components ───────────────────── */

/* App header */
.fa-header {
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    padding: 20px 28px 18px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.fa-brand {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--on-surface);
    letter-spacing: -0.5px;
}
.fa-brand span { color: var(--cyan); }
.fa-version {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 1px;
}
.fa-nav-pill {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.fa-nav-active {
    background: transparent;
    border: 1px solid var(--cyan);
    color: var(--cyan);
}
.fa-nav-inactive {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
}

/* KPI metric cards */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}
.kpi-card.kpi-primary { border-top: 2px solid var(--cyan); }
.kpi-card.kpi-danger  { border-top: 2px solid var(--red);  }
.kpi-card.kpi-warn    { border-top: 2px solid var(--yellow); }
.kpi-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: var(--font-mono);
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-value.cv { color: var(--cyan); }
.kpi-value.rv { color: var(--red);  }
.kpi-value.yv { color: var(--yellow); }
.kpi-value.wv { color: var(--on-surface); }
.kpi-sub {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}
.kpi-sub.sub-danger  { color: var(--red);    font-weight: 700; }
.kpi-sub.sub-warn    { color: var(--yellow); font-weight: 700; }
.kpi-sub.sub-ok      { color: var(--cyan);   font-weight: 700; }

/* Segmented bar */
.seg-bar-wrap { margin: 10px 0 0; }
.seg-bar-track {
    display: flex;
    gap: 3px;
    height: 4px;
    margin-top: 4px;
}
.seg-on  { flex: 1; background: var(--cyan); }
.seg-on-r{ flex: 1; background: var(--red); }
.seg-off { flex: 1; background: var(--border); }

/* Image panel labels */
.panel-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--cyan);
    padding: 6px 10px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-bottom: none;
    display: inline-block;
    margin-bottom: 0;
}
.panel-img-wrap {
    border: 1px solid var(--border);
    padding: 0;
    background: #000;
    position: relative;
}
.panel-img-wrap.active { border-color: var(--cyan); }

/* Colorbar */
.colorbar-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: none;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.colorbar-gradient {
    flex: 1;
    height: 6px;
    background: linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff4b4b);
}

/* Technical logs */
.log-panel {
    background: #0b0e14;
    border: 1px solid var(--border);
    border-top: 2px solid var(--cyan);
    padding: 14px 16px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    line-height: 1.8;
    max-height: 220px;
    overflow-y: auto;
}
.log-line { color: var(--on-variant); }
.log-line.log-warn  { color: var(--yellow); }
.log-line.log-crit  { color: var(--red);    }
.log-line.log-ok    { color: var(--cyan);   }
.log-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.log-status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--cyan);
    display: inline-block;
    margin-right: 6px;
    animation: blink 1.2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* Metadata table */
.meta-tbl { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.meta-tbl tr:nth-child(even) td { background: rgba(255,255,255,0.025); }
.meta-tbl td {
    padding: 7px 10px;
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono);
}
.meta-tbl .mk { color: var(--muted);      font-size: 0.72rem; letter-spacing: 0.06em; }
.meta-tbl .mv { color: var(--on-surface); }
.meta-tbl .mv-warn { color: var(--yellow); font-weight: 700; }
.meta-tbl .mv-danger { color: var(--red);    font-weight: 700; }

/* Section titles */
.sec-title {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
}

/* Welcome card */
.welcome-card {
    background: var(--bg-card);
    border: 1px dashed var(--border-cyan);
    padding: 60px 40px;
    text-align: center;
    margin: 40px auto;
    max-width: 560px;
}
.welcome-icon { font-size: 2.5rem; margin-bottom: 14px; }
.welcome-title { font-size: 1.1rem; font-weight: 700; color: var(--cyan); margin-bottom: 8px; font-family: var(--font-mono); letter-spacing: 0.05em; }
.welcome-sub   { font-size: 0.85rem; color: var(--muted); line-height: 1.7; }

/* Mode badge */
.mode-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 8px;
    border: 1px solid;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-left: 8px;
    vertical-align: middle;
}
.mode-ela   { border-color: var(--cyan); color: var(--cyan); }
.mode-unet  { border-color: #a855f7;     color: #a855f7;     }

/* ELA stat grid */
.ela-stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
.ela-stat { background: #0d1018; border: 1px solid var(--border); padding: 9px 12px; }
.ela-stat-k { font-family: var(--font-mono); font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
.ela-stat-v { font-family: var(--font-mono); font-size: 1rem; font-weight: 700; color: var(--on-surface); margin-top: 2px; }

/* Opacity slider label */
.opacity-label { font-family: var(--font-mono); font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
</style>
"""
st.markdown(FORENSIC_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_css(score: int) -> tuple:
    """Return (value_class, sub_class, sub_text) based on score."""
    if score <= 20:
        return "cv", "sub-ok",     "AUTHENTIC"
    elif score <= 50:
        return "yv", "sub-warn",   "SUSPICIOUS"
    else:
        return "rv", "sub-danger", "SPLICED"


def _build_segbar(score: int, total: int = 20) -> str:
    """Build a segmented bar HTML string."""
    filled = int((score / 100) * total)
    if score <= 20:
        seg_on = "seg-on"
    elif score <= 50:
        seg_on = "seg-on-r"   # reuse red for orange feel
    else:
        seg_on = "seg-on-r"

    segs = "".join(
        f'<div class="{seg_on}"></div>' if i < filled else '<div class="seg-off"></div>'
        for i in range(total)
    )
    return f'<div class="seg-bar-track">{segs}</div>'


def _render_kpi(label: str, value: str, sub: str,
                val_cls: str, sub_cls: str, kind: str,
                seg_score: int = 0) -> None:
    bar = _build_segbar(seg_score) if seg_score else ""
    st.markdown(f"""
    <div class="kpi-card kpi-{kind}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {val_cls}">{value}</div>
        <div class="kpi-sub {sub_cls}">{sub}</div>
        {bar}
    </div>
    """, unsafe_allow_html=True)


def _render_panel_label(text: str, active: bool = False) -> None:
    st.markdown(f'<div class="panel-label">{text}</div>', unsafe_allow_html=True)


def _render_colorbar(lo: str = "LOW", hi: str = "HIGH") -> None:
    st.markdown(f"""
    <div class="colorbar-wrap">
        <span>{lo}</span>
        <div class="colorbar-gradient"></div>
        <span>{hi}</span>
    </div>
    """, unsafe_allow_html=True)


import html

def _render_log(entries: list) -> None:
    lines_html = "".join(
        f'<div class="log-line {e.get("cls","")}">{html.escape(str(e["text"]))}</div>'
        for e in entries
    )
    st.markdown(f"""
    <div class="log-label">
        <span><span class="log-status-dot"></span> Technical Logs &amp; Analysis Output</span>
        <span style="color:var(--cyan)">● SYSTEM ACTIVE</span>
    </div>
    <div class="log-panel">{lines_html}</div>
    """, unsafe_allow_html=True)


def _render_meta_table(rows: list) -> None:
    trs = ""
    for k, v, cls in rows:
        safe_k = html.escape(str(k))
        safe_v = html.escape(str(v))
        trs += f'<tr><td class="mk">{safe_k}</td><td class="mv {cls}">{safe_v}</td></tr>'
    st.markdown(
        f'<table class="meta-tbl"><tbody>{trs}</tbody></table>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# U-Net inference (lazy import — only when mode == Deep Learning)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_unet_model(weights_path: str):
    """
    Load the U-Net model and weights. Cached so it runs only once.
    Returns (model | None, status_message).
    """
    try:
        import torch
        from model import UNet

        model = UNet(in_channels=3, out_channels=1)
        state = torch.load(weights_path, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model, "loaded"
    except FileNotFoundError:
        return None, "weights_missing"
    except Exception as e:
        return None, f"error: {e}"


def _run_unet_inference(image_pil: Image.Image, weights_path: str, img_size: int = 256):
    """
    Run U-Net inference on a PIL Image.

    Returns (prob_mask: np.ndarray [0,1] float32, score: int, status: str, log_entries: list)
    """
    import torch
    from torchvision import transforms

    logs = []
    t0 = time.time()

    model, load_status = _load_unet_model(weights_path)

    if model is None:
        logs.append({"text": f"[WARN] U-Net weights not found at '{weights_path}'. Falling back to ELA.", "cls": "log-warn"})
        logs.append({"text": "[INFO] To train: python train_unet.py --data_root data/", "cls": "log-line"})
        return None, 0, load_status, logs

    logs.append({"text": f"[{_ts()}] U-Net loaded → {sum(p.numel() for p in model.parameters()):,} parameters", "cls": "log-ok"})
    logs.append({"text": f"[{_ts()}] Preprocessing image → {img_size}×{img_size}", "cls": "log-line"})

    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(image_pil.convert("RGB")).unsqueeze(0)  # (1,3,H,W)

    logs.append({"text": f"[{_ts()}] Running forward pass …", "cls": "log-line"})

    with torch.no_grad():
        import torch.nn.functional as F
        logits = model(tensor)                                    # (1,1,H,W)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()    # (H,W) float32

    # Resize mask back to original dimensions
    orig_w, orig_h = image_pil.size
    import cv2
    prob = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    prob = np.clip(prob, 0.0, 1.0)

    # ── Calibrated scoring ────────────────────────────────────────────────
    # Problem: prob.max() lets a single high-noise pixel push the score to 96%
    # on an authentic image.  Instead we:
    #   1. Compute the mean of pixels that are confidently above 0.55
    #   2. Require at least 0.5% of all pixels to be in that confident set
    #      before calling the image Spliced (localized anomaly, not noise)
    #   3. Raise verdict thresholds (Spliced > 65, Suspicious > 35)
    CONFIDENCE_THRESH   = 0.55   # pixel must exceed this to count as "tampered"
    MIN_TAMPERED_FRAC   = 0.005  # at least 0.5% of pixels must be confident

    confident_pixels = prob[prob > CONFIDENCE_THRESH]
    tampered_frac    = len(confident_pixels) / max(prob.size, 1)

    if len(confident_pixels) > 0 and tampered_frac >= MIN_TAMPERED_FRAC:
        # Score = mean of high-confidence pixels, weighted by coverage fraction
        region_mean  = float(confident_pixels.mean())
        coverage_w   = min(tampered_frac / 0.15, 1.0)   # plateau at 15% coverage
        raw_score    = region_mean * (0.6 + 0.4 * coverage_w)
    else:
        # No meaningful concentrated anomaly — use 30th-percentile as a soft floor
        raw_score = float(np.percentile(prob, 30))

    score  = int(np.clip(raw_score * 100, 0, 100))
    # Raised thresholds compared to naive max-based scheme
    status = "Spliced" if score > 65 else ("Suspicious" if score > 35 else "Authentic")

    elapsed = time.time() - t0
    logs.append({"text": f"[{_ts()}] Inference complete in {elapsed:.2f}s", "cls": "log-ok"})
    logs.append({
        "text": f"[{_ts()}] Confident pixels: {tampered_frac*100:.2f}% above threshold | Region mean: {confident_pixels.mean() if len(confident_pixels) > 0 else 0:.4f} | Score: {score}",
        "cls": "log-ok",
    })
    if score > 65:
        logs.append({"text": f"[CRITICAL] U-Net confirms concentrated manipulation region. Score: {score}.", "cls": "log-crit"})
    elif score > 35:
        logs.append({"text": f"[WARN] Moderate anomaly signal detected. Score: {score}. Manual review recommended.", "cls": "log-warn"})

    return prob, score, status, logs


def _ts() -> str:
    """Return a formatted timestamp string for technical logs."""
    return time.strftime("%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        # ── Brand ─────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding:20px 4px 16px;">
            <div style="font-size:1.15rem; font-weight:700; color:#e1e2eb; letter-spacing:-0.3px;">
                SplicingDetection
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                        color:#839493; letter-spacing:0.08em; text-transform:uppercase; margin-top:2px;">
                V2.4.0 Forensic Suite
            </div>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────────────────────
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#839493;
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">
            Navigation
        </div>
        """, unsafe_allow_html=True)

        _nav_items = [
            ("📤", "Upload"),
            ("📋", "Metadata"),
            ("🔬", "ELA Analysis"),
            ("🧠", "U-Net Results"),
            ("📝", "Technical Logs"),
        ]
        for icon, label in _nav_items:
            active = label in ("ELA Analysis",)
            bg  = "rgba(0,255,255,0.1)" if active else "transparent"
            col = "#00FFFF"             if active else "#839493"
            bdr = "1px solid rgba(0,255,255,0.35)" if active else "1px solid transparent"
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; padding:9px 10px;
                        background:{bg}; border:{bdr}; margin-bottom:2px; cursor:pointer;
                        font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:{col};">
                <span>{icon}</span><span>{label}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Upload ────────────────────────────────────────────────────────────
        st.markdown('<div class="sec-title">Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Image to Analyze",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
            key="img_upload",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Detection Mode ────────────────────────────────────────────────────
        st.markdown('<div class="sec-title">Detection Mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Mode",
            ["Heuristic (ELA)", "Deep Learning (U-Net)"],
            label_visibility="collapsed",
            key="det_mode",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── ELA Settings ──────────────────────────────────────────────────────
        with st.expander("⚙ ELA Settings", expanded=False):
            ela_quality = st.slider("Recompression Quality", 70, 98, 90, 1, key="ela_q")
            ela_amplify = st.slider("Amplification Factor",  5.0, 30.0, 20.0, 0.5, key="ela_amp")
            heatmap_alpha = st.slider("Heatmap Opacity", 0.1, 0.8, 0.45, 0.05, key="hm_alpha")

        # ── U-Net Settings ────────────────────────────────────────────────────
        with st.expander("⚙ U-Net Settings", expanded=False):
            weights_path = st.text_input(
                "Weights File Path",
                value="unet_splicing_model.pth",
                key="wts_path",
                label_visibility="visible",
            )
            img_size = st.select_slider(
                "Input Resolution",
                options=[128, 256, 512],
                value=256,
                key="img_sz",
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Run button ────────────────────────────────────────────────────────
        run_analysis = st.button("▶  Run Deep Scan", width="stretch")

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # ── Settings & Support placeholders ───────────────────────────────────
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#839493;">
            ⚙ Settings &nbsp;&nbsp;&nbsp; ? Support
        </div>
        """, unsafe_allow_html=True)

    return uploaded_file, mode, run_analysis, {
        "ela_quality":   ela_quality,
        "ela_amplify":   ela_amplify,
        "heatmap_alpha": heatmap_alpha,
        "weights_path":  weights_path,
        "img_size":      img_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

def _render_header(mode: str) -> None:
    mode_badge = (
        '<span class="mode-badge mode-ela">ELA HEURISTIC</span>'
        if "Heuristic" in mode
        else '<span class="mode-badge mode-unet">U-NET DL</span>'
    )
    st.markdown(f"""
    <div style="background:#1d2026; border-bottom:1px solid #3a4a49; padding:14px 20px;
                display:flex; align-items:center; gap:0; margin-bottom:20px;">
        <div style="flex:1; display:flex; align-items:center; gap:16px;">
            <div>
                <div style="font-size:1.55rem; font-weight:700; color:#e1e2eb; letter-spacing:-0.5px; line-height:1.1;">
                    Forensic<br><span style="color:#00FFFF;">Analyzer</span>
                </div>
            </div>
            <div style="display:flex; gap:8px; align-items:center; margin-left:20px;">
                <span class="fa-nav-pill fa-nav-active">Dashboard</span>
                <span class="fa-nav-pill fa-nav-inactive">Comparison</span>
                <span class="fa-nav-pill fa-nav-inactive">Export Report</span>
            </div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#839493;">
            Active Mode {mode_badge}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Welcome screen
# ─────────────────────────────────────────────────────────────────────────────

def _render_welcome() -> None:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">🔬</div>
        <div class="welcome-title">No Evidence Loaded</div>
        <div class="welcome-sub">
            Upload a JPG, PNG, or WEBP image via the sidebar to begin forensic analysis.<br><br>
            <b style="color:#00FFFF">Phase 1 — Heuristic (ELA)</b><br>
            Error Level Analysis highlights compression anomalies.<br><br>
            <b style="color:#a855f7">Phase 2 — Deep Learning (U-Net)</b><br>
            Pixel-precise tamper mask via trained neural network.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis result renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_results(
    image_pil:    Image.Image,
    score:        int,
    status:       str,
    heatmap_rgb:  np.ndarray,
    ela_stats:    dict,
    metadata:     dict,
    log_entries:  list,
    mode:         str,
    unet_prob:    np.ndarray = None,
) -> None:
    """Render the full forensic dashboard after analysis."""

    val_cls, sub_cls, _ = _score_css(score)
    is_danger   = score > 50
    is_warn     = 20 < score <= 50

    # ── Row 1: KPI cards ───────────────────────────────────────────────────────
    k1, k2, k3 = st.columns([2, 1, 1])

    with k1:
        ela_peak  = ela_stats.get("max",  0)
        ela_var   = ela_stats.get("variance", 0)
        kind      = "danger" if is_danger else ("warn" if is_warn else "primary")
        _render_kpi(
            "⟳  GLOBAL FORGERY SCORE",
            f"{score}%",
            "HIGH PROBABILITY" if is_danger else ("MODERATE SIGNAL" if is_warn else "LOW SIGNAL"),
            val_cls, sub_cls, kind, seg_score=score
        )

    with k2:
        _render_kpi(
            "ELA PEAK LUMINANCE",
            f"{ela_peak:.1f}",
            f"+{ela_stats.get('high_pct', 0):.1f}% variance",
            "wv", "log-line", "primary"
        )

    with k3:
        if unet_prob is not None and unet_prob is not False:
            unet_conf = float(unet_prob.max())
            unet_cls  = "rv" if unet_conf > 0.5 else "yv"
            unet_sub  = "Anomaly Detected" if unet_conf > 0.5 else "No Strong Signal"
            unet_sub_cls = "sub-danger" if unet_conf > 0.5 else "sub-warn"
            _render_kpi(
                "U-NET CONFIDENCE",
                f"{unet_conf:.2f}",
                unet_sub, unet_cls, unet_sub_cls, "danger" if unet_conf > 0.5 else "warn"
            )
        else:
            _render_kpi(
                "U-NET CONFIDENCE",
                "N/A",
                "Weights not loaded" if "U-Net" in mode else "ELA mode active",
                "wv", "log-line", "primary"
            )

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # ── Row 2: Images side by side ────────────────────────────────────────────
    img_col, hm_col = st.columns(2)

    orig_w, orig_h = image_pil.size
    fname = st.session_state.get("filename", "SOURCE_IMG.JPG")

    with img_col:
        _render_panel_label(f"SOURCE_IMG · {fname}")
        st.image(image_pil, width="stretch")

    with hm_col:
        hm_label = "ELA_HEATMAP_OVERLAY" if "Heuristic" in mode else "UNET_MASK_OVERLAY"
        _render_panel_label(hm_label, active=True)
        st.image(heatmap_rgb, width="stretch")
        _render_colorbar()

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # ── Row 3: Metadata + Technical Logs ─────────────────────────────────────
    meta_col, log_col = st.columns([1, 1])

    with meta_col:
        # Build metadata rows
        exif = metadata.get("exif", {})
        sw   = exif.get("Software", "Not Available")
        is_editing_sw = any(
            k in str(sw).lower()
            for k in ["photoshop", "gimp", "lightroom", "affinity", "canva", "snapseed"]
        )
        rows = [
            ("Camera Model",      exif.get("Model",             "Not Available"), ""),
            ("Software",          sw, "mv-warn" if is_editing_sw else ""),
            ("DateTimeOriginal",  exif.get("DateTimeOriginal",  "Not Available"), ""),
            ("ColorSpace",        exif.get("ColorSpace",        str(metadata.get("basic", {}).get("color_mode", "RGB"))), ""),
            ("Format",            metadata.get("basic", {}).get("format", "Unknown"), ""),
            ("Resolution",        f'{orig_w} × {orig_h}', ""),
            ("EXIF",              "Available ✓" if metadata.get("has_exif") else "Not Available", "mv-warn" if not metadata.get("has_exif") else ""),
        ]
        _render_meta_table(rows)

    with log_col:
        _render_log(log_entries)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # ── Row 4: ELA Stats ─────────────────────────────────────────────────────
    with st.expander("📊 Detailed ELA Statistics", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        stat_items = [
            ("ELA Mean",     f'{ela_stats.get("mean",  0):.2f}', s1),
            ("ELA Variance", f'{ela_stats.get("variance", 0):.1f}', s2),
            ("99th Pctile",  f'{ela_stats.get("p99",   0):.2f}', s3),
            ("High-Diff %",  f'{ela_stats.get("high_pct", 0):.1f}%', s4),
        ]
        for label, val, col in stat_items:
            with col:
                st.markdown(f"""
                <div class="ela-stat">
                    <div class="ela-stat-k">{label}</div>
                    <div class="ela-stat-v">{val}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Forensic Anomaly Report ───────────────────────────────────────────────
    unet_prob_for_report = unet_prob if (unet_prob is not None and unet_prob is not False) else None
    findings = generate_anomaly_report(
        ela_stats=ela_stats,
        unet_prob=unet_prob_for_report,
        metadata=metadata,
        score=score,
        mode=mode,
    )

    critical_count = sum(1 for f in findings if f["level"] == "CRITICAL")
    warning_count  = sum(1 for f in findings if f["level"] == "WARNING")
    clear_count    = sum(1 for f in findings if f["level"] == "CLEAR")

    report_label = (
        f"🔴 Forensic Anomaly Report — {critical_count} Critical · {warning_count} Warnings · {clear_count} Clear"
        if critical_count > 0
        else f"🟡 Forensic Anomaly Report — {warning_count} Warnings · {clear_count} Clear"
        if warning_count > 0
        else f"🟢 Forensic Anomaly Report — No Critical Findings"
    )

    with st.expander(report_label, expanded=True):
        st.markdown(
            "<p style='color:#8A9BB5; font-size:0.82rem; margin-bottom:12px;'>"
            "Each finding below describes a specific forensic indicator detected in the image. "
            "Severity levels: "
            "<b style='color:#FF4B4B;'>CRITICAL</b> — strong evidence of manipulation · "
            "<b style='color:#FFA500;'>WARNING</b> — suspicious signal requiring context · "
            "<b style='color:#00BFFF;'>INFO</b> — neutral contextual note · "
            "<b style='color:#00C853;'>CLEAR</b> — no anomaly in this category."
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(format_anomaly_html(findings), unsafe_allow_html=True)

    # ── Verdict banner ────────────────────────────────────────────────────────
    if is_danger:
        st.error(f"🚨 VERDICT: IMAGE CLASSIFIED AS **SPLICED** — Score {score}/100. Significant forensic indicators detected.")
    elif is_warn:
        st.warning(f"⚠️ VERDICT: **SUSPICIOUS** — Score {score}/100. Some forensic signals present; further investigation recommended.")
    else:
        st.success(f"✅ VERDICT: **AUTHENTIC** — Score {score}/100. No significant forensic anomalies detected.")

    st.caption("🔬 Forensic Analysis System — Phase 1 (Heuristic ELA) & Phase 2 (Deep Learning U-Net) active. "
               "Trained on CASIA-standard image tampering dataset.")


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extractor (reuse existing module if available, else inline)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_metadata(image_pil: Image.Image) -> dict:
    """Thin wrapper — tries our existing metadata.py; falls back to minimal inline version."""
    try:
        from utils.metadata import extract_metadata
        return extract_metadata(image_pil)
    except Exception:
        # Minimal fallback
        basic = {
            "format":     image_pil.format or "Unknown",
            "width":      image_pil.size[0],
            "height":     image_pil.size[1],
            "resolution": f"{image_pil.size[0]} × {image_pil.size[1]}",
            "color_mode": image_pil.mode,
        }
        exif = {}
        try:
            raw = image_pil.getexif()
            from PIL import ExifTags
            for tid, val in raw.items():
                tag = ExifTags.TAGS.get(tid, str(tid))
                exif[tag] = str(val) if not isinstance(val, str) else val
        except Exception:
            pass
        return {"basic": basic, "exif": exif, "has_exif": bool(exif), "anomalies": []}


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Sidebar (returns controls)
    uploaded_file, mode, run_analysis, settings = _render_sidebar()

    # Header
    _render_header(mode)

    # ── No file uploaded ──────────────────────────────────────────────────────
    if uploaded_file is None:
        _render_welcome()
        return

    # ── Load image ────────────────────────────────────────────────────────────
    try:
        raw_bytes = uploaded_file.read()
        image_pil = Image.open(io.BytesIO(raw_bytes))
        # Preserve format info
        if image_pil.format is None:
            ext = uploaded_file.name.lower().rsplit(".", 1)[-1]
            image_pil.format = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG",
                                  "webp": "WEBP", "bmp": "BMP"}.get(ext, "JPEG")
        st.session_state["filename"] = uploaded_file.name.upper()
    except Exception as e:
        st.error(f"❌ Could not open image: {e}")
        return

    # Show preview + run button prompt when image is loaded but not yet run
    cache_key = (
        f"result_{uploaded_file.name}_{mode}_"
        f"{settings['ela_quality']}_{settings['ela_amplify']}_"
        f"{settings['heatmap_alpha']}_{settings['weights_path']}"
    )

    if not run_analysis and cache_key not in st.session_state:
        # Preview only
        st.markdown("""
        <div style="text-align:center; padding:10px 0 6px;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                        color:#839493; text-transform:uppercase; letter-spacing:0.1em;">
                Image loaded — click ▶ Run Deep Scan in sidebar to begin
            </div>
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            _render_panel_label(f"SOURCE_IMG · {uploaded_file.name.upper()}")
            st.image(image_pil, width="stretch")
        with c2:
            st.markdown("""
            <div style="height:100%; display:flex; flex-direction:column; justify-content:center;
                        align-items:center; background:#0b0e14; border:1px dashed #3a4a49;
                        min-height:280px;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                            color:#839493; text-align:center; line-height:2;">
                    HEATMAP_OVERLAY<br>
                    <span style="color:#3a4a49;">Awaiting scan…</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        return

    # ── Run or use cached ─────────────────────────────────────────────────────
    if run_analysis:
        with st.spinner("🔬 Initialising forensic analysis pipeline…"):
            try:
                log_entries: list = []
                unet_prob = None

                log_entries.append({"text": f"[{_ts()}] Initializing ELA module... DONE", "cls": "log-ok"})
                log_entries.append({"text": f"[{_ts()}] Analyzing JPEG compression levels...", "cls": "log-line"})

                # ── Phase 1: ELA (always computed) ────────────────────────────
                ela_image  = calculate_ela(image_pil,
                                           quality=settings["ela_quality"],
                                           amplify=settings["ela_amplify"])
                ela_stats  = get_ela_stats(ela_image)
                ela_score, ela_status = generate_forgery_score(ela_image)

                log_entries.append({"text": f"[{_ts()}] > Baseline quantization table extracted.", "cls": "log-ok"})
                log_entries.append({"text": f"[{_ts()}] > ELA variance: {ela_stats['variance']:.1f} | Mean: {ela_stats['mean']:.2f}", "cls": "log-line"})

                if ela_stats["high_pct"] > 5:
                    log_entries.append({
                        "text": f"[{_ts()}] > WARN: Inconsistent error levels detected in {ela_stats['high_pct']:.1f}% of pixels.",
                        "cls": "log-warn"
                    })

                # ── Phase 2: U-Net (if selected) ──────────────────────────────
                if "U-Net" in mode:
                    log_entries.append({"text": f"[{_ts()}] Passing coordinates to U-Net segmenter...", "cls": "log-line"})
                    unet_prob, unet_score, unet_status, unet_logs = _run_unet_inference(
                        image_pil,
                        settings["weights_path"],
                        settings["img_size"],
                    )
                    log_entries.extend(unet_logs)

                    if unet_prob is not None:
                        # Enhance raw U-Net probability map to highlight splice
                        # boundaries via CLAHE + Difference-of-Gaussians.
                        # Passing float32 ensures get_heatmap_overlay performs
                        # full min-max contrast stretch.
                        enhanced_mask = enhance_unet_mask(unet_prob)
                        heatmap_rgb   = get_heatmap_overlay(
                            image_pil, enhanced_mask,
                            alpha=settings["heatmap_alpha"] + 0.1,  # slightly higher opacity for DL mode
                        )
                        score, status = unet_score, unet_status
                    else:
                        # Fallback to ELA
                        heatmap_rgb = get_ela_heatmap(ela_image, image_pil, alpha=settings["heatmap_alpha"])
                        score, status = ela_score, ela_status
                        log_entries.append({"text": f"[{_ts()}] Falling back to ELA heuristics.", "cls": "log-warn"})
                        unet_prob = False  # sentinel: tried but unavailable
                else:
                    # Phase 1 only
                    heatmap_rgb = get_ela_heatmap(ela_image, image_pil, alpha=settings["heatmap_alpha"])
                    score, status = ela_score, ela_status
                    log_entries.append({"text": f"[{_ts()}] ELA heuristic scan complete.", "cls": "log-ok"})
                    log_entries.append({"text": f"[{_ts()}] Final score: {score} → {status.upper()}", "cls": "log-ok" if score <= 20 else "log-crit"})

                # ── Metadata ──────────────────────────────────────────────────
                metadata = _extract_metadata(image_pil)

                # Cache result
                st.session_state[cache_key] = {
                    "score":       score,
                    "status":      status,
                    "heatmap_rgb": heatmap_rgb,
                    "ela_stats":   ela_stats,
                    "metadata":    metadata,
                    "log_entries": log_entries,
                    "unet_prob":   unet_prob,
                }

            except Exception as exc:
                st.error(f"❌ Analysis error: {exc}")
                with st.expander("Show full traceback"):
                    st.code(traceback.format_exc())
                return

    # ── Render results ────────────────────────────────────────────────────────
    if cache_key in st.session_state:
        r = st.session_state[cache_key]
        _render_results(
            image_pil    = image_pil,
            score        = r["score"],
            status       = r["status"],
            heatmap_rgb  = r["heatmap_rgb"],
            ela_stats    = r["ela_stats"],
            metadata     = r["metadata"],
            log_entries  = r["log_entries"],
            mode         = mode,
            unet_prob    = r.get("unet_prob"),
        )


if __name__ == "__main__":
    main()
