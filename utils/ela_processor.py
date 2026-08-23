"""
utils/ela_processor.py — ELA + Heatmap + Scoring Engine
=========================================================
All image-forensics helper functions used by both app.py (Phase 1 heuristics)
and the U-Net inference path (Phase 2 heatmap blending).

Functions
---------
calculate_ela()         — Pillow-based Error Level Analysis
get_heatmap_overlay()   — OpenCV JET colormap blend over original image
generate_forgery_score() — Rule-based 0-100 score + label
get_ela_stats()         — Extended ELA statistical metrics dict
"""

import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
from typing import Tuple


# ─── Constants ────────────────────────────────────────────────────────────────

# Thresholds for ELA-based score → label mapping
SCORE_AUTHENTIC  = 20   # 0–20   → Authentic
SCORE_SUSPICIOUS = 50   # 21–50  → Suspicious
                         # 51–100 → Spliced


# ─── Core ELA ─────────────────────────────────────────────────────────────────

def calculate_ela(
    image: Image.Image,
    quality: int   = 90,
    amplify: float = 20.0,
) -> Image.Image:
    """
    Perform Error Level Analysis (ELA) on a PIL Image.

    ELA identifies regions where the JPEG compression error level deviates
    significantly from surrounding areas. In a spliced image, the inserted
    patch was typically saved independently at a different quality level;
    when the composite image is re-saved, the patched region shows a higher
    (or different) residual error than the authentic background.

    Process
    -------
    1. Re-compress the image at ``quality`` as JPEG into an in-memory buffer.
    2. Compute pixel-wise absolute difference: |original − recompressed|.
    3. Amplify the difference by ``amplify`` and clip to [0, 255].
    4. Return as an RGB PIL Image.

    Parameters
    ----------
    image   : PIL Image — input image (any mode; internally converted to RGB)
    quality : int       — JPEG recompression quality  (default 90)
    amplify : float     — difference amplification factor (default 20)

    Returns
    -------
    ela_pil : PIL Image (RGB) — amplified ELA difference image
    """
    # Normalise to RGB so we always get 3 channels
    rgb = image.convert("RGB")

    # Re-save to an in-memory buffer at the target quality
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    # Pixel-wise absolute difference
    ela_pil = ImageChops.difference(rgb, recompressed)

    # Amplify so subtle differences become visible
    orig_arr   = np.array(rgb,          dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)
    diff_arr   = np.abs(orig_arr - recomp_arr) * amplify
    diff_arr   = np.clip(diff_arr, 0, 255).astype(np.uint8)

    ela_pil = Image.fromarray(diff_arr, mode="RGB")

    # Light sharpening to accentuate region boundaries
    ela_pil = ela_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2))

    return ela_pil


def get_ela_stats(ela_image: Image.Image) -> dict:
    """
    Compute a comprehensive set of statistics from an ELA image.

    Parameters
    ----------
    ela_image : PIL Image (RGB) — output of calculate_ela()

    Returns
    -------
    dict with keys: mean, max, std, variance, p95, p99, high_pct
    """
    arr  = np.array(ela_image, dtype=np.float32)
    gray = arr.mean(axis=2)          # convert to single-channel luminance

    high_thresh = 15.0               # raw-pixel threshold for "high" ELA
    return {
        "mean":     float(gray.mean()),
        "max":      float(gray.max()),
        "std":      float(gray.std()),
        "variance": float(np.var(gray)),
        "p95":      float(np.percentile(gray, 95)),
        "p99":      float(np.percentile(gray, 99)),
        "high_pct": float((gray > high_thresh).mean() * 100),
    }


# ─── Heatmap ──────────────────────────────────────────────────────────────────

def get_heatmap_overlay(
    original_pil: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a grayscale mask as a colour heatmap over the original image.

    The heatmap is applied with OpenCV's COLORMAP_JET (blue=low, red=high),
    matching the reference UI design spec.

    Parameters
    ----------
    original_pil : PIL Image  — the source image shown beside the heatmap
    mask         : np.ndarray — float32 [0,1] or uint8 [0,255] grayscale mask
    alpha        : float      — heatmap opacity weight  (default 0.45)
    colormap     : int        — OpenCV colormap constant (default JET)

    Returns
    -------
    np.ndarray (H, W, 3) uint8 RGB — blended overlay ready for st.image()
    """
    # Convert PIL → BGR for OpenCV
    orig_rgb  = np.array(original_pil.convert("RGB"), dtype=np.uint8)
    orig_bgr  = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    # Always normalize to [0,1] float first for proper contrast stretch
    if mask.dtype == np.uint8:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)

    m_min, m_max = mask_f.min(), mask_f.max()
    if m_max > m_min:
        mask_normalized = ((mask_f - m_min) / (m_max - m_min) * 255).astype(np.uint8)
    else:
        mask_normalized = np.zeros_like(mask_f, dtype=np.uint8)

    # Resize mask to match original if needed
    h, w = orig_bgr.shape[:2]
    if mask_normalized.shape[:2] != (h, w):
        mask_normalized = cv2.resize(mask_normalized, (w, h), interpolation=cv2.INTER_LINEAR)

    # Apply colormap and blend
    heatmap_bgr  = cv2.applyColorMap(mask_normalized, colormap)
    blended_bgr  = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap_bgr, alpha, 0)

    # Return as RGB for Streamlit / PIL
    return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)


def enhance_unet_mask(prob, confidence_threshold=0.50):
    """Threshold-gated U-Net mask enhancer. Suppresses natural texture noise."""
    import numpy as _np
    import cv2 as _cv2

    if prob.max() <= 0:
        return prob
    prob = prob.astype(_np.float32)

    # Gate: zero pixels below threshold to suppress authentic-image noise floor
    gated = _np.where(prob >= confidence_threshold, prob, 0.0)

    if gated.max() <= 0:
        # Nothing crossed threshold -> cool uniform baseline (Authentic reading)
        return _np.full_like(prob, 0.1, dtype=_np.float32)

    # Smooth: large Gaussian merges nearby anomaly pixels into coherent blobs
    h, w = gated.shape
    k_size = max(11, int(min(h, w) * 0.04))
    if k_size % 2 == 0:
        k_size += 1
    smoothed = _cv2.GaussianBlur(gated, (k_size, k_size), k_size / 6)

    # Close: fill small holes inside blobs for solid region appearance
    kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (7, 7))
    sm_u8 = (smoothed * 255).astype(_np.uint8)
    closed = _cv2.morphologyEx(sm_u8, _cv2.MORPH_CLOSE, kernel)
    smoothed = closed.astype(_np.float32) / 255.0

    # Stretch survivors to full [0, 1] dynamic range for JET colormap
    s_min, s_max = smoothed.min(), smoothed.max()
    if s_max > s_min:
        enhanced = (smoothed - s_min) / (s_max - s_min)
    else:
        enhanced = smoothed
    return enhanced.astype(_np.float32)


def get_ela_heatmap(ela_image: Image.Image, original_pil: Image.Image, alpha: float = 0.5) -> np.ndarray:
    """
    Convert an ELA PIL image into a JET heatmap overlay.

    Convenience wrapper: converts ELA → grayscale → get_heatmap_overlay().
    """
    ela_gray = np.array(ela_image.convert("L"), dtype=np.uint8)
    return get_heatmap_overlay(original_pil, ela_gray, alpha=alpha)


# ─── Scoring ──────────────────────────────────────────────────────────────────

def generate_forgery_score(ela_image: Image.Image) -> Tuple[int, str]:
    """
    Produce a heuristic forgery score (0–100) and a label string.

    Scoring components (transparent heuristics):
      • 60 % weight — variance of the ELA luminance (higher = more suspicious)
      • 20 % weight — 99th-percentile ELA spike
      • 20 % weight — percentage of pixels with high ELA difference

    Classification thresholds (prototype / demo only):
      0–20   → "Authentic"
      21–50  → "Suspicious"
      51–100 → "Spliced"

    Parameters
    ----------
    ela_image : PIL Image — output of calculate_ela()

    Returns
    -------
    (score: int, label: str)
    """
    stats = get_ela_stats(ela_image)

    # --- Component A: Variance contribution (0–60 pts) ---
    # Variance normalised to [0,1] using empirical upper bound of 1500
    var_norm  = min(stats["variance"] / 1500.0, 1.0)
    score_var = int(var_norm * 60)

    # --- Component B: 99th-percentile spike (0–20 pts) ---
    # p99 normalised to empirical upper bound of 80
    p99_norm  = min(stats["p99"] / 80.0, 1.0)
    score_p99 = int(p99_norm * 20)

    # --- Component C: High-diff pixel percentage (0–20 pts) ---
    # high_pct normalised to upper bound of 30%
    pct_norm  = min(stats["high_pct"] / 30.0, 1.0)
    score_pct = int(pct_norm * 20)

    total = min(score_var + score_p99 + score_pct, 100)

    if total <= SCORE_AUTHENTIC:
        label = "Authentic"
    elif total <= SCORE_SUSPICIOUS:
        label = "Suspicious"
    else:
        label = "Spliced"

    return total, label
