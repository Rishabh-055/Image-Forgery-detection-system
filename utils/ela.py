"""
ela.py — Error Level Analysis (ELA) Module
==========================================
Implements ELA using Pillow by comparing the original image
with a recompressed version at a fixed JPEG quality level.

ELA highlights regions with unusually high compression differences,
which may indicate digital editing or splicing. It is a heuristic
technique and should not be used as conclusive proof of forgery.
"""

import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def perform_ela(image: Image.Image, quality: int = 90, amplify: float = 15.0) -> tuple:
    """
    Perform Error Level Analysis on a PIL Image.

    Args:
        image:    PIL Image object (original)
        quality:  JPEG recompression quality (default 90)
        amplify:  Amplification factor for the ELA difference visualization

    Returns:
        ela_image:   PIL Image of the amplified ELA difference (RGB)
        ela_array:   NumPy array of the raw absolute difference (float32)
        ela_stats:   Dict with statistical metrics from the ELA
    """
    # Convert to RGB to ensure consistent processing (handles RGBA, L, P, etc.)
    original_rgb = image.convert("RGB")

    # --- Recompress at the target quality ---
    buffer = io.BytesIO()
    original_rgb.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    # --- Pixel-level absolute difference ---
    orig_arr = np.array(original_rgb, dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)
    diff_arr = np.abs(orig_arr - recomp_arr)           # shape: (H, W, 3)

    # --- Per-channel amplification then clip ---
    ela_arr_amplified = np.clip(diff_arr * amplify, 0, 255).astype(np.uint8)

    # Build ELA PIL image
    ela_pil = Image.fromarray(ela_arr_amplified, mode="RGB")

    # Optional: apply a subtle sharpening to make regions more visible
    ela_pil = ela_pil.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

    # --- Statistics ---
    # Convert diff to grayscale-like magnitude for scalar stats
    diff_gray = diff_arr.mean(axis=2)                  # shape: (H, W)
    total_pixels = diff_gray.size
    high_threshold = 15.0                              # raw diff threshold for "high"

    ela_stats = {
        "ela_mean":          float(diff_gray.mean()),
        "ela_max":           float(diff_gray.max()),
        "ela_std":           float(diff_gray.std()),
        "ela_p95":           float(np.percentile(diff_gray, 95)),
        "ela_p99":           float(np.percentile(diff_gray, 99)),
        "high_diff_pct":     float((diff_gray > high_threshold).sum() / total_pixels * 100),
        "recomp_quality":    quality,
        "amplify_factor":    amplify,
    }

    return ela_pil, diff_arr, ela_stats


def generate_ela_heatmap(ela_array: np.ndarray) -> Image.Image:
    """
    Generate a colour-mapped heatmap from the raw ELA difference array.

    Uses a simple red-gradient colouring: high-difference pixels appear
    brighter/more saturated.

    Args:
        ela_array: Raw float32 difference array (H, W, 3)

    Returns:
        heatmap PIL Image in RGB mode
    """
    diff_gray = ela_array.mean(axis=2)                 # (H, W)
    max_val = diff_gray.max() if diff_gray.max() > 0 else 1.0
    norm = diff_gray / max_val                         # 0..1

    # Red-channel dominant heatmap
    r = np.clip(norm * 255, 0, 255).astype(np.uint8)
    g = np.clip((1 - norm) * 80, 0, 255).astype(np.uint8)
    b = np.zeros_like(r)

    heatmap_arr = np.stack([r, g, b], axis=2)
    return Image.fromarray(heatmap_arr, mode="RGB")


def get_ela_score(ela_stats: dict) -> int:
    """
    Convert ELA statistics to a sub-score (0–50 points).

    Scoring logic (transparent heuristics):
      - High mean ELA difference            → more suspicious
      - Large percentage of high-diff pixels → more suspicious
      - High 99th-percentile ELA            → more suspicious

    Returns an integer score in [0, 50].
    """
    score = 0

    # Component 1: Mean ELA difference (0–20 pts)
    mean = ela_stats["ela_mean"]
    if mean < 2.0:
        score += 0
    elif mean < 5.0:
        score += 8
    elif mean < 10.0:
        score += 15
    elif mean < 20.0:
        score += 20
    else:
        score += 20

    # Component 2: High-difference pixel percentage (0–20 pts)
    pct = ela_stats["high_diff_pct"]
    if pct < 1.0:
        score += 0
    elif pct < 5.0:
        score += 6
    elif pct < 15.0:
        score += 12
    elif pct < 30.0:
        score += 17
    else:
        score += 20

    # Component 3: 99th-percentile spike (0–10 pts)
    p99 = ela_stats["ela_p99"]
    if p99 < 10.0:
        score += 0
    elif p99 < 25.0:
        score += 4
    elif p99 < 50.0:
        score += 7
    else:
        score += 10

    return min(score, 50)
