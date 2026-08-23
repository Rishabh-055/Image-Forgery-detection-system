"""
scoring.py — Rule-Based Forgery Scoring Module
===============================================
Implements the transparent, rule-based scoring system used by
the Image Forgery Detection prototype.

Score breakdown (max 100):
  ELA Anomaly Score          0 – 50  (computed in ela.py)
  Metadata Anomaly Score     0 – 20  (computed in metadata.py)
  Compression/Noise Score    0 – 20  (computed here)
  Other Signals Score        0 – 10  (computed here)

Classification thresholds (prototype / demo only):
  0–30   → Likely Authentic
  31–60  → Suspicious
  61–100 → Likely Forged

These thresholds are heuristic estimates, NOT scientifically validated.
"""

import io
import math
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Compression / Noise Score  (0–20 pts)
# ─────────────────────────────────────────────────────────────────────────────

def compute_compression_score(
    original_image: Image.Image,
    noise_stats: dict,
    ela_stats: dict,
) -> tuple:
    """
    Estimate compression-related forgery signals (0–20 points).

    Signals used:
      - Standard deviation of ELA (high std → inconsistent compression → suspicious)
      - Local image variation (very uniform images can indicate heavy re-saves)
      - ELA 95th-percentile spike

    Args:
        original_image: Original PIL Image
        noise_stats:    Dict from analyzer.compute_noise_statistics()
        ela_stats:      Dict from ela.perform_ela()

    Returns:
        (score: int, reasons: list[str])
    """
    score = 0
    reasons = []

    # --- Signal 1: ELA standard deviation (0–8 pts) ---
    ela_std = ela_stats.get("ela_std", 0)
    if ela_std > 20:
        score += 8
        reasons.append(f"High ELA standard deviation ({ela_std:.1f}) — inconsistent recompression (+8)")
    elif ela_std > 10:
        score += 5
        reasons.append(f"Moderate ELA standard deviation ({ela_std:.1f}) — some regional variation (+5)")
    elif ela_std > 5:
        score += 2
        reasons.append(f"Low ELA standard deviation ({ela_std:.1f}) — mostly uniform compression (+2)")

    # --- Signal 2: ELA 95th-percentile spike (0–7 pts) ---
    p95 = ela_stats.get("ela_p95", 0)
    if p95 > 40:
        score += 7
        reasons.append(f"Very high ELA 95th-percentile ({p95:.1f}) — large bright spots in ELA (+7)")
    elif p95 > 20:
        score += 4
        reasons.append(f"Elevated ELA 95th-percentile ({p95:.1f}) — some bright ELA regions (+4)")
    elif p95 > 10:
        score += 1
        reasons.append(f"Mild ELA 95th-percentile ({p95:.1f}) (+1)")

    # --- Signal 3: Local variation analysis (0–5 pts) ---
    local_var = noise_stats.get("local_variation", 0)
    if local_var < 3.0:
        # Suspiciously smooth — may have been synthetically generated
        score += 3
        reasons.append(
            f"Very low local pixel variation ({local_var:.2f}) — image may be synthetically smooth (+3)"
        )
    elif local_var > 60:
        score += 5
        reasons.append(
            f"Very high local pixel variation ({local_var:.2f}) — may indicate heavy noise or compositing (+5)"
        )

    return min(score, 20), reasons


# ─────────────────────────────────────────────────────────────────────────────
# Other Signals Score  (0–10 pts)
# ─────────────────────────────────────────────────────────────────────────────

def compute_other_signals_score(
    original_image: Image.Image,
    analysis_image: Image.Image,
    ela_stats: dict,
    noise_stats: dict,
) -> tuple:
    """
    Compute miscellaneous image forensics signals (0–10 points).

    Signals:
      - Unusual aspect ratio
      - Colour channel imbalance
      - Very low or zero ELA (perfect copy → suspicious for a camera photo)
      - Unusually high image complexity vs ELA mean

    Args:
        original_image:  Original PIL Image (for format/size info)
        analysis_image:  RGB PIL Image used for analysis
        ela_stats:       ELA statistics dict
        noise_stats:     Noise statistics dict

    Returns:
        (score: int, reasons: list[str])
    """
    score = 0
    reasons = []

    w, h = analysis_image.size

    # --- Signal 1: Unusual aspect ratio (0–3 pts) ---
    ratio = w / h if h > 0 else 1.0
    # Very wide or very tall images outside normal camera/document ratios
    if ratio > 5.0 or ratio < 0.2:
        score += 3
        reasons.append(
            f"Unusual aspect ratio ({ratio:.2f}) — far outside typical photo/document ratios (+3)"
        )
    elif ratio > 3.0 or ratio < 0.33:
        score += 1
        reasons.append(
            f"Non-standard aspect ratio ({ratio:.2f}) (+1)"
        )

    # --- Signal 2: Colour channel imbalance (0–3 pts) ---
    channel_range = noise_stats.get("channel_range", 0)
    if channel_range > 60:
        score += 3
        reasons.append(
            f"Large colour channel imbalance ({channel_range:.1f}) — possibly colour-graded or composited (+3)"
        )
    elif channel_range > 30:
        score += 1
        reasons.append(
            f"Moderate colour channel imbalance ({channel_range:.1f}) (+1)"
        )

    # --- Signal 3: Very low ELA mean on a JPEG (0–4 pts) ---
    fmt = original_image.format or ""
    ela_mean = ela_stats.get("ela_mean", 999)
    if fmt == "JPEG" and ela_mean < 0.5:
        score += 4
        reasons.append(
            "Near-zero ELA on a JPEG — image may have been re-saved many times (multiple re-compressions) (+4)"
        )
    elif ela_mean > 30:
        score += 2
        reasons.append(
            f"Very high ELA mean ({ela_mean:.1f}) — strong overall compression inconsistency (+2)"
        )

    return min(score, 10), reasons


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_result(total_score: int) -> tuple:
    """
    Classify the total forgery score into a label and UI colour.

    Thresholds (prototype demo only):
      0–30   → Likely Authentic  (green)
      31–60  → Suspicious        (orange/yellow)
      61–100 → Likely Forged     (red)

    Returns:
        (label: str, color_hex: str)
    """
    if total_score <= 30:
        return "LIKELY AUTHENTIC", "#2ecc71"
    elif total_score <= 60:
        return "SUSPICIOUS", "#f39c12"
    else:
        return "LIKELY FORGED", "#e74c3c"


# ─────────────────────────────────────────────────────────────────────────────
# Human-Readable Explanation Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_explanation(
    total_score: int,
    classification: str,
    ela_score: int,
    ela_stats: dict,
    metadata_score: int,
    metadata_reasons: list,
    compression_score: int,
    compression_reasons: list,
    other_score: int,
    other_reasons: list,
    metadata: dict,
    was_resized: bool,
) -> str:
    """
    Generate a dynamic, human-readable explanation of the analysis result.

    The explanation is composed from actual computed values — it is NOT
    hardcoded per image.

    Returns:
        A multi-sentence string explanation.
    """
    parts = []

    # --- ELA summary ---
    ela_mean = ela_stats.get("ela_mean", 0)
    high_pct  = ela_stats.get("high_diff_pct", 0)
    ela_p99   = ela_stats.get("ela_p99", 0)

    if ela_score < 10:
        ela_desc = (
            f"The ELA analysis shows very low compression differences (mean: {ela_mean:.2f}, "
            f"high-diff pixels: {high_pct:.1f}%), which is consistent with an un-edited image or "
            "a PNG/screenshot where ELA is less meaningful."
        )
    elif ela_score < 25:
        ela_desc = (
            f"The ELA analysis reveals moderate compression variation (mean: {ela_mean:.2f}, "
            f"high-diff pixels: {high_pct:.1f}%). Some regions show elevated differences, "
            "which could indicate minor editing or multiple JPEG re-saves."
        )
    else:
        ela_desc = (
            f"The ELA analysis shows significant compression inconsistencies (mean: {ela_mean:.2f}, "
            f"high-diff pixels: {high_pct:.1f}%, 99th-percentile: {ela_p99:.1f}). "
            "Bright regions in the ELA visualisation may indicate areas of digital manipulation or splicing."
        )
    parts.append(ela_desc)

    # --- Metadata summary ---
    has_exif = metadata.get("has_exif", False)
    anomalies = metadata.get("anomalies", [])

    if metadata_score == 0 and not anomalies:
        meta_desc = (
            "No suspicious metadata signals were detected. "
            + ("EXIF data is present and shows no editing software traces." if has_exif
               else "EXIF data is not available for this image type, which is normal for PNGs and screenshots.")
        )
    else:
        anomaly_texts = "; ".join(anomalies) if anomalies else "minor irregularities"
        meta_desc = (
            f"Metadata analysis found the following signals: {anomaly_texts}. "
            f"This contributed {metadata_score} points to the overall score."
        )
    parts.append(meta_desc)

    # --- Compression summary ---
    if compression_score == 0:
        comp_desc = "Compression and noise characteristics appear consistent with normal camera output or standard image processing."
    else:
        comp_details = "; ".join(compression_reasons) if compression_reasons else "some compression irregularities"
        comp_desc = f"Compression analysis detected: {comp_details}."
    parts.append(comp_desc)

    # --- Other signals ---
    if other_score > 0 and other_reasons:
        other_desc = "Additional signals: " + "; ".join(other_reasons) + "."
        parts.append(other_desc)

    # --- Final verdict ---
    if classification == "LIKELY AUTHENTIC":
        verdict = (
            f"Based on the heuristic prototype rules, the image received a total score of "
            f"{total_score}/100 and has been classified as LIKELY AUTHENTIC. "
            "No strong evidence of digital manipulation was detected by this prototype."
        )
    elif classification == "SUSPICIOUS":
        verdict = (
            f"Based on the heuristic prototype rules, the image received a total score of "
            f"{total_score}/100 and has been classified as SUSPICIOUS. "
            "Some signals suggest possible editing, but the evidence is not conclusive."
        )
    else:
        verdict = (
            f"Based on the heuristic prototype rules, the image received a total score of "
            f"{total_score}/100 and has been classified as LIKELY FORGED. "
            "Multiple signals suggest significant digital manipulation."
        )
    parts.append(verdict)

    # --- Disclaimer ---
    parts.append(
        "⚠️ Disclaimer: This is a prototype heuristic system. Results should not be used "
        "as legal or conclusive evidence. A future version will incorporate CNN-based "
        "classification trained on the CASIA and MICC-F220 datasets."
    )

    if was_resized:
        parts.append(
            "ℹ️ Note: The image was resized for analysis to stay within processing limits. "
            "Display uses the original resolution."
        )

    return "\n\n".join(parts)
