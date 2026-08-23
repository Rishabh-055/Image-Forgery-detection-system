"""
analyzer.py — Core Image Analyzer Module
=========================================
Orchestrates all analysis modules and aggregates results
into a single unified analysis result dictionary.
"""

import io
import numpy as np
from PIL import Image
from typing import Optional

from utils.ela import perform_ela, get_ela_score
from utils.metadata import extract_metadata, get_metadata_score
from utils.scoring import (
    compute_compression_score,
    compute_other_signals_score,
    classify_result,
    generate_explanation,
)


# Maximum dimension for analysis copy (preserves original for display)
MAX_ANALYSIS_DIM = 2048


def prepare_image_for_analysis(image: Image.Image) -> Image.Image:
    """
    Prepare a working copy of the image for analysis.

    - Converts unusual modes (RGBA, P, CMYK, etc.) to RGB
    - Resizes very large images to MAX_ANALYSIS_DIM on the longest edge

    Args:
        image: Original PIL Image

    Returns:
        RGB PIL Image suitable for analysis
    """
    # Convert to RGB for uniform processing
    if image.mode != "RGB":
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode == "P":
            image = image.convert("RGBA").convert("RGB")
        else:
            image = image.convert("RGB")

    # Resize if too large
    w, h = image.size
    if max(w, h) > MAX_ANALYSIS_DIM:
        ratio = MAX_ANALYSIS_DIM / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    return image


def compute_noise_statistics(image_rgb: Image.Image) -> dict:
    """
    Compute basic image noise/quality statistics.

    Args:
        image_rgb: RGB PIL Image

    Returns:
        dict with noise-related metrics
    """
    arr = np.array(image_rgb, dtype=np.float32)

    # Laplacian-like edge detection for noise estimation
    # Simple approach: difference between pixel and its right/bottom neighbour
    diff_h = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean()
    diff_v = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean()
    local_variation = float((diff_h + diff_v) / 2)

    # Channel statistics
    channel_means = arr.mean(axis=(0, 1))
    channel_stds  = arr.std(axis=(0, 1))

    # Brightness and contrast
    gray = arr.mean(axis=2)
    brightness = float(gray.mean())
    contrast   = float(gray.std())

    # Colour balance check: large deviation between channels might be unusual
    channel_range = float(channel_means.max() - channel_means.min())

    return {
        "local_variation":  local_variation,
        "channel_means":    channel_means.tolist(),
        "channel_stds":     channel_stds.tolist(),
        "brightness":       brightness,
        "contrast":         contrast,
        "channel_range":    channel_range,
    }


def analyze_image(image: Image.Image, ela_quality: int = 90, ela_amplify: float = 15.0) -> dict:
    """
    Full pipeline: run all analysis modules and return aggregated results.

    Args:
        image:        Original PIL Image (may be any supported mode/size)
        ela_quality:  JPEG quality for ELA recompression
        ela_amplify:  ELA difference amplification factor

    Returns:
        result dict containing all sub-scores, visualisations, metadata, and
        the final forgery classification.
    """
    original_size = image.size
    original_mode = image.mode
    original_format = image.format  # May be None if loaded from bytes

    # --- Prepare analysis copy ---
    analysis_img = prepare_image_for_analysis(image)
    was_resized = (analysis_img.size != original_size)

    # ── 1. Error Level Analysis ──────────────────────────────────────────────
    ela_image, ela_raw_array, ela_stats = perform_ela(
        analysis_img, quality=ela_quality, amplify=ela_amplify
    )
    ela_score = get_ela_score(ela_stats)

    # ── 2. Metadata Analysis ─────────────────────────────────────────────────
    metadata = extract_metadata(image)          # Use original (has format info)
    metadata_score, metadata_reasons = get_metadata_score(metadata)

    # ── 3. Noise / Compression Analysis ─────────────────────────────────────
    noise_stats = compute_noise_statistics(analysis_img)
    compression_score, compression_reasons = compute_compression_score(
        image, noise_stats, ela_stats
    )

    # ── 4. Other Signals ─────────────────────────────────────────────────────
    other_score, other_reasons = compute_other_signals_score(
        image, analysis_img, ela_stats, noise_stats
    )

    # ── 5. Final Score & Classification ─────────────────────────────────────
    total_score = ela_score + metadata_score + compression_score + other_score
    classification, classification_color = classify_result(total_score)

    # ── 6. Human-readable Explanation ───────────────────────────────────────
    explanation = generate_explanation(
        total_score=total_score,
        classification=classification,
        ela_score=ela_score,
        ela_stats=ela_stats,
        metadata_score=metadata_score,
        metadata_reasons=metadata_reasons,
        compression_score=compression_score,
        compression_reasons=compression_reasons,
        other_score=other_score,
        other_reasons=other_reasons,
        metadata=metadata,
        was_resized=was_resized,
    )

    return {
        # Image info
        "original_size":        original_size,
        "original_mode":        original_mode,
        "original_format":      original_format,
        "analysis_size":        analysis_img.size,
        "was_resized":          was_resized,

        # ELA
        "ela_image":            ela_image,
        "ela_raw_array":        ela_raw_array,
        "ela_stats":            ela_stats,
        "ela_score":            ela_score,

        # Metadata
        "metadata":             metadata,
        "metadata_score":       metadata_score,
        "metadata_reasons":     metadata_reasons,

        # Compression / noise
        "noise_stats":          noise_stats,
        "compression_score":    compression_score,
        "compression_reasons":  compression_reasons,

        # Other
        "other_score":          other_score,
        "other_reasons":        other_reasons,

        # Final
        "total_score":          total_score,
        "classification":       classification,
        "classification_color": classification_color,
        "explanation":          explanation,

        # Working image (for side-by-side display)
        "analysis_image":       analysis_img,
    }
