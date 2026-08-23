"""
metadata.py — Image Metadata Extraction Module
===============================================
Extracts and interprets image metadata using Pillow (PIL).

Metadata provides contextual clues about image origin and editing history.
Missing metadata alone is NOT sufficient to classify an image as forged.
"""

from PIL import Image, ExifTags
from typing import Optional


# EXIF tag IDs for commonly interesting fields
_EXIF_TAG_MAP = {
    271:  "Camera Make",
    272:  "Camera Model",
    305:  "Software",
    306:  "DateTime",
    36867: "DateTimeOriginal",
    36868: "DateTimeDigitized",
    33434: "ExposureTime",
    33437: "FNumber",
    34855: "ISOSpeedRatings",
    37377: "ShutterSpeedValue",
    37378: "ApertureValue",
    37385: "Flash",
    37386: "FocalLength",
    41729: "SceneType",
    41730: "CFAPattern",
    42035: "LensMake",
    42036: "LensModel",
}

# Software names that are known editing tools
_EDITING_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "lightroom", "affinity", "paintshop",
    "pixelmator", "canva", "snapseed", "vsco", "fotor",
    "paint.net", "corel", "darktable", "rawtherapee",
    "capture one", "luminar", "sketchbook", "krita",
    "illustrator", "inkscape", "figma",
]


def extract_metadata(image: Image.Image) -> dict:
    """
    Extract all available metadata from a PIL Image object.

    Args:
        image: PIL Image object

    Returns:
        A dict with keys:
          "basic"   → dict of basic image properties
          "exif"    → dict of decoded EXIF tags (may be empty)
          "has_exif"→ bool
          "anomalies" → list of string descriptions of suspicious findings
    """
    metadata = {
        "basic": _extract_basic(image),
        "exif": {},
        "has_exif": False,
        "anomalies": [],
    }

    # --- Try to extract EXIF ---
    try:
        exif_data = image._getexif()  # Returns None for non-JPEG / EXIF-less images
        if exif_data:
            metadata["has_exif"] = True
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, f"Tag_{tag_id}")
                # Convert bytes to string if needed
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="replace").strip()
                    except Exception:
                        value = repr(value)
                metadata["exif"][tag_name] = value
    except (AttributeError, Exception):
        pass  # PNG, BMP, WEBP images may not have _getexif

    # Also try the newer getexif() API (Pillow >= 6.0)
    if not metadata["has_exif"]:
        try:
            exif_obj = image.getexif()
            if exif_obj:
                metadata["has_exif"] = True
                for tag_id, value in exif_obj.items():
                    tag_name = ExifTags.TAGS.get(tag_id, f"Tag_{tag_id}")
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8", errors="replace").strip()
                        except Exception:
                            value = repr(value)
                    metadata["exif"][tag_name] = value
        except Exception:
            pass

    # --- Detect anomalies ---
    metadata["anomalies"] = _detect_anomalies(metadata)

    return metadata


def _extract_basic(image: Image.Image) -> dict:
    """Extract basic non-EXIF image properties."""
    fmt = image.format if image.format else "Unknown"
    w, h = image.size
    mode = image.mode

    # DPI
    dpi = None
    try:
        dpi = image.info.get("dpi") or image.info.get("jfif_density")
    except Exception:
        pass

    return {
        "format":     fmt,
        "width":      w,
        "height":     h,
        "resolution": f"{w} × {h}",
        "color_mode": mode,
        "dpi":        str(dpi) if dpi else "Not Available",
        "file_size_approx": f"~{(w * h * _channels(mode)) // 1024} KB (uncompressed)",
    }


def _channels(mode: str) -> int:
    """Return approximate number of colour channels for a PIL mode string."""
    return {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4,
            "CMYK": 4, "YCbCr": 3, "LAB": 3, "HSV": 3, "I": 1, "F": 1}.get(mode, 3)


def _detect_anomalies(metadata: dict) -> list:
    """
    Inspect metadata for suspicious characteristics.

    Returns a list of human-readable anomaly description strings.
    """
    anomalies = []
    exif = metadata["exif"]
    basic = metadata["basic"]

    # 1. Editing software detected in EXIF
    software = str(exif.get("Software", "")).strip()
    if software:
        sw_lower = software.lower()
        for kw in _EDITING_SOFTWARE_KEYWORDS:
            if kw in sw_lower:
                anomalies.append(
                    f'Editing software detected in EXIF: "{software}"'
                )
                break

    # 2. DateTime field vs DateTimeOriginal mismatch
    dt_mod = exif.get("DateTime", "")
    dt_orig = exif.get("DateTimeOriginal", "")
    if dt_mod and dt_orig and dt_mod != dt_orig:
        anomalies.append(
            f"Modification datetime ({dt_mod}) differs from original capture datetime ({dt_orig})"
        )

    # 3. No EXIF on a JPEG (not necessarily forged, but worth noting)
    if basic.get("format") == "JPEG" and not metadata["has_exif"]:
        anomalies.append(
            "JPEG image has no EXIF metadata — original camera metadata may have been stripped"
        )

    # 4. CMYK or unusual colour mode
    if basic.get("color_mode") in ("CMYK", "P", "LAB"):
        anomalies.append(
            f"Unusual colour mode '{basic['color_mode']}' detected — may indicate processing pipeline"
        )

    return anomalies


def get_display_table(metadata: dict) -> list:
    """
    Build a list of (Property, Value) tuples for display in the UI.

    Args:
        metadata: dict returned by extract_metadata()

    Returns:
        List of (str, str) tuples
    """
    basic = metadata["basic"]
    exif  = metadata["exif"]

    rows = [
        ("Format",        basic.get("format", "Not Available")),
        ("Resolution",    basic.get("resolution", "Not Available")),
        ("Color Mode",    basic.get("color_mode", "Not Available")),
        ("DPI",           basic.get("dpi", "Not Available")),
        ("EXIF Data",     "Available ✓" if metadata["has_exif"] else "Not Available"),
        ("Camera Make",   str(exif.get("Make",        exif.get("Camera Make",  "Not Available")))),
        ("Camera Model",  str(exif.get("Model",       exif.get("Camera Model", "Not Available")))),
        ("Software",      str(exif.get("Software",    "Not Available"))),
        ("Date Taken",    str(exif.get("DateTimeOriginal", exif.get("DateTime", "Not Available")))),
        ("Date Modified", str(exif.get("DateTime",    "Not Available"))),
        ("Flash",         str(exif.get("Flash",        "Not Available"))),
        ("Focal Length",  str(exif.get("FocalLength",  "Not Available"))),
        ("ISO",           str(exif.get("ISOSpeedRatings", "Not Available"))),
    ]

    # Clean up "None" strings
    rows = [(k, v if v not in (None, "None", "") else "Not Available") for k, v in rows]
    return rows


def get_metadata_score(metadata: dict) -> tuple:
    """
    Score metadata anomalies (0–20 points).

    Returns (score: int, reasons: list[str]).
    """
    score = 0
    reasons = []
    anomalies = metadata.get("anomalies", [])

    for anomaly in anomalies:
        if "editing software" in anomaly.lower():
            score += 12
            reasons.append("Known editing software found in EXIF (+12)")
        elif "datetime" in anomaly.lower():
            score += 6
            reasons.append("Datetime mismatch between modification and capture (+6)")
        elif "no exif" in anomaly.lower() or "stripped" in anomaly.lower():
            score += 4
            reasons.append("EXIF metadata stripped from JPEG (+4)")
        elif "colour mode" in anomaly.lower():
            score += 3
            reasons.append("Unusual colour mode detected (+3)")
        else:
            score += 2
            reasons.append(f"Anomaly: {anomaly} (+2)")

    return min(score, 20), reasons
