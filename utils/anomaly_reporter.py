"""
utils/anomaly_reporter.py — Forensic Anomaly Description Engine
================================================================
Converts raw numeric forensic signals (ELA stats, U-Net probability map,
EXIF metadata) into structured, human-readable anomaly descriptions shown
inside the dashboard's Forensic Anomaly Report panel.

Each anomaly is a dict:
    {
        "level":   "CRITICAL" | "WARNING" | "INFO" | "CLEAR",
        "tag":     short label (e.g. "ELA Variance Spike"),
        "detail":  full human-readable explanation,
        "icon":    emoji prefix
    }
"""

from __future__ import annotations
import numpy as np


# ─── ELA Thresholds (calibrated on CASIA v2 statistics) ──────────────────────

_ELA_VAR_AUTHENTIC  = 200.0     # typical variance for single-save authentic JPEG
_ELA_VAR_SUSPICIOUS = 600.0     # starts showing multi-history signals
_ELA_VAR_CRITICAL   = 1200.0    # strong splice evidence

_ELA_MEAN_AUTHENTIC  = 5.0
_ELA_MEAN_SUSPICIOUS = 12.0
_ELA_MEAN_CRITICAL   = 20.0

_ELA_P99_AUTHENTIC   = 30.0
_ELA_P99_SUSPICIOUS  = 60.0
_ELA_P99_CRITICAL    = 100.0

_ELA_HIGHPCT_AUTH    = 5.0      # % of pixels above high-diff threshold
_ELA_HIGHPCT_SUSP    = 15.0
_ELA_HIGHPCT_CRIT    = 30.0

# U-Net thresholds
_UNET_FRAC_SUSPICIOUS = 0.03    # 3% of pixels confidently tampered
_UNET_FRAC_CRITICAL   = 0.10   # 10%+ → strong splice region


def generate_anomaly_report(
    ela_stats: dict,
    unet_prob: "np.ndarray | None",
    metadata:  dict,
    score:     int,
    mode:      str,
) -> list[dict]:
    """
    Analyse all available forensic signals and produce a ranked list of
    human-readable anomaly descriptions.

    Parameters
    ----------
    ela_stats : dict   — output of ela_processor.get_ela_stats()
    unet_prob : ndarray or None — raw float32 U-Net probability map, or None
    metadata  : dict   — output of extract_metadata() / inline extractor
    score     : int    — final aggregated forgery score (0-100)
    mode      : str    — "Heuristic (ELA)" or "Deep Learning (U-Net)"

    Returns
    -------
    list of anomaly dicts ordered by severity (CRITICAL → WARNING → INFO → CLEAR)
    """
    findings: list[dict] = []

    # ── 1. ELA Compression History Anomalies ─────────────────────────────────
    variance  = ela_stats.get("variance",  0.0)
    mean_val  = ela_stats.get("mean",      0.0)
    p99       = ela_stats.get("p99",       0.0)
    high_pct  = ela_stats.get("high_pct",  0.0)

    # 1a. Variance (primary ELA signal)
    if variance >= _ELA_VAR_CRITICAL:
        findings.append({
            "level": "CRITICAL",
            "tag":   "JPEG Compression History Mismatch",
            "icon":  "🔴",
            "detail": (
                f"ELA luminance variance is **{variance:.1f}** — significantly above the authentic "
                f"baseline of ~{_ELA_VAR_AUTHENTIC:.0f}. This indicates regions of the image have "
                "different JPEG compression histories, a hallmark of image splicing where a donor "
                "patch was saved at a different quality factor than the host image."
            ),
        })
    elif variance >= _ELA_VAR_SUSPICIOUS:
        findings.append({
            "level": "WARNING",
            "tag":   "Elevated JPEG Compression Variance",
            "icon":  "🟡",
            "detail": (
                f"ELA variance ({variance:.1f}) exceeds the typical authentic image baseline "
                f"(~{_ELA_VAR_AUTHENTIC:.0f}). Moderate compression history inconsistency detected. "
                "This may indicate light editing, JPEG recompression from a messaging app, or "
                "a low-intensity splice."
            ),
        })
    else:
        findings.append({
            "level": "CLEAR",
            "tag":   "Uniform JPEG Compression Detected",
            "icon":  "🟢",
            "detail": (
                f"ELA variance ({variance:.1f}) is within the authentic range (<{_ELA_VAR_SUSPICIOUS:.0f}). "
                "The image appears to have a single, consistent JPEG compression history across "
                "all 8×8 DCT blocks — consistent with an unedited photograph."
            ),
        })

    # 1b. Mean ELA luminance
    if mean_val >= _ELA_MEAN_CRITICAL:
        findings.append({
            "level": "CRITICAL",
            "tag":   "High ELA Mean Luminance",
            "icon":  "🔴",
            "detail": (
                f"The average ELA residual brightness across all pixels is **{mean_val:.2f}** "
                f"(critical threshold: {_ELA_MEAN_CRITICAL}). A high mean indicates that the "
                "image globally did not reach compression equilibrium upon re-save, suggesting "
                "multiple saves at differing quality levels or heavy post-processing."
            ),
        })
    elif mean_val >= _ELA_MEAN_SUSPICIOUS:
        findings.append({
            "level": "WARNING",
            "tag":   "Above-Average ELA Mean Residual",
            "icon":  "🟡",
            "detail": (
                f"ELA mean luminance ({mean_val:.2f}) is moderately elevated. Some image regions "
                "have not fully reached JPEG equilibrium. This may reflect light social-media "
                "recompression (e.g., WhatsApp, Telegram) rather than deliberate tampering."
            ),
        })

    # 1c. 99th percentile spike
    if p99 >= _ELA_P99_CRITICAL:
        findings.append({
            "level": "CRITICAL",
            "tag":   "Extreme Local Error Spike (P99)",
            "icon":  "🔴",
            "detail": (
                f"The 99th-percentile ELA residual is **{p99:.1f}** — far above the authentic "
                f"ceiling of ~{_ELA_P99_AUTHENTIC:.0f}. Extreme localized spikes indicate isolated "
                "regions with fundamentally different compression characteristics — consistent with "
                "a donor patch boundary where pixel quantization tables conflict."
            ),
        })
    elif p99 >= _ELA_P99_SUSPICIOUS:
        findings.append({
            "level": "WARNING",
            "tag":   "Localized ELA Spike Detected (P99)",
            "icon":  "🟡",
            "detail": (
                f"99th-percentile ELA value ({p99:.1f}) suggests isolated areas with higher-than-"
                "expected compression residuals. Edge regions, over-sharpened details, or light "
                "retouching can produce similar signals."
            ),
        })

    # 1d. High-difference pixel fraction
    if high_pct >= _ELA_HIGHPCT_CRIT:
        findings.append({
            "level": "CRITICAL",
            "tag":   "Widespread Compression Anomaly",
            "icon":  "🔴",
            "detail": (
                f"**{high_pct:.1f}%** of all pixels show abnormally high ELA difference values. "
                "A widespread pattern (>{_ELA_HIGHPCT_CRIT:.0f}%) across the spatial domain "
                "suggests either large-area manipulation or multiple spliced regions distributed "
                "across the image."
            ),
        })
    elif high_pct >= _ELA_HIGHPCT_SUSP:
        findings.append({
            "level": "WARNING",
            "tag":   "Scattered High-Difference Pixels",
            "icon":  "🟡",
            "detail": (
                f"{high_pct:.1f}% of pixels exceed the high-difference threshold. Scattered "
                "anomalies may correspond to fine-detail retouching (hair, text edges) or "
                "JPEG blocking artefacts at format conversion boundaries."
            ),
        })
    else:
        findings.append({
            "level": "CLEAR",
            "tag":   "No Widespread Pixel Anomalies",
            "icon":  "🟢",
            "detail": (
                f"Only {high_pct:.1f}% of pixels show elevated ELA residuals. The compression "
                "error distribution is spatially uniform — no concentrated anomaly patches detected."
            ),
        })

    # ── 2. U-Net Deep Learning Findings ──────────────────────────────────────
    if unet_prob is not None and "U-Net" in mode:
        confident_mask   = unet_prob[unet_prob > 0.55]
        tampered_frac    = len(confident_mask) / max(unet_prob.size, 1)
        peak_prob        = float(unet_prob.max())

        if tampered_frac >= _UNET_FRAC_CRITICAL:
            findings.append({
                "level": "CRITICAL",
                "tag":   "U-Net: Concentrated Tampered Region Detected",
                "icon":  "🔴",
                "detail": (
                    f"The U-Net segmentation network flagged **{tampered_frac*100:.1f}%** of image "
                    f"pixels with tamper confidence >55% (peak: {peak_prob:.3f}). A concentrated, "
                    "spatially coherent high-probability region was isolated — consistent with a "
                    "donor patch pasted at specific image coordinates. Refer to the heatmap overlay "
                    "for exact spatial localization."
                ),
            })
        elif tampered_frac >= _UNET_FRAC_SUSPICIOUS:
            findings.append({
                "level": "WARNING",
                "tag":   "U-Net: Moderate Anomaly Region",
                "icon":  "🟡",
                "detail": (
                    f"U-Net identified {tampered_frac*100:.1f}% of pixels with elevated tamper "
                    f"probability (peak: {peak_prob:.3f}). The detected region is modest in size "
                    "and may correspond to light retouching, colour correction, or a small "
                    "inserted object."
                ),
            })
        else:
            findings.append({
                "level": "CLEAR",
                "tag":   "U-Net: No Significant Tampered Region",
                "icon":  "🟢",
                "detail": (
                    f"U-Net confident pixel coverage is only {tampered_frac*100:.2f}% "
                    f"(peak probability: {peak_prob:.3f}). No spatially coherent anomaly "
                    "region was detected by the neural segmentation model — the image "
                    "spatial feature distribution is consistent with authentic content."
                ),
            })

        # Spatial distribution (is the anomaly localized or scattered?)
        if tampered_frac >= 0.005:
            # Compute how clustered the anomaly is
            binary_map = (unet_prob > 0.55).astype(np.uint8)
            rows_hit   = binary_map.any(axis=1).sum()
            cols_hit   = binary_map.any(axis=0).sum()
            row_pct    = rows_hit / max(unet_prob.shape[0], 1) * 100
            col_pct    = cols_hit / max(unet_prob.shape[1], 1) * 100
            if row_pct < 40 and col_pct < 40:
                findings.append({
                    "level": "WARNING",
                    "tag":   "Spatially Localized Anomaly (Likely Donor Patch)",
                    "icon":  "🟡",
                    "detail": (
                        f"The anomalous region spans approximately {row_pct:.0f}% of image rows "
                        f"and {col_pct:.0f}% of columns — a concentrated, localized pattern strongly "
                        "consistent with a single inserted donor patch rather than global image "
                        "degradation or recompression."
                    ),
                })
            else:
                findings.append({
                    "level": "INFO",
                    "tag":   "Diffuse Anomaly Pattern",
                    "icon":  "🔵",
                    "detail": (
                        f"The anomalous region spans {row_pct:.0f}% of rows and {col_pct:.0f}% of "
                        "columns — a diffuse distribution more consistent with global recompression, "
                        "aggressive JPEG artefacts, or multiple small edits rather than a single "
                        "large donor patch."
                    ),
                })

    # ── 3. Metadata Forensic Findings ─────────────────────────────────────────
    exif         = metadata.get("exif", {})
    has_exif     = metadata.get("has_exif", False)
    sw_field     = exif.get("Software", "")
    editing_sw   = ["photoshop", "gimp", "lightroom", "pixelmator",
                    "canva", "paint", "corel", "affinity", "snapseed"]
    sw_detected  = any(s in sw_field.lower() for s in editing_sw) if sw_field else False

    if not has_exif:
        findings.append({
            "level": "WARNING",
            "tag":   "EXIF Metadata Absent",
            "icon":  "🟡",
            "detail": (
                "No EXIF metadata was found in this image. Authentic camera photographs "
                "always embed EXIF (camera model, capture timestamp, sensor info). Missing "
                "EXIF can indicate: (1) deliberate stripping by an editing tool to hide "
                "modification history, (2) a screenshot or screen-recorded image, or "
                "(3) saving via a messaging app that strips metadata."
            ),
        })
    else:
        findings.append({
            "level": "CLEAR",
            "tag":   "EXIF Metadata Present",
            "icon":  "🟢",
            "detail": (
                f"EXIF metadata is present ({len(exif)} fields). Camera model: "
                f"'{exif.get('Model', 'Unknown')}'. Original capture time: "
                f"'{exif.get('DateTimeOriginal', 'Not embedded')}'. "
                "Presence of EXIF does not guarantee authenticity but reduces suspicion "
                "of deliberate metadata scrubbing."
            ),
        })

    if sw_detected:
        findings.append({
            "level": "WARNING",
            "tag":   "Image Editing Software Signature Detected",
            "icon":  "🟡",
            "detail": (
                f"The EXIF 'Software' field contains: **\"{sw_field}\"** — a known image "
                "editing application. While many photographers legitimately use editing tools "
                "for colour correction and exposure adjustment, this confirms the image was "
                "opened and re-saved in an editing environment, which is a prerequisite for "
                "deliberate pixel manipulation."
            ),
        })

    # Camera model check
    camera = exif.get("Model", "")
    if has_exif and not camera:
        findings.append({
            "level": "INFO",
            "tag":   "Camera Model Field Empty",
            "icon":  "🔵",
            "detail": (
                "EXIF data is present but the Camera Model field is empty. Some editing "
                "tools strip specific tags while leaving others intact. This is a minor "
                "soft-indicator of post-processing."
            ),
        })

    # Timestamp check
    dt_orig = exif.get("DateTimeOriginal", "")
    dt_mod  = exif.get("DateTime", "")
    if dt_orig and dt_mod and dt_orig != dt_mod:
        findings.append({
            "level": "WARNING",
            "tag":   "Capture / Modification Timestamp Mismatch",
            "icon":  "🟡",
            "detail": (
                f"EXIF DateTimeOriginal ('{dt_orig}') differs from DateTime ('{dt_mod}'). "
                "This indicates the image was modified after its original capture — "
                "consistent with post-processing or editing."
            ),
        })

    # ── 4. Format-level Findings ──────────────────────────────────────────────
    img_format = metadata.get("basic", {}).get("format", "Unknown")
    if img_format == "PNG":
        findings.append({
            "level": "INFO",
            "tag":   "Lossless PNG Format — ELA Reliability Reduced",
            "icon":  "🔵",
            "detail": (
                "This image is in PNG (lossless) format. ELA relies on JPEG's lossy DCT "
                "quantization to create compression history artefacts. PNG has no compression "
                "quality factor, so ELA analysis produces uniformly high residuals that do "
                "not reliably indicate tampering. The U-Net analysis is more meaningful for "
                "this image format."
            ),
        })
    elif img_format == "WEBP":
        findings.append({
            "level": "INFO",
            "tag":   "WebP Format — Partial ELA Applicability",
            "icon":  "🔵",
            "detail": (
                "WebP uses a hybrid codec (VP8 for lossy, VP8L for lossless). ELA is partially "
                "applicable to lossy WebP images since they share DCT-like compression concepts, "
                "but calibration differs from JPEG. Interpret ELA scores with moderate caution "
                "for WebP files."
            ),
        })

    # ── Sort: CRITICAL first, then WARNING, INFO, CLEAR ──────────────────────
    order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2, "CLEAR": 3}
    findings.sort(key=lambda f: order.get(f["level"], 4))

    return findings


def format_anomaly_html(findings: list[dict]) -> str:
    """
    Render findings as a styled HTML report for st.markdown(unsafe_allow_html=True).
    """
    import html as _html
    rows = []
    for f in findings:
        level_colours = {
            "CRITICAL": ("#FF4B4B", "#2A1010"),
            "WARNING":  ("#FFA500", "#221A08"),
            "INFO":     ("#00BFFF", "#08141F"),
            "CLEAR":    ("#00C853", "#08201A"),
        }
        text_c, bg_c = level_colours.get(f["level"], ("#AAAAAA", "#1A1A1A"))
        icon   = _html.escape(f.get("icon", ""))
        tag    = _html.escape(f.get("tag", ""))
        detail = f.get("detail", "")   # allow **bold** markdown-style — rendered as-is

        # Convert **text** to <b>text</b> for inline bold
        import re
        detail_html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', _html.escape(detail))

        rows.append(f"""
<div style="
    background:{bg_c};
    border-left: 4px solid {text_c};
    border-radius: 4px;
    padding: 10px 14px;
    margin-bottom: 8px;
">
  <div style="color:{text_c}; font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:700; letter-spacing:0.05em; margin-bottom:4px;">
    {icon} {tag} &nbsp;<span style="opacity:0.6; font-weight:400;">[{f['level']}]</span>
  </div>
  <div style="color:#C8D6E8; font-size:0.84rem; line-height:1.55;">
    {detail_html}
  </div>
</div>""")

    return "\n".join(rows)
