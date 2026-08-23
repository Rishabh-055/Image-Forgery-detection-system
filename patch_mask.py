import re

with open('utils/ela_processor.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_fn = """def enhance_unet_mask(prob, confidence_threshold=0.50):
    \"\"\"Threshold-gated U-Net mask enhancer. Suppresses natural texture noise.\"\"\"
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

"""

content_new = re.sub(
    r'def enhance_unet_mask\(prob[^\n]*\n.*?(?=\ndef )',
    new_fn,
    content,
    flags=re.DOTALL
)

if content_new == content:
    print('ERROR: regex did not match')
else:
    with open('utils/ela_processor.py', 'w', encoding='utf-8') as f:
        f.write(content_new)
    print('OK: enhance_unet_mask replaced')
