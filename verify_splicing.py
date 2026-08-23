"""
verify_splicing.py — Rigorous evaluation script to test if ELA & Heatmaps
actually detect and highlight splicing on the user's test images.
"""

import os
from PIL import Image
import numpy as np
import cv2

from utils.ela_processor import (
    calculate_ela,
    get_ela_stats,
    get_ela_heatmap,
    generate_forgery_score,
    get_heatmap_overlay,
)

def evaluate_image(img_path, output_prefix="eval"):
    os.makedirs("samples/eval_results", exist_ok=True)
    img = Image.open(img_path).convert("RGB")
    
    # 1. Compute ELA
    ela_img = calculate_ela(img, quality=90, amplify=20.0)
    stats = get_ela_stats(ela_img)
    score, label = generate_forgery_score(ela_img)
    
    # 2. Compute Heatmap Overlay
    heatmap = get_ela_heatmap(ela_img, img, alpha=0.5)
    
    # 3. Save outputs
    ela_path = f"samples/eval_results/{output_prefix}_ela.png"
    hm_path = f"samples/eval_results/{output_prefix}_heatmap.png"
    orig_path = f"samples/eval_results/{output_prefix}_orig.png"
    
    img.save(orig_path)
    ela_img.save(ela_path)
    cv2.imwrite(hm_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    
    print(f"[{output_prefix}] Path: {img_path}")
    print(f"  Dimensions: {img.size}")
    print(f"  Score: {score}/100 | Label: {label}")
    print(f"  Stats: Mean={stats['mean']:.2f}, Max={stats['max']:.2f}, Variance={stats['variance']:.2f}, P99={stats['p99']:.2f}, HighPct={stats['high_pct']:.1f}%")
    print(f"  Saved: {orig_path}, {ela_path}, {hm_path}\n")
    return score, label, stats

print("=== 1. Testing User Test Images directly ===")
evaluate_image("test img 1.webp", "test_img_1")
evaluate_image("test img2.webp", "test_img_2")

print("=== 2. Cropping individual panels from test img2.webp ===")
img2 = Image.open("test img2.webp").convert("RGB")
w, h = img2.size
# test img2 has 3 equal columns: a, b, c
w_third = w // 3
panel_a = img2.crop((0, 0, w_third, h))               # Authentic background
panel_b = img2.crop((w_third, 0, 2 * w_third, h))       # Donor zebra
panel_c = img2.crop((2 * w_third, 0, w, h))           # Spliced composite

panel_a.save("samples/eval_results/panel_a_authentic.png")
panel_b.save("samples/eval_results/panel_b_donor.png")
panel_c.save("samples/eval_results/panel_c_spliced.png")

evaluate_image("samples/eval_results/panel_a_authentic.png", "panel_a_authentic")
evaluate_image("samples/eval_results/panel_b_donor.png", "panel_b_donor")
evaluate_image("samples/eval_results/panel_c_spliced.png", "panel_c_spliced")

print("=== 3. Creating a Realistic Spliced JPEG Test Case ===")
# Take panel A (mountain background), save as JPEG quality 95
# Take zebra from test img 1, save as JPEG quality 65
# Paste the zebra onto panel A, save composite as JPEG quality 90
bg = panel_a.copy()
zebra = Image.open("test img 1.webp").convert("RGB")
# resize zebra to fit in right half
zebra_crop = zebra.crop((int(zebra.width * 0.4), int(zebra.height * 0.35), int(zebra.width * 0.95), int(zebra.height * 0.85)))
zebra_crop = zebra_crop.resize((int(bg.width * 0.4), int(bg.height * 0.45)))

# Save with different compression levels to simulate splicing two distinct sources
import io
buf_bg = io.BytesIO()
bg.save(buf_bg, format="JPEG", quality=95)
buf_bg.seek(0)
bg_jpg = Image.open(buf_bg).convert("RGB")

buf_z = io.BytesIO()
zebra_crop.save(buf_z, format="JPEG", quality=60)
buf_z.seek(0)
zebra_jpg = Image.open(buf_z).convert("RGB")

# Paste zebra into background
spliced_comp = bg_jpg.copy()
paste_box = (int(bg.width * 0.45), int(bg.height * 0.4))
spliced_comp.paste(zebra_jpg, paste_box)

# Final save as JPEG quality 90 (standard spliced file on disk)
spliced_jpg_path = "samples/eval_results/synthetic_spliced.jpg"
spliced_comp.save(spliced_jpg_path, format="JPEG", quality=90)

evaluate_image(spliced_jpg_path, "synthetic_spliced_jpg")
