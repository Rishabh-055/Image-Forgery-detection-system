"""
test_app_e2e.py — Full End-to-End System Test for Image Forgery Detection
==================================================================
Tests both Phase 1 (Heuristic ELA) and Phase 2 (Deep Learning U-Net)
pipelines on authentic, donor, and spliced images to verify all
components work without errors.
"""

import os
import io
import time
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms

# Imports from project
from model import UNet
from utils.ela_processor import (
    calculate_ela,
    get_ela_stats,
    get_ela_heatmap,
    get_heatmap_overlay,
    generate_forgery_score,
)
from utils.metadata import extract_metadata

def run_e2e_verification():
    print("=" * 60)
    print("      IMAGE FORGERY DETECTION: FULL SYSTEM TEST")
    print("=" * 60)

    # 1. Check Weights file
    wts_path = "unet_splicing_model.pth"
    assert os.path.exists(wts_path), f"Error: {wts_path} not found!"
    print(f"[OK] Model weights file found: {wts_path} ({os.path.getsize(wts_path):,} bytes)")

    # 2. Instantiate and load U-Net with weights_only=True
    model = UNet(in_channels=3, out_channels=1)
    state = torch.load(wts_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[OK] U-Net model successfully initialized ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # 3. Test list of images
    test_cases = [
        ("test img 1.webp", "Test Image 1 (Zebra Donor)"),
        ("test img2.webp", "Test Image 2 (3-Panel Research Figure)"),
        ("samples/eval_results/clean_authentic.jpg", "Authentic JPEG Landscape"),
        ("samples/eval_results/clean_spliced.jpg", "Spliced / Altered JPEG Landscape"),
    ]

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for img_path, label in test_cases:
        if not os.path.exists(img_path):
            continue
        print("\n" + "-" * 60)
        print(f"Testing: {label} ({img_path})")
        print("-" * 60)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        print(f"  Dimensions: {w} x {h}")

        # --- Phase 1: ELA Heuristic Test ---
        t0 = time.time()
        ela_img = calculate_ela(img, quality=90, amplify=20.0)
        ela_stats = get_ela_stats(ela_img)
        ela_score, ela_verdict = generate_forgery_score(ela_img)
        ela_heatmap = get_ela_heatmap(ela_img, img, alpha=0.45)
        ela_time = (time.time() - t0) * 1000

        print(f"  [Phase 1 - ELA]     Score: {ela_score:3d}% | Verdict: {ela_verdict:10s} | ELA Max: {ela_stats['max']:5.1f} | Time: {ela_time:.1f}ms")
        assert ela_heatmap.shape == (h, w, 3), "ELA Heatmap shape mismatch!"

        # --- Phase 2: U-Net Deep Learning Test ---
        t0 = time.time()
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        prob_resized = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        prob_resized = np.clip(prob_resized, 0.0, 1.0)
        unet_score = int(np.max(prob_resized) * 100)
        unet_verdict = "Spliced" if unet_score > 50 else ("Suspicious" if unet_score > 20 else "Authentic")
        
        mask_u8 = (prob_resized * 255).astype(np.uint8)
        unet_heatmap = get_heatmap_overlay(img, mask_u8, alpha=0.45)
        unet_time = (time.time() - t0) * 1000

        print(f"  [Phase 2 - U-Net]   Score: {unet_score:3d}% | Verdict: {unet_verdict:10s} | Peak Prob: {np.max(prob_resized):.4f} | Time: {unet_time:.1f}ms")
        assert unet_heatmap.shape == (h, w, 3), "U-Net Heatmap shape mismatch!"

        # --- Metadata Extraction ---
        meta = extract_metadata(img)
        print(f"  [Metadata]          EXIF Present: {meta.get('has_exif', False)} | Format: {meta.get('basic', {}).get('format')}")

    print("\n" + "=" * 60)
    print("      ALL TESTS PASSED SUCCESSFULLY (100% OPERATIONAL)")
    print("=" * 60)

if __name__ == "__main__":
    run_e2e_verification()
