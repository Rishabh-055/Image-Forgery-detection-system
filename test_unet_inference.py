"""
test_unet_inference.py — Test Phase 2 U-Net Inference on Real and Test Images
"""

import os
from PIL import Image
import torch
import numpy as np
import cv2
from torchvision import transforms

from model import UNet
from utils.ela_processor import get_heatmap_overlay

def test_inference():
    print("=== Testing U-Net Model Weights ===")
    model = UNet(in_channels=3, out_channels=1)
    state = torch.load("unet_splicing_model.pth", map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded successfully with weights_only=True. Params:", sum(p.numel() for p in model.parameters()))

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_files = [
        ("test img 1.webp", "User Test 1 (Zebra donor)"),
        ("test img2.webp", "User Test 2 (3-panel splicing figure)"),
        ("samples/eval_results/clean_authentic.jpg", "Clean Authentic JPEG"),
        ("samples/eval_results/clean_spliced.jpg", "Altered Spliced JPEG"),
    ]

    # Add a sample from data/val
    val_imgs = [f for f in os.listdir("data/val/images") if f.endswith(".jpg")]
    for vf in val_imgs[:2]:
        test_files.append((os.path.join("data/val/images", vf), f"CASIA Val Sample: {vf}"))

    os.makedirs("samples/unet_eval_results", exist_ok=True)

    for path, desc in test_files:
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        orig_w, orig_h = img.size
        prob_resized = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        score = int(np.max(prob_resized) * 100)
        status = "Spliced" if score > 50 else ("Suspicious" if score > 20 else "Authentic")

        # Save heatmap
        mask_u8 = (prob_resized * 255).astype(np.uint8)
        heatmap = get_heatmap_overlay(img, mask_u8, alpha=0.45)
        out_path = f"samples/unet_eval_results/{os.path.basename(path)}_unet_heatmap.png"
        cv2.imwrite(out_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

        print(f"[{status:10s}] Score: {score:3d}% | Peak Prob: {np.max(prob_resized):.4f} | {desc} -> Saved: {out_path}")

if __name__ == "__main__":
    test_inference()
