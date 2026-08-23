"""Integration test for the full pipeline."""
import torch, torchvision
from model import UNet
from utils.ela_processor import calculate_ela, get_ela_stats, get_ela_heatmap, generate_forgery_score
from PIL import Image
import io, numpy as np

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)

# ── Test U-Net forward pass ──
model = UNet(in_channels=3, out_channels=1)
model.eval()
x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    out = model(x)
assert out.shape == (1, 1, 256, 256), f"Unexpected shape: {out.shape}"
print(f"U-Net output shape: {out.shape}  [OK]")

# ── Test ELA pipeline ──
img = Image.new("RGB", (300, 200), color=(80, 120, 60))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=85)
buf.seek(0)
img_j = Image.open(buf)
ela   = calculate_ela(img_j, quality=90, amplify=20.0)
stats = get_ela_stats(ela)
score, label = generate_forgery_score(ela)
print(f"ELA mean={stats['mean']:.2f}  variance={stats['variance']:.2f}")
print(f"Score: {score}  ->  {label}")

# ── Test heatmap overlay ──
heatmap = get_ela_heatmap(ela, img_j, alpha=0.45)
assert heatmap.shape == (200, 300, 3), f"Unexpected heatmap shape: {heatmap.shape}"
print(f"Heatmap shape: {heatmap.shape}  [OK]")

print("\n=== ALL SYSTEMS GO — ready to launch app ===")
