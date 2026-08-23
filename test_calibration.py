"""Quick calibration verification for the false-positive fix."""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import os

from model import UNet
from utils.ela_processor import enhance_unet_mask

model = UNet(in_channels=3, out_channels=1)
state = torch.load('unet_splicing_model.pth', map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CONFIDENCE_THRESH = 0.55
MIN_TAMPERED_FRAC = 0.005

def score_image(path, label):
    if not os.path.exists(path):
        return
    img = Image.open(path).convert('RGB')
    t = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(t)).squeeze().numpy()

    confident = prob[prob > CONFIDENCE_THRESH]
    frac = len(confident) / max(prob.size, 1)

    if len(confident) > 0 and frac >= MIN_TAMPERED_FRAC:
        region_mean  = float(confident.mean())
        coverage_w   = min(frac / 0.15, 1.0)
        raw_score    = region_mean * (0.6 + 0.4 * coverage_w)
    else:
        raw_score = float(np.percentile(prob, 30))

    score   = int(np.clip(raw_score * 100, 0, 100))
    verdict = 'Spliced'    if score > 65 else \
              'Suspicious' if score > 35 else \
              'Authentic'

    enhanced = enhance_unet_mask(prob)
    print(f"  [{verdict:10s}] Score:{score:3d}%  | conf_frac:{frac*100:.2f}%  | enh_max:{enhanced.max():.3f}  | {label}")


print("=" * 65)
print("  CALIBRATED SCORING VERIFICATION")
print("=" * 65)
score_image('samples/eval_results/clean_authentic.jpg', 'Authentic landscape (should be Authentic/Suspicious)')
score_image('samples/eval_results/clean_spliced.jpg',   'Spliced landscape  (should be Spliced)')
score_image('test img 1.webp',  'Test img 1 - Zebra donor      (known spliced)')
score_image('test img2.webp',   'Test img 2 - Research panel   (known spliced)')

print()
print("CASIA Validation Set:")
val_imgs = sorted(os.listdir('data/val/images'))[:6]
for vf in val_imgs:
    gt = 'Spliced' if vf.startswith('Tp') else 'Authentic'
    score_image(os.path.join('data/val/images', vf), f'{vf} (GT: {gt})')

print("=" * 65)
