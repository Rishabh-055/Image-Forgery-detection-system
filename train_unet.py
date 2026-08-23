"""
train_unet.py — U-Net Training Pipeline for Image Splicing Detection
=====================================================================
Complete PyTorch training pipeline for pixel-level splicing segmentation.
Includes:
  - Combined BCE + Soft Dice Loss for severe foreground/background class imbalance
  - Cosine annealing learning rate scheduling
  - Deterministic validation evaluation (no stochastic augmentations in val)
  - Automatic best checkpoint saving to unet_splicing_model.pth
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Import U-Net architecture from model.py
sys.path.insert(0, str(Path(__file__).parent))
from model import UNet

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions for Splicing Segmentation
# ─────────────────────────────────────────────────────────────────────────────

class SoftDiceLoss(nn.Module):
    """
    Differentiable Soft Dice Loss for binary segmentation.
    Addresses extreme class imbalance where spliced regions occupy <10% of pixels.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class CombinedBCEDiceLoss(nn.Module):
    """
    Weighted combination of Binary Cross Entropy with Logits and Soft Dice Loss.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SplicingDataset(Dataset):
    """
    Loads (image, binary_mask) pairs.
    Mask values: 255 = tampered, 0 = authentic.
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        image_dir: str,
        mask_dir:  str,
        img_size:  int  = 256,
        augment:   bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.img_size  = img_size
        self.augment   = augment

        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        self.samples = []

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        for img_path in sorted(self.image_dir.iterdir()):
            if img_path.suffix.lower() not in img_extensions:
                continue
            mask_path = self._find_mask(img_path.stem)
            if mask_path is not None:
                self.samples.append((img_path, mask_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No (image, mask) pairs found in {self.image_dir} and {self.mask_dir}"
            )

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def _find_mask(self, stem: str) -> Optional[Path]:
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            candidate = self.mask_dir / (stem + ext)
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L")
        
        # Binarise mask
        mask_arr = np.array(mask_pil)
        mask_bin = (mask_arr > 127).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_bin)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
            if torch.rand(1).item() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)

        image_t = self.img_transform(image)
        mask_t  = self.mask_transform(mask_pil)

        return image_t, mask_t


# ─────────────────────────────────────────────────────────────────────────────
# Metric
# ─────────────────────────────────────────────────────────────────────────────

def dice_coefficient(
    pred_logits: torch.Tensor,
    targets:     torch.Tensor,
    threshold:   float = 0.5,
    eps:         float = 1e-7,
) -> float:
    preds = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (preds * targets).sum()
    return (2.0 * intersection / (preds.sum() + targets.sum() + eps)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Training and Validation Loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_dice, n_batches = 0.0, 0.0, 0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_coefficient(logits.detach(), masks)
        n_batches  += 1

    return total_loss / max(1, n_batches), total_dice / max(1, n_batches)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, n_batches = 0.0, 0.0, 0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)
        loss   = criterion(logits, masks)

        total_loss += loss.item()
        total_dice += dice_coefficient(logits, masks)
        n_batches  += 1

    return total_loss / max(1, n_batches), total_dice / max(1, n_batches)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training device: {device}")

    train_image_dir = os.path.join(args.data_root, "train", "images")
    train_mask_dir  = os.path.join(args.data_root, "train", "masks")
    val_image_dir   = os.path.join(args.data_root, "val",   "images")
    val_mask_dir    = os.path.join(args.data_root, "val",   "masks")

    # If directories don't exist, auto-run prep_dataset
    if not os.path.isdir(train_image_dir):
        log.info("Dataset directories not found. Generating CASIA-standard dataset...")
        from prep_dataset import build_dataset
        build_dataset(num_train=80, num_val=20)

    train_ds = SplicingDataset(train_image_dir, train_mask_dir, args.img_size, augment=True)
    val_ds   = SplicingDataset(val_image_dir,   val_mask_dir,   args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    log.info(f"Dataset: {len(train_ds)} train samples, {len(val_ds)} val samples")

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    output_path = args.output

    log.info(f"Starting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice     = validate(model, val_loader, criterion, device)
        scheduler.step()

        log.info(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}  Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f}  Dice: {val_dice:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            log.info(f"  [SAVED] Best checkpoint saved to {output_path} (Val Loss: {val_loss:.4f})")

    # Always ensure model weight file exists
    if not os.path.exists(output_path):
        torch.save(model.state_dict(), output_path)

    log.info(f"\nTraining complete. Model weights saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net for Image Splicing Detection.")
    parser.add_argument("--data_root",  type=str,   default="data", help="Root data directory.")
    parser.add_argument("--epochs",     type=int,   default=10,   help="Number of epochs.")
    parser.add_argument("--batch_size", type=int,   default=4,    help="Batch size.")
    parser.add_argument("--lr",         type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--img_size",   type=int,   default=256,  help="Image size.")
    parser.add_argument("--output",     type=str,   default="unet_splicing_model.pth", help="Output model path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
