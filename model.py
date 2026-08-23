"""
model.py — Lightweight U-Net Architecture for Image Splicing Detection
=======================================================================
Defines a PyTorch U-Net that takes a 3-channel RGB image as input and
produces a single-channel binary probability mask indicating tampered pixels.

Architecture: Encoder → Bottleneck → Decoder with skip connections
Input  : (B, 3, 256, 256) — normalised RGB tensor
Output : (B, 1, 256, 256) — raw logits (apply sigmoid for probability)

Reference:
    Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
    Image Segmentation", MICCAI 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building Block ──────────────────────────────────────────────────────────

def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """
    Two consecutive 3×3 Conv → BatchNorm → ReLU layers.
    Padding=1 preserves spatial dimensions.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


# ─── U-Net ────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Lightweight U-Net for pixel-wise splicing segmentation.

    Channel progression  (encoder): 3 → 64 → 128 → 256 → bottleneck 512
    Channel progression  (decoder): 512 → 256 → 128 → 64 → output 1

    Skip connections concatenate encoder feature maps with decoder maps at
    matching spatial resolution, allowing the network to recover fine
    spatial detail lost during max-pooling.

    Parameters
    ----------
    in_channels  : int  — number of input image channels (default: 3 for RGB)
    out_channels : int  — number of output mask channels (default: 1 binary)
    base_filters : int  — base number of convolutional filters (default: 64)
    dropout_p    : float — dropout probability in bottleneck (default: 0.3)
    """

    def __init__(
        self,
        in_channels: int  = 3,
        out_channels: int = 1,
        base_filters: int = 64,
        dropout_p: float  = 0.3,
    ):
        super().__init__()
        f = base_filters  # shorthand

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1   = _conv_block(in_channels, f)       # 3   → 64
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2   = _conv_block(f, f * 2)             # 64  → 128
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3   = _conv_block(f * 2, f * 4)         # 128 → 256
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            _conv_block(f * 4, f * 8),                  # 256 → 512
            nn.Dropout2d(p=dropout_p),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        # ConvTranspose2d upsamples by ×2; then we cat the skip connection,
        # so the conv_block input channels are doubled.

        self.up3    = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3   = _conv_block(f * 8, f * 4)         # 512 → 256

        self.up2    = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2   = _conv_block(f * 4, f * 2)         # 256 → 128

        self.up1    = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1   = _conv_block(f * 2, f)             # 128 → 64

        # ── Final 1×1 Projection ─────────────────────────────────────────────
        self.final  = nn.Conv2d(f, out_channels, kernel_size=1)

        # ── Weight Initialisation ─────────────────────────────────────────────
        self._init_weights()

    # ─── Helper ──────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Kaiming (He) initialisation for conv layers — recommended for
        networks with ReLU activations.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _pad_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Pad `upsampled` to match `skip` spatial size if they differ
        (can happen with odd input dimensions), then concatenate on C axis.
        """
        if upsampled.shape != skip.shape:
            upsampled = F.interpolate(
                upsampled, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        return torch.cat([upsampled, skip], dim=1)

    # ─── Forward Pass ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W) input image tensor

        Returns
        -------
        (B, 1, H, W) raw logit mask  — apply torch.sigmoid() for [0,1] probs
        """
        # Encoder
        e1 = self.enc1(x)            # → (B,  64, H,    W)
        e2 = self.enc2(self.pool1(e1))  # → (B, 128, H/2,  W/2)
        e3 = self.enc3(self.pool2(e2))  # → (B, 256, H/4,  W/4)

        # Bottleneck
        b  = self.bottleneck(self.pool3(e3))  # → (B, 512, H/8, W/8)

        # Decoder with skip connections
        d3 = self._pad_and_cat(self.up3(b),  e3)  # → (B, 512, H/4,  W/4)
        d3 = self.dec3(d3)                          # → (B, 256, H/4,  W/4)

        d2 = self._pad_and_cat(self.up2(d3), e2)  # → (B, 256, H/2,  W/2)
        d2 = self.dec2(d2)                          # → (B, 128, H/2,  W/2)

        d1 = self._pad_and_cat(self.up1(d2), e1)  # → (B, 128, H,    W)
        d1 = self.dec1(d1)                          # → (B,  64, H,    W)

        return self.final(d1)                       # → (B,   1, H,    W)
