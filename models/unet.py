"""
models/unet.py — U-Net Segmentation Model

Full implementation per Ronneberger et al. (2015), trained on real
NIH ChestX-ray images with anatomical lung masks.

Architecture:
  - Encoder: 4 downsampling blocks (DoubleConv + MaxPool)
  - Bottleneck: deepest feature extraction
  - Decoder: 4 upsampling blocks with skip connections
  - Output: sigmoid mask [0,1]

Training:
  - Combined Dice + BCE loss
  - Adam optimizer with cosine annealing LR schedule
  - Full training/validation loop with IoU and Dice metrics
  - Saves best model by validation Dice score
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  U-Net Architecture
# ─────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Conv2d → BN → ReLU → Conv2d → BN → ReLU"""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d(2) → DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Bilinear upsample → concat skip → DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x  = self.up(x)
        # Pad if needed (odd spatial dims)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x  = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x  = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net with 4 encoder/decoder stages.

    in_channels  : 1 (grayscale X-ray)
    out_channels : 1 (lung mask)
    base_features: feature channels in first encoder block
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = Down(f,     f * 2)
        self.enc3 = Down(f * 2, f * 4)
        self.enc4 = Down(f * 4, f * 8)

        # Bottleneck
        self.bottleneck = Down(f * 8, f * 16)

        # Decoder
        self.dec4 = Up(f * 16 + f * 8, f * 8)
        self.dec3 = Up(f * 8  + f * 4, f * 4)
        self.dec2 = Up(f * 4  + f * 2, f * 2)
        self.dec1 = Up(f * 2  + f,     f)

        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b  = self.bottleneck(e4)

        # Decoder path with skip connections
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return torch.sigmoid(self.out_conv(d1))


# ─────────────────────────────────────────────────────────────
#  Loss Functions
# ─────────────────────────────────────────────────────────────

def dice_loss(pred, target, smooth=1e-5):
    """Differentiable Dice loss."""
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def combined_loss(pred, target, bce_weight=0.5):
    """BCE + Dice combined loss (standard for medical segmentation)."""
    bce  = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + (1 - bce_weight) * dice


# ─────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────

def compute_dice(pred, target, threshold=0.5, smooth=1e-5):
    """Compute Dice coefficient (scalar)."""
    pred_bin    = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    return ((2.0 * intersection + smooth) / (pred_bin.sum() + target.sum() + smooth)).item()


def compute_iou(pred, target, threshold=0.5, smooth=1e-5):
    """Compute Intersection over Union (scalar)."""
    pred_bin    = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union        = pred_bin.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


# ─────────────────────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────────────────────

def train_unet(
    train_loader,
    val_loader,
    save_path   : str   = "models/unet_weights.pth",
    log_path    : str   = "logs/unet_training.csv",
    epochs      : int   = 50,
    lr          : float = 1e-3,
    base_features: int  = 32,
    device      : str   = "cpu",
) -> UNet:
    """
    Full training loop for U-Net on real chest X-ray data.

    Returns trained model.
    Saves best weights and full training log (CSV).
    """
    print(f"\n{'='*60}")
    print(f"  U-Net Training — REAL NIH ChestX-ray Dataset")
    print(f"  Epochs={epochs}  LR={lr}  Features={base_features}  Device={device}")
    print(f"{'='*60}")

    model     = UNet(in_channels=1, out_channels=1, base_features=base_features).to(device)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_p:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_dice = 0.0
    history = []

    os.makedirs(os.path.dirname(save_path)  or ".", exist_ok=True)
    os.makedirs(os.path.dirname(log_path)   or ".", exist_ok=True)

    # CSV header
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_dice,val_loss,val_dice,val_iou,lr\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Training phase ────────────────────────────────────
        model.train()
        train_loss = train_dice = 0.0
        n_train    = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = combined_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_dice += compute_dice(pred.detach(), yb)
            n_train    += 1

        train_loss /= max(n_train, 1)
        train_dice /= max(n_train, 1)

        # ── Validation phase ──────────────────────────────────
        model.eval()
        val_loss = val_dice = val_iou = 0.0
        n_val    = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred    = model(xb)
                loss    = combined_loss(pred, yb)
                val_loss += loss.item()
                val_dice += compute_dice(pred, yb)
                val_iou  += compute_iou(pred, yb)
                n_val    += 1

        val_loss /= max(n_val, 1)
        val_dice /= max(n_val, 1)
        val_iou  /= max(n_val, 1)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        # Save best model
        marker = ""
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), save_path)
            marker = " ← BEST"

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"Loss {train_loss:.4f}/{val_loss:.4f} | "
              f"Dice {train_dice:.4f}/{val_dice:.4f} | "
              f"IoU {val_iou:.4f} | "
              f"LR {current_lr:.2e} | {elapsed:.1f}s{marker}")

        # Log to CSV
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_dice:.6f},"
                    f"{val_loss:.6f},{val_dice:.6f},{val_iou:.6f},{current_lr:.2e}\n")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_dice": train_dice,
            "val_loss": val_loss, "val_dice": val_dice, "val_iou": val_iou,
        })

    print(f"\n  Best Val Dice: {best_val_dice:.4f}")
    print(f"  Weights saved → {save_path}")
    print(f"  Training log  → {log_path}")

    # Load best weights before returning
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model


# ─────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────

def load_unet(weights_path: str = "models/unet_weights.pth",
              base_features: int = 32, device: str = "cpu") -> UNet:
    """Load trained U-Net from disk."""
    model = UNet(in_channels=1, out_channels=1, base_features=base_features).to(device)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"[UNet] Loaded weights from {weights_path}")
    else:
        raise FileNotFoundError(
            f"U-Net weights not found at '{weights_path}'. "
            f"Run: python models/train_all.py first."
        )
    model.eval()
    return model


def predict_roi(model: UNet, arr: np.ndarray,
                threshold: float = 0.5, device: str = "cpu") -> np.ndarray:
    """
    Run trained U-Net on uint8 grayscale image → binary ROI mask.

    Args:
        model    : trained UNet instance
        arr      : H×W uint8 grayscale array
        threshold: sigmoid output binarization threshold

    Returns:
        roi_mask : bool H×W array (True = lung / sensitive ROI)
    """
    H, W = arr.shape
    # Pad to multiple of 16 (for 4 MaxPool layers)
    ph = ((H + 15) // 16) * 16
    pw = ((W + 15) // 16) * 16

    padded = np.zeros((ph, pw), dtype=np.float32)
    padded[:H, :W] = arr.astype(np.float32) / 255.0

    tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)

    prob     = out.squeeze().cpu().numpy()[:H, :W]
    roi_mask = (prob > threshold).astype(bool)

    return roi_mask


def predict_probability(model: UNet, arr: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Return soft probability map [0,1] instead of binary mask."""
    H, W = arr.shape
    ph = ((H + 15) // 16) * 16
    pw = ((W + 15) // 16) * 16
    padded = np.zeros((ph, pw), dtype=np.float32)
    padded[:H, :W] = arr.astype(np.float32) / 255.0
    tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    return out.squeeze().cpu().numpy()[:H, :W]
