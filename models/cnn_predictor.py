"""
models/cnn_predictor.py — CNN Pixel Predictor for Improved RDH Capacity

Trained on real NIH chest X-ray patches.
Goal: predict each pixel from context → minimize prediction errors →
      more zero-residual pixels → higher embedding capacity.

Architecture: Residual CNN operating on local neighbourhoods.
Loss: MAE (L1) — standard for pixel prediction tasks.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# ─────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block for the predictor."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class CNNPredictor(nn.Module):
    """
    Fully-convolutional residual predictor.
    Input : (B, 1, H, W) normalised [0, 1]  — centre pixel masked to 0
    Output: (B, 1, H, W) predicted pixel values in [0, 255]
    """
    def __init__(self, n_channels=64, n_res_blocks=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(1, n_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(n_channels) for _ in range(n_res_blocks)])
        self.tail = nn.Conv2d(n_channels, 1, 1)

    def forward(self, x):
        feat = self.head(x)
        feat = self.res_blocks(feat)
        pred = self.tail(feat)
        # Scale from [~0,1] → pixel range (model learns to output [0,255]-normalised)
        return pred * 255.0


# ─────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    """
    Random patch dataset from real X-ray images for predictor training.
    Masks centre pixel → model learns to predict it from context.
    """
    def __init__(self, image_paths: list, n_patches: int = 15000,
                 patch_size: int = 32, img_size: int = 256):
        self.paths      = image_paths
        self.n_patches  = n_patches
        self.patch_size = patch_size
        self.img_size   = img_size
        self.images     = self._load_images()

    def _load_images(self):
        imgs = []
        for p in self.paths:
            img = Image.open(p).convert("L").resize(
                (self.img_size, self.img_size), Image.LANCZOS)
            imgs.append(np.array(img, dtype=np.uint8))
        print(f"[Predictor Dataset] Loaded {len(imgs)} images")
        return imgs

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        rng  = np.random.default_rng(idx)
        img  = self.images[rng.integers(len(self.images))]
        H, W = img.shape
        p    = self.patch_size

        if H <= p or W <= p:
            patch = np.zeros((p, p), dtype=np.float32)
        else:
            r = rng.integers(0, H - p)
            c = rng.integers(0, W - p)
            patch = img[r:r+p, c:c+p].astype(np.float32) / 255.0

        target_val = patch[p//2, p//2]

        # Mask centre pixel so model learns from surrounding context
        inp = patch.copy()
        inp[p//2, p//2] = 0.0

        return (
            torch.from_numpy(inp).unsqueeze(0),           # (1, p, p)
            torch.tensor(target_val * 255.0).float()      # scalar target in [0,255]
        )


# ─────────────────────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────────────────────

def train_predictor(
    image_paths : list,
    save_path   : str   = "models/cnn_predictor.pth",
    log_path    : str   = "logs/predictor_training.csv",
    epochs      : int   = 30,
    n_patches   : int   = 12000,
    batch_size  : int   = 64,
    lr          : float = 1e-3,
    img_size    : int   = 256,
    patch_size  : int   = 32,
    device      : str   = "cpu",
) -> CNNPredictor:
    """
    Train CNN predictor on real X-ray patches.
    Returns trained model. Saves weights and CSV log.
    """
    print(f"\n{'='*60}")
    print(f"  CNN Predictor Training — Real NIH ChestX-ray Patches")
    print(f"  Epochs={epochs}  Patches={n_patches}  PatchSize={patch_size}")
    print(f"{'='*60}")

    dataset   = PatchDataset(image_paths, n_patches, patch_size, img_size)
    loader    = DataLoader(dataset, batch_size=batch_size,
                           shuffle=True, num_workers=0)

    model     = CNNPredictor(n_channels=32, n_res_blocks=3).to(device)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_p:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.L1Loss()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(log_path)  or ".", exist_ok=True)

    with open(log_path, "w") as f:
        f.write("epoch,train_mae,lr\n")

    best_mae = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        epoch_mae = 0.0
        n_batches = 0

        for inp, target in loader:
            inp, target = inp.to(device), target.to(device)
            optimizer.zero_grad()

            # Predict centre pixel
            pred   = model(inp)[:, 0, patch_size//2, patch_size//2]
            loss   = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_mae += loss.item()
            n_batches += 1

        scheduler.step()
        avg_mae    = epoch_mae / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        marker = ""
        if avg_mae < best_mae:
            best_mae = avg_mae
            torch.save(model.state_dict(), save_path)
            marker = " ← BEST"

        print(f"  Epoch {epoch:3d}/{epochs} | MAE={avg_mae:.3f} | "
              f"LR={current_lr:.2e} | {elapsed:.1f}s{marker}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_mae:.6f},{current_lr:.2e}\n")

    print(f"\n  Best MAE: {best_mae:.3f}")
    print(f"  Weights saved → {save_path}")

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model


# ─────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────

def load_predictor(weights_path: str = "models/cnn_predictor.pth",
                   device: str = "cpu") -> CNNPredictor:
    model = CNNPredictor(n_channels=32, n_res_blocks=3).to(device)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"[Predictor] Loaded from {weights_path}")
    else:
        raise FileNotFoundError(
            f"Predictor weights not found at '{weights_path}'. "
            f"Run: python models/train_all.py first."
        )
    model.eval()
    return model


def predict_all(model: CNNPredictor, arr: np.ndarray,
                device: str = "cpu") -> np.ndarray:
    """
    Run predictor on full image → float32 prediction map same size as arr.
    Row 0 falls back to neighbour copy.
    """
    model.eval()
    x = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)
    return np.clip(pred.squeeze().cpu().numpy(), 0, 255).astype(np.float32)
