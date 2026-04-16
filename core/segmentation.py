"""
core/segmentation.py — Deep Learning Segmentation (U-Net only, no fallbacks)

Uses the trained U-Net to produce:
  - Binary ROI mask (lung region)
  - Sensitivity heatmap (gradient-weighted probability)
"""
import numpy as np
import torch
from models.unet import load_unet, predict_roi, predict_probability

UNET_WEIGHTS  = "models/unet_weights.pth"
BASE_FEATURES = 16


def segment(arr: np.ndarray, device: str = "cpu") -> tuple:
    """
    Run trained U-Net on grayscale image.
    Returns (roi_mask: bool H×W, sensitivity: float32 H×W)
    """
    print(f"\n[segmentation] Image shape: {arr.shape}")

    model    = load_unet(UNET_WEIGHTS, BASE_FEATURES, device)
    prob_map = predict_probability(model, arr, device)  # [0,1] soft map

    # Binary ROI mask at 0.5 threshold
    roi_mask    = (prob_map > 0.5).astype(bool)

    # Sensitivity = blend of ROI probability + gradient magnitude
    sensitivity = _build_sensitivity(arr, prob_map)

    roi_pct = roi_mask.mean() * 100
    print(f"[segmentation] ROI coverage : {roi_pct:.1f}%")
    print(f"[segmentation] Sensitivity  : mean={sensitivity.mean():.3f}  max={sensitivity.max():.3f}")
    print(f"[segmentation] Method       : U-Net (trained DL model)")
    return roi_mask.astype(bool), sensitivity


def embedding_mask(sensitivity: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """Return bool mask of pixels safe for embedding (sensitivity < threshold)."""
    safe = sensitivity < threshold
    print(f"[segmentation] Safe pixels  : {safe.mean()*100:.1f}% (threshold={threshold})")
    return safe


def _build_sensitivity(arr: np.ndarray, prob_map: np.ndarray) -> np.ndarray:
    """
    Sensitivity = 0.7 * U-Net probability + 0.3 * normalised gradient magnitude.
    High probability → lung → high sensitivity (avoid embedding).
    High gradient → edge/boundary → also sensitive.
    """
    arr_f = arr.astype(np.float32)
    gy    = np.gradient(arr_f, axis=0)
    gx    = np.gradient(arr_f, axis=1)
    grad  = np.sqrt(gx**2 + gy**2)
    g_max = grad.max()
    if g_max > 0:
        grad /= g_max

    sensitivity = np.clip(0.7 * prob_map + 0.3 * grad, 0.0, 1.0).astype(np.float32)
    return sensitivity
