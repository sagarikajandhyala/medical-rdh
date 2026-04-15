"""
models/train_all.py — Master Training Script

Trains all models on real NIH ChestX-ray data:
  1. U-Net segmentation model (lung ROI detection)
  2. CNN pixel predictor (improved embedding capacity)

Usage:
    python models/train_all.py
    python models/train_all.py --epochs 50 --img_size 256
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_dataloaders, get_image_paths
from models.unet import train_unet
from models.cnn_predictor import train_predictor


def main():
    parser = argparse.ArgumentParser(description="Train Medical RDH Models on Real Data")
    parser.add_argument("--data_dir",    default="data/raw",         help="X-ray image directory")
    parser.add_argument("--mask_dir",    default="data/masks",       help="GT mask directory (optional)")
    parser.add_argument("--img_size",    type=int, default=256,      help="Training image size")
    parser.add_argument("--unet_epochs", type=int, default=40,       help="U-Net training epochs")
    parser.add_argument("--pred_epochs", type=int, default=25,       help="Predictor training epochs")
    parser.add_argument("--batch_size",  type=int, default=2,        help="Batch size")
    parser.add_argument("--lr",          type=float, default=1e-3,   help="Learning rate")
    parser.add_argument("--n_patches",   type=int, default=10000,    help="Patches for predictor")
    parser.add_argument("--base_feat",   type=int, default=32,       help="U-Net base features")
    parser.add_argument("--skip_pred",   action="store_true",        help="Skip predictor training")
    args = parser.parse_args()

    print("=" * 60)
    print("  Medical RDH — Full Model Training on Real NIH X-rays")
    print("=" * 60)
    print(f"  Data     : {args.data_dir}")
    print(f"  Img size : {args.img_size}×{args.img_size}")
    print(f"  U-Net    : {args.unet_epochs} epochs, base_features={args.base_feat}")
    print(f"  Predictor: {args.pred_epochs} epochs, {args.n_patches} patches")
    print("=" * 60)

    t_start = time.time()

    # ── 1. U-Net Segmentation ──────────────────────────────────
    print(f"\n[1/2] Training U-Net Segmentation Model...")
    mask_dir = args.mask_dir if os.path.exists(args.mask_dir) else None

    train_loader, val_loader = get_dataloaders(
        data_dir   = args.data_dir,
        mask_dir   = mask_dir,
        img_size   = args.img_size,
        batch_size = args.batch_size,
    )

    unet = train_unet(
        train_loader  = train_loader,
        val_loader    = val_loader,
        save_path     = "models/unet_weights.pth",
        log_path      = "logs/unet_training.csv",
        epochs        = args.unet_epochs,
        lr            = args.lr,
        base_features = args.base_feat,
    )
    print(f"\n✓ U-Net training complete.")

    # ── 2. CNN Predictor ───────────────────────────────────────
    if not args.skip_pred:
        print(f"\n[2/2] Training CNN Pixel Predictor...")
        img_paths = get_image_paths(args.data_dir)

        predictor = train_predictor(
            image_paths = img_paths,
            save_path   = "models/cnn_predictor.pth",
            log_path    = "logs/predictor_training.csv",
            epochs      = args.pred_epochs,
            n_patches   = args.n_patches,
            img_size    = args.img_size,
        )
        print(f"\n✓ Predictor training complete.")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  All training complete in {total_time/60:.1f} minutes")
    print(f"  Saved:")
    print(f"    models/unet_weights.pth")
    if not args.skip_pred:
        print(f"    models/cnn_predictor.pth")
    print(f"    logs/unet_training.csv")
    print(f"    logs/predictor_training.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
