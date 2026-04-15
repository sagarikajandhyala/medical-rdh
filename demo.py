"""
demo.py — Quick demo using real NIH ChestX-ray images

Runs the complete pipeline on the first available real X-ray:
  1. Loads real PNG from data/raw/
  2. Preprocesses (grayscale + CLAHE)
  3. U-Net segmentation (requires trained weights)
  4. Agent policy
  5. AES-256-GCM encrypt payload
  6. RDH embed
  7. Extract + decrypt (verify)
  8. Evaluate PSNR/SSIM

Usage:
    python demo.py
    python demo.py --image data/raw/00000003_001.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import get_image_paths
from main         import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   default=None,         help="Specific image path")
    parser.add_argument("--payload", default="Patient ID: P-00123 | Study: NIH-ChestXray14 | Date: 2025",
                        help="Secret payload to embed")
    parser.add_argument("--size",    type=int, default=256, help="Process at this size")
    args = parser.parse_args()

    print("=" * 60)
    print("  Medical RDH — Demo on Real NIH ChestX-ray14")
    print("=" * 60)

    # Pick image
    if args.image:
        image_path = args.image
    else:
        paths = get_image_paths("data/raw")
        if not paths:
            print("ERROR: No PNG images in data/raw/")
            sys.exit(1)
        image_path = paths[0]
        print(f"  Using: {os.path.basename(image_path)}")

    if not os.path.exists("models/unet_weights.pth"):
        print("\n⚠  WARNING: models/unet_weights.pth not found.")
        print("   Run training first:  python models/train_all.py")
        print("   Then re-run demo.\n")
        sys.exit(1)

    result = run_pipeline(
        image_path  = image_path,
        payload_str = args.payload,
        modality    = "xray",
        consent     = "anonymized",
        proc_size   = args.size,
    )

    print("\n" + "=" * 60)
    print("  Demo Summary")
    print("=" * 60)
    print(f"  PSNR       : {result['psnr']:.2f} dB  {'✓' if result['psnr']>50 else '⚠'}")
    print(f"  SSIM       : {result['ssim']:.6f}")
    print(f"  Reversible : {'YES ✓' if result['reversible'] else 'NO ✗'}")
    print(f"  Payload    : {'YES ✓' if result['payload_ok'] else 'NO ✗'}")
    print(f"  Embedded   : {result['bits_embedded']} bits")
    print("=" * 60)


if __name__ == "__main__":
    main()
