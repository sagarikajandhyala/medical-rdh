"""
core/preprocess.py — Medical Image Preprocessing

Handles real X-ray PNGs: RGB→grayscale, CLAHE normalization,
proper resizing for pipeline consistency.
"""
import numpy as np
import cv2
import os
from PIL import Image


def preprocess(input_path: str, output_path: str,
               size: int = None, apply_clahe: bool = True) -> np.ndarray:
    """
    Load X-ray → grayscale → optional CLAHE → save → return uint8 array.
    Size=None keeps original size.
    """
    print(f"\n[preprocess] Loading: {input_path}")
    img = Image.open(input_path).convert("L")

    if size:
        img = img.resize((size, size), Image.LANCZOS)

    arr = np.array(img, dtype=np.uint8)
    print(f"[preprocess] Shape: {arr.shape}  Range: [{arr.min()}, {arr.max()}]")

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr   = clahe.apply(arr)
        print(f"[preprocess] CLAHE applied")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(arr).save(output_path)
    print(f"[preprocess] Saved → {output_path}")
    return arr
