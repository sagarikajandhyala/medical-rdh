"""
dicom/dicom_handler.py — DICOM and PNG image I/O

Reads DICOM (.dcm) files or standard PNGs, normalises pixel data to uint8,
and writes stego images back in the original format.
"""
import os
import numpy as np
from PIL import Image

try:
    import pydicom
    PYDICOM = True
except ImportError:
    PYDICOM = False


def read_image(path: str) -> tuple:
    """
    Read any supported image file.
    Returns (arr: uint8 H×W grayscale, meta: dict).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm" and PYDICOM:
        return _read_dcm(path)
    return _read_png(path)


def _read_dcm(path):
    ds  = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    a, b = arr.min(), arr.max()
    if b > a:
        arr = (arr - a) / (b - a) * 255.0
    arr  = arr.astype(np.uint8)
    meta = {
        "format"  : "DICOM",
        "modality": getattr(ds, "Modality", "Unknown"),
        "patient" : getattr(ds, "PatientID", "Unknown"),
        "path"    : path,
    }
    print(f"[dicom] DICOM: modality={meta['modality']}  shape={arr.shape}")
    return arr, meta


def _read_png(path):
    img  = Image.open(path).convert("L")
    arr  = np.array(img, dtype=np.uint8)
    meta = {"format": "PNG", "path": path}
    print(f"[dicom] PNG: shape={arr.shape}")
    return arr, meta


def write_image(arr: np.ndarray, original_path: str, output_path: str):
    """Write stego image, preserving original format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = os.path.splitext(original_path)[1].lower()
    if ext == ".dcm" and PYDICOM:
        ds             = pydicom.dcmread(original_path)
        ds.PixelData   = arr.astype(np.uint8).tobytes()
        ds.BitsAllocated = ds.BitsStored = 8
        ds.HighBit       = 7
        ds.save_as(output_path)
    else:
        Image.fromarray(arr.astype(np.uint8)).save(output_path)
    print(f"[dicom] Saved → {output_path}")


def load_dataset(folder: str, extensions=(".dcm", ".png", ".jpg")) -> list:
    """Load all images from a folder. Returns list of (arr, meta) tuples."""
    results = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if any(f.lower().endswith(e) for e in extensions):
                try:
                    arr, meta = read_image(os.path.join(root, f))
                    results.append((arr, meta))
                except Exception as e:
                    print(f"[dicom] Skipped {f}: {e}")
    print(f"[dicom] Loaded {len(results)} images from {folder}")
    return results
