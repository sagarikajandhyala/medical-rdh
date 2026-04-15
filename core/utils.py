"""core/utils.py — Shared utilities: bit ops, checksums, I/O"""
import numpy as np, hashlib, json, os, datetime
from PIL import Image

def load_image(path):
    img = Image.open(path)
    if img.mode != "L": img = img.convert("L")
    return np.array(img)

def save_image(arr, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)
    print(f"  [utils] Saved → {path}")

def compute_checksum(arr): return hashlib.sha256(arr.tobytes()).hexdigest()

def save_manifest(m, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f: json.dump(m, f, indent=2)
    print(f"  [utils] Manifest → {path}")

def timestamp(): return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def int_to_bits(v, n): return [(v >> (n-1-i)) & 1 for i in range(n)]

def bits_to_int(bits):
    r = 0
    for b in bits: r = (r << 1) | b
    return r

def bytes_to_bits(data):
    bits = []
    for byte in data: bits.extend(int_to_bits(byte, 8))
    return bits

def bits_to_bytes(bits):
    while len(bits) % 8: bits.append(0)
    return bytes(bits_to_int(bits[i:i+8]) for i in range(0, len(bits), 8))
