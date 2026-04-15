"""core/extract.py — Exact reversal of histogram-shifting embed."""
import numpy as np
from core.utils import compute_checksum, bits_to_bytes, bits_to_int, timestamp
from core.embed import _build_tight_safe


def extract(stego: np.ndarray, safe_mask: np.ndarray,
            T: int = 1, original_arr: np.ndarray = None):
    """
    Extract payload and restore original image.
    original_arr: used to rebuild tight_safe exactly as during embed.
    If not provided, tries to rebuild from stego (may miss boundary edge cases).
    """
    print(f"\n[extract] Shape={stego.shape}")
    ref  = original_arr if original_arr is not None else stego
    tight_safe = _build_tight_safe(ref, safe_mask)

    arr  = stego.copy().astype(np.int32)
    H, W = stego.shape
    bits, payload_len = [], None

    for r in range(2, H, 2):
        for c in range(W):
            if not tight_safe[r, c]: continue
            pred = int(arr[r-1, c])
            e_s  = int(arr[r, c]) - pred
            if e_s >= 2:    arr[r, c] -= 1
            elif e_s == 1:  bits.append(1); arr[r, c] -= 1
            elif e_s == 0:  bits.append(0)
            elif e_s <= -2: arr[r, c] += 1
            if payload_len is None and len(bits) == 32:
                payload_len = bits_to_int(bits[:32])
                print(f"[extract] Header: {payload_len} bits ({payload_len//8} bytes)")

    restored = arr.clip(0, 255).astype(np.uint8)
    payload_bits  = bits[32: 32+payload_len] if payload_len else bits[32:]
    payload_bytes = bits_to_bytes(list(payload_bits))

    print(f"[extract] Recovered: {len(payload_bits)} bits → {len(payload_bytes)} bytes")
    print(f"[extract] Restored checksum: {compute_checksum(restored)}")
    print(f"[extract] Done at {timestamp()}")
    return payload_bytes, restored
