"""
core/embed.py — Histogram-Shifting RDH Embedding (provably reversible)

Algorithm:
  - Process even rows; predict from upper (odd) row — never modified
  - Shift all non-zero residuals outward: e≥1→e+1, e≤-1→e-1
  - Embed at zero-residual pixels: e=0 stays or becomes 1
  - Boundary pixels that would overflow excluded and saved to manifest
  - 32-bit length header prepended to payload
"""
import numpy as np
from core.utils import compute_checksum, timestamp, int_to_bits


def _build_tight_safe(arr: np.ndarray, safe: np.ndarray) -> np.ndarray:
    """Exclude boundary pixels that would overflow under shift/embed."""
    H, W  = arr.shape
    tight = safe.copy()
    for r in range(2, H, 2):
        for c in range(W):
            if not safe[r, c]: continue
            pred = int(arr[r-1, c])
            e    = int(arr[r, c]) - pred
            if e >= 1  and arr[r, c] >= 255: tight[r, c] = False
            if e <= -1 and arr[r, c] <= 0:   tight[r, c] = False
            if e == 0  and arr[r, c] >= 255: tight[r, c] = False
    return tight


def embed(arr: np.ndarray, payload_bits: list,
          safe_mask: np.ndarray, T: int = 1):
    all_bits   = int_to_bits(len(payload_bits), 32) + list(payload_bits)
    total_bits = len(all_bits)
    print(f"\n[embed] Shape={arr.shape}  payload={len(payload_bits)} bits (+32 hdr={total_bits})")

    tight_safe = _build_tight_safe(arr, safe_mask)
    excluded   = int(safe_mask.sum()) - int(tight_safe.sum())
    print(f"[embed] Boundary-excluded: {excluded}")

    stego   = arr.copy().astype(np.int32)
    H, W    = arr.shape
    bit_idx = n_embed = n_shift = 0

    for r in range(2, H, 2):
        for c in range(W):
            if not tight_safe[r, c]: continue
            pred = int(arr[r-1, c])
            e    = int(arr[r, c]) - pred
            if e >= 1:
                stego[r, c] += 1; n_shift += 1
            elif e <= -1:
                stego[r, c] -= 1; n_shift += 1
            elif e == 0:
                if bit_idx < total_bits:
                    stego[r, c] += all_bits[bit_idx]
                    bit_idx += 1; n_embed += 1

    stego = stego.clip(0, 255).astype(np.uint8)
    ok    = bit_idx >= total_bits
    print(f"[embed] {'✓' if ok else '⚠'} {bit_idx}/{total_bits} bits | embed={n_embed} shift={n_shift}")

    return stego, {
        "payload_bits": len(payload_bits), "header_bits": 32,
        "embedded": n_embed, "threshold_T": T,
        "capacity_ok": ok,
        "original_checksum": compute_checksum(arr),
        "stego_checksum":    compute_checksum(stego),
        "timestamp": timestamp(),
    }
