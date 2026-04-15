"""
experiments/baseline.py — Baseline RDH Method Comparison

Implements and compares:
  1. LSB Substitution (not reversible — baseline)
  2. Difference Expansion — Tian 2003 (reversible)
  3. Histogram Shifting — Ours (reversible, DL-guided)

Run: python experiments/baseline.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import bytes_to_bits, bits_to_bytes, int_to_bits, bits_to_int
from evaluation.metrics import psnr, ssim


# ── LSB ──────────────────────────────────────────────────────

def lsb_embed(arr, payload_bits):
    stego   = arr.copy()
    all_bits = int_to_bits(len(payload_bits), 32) + list(payload_bits)
    idx = 0
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if idx >= len(all_bits): break
            stego[r, c] = (int(stego[r, c]) & 0xFE) | all_bits[idx]
            idx += 1
        if idx >= len(all_bits): break
    return stego, {"method": "LSB", "bits": idx, "reversible": False}


def lsb_extract(stego, n_bits):
    bits = []
    hdr  = []
    for r in range(stego.shape[0]):
        for c in range(stego.shape[1]):
            b = int(stego[r, c]) & 1
            if len(hdr) < 32:
                hdr.append(b)
                if len(hdr) == 32: n_bits = bits_to_int(hdr)
            else:
                bits.append(b)
            if len(bits) >= n_bits: break
        if len(bits) >= n_bits: break
    return bits_to_bytes(bits[:n_bits])


# ── Difference Expansion (Tian 2003) ─────────────────────────

def de_embed(arr, payload_bits):
    stego    = arr.copy().astype(np.int32)
    all_bits = int_to_bits(len(payload_bits), 32) + list(payload_bits)
    idx      = 0
    loc_map  = []
    H, W     = arr.shape

    for r in range(H):
        c = 0
        while c + 1 < W:
            if idx >= len(all_bits):
                loc_map.append(0); c += 2; continue
            u, v = int(stego[r, c]), int(stego[r, c+1])
            l    = (u + v) // 2
            h    = u - v
            h_new   = 2 * h + all_bits[idx]
            u_new   = l + int(np.ceil(h_new / 2))
            v_new   = l - int(np.floor(h_new / 2))
            if 0 <= u_new <= 255 and 0 <= v_new <= 255:
                stego[r, c]   = u_new
                stego[r, c+1] = v_new
                loc_map.append(1); idx += 1
            else:
                loc_map.append(0)
            c += 2

    return stego.clip(0, 255).astype(np.uint8), {
        "method": "DifferenceExpansion", "bits": idx,
        "reversible": True, "loc_map": loc_map
    }


def de_extract(stego, loc_map):
    arr  = stego.copy().astype(np.int32)
    H, W = stego.shape
    bits = []
    li   = 0
    for r in range(H):
        c = 0
        while c + 1 < W:
            if li < len(loc_map) and loc_map[li]:
                u, v   = int(arr[r, c]), int(arr[r, c+1])
                l      = (u + v) // 2
                h      = u - v
                h_orig = h // 2
                bit    = abs(h) % 2
                bits.append(bit)
                arr[r, c]   = l + int(np.ceil(h_orig / 2))
                arr[r, c+1] = l - int(np.floor(h_orig / 2))
            li += 1; c += 2
    arr = arr.clip(0, 255).astype(np.uint8)
    if len(bits) < 32: return b"", arr
    n    = bits_to_int(bits[:32])
    return bits_to_bytes(bits[32:32+n]), arr


# ── Our Method ────────────────────────────────────────────────

def our_embed(arr, payload_bits, safe_mask=None):
    from core.embed import embed, _build_tight_safe
    if safe_mask is None:
        safe_mask = np.ones(arr.shape, dtype=bool)
    return embed(arr, payload_bits, safe_mask)


def our_extract(stego, tight_safe):
    from core.extract import extract
    return extract(stego, tight_safe)


# ── Comparison Runner ─────────────────────────────────────────

def run_comparison(arr: np.ndarray, payload: bytes = None) -> dict:
    if payload is None:
        payload = b"Patient ID: P-001 | Study: RDH-2025"
    payload_bits = bytes_to_bits(payload)

    print(f"\n{'='*60}")
    print(f"  Baseline Comparison  payload={len(payload)} bytes")
    print(f"{'='*60}")
    results = {}

    # LSB
    print("\n[1] LSB Substitution")
    stego_lsb, info = lsb_embed(arr, payload_bits)
    p = psnr(arr, stego_lsb); s = ssim(arr, stego_lsb)
    rec = lsb_extract(stego_lsb, len(payload_bits))
    ok  = rec[:len(payload)] == payload
    print(f"  PSNR={p:.2f}dB  SSIM={s:.4f}  Rev=NO  Match={'✓' if ok else '✗'}")
    results["LSB"] = {"psnr": round(p,2), "ssim": round(s,4),
                       "reversible": False, "payload_ok": ok,
                       "bpp": round(len(payload_bits)/arr.size, 5)}

    # DE
    print("\n[2] Difference Expansion (Tian 2003)")
    stego_de, info_de = de_embed(arr, payload_bits)
    p = psnr(arr, stego_de); s = ssim(arr, stego_de)
    rec_de, rest_de = de_extract(stego_de, info_de["loc_map"])
    rev = bool(np.all(arr == rest_de)); ok = rec_de[:len(payload)] == payload
    print(f"  PSNR={p:.2f}dB  SSIM={s:.4f}  Rev={'YES' if rev else 'NO'}  Match={'✓' if ok else '✗'}")
    results["DifferenceExpansion"] = {"psnr": round(p,2), "ssim": round(s,4),
                                       "reversible": rev, "payload_ok": ok,
                                       "bpp": round(info_de["bits"]/arr.size, 5)}

    # Ours
    print("\n[3] Histogram Shifting — Ours (DL-guided)")
    from core.embed import _build_tight_safe
    safe  = np.ones(arr.shape, dtype=bool)
    tight = _build_tight_safe(arr, safe)
    stego_hs, info_hs = our_embed(arr, payload_bits, safe)
    p = psnr(arr, stego_hs); s = ssim(arr, stego_hs)
    rec_hs, rest_hs = our_extract(stego_hs, tight)
    rev = bool(np.all(arr == rest_hs)); ok = rec_hs[:len(payload)] == payload
    print(f"  PSNR={p:.2f}dB  SSIM={s:.4f}  Rev={'YES' if rev else 'NO'}  Match={'✓' if ok else '✗'}")
    results["HistShift_Ours"] = {"psnr": round(p,2), "ssim": round(s,4),
                                  "reversible": rev, "payload_ok": ok,
                                  "bpp": round(info_hs["embedded"]/arr.size, 5)}

    # Summary table
    print(f"\n{'─'*60}")
    print(f"  {'Method':<26} {'PSNR':>7} {'SSIM':>8} {'Rev':>5} {'BPP':>9}")
    print(f"  {'─'*56}")
    for m, r in results.items():
        print(f"  {m:<26} {r['psnr']:>6.2f}  {r['ssim']:>7.4f}  "
              f"{'Y' if r['reversible'] else 'N':>4}  {r['bpp']:>9.5f}")
    print(f"{'─'*60}")
    return results


if __name__ == "__main__":
    from data.dataset import load_xray_gray, get_image_paths
    paths = get_image_paths("data/raw")
    arr   = load_xray_gray(paths[0], size=256)
    run_comparison(arr, b"Patient ID: P-001 | Study: NIH-Comparison")
