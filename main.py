"""
main.py — Medical RDH Full Pipeline (Real NIH ChestX-ray)

DICOM/PNG → Preprocess → U-Net Segment → Agent Policy →
AES-256-GCM Encrypt → RDH Embed → Verify → Evaluate

Usage:
    python main.py --image data/raw/00000001_000.png --payload "Patient ID: P-001"
    python main.py --image data/raw/00000001_000.png --modality xray --consent anonymized
"""

import argparse
import os
import sys
import numpy as np

from core.preprocess   import preprocess
from core.segmentation import segment, embedding_mask
from core.embed        import embed, _build_tight_safe
from core.extract      import extract
from core.utils        import (save_image, save_manifest, bytes_to_bits,
                                compute_checksum, timestamp)
from security.encrypt  import encrypt_payload
from security.decrypt  import decrypt_payload
from agent.policy      import plan_policy, enforce_compliance, generate_audit_entry
from evaluation.psnr   import print_psnr
from evaluation.ssim   import print_ssim


def run_pipeline(
    image_path  : str,
    payload_str : str,
    modality    : str = "xray",
    consent     : str = "anonymized",
    proc_size   : int = None,
):
    print("=" * 60)
    print("  Medical RDH Pipeline — Secure Lossless Data Hiding")
    print("  Dataset: Real NIH ChestX-ray14")
    print("=" * 60)

    base          = os.path.splitext(os.path.basename(image_path))[0]
    proc_path     = f"data/processed/{base}_proc.png"
    stego_path    = f"data/stego/{base}_stego.png"
    restore_path  = f"data/stego/{base}_restored.png"
    manifest_path = f"data/stego/{base}_manifest.json"
    mask_path     = f"data/stego/{base}_tight_safe.npy"

    # ── 1. Preprocess ─────────────────────────────────────────
    arr = preprocess(image_path, proc_path, size=proc_size, apply_clahe=True)

    # ── 2. U-Net Segmentation ─────────────────────────────────
    roi_mask, sensitivity = segment(arr)

    # ── 3. Agent Policy ───────────────────────────────────────
    policy    = plan_policy(modality, consent, sensitivity)
    safe_mask = embedding_mask(sensitivity, policy.sensitivity_threshold)

    if not enforce_compliance(policy, "anonymized_id"):
        print("\n[main] ✗ Embedding blocked by policy.")
        sys.exit(1)

    # ── 4. Build & save tight_safe ────────────────────────────
    tight_safe = _build_tight_safe(arr, safe_mask)
    np.save(mask_path, tight_safe)
    print(f"  [main] Tight-safe mask: {tight_safe.sum()} pixels → {mask_path}")

    # ── 5. Encrypt ────────────────────────────────────────────
    print(f"\n[main] Encrypting payload...")
    payload_bytes             = payload_str.encode("utf-8")
    encrypted_bytes, enc_meta = encrypt_payload(payload_bytes)

    # ── 6. Embed ──────────────────────────────────────────────
    payload_bits      = bytes_to_bits(encrypted_bytes)
    stego, embed_info = embed(arr, payload_bits, safe_mask)
    save_image(stego, stego_path)

    # ── 7. Save manifest ──────────────────────────────────────
    audit    = generate_audit_entry(policy, compute_checksum(arr),
                                    embed_info["embedded"], timestamp())
    manifest = {
        "pipeline_version": "2.0",
        "dataset"         : "NIH_ChestX-ray14",
        "image_base"      : base,
        "modality"        : modality,
        "consent_level"   : consent,
        "encryption"      : enc_meta,
        "embedding"       : embed_info,
        "audit"           : audit,
        "paths"           : {
            "original": image_path, "processed": proc_path,
            "stego"   : stego_path, "restored" : restore_path,
            "mask"    : mask_path,
        },
    }
    save_manifest(manifest, manifest_path)

    # ── 8. Extract + Decrypt (verification) ───────────────────
    print("\n" + "─"*50)
    print("  Verification: Extract → Decrypt → Compare")
    print("─"*50)

    tight_safe_loaded    = np.load(mask_path)
    recovered_packed, restored = extract(stego, tight_safe_loaded, original_arr=arr)

    expected_len         = len(encrypted_bytes)
    recovered_packed     = recovered_packed[:expected_len]

    try:
        recovered_text   = decrypt_payload(recovered_packed, enc_meta["key_b64"])
        print(f"  [main] Recovered: '{recovered_text.decode('utf-8')}'")
        match            = recovered_text == payload_bytes
        print(f"  [main] Payload match: {'✓ PERFECT' if match else '✗ MISMATCH'}")
    except ValueError as e:
        print(f"  [main] Decryption error: {e}")
        match = False

    # ── 9. Evaluate ───────────────────────────────────────────
    print("\n" + "─"*50)
    print("  Image Quality Metrics")
    print("─"*50)
    psnr_val = print_psnr(arr, stego)
    ssim_val = print_ssim(arr, stego)

    diff       = np.abs(arr.astype(int) - restored.astype(int))
    reversible = bool(np.all(diff == 0))
    print(f"  [Reversibility] Max pixel diff: {diff.max()}")
    print(f"  [Reversibility] Perfect: {'✓ YES' if reversible else '✗ NO'}")

    save_image(restored, restore_path)

    print("\n" + "="*60)
    print(f"  Pipeline Complete")
    print(f"  Image    : {base}")
    print(f"  PSNR     : {psnr_val:.2f} dB")
    print(f"  SSIM     : {ssim_val:.6f}")
    print(f"  Reversible: {'YES ✓' if reversible else 'NO ✗'}")
    print(f"  Payload  : {'YES ✓' if match else 'NO ✗'}")
    print("="*60)

    return {
        "psnr": psnr_val, "ssim": ssim_val,
        "reversible": reversible, "payload_ok": match,
        "bits_embedded": embed_info["embedded"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical RDH Pipeline — Real X-ray")
    parser.add_argument("--image",    required=True)
    parser.add_argument("--payload",  default="Patient ID: P-00123 | Study: NIH-ChestXray")
    parser.add_argument("--modality", default="xray", choices=["xray","mri","ct"])
    parser.add_argument("--consent",  default="anonymized", choices=["full","anonymized","none"])
    parser.add_argument("--size",     type=int, default=None, help="Resize image (default=keep original)")
    args = parser.parse_args()
    run_pipeline(args.image, args.payload, args.modality, args.consent, args.size)
