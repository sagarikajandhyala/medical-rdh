"""
experiments/run_experiments.py — Full Experiment Suite on Real NIH X-rays

Experiments:
  1. Capacity-distortion curve (payload size vs PSNR/SSIM)
  2. Method comparison (LSB vs DE vs Ours)
  3. Per-image evaluation across all 20 real X-rays
  4. U-Net segmentation quality (Dice/IoU on training set)
  5. Robustness to common image attacks

Results saved to results/experiment_results.json

Run: python experiments/run_experiments.py
"""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset       import load_xray_gray, get_image_paths
from core.utils         import bytes_to_bits
from core.embed         import embed, _build_tight_safe
from core.extract       import extract
from core.segmentation  import segment, embedding_mask
from agent.policy       import plan_policy
from evaluation.metrics import psnr, ssim, bpp
from experiments.baseline import run_comparison


def exp1_capacity_distortion(arr, modality="xray"):
    """PSNR/SSIM vs payload size on one real X-ray."""
    print(f"\n{'='*55}")
    print("  Exp 1: Capacity-Distortion Curve (Real X-ray)")
    print(f"{'='*55}")

    _, sensitivity = segment(arr)
    policy         = plan_policy(modality, "anonymized", sensitivity, use_llm=False)
    safe           = embedding_mask(sensitivity, policy.sensitivity_threshold)
    tight          = _build_tight_safe(arr, safe)

    results = []
    for n_bytes in [5, 10, 25, 50, 100, 200, 300]:
        bits        = [0] * (n_bytes * 8)
        stego, info = embed(arr, bits, safe)
        if not info["capacity_ok"]:
            print(f"  {n_bytes:4d} bytes: CAPACITY EXCEEDED"); break
        p = psnr(arr, stego); s = ssim(arr, stego)
        b = bpp(info["embedded"], arr.shape)
        _, restored = extract(stego, tight, original_arr=arr)
        rev = bool(np.all(arr == restored))
        print(f"  {n_bytes:4d} bytes: PSNR={p:.2f}dB  SSIM={s:.5f}  "
              f"BPP={b:.5f}  Rev={'YES' if rev else 'NO'}")
        results.append({"bytes": n_bytes, "psnr": round(p,4),
                         "ssim": round(s,6), "bpp": round(b,6), "reversible": rev})
    return results


def exp2_method_comparison(arr):
    """Compare LSB vs DE vs Ours on one real X-ray."""
    print(f"\n{'='*55}")
    print("  Exp 2: Method Comparison")
    print(f"{'='*55}")
    return run_comparison(arr, b"Patient ID: P-001 | NIH-CXR14 Study")


def exp3_per_image_eval(paths, payload=b"PatientID:001"):
    """Run our pipeline on every real X-ray image."""
    print(f"\n{'='*55}")
    print(f"  Exp 3: Per-Image Evaluation ({len(paths)} real X-rays)")
    print(f"{'='*55}")

    payload_bits = bytes_to_bits(payload)
    results      = []

    for path in paths:
        fname = os.path.basename(path)
        arr   = load_xray_gray(path, size=256)

        _, sensitivity = segment(arr)
        policy         = plan_policy("xray", "anonymized", sensitivity, use_llm=False)
        safe           = embedding_mask(sensitivity, policy.sensitivity_threshold)
        tight          = _build_tight_safe(arr, safe)

        stego, info = embed(arr, payload_bits, safe)
        if not info["capacity_ok"]:
            print(f"  {fname}: CAPACITY EXCEEDED"); continue

        rec, restored = extract(stego, tight, original_arr=arr)
        p   = psnr(arr, stego); s = ssim(arr, stego)
        rev = bool(np.all(arr == restored))
        ok  = rec[:len(payload)] == payload

        print(f"  {fname}: PSNR={p:.2f}dB  SSIM={s:.5f}  Rev={'Y' if rev else 'N'}  Match={'✓' if ok else '✗'}")
        results.append({
            "image": fname, "psnr": round(p,4), "ssim": round(s,6),
            "reversible": rev, "payload_ok": ok,
            "bits_embedded": info["embedded"],
        })

    if results:
        avg_psnr = np.mean([r["psnr"] for r in results])
        avg_ssim = np.mean([r["ssim"] for r in results])
        rev_pct  = np.mean([r["reversible"] for r in results]) * 100
        ok_pct   = np.mean([r["payload_ok"]  for r in results]) * 100
        print(f"\n  SUMMARY over {len(results)} images:")
        print(f"  Avg PSNR  : {avg_psnr:.2f} dB")
        print(f"  Avg SSIM  : {avg_ssim:.6f}")
        print(f"  Rev rate  : {rev_pct:.0f}%")
        print(f"  Payload OK: {ok_pct:.0f}%")

    return results


def exp4_unet_quality(paths, n_eval=5):
    """Evaluate U-Net segmentation Dice/IoU on sample images."""
    print(f"\n{'='*55}")
    print(f"  Exp 4: U-Net Segmentation Quality")
    print(f"{'='*55}")

    from data.dataset import make_pseudo_masks_from_anatomy
    from models.unet  import load_unet, predict_roi
    from evaluation.metrics import dice_score, iou_score

    try:
        model = load_unet("models/unet_weights.pth", base_features=16)
    except FileNotFoundError as e:
        print(f"  {e}"); return []

    results = []
    for path in paths[:n_eval]:
        arr   = load_xray_gray(path, size=192)
        # GT pseudo-mask as reference
        gt    = make_pseudo_masks_from_anatomy([arr], 192)[0]
        pred  = predict_roi(model, arr, threshold=0.5).astype(np.float32)
        d     = dice_score(pred, gt)
        iou   = iou_score(pred, gt)
        fname = os.path.basename(path)
        print(f"  {fname}: Dice={d:.4f}  IoU={iou:.4f}")
        results.append({"image": fname, "dice": round(d,4), "iou": round(iou,4)})

    if results:
        print(f"  Avg Dice={np.mean([r['dice'] for r in results]):.4f}  "
              f"Avg IoU={np.mean([r['iou']  for r in results]):.4f}")
    return results


def exp5_robustness(arr):
    """Test payload recovery under common image attacks."""
    print(f"\n{'='*55}")
    print("  Exp 5: Robustness to Image Attacks")
    print(f"{'='*55}")

    payload      = b"PatID:001"
    payload_bits = bytes_to_bits(payload)
    safe         = np.ones(arr.shape, dtype=bool)
    tight        = _build_tight_safe(arr, safe)
    stego, _     = embed(arr, payload_bits, safe)

    def try_extract(attacked):
        try:
            rec, _ = extract(attacked, tight, original_arr=arr)
            return rec[:len(payload)] == payload
        except Exception:
            return False

    results = {}
    ok = try_extract(stego)
    print(f"  No attack     : {'✓ intact' if ok else '✗ corrupted'}")
    results["no_attack"] = ok

    for sigma in [1, 3]:
        noise    = np.random.normal(0, sigma, stego.shape)
        attacked = np.clip(stego.astype(float)+noise, 0, 255).astype(np.uint8)
        ok       = try_extract(attacked)
        print(f"  Gaussian σ={sigma}  : {'✓ intact' if ok else '✗ corrupted'}")
        results[f"gaussian_s{sigma}"] = ok

    try:
        from PIL import Image; import io
        for q in [95, 80]:
            buf = io.BytesIO()
            Image.fromarray(stego).save(buf, format="JPEG", quality=q)
            buf.seek(0)
            attacked = np.array(Image.open(buf).convert("L"))
            ok       = try_extract(attacked)
            print(f"  JPEG q={q}       : {'✓ intact' if ok else '✗ corrupted (expected — lossy)'}")
            results[f"jpeg_q{q}"] = ok
    except Exception as e:
        print(f"  JPEG test skipped: {e}")

    print("  (Note: RDH is intentionally fragile to lossy attacks — by design)")
    return results


def main():
    print("=" * 55)
    print("  Medical RDH — Full Experiment Suite")
    print("  Dataset: Real NIH ChestX-ray14 (20 images)")
    print("=" * 55)

    paths  = get_image_paths("data/raw")
    arr    = load_xray_gray(paths[0], size=256)   # use first real image
    os.makedirs("results", exist_ok=True)

    t0  = time.time()
    all_results = {
        "dataset"             : "NIH_ChestXray14",
        "n_images"            : len(paths),
        "capacity_distortion" : exp1_capacity_distortion(arr),
        "method_comparison"   : exp2_method_comparison(arr),
        "per_image_eval"      : exp3_per_image_eval(paths),
        "unet_quality"        : exp4_unet_quality(paths),
        "robustness"          : exp5_robustness(arr),
    }

    elapsed = time.time() - t0
    print(f"\n✓ All experiments done in {elapsed:.1f}s")

    out = "results/experiment_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()
