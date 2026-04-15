from evaluation.metrics import ssim as _ssim
def ssim(orig, stego): return _ssim(orig, stego)
def print_ssim(orig, stego):
    v = ssim(orig, stego)
    r = "✓ EXCELLENT" if v>0.99 else ("✓ GOOD" if v>0.95 else "⚠ LOW")
    print(f"  [SSIM] {v:.6f}  {r}"); return v
