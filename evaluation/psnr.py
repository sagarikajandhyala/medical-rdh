from evaluation.metrics import psnr as _psnr
def psnr(orig, stego): return _psnr(orig, stego)
def print_psnr(orig, stego):
    v = psnr(orig, stego)
    r = "✓ EXCELLENT" if v>50 else ("✓ GOOD" if v>40 else "⚠ LOW")
    s = f"{v:.2f} dB" if v!=float("inf") else "∞ dB"
    print(f"  [PSNR] {s}  {r}"); return v
