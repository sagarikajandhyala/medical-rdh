"""evaluation/metrics.py — PSNR, SSIM, BPP, NCC, Dice, IoU, BER"""
import numpy as np

def psnr(orig, stego):
    mse = np.mean((orig.astype(np.float64)-stego.astype(np.float64))**2)
    return float("inf") if mse==0 else 20*np.log10(255/np.sqrt(mse))

def ssim(orig, stego, K1=0.01, K2=0.03, L=255):
    x,y = orig.astype(np.float64), stego.astype(np.float64)
    C1,C2 = (K1*L)**2,(K2*L)**2
    num = (2*x.mean()*y.mean()+C1)*(2*((x-x.mean())*(y-y.mean())).mean()+C2)
    den = (x.mean()**2+y.mean()**2+C1)*(x.var()+y.var()+C2)
    return float(num/den)

def bpp(n_bits, shape): return n_bits/(shape[0]*shape[1])

def ncc(orig, stego):
    o = orig.astype(np.float64)-orig.mean()
    s = stego.astype(np.float64)-stego.mean()
    d = np.sqrt((o**2).sum()*(s**2).sum())
    return float((o*s).sum()/d) if d>0 else 1.0

def dice_score(pred, target, thresh=0.5, smooth=1e-5):
    pb = (pred>thresh).astype(float)
    inter = (pb*target).sum()
    return float((2*inter+smooth)/(pb.sum()+target.sum()+smooth))

def iou_score(pred, target, thresh=0.5, smooth=1e-5):
    pb = (pred>thresh).astype(float)
    inter = (pb*target).sum()
    union = pb.sum()+target.sum()-inter
    return float((inter+smooth)/(union+smooth))

def full_report(orig, stego, restored, n_bits):
    p   = psnr(orig, stego)
    s   = ssim(orig, stego)
    n   = ncc(orig, stego)
    b   = bpp(n_bits, orig.shape)
    rev = bool(np.all(orig==restored))
    md  = int(np.abs(orig.astype(int)-restored.astype(int)).max())
    print(f"\n{'─'*48}")
    print(f"  Evaluation Report")
    print(f"{'─'*48}")
    ps = f"{p:.2f} dB" if p!=float("inf") else "∞ dB"
    print(f"  PSNR          : {ps}  {'✓' if p>50 else '⚠'}")
    print(f"  SSIM          : {s:.6f}  {'✓' if s>0.99 else '⚠'}")
    print(f"  NCC           : {n:.6f}")
    print(f"  BPP           : {b:.6f}")
    print(f"  Bits embedded : {n_bits}")
    print(f"  Reversible    : {'YES ✓' if rev else f'NO ✗ (max_diff={md})'}")
    print(f"{'─'*48}")
    return {"psnr_db":round(p,4) if p!=float("inf") else "inf",
            "ssim":round(s,6),"ncc":round(n,6),"bpp":round(b,6),
            "bits_embedded":n_bits,"reversible":rev,"max_pixel_diff":md}
