"""
scripts/plot_results.py — Generate all result figures

Produces:
  results/training_curves.png     — U-Net & predictor loss/Dice over epochs
  results/capacity_distortion.png — PSNR/SSIM vs payload size
  results/method_comparison.png   — LSB vs DE vs Ours bar chart
  results/per_image_psnr.png      — PSNR per real X-ray image
  results/sensitivity_pipeline.png— Original / Sensitivity / Safe-mask visual
  results/stego_comparison.png    — Original / Stego / Restored visual

Run: python scripts/plot_results.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
STYLE = {
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}
plt.rcParams.update(STYLE)
COLORS = ["#2563EB", "#16A34A", "#DC2626", "#9333EA", "#F59E0B"]


# ── 1. Training curves ────────────────────────────────────────

def plot_training_curves():
    unet_log = "logs/unet_training.csv"
    pred_log = "logs/predictor_training.csv"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Model Training on Real NIH ChestX-ray14", fontsize=12, fontweight="bold")

    if os.path.exists(unet_log):
        data = np.genfromtxt(unet_log, delimiter=",", skip_header=1)
        if data.ndim == 1: data = data[np.newaxis]
        epochs = data[:, 0]
        axes[0].plot(epochs, data[:, 1], label="Train", color=COLORS[0], lw=2)
        axes[0].plot(epochs, data[:, 3], label="Val",   color=COLORS[2], lw=2, linestyle="--")
        axes[0].set_title("U-Net Loss (Dice+BCE)"); axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss"); axes[0].legend()

        axes[1].plot(epochs, data[:, 2], label="Train Dice", color=COLORS[0], lw=2)
        axes[1].plot(epochs, data[:, 4], label="Val Dice",   color=COLORS[2], lw=2, linestyle="--")
        axes[1].plot(epochs, data[:, 5], label="Val IoU",    color=COLORS[3], lw=2, linestyle=":")
        axes[1].set_title("U-Net Dice / IoU"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score"); axes[1].legend()
    else:
        for ax in axes[:2]: ax.text(0.5, 0.5, "Train U-Net first", ha="center", va="center",
                                     transform=ax.transAxes, fontsize=11, color="gray")

    if os.path.exists(pred_log):
        data = np.genfromtxt(pred_log, delimiter=",", skip_header=1)
        if data.ndim == 1: data = data[np.newaxis]
        axes[2].plot(data[:, 0], data[:, 1], color=COLORS[1], lw=2, marker="o", ms=4)
        axes[2].set_title("CNN Predictor MAE"); axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("MAE (pixel units)")
    else:
        axes[2].text(0.5, 0.5, "Train predictor first", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=11, color="gray")

    plt.tight_layout()
    out = "results/training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {out}")


# ── 2. Capacity-distortion ────────────────────────────────────

def plot_capacity_distortion(data):
    if not data: return
    sizes = [r["bytes"] for r in data]
    psnrs = [r["psnr"]  for r in data]
    ssims = [r["ssim"]  for r in data]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Capacity-Distortion Curve — Real NIH ChestX-ray", fontsize=12, fontweight="bold")

    axes[0].plot(sizes, psnrs, "o-", color=COLORS[0], lw=2, ms=7)
    axes[0].axhline(50, color=COLORS[2], ls="--", lw=1.2, label="Target 50 dB")
    axes[0].set_xlabel("Payload (bytes)"); axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR vs Payload Size"); axes[0].legend()

    axes[1].plot(sizes, ssims, "s-", color=COLORS[1], lw=2, ms=7)
    axes[1].axhline(0.99, color=COLORS[2], ls="--", lw=1.2, label="Target 0.99")
    axes[1].set_xlabel("Payload (bytes)"); axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM vs Payload Size"); axes[1].legend()

    plt.tight_layout()
    out = "results/capacity_distortion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {out}")


# ── 3. Method comparison ──────────────────────────────────────

def plot_method_comparison(data):
    if not data: return
    methods = list(data.keys())
    psnrs   = [data[m]["psnr"] for m in methods]
    ssims   = [data[m]["ssim"] for m in methods]
    colors  = [COLORS[2], COLORS[3], COLORS[0]]
    labels  = [m.replace("_", "\n") for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Method Comparison: LSB vs Difference Expansion vs Ours",
                 fontsize=12, fontweight="bold")

    bars = axes[0].bar(labels, psnrs, color=colors, edgecolor="white", lw=0.8)
    axes[0].axhline(50, color="black", ls="--", lw=1.2, label="50 dB target")
    axes[0].set_ylabel("PSNR (dB)"); axes[0].set_title("PSNR"); axes[0].legend()
    for bar, v in zip(bars, psnrs):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.1f}", ha="center", fontsize=9)

    bars = axes[1].bar(labels, ssims, color=colors, edgecolor="white", lw=0.8)
    axes[1].set_ylabel("SSIM"); axes[1].set_title("SSIM"); axes[1].set_ylim([0.97, 1.002])
    for bar, v in zip(bars, ssims):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+0.0002, f"{v:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    out = "results/method_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {out}")


# ── 4. Per-image PSNR ─────────────────────────────────────────

def plot_per_image(data):
    if not data: return
    names = [r["image"][:15] for r in data]
    psnrs = [r["psnr"] for r in data]
    ssims = [r["ssim"] for r in data]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle("Per-Image RDH Quality — All 20 Real NIH ChestX-rays",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(names))
    axes[0].bar(x, psnrs, color=COLORS[0], alpha=0.85)
    axes[0].axhline(50, color=COLORS[2], ls="--", lw=1.2, label="50 dB target")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("PSNR (dB)"); axes[0].set_title("PSNR per Image"); axes[0].legend()

    axes[1].bar(x, ssims, color=COLORS[1], alpha=0.85)
    axes[1].axhline(0.99, color=COLORS[2], ls="--", lw=1.2, label="0.99 target")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("SSIM"); axes[1].set_title("SSIM per Image"); axes[1].legend()
    axes[1].set_ylim([0.99, 1.001])

    plt.tight_layout()
    out = "results/per_image_psnr.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {out}")


# ── 5. Sensitivity pipeline visual ───────────────────────────

def plot_sensitivity_pipeline():
    from data.dataset       import load_xray_gray, get_image_paths
    from core.segmentation  import segment, embedding_mask
    from agent.policy       import plan_policy

    paths = get_image_paths("data/raw")
    arr   = load_xray_gray(paths[0], size=256)

    try:
        _, sensitivity = segment(arr)
    except Exception as e:
        print(f"  Segmentation failed ({e}) — train U-Net first"); return

    policy    = plan_policy("xray", "anonymized", sensitivity, use_llm=False)
    safe_mask = embedding_mask(sensitivity, policy.sensitivity_threshold)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("RDH Pipeline: Segmentation → Sensitivity → Safe Embedding Region",
                 fontsize=11, fontweight="bold")

    axes[0].imshow(arr, cmap="gray"); axes[0].set_title("Real NIH ChestX-ray"); axes[0].axis("off")
    im = axes[1].imshow(sensitivity, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Sensitivity Map (U-Net)"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(safe_mask, cmap="RdYlGn")
    axes[2].set_title(f"Safe Embedding Region\n(green = safe, threshold={policy.sensitivity_threshold})")
    axes[2].axis("off")

    plt.tight_layout()
    out = "results/sensitivity_pipeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {out}")


# ── 6. Stego comparison ───────────────────────────────────────

def plot_stego_comparison():
    paths_to_check = [
        ("data/processed", "proc"),
        ("data/stego",     "stego"),
    ]
    avail = []
    for folder, suffix in paths_to_check:
        if os.path.exists(folder):
            imgs = [f for f in os.listdir(folder) if f.endswith(".png")]
            if imgs:
                avail.append((os.path.join(folder, imgs[0]), suffix))
    if not avail:
        print("  No stego images yet — run main.py first"); return

    from PIL import Image
    fig, axes = plt.subplots(1, len(avail), figsize=(5*len(avail), 5))
    fig.suptitle("Stego Image Visual Quality", fontsize=11, fontweight="bold")
    if len(avail) == 1: axes = [axes]
    for ax, (path, label) in zip(axes, avail):
        img = np.array(Image.open(path).convert("L"))
        ax.imshow(img, cmap="gray"); ax.set_title(label); ax.axis("off")

    plt.tight_layout()
    out = "results/stego_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {out}")


# ── Main ──────────────────────────────────────────────────────

def main():
    print("Generating result plots from real NIH ChestX-ray experiments...")

    # Load experiment results if available
    data = {}
    if os.path.exists("results/experiment_results.json"):
        with open("results/experiment_results.json") as f:
            data = json.load(f)
    else:
        print("  No experiment_results.json — run experiments/run_experiments.py first")

    plot_training_curves()
    plot_capacity_distortion(data.get("capacity_distortion", []))
    plot_method_comparison(data.get("method_comparison", {}))
    plot_per_image(data.get("per_image_eval", []))
    plot_sensitivity_pipeline()
    plot_stego_comparison()

    print("\n✓ All plots saved to results/")


if __name__ == "__main__":
    main()
