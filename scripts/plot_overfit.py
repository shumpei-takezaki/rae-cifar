"""Figures for the single-image overfitting run (scripts/overfit_single_image.py output).

Usage: uv run python scripts/plot_overfit.py samples/overfit/tiny_n192 docs/figures/03_single_image_overfit/overfit_tiny_n192
Writes <prefix>.png (final loss vs width with the (n-d)/n bound, loss curves, sample PSNR) and <prefix>_samples.png.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE = "#2a78d6"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue, validated (dataviz validate_palette --ordinal)


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / ((a - b) ** 2).mean())


def main():
    run, prefix = Path(sys.argv[1]), Path(sys.argv[2])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    r = json.loads((run / "results.json").read_text())
    n, res = r["token_dim"], r["results"]
    widths = [x["width"] for x in res]
    k = r["args"]["num_samples"] + 2  # columns: input, RAE recon, samples

    # sample PSNR vs the RAE reconstruction, read back from the saved grid (1px padding, 32px tiles)
    g = np.asarray(Image.open(run / "grid.png").convert("RGB")).astype(float)
    tile = lambda row, col: g[row * 33 + 1: row * 33 + 33, col * 33 + 1: col * 33 + 33]
    sample_psnr = [np.mean([psnr(tile(i, c), tile(i, 1)) for c in range(2, k)]) for i in range(len(widths))]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a in ax:
        a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
        for s in ("top", "right"): a.spines[s].set_visible(False)

    # (a) final loss vs width, with the bound
    dd = np.linspace(min(widths), max(widths), 200)
    ax[0].plot(dd, np.clip((n - dd) / n, 0, None), "--", color=INK2, lw=1.5, label="bound (n−d)/n")
    ax[0].plot(widths, [x["final_loss"] for x in res], "-o", color=BLUE, lw=2, ms=6,
               markeredgecolor=SURFACE, markeredgewidth=1.5, label="final loss (last 200 steps)")
    ax[0].axvline(n, color=INK2, lw=0.8); ax[0].text(n + 6, 0.32, f"d = n = {n}", color=INK2)
    ax[0].set(xlabel="DiT width d", ylabel="flow-matching loss (per dim)", title="(a) Final loss vs. width")
    ax[0].legend(frameon=False, loc="upper right")

    # (b) loss curves for four widths
    steps = np.arange(len(res[0]["loss_curve"])) * 100 + 50
    for color, d in zip(RAMP, [64, 128, 192, 384]):
        x = next(x for x in res if x["width"] == d)
        ax[1].plot(steps, x["loss_curve"], color=color, lw=2, label=f"d = {d}")
    ax[1].set(yscale="log", ylim=(2e-4, 3), xlabel="step", ylabel="loss (100-step mean, log)", title="(b) Training loss")
    ax[1].legend(frameon=False, loc="lower left", ncol=2)

    # (c) sample PSNR vs width
    ax[2].plot(widths, sample_psnr, "-o", color=BLUE, lw=2, ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax[2].axvline(n, color=INK2, lw=0.8)
    for d, p in zip(widths, sample_psnr):
        if d in (64, 192, 384): ax[2].annotate(f"{p:.1f}", (d, p), textcoords="offset points", xytext=(0, 7), ha="center", color=INK)
    ax[2].set(ylim=(27, 56), xlabel="DiT width d", ylabel="PSNR [dB]", title="(c) Samples vs. RAE reconstruction")
    fig.suptitle(f"Single-image overfitting on the CIFAR-10 RAE latent (token dim n = {n}; plain DiT, 12 layers, "
                 f"{r['args']['steps']} steps)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{prefix}.png", dpi=150)

    # labeled, upscaled sample grid
    s = 3
    grid = Image.open(run / "grid.png").convert("RGB")
    grid = grid.resize((grid.width * s, grid.height * s), Image.NEAREST)
    lw, th = 70, 22
    out = Image.new("RGB", (grid.width + lw, grid.height + th), SURFACE)
    out.paste(grid, (lw, th))
    dr = ImageDraw.Draw(out)
    for c, name in enumerate(["input", "RAE recon"] + [f"sample {i + 1}" for i in range(k - 2)]):
        dr.text((lw + c * 33 * s + 4, 5), name, fill=INK)
    for i, d in enumerate(widths):
        dr.text((6, th + i * 33 * s + 40), f"d = {d}", fill=INK)
    out.save(f"{prefix}_samples.png")
    print(f"wrote {prefix}.png and {prefix}_samples.png; sample PSNR:", [round(p, 1) for p in sample_psnr])


if __name__ == "__main__":
    main()
