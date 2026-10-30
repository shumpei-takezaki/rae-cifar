"""Figures for the radial/angular split (cf. RJF Fig. 4): scripts/overfit_single_image.py runs with --objective full / angular.

Usage: uv run python scripts/plot_radial_angular.py samples/overfit/tiny_n192_full samples/overfit/tiny_n192_angular \
    docs/figures/05_radial_angular/radial_angular_tiny_n192
Writes <prefix>.png, <prefix>_samples.png (angular run) and <prefix>.json (final numbers and sample PSNR).
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
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated pair)
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue, validated


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / ((a - b) ** 2).mean())


def sample_psnr(run, n_rows, k):  # mean PSNR of samples vs the RAE recon, read back from grid.png (1px padding, 32px tiles)
    g = np.asarray(Image.open(run / "grid.png").convert("RGB")).astype(float)
    tile = lambda row, col: g[row * 33 + 1: row * 33 + 33, col * 33 + 1: col * 33 + 33]
    return [float(np.mean([psnr(tile(i, c), tile(i, 1)) for c in range(2, k)])) for i in range(n_rows)]


def main():
    full_run, ang_run, prefix = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    F, A = (json.loads((p / "results.json").read_text()) for p in (full_run, ang_run))
    n, k = F["token_dim"], F["args"]["num_samples"] + 2
    widths = [x["width"] for x in F["results"]]
    psnr_f, psnr_a = sample_psnr(full_run, len(widths), k), sample_psnr(ang_run, len(widths), k)
    col = lambda R, key: [x[key] for x in R["results"]]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.7))
    for a in ax:
        a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
        for s in ("top", "right"): a.spines[s].set_visible(False)

    # (a) full objective: final total / radial / angular vs width
    dd = np.linspace(min(widths), max(widths), 200)
    ax[0].plot(dd, np.clip((n - dd) / n, 0, None), "--", color=INK2, lw=1.5, label="bound (n−d)/n")
    for key, color, ls, label in (("final_loss", INK, "-", "total"), ("final_radial", ORANGE, "-", "radial part"),
                                  ("final_angular", BLUE, "-", "angular part")):
        ax[0].plot(widths, col(F, key), ls, marker="o", color=color, lw=2, ms=5, markeredgecolor=SURFACE, label=label)
    ax[0].axvline(n, color=INK2, lw=0.8)
    ax[0].set(xlabel="DiT width d", ylabel="loss (per dim, last 200 steps)", title="(a) Full objective: loss split")
    ax[0].legend(frameon=False, loc="upper right")

    # (b) final angular loss vs width: full vs angular-only objective
    ax[1].plot(widths, col(F, "final_angular"), "-o", color=BLUE, lw=2, ms=5, markeredgecolor=SURFACE, label="trained on full loss")
    ax[1].plot(widths, col(A, "final_angular"), "-o", color=ORANGE, lw=2, ms=5, markeredgecolor=SURFACE, label="trained on angular loss only")
    ax[1].axvline(n, color=INK2, lw=0.8)
    ax[1].set(yscale="log", ylim=(5e-5, 1), xlabel="DiT width d", ylabel="angular loss (log)", title="(b) Angular loss vs. width")
    ax[1].legend(frameon=False, loc="lower left")

    # (c) angular-loss curves of the angular-only runs
    steps = np.arange(len(A["results"][0]["angular_curve"])) * 100 + 50
    for color, d in zip(RAMP, [64, 128, 192, 384]):
        ax[2].plot(steps, next(x for x in A["results"] if x["width"] == d)["angular_curve"], color=color, lw=2, label=f"d = {d}")
    ax[2].set(yscale="log", ylim=(5e-5, 3), xlabel="step", ylabel="angular loss (100-step mean, log)",
              title="(c) Angular-only training")
    ax[2].legend(frameon=False, loc="upper right", ncol=2)

    # (d) sample PSNR vs the RAE reconstruction
    ax[3].plot(widths, psnr_f, "-o", color=BLUE, lw=2, ms=5, markeredgecolor=SURFACE, label="full objective")
    ax[3].plot(widths, psnr_a, "-o", color=ORANGE, lw=2, ms=5, markeredgecolor=SURFACE, label="angular-only (tangent sampling)")
    ax[3].axvline(n, color=INK2, lw=0.8)
    ax[3].set(xlabel="DiT width d", ylabel="PSNR [dB]", title="(d) Samples vs. RAE reconstruction")
    ax[3].legend(frameon=False, loc="lower right")
    fig.suptitle(f"Radial / angular split of the velocity error, single-image overfitting (token dim n = {n}; plain DiT, "
                 f"12 layers, {F['args']['steps']} steps)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{prefix}.png", dpi=150)

    # labeled, upscaled sample grid of the angular-only run
    s = 3
    grid = Image.open(ang_run / "grid.png").convert("RGB")
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

    rows = lambda R, ps: [dict(width=x["width"], total=x["final_loss"] if R is F else None, radial=x["final_radial"],
                               angular=x["final_angular"], sample_latent_mse=x["sample_latent_mse"], sample_psnr=p)
                          for x, p in zip(R["results"], ps)]
    summary = dict(token_dim=n, full=rows(F, psnr_f), angular=rows(A, psnr_a))
    Path(f"{prefix}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
