"""Figure for experiment 07 (Stage 2 model size): FID during training and final 50k FID vs. parameters.

Usage: uv run python scripts/plot_stage2_size.py docs/figures/07_stage2_model_size/fid.json docs/figures/07_stage2_model_size/fid
Input: {run: {params_M, fid10k_by_epoch: [[epoch, fid], ...], fid50k_final}} (FIDs parsed from the training logs and eval_stage2.sh).
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (all-pairs validated in the dataviz reference palette)


def main():
    runs = json.load(open(sys.argv[1]))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for a in ax:
        a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
        for s in ("top", "right"): a.spines[s].set_visible(False)

    # (a) FID (10k) during training, epoch 0 left out (FID ~238 for all)
    for color, (name, r) in zip(SERIES, runs.items()):
        ep, fid = zip(*[(e, f) for e, f in r["fid10k_by_epoch"] if e > 0])
        ax[0].plot(ep, fid, "-o", color=color, lw=2, ms=5, markeredgecolor=SURFACE, label=f"{name.replace('^DH', '$^{\\mathrm{DH}}$')} ({r['params_M']:.0f}M)")
    ax[0].set(xlabel="epoch", ylabel="FID (10k samples, no CFG)", title="(a) FID during training", ylim=(9, 17))
    ax[0].legend(frameon=False, loc="upper right")

    # (b) final FID (50k) vs. parameters
    for color, (name, r) in zip(SERIES, runs.items()):
        ax[1].scatter(r["params_M"], r["fid50k_final"], s=60, color=color, edgecolor=SURFACE, zorder=3)
        ax[1].annotate(f"{r['fid50k_final']:.2f}", (r["params_M"], r["fid50k_final"]), textcoords="offset points",
                       xytext=(0, 8), ha="center", color=INK)
    ax[1].set(xscale="log", xlim=(10, 400), ylim=(6, 15), xlabel="parameters [M] (log)", ylabel="FID (50k samples, no CFG)",
              title="(b) Final epoch (2000)")
    fig.tight_layout()
    fig.savefig(f"{sys.argv[2]}.png", dpi=150)


if __name__ == "__main__":
    main()
