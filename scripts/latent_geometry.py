"""Hyperspherical geometry of the RAE latent (cf. RJF, "Learning on the Manifold", arXiv 2602.10099, Fig. 2-3).

(a) Per-token norm distributions: raw encoder tokens (LayerNorm without affine), the normalized Stage-2 latent, and the
    Gaussian prior. (b) Per-token norm along the interpolation path between a latent and Gaussian noise: the linear
    (Euclidean) path used by flow matching vs. the spherical (slerp) path.

Usage (repo root): PYTHONPATH=src uv run python scripts/latent_geometry.py \
    --config configs/stage2/sampling/CIFAR10/DiTDH-S_dinov2-r0_decXL-gan.yaml --out docs/figures/04_latent_geometry/latent_geometry_tiny
Writes <out>.png and <out>.json.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (all-pairs validated in the dataviz reference palette)


def tokens(z):  # [B, C, H, W] -> [B*H*W, C]
    return z.flatten(2).transpose(1, 2).reshape(-1, z.shape[1])


def slerp(a, b, t):  # per-token spherical interpolation, radius interpolated linearly
    ra, rb = a.norm(dim=-1, keepdim=True), b.norm(dim=-1, keepdim=True)
    ua, ub = a / ra, b / rb
    om = torch.arccos((ua * ub).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7))
    u = (torch.sin((1 - t) * om) * ua + torch.sin(t * om) * ub) / torch.sin(om)
    return u * ((1 - t) * ra + t * rb)


def stats(x):
    x = x.double()
    return dict(mean=x.mean().item(), std=x.std().item(), rel_std=(x.std() / x.mean()).item(), min=x.min().item(), max=x.max().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", default="data/cifar10/test.npz")
    p.add_argument("--num-images", type=int, default=10000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.manual_seed(0)
    cfg, *_ = parse_configs(OmegaConf.load(args.config))
    rae = instantiate_from_config(cfg).cuda().eval()
    imgs = np.load(args.data)["arr_0"][: args.num_images]
    zn = []
    with torch.no_grad():
        for i in range(0, len(imgs), 500):
            x = torch.from_numpy(imgs[i:i + 500]).permute(0, 3, 1, 2).float().cuda() / 255
            zn.append(rae.encode(x).float())
    zn = torch.cat(zn)  # normalized latent (Stage-2 input)
    zr = zn * torch.sqrt(rae.latent_var.cuda() + rae.eps) + rae.latent_mean.cuda()  # raw encoder output
    C = zn.shape[1]
    tn, tr = tokens(zn), tokens(zr)
    noise = torch.randn_like(tn)
    norms = {"raw encoder tokens": tr.norm(dim=-1), "normalized latent (Stage 2)": tn.norm(dim=-1), "Gaussian prior": noise.norm(dim=-1)}

    ts = torch.linspace(0, 1, 21)
    path = {}
    for name, a in (("linear", None), ("slerp", None)):
        m, s = [], []
        for t in ts:
            xt = (1 - t) * tn + t * noise if name == "linear" else slerp(tn, noise, t.item())
            r = xt.norm(dim=-1)
            m.append(r.mean().item()); s.append(r.std().item())
        path[name] = dict(mean=m, std=s)
    cos = torch.nn.functional.cosine_similarity(tn, noise, dim=-1)

    result = dict(num_images=len(imgs), token_dim=C, sqrt_C=C ** 0.5, num_tokens=tn.shape[0],
                  token_norm={k: stats(v) for k, v in norms.items()},
                  cos_data_noise=stats(cos), t=ts.tolist(), path_token_norm=path,
                  midpoint_linear_over_endpoint=path["linear"]["mean"][10] / path["linear"]["mean"][0])
    with open(f"{args.out}.json", "w") as f:
        json.dump(result, f, indent=1)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for a in ax:
        a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)

    # (a) norm distributions (log density); the raw tokens are a delta at sqrt(C)
    bins = np.linspace(9, 18, 181)
    for color, (name, v) in zip(SERIES[1:], list(norms.items())[1:]):
        ax[0].hist(v.cpu().numpy(), bins=bins, density=True, histtype="step", lw=2, color=color, label=name)
    ax[0].vlines(C ** 0.5, 1e-3, 5, color=SERIES[0], lw=2, label=f"raw encoder tokens (all at √{C} = {C ** 0.5:.2f})")
    ax[0].set(yscale="log", ylim=(1e-3, 3e3), xlabel="per-token L2 norm", ylabel="density (log)",
              title=f"(a) Token norms (CIFAR-10 test, {len(imgs)} images × 64 tokens)")
    ax[0].legend(frameon=False, loc="upper left", fontsize=8)

    # (b) norm along the path
    tt = ts.numpy()
    for color, name, label in ((SERIES[1], "linear", "linear (Euclidean) path"), (SERIES[0], "slerp", "spherical (slerp) path")):
        m, s = np.array(path[name]["mean"]), np.array(path[name]["std"])
        ax[1].fill_between(tt, m - s, m + s, color=color, alpha=0.15, lw=0)
        ax[1].plot(tt, m, color=color, lw=2, label=label)
    ax[1].plot(tt, C ** 0.5 * np.sqrt((1 - tt) ** 2 + tt ** 2), "--", color=INK2, lw=1.2, label="√((1−t)² + t²) · √C")
    mid = path["linear"]["mean"][10]
    ax[1].annotate(f"{mid:.2f} ({mid / path['linear']['mean'][0]:.0%})", (0.5, mid), textcoords="offset points",
                   xytext=(0, -16), ha="center", color=INK)
    ax[1].set(ylim=(8, 16), xlabel="t  (0 = data, 1 = noise)", ylabel="per-token L2 norm",
              title="(b) Norm along the path data → noise (mean ± std)")
    ax[1].legend(frameon=False, loc="upper center", fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    print(json.dumps({k: result[k] for k in ("token_norm", "cos_data_noise", "midpoint_linear_over_endpoint")}, indent=1))


if __name__ == "__main__":
    main()
