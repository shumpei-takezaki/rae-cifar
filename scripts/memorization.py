"""Memorization of generated samples: nearest neighbours in the training set (pixel L2).

For each sample x, d1 / d2 are the L2 distances to its nearest and second-nearest training images (train 50k plus their
horizontal flips, since Stage 2 trains with flips; a flipped copy of the same image is not counted as the second neighbour).
A sample counts as memorized if d1 / d2 < 1/3 (empirical threshold of Gu et al., "On Memorization in Diffusion Models",
arXiv 2310.02664, who credit Carlini et al. 2023).
Also the score of Carlini et al., "Extracting Training Data from Diffusion Models" (arXiv 2301.13188, Sec. 5.1):
d1 / (alpha * mean distance to the n = 50 nearest training images), alpha = 0.5. The paper gives no numeric cutoff (an
"abnormally low" distance with an adaptive threshold); we count score < 1 (d1 below half the mean 50-NN distance) and also
report the number of distinct training images matched.
CIFAR-10 test images are scored the same way as a reference for non-memorized data.

Usage: PYTHONPATH=src uv run python scripts/memorization.py --out docs/figures/<exp>/memorization samples/stage2/<name>.npz ...
Writes <out>.json (per input: memorized fraction, d1 and d1/d2 statistics) and <out>_nn.png (the closest samples and their neighbours).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def knn(q, ref, n_img, k=50, bs=1000):
    """Distances to the k nearest training images (sorted) and the nearest index; ref = [train; flipped train], index i and
    i + n_img are one image."""
    r = ref.flatten(1)
    rn = (r * r).sum(1)
    dk, idx = [], []
    for i in range(0, len(q), bs):
        x = q[i:i + bs].flatten(1)
        d = ((x * x).sum(1, keepdim=True) - 2 * x @ r.T + rn).clamp_min(0)
        d = torch.minimum(d[:, :n_img], d[:, n_img:])  # one distance per training image (best of the two flips)
        v, j = d.topk(k, dim=1, largest=False)
        dk.append(v.sqrt()); idx.append(j[:, 0])
    return torch.cat(dk), torch.cat(idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("samples", nargs="+")
    p.add_argument("--train", default="data/cifar10/train.npz")
    p.add_argument("--test", default="data/cifar10/test.npz")
    p.add_argument("--n", type=int, default=50000, help="samples scored per input (random subset)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    load = lambda f: torch.from_numpy(np.load(f)["arr_0"]).cuda().float() / 255  # [N, 32, 32, 3]
    train = load(args.train)
    ref = torch.cat([train, train.flip(2)])  # flip along width
    rng = np.random.default_rng(0)
    out, rows = {}, []
    for f in [args.test] + args.samples:
        x = load(f)
        x = x[torch.from_numpy(rng.choice(len(x), min(args.n, len(x)), replace=False)).cuda()]
        dk, idx = knn(x, ref, len(train))
        d1, ratio = dk[:, 0], dk[:, 0] / dk[:, 1]
        carlini = d1 / (0.5 * dk.mean(1))
        name = "CIFAR-10 test (reference)" if f == args.test else Path(f).stem
        out[name] = dict(n=len(x), memorized_frac=(ratio < 1 / 3).float().mean().item(),
                         ratio_median=ratio.median().item(), ratio_p01=ratio.quantile(0.01).item(),
                         d1_median=d1.median().item(), d1_p01=d1.quantile(0.01).item(),
                         carlini_frac=(carlini < 1).float().mean().item(), carlini_unique_train=idx[carlini < 1].unique().numel(),
                         carlini_min=carlini.min().item(), carlini_p001=carlini.quantile(0.001).item(),
                         carlini_by_threshold={str(th): dict(frac=(carlini < th).float().mean().item(),
                                                             unique_train=idx[carlini < th].unique().numel())
                                               for th in (0.6, 0.8, 1.0, 1.2)})
        print(name, json.dumps(out[name]))
        k = carlini.argsort()[:8]  # the 8 samples with the lowest Carlini score
        rows.append([x[k], train[idx[k]], name])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(out, indent=1))

    # grid: per input, a row of the 8 lowest-score samples above their nearest training images
    tile = lambda t: (t.clamp(0, 1) * 255).byte().cpu().numpy()
    pad = 2
    grid = np.full((len(rows) * 2 * (32 + pad) + len(rows) * 6, 8 * (32 + pad), 3), 255, np.uint8)
    y = 0
    for s, nn, _ in rows:
        for t in (s, nn):
            for c, im in enumerate(tile(t)):
                grid[y:y + 32, c * (32 + pad):c * (32 + pad) + 32] = im
            y += 32 + pad
        y += 6
    img = Image.fromarray(grid).resize((grid.shape[1] * 3, grid.shape[0] * 3), Image.NEAREST)
    lw = 330  # left margin: input name; upper row = samples, lower row = their nearest training images
    out_img = Image.new("RGB", (img.width + lw, img.height), "white")
    out_img.paste(img, (lw, 0))
    draw = ImageDraw.Draw(out_img)
    for i, (_, _, name) in enumerate(rows):
        y0 = 3 * i * (2 * (32 + pad) + 6)
        draw.text((6, y0 + 40), name[:48], fill="black")
        draw.text((6, y0 + 40 + 3 * (32 + pad)), "nearest train image", fill="#52514e")
    out_img.save(f"{args.out}_nn.png")


if __name__ == "__main__":
    main()
