"""Single-image overfitting (RAE paper Sec. 4.1 / Table 3): can a plain DiT of width d fit ONE RAE latent of token dim n?

The paper's Theorem 1 says the per-dim flow-matching loss on a single image is bounded below by (n - d) / n when d < n.
For each width we train a plain LightningDiT (no DDT head) on copies of one latent, report the final loss next to the
bound, and decode samples with the RAE decoder into one grid (first column: input image, second: RAE reconstruction).

The velocity error is also split per token into a radial part (along x_t / ||x_t||) and an angular part (the rest), as in
RJF ("Learning on the Manifold", arXiv 2602.10099, Eq. 4 / Fig. 4; the paper does not define the radial direction, we use
x_t). --objective angular trains on the angular part only (RJF's "radial loss masked"); its samples then integrate the
tangent-projected velocity, since the radial velocity is never trained.

Usage (repo root): PYTHONPATH=src uv run python scripts/overfit_single_image.py \
    --config configs/stage2/sampling/CIFAR10/DiTDH-S_dinov2-r0_decXL-gan.yaml --widths 64 96 128 160 192 256 384
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid, save_image

from stage2.models.lightningDiT import LightningDiT
from stage2.transport import Sampler, create_transport
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Stage 2 config: stage_1 (RAE + latent stats), transport, sampler, misc")
    p.add_argument("--image", default="data/cifar10/train/airplane/00029.png")
    p.add_argument("--label", type=int, default=0)
    p.add_argument("--widths", type=int, nargs="+", default=[64, 96, 128, 160, 192, 256, 384])
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--head-dim", type=int, default=32)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--objective", choices=["full", "angular"], default="full")
    p.add_argument("--out", default="samples/overfit")
    args = p.parse_args()

    dev = torch.device("cuda")
    autocast = dict(device_type="cuda", dtype=torch.bfloat16)  # rae-cifar trains in bf16
    rae_cfg, _, transport_cfg, sampler_cfg, _, misc, _, _ = parse_configs(OmegaConf.load(args.config))
    rae = instantiate_from_config(rae_cfg).to(dev).eval()
    misc = dict(misc)
    shift = math.sqrt(misc["time_dist_shift_dim"] / misc["time_dist_shift_base"])
    transport = create_transport(**dict(transport_cfg.get("params", {})), time_dist_shift=shift)
    sample_fn = Sampler(transport).sample_ode(**dict(sampler_cfg.get("params", {})))

    x = torch.from_numpy(np.asarray(Image.open(args.image).convert("RGB"))).permute(2, 0, 1).float().div(255)[None].to(dev)
    with torch.no_grad():
        z = rae.encode(x)  # normalized latent [1, C, H, W]
        recon = rae.decode(z).clamp(0, 1)
    C, H, W = z.shape[1:]
    print(f"image {args.image}  latent {tuple(z.shape[1:])}  token dim n={C}  time shift {shift:.3f}")

    def radial_angular(v, xt):  # split [B, C, H, W] per token into the part along x_t and the rest
        r = xt / xt.norm(dim=1, keepdim=True)
        vr = (v * r).sum(1, keepdim=True) * r
        return vr, v - vr

    def step_losses(model, zb, kw):  # same t / path as transport.training_losses
        t, x0, x1 = transport.sample(zb)
        t, xt, ut = transport.path_sampler.plan(t, x0, x1)
        er, ea = radial_angular(model(xt, t, **kw) - ut, xt)
        radial, angular = er.pow(2).mean(), ea.pow(2).mean()  # per-dim means; radial + angular = full loss
        return radial + angular, radial, angular

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, results = [], []
    for d in args.widths:
        torch.manual_seed(args.seed)
        model = LightningDiT(input_size=H, patch_size=1, in_channels=C, hidden_size=d, depth=args.depth,
                             num_heads=d // args.head_dim, class_dropout_prob=0.0, num_classes=10).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
        zb = z.expand(args.batch_size, -1, -1, -1)
        yb = torch.full((args.batch_size,), args.label, device=dev)
        losses, rads, angs = [], [], []
        for step in range(args.steps):
            with torch.autocast(**autocast):
                full, radial, angular = step_losses(model, zb, dict(y=yb))
            loss = full if args.objective == "full" else angular
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item()); rads.append(radial.item()); angs.append(angular.item())
        final = float(np.mean(losses[-200:]))
        curve = lambda v: [float(np.mean(v[i:i + 100])) for i in range(0, args.steps, 100)]
        bound = max(C - d, 0) / C
        model.eval()
        with torch.no_grad(), torch.autocast(**autocast):
            noise = torch.randn(args.num_samples, C, H, W, device=dev, generator=torch.Generator(dev).manual_seed(args.seed))
            fn = model.forward if args.objective == "full" else (lambda xt, t, **kw: radial_angular(model(xt, t, **kw), xt)[1])
            zs = sample_fn(noise, fn, y=yb[: args.num_samples])[-1].float()
            imgs = rae.decode(zs).float().clamp(0, 1)
        latent_mse = float(((zs - z) ** 2).mean())
        results.append(dict(width=d, heads=d // args.head_dim, params_M=sum(p.numel() for p in model.parameters()) / 1e6,
                            final_loss=final, bound=bound, sample_latent_mse=latent_mse, loss_curve=curve(losses),
                            final_radial=float(np.mean(rads[-200:])), final_angular=float(np.mean(angs[-200:])),
                            radial_curve=curve(rads), angular_curve=curve(angs)))
        print(f"d={d:4d}  {args.objective} loss(last 200)={final:.4f}  radial={np.mean(rads[-200:]):.4f}  "
              f"angular={np.mean(angs[-200:]):.4f}  bound (n-d)/n={bound:.4f}  sample latent MSE={latent_mse:.4f}", flush=True)
        rows.append(torch.cat([x, recon, imgs]))
        del model, opt
    save_image(make_grid(torch.cat(rows), nrow=args.num_samples + 2, padding=1), out / "grid.png")
    (out / "results.json").write_text(json.dumps(dict(args=vars(args), token_dim=C, results=results), indent=1))
    print(f"wrote {out}/grid.png and results.json  (rows = widths {args.widths}; cols = input, RAE recon, samples)")


if __name__ == "__main__":
    main()
