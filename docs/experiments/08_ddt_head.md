# 08. Is the DDT head needed? Plain LightningDiT vs. DiT<sup>DH</sup>

- **Status:** done
- **Paper:** RAE Sec. 5 and Table 6 (DiT-XL vs. DiT<sup>DH</sup>-XL); RJF (Learning on the Manifold) uses plain LightningDiT

## Goal

The DDT head exists to satisfy the width requirement (width ≥ token dim n) cheaply. With our latent (n = 192) a plain DiT of
width 384 already satisfies it, so by RAE Theorem 1 the head is not required. In RAE Table 6, however, the head helps even when
the body is wide enough (DINOv2-S, n = 384, DiT-XL width 1152: 3.50 → 2.42). We check whether the head helps on CIFAR-10, using
the best model of [07](07_stage2_model_size.md) (DiT<sup>DH</sup>-S with a 512 head, FID 7.94) as the reference.

## Setup

Plain `LightningDiT` (no DDT head; velocity predicted directly from the body), patch 1 (64 tokens), same recipe as
[07](07_stage2_model_size.md): 2000 epochs, batch 512, bf16, FID (10k) every 250 epochs, then 50k FID (no CFG, CFG 3.0) and
memorization. LightningDiT and the DDT body share the time embedding (Gaussian Fourier), fixed sin-cos position embedding,
RoPE, SwiGLU and RMSNorm; the only other difference is that the final layer is conditioned globally (t, y) instead of per token.

| Model | Body (width, depth, heads) | DDT head | Params |
|---|---|---|---|
| DiT<sup>DH</sup>-S, head 512 ([07](07_stage2_model_size.md)) | 384, 12, 6 | 512 × 2 | 42.8M |
| **LightningDiT-S/1** | 384, 12, 6 | none | 32.8M |
| **LightningDiT-Ti/1** | 192, 12, 3 | none | ≈ 8M |

- LightningDiT-S/1 vs. the 512-head model: the same body with and without the head.
- LightningDiT-Ti/1 has width = n, the same width / n ratio as RJF's LightningDiT-B/1 on DINOv2-B; it is also the baseline for an RJF follow-up.

Configs: `configs/stage2/{training,sampling}/CIFAR10/LightningDiT-{S,Ti}_dinov2-r0_decXL-gan.yaml`

## Results

FID (10k samples, no CFG) during training and the final 50k FID:

| Model | Params | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | Final, 50k (no CFG) |
|---|---|---|---|---|---|---|---|---|---|
| DiT<sup>DH</sup>-S, head 512 ([07](07_stage2_model_size.md)) | 42.8M | 11.47 | 10.84 | 10.58 | 10.19 | 10.32 | 10.04 | 9.89 | **7.94** |
| LightningDiT-S/1 (no head) | 32.8M | 14.02 | 12.47 | 11.82 | 11.49 | 11.17 | 11.00 | 11.04 | 8.70 |
| LightningDiT-Ti/1 (no head, width = n) | ≈ 8M | 29.2 | 43.9 | 38.8 | 37.0 | 38.0 | 167.6 | 221.3 | 235.6 (diverged) |

With CFG 3.0 (50k): head 512 14.34, LightningDiT-S/1 14.56, LightningDiT-Ti/1 159.0.

Memorization (50k samples each; see [07](07_stage2_model_size.md) for the criteria):

| | Gu: d1/d2 < 1/3 | Carlini: score < 1 | Carlini: distinct train images | Median d1 |
|---|---|---|---|---|
| Head 512, no CFG / CFG 3.0 | 0% / 0% | 0.026% / 0.046% | 13 / 19 | 8.96 / 8.66 |
| LightningDiT-S/1, no CFG / CFG 3.0 | 0% / 0% | 0.008% / 0.016% | 4 / 8 | 8.94 / 8.58 |

Neither model memorizes by the Gu criterion; the plain model makes slightly fewer near-copies than the 512-head model
(4 vs. 13 distinct training images). The diverged Ti/1 is left out (its samples are degenerate).

- **Removing the head costs some FID** (7.94 → 8.70) with the same body, so a head of moderate width (512 ≈ 2.7n) helps
  even though the body alone already satisfies the width requirement (384 = 2n). This agrees in direction with RAE Table 6,
  with a much smaller gap than the oversized head of [07](07_stage2_model_size.md) (13.59). The plain model is also
  10M parameters smaller, so part of the gap may be capacity.
- **The plain DiT with width = n (Ti/1) was unstable.** Its training loss jumped at epoch ~290 (1.01 → 2.10), recovered to
  about 0.82, and rose again from epoch ~1200 to about 1.9 (the other models end at 0.35–0.41); FID followed. We record this
  as is and do not investigate the cause here. Candidates: a plain DiT whose width only just meets the requirement is
  hard to train (RJF reports FID 24.3 for plain LightningDiT-B/1 with width = n on DINOv2-B), or an optimization instability
  of the small model (bf16, lr 2e-4, no QK-norm). It will be revisited as the baseline of the RJF follow-up.
