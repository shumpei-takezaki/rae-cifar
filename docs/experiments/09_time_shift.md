# 09. Ablation: dimension-dependent noise-schedule shift

- **Status:** done
- **Paper:** RAE Table 4 (gFID 23.08 without shift vs. 4.81 with shift, DINOv2-B)

## Goal

RAE shifts the time distribution by the effective data dimension m (tokens × token dim): t_m = α t / (1 + (α − 1) t) with
α = √(m / 4096). For DINOv2-B at 256px, m = 256 × 768 and α ≈ 6.93. Our latent is much smaller (m = 8 × 8 × 192 = 12,288,
α ≈ 1.73), so the effect may be small. We train the reference model of [07](07_stage2_model_size.md) without the shift.

## Setup

DiT<sup>DH</sup>-S with a 512 head ([07](07_stage2_model_size.md), FID 7.94), same recipe (2000 epochs), with
`time_dist_shift_dim: 4096` (α = 1). In this code base the shift applies both to the training time distribution
(`train.py`) and to the sampling time grid (`sample_ddp.py`), so both are unshifted. Evaluation: 50k FID (no CFG) and memorization.

Configs: `configs/stage2/{training,sampling}/CIFAR10/DiTDH-S-h512-noshift_dinov2-r0_decXL-gan.yaml`

## Results

FID (10k samples, no CFG) during training and the final 50k FID:

| Model | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | Final, 50k (no CFG) |
|---|---|---|---|---|---|---|---|---|
| With shift (α ≈ 1.73, [07](07_stage2_model_size.md)) | 11.47 | 10.84 | 10.58 | 10.19 | 10.32 | 10.04 | 9.89 | **7.94** |
| Without shift (α = 1) | 12.75 | 12.38 | 11.75 | 11.62 | 11.46 | 11.15 | 10.75 | 9.01 |

Memorization (50k samples; criteria as in [07](07_stage2_model_size.md)):

| | Gu: d1/d2 < 1/3 | Carlini: score < 1 | Carlini: distinct train images | Median d1 |
|---|---|---|---|---|
| With shift | 0% | 0.026% | 13 | 8.96 |
| Without shift | 0% | 0.004% | 2 | 8.94 |

- **The shift helps, as in the paper, but much less** (FID 9.01 → 7.94; RAE Table 4: 23.08 → 4.81). This is expected: our latent is
  much smaller than the paper's, so the shift itself is much smaller (α ≈ 1.73 vs. about 6.93).
- Without the shift, training still works and FID keeps improving until the end.
- Neither model memorizes by the Gu criterion. The Carlini score counts few near-copies in both, and the difference (13 vs. 2) is
  too small to interpret.
- One seed per model. The training and sampling shifts were removed together, so their separate effects are not known.

## Reproduce

```bash
scripts/train_stage2.sh configs/stage2/training/CIFAR10/DiTDH-S-h512-noshift_dinov2-r0_decXL-gan.yaml ditdh_s_h512_noshift_decXLgan
scripts/eval_stage2.sh configs/stage2/sampling/CIFAR10/DiTDH-S-h512-noshift_dinov2-r0_decXL-gan.yaml ditdh_s_h512_noshift_decXLgan_last_cfg1.0 1.0
```
