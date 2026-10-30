# 02. Stage 2: generation with DiT<sup>DH</sup>-S

- **Status:** done
- **Paper:** Sec. 5 (DiT<sup>DH</sup>); the paper's main results use DiT<sup>DH</sup>-XL on ImageNet 256

## Goal

Train DiT<sup>DH</sup> on the latent of the RAE from [01](01_stage1_decoder_capacity.md) (Tiny encoder + XL decoder with GAN)
and measure generation quality (FID).

## Setup

| Item | Value |
|---|---|
| Latent | `[192, 8, 8]`, normalized with per-channel, per-position mean / variance from train 50k (with horizontal flips) |
| Model | the official DiT<sup>DH</sup>-S architecture (hidden [384, 2048], depth [12, 2], heads [6, 16]), patch 1, 10 classes; 193M parameters |
| Training | 2000 epochs, batch 512 (97 steps/epoch), AdamW 2e-4 (linear, 40 warmup epochs), EMA 0.9995, bf16, label dropout 0.1 |
| Transport | linear path, velocity prediction, logit-normal time, time shift √(12288/4096) ≈ 1.73 |
| Sampling | ODE (Euler, 50 steps) |
| FID | against Inception statistics of train 50k (the real test 10k images score 3.15) |

Configs: `configs/stage2/training/CIFAR10/DiTDH-S_dinov2-r0_decXL-gan.yaml`, sampling:
`configs/stage2/sampling/CIFAR10/DiTDH-S_dinov2-r0_decXL-gan.yaml`

## Results

FID during training (EMA, no CFG, 10k samples with the test labels):

| Epoch | 0 | 500 | 1000 | 1500 |
|---|---|---|---|---|
| FID (10k) | 238.0 | **11.42** | 13.10 | 14.71 |

FID after training (final-epoch EMA, 50k samples, 5000 per class):

| CFG scale | FID (50k) |
|---|---|
| 1.0 (no CFG) | **13.59** |
| 3.0 | 23.50 |

- **FID is best at epoch 500 and gets worse afterwards**, while the training loss keeps decreasing (0.233 at the end). Possibly overfitting.
- **CFG 3.0 makes FID worse.** We think the guidance is too strong and reduces diversity.
- Not done: smaller CFG scales (1.25–2), and a 50k FID of the epoch-500 checkpoint.

## Reproduce

```bash
scripts/train_stage2.sh configs/stage2/training/CIFAR10/DiTDH-S_dinov2-r0_decXL-gan.yaml ditdhS_decXLgan
scripts/eval_stage2.sh configs/stage2/sampling/CIFAR10/DiTDH-S_dinov2-r0_decXL-gan.yaml ditdhS_decXLgan_last_cfg1.0 1.0
```

Training takes about 31 h; evaluating 50k samples takes about 20 min without CFG and 35 min with CFG.
