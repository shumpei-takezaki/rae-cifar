# 10. Ablation: noise augmentation in decoder training

- **Status:** done
- **Paper:** RAE Table 5 and Appendix Table 15c (DINOv2-B: τ = 0 → gFID 4.81 / rFID 0.49; τ = 0.8 → 4.28 / 0.57)

## Goal

RAE trains the decoder on noisy latents so that it tolerates the imperfect latents produced by the diffusion model: noise
augmentation costs some reconstruction quality (rFID) but improves generation (gFID). We check this on CIFAR-10.

## Setup

- A ViT-XL decoder trained exactly like the one used for Stage 2 ([01](01_stage1_decoder_capacity.md): 400 epochs without GAN,
  then 100 epochs of GAN fine-tuning), but with `noise_tau: 0` in both stages. Only the GAN versions are compared.
- Noise in this code base (`src/stage1/rae.py`, unchanged from upstream): σ = τ · U(0, 1) per sample, noise σ · N(0, I).
  The paper describes σ ~ |N(0, τ²)| instead; we keep the code's version.
- Generation does not need a new Stage 2: the encoder, the latent and its normalization are unchanged, so the 50k latents of the
  reference Stage 2 model ([07](07_stage2_model_size.md), DiT<sup>DH</sup>-S with a 512 head, same sampling seed) are decoded
  with each decoder.
- Metrics: reconstruction on test and train (PSNR, SSIM, LPIPS, rFID; `scripts/eval_stage1.sh`) and 50k gFID (no CFG).

Configs: `configs/stage1/training/cifar_dinov2_decXL_{nogan,gan_ft}_nonoise.yaml`,
`configs/stage1/pretrained/cifar_dinov2_decXL_gan_ft_nonoise.yaml`, `configs/stage2/sampling/CIFAR10/DiTDH-S-h512_dinov2-r0_decXL-gan-nonoise.yaml`

## Results

XL decoder with GAN, final-epoch EMA. Train rFID is given on 50k / a random 10k subset. gFID: 50k samples, no CFG.

| Training noise | Test PSNR | Test SSIM | Test LPIPS | Test rFID | Train PSNR | Train LPIPS | Train rFID | **gFID** |
|---|---|---|---|---|---|---|---|---|
| τ = 0.8 (default) | 22.35 | 0.835 | 0.102 | 5.71 | 31.84 | 0.014 | 0.89 / 2.69 | **7.94** |
| τ = 0 (no noise) | **24.81** | **0.908** | **0.064** | **4.22** | **35.03** | **0.007** | **0.79 / 2.05** | 10.45 |

- **This reproduces the trade-off in the paper:** without noise, reconstruction improves on every metric, while generation gets
  worse. Decoding the same Stage 2 model, gFID goes from 7.94 with the noise-trained decoder to 10.45 without noise
  (RAE Table 5: 4.28 vs. 4.81).
- **The effect is larger than in the paper in both directions**, in particular for reconstruction (test PSNR +2.5 dB, test rFID
  5.71 → 4.22). We did not investigate why.
- The noise-free decoder has a better test rFID than any Tiny-encoder decoder in [01](01_stage1_decoder_capacity.md).
  Noise augmentation therefore explains part of the low test PSNR of those decoders.
- One seed per decoder. Only the τ = 0 and τ = 0.8 endpoints were tested, and the noise follows the code (uniform σ), not the paper's description.

## Reproduce

```bash
scripts/train_stage1.sh configs/stage1/training/cifar_dinov2_decXL_nogan_nonoise.yaml dinov2_decXL_nogan_nonoise
# extract the EMA decoder, then the GAN fine-tuning and its extraction, as in 01
scripts/train_stage1.sh configs/stage1/training/cifar_dinov2_decXL_gan_ft_nonoise.yaml dinov2_decXL_gan_ft_nonoise
scripts/eval_stage1.sh configs/stage1/pretrained/cifar_dinov2_decXL_gan_ft_nonoise.yaml dinov2_decXL_gan_ft_nonoise test   # and train
scripts/eval_stage2.sh configs/stage2/sampling/CIFAR10/DiTDH-S-h512_dinov2-r0_decXL-gan-nonoise.yaml ditdh_s_h512_decXLgan_nonoise_last_cfg1.0 1.0
```
