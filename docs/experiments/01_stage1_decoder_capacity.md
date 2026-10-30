# 01. Stage 1: decoder size and GAN loss (RAE Table 1b)

- **Status:** done
- **Paper:** Table 1b (a larger decoder, ViT-B / L / XL, improves rFID; encoder fixed to DINOv2-B)

## Goal

Measure how the size of the RAE decoder (ViT-B / L / XL) and the GAN loss affect reconstruction on CIFAR-10.

## Setup

| Item | Value |
|---|---|
| Encoder | DINOv2 ViT-Tiny/4 from dino-cifar (no registers, frozen). Latent `[192, 8, 8]` |
| Decoder | official ViT decoder (B: 12 layers, width 768 / L: 24, 1024 / XL: 28, 1152), patch 4 |
| Without GAN | L1 + LPIPS, 400 epochs, batch 512, AdamW 2e-4 → 2e-5 (cosine), latent noise augmentation σ ≤ 0.8 |
| With GAN | 100 more epochs from the final EMA decoder without GAN; generator GAN loss from epoch 10 |
| Discriminator | official design (frozen DINO + trained heads, hinge loss, DiffAug), with the frozen backbone replaced by a 32px DINO (v1) ViT-Tiny/4 from dino-cifar (`recipe: 'cifar'`) |
| Precision | B in fp32; L and XL in bf16 (+ gradient checkpointing) |
| Evaluation | final-epoch EMA decoder; test 10k and train 50k images; rFID with the torch-fidelity Inception |

Configs: `configs/stage1/training/cifar_dinov2_dec{B,L,XL}_{nogan,gan_ft}.yaml`

## Results

| Decoder | Test PSNR | Test SSIM | Test LPIPS | Test rFID | Train PSNR | Train LPIPS | Train rFID (50k / random 10k) |
|---|---|---|---|---|---|---|---|
| B, no GAN | **22.90** | **0.858** | **0.092** | 5.51 | 30.85 | 0.018 | 2.30 / 4.37 |
| B, GAN | 22.62 | 0.846 | 0.095 | 6.57 | 29.71 | 0.022 | 2.02 / 4.19 |
| L, no GAN | 22.55 | 0.847 | 0.098 | **5.15** | 33.16 | 0.011 | 1.83 / 3.54 |
| L, GAN | 22.34 | 0.837 | 0.101 | 5.80 | 31.29 | 0.016 | 1.05 / 2.92 |
| XL, no GAN | 22.49 | 0.843 | 0.101 | 5.25 | **33.76** | **0.009** | 1.71 / 3.32 |
| XL, GAN | 22.35 | 0.835 | 0.102 | 5.71 | 31.84 | 0.014 | **0.89 / 2.69** |

- **On train, a larger decoder is better on every metric**, as in Table 1b of the paper. The GAN loss lowers the train rFID further.
- **On test, the paper's trend does not hold.** The best rFID is L without GAN (5.15), and the best PSNR / SSIM / LPIPS is B
  without GAN. On CIFAR-10 (50k images), larger decoders overfit train more.
  - For B without GAN, the test PSNR peaks around epoch 150 (23.4 dB) and then declines while the train PSNR keeps rising.
- **The discriminator stops discriminating as soon as the generator GAN loss is switched on.** While only the discriminator
  trains, its accuracy rises to about 0.7; within 20 steps of enabling the GAN loss it drops back to 0.5 and stays there (B, L
  and XL alike). We think the generator GAN weight (adaptive weight × 0.75, about 8–20) is so large that the decoder quickly learns
  to fool the frozen ViT-Tiny. Training only the discriminator heads offline on reconstructions without GAN reaches 0.67–0.77
  accuracy in 600 steps, so the discriminator itself can tell them apart.
- Stage 2 ([02](02_stage2_ditdh_s.md)) uses **XL with GAN**, which has the best train rFID.

## Caveats

- B (fp32) and L / XL (bf16) differ in training precision. The effect of inference precision is negligible (XL with GAN, test 10k:
  PSNR 22.346 / 22.343 and rFID 5.709 / 5.708 in fp32 / bf16); the effect of training precision was not measured.
- One seed per run.

## Reproduce

```bash
scripts/train_stage1.sh configs/stage1/training/cifar_dinov2_decXL_nogan.yaml dinov2_decXL_nogan
scripts/train_stage1.sh configs/stage1/training/cifar_dinov2_decXL_gan_ft.yaml dinov2_decXL_gan_ft
# extract the EMA decoder, then reconstruct and score test (or train)
PYTHONPATH=src uv run python src/extract_decoder.py --config configs/stage1/training/cifar_dinov2_decXL_gan_ft.yaml \
  --ckpt ckpts/stage1/dinov2_decXL_gan_ft/checkpoints/ep-last.pt --use-ema --out ckpts/stage1/dinov2_decXL_gan_ft/decoder_ema.pt
scripts/eval_stage1.sh configs/stage1/pretrained/cifar_dinov2_decXL_gan_ft.yaml dinov2_decXL_gan_ft test
```
