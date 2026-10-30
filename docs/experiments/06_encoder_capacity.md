# 06. Encoder size: DINOv2 ViT-Tiny/4 vs. ViT-S/4 (RAE Table 1c)

- **Status:** done. ViT-B/4 coming soon.
- **Paper:** Table 1c (DINOv2-S / B / L with the default ViT-XL decoder; rFID is nearly constant across encoder sizes)

## Goal

See how RAE reconstruction (and later generation) changes with a larger encoder.

## Setup

- Encoder: DINOv2 from dino-cifar, no registers. Besides Tiny/4 (width 192), we add **S/4** (width 384, 12 layers, 6 heads;
  the size of the official ViT-S), trained with the same recipe (1000 epochs) except for width and heads. Latent `[384, 8, 8]`.
- Decoder: **fixed to ViT-XL** (as in the paper), with the same procedure as XL in [01](01_stage1_decoder_capacity.md)
  (400 epochs without GAN, then 100 epochs of GAN fine-tuning, bf16).
- Compared: kNN / linear probe in dino-cifar, and Stage 1 reconstruction on train and test (PSNR, SSIM, LPIPS, rFID).

## Caveats

- The encoders were trained with different precision: Tiny in strict fp32, S with TF32 (S is about 2.4× slower in fp32).
  The loss after 2 epochs differs by 0.0005 between the two.
- One seed per run.

## Results

### Encoders (dino-cifar, CIFAR-10 test top-1 %)

| Encoder | Width | kNN CLS | kNN PatchMean | Linear CLS | Linear PatchMean |
|---|---:|---:|---:|---:|---:|
| ViT-Tiny/4 | 192 | 85.49 | 86.05 | **81.25** | 86.59 |
| ViT-S/4 | 384 | **86.96** | **87.04** | 76.59 | **89.54** |

S gives better representations (PatchMean: kNN +1.0 pt, linear +3.0 pt). Only the CLS linear probe is lower, but it is
provisional because the linear-probe hyperparameters are untuned. Details:
[dino-cifar experiment 03](https://github.com/shumpei-takezaki/dino-cifar/blob/main/docs/experiments/03_encoder_size.md).

### Stage 1 (XL decoder)

Final-epoch EMA decoders; test 10k, train 50k images. Train rFID is given on 50k / a random 10k subset (the same subset, seed 0).

| Encoder | GAN | Test PSNR | Test SSIM | Test LPIPS | Test rFID | Train PSNR | Train SSIM | Train LPIPS | Train rFID |
|---|---|---|---|---|---|---|---|---|---|
| Tiny/4 | no | 22.49 | 0.843 | 0.101 | 5.25 | 33.76 | 0.969 | 0.0093 | 1.71 / 3.32 |
| Tiny/4 | yes | 22.35 | 0.835 | 0.102 | 5.71 | 31.84 | 0.957 | 0.0138 | 0.89 / 2.69 |
| S/4 | no | **25.06** | **0.926** | **0.062** | 3.97 | **35.39** | **0.979** | **0.0067** | 1.52 / 2.84 |
| S/4 | yes | 24.83 | 0.919 | 0.065 | **3.71** | 33.33 | 0.970 | 0.0106 | **0.72 / 2.23** |

Final training loss without GAN: Tiny recon 0.021 / LPIPS 0.020, S 0.018 / 0.017.

- **With the S encoder, reconstruction improves on every metric**, most of all on test (PSNR +2.6 dB, rFID 5.25 → 3.97).
  The train–test gap also shrinks a little (11.3 dB for Tiny vs. 10.3 dB for S in PSNR).
- **This differs from the paper**, where rFID hardly changes with encoder size. Possible reasons (none verified):
  - The latent doubles (token dim 192 → 384). The paper's DINOv2-S / B / L have token dims 384 / 768 / 1024, ample for 16×16-pixel
    patches. Our patches are 4×4 pixels (48 values), but Tiny may still lose information.
  - The encoder representation itself is better (PatchMean linear probe 86.6 → 89.5).
- **With S, the GAN loss also improves the test rFID** (3.97 → 3.71), whereas with Tiny it got worse (5.25 → 5.71).
  PSNR / SSIM / LPIPS get slightly worse with GAN, as for Tiny.
- As with Tiny, the discriminator stops discriminating right after the generator GAN loss is enabled
  (accuracy 0.61 at epoch 9, then 0.47–0.50 from epoch 10 on).
- **With only two encoder sizes we do not claim a trend ("larger is better").** ViT-B/4 (width 768) will be added later.
- Stage 2 (generation) has not been run with S. Following the RAE argument (DiT width ≥ token dim), a doubled latent also needs a
  DiT of width ≥ 384.

Configs: `configs/stage1/training/cifar_dinov2s_decXL_{nogan,gan_ft}.yaml` (the Tiny XL configs with the encoder swapped).

## Reproduce

```bash
# Encoder (dino-cifar): configs/dinov2_vits.yaml, scripts/train_dinov2_vits.sh
scripts/train_stage1.sh configs/stage1/training/cifar_dinov2s_decXL_nogan.yaml dinov2s_decXL_nogan
PYTHONPATH=src uv run python src/extract_decoder.py --config configs/stage1/training/cifar_dinov2s_decXL_nogan.yaml \
  --ckpt ckpts/stage1/dinov2s_decXL_nogan/checkpoints/ep-last.pt --use-ema --out ckpts/stage1/dinov2s_decXL_nogan/decoder_ema.pt
scripts/train_stage1.sh configs/stage1/training/cifar_dinov2s_decXL_gan_ft.yaml dinov2s_decXL_gan_ft   # then extract_decoder likewise
scripts/eval_stage1.sh configs/stage1/pretrained/cifar_dinov2s_decXL_gan_ft.yaml dinov2s_decXL_gan_ft test   # and train
```
