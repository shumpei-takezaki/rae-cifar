# Experiments

Reproductions of findings from RAE-related papers on CIFAR-10. One file per experiment in `experiments/`, figures and numbers
in `figures/<experiment>/`. Unless noted, the encoder is the DINOv2 ViT-Tiny/4 from dino-cifar (no registers, frozen) and the
latent is `[192, 8, 8]`.

| # | Experiment | Paper | Main result |
|---|---|---|---|
| 01 | [Stage 1: decoder size and GAN](experiments/01_stage1_decoder_capacity.md) | RAE Table 1b | On train, larger decoders are better (XL + GAN: rFID 0.89). On test, all decoders overfit and the paper's trend does not hold |
| 02 | [Stage 2: DiT<sup>DH</sup>-S generation](experiments/02_stage2_ditdh_s.md) | RAE Sec. 5 | FID 13.59 (50k, no CFG). Best at epoch 500; CFG 3.0 hurts |
| 03 | [Single-image overfitting](experiments/03_single_image_overfit.md) | RAE Sec. 4.1, Table 3 | For d < n the loss stops at the bound (n − d)/n; it converges for d ≥ n. Reproduced |
| 04 | [Latent geometry](experiments/04_latent_geometry.md) | RJF Fig. 2–3 | The latent lies on a sphere of radius √192; the straight path shrinks the norm to 71% midway |
| 05 | [Radial / angular parts of the loss](experiments/05_radial_angular.md) | RJF Fig. 4 | The angular loss alone goes down to 0.004 even at d = 64 (full stays above the bound); nearly trivial for a single image |
| 06 | [Encoder size (Tiny vs. S; B coming soon)](experiments/06_encoder_capacity.md) | RAE Table 1c | With S, reconstruction improves on every metric (test rFID 5.25 → 3.97, PSNR +2.6 dB), unlike the paper (nearly constant) |

**Coming soon**

- Encoder size: DINOv2 ViT-B/4 (token dim 768).
- Stage 2 ablations of RAE design choices that are on by default here: noise augmentation in decoder training (`noise_tau: 0.8`),
  the wide DDT head of DiT<sup>DH</sup>, and the dimension-dependent noise-schedule shift (√(12288/4096) ≈ 1.73).
- Follow-ups: RAEv2, RJF (Learning on the Manifold), CIFAR-100.

Papers: RAE [arXiv 2510.11690](https://arxiv.org/abs/2510.11690), RJF (Learning on the Manifold) [arXiv 2602.10099](https://arxiv.org/abs/2602.10099).
