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
| 07 | [Stage 2 model size (smaller DiT<sup>DH</sup>)](experiments/07_stage2_model_size.md) | RAE Fig. 6, Table 17 | A 512-wide DDT head (≈ 2.7n) beats the official 2048 head: FID (50k) 7.94 vs. 13.59 with 4.5× fewer parameters; DiT<sup>DH</sup>-Ti (18.5M) 9.64. No model memorizes by the Gu et al. criterion; the Carlini score shows the official model makes ~10× more near-copies |
| 08 | [Is the DDT head needed?](experiments/08_ddt_head.md) | RAE Table 6, RJF | Removing the 512 head (plain LightningDiT-S/1) costs a little: FID 8.70 vs. 7.94. A plain DiT with width = n (Ti/1) diverged |
| 09 | [Ablation: noise-schedule shift](experiments/09_time_shift.md) | RAE Table 4 | The shift helps less than in the paper: FID 9.01 without vs. 7.94 with (α ≈ 1.73) |
| 10 | [Ablation: decoder noise augmentation](experiments/10_decoder_noise.md) | RAE Table 5 | Reproduces the trade-off: without noise, test rFID improves (5.71 → 4.22) but gFID worsens (7.94 → 10.45) |

**Coming soon**

- Encoder size: DINOv2 ViT-B/4 (token dim 768).
- Follow-ups: RAEv2, RJF (Learning on the Manifold), CIFAR-100.

Papers: RAE [arXiv 2510.11690](https://arxiv.org/abs/2510.11690), RJF (Learning on the Manifold) [arXiv 2602.10099](https://arxiv.org/abs/2602.10099).
