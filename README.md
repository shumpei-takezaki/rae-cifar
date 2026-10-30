# RAE on CIFAR-10

A small-scale reproduction of [Diffusion Transformers with Representation Autoencoders (RAE)](https://arxiv.org/abs/2510.11690)
on CIFAR-10, built on the [official implementation](https://github.com/bytetriper/RAE) (upstream README: [README_upstream.md](README_upstream.md)).

- **Encoder:** a DINOv2 ViT-Tiny/4 (no registers) trained from scratch on CIFAR-10 with
  [dino-cifar](https://github.com/shumpei-takezaki/dino-cifar), frozen. Latent: `[192, 8, 8]` (64 tokens of dim 192).
- **Decoder:** the official ViT decoder (patch 4), trained with L1 + LPIPS (+ GAN) and latent noise augmentation.
- **Generator:** DiT<sup>DH</sup>-S on the normalized latent, class-conditional (10 classes).

Detailed write-ups are in [docs/](docs/README.md), one file per experiment.

## Results

| # | Experiment | Paper | Main result |
|---|---|---|---|
| 01 | [Decoder size (B / L / XL) and GAN](docs/experiments/01_stage1_decoder_capacity.md) | Table 1b | On train, larger decoders are better (XL + GAN: rFID 0.89 on 50k). On test, all decoders overfit (rFID 5.2–6.6) |
| 02 | [DiT<sup>DH</sup>-S generation](docs/experiments/02_stage2_ditdh_s.md) | Sec. 5 | FID 13.59 (50k samples, no CFG). Best at epoch 500; CFG 3.0 hurts (23.50) |
| 03 | [Single-image overfitting: DiT width vs. token dim](docs/experiments/03_single_image_overfit.md) | Sec. 4.1, Table 3 | For width d < n = 192 the loss stops just above the bound (n − d)/n; it converges for d ≥ n |
| 04 | [Latent geometry](docs/experiments/04_latent_geometry.md) | RJF Fig. 2–3 | Tokens lie on a sphere of radius √192; the linear path shrinks the norm to 71% midway |
| 05 | [Radial / angular split of the loss](docs/experiments/05_radial_angular.md) | RJF Fig. 4 | Training on the angular part alone converges even at d = 64 (0.004 vs. 0.74), but this is nearly trivial for a single image |
| 06 | [Encoder size (Tiny vs. S; B coming soon)](docs/experiments/06_encoder_capacity.md) | Table 1c | A larger encoder (ViT-S/4, token dim 384) improved reconstruction on every metric (test rFID 5.25 → 3.97, PSNR +2.6 dB), unlike the paper (nearly constant) |

RJF: [Learning on the Manifold](https://arxiv.org/abs/2602.10099).

**Coming soon**

- Encoder size: DINOv2 ViT-B/4 (token dim 768).
- Stage 2 ablations of the RAE design choices that are already on by default here: noise-augmented decoder training
  (`noise_tau: 0.8`), the wide DDT head of DiT<sup>DH</sup>, and the dimension-dependent noise-schedule shift (√(12288/4096) ≈ 1.73).
- Follow-ups: RAEv2, RJF (Learning on the Manifold), CIFAR-100.

## Setup

```bash
git clone https://github.com/shumpei-takezaki/rae-cifar.git && cd rae-cifar
uv sync                                      # Python 3.12; pulls dino-cifar (encoder definition)
uv run python scripts/prepare_cifar10.py     # downloads CIFAR-10 to data/; writes data/cifar10/: PNGs, {train,test}.npz, train FID stats (GPU)
```

## Pretrained weights

From [huggingface.co/shumpei777/rae-cifar](https://huggingface.co/shumpei777/rae-cifar):

```bash
uv run hf download shumpei777/rae-cifar --local-dir pretrained_models
PYTHONPATH=src uv run python src/sample.py --config configs/stage2/sampling/CIFAR10/DiTDH-S_pretrained.yaml   # -> sample.png, one per class
```

| File | Content | Size |
|---|---|---|
| `encoder_dinov2_vit_tiny4.pt` | DINOv2 ViT-Tiny/4 encoder (EMA teacher) | 22 MB |
| `normalization_stats.pt` | per-channel, per-position latent mean / var (train, with flips) | 0.1 MB |
| `decoder_vitxl_gan.pt` | ViT-XL decoder, GAN fine-tuned (EMA) | 1.7 GB |
| `ditdh_s_ema.pt` | DiT<sup>DH</sup>-S, 2000 epochs (EMA) | 0.8 GB |
| `encoder_dinov2_vit_s4.pt` | DINOv2 ViT-S/4 encoder (experiment 06) | 85 MB |
| `decoder_vitxl_gan_s4.pt` | ViT-XL decoder for the ViT-S/4 encoder, GAN fine-tuned (EMA) | 1.7 GB |
| `disc_dino_vit_tiny4.pt` | DINO (v1) ViT-Tiny/4, frozen backbone of the GAN discriminator (only for training) | 22 MB |

## Training from scratch

The configs read the frozen encoder and the frozen backbone of the GAN discriminator (a DINO (v1) ViT-Tiny/4) from
`pretrained_models/` (download above); both were trained with dino-cifar.
All runs use a single GPU and bf16 (training times below). Each launcher resumes automatically; add `--wandb` to log to Weights & Biases
(`WANDB_KEY`, `ENTITY`, `PROJECT` in `.env`).

```bash
# Stage 1: decoder without GAN (400 epochs), then GAN fine-tuning (100 epochs)
scripts/train_stage1.sh configs/stage1/training/cifar_dinov2_decXL_nogan.yaml dinov2_decXL_nogan
scripts/train_stage1.sh configs/stage1/training/cifar_dinov2_decXL_gan_ft.yaml dinov2_decXL_gan_ft
PYTHONPATH=src uv run python src/extract_decoder.py --config configs/stage1/training/cifar_dinov2_decXL_gan_ft.yaml \
  --ckpt ckpts/stage1/dinov2_decXL_gan_ft/checkpoints/ep-last.pt --use-ema --out ckpts/stage1/dinov2_decXL_gan_ft/decoder_ema.pt
scripts/eval_stage1.sh configs/stage1/pretrained/cifar_dinov2_decXL_gan_ft.yaml dinov2_decXL_gan_ft test   # PSNR / SSIM / LPIPS / rFID

# Latent normalization statistics -> stats/
uv run torchrun --standalone --nproc_per_node=1 src/calculate_stat.py --config configs/stage1/pretrained/cifar_dinov2_decXL_gan_ft.yaml \
  --data-path data/cifar10/train --sample-dir stats --image-size 32 --per-proc-batch-size 500 --num-workers 4 --momentum 0.02

# Stage 2: DiT^DH-S (2000 epochs), then FID on 50k samples against CIFAR-10 train
scripts/train_stage2.sh configs/stage2/training/CIFAR10/DiTDH-S_dinov2-r0_decXL-gan.yaml ditdhS_decXLgan
scripts/eval_stage2.sh configs/stage2/sampling/CIFAR10/DiTDH-S_dinov2-r0_decXL-gan.yaml ditdhS_decXLgan_cfg1.0 1.0
```

Configs for the other decoders (B / L, with and without GAN) are in `configs/stage1/`. The commands for each experiment
are at the end of its file in `docs/experiments/`.

Wall-clock times on one A100-40GB (bf16 unless noted):

| Run | Time |
|---|---|
| Encoder, dino-cifar DINOv2 ViT-Tiny/4 / ViT-S/4, 1000 epochs | ~16.5 h (fp32) / ~18 h (TF32) |
| Stage 1 decoder without GAN, 400 epochs: B / L / XL | ~21 h (fp32) / ~10.5 h / ~12.5 h |
| Stage 1 GAN fine-tuning, 100 epochs: B / L / XL | ~6 h (fp32) / ~4 h / ~5 h |
| Stage 2 DiT<sup>DH</sup>-S, 2000 epochs (incl. 4 evaluations of 10k samples) | ~31 h |
| FID on 50k samples: without / with CFG | ~20 min / ~35 min |

## Changes from upstream

- `src/stage1/encoders/cifar_dino.py`: the dino-cifar encoder (`CifarDinoViT`).
- `src/disc/dinodisc.py`: a 32px discriminator recipe (`recipe: 'cifar'`) whose frozen backbone is a DINO ViT-Tiny/4 from dino-cifar.
- `src/train_stage1.py`: flip-only augmentation at 32px; optional gradient checkpointing of the decoder.
- `src/train.py`, `src/sample.py`: 32px / 10-class support; resuming from `ep-last.pt` (`src/utils/resume_utils.py`).
- `pyproject.toml` / `uv.lock`: environment managed with uv instead of conda.

## Acknowledgements

- [RAE](https://github.com/bytetriper/RAE) ([Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690)):
  this repository is built on the official implementation, which in turn builds on [SiT](https://github.com/willisma/sit),
  [DDT](https://github.com/MCG-NJU/DDT), [LightningDiT](https://github.com/hustvl/LightningDiT/) and [MAE](https://github.com/facebookresearch/mae)
  (see [README_upstream.md](README_upstream.md)).
- [dino-cifar](https://github.com/shumpei-takezaki/dino-cifar): the CIFAR-10 DINOv2 encoders and the DINO backbone of the discriminator,
  following [DINO](https://github.com/facebookresearch/dino) and [DINOv2](https://github.com/facebookresearch/dinov2).
- [Learning on the Manifold (RJF)](https://arxiv.org/abs/2602.10099): the analyses in experiments 04 and 05.

## Citation

Please cite the RAE paper:

```bibtex
@misc{zheng2025diffusiontransformersrepresentationautoencoders,
      title={Diffusion Transformers with Representation Autoencoders},
      author={Boyang Zheng and Nanye Ma and Shengbang Tong and Saining Xie},
      year={2025},
      eprint={2510.11690},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.11690},
}
```

## License

MIT, as upstream ([LICENSE](LICENSE), Copyright (c) 2025 Boyang Zheng).
