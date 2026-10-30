"""DINO discriminator with the dino-cifar DINO (v1) ViT (recipe 'cifar'): 32px in, frozen proxy, grads reach the input."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
CKPT = ROOT / "pretrained_models/disc_dino_vit_tiny4.pt"


def test_cifar_disc_shapes_and_grads():
    if not CKPT.exists():
        pytest.skip(f"missing {CKPT}")
    from disc import build_discriminator

    cfg = {"arch": {"dino_ckpt_path": str(CKPT), "recipe": "cifar", "ks": 9}}
    disc, _ = build_discriminator(cfg, torch.device("cpu"))
    x = (torch.rand(2, 3, 32, 32) * 2 - 1).requires_grad_(True)
    logits_fake, logits_real = disc(x, x.detach())
    assert logits_fake.shape == (2, 5 * 64)  # 4 key depths + last block, 8x8 tokens each
    assert logits_real.shape == logits_fake.shape
    logits_fake.sum().backward()
    assert x.grad.abs().sum() > 0  # generator gets a GAN gradient through the frozen ViT
    assert all(not p.requires_grad for p in disc.dino_proxy[0].parameters())
    assert all(p.requires_grad for p in disc.heads.parameters())
