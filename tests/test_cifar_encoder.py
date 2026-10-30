"""RAE with the dino-cifar encoder: shapes, frozen encoder, affine-free final LN. Needs the dino-cifar checkpoint."""

import os
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
CONFIG = ROOT / "configs/stage1/training/cifar_dinov2_decB.yaml"


def test_rae_cifar_roundtrip_shapes():
    from utils.model_utils import instantiate_from_config
    from utils.train_utils import parse_configs

    rae_cfg, *_ = parse_configs(OmegaConf.load(CONFIG))
    ckpt = ROOT / rae_cfg.params.encoder_params.ckpt_path
    if not ckpt.exists():
        pytest.skip(f"missing {ckpt}")
    os.chdir(ROOT)  # config paths are relative to the repo root
    rae = instantiate_from_config(rae_cfg).eval()
    x = torch.rand(2, 3, 32, 32)
    z = rae.encode(x)
    assert z.shape == (2, 192, 8, 8)
    assert rae.decode(z).shape == (2, 3, 32, 32)
    assert not any(p.requires_grad for p in rae.encoder.parameters())
    assert rae.encoder.encoder.norm.weight is None  # final LN affine dropped (normalize: true)
