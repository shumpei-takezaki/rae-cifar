"""Dump CIFAR-10 to ImageFolder PNGs (what the RAE train/eval scripts read).

    uv run python scripts/prepare_cifar10.py   # downloads CIFAR-10 into data/ (cifar-10-batches-py) if missing
    -> data/cifar10/{train,test}/<class>/<index>.png, {train,test}.npz (uint8, ImageFolder order; reference for
       reconstruction metrics) and train_fid_stats.npz (Inception mu/sigma of train, the gFID reference; needs a GPU)
"""

import argparse
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

p = argparse.ArgumentParser()
p.add_argument("--src", default="data", help="dir containing (or to download) cifar-10-batches-py")
p.add_argument("--out", default="data/cifar10")
args = p.parse_args()

for split in ("train", "test"):
    ds = CIFAR10(args.src, train=split == "train", download=True)
    for c in ds.classes:
        (Path(args.out) / split / c).mkdir(parents=True, exist_ok=True)
    for i, (img, y) in enumerate(ds):
        img.save(Path(args.out) / split / ds.classes[y] / f"{i:05d}.png")
    print(f"{split}: {len(ds)} images")
    order = np.argsort(ds.targets, kind="stable")  # ImageFolder order: class (alphabetical = label order), then index
    np.savez(Path(args.out) / f"{split}.npz", arr_0=ds.data[order])

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval.fid import _compute_inception_moments_from_arr
mu, sigma = _compute_inception_moments_from_arr(np.load(Path(args.out) / "train.npz")["arr_0"], batch_size=500, device="cuda")
np.savez(Path(args.out) / "train_fid_stats.npz", mu=mu, sigma=sigma)
