#!/usr/bin/env bash
# Usage: scripts/eval_stage1.sh <stage-1 inference config> <name> [split: test (default) | train]
# Reconstructs all of data/cifar10/<split> into samples/<name>_ema_<split>/ (+ .npz) and prints PSNR / SSIM / LPIPS / rFID
# against data/cifar10/<split>.npz (scripts/prepare_cifar10.py). For train it also prints rFID on a random 10k subset.
set -euo pipefail
cfg=$1; name=${2}_ema_${3:-test}; split=${3:-test}
[[ -f samples/$name.npz ]] || SAVE_FOLDER=$name uv run torchrun --standalone --nproc_per_node=1 src/stage1_sample_ddp.py \
  --config "$cfg" --data-path data/cifar10/$split --sample-dir samples --image-size 32 --per-proc-batch-size 500
PYTHONPATH=src uv run python -c "
import numpy as np, torch
from eval import compute_reconstruction_metrics
from eval.ref_iqa import calculate_lpips
from eval.fid import calculate_rfid
ref, rec = np.load('data/cifar10/$split.npz')['arr_0'], np.load('samples/$name.npz')['arr_0']
assert ref.shape == rec.shape, (ref.shape, rec.shape)
m = compute_reconstruction_metrics(ref, rec, torch.device('cuda'), batch_size=500)
m['lpips'] = calculate_lpips(ref, rec, 500, 'cuda')
if len(ref) > 10000:
    i = np.random.default_rng(0).choice(len(ref), 10000, replace=False)
    m['rfid_10k'] = calculate_rfid(ref[i], rec[i], 500, 'cuda')
print('$name', {k: round(float(v), 4) for k, v in m.items()})"
