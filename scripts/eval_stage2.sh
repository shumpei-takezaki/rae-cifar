#!/usr/bin/env bash
# Usage: [BS=500] scripts/eval_stage2.sh <sampling config> <name> [cfg scale (default: the config's)] [num samples (default 50000)]
# Generates class-balanced samples into samples/stage2/<name>/ (+ .npz) and prints FID vs CIFAR-10 train (50k) stats.
set -euo pipefail
cfg=$1; name=$2; scale=${3:-}; n=${4:-50000}
if [[ -n $scale ]]; then  # sample_ddp.py has no CLI override for the guidance scale
  tmp=$(mktemp --suffix=.yaml); trap 'rm -f "$tmp"' EXIT
  sed -E "s/^(  scale: )[0-9.]+/\1$scale/" "$cfg" > "$tmp"; cfg=$tmp
fi
SAVE_FOLDER=$name uv run torchrun --standalone --nproc_per_node=1 src/sample_ddp.py --config "$cfg" \
  --sample-dir samples/stage2 --per-proc-batch-size ${BS:-500} --num-fid-samples "$n" --precision bf16
PYTHONPATH=src uv run python -c "
import numpy as np
from eval.fid import calculate_gfid
fid = calculate_gfid(np.load('samples/stage2/$name.npz')['arr_0'], np.load('data/cifar10/train_fid_stats.npz'), 500, 'cuda')
print(f'FID $name (n=$n): {fid:.3f}')"
