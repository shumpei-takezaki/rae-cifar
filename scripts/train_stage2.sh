#!/usr/bin/env bash
# Usage: scripts/train_stage2.sh <config> <experiment name> [extra train.py args, e.g. --wandb]
# Writes ckpts/stage2/<experiment name>/ and resumes automatically if it already exists.
set -euo pipefail
set -a; [[ -f .env ]] && source .env; set +a
export EXPERIMENT_NAME=$2
uv run torchrun --standalone --nproc_per_node=1 src/train.py --config "$1" \
  --data-path data/cifar10/train --results-dir ckpts/stage2 --image-size 32 --precision bf16 "${@:3}"
