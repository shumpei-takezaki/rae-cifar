#!/usr/bin/env bash
# Usage: scripts/train_stage1.sh <config> <experiment name> [extra train_stage1.py args, e.g. --wandb]
# Writes ckpts/stage1/<experiment name>/ and resumes automatically if it already exists. bf16 by default (project rule since 2026-10-28).
set -euo pipefail
set -a; [[ -f .env ]] && source .env; set +a
export EXPERIMENT_NAME=$2
uv run torchrun --standalone --nproc_per_node=1 src/train_stage1.py --config "$1" \
  --data-path data/cifar10/train --results-dir ckpts/stage1 --image-size 32 --precision bf16 "${@:3}"
