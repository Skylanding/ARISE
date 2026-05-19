#!/usr/bin/env bash
# Train ARISE on Phi-4-mini-instruct + DeepScaleR (Table 1 row 2 of the paper).
set -xeuo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PROJECT_NAME="${PROJECT_NAME:-arise}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-ARISE-DAPO-Phi4-Mini}"
export ARISE_CKPTS_DIR="${ARISE_CKPTS_DIR:-ckpts/arise/${EXPERIMENT_NAME}}"
mkdir -p "${ARISE_CKPTS_DIR}"

CONFIG_FILE="${CONFIG_FILE:-recipe/arise/config/arise_phi4_mini.yaml}"

python -m verl.trainer.main_ppo \
    --config "${CONFIG_FILE}" \
    --recipe arise \
    "$@"
