#!/usr/bin/env bash
# Train ARISE on Qwen3-4B-Instruct-2507 + DeepScaleR (Table 1 row 1 of the paper).
#
# Override any path through env vars: ARISE_MODEL_PATH, ARISE_TRAIN_FILE,
# ARISE_AMC23, ARISE_AIME24, ARISE_AIME25, ARISE_OMNI, ARISE_CKPTS_DIR.
set -xeuo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PROJECT_NAME="${PROJECT_NAME:-arise}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-ARISE-DAPO-Qwen3-4B}"
export ARISE_CKPTS_DIR="${ARISE_CKPTS_DIR:-ckpts/arise/${EXPERIMENT_NAME}}"
mkdir -p "${ARISE_CKPTS_DIR}"

CONFIG_FILE="${CONFIG_FILE:-recipe/arise/config/arise_qwen3_4b.yaml}"

python -m verl.trainer.main_ppo \
    --config "${CONFIG_FILE}" \
    --recipe arise \
    "$@"
