#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate a trained DAPO checkpoint on math benchmarks (val_only mode).
#
# Usage:
#   bash eval_math.sh                                    # latest ckpt, math500-sample50
#   bash eval_math.sh /abs/path/to/global_step_150       # specific ckpt
#   bash eval_math.sh latest full                        # latest ckpt, full math500
#   bash eval_math.sh baseline                           # base model (no ckpt)
#
# Environment overrides:
#   EVAL_DATASET   math500 | math500-sample50 | aime24 | gsm8k  (default: math500-sample50)
#   PROJECT_NAME   (default: agent-lib)
#   EXP_NAME       (default: DAPO-Qwen3-4B)
#   CKPTS_ROOT     (default: ~/verl/ckpts)
# ---------------------------------------------------------------------------
set -Eeuo pipefail
IFS=$'\n\t'
log() { echo "[$(date '+%F %T')] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export VLLM_USE_V1="1"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

N_GPUS=4

# ---- Experiment naming (same convention as training script) ----
project_name=${PROJECT_NAME:-agent-lib}
exp_name=${EXP_NAME:-DAPO-Qwen3-4B}
CKPTS_ROOT=${CKPTS_ROOT:-"${HOME}/verl/ckpts"}
CKPTS_DIR="${CKPTS_ROOT}/${project_name}/${exp_name}"

MODEL_PATH=${MODEL_PATH_OVERRIDE:-"/home/ubuntu/basemodels/qwen3/qwen3-4b-it"}

# ---- Checkpoint resolution ----
ARG1="${1:-latest}"
ARG2="${2:-}"
BASELINE_MODE=0

if [[ "${ARG1,,}" == "baseline" ]]; then
    BASELINE_MODE=1
    TRAINER_RESUME_MODE="disable"
    TRAINER_RESUME_PATH=""
    RESUME_DESC="baseline (base model only)"
elif [[ "${ARG1}" == /* ]]; then
    # Absolute path to a specific global_step_xxx directory
    if [[ ! -d "${ARG1}" ]]; then
        log "ERROR: Checkpoint path does not exist: ${ARG1}"
        exit 1
    fi
    TRAINER_RESUME_MODE="auto"
    TRAINER_RESUME_PATH="${ARG1}"
    if [[ "$(basename "${ARG1}")" =~ ^global_step_ ]]; then
        CKPTS_DIR="$(dirname "${ARG1}")"
    fi
    RESUME_DESC="${ARG1}"
elif [[ "${ARG1}" == "latest" ]]; then
    TRAINER_RESUME_MODE="auto"
    TRAINER_RESUME_PATH="${CKPTS_DIR}"
    RESUME_DESC="${CKPTS_DIR} (latest)"
elif [[ "${ARG1}" =~ ^[0-9]+$ ]]; then
    # Numeric step
    TRAINER_RESUME_PATH="${CKPTS_DIR}/global_step_${ARG1}"
    if [[ ! -d "${TRAINER_RESUME_PATH}" ]]; then
        log "ERROR: Checkpoint does not exist: ${TRAINER_RESUME_PATH}"
        exit 1
    fi
    TRAINER_RESUME_MODE="auto"
    RESUME_DESC="${TRAINER_RESUME_PATH}"
else
    log "ERROR: unrecognized argument '${ARG1}'. Use: latest | <step_number> | /abs/path | baseline"
    exit 1
fi

# ---- Dataset resolution ----
EVAL_DATASET=${EVAL_DATASET:-math500-sample50}
case "${ARG2}" in
    full)    EVAL_DATASET="math500" ;;
    sample*) EVAL_DATASET="math500-sample50" ;;
    aime*)   EVAL_DATASET="aime24" ;;
    gsm*)    EVAL_DATASET="gsm8k" ;;
    "")      ;; # keep EVAL_DATASET from env
    *)       EVAL_DATASET="${ARG2}" ;;
esac

case "${EVAL_DATASET}" in
    math500)         TEST_FILE="/home/ubuntu/datasets/math_datasets/math500/math500.parquet" ;;
    math500-sample50) TEST_FILE="/home/ubuntu/datasets/math_datasets/math500/math500-sample50.parquet" ;;
    aime24)          TEST_FILE="/home/ubuntu/datasets/math_datasets/aime24/eval/aime-2024.parquet" ;;
    aime25)          TEST_FILE="/home/ubuntu/datasets/math_datasets/aime24/eval/aime-2025.parquet" ;;
    gsm8k)           TEST_FILE="/home/ubuntu/verl/data/gsm8k-test-dapo.parquet" ;;
    *)               TEST_FILE="${EVAL_DATASET}" ;; # allow raw path
esac

if [[ ! -f "${TEST_FILE}" ]]; then
    log "ERROR: Test file does not exist: ${TEST_FILE}"
    exit 1
fi

# Use a dummy train file (val_only mode never trains, but Hydra requires it)
TRAIN_FILE="/home/ubuntu/verl/data/dapo-math-17k-sampled-1pct.parquet"

# ---- Length / batch config ----
max_prompt_length=2048
max_response_length=3072
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 6 / 5))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3 / 2))
DATA_BATCH_SIZE=4

# ---- Eval output dir & log ----
EVAL_CKPTS_DIR="${CKPTS_DIR}/eval"
mkdir -p "${EVAL_CKPTS_DIR}"
LOG_DIR="${REPO_ROOT}/recipe/dapo/eval/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_${EVAL_DATASET}_$(date +%Y%m%d_%H%M%S).log"

log "===== Eval Config ====="
log "PROJECT_NAME=${project_name}"
log "EXP_NAME=${exp_name}"
log "CKPTS_DIR=${CKPTS_DIR}"
log "RESUME=${RESUME_DESC}"
log "MODEL_PATH=${MODEL_PATH}"
log "EVAL_DATASET=${EVAL_DATASET}"
log "TEST_FILE=${TEST_FILE}"
log "LOG_FILE=${LOG_FILE}"
log "======================="

# ---- Build resume args as an array (avoids quoting issues with Hydra) ----
RESUME_ARGS=("trainer.resume_mode=${TRAINER_RESUME_MODE}")
if [[ ${BASELINE_MODE} -eq 0 && -n "${TRAINER_RESUME_PATH}" ]]; then
    RESUME_ARGS+=("trainer.resume_from_path=${TRAINER_RESUME_PATH}")
fi

# ---- Run evaluation ----
{ python3 -u -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.return_full_prompt=True \
    +data.default_data_source=math_dapo \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${DATA_BATCH_SIZE} \
    data.train_batch_size=${DATA_BATCH_SIZE} \
    data.val_batch_size=${DATA_BATCH_SIZE} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${DATA_BATCH_SIZE} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.temperature=0.2 \
    actor_rollout_ref.rollout.top_p=0.7 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.2 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=False \
    "trainer.logger=[console,wandb]" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}-eval-${EVAL_DATASET}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    +trainer.val_num_examine=100 \
    trainer.log_val_generations=100 \
    trainer.test_freq=5 \
    trainer.save_freq=999999 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${EVAL_CKPTS_DIR}" \
    "${RESUME_ARGS[@]}" \
    trainer.use_agent_system=False \
    env.env_name=math \
    env.max_steps=1 \
    env.rollout.n=1 \
    skill_library.enable=False ; } 2>&1 | tee -a "${LOG_FILE}"

ret=${PIPESTATUS[0]}

# ---- Post-run summary ----
if [[ -f "$LOG_FILE" ]]; then
    acc_line=$(grep -a "acc/mean@1" "$LOG_FILE" | tail -1 || true)
    if [[ -n "$acc_line" ]]; then
        log "Result: ${acc_line}"
    fi

    acc_summary=$(awk '
        BEGIN {t=0; f=0}
        /\[acc\]/ {
            if ($0 ~ /True/) t++;
            else if ($0 ~ /False/) f++;
        }
        END {
            n = t + f;
            if (n > 0) printf("samples=%d, correct=%d, wrong=%d, acc=%.2f%%", n, t, f, 100.0*t/n);
            else print "samples=0 (no per-sample acc lines found)"
        }' "$LOG_FILE")
    invalid_count=$(grep -a "\[pred\]" "$LOG_FILE" | grep -c "\[INVALID\]" || true)
    log "Eval summary: ${acc_summary}, invalid=${invalid_count}"
    echo "[Summary] ${acc_summary}, invalid=${invalid_count}" >> "$LOG_FILE"
fi

if [[ $ret -eq 0 ]]; then
    log "Evaluation finished successfully"
else
    log "Evaluation failed with exit code $ret"
fi
log "Logs saved to: ${LOG_FILE}"
