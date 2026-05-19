#!/usr/bin/env bash
set -xeuo pipefail

# Always run from this repository root to avoid importing another `recipe` package.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# -----------------------------------------------------------------------------
# Experiment management conventions (recommended):
# 1) Use normalized names (no spaces) for stable checkpoint paths.
# 2) First run from scratch with RESUME_MODE=disable.
# 3) Resume interrupted runs with same PROJECT_NAME/EXP_NAME + RESUME_MODE=auto.
# 4) Optional explicit checkpoint recovery:
#    RESUME_MODE=resume_path RESUME_PATH=/abs/path/to/global_step_xxx
#
# Examples:
#   PROJECT_NAME=agent-lib EXP_NAME=DAPO-Qwen3-4B RESUME_MODE=disable bash run_dapo_qwen3_4b.sh
#   PROJECT_NAME=agent-lib EXP_NAME=DAPO-Qwen3-4B RESUME_MODE=auto    bash run_dapo_qwen3_4b.sh
# -----------------------------------------------------------------------------
# Naming (override from env for clean experiment management)
project_name=${PROJECT_NAME:-agent-lib}
exp_name=${EXP_NAME:-DAPO-Qwen3-4B}

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=2048
max_response_length=3072
enable_overlong_buffer=False
overlong_buffer_len=128
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=4
n_resp_per_prompt=4
train_prompt_mini_bsz=4
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 6 / 5))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3 / 2))

# Distributed: 4 GPUs on single node
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
NNODES=${NNODES:-1}
USE_AGENT_SYSTEM=${USE_AGENT_SYSTEM:-true}
# Resume mode: disable (start fresh), auto (resume latest), resume_path
RESUME_MODE=${RESUME_MODE:-auto}
RESUME_PATH=${RESUME_PATH:-}
if [[ "${RESUME_MODE}" != "disable" && "${RESUME_MODE}" != "auto" && "${RESUME_MODE}" != "resume_path" ]]; then
    echo "ERROR: unsupported RESUME_MODE=${RESUME_MODE}. Use disable|auto|resume_path." >&2
    exit 1
fi
if [[ "${RESUME_MODE}" == "resume_path" && -z "${RESUME_PATH}" ]]; then
    echo "ERROR: RESUME_MODE=resume_path requires RESUME_PATH." >&2
    exit 1
fi
if [[ "${RESUME_MODE}" != "resume_path" && -n "${RESUME_PATH}" ]]; then
    echo "WARN: RESUME_PATH is set but RESUME_MODE=${RESUME_MODE}; RESUME_PATH will be ignored."
fi
# Skill library switches
SKILL_LIBRARY_ENABLE=${SKILL_LIBRARY_ENABLE:-true}
SKILL_CONCLUDE_ENABLE=${SKILL_CONCLUDE_ENABLE:-true}
SKILL_ABSTRACTION_ENABLE=${SKILL_ABSTRACTION_ENABLE:-true}
SKILL_REWARD_ENABLE=${SKILL_REWARD_ENABLE:-true}
SKILL_REWARD_WHEN_USED_AND_CORRECT=${SKILL_REWARD_WHEN_USED_AND_CORRECT:-2.0}
SKILL_SELECTION_MODE=${SKILL_SELECTION_MODE:-llm_logprob}
SKILL_SELECTION_CONFIDENCE_THRESHOLD=${SKILL_SELECTION_CONFIDENCE_THRESHOLD:-null}
SKILL_SELECTION_EPSILON=${SKILL_SELECTION_EPSILON:-0.05}
SKILL_SELECTION_LLM_CANDIDATE_TOP_K=${SKILL_SELECTION_LLM_CANDIDATE_TOP_K:-4}
SKILL_MANAGER_ENABLE=${SKILL_MANAGER_ENABLE:-true}
SKILL_MANAGER_ACTIONS=${SKILL_MANAGER_ACTIONS:-'["NOOP","FETCH","LOAD","MODIFY"]'}
SKILL_MANAGER_PAYLOAD_SKILLS_LIMIT=${SKILL_MANAGER_PAYLOAD_SKILLS_LIMIT:-5}
SKILL_MANAGER_SYNC_EVERY=${SKILL_MANAGER_SYNC_EVERY:-5}
SKILL_MANAGER_USE_MODEL=${SKILL_MANAGER_USE_MODEL:-true}
SKILL_MANAGER_USE_MODEL_GENERATE=${SKILL_MANAGER_USE_MODEL_GENERATE:-false}
SKILL_MANAGER_ACTION_FORMAT=${SKILL_MANAGER_ACTION_FORMAT:-plain}
SKILL_MANAGER_FETCH_MODE=${SKILL_MANAGER_FETCH_MODE:-utility}
SKILL_MANAGER_LOAD_MODE=${SKILL_MANAGER_LOAD_MODE:-utility}
MASTER_WEIGHT=${MASTER_WEIGHT:-0.05}
MANAGER_WEIGHT=${MANAGER_WEIGHT:-0.0}
# Cold-start seeds prevent empty skill cache when early success is rare.
SKILL_SEED_SKILLS=${SKILL_SEED_SKILLS:-'[
  {"skill_name":"equation_setup","problem_type":"algebra","key_insight":"Translate word-problem quantities into variables and equations before solving","method":["Assign a variable to each unknown","Write equations from the given conditions","Solve by substitution or elimination"],"check":"Plug solution back into original conditions"},
  {"skill_name":"step_by_step_arithmetic","problem_type":"general","key_insight":"Sequential decomposition prevents compounding arithmetic errors","method":["Break the computation into single-operation steps","Carry out each step and record intermediate results","Combine intermediate results for the final answer"],"check":"Re-compute key steps or estimate order of magnitude"},
  {"skill_name":"case_analysis","problem_type":"combinatorics","key_insight":"Partition the problem into exhaustive non-overlapping cases","method":["Identify the branching criterion that splits the problem","Solve each case independently","Sum or unify the case results"],"check":"Verify cases are exhaustive and mutually exclusive"}
]'}
# Skill library warmup: first N training updates do plain DAPO rollout, while
# still extracting successful traces to populate skill library.
SKILL_WARMUP_ENABLE=${SKILL_WARMUP_ENABLE:-true}
SKILL_WARMUP_STEPS=${SKILL_WARMUP_STEPS:-10}
SKILL_CONCLUDE_MIN_CHARS=${SKILL_CONCLUDE_MIN_CHARS:-24}
SKILL_CONCLUDE_MIN_WORDS=${SKILL_CONCLUDE_MIN_WORDS:-4}
SKILL_CONCLUDE_MIN_UNIQUE_WORD_RATIO=${SKILL_CONCLUDE_MIN_UNIQUE_WORD_RATIO:-0.3}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
# Optional root override for all checkpoints
CKPTS_ROOT=${CKPTS_ROOT:-"${RAY_DATA_HOME}/ckpts"}
# from parent shells. To override intentionally, set MODEL_PATH_OVERRIDE.
MODEL_PATH=${MODEL_PATH_OVERRIDE:-"/home/ubuntu/basemodels/qwen3/qwen3-4b-it"}
if [[ "${MODEL_PATH}" == "~/"* ]]; then
    MODEL_PATH="${HOME}/${MODEL_PATH#~/}"
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    echo "Please set a valid local model directory, e.g. /home/ubuntu/basemodels/qwen3/qwen3-4b-it" >&2
    exit 1
fi
echo "Using MODEL_PATH=${MODEL_PATH}"
CKPTS_DIR=${CKPTS_DIR:-"${CKPTS_ROOT}/${project_name}/${exp_name}"}
mkdir -p "${CKPTS_DIR}"
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k-sampled-1pct.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}
DATASET_SOURCE=${DATASET_SOURCE:-dapo}
OPENR1_TRAIN_FILE=${OPENR1_TRAIN_FILE:-"${RAY_DATA_HOME}/data/openr1-math-default-1pct-dapo.parquet"}
if [[ "${DATASET_SOURCE}" == "openr1" ]]; then
    TRAIN_FILE="${OPENR1_TRAIN_FILE}"
elif [[ "${DATASET_SOURCE}" == "dapo" ]]; then
    TRAIN_FILE="${TRAIN_FILE}"
else
    echo "ERROR: unsupported DATASET_SOURCE=${DATASET_SOURCE}. Use dapo or openr1." >&2
    exit 1
fi

echo "===== Run Config ====="
echo "PROJECT_NAME=${project_name}"
echo "EXP_NAME=${exp_name}"
echo "CKPTS_ROOT=${CKPTS_ROOT}"
echo "CKPTS_DIR=${CKPTS_DIR}"
echo "RESUME_MODE=${RESUME_MODE}"
if [[ "${RESUME_MODE}" == "resume_path" ]]; then
    echo "RESUME_PATH=${RESUME_PATH}"
fi
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATASET_SOURCE=${DATASET_SOURCE}"
echo "======================"

temperature=0.2
top_p=0.7
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# CUDA allocator (vLLM memory pool is incompatible with expandable_segments)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

sp_size=2
use_dynamic_bsz=True
offload=False
gen_tp=1

# In agent_system path, keep per-step generation as single response.
# Group repetition is controlled by env.rollout.n.
python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.return_full_prompt=True \
    +data.default_data_source=math_dapo \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${train_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.test_freq=0 \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=${RESUME_MODE} \
    ${RESUME_PATH:+trainer.resume_from_path="${RESUME_PATH}"} \
    trainer.use_agent_system=${USE_AGENT_SYSTEM} \
    env.env_name=math \
    env.max_steps=1 \
    env.rollout.n=${n_resp_per_prompt} \
    skill_library.enable=${SKILL_LIBRARY_ENABLE} \
    "+skill_library.seed_skills=${SKILL_SEED_SKILLS}" \
    skill_library.selection.score_mode=${SKILL_SELECTION_MODE} \
    skill_library.selection.confidence_threshold=${SKILL_SELECTION_CONFIDENCE_THRESHOLD} \
    skill_library.selection.epsilon=${SKILL_SELECTION_EPSILON} \
    skill_library.selection.llm_candidate_top_k=${SKILL_SELECTION_LLM_CANDIDATE_TOP_K} \
    skill_library.selection.inject_to_prompt=True \
    skill_library.manager.enable=${SKILL_MANAGER_ENABLE} \
    "+skill_library.manager.actions=${SKILL_MANAGER_ACTIONS}" \
    skill_library.manager.payload_skills_limit=${SKILL_MANAGER_PAYLOAD_SKILLS_LIMIT} \
    skill_library.manager.sync_every=${SKILL_MANAGER_SYNC_EVERY} \
    skill_library.manager.use_model=${SKILL_MANAGER_USE_MODEL} \
    skill_library.manager.use_model_generate=${SKILL_MANAGER_USE_MODEL_GENERATE} \
    skill_library.manager.action_format=${SKILL_MANAGER_ACTION_FORMAT} \
    skill_library.manager.fetch_mode=${SKILL_MANAGER_FETCH_MODE} \
    skill_library.manager.load_mode=${SKILL_MANAGER_LOAD_MODE} \
    ++actor_rollout_ref.actor.policy_loss.master_weight=${MASTER_WEIGHT} \
    ++actor_rollout_ref.actor.policy_loss.manager_weight=${MANAGER_WEIGHT} \
    skill_library.reward.enable=${SKILL_REWARD_ENABLE} \
    +skill_library.reward.reward_when_skill_used_and_correct=${SKILL_REWARD_WHEN_USED_AND_CORRECT} \
    +skill_library.conclude_skill.enable=${SKILL_CONCLUDE_ENABLE} \
    +skill_library.conclude_skill.validation.min_chars=${SKILL_CONCLUDE_MIN_CHARS} \
    +skill_library.conclude_skill.validation.min_words=${SKILL_CONCLUDE_MIN_WORDS} \
    +skill_library.conclude_skill.validation.min_unique_word_ratio=${SKILL_CONCLUDE_MIN_UNIQUE_WORD_RATIO} \
    skill_library.abstraction.enable=${SKILL_ABSTRACTION_ENABLE} \
    skill_library.warmup.enable=${SKILL_WARMUP_ENABLE} \
    skill_library.warmup.steps=${SKILL_WARMUP_STEPS}
