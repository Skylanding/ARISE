<p align="center">
  <h1 align="center">
  ARISE: Agent Reasoning with Intrinsic Skill Evolution<br>in Hierarchical Reinforcement Learning<br>
  <sub><sub></sub></sub>
  </h1>
  <p align="center">
    <strong>Yu Li</strong><sup>1</sup>
    &nbsp;&nbsp;
    <strong>Rui Miao</strong><sup>2</sup>
    &nbsp;&nbsp;
    <strong>Zhengling Qi</strong><sup>1✉</sup>
    &nbsp;&nbsp;
    <strong>Tian Lan</strong><sup>1✉</sup>
    <br>
    <sup>1</sup>George Washington University&nbsp;&nbsp;
    <sup>2</sup>University of Texas at Dallas
    <br>
    <a href='https://github.com/Skylanding/ARISE'><img src='https://img.shields.io/badge/GitHub-Code-black?logo=github'></a>&nbsp;
    <a href='https://neurips.cc/'><img src='https://img.shields.io/badge/NeurIPS-2026-blue'></a>&nbsp;
    <br>
    <img src="figure/overview.png">
  </p>
  <br>
</p>

## Abstract

Reinforcement learning with verifiable rewards has become the dominant paradigm for improving mathematical reasoning in language models, yet existing methods treat each problem instance in isolation and discard the reusable strategies that emerge during training. We introduce **ARISE** (Agent Reasoning via Intrinsic Skill Evolution), a hierarchical reinforcement learning framework in which a single shared policy operates at two levels under one set of parameters, serving simultaneously as a policy over options that selects from a persistent skill library and as an intra-option policy that generates solutions conditioned on the selected skill. Every library entry is cast as an option induced by the policy itself; skill selection scores each candidate through the policy's own conditional log-likelihood; and skill conclusion distills ground-truth traces into structured documents through a dedicated inference step, so that selection, generation, and solution reasoning are jointly optimized by one unified policy gradient. The reward structure is hierarchical, coupling a binary solution reward for the intra-option policy with a marginal skill quality reward that attributes credit through a counterfactual comparison against unaided rollouts, thereby driving the co-evolution of reasoning ability and library quality through a shared advantage signal. Experiments on two instruction-tuned base models across seven competition and Olympiad-level benchmarks show that ARISE consistently outperforms GRPO-family algorithms and memory-augmented baselines, with the largest gains on out-of-distribution tasks. Ablation studies confirm that each component contributes to the observed improvements, and that library quality and reasoning performance rise in tandem throughout training.

## Installation

```bash
git clone https://github.com/Skylanding/ARISE.git
cd ARISE

pip install -e .
pip install -r requirements.txt
pip install vllm
pip install flash-attn --no-build-isolation
```

**Requirements:** Python >= 3.10, CUDA >= 12.1, 8x GPUs.

## Training

### Step 1: Configure Paths

Open `recipe/arise/run_arise_qwen3_4b.sh` and set the following environment variables to your local paths:

```bash
export ARISE_MODEL_PATH="/path/to/qwen3-4b-instruct"        # Qwen3-4B-Instruct-2507
export ARISE_TRAIN_FILE="/path/to/deepscaler/train.parquet" # DeepScaleR (~40K math problems)
export ARISE_AMC23="/path/to/amc23/test.parquet"            # in-distribution
export ARISE_AIME24="/path/to/aime24/test.parquet"          # in-distribution
export ARISE_AIME25="/path/to/aime25/test.parquet"          # in-distribution
export ARISE_OMNI="/path/to/omni-math/test.parquet"         # out-of-distribution
export ARISE_CKPTS_DIR="/path/to/checkpoints"               # checkpoint output directory
```

### Step 2: Launch Training

```bash
cd ARISE
bash recipe/arise/run_arise_qwen3_4b.sh
```

The script invokes `python -m verl.trainer.main_ppo --recipe arise` with the full set of Hydra overrides. It will:

1. Initialize Ray and distributed FSDP workers (8 GPUs by default).
2. Seed the skill library with 5 generic mathematical heuristics from `recipe/arise/seeds/seed_skills.json`.
3. Run **Phase I** (warm-up, `N_w = 500` steps): standard GRPO/DAPO objective on the binary task reward while populating the skill library through inference-only summary rollouts `O_{G+1}`.
4. Run **Phase II** (skill-augmented): activate the upper-level skill manager, score each cache entry via length-normalized log-probability (Eq. 6), sample an option through a temperature softmax with confidence gate (Eq. 10), and credit selection + generation through one unified group-relative advantage (Eq. 7).
5. After each step, apply the five library management operations (`UPDATE → ADD → EVICT → LOAD → DELETE`) and checkpoint the library to `${ARISE_CKPTS_DIR}/library_step*.json`.
6. Save model checkpoints periodically and log to both console and Weights & Biases.

### Key Parameters

| Category | Parameter | Value |
|----------|-----------|-------|
| Manager | `arise.temperature` (σ) | `1.0` |
| Manager | `arise.exploration_eps` (ε) | `0.10` |
| Manager | `arise.confidence_threshold` (δ) | `0.35` |
| Manager | `arise.max_skill_score_tokens` | `128` |
| Library | `arise.cache_capacity` (C_c) | `10` |
| Library | `arise.reservoir_capacity` (C_r) | `100` |
| Library | `arise.ema_beta` (β) | `0.9` |
| Conclusion | `arise.summary_max_chars` | `350` |
| Conclusion | `arise.max_summary_traces` | `2` |
| Schedule | `arise.warmup_steps` (N_w) | `500` |
| Reward | `arise.rs_values` (r_1, r_2, r_3, r_4) | `[-1.0, 0.0, 0.5, 1.0]` |
| Reward | `arise.skill_bonus` (r_skill) | `1` |
| Data | `data.max_prompt_length` / `max_response_length` | `2048` / `4096` |
| Data | `data.train_batch_size` | `64` |
| Data | `actor_rollout_ref.rollout.n` | `8` (G in Eq. 8) |
| Optim | `actor.optim.lr` | `1e-6` |
| Optim | `actor.clip_ratio` (ε_c) | `0.2` |
| Rollout | `rollout.name` / `temperature` / `top_p` | `vllm` / `0.7` / `0.95` |
| Trainer | `n_gpus_per_node` / `nnodes` | `8` / `1` |

See `recipe/arise/config/arise_qwen3_4b.yaml` for the complete configuration. A second recipe `recipe/arise/run_arise_phi4_mini.sh` reproduces the Phi-4-mini-instruct row of Table 1 in the paper.

## Evaluation

We evaluate on three in-distribution math competition benchmarks (**AMC 2023**, **AIME 2024**, **AIME 2025**) and the out-of-distribution Olympiad benchmark **Omni-MATH** (with Algebra / Number Theory / Combinatorics / Geometry subdomains). All numbers in Table 1 of the paper are Pass@1 averaged over 32 independent evaluation runs.

### AIME 2024

```bash
cd ARISE

bash recipe/dapo/eval/scripts/eval_math.sh \
    "${CKPTS_DIR}/global_step_XXX" \
    aime24
```

### AIME 2025

```bash
cd ARISE

bash recipe/dapo/eval/scripts/eval_math.sh \
    "${CKPTS_DIR}/global_step_XXX" \
    aime25
```

### AMC 2023

```bash
cd ARISE

EVAL_DATASET=amc23 \
bash recipe/dapo/eval/scripts/eval_math.sh \
    "${CKPTS_DIR}/global_step_XXX"
```

### Omni-MATH (out-of-distribution)

```bash
cd ARISE

EVAL_DATASET=omni-math \
bash recipe/dapo/eval/scripts/eval_math.sh \
    "${CKPTS_DIR}/global_step_XXX"
```

> Replace `global_step_XXX` with the actual checkpoint step you want to evaluate. Evaluation uses greedy decoding with ε = 0 (exploration disabled), so the manager selects deterministically from the trained skill cache. To use the `amc23` and `omni-math` aliases, add the corresponding `TEST_FILE` entries to the dataset `case` block in `recipe/dapo/eval/scripts/eval_math.sh`.

## Project Structure

```
ARISE/
├── docs/                            # Documentation
├── examples/                        # Data preprocessing
│   └── data_preprocess/             # DeepScaleR / AMC / AIME / Omni-MATH prep
├── figure/                          # Paper figures
├── recipe/
│   ├── arise/                       # ARISE recipe (this work)
│   │   ├── config/                  # arise_qwen3_4b.yaml, arise_phi4_mini.yaml
│   │   ├── seeds/                   # seed_skills.json (Table 7)
│   │   ├── prompts/                 # skill conclusion prompt (Figure 4)
│   │   ├── run_arise_qwen3_4b.sh
│   │   ├── run_arise_phi4_mini.sh
│   │   └── README.md
│   └── dapo/                        # DAPO base trainer + eval scripts
├── scripts/                         # Utility scripts (HF↔Megatron, diagnose)
├── tests/                           # Unit tests
└── verl/                            # verl framework core
    ├── trainer/ppo/
    │   ├── arise_trainer.py         # AriseStep composer (Algorithm 1)
    │   └── ray_trainer.py           # GRPO/DAPO training loop
    └── workers/arise/               # ARISE core modules
        ├── skill_document.py        # Table 6 schema + 4-stage validation
        ├── skill_library.py         # Cache + Reservoir + 5 ops (Appendix C.2)
        ├── skill_manager.py         # Log-prob scoring + softmax gate (Eq. 6/10)
        ├── skill_conclusion.py      # Inference summary rollout (Appendix D.1)
        ├── reward.py                # Four-level R_s + R_t (Table 8 / D.2)
        └── phase_scheduler.py       # Phase I / II transition (Section 3.4)
```

## Citation

```bibtex
@article{li2026arise,
  title={Arise: Agent reasoning with intrinsic skill evolution in hierarchical reinforcement learning},
  author={Li, Yu and Miao, Rui and Qi, Zhengling and Lan, Tian},
  journal={arXiv preprint arXiv:2603.16060},
  year={2026}
}
```

## Acknowledgements

Built on top of [verl](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning for LLMs).

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
