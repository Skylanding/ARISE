# ARISE recipe

Training recipe corresponding to **ARISE: Agent Reasoning with Intrinsic Skill
Evolution in Hierarchical Reinforcement Learning** (NeurIPS 2026 submission).

This recipe assembles the ARISE training loop on top of veRL's existing GRPO /
DAPO trainer. It is a **reference implementation** of the algorithm described
in Sections 3.1-3.4 and Appendices A-D of the paper. The core ARISE modules
live in [`verl/workers/arise/`](../../verl/workers/arise/) and the trainer
wrapper in
[`verl/trainer/ppo/arise_trainer.py`](../../verl/trainer/ppo/arise_trainer.py).

## Layout

```
recipe/arise/
├── README.md                            this file
├── config/
│   ├── arise_qwen3_4b.yaml              Qwen3-4B-Instruct-2507 (Table 1 row 1)
│   └── arise_phi4_mini.yaml             Phi-4-mini-instruct  (Table 1 row 2)
├── prompts/
│   └── skill_conclusion.txt             Figure 4 summary-rollout template
├── seeds/
│   └── seed_skills.json                 5 seed skills from Table 7
├── run_arise_qwen3_4b.sh
└── run_arise_phi4_mini.sh
```

## Quick start

```bash
# 1. Prepare DeepScaleR + the four eval benchmarks (see ../../examples/data_preprocess/).
export ARISE_MODEL_PATH=/path/to/qwen3-4b-it
export ARISE_TRAIN_FILE=/path/to/deepscaler/train.parquet
export ARISE_AMC23=/path/to/amc23/test.parquet
export ARISE_AIME24=/path/to/aime24/test.parquet
export ARISE_AIME25=/path/to/aime25/test.parquet
export ARISE_OMNI=/path/to/omni-math/test.parquet

# 2. Launch training.
bash recipe/arise/run_arise_qwen3_4b.sh
```

The script forwards to `python -m verl.trainer.main_ppo --recipe arise`. Any
remaining CLI arguments are passed through to verl's trainer so the usual
Slurm / wandb overrides keep working.

## Mapping between the paper and the code

| Paper artefact | Code |
|---|---|
| `SkillDocument` (Table 6 / Appendix C.1) | `verl/workers/arise/skill_document.py` |
| 5 library ops (Appendix C.2 / C.3) | `verl/workers/arise/skill_library.py` |
| `\mu_\theta` scoring + injection (Eq. 6, 10 / Appendix C.4) | `verl/workers/arise/skill_manager.py` |
| `O_{G+1}` summary rollout (Appendix D.1 / Figure 4) | `verl/workers/arise/skill_conclusion.py` |
| Four-level `R_s` + `R = r_task + r_skill` (Section 3.3 / Table 8 / D.2) | `verl/workers/arise/reward.py` |
| Phase I / Phase II schedule (Section 3.4) | `verl/workers/arise/phase_scheduler.py` |
| Per-step composition (Algorithm 1) | `verl/trainer/ppo/arise_trainer.py` |
| Hyperparameters of Table 5 | `recipe/arise/config/arise_*.yaml` |
| Seed skills of Table 7 | `recipe/arise/seeds/seed_skills.json` |

## Skill-library hyperparameters (Table 5)

| Group | Symbol | Default | YAML key |
|---|---|---|---|
| Manager | `\sigma` (softmax temp) | 1.0 | `arise.temperature` |
| Manager | `\epsilon` (\epsilon-greedy) | 0.10 | `arise.exploration_eps` |
| Manager | `\delta` (confidence gate) | 0.35 | `arise.confidence_threshold` |
| Manager | max skill tokens for scoring | 128 | `arise.max_skill_score_tokens` |
| Library | `C_c` (cache capacity) | 10 | `arise.cache_capacity` |
| Library | `C_r` (reservoir capacity) | 100 | `arise.reservoir_capacity` |
| Library | `\beta` (utility EMA) | 0.9 | `arise.ema_beta` |
| Schedule | `N_w` (warm-up steps) | 500 | `arise.warmup_steps` |
| Reward | `r_skill` bonus | 1 | `arise.skill_bonus` |
| Reward | `(r_1, r_2, r_3, r_4)` | `(-1, 0, 0.5, 1)` | `arise.rs_values` |

## Notes

* This recipe is intended as a *reference implementation*: it documents how
  every component of the paper maps to a class / function. The default config
  numbers reproduce the Table 1 setup; whether you can hit Table 1 accuracy
  depends on your verl deployment (rollout engine, dynamic-sampling
  implementation, evaluation protocol, etc.).
* `skill_conclusion.txt` is shipped both as a stand-alone prompt file (for
  manual inspection / patching) and as a constant in
  `skill_conclusion.SUMMARY_PROMPT_TEMPLATE`. They must stay in sync.
* The library is checkpointed to `ARISE_CKPTS_DIR/library_step*.json` and can
  be reloaded with `SkillLibrary.load_from(path)` if you need to resume
  training.
