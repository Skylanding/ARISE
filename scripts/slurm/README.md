# SLURM Script Staging

Use this directory to keep cluster submission wrappers separate from core training code.

## Suggested Structure

- `scripts/slurm/oppo/` - OPPO-related submission wrappers
- `scripts/slurm/dapo/` - DAPO/agent-library submission wrappers
- `scripts/slurm/common/` - shared helper snippets (optional)

## Minimal Conventions

- Name files by intent and hardware profile, e.g.:
  - `submit_oppo_tmw_h200_1gpu.sh`
  - `submit_dapo_tmw_h200_2gpu.sh`
- Keep algorithm-specific hyperparameters in recipe scripts/config files.
- Keep submission scripts focused on:
  - partition/nodelist/gres/cpu/mem/time
  - log path conventions
  - environment bootstrap (conda/CUDA)

## Why this split

This avoids mixing operational cluster details with algorithm source files under `recipe/` and `verl/`,
which makes upgrades and debugging easier.
