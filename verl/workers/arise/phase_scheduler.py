# Copyright 2026 ARISE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Two-phase training schedule for ARISE.

Implements Section 3.4 of the paper. During Phase I (the first ``N_w``
optimisation steps), the policy over options is disabled and the shared
policy ``\u03c0_\u03b8`` is trained as a single flat worker against the binary task
reward through the standard GRPO objective. The self-evolving skill cycle
(conclusion + heuristic maintenance) still runs every step so that by the time
Phase II activates the library already contains a diverse set of calibrated
seed/early skills.

At step ``N_w + 1`` the policy over options activates: skill selection,
``\u03b5``-greedy exploration, and the hierarchical reward ``R = r_task + r_skill``
all come online together. Skill conclusion continues unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    WARMUP = "warmup"
    SKILL_AUGMENTED = "skill_augmented"


@dataclass
class PhaseDecision:
    """Per-step decision broadcast to the trainer.

    Attributes
    ----------
    phase:
        Which phase the step belongs to.
    score_skills:
        Whether the upper-level manager should score and inject skills.
        ``False`` in Phase I (Algorithm 1 line 4-11 skips selection entirely).
    use_hierarchical_reward:
        Whether to add ``r_skill`` to ``r_task``. ``False`` in Phase I.
    conclude_skills:
        Whether to run the inference-only summary rollout ``O_{G+1}``. Always
        ``True`` in both phases (see Section 3.4: "Skill conclusion executes
        at every step").
    """

    phase: Phase
    score_skills: bool
    use_hierarchical_reward: bool
    conclude_skills: bool

    @property
    def in_warmup(self) -> bool:
        return self.phase is Phase.WARMUP


class PhaseScheduler:
    """Map an optimisation step to a :class:`PhaseDecision`.

    Parameters
    ----------
    warmup_steps:
        ``N_w`` (Table 5 default = 500).
    """

    def __init__(self, warmup_steps: int = 500) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.warmup_steps = warmup_steps

    def decide(self, step: int) -> PhaseDecision:
        """Return the schedule decision for a 1-indexed optimisation ``step``.

        Step ``1..N_w`` is Phase I; step ``N_w + 1`` onwards is Phase II.
        """
        if step <= 0:
            raise ValueError("step must be >= 1 (training steps are 1-indexed)")
        if step <= self.warmup_steps:
            return PhaseDecision(
                phase=Phase.WARMUP,
                score_skills=False,
                use_hierarchical_reward=False,
                conclude_skills=True,
            )
        return PhaseDecision(
            phase=Phase.SKILL_AUGMENTED,
            score_skills=True,
            use_hierarchical_reward=True,
            conclude_skills=True,
        )

    def is_phase_transition(self, step: int) -> bool:
        """``True`` exactly on the step that activates Phase II."""
        return step == self.warmup_steps + 1
