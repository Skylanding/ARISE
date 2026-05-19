# Copyright 2026 ARISE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hierarchical reward composition and the four-level marginal skill reward.

Implements Section 3.3 and Appendix D.2 of the ARISE paper. Two reward
signals are produced per training step:

* The *task* reward ``r_task \u2208 {0, 1}`` is the standard binary verifier
  outcome, which drives the GRPO objective at the worker level regardless of
  skill usage (Eq. 9).
* The *marginal skill quality* reward ``R_s \u2208 {r_1, r_2, r_3, r_4}`` is a
  counterfactual comparison between the ``G`` original rollouts and the
  verification rollout ``\u03c4_{G+1}`` conditioned on the newly concluded skill
  (the table next to Eq. of \u00a73.3).

The total reward used by Phase II is the simple composition
``R = r_task + r_skill \u2208 {0, 1, 2}`` summarised in Table 8 of the paper,
where ``r_skill = 1`` exactly when the trajectory both used a selected skill
and produced a correct answer. Combined with the GRPO group-relative
normalisation (Eq. 8), this gives strictly higher advantage to skill-augmented
correct trajectories whenever they coexist with unaided correct trajectories
in the same group.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence


class RsLevel(Enum):
    """Symbolic names for the four marginal skill reward levels.

    The numerical values are not the rewards themselves -- those are
    configurable through :class:`AriseReward.rs_values` -- they just index the
    four counterfactual outcomes in the table of Section 3.3.
    """

    R1_HARMFUL = 1  # \u2203 correct unaided, but skill rollout fails
    R2_NEUTRAL = 2  # all unaided incorrect, skill rollout also incorrect
    R3_REDUNDANT = 3  # \u2203 correct unaided, skill rollout also correct
    R4_CRITICAL = 4  # all unaided incorrect, skill rollout correct


@dataclass
class HierarchicalReward:
    """Per-trajectory reward bookkeeping for one rollout.

    ``r_task`` and ``r_skill`` follow Table 8: ``r_skill`` is only granted on
    trajectories that *both* used a selected skill *and* produced a correct
    answer. The total ``R = r_task + r_skill`` is what gets fed into the
    GRPO advantage normaliser of Eq. 8.
    """

    r_task: int
    r_skill: int

    @property
    def total(self) -> int:
        return int(self.r_task) + int(self.r_skill)


class AriseReward:
    """Compose task + skill rewards and compute the marginal ``R_s``.

    Parameters
    ----------
    rs_values:
        Numerical reward attached to the four ``R_s`` levels, in the order
        ``(r_1, r_2, r_3, r_4)``. The paper enforces ``r_1 < r_2 < r_3 < r_4``
        but does not pin down the exact magnitudes; defaults follow common
        practice for binary-outcome counterfactual baselines.
    skill_bonus:
        Magnitude of ``r_skill`` granted under the "skill used + task
        correct" cell of Table 8 (default 1).
    """

    def __init__(
        self,
        rs_values: Sequence[float] = (-1.0, 0.0, 0.5, 1.0),
        skill_bonus: int = 1,
    ) -> None:
        if len(rs_values) != 4:
            raise ValueError("rs_values must be a 4-tuple (r1, r2, r3, r4)")
        r1, r2, r3, r4 = rs_values
        if not (r1 < r2 < r3 < r4):
            raise ValueError(
                "rs_values must satisfy r1 < r2 < r3 < r4 (Section 3.3 ordering)"
            )
        if skill_bonus < 0:
            raise ValueError("skill_bonus must be non-negative")
        self.rs_values = tuple(float(v) for v in rs_values)
        self.skill_bonus = int(skill_bonus)

    # -- Per-trajectory composition (Table 8) -------------------------------

    def trajectory_reward(self, skill_used: bool, task_correct: bool) -> HierarchicalReward:
        r_task = 1 if task_correct else 0
        if skill_used and task_correct:
            r_skill = self.skill_bonus
        else:
            r_skill = 0
        return HierarchicalReward(r_task=r_task, r_skill=r_skill)

    def batch_trajectory_rewards(
        self,
        skill_used: Sequence[bool],
        task_correct: Sequence[bool],
    ) -> List[HierarchicalReward]:
        if len(skill_used) != len(task_correct):
            raise ValueError("skill_used and task_correct must have the same length")
        return [
            self.trajectory_reward(bool(s), bool(c))
            for s, c in zip(skill_used, task_correct)
        ]

    # -- Marginal skill quality (Section 3.3 table) -------------------------

    @staticmethod
    def classify_rs(
        original_outcomes: Sequence[bool],
        verification_outcome: bool,
    ) -> RsLevel:
        """Locate the counterfactual ``R_s`` cell.

        ``original_outcomes`` are the binary correctness of the ``G`` unaided
        rollouts, ``verification_outcome`` is the binary correctness of the
        skill-conditioned verification rollout ``\u03c4_{G+1}``.
        """
        if len(original_outcomes) == 0:
            raise ValueError("original_outcomes must not be empty")
        any_correct = any(bool(o) for o in original_outcomes)
        verify_correct = bool(verification_outcome)

        if any_correct and verify_correct:
            return RsLevel.R3_REDUNDANT
        if any_correct and not verify_correct:
            return RsLevel.R1_HARMFUL
        if not any_correct and verify_correct:
            return RsLevel.R4_CRITICAL
        return RsLevel.R2_NEUTRAL

    def marginal_skill_reward(
        self,
        original_outcomes: Sequence[bool],
        verification_outcome: bool,
    ) -> float:
        """Return the numerical ``R_s`` for use as the new entry's seed utility.

        This is what initialises ``r\u0304_{m\u0303}`` in Algorithm 1 line 7 / 18,
        and what flows through the unified policy gradient of Eq. 7 via the
        group-relative advantage of Eq. 8.
        """
        level = self.classify_rs(original_outcomes, verification_outcome)
        return self.rs_values[level.value - 1]

    # -- Diagnostics --------------------------------------------------------

    def rs_distribution(
        self, levels: Sequence[RsLevel]
    ) -> "dict[RsLevel, float]":
        """Empirical distribution over ``R_s`` levels (Table 4 in the paper)."""
        total = len(levels)
        if total == 0:
            return {lvl: 0.0 for lvl in RsLevel}
        counts = {lvl: 0 for lvl in RsLevel}
        for lvl in levels:
            counts[lvl] = counts.get(lvl, 0) + 1
        return {lvl: counts[lvl] / total for lvl in RsLevel}
