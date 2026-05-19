# Copyright 2026 ARISE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ARISE training step composer.

This module glues together the ARISE building blocks (skill library, manager,
concluder, reward composer, phase scheduler) into a single :class:`AriseStep`
object that wraps one optimisation step on top of an existing verl GRPO / DAPO
training loop.

It is deliberately *not* a Ray trainer of its own: the heavy lifting -- model
sharding, vLLM rollout, FSDP optimisation, advantage estimation, the clipped
GRPO surrogate -- continues to live in :mod:`verl.trainer.ppo.ray_trainer` and
:mod:`verl.trainer.ppo.core_algos`. The wrapper merely exposes the ARISE hooks
that the underlying trainer needs to call at well-defined points in the step::

    decision = step.begin_step(global_step)
    selection = step.select_skill(query, log_prob_fn, ...)   # Phase II only
    prompt    = step.inject(query, selection)
    # ... run G solution rollouts with `prompt` ...
    rewards   = step.compose_rewards(skill_used, task_correct)
    # ... GRPO advantage + clipped surrogate update through verl ...
    step.conclude_and_maintain(question, positive_traces, generate_fn,
                               original_outcomes, verify_outcome,
                               selected_skill_name=selection.entry.name if selection.entry else None,
                               selected_reward=rewards[selected_idx].total
                                              if selection.used_skill else None)

The two callables ``log_prob_fn`` and ``generate_fn`` are abstract so the
caller can plug in either an HF model loop or a vLLM endpoint without this
module having to know about either.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from verl.workers.arise import (
    AriseReward,
    HierarchicalReward,
    Phase,
    PhaseDecision,
    PhaseScheduler,
    SelectionResult,
    SkillConcluder,
    SkillDocument,
    SkillLibrary,
    SkillManager,
)

# Re-export to keep the trainer namespace flat for downstream imports.
HierarchicalReward = HierarchicalReward  # noqa: PLW0127

logger = logging.getLogger(__name__)


@dataclass
class AriseConfig:
    """Hyperparameters mirroring Table 5 of the paper.

    Defaults match the "ARISE-specific" rows of that table. Everything below
    the double rule (group size, learning rate, etc.) is owned by the
    underlying verl trainer config and *not* duplicated here.
    """

    # Skill selection (Manager)
    temperature: float = 1.0
    exploration_eps: float = 0.1
    confidence_threshold: float = 0.35
    max_skill_score_tokens: int = 128

    # Skill library
    cache_capacity: int = 10
    reservoir_capacity: int = 100
    ema_beta: float = 0.9
    delete_percentile: float = 0.10

    # Skill conclusion (O_{G+1})
    max_summary_traces: int = 2
    summary_trace_clip_chars: int = 400
    summary_max_chars: int = 220

    # Training schedule
    warmup_steps: int = 500
    skill_bonus: int = 1
    rs_values: Sequence[float] = (-1.0, 0.0, 0.5, 1.0)

    # Optional: path to a JSON file with seed skill documents (Table 7).
    seed_skills_path: Optional[str] = None
    # Optional: directory in which to checkpoint the library each step.
    library_checkpoint_dir: Optional[str] = None

    rng_seed: Optional[int] = None


@dataclass
class AriseStepLog:
    """Diagnostic record returned at the end of every ``AriseStep`` call.

    Useful for wandb / tensorboard logging without leaking framework details
    into the inner ARISE modules.
    """

    global_step: int
    phase: Phase
    selected_skill: Optional[str]
    used_skill: bool
    gated: bool
    explored: bool
    rs_level: Optional[str] = None
    rs_value: Optional[float] = None
    library_cache_size: int = 0
    library_reservoir_size: int = 0
    new_skill_name: Optional[str] = None
    used_fallback: bool = False
    maintenance: dict = field(default_factory=dict)


class AriseStep:
    """One optimisation step's worth of ARISE bookkeeping.

    The class is stateless across steps in everything except the library and
    the RNG; both are intentional so checkpointing is a single JSON file.
    """

    def __init__(self, config: AriseConfig) -> None:
        self.config = config
        self.library = SkillLibrary(
            cache_capacity=config.cache_capacity,
            reservoir_capacity=config.reservoir_capacity,
            ema_beta=config.ema_beta,
            delete_percentile=config.delete_percentile,
        )
        self.manager = SkillManager(
            temperature=config.temperature,
            confidence_threshold=config.confidence_threshold,
            exploration_eps=config.exploration_eps,
            max_skill_score_tokens=config.max_skill_score_tokens,
            rng_seed=config.rng_seed,
        )
        self.concluder = SkillConcluder(
            max_traces=config.max_summary_traces,
            trace_clip_chars=config.summary_trace_clip_chars,
            max_chars=config.summary_max_chars,
        )
        self.reward = AriseReward(
            rs_values=config.rs_values,
            skill_bonus=config.skill_bonus,
        )
        self.schedule = PhaseScheduler(warmup_steps=config.warmup_steps)

        if config.seed_skills_path:
            self._load_seed_skills(config.seed_skills_path)

    # -- Initialisation helpers ---------------------------------------------

    def _load_seed_skills(self, path: str) -> None:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, list):
            raise ValueError(f"seed skills file must contain a JSON list, got {type(payload).__name__}")
        seed_docs = [SkillDocument(**entry) for entry in payload]
        self.library.seed(seed_docs)
        logger.info(
            "ARISE: seeded cache with %d skills from %s", self.library.cache_size(), path
        )

    # -- Per-step hooks (called by the surrounding trainer) -----------------

    def begin_step(self, global_step: int) -> PhaseDecision:
        """Return the :class:`PhaseDecision` for this step."""
        return self.schedule.decide(global_step)

    def select_skill(
        self,
        query: str,
        log_prob_fn,
        token_count_fn=None,
    ) -> SelectionResult:
        """Score the cache against ``query`` and sample one option.

        Returns an empty :class:`SelectionResult` (``entry is None``) whenever
        no candidates are available or the confidence gate fires.
        """
        return self.manager.score_and_select(
            query=query,
            library=self.library,
            log_prob_fn=log_prob_fn,
            token_count_fn=token_count_fn,
        )

    @staticmethod
    def inject(query: str, selection: SelectionResult) -> str:
        return SkillManager.inject(query, selection.entry.document if selection.used_skill else None)

    def compose_rewards(
        self,
        skill_used: Sequence[bool],
        task_correct: Sequence[bool],
        decision: Optional[PhaseDecision] = None,
    ) -> List[HierarchicalReward]:
        """Per-trajectory ``(r_task, r_skill)`` composition.

        In Phase I ``decision.use_hierarchical_reward`` is ``False`` so the
        ``r_skill`` column is zeroed out regardless of ``skill_used``.
        """
        rewards = self.reward.batch_trajectory_rewards(skill_used, task_correct)
        if decision is not None and not decision.use_hierarchical_reward:
            rewards = [HierarchicalReward(r_task=r.r_task, r_skill=0) for r in rewards]
        return rewards

    def conclude_and_maintain(
        self,
        question: str,
        positive_traces: Sequence[str],
        generate_fn,
        original_outcomes: Sequence[bool],
        verification_outcome: bool,
        selected_skill_name: Optional[str] = None,
        selected_reward: Optional[float] = None,
        global_step: int = -1,
        decision: Optional[PhaseDecision] = None,
        selection: Optional[SelectionResult] = None,
    ) -> AriseStepLog:
        """Run the ``O_{G+1}`` rollout, compute ``R_s``, and tick the library.

        ``positive_traces`` should be the trajectories with ``\u00c2_i > 0`` from
        the GRPO group (Algorithm 1, line 8 / 17). When the list is empty the
        conclusion rollout is skipped and the library is only ``UPDATE``-d
        with the selected skill's most recent reward.

        ``global_step``, ``decision`` and ``selection`` are optional bookkeeping
        threaded through to the returned :class:`AriseStepLog` so the trainer
        can log accurate phase / gate / exploration information without
        post-processing.
        """
        new_doc: Optional[SkillDocument] = None
        used_fallback = False
        rs_level_name: Optional[str] = None
        rs_value: Optional[float] = None

        if positive_traces and len(original_outcomes) > 0:
            result = self.concluder.conclude(question, positive_traces, generate_fn)
            new_doc = result.document
            used_fallback = result.used_fallback
            if new_doc is not None:
                rs_level = self.reward.classify_rs(original_outcomes, verification_outcome)
                rs_value = self.reward.marginal_skill_reward(original_outcomes, verification_outcome)
                rs_level_name = rs_level.name

        maintenance = self.library.step(
            selected_name=selected_skill_name,
            selected_reward=selected_reward,
            new_document=new_doc,
            new_initial_utility=rs_value if rs_value is not None else 0.0,
        )

        phase = decision.phase if decision is not None else Phase.SKILL_AUGMENTED
        gated = selection.gated if selection is not None else False
        explored = selection.explored if selection is not None else False

        return AriseStepLog(
            global_step=int(global_step),
            phase=phase,
            selected_skill=selected_skill_name,
            used_skill=selected_skill_name is not None,
            gated=gated,
            explored=explored,
            rs_level=rs_level_name,
            rs_value=rs_value,
            library_cache_size=self.library.cache_size(),
            library_reservoir_size=self.library.reservoir_size(),
            new_skill_name=new_doc.skill_name if new_doc else None,
            used_fallback=used_fallback,
            maintenance=maintenance,
        )

    # -- Checkpointing ------------------------------------------------------

    def save_library(self, step: int) -> Optional[str]:
        if not self.config.library_checkpoint_dir:
            return None
        ckpt_dir = Path(self.config.library_checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"library_step{step:06d}.json"
        self.library.save(str(path))
        return str(path)
