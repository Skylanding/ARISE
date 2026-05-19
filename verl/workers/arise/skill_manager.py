# Copyright 2026 ARISE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Policy-driven skill scoring and selection.

Implements the upper-level *Manager* of the hierarchical policy described in
Sections 3.1-3.2 of the ARISE paper. Concretely:

* Each candidate option ``o_k`` is scored by the length-normalised conditional
  log-likelihood of its skill document under the shared policy ``\u03c0_\u03b8``
  (Eq. 6).
* Scores are turned into a selection distribution via a temperature softmax
  with a confidence gate ``\u03b4`` and an ``\u03b5``-greedy exploration tail (Eq. 10).
* On selection, the chosen document is prepended to the query under the
  ``SKILL:`` prefix (Appendix C.4).

The :meth:`SkillManager.score` API is intentionally minimal so the trainer
can plug in either an HF model (``forward(input_ids).logits``) or a vLLM-style
log-prob endpoint; see ``score_with_callable`` for the latter.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from verl.workers.arise.skill_document import SkillDocument
from verl.workers.arise.skill_library import LibraryEntry, SkillLibrary

# A callable that returns the *sum* of log-probabilities of ``skill_text`` under
# the model given the query as context. Implementations are free to truncate
# ``skill_text`` to a maximum number of tokens (the paper uses 128, see
# Appendix C.4).
LogProbFn = Callable[[str, str], float]


@dataclass
class SelectionResult:
    """Outcome of one selection step.

    ``entry`` is ``None`` when the confidence gate forced the unaided branch
    ``o_\u2205`` (Eq. 10) or when no candidates are available.
    """

    entry: Optional[LibraryEntry]
    probabilities: List[float]
    scores: List[float]
    gated: bool
    explored: bool

    @property
    def used_skill(self) -> bool:
        return self.entry is not None


class SkillManager:
    """Upper-level policy ``\u03bc_\u03b8`` over options.

    Parameters
    ----------
    temperature:
        Softmax temperature ``\u03c3`` (Table 5 default = 1.0).
    confidence_threshold:
        ``\u03b4`` (Table 5 default = 0.35). Below this the manager abstains and
        falls back to ``o_\u2205``.
    exploration_eps:
        ``\u03b5`` for the ``\u03b5``-greedy exploration tail (Table 5 default = 0.1).
    max_skill_score_tokens:
        Hard cap on how many tokens of the skill document are passed through
        the log-prob callable (Table 5 = 128). Implementations may treat this
        as advisory.
    rng_seed:
        Optional integer seed for the exploration RNG, useful in tests.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        confidence_threshold: float = 0.35,
        exploration_eps: float = 0.1,
        max_skill_score_tokens: int = 128,
        rng_seed: Optional[int] = None,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 <= exploration_eps <= 1.0:
            raise ValueError("exploration_eps must be in [0, 1]")
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.exploration_eps = exploration_eps
        self.max_skill_score_tokens = max_skill_score_tokens
        self._rng = random.Random(rng_seed)

    # -- Scoring (Eq. 6) -----------------------------------------------------

    @staticmethod
    def length_normalised(log_prob_sum: float, num_tokens: int) -> float:
        if num_tokens <= 0:
            return float("-inf")
        return log_prob_sum / float(num_tokens)

    def score_with_callable(
        self,
        query: str,
        candidates: Sequence[LibraryEntry],
        log_prob_fn: LogProbFn,
        token_count_fn: Optional[Callable[[str], int]] = None,
    ) -> List[float]:
        """Compute per-candidate length-normalised log-probability scores.

        ``log_prob_fn(query, skill_text)`` must return the *sum* of token-level
        log-probabilities of the skill text conditioned on the query. Length
        normalisation uses ``token_count_fn`` if provided (recommended: a real
        tokeniser), else falls back to whitespace splitting.
        """
        scores: List[float] = []
        for entry in candidates:
            text = entry.document.to_json()
            lp_sum = float(log_prob_fn(query, text))
            n_tokens = token_count_fn(text) if token_count_fn else len(text.split())
            scores.append(self.length_normalised(lp_sum, n_tokens))
        return scores

    # -- Selection (Eq. 10) --------------------------------------------------

    def softmax(self, scores: Sequence[float]) -> List[float]:
        if not scores:
            return []
        # Numerically stable softmax.
        m = max(scores)
        exps = [math.exp((s - m) / self.temperature) for s in scores]
        total = sum(exps)
        if total <= 0.0:
            n = len(scores)
            return [1.0 / n] * n
        return [e / total for e in exps]

    def select(
        self,
        candidates: Sequence[LibraryEntry],
        scores: Sequence[float],
    ) -> SelectionResult:
        """Sample an option using the temperature softmax with gate + epsilon.

        Returns a :class:`SelectionResult`; ``entry is None`` whenever
        ``max_k p_k < \u03b4`` (the unaided fallback ``o_\u2205``).
        """
        if not candidates:
            return SelectionResult(entry=None, probabilities=[], scores=[], gated=True, explored=False)

        probs = self.softmax(scores)
        max_prob = max(probs)
        if max_prob < self.confidence_threshold:
            return SelectionResult(
                entry=None,
                probabilities=probs,
                scores=list(scores),
                gated=True,
                explored=False,
            )

        explored = False
        if self.exploration_eps > 0.0 and self._rng.random() < self.exploration_eps:
            chosen_idx = self._rng.randrange(len(candidates))
            explored = True
        else:
            # Argmax with deterministic tie-breaking on the lower index.
            chosen_idx = max(range(len(candidates)), key=lambda i: probs[i])

        return SelectionResult(
            entry=candidates[chosen_idx],
            probabilities=probs,
            scores=list(scores),
            gated=False,
            explored=explored,
        )

    # -- Convenience: score + select in one pass -----------------------------

    def score_and_select(
        self,
        query: str,
        library: SkillLibrary,
        log_prob_fn: LogProbFn,
        token_count_fn: Optional[Callable[[str], int]] = None,
    ) -> SelectionResult:
        candidates = list(library.iter_candidates())
        scores = self.score_with_callable(query, candidates, log_prob_fn, token_count_fn)
        return self.select(candidates, scores)

    # -- Prompt injection (Appendix C.4) ------------------------------------

    @staticmethod
    def inject(query: str, document: Optional[SkillDocument]) -> str:
        """Prepend the skill document to the query under the ``SKILL:`` prefix.

        When ``document is None`` (the unaided branch) the query is returned
        unchanged, so the worker sees the original question without any prefix.
        """
        if document is None:
            return query
        return f"{document.to_prompt_prefix()}{query}"
