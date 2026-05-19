# Copyright 2026 ARISE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Inference-only skill conclusion (the ``O_{G+1}`` rollout).

Implements the summary rollout described in Appendix D.1 of the ARISE paper.
After every rollout group, the same policy ``\u03c0_\u03b8`` distils a representative
successful trace ``\u03c4^+`` into a structured :class:`SkillDocument` via a single
inference rollout (no gradients), governed by the prompt template in
Figure 4.

The class deliberately keeps the LLM call abstract behind a callable so the
trainer can wire in either an HF ``generate`` loop or a vLLM endpoint without
this module knowing about either.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from verl.workers.arise.skill_document import (
    MAX_DOCUMENT_CHARS,
    SkillDocument,
    trace_abstract_fallback,
    validate_and_parse,
)

logger = logging.getLogger(__name__)

GenerationFn = Callable[[str], str]

SUMMARY_PROMPT_TEMPLATE = """You are a skill distiller for math reasoning.
Given one question and trajectories from the same rollout group, summarize ONE
reusable skill.

Question: {question}

Group trajectories:
{traces_block}

Output MUST be a valid JSON object and nothing else (no markdown / code
fences). Schema:
{{"skill_name", "problem_type", "key_insight", "method", "check"}}

Rules:
- Be generic and transferable; do not copy specific numbers.
- Keep the whole skill within {max_chars} content characters (counting
  field values only, not JSON framing).
- key_insight is the most important field.
- method must contain 2-3 concise steps.
- Focus on improving correctness, not style.
"""


@dataclass
class ConclusionResult:
    """Outcome of one :meth:`SkillConcluder.conclude` call.

    ``document`` is ``None`` if both the primary parse and the fallback failed
    (extremely rare in practice; logged at warning level).
    """

    document: Optional[SkillDocument]
    used_fallback: bool
    raw_generation: str
    prompt: str


class SkillConcluder:
    """Build the summary prompt, invoke the model, validate the output.

    Parameters
    ----------
    max_traces:
        Up to how many positive-advantage traces to include in the prompt.
        The paper uses 2 (Figure 4); the worker discards any extras.
    trace_clip_chars:
        Per-trace truncation in characters (Figure 4 caption: 400).
    max_chars:
        Hard cap on the produced skill document, propagated into the prompt
        for the model and verified at parse time. Defaults to
        :data:`MAX_DOCUMENT_CHARS` (220, Appendix D.1).
    """

    def __init__(
        self,
        max_traces: int = 2,
        trace_clip_chars: int = 400,
        max_chars: int = MAX_DOCUMENT_CHARS,
    ) -> None:
        if max_traces <= 0:
            raise ValueError("max_traces must be > 0")
        if trace_clip_chars <= 0:
            raise ValueError("trace_clip_chars must be > 0")
        self.max_traces = max_traces
        self.trace_clip_chars = trace_clip_chars
        self.max_chars = max_chars

    # -- Prompt assembly -----------------------------------------------------

    def _clip(self, text: str) -> str:
        if text is None:
            return ""
        text = str(text).strip()
        if len(text) <= self.trace_clip_chars:
            return text
        return text[: self.trace_clip_chars] + "\u2026"

    def build_prompt(self, question: str, positive_traces: Sequence[str]) -> str:
        """Render the Figure 4 template.

        ``positive_traces`` should contain *successful* trajectories
        ``\u03c4^+ = {\u03c4_i | \u00c2_i > 0}`` (see Algorithm 1, line 6 / 17).
        """
        if not positive_traces:
            raise ValueError("positive_traces must contain at least one trace")
        selected = list(positive_traces)[: self.max_traces]
        block_lines: List[str] = []
        for idx, trace in enumerate(selected, start=1):
            block_lines.append(f"[SUCCESS #{idx}] {self._clip(trace)}")
        return SUMMARY_PROMPT_TEMPLATE.format(
            question=question,
            traces_block="\n".join(block_lines),
            max_chars=self.max_chars,
        )

    # -- End-to-end conclusion ----------------------------------------------

    def conclude(
        self,
        question: str,
        positive_traces: Sequence[str],
        generate_fn: GenerationFn,
    ) -> ConclusionResult:
        """Run one inference rollout and return the validated skill.

        ``generate_fn(prompt) -> str`` should perform a single generation with
        sampling parameters as in Figure 4 (temperature 0.7, top-p 0.95, max
        new tokens 192). All four validation stages from Appendix D.1 are
        applied, and a :func:`trace_abstract_fallback` is constructed when the
        primary parse fails.
        """
        if not positive_traces:
            return ConclusionResult(
                document=None,
                used_fallback=False,
                raw_generation="",
                prompt="",
            )
        prompt = self.build_prompt(question, positive_traces)
        try:
            raw = str(generate_fn(prompt))
        except Exception as exc:
            logger.warning("skill conclusion generation failed: %s", exc)
            raw = ""

        doc = validate_and_parse(raw)
        used_fallback = False
        if doc is None:
            used_fallback = True
            doc = trace_abstract_fallback(positive_traces[0])
            if doc.length() > self.max_chars:
                doc = None

        return ConclusionResult(
            document=doc,
            used_fallback=used_fallback,
            raw_generation=raw,
            prompt=prompt,
        )
