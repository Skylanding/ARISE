# Copyright 2026 ARISE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Skill document schema and the four-stage validation pipeline.

Implements the JSON document layout described in Appendix C.1 (Table 6) of the
ARISE paper. Every entry in the skill library is one :class:`SkillDocument`
instance, which is rendered into prompts during selection and injection
(Appendix C.4).

:func:`validate_and_parse` realises the four stages listed in Appendix D.1:

1. extraction of the JSON object from a free-form generation,
2. parsing and type checking against the schema in Table 6,
3. per-field truncation to the schema limits, and
4. a final hard cap on the total document length (220 characters).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

PROBLEM_TYPES = (
    "algebra",
    "geometry",
    "combinatorics",
    "number_theory",
    "calculus",
    "general",
)

FIELD_LIMITS = {
    "skill_name": 40,
    "key_insight": 160,
    "method_step": 100,
    "check": 100,
}

MAX_METHOD_STEPS = 3
MIN_METHOD_STEPS = 2

# Hard cap on the field-content length (sum of `skill_name`, `problem_type`,
# `key_insight`, every method step, and `check`).
#
# Table 5 of the paper quotes a "220 char" target for skill documents, but
# Figure 3's own example is ~283 content chars and the Table 7 seed skills sit
# in the 257-309 range. We therefore use 350 as the effective hard cap so the
# canonical seeds and the published example all pass validation; the trainer
# can tighten this further through `summary_max_chars` in the YAML config.
MAX_DOCUMENT_CHARS = 350

DEFAULT_CHECK = "Substitute back to verify"

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class SkillDocument:
    """Structured representation of one library entry.

    The five fields mirror Table 6 of the paper. ``method`` holds 2 to 3
    procedural steps; each step is itself capped at
    ``FIELD_LIMITS['method_step']`` characters.
    """

    skill_name: str
    problem_type: str
    key_insight: str
    method: List[str] = field(default_factory=list)
    check: str = DEFAULT_CHECK

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    def content_length(self) -> int:
        """Sum of field-content characters, matching Table 5's "Max skill document length".

        The paper measures the 220-character cap on user-visible content (the
        actual reasoning text of the document), not on JSON framing such as
        field names, quotes, brackets and separators. This is consistent with
        Figure 3, whose example exceeds 220 chars when serialised verbatim,
        and with Appendix C.4's "60-80 tokens after tokenization" estimate.
        """
        return (
            len(self.skill_name)
            + len(self.problem_type)
            + len(self.key_insight)
            + sum(len(step) for step in self.method)
            + len(self.check)
        )

    def length(self) -> int:
        """Backward-compatible alias for :meth:`content_length`."""
        return self.content_length()

    def json_length(self) -> int:
        """Length of the serialised JSON document, including framing overhead."""
        return len(self.to_json())

    def to_prompt_prefix(self) -> str:
        """Format the document for injection in front of a query.

        The ``SKILL:`` prefix acts as a structured delimiter that the worker
        learns to attend to during Phase II (Appendix C.4).
        """
        return f"SKILL:{self.to_json()}\n"


def _truncate(text, limit: int) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "\u2026"


def _normalise_skill_name(name) -> str:
    if not name:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_").lower()
    return cleaned[: FIELD_LIMITS["skill_name"]]


def _normalise_problem_type(value) -> str:
    if not value:
        return "general"
    canonical = str(value).strip().lower().replace(" ", "_")
    return canonical if canonical in PROBLEM_TYPES else "general"


def extract_json_object(raw: str) -> Optional[str]:
    """Stage 1 of the validation pipeline.

    Strip Markdown fences, locate the first balanced ``{...}`` block, and return
    it as a substring. Returns ``None`` if no candidate is found.
    """
    if not raw:
        return None
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "")
    match = _JSON_OBJECT_RE.search(cleaned)
    return match.group(0) if match else None


def _coerce_method(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        chunks = re.split(r"[\n;]|\s\d+[.)]\s", value)
        return [c.strip(" -*\t").strip() for c in chunks if c.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def validate_and_parse(raw: str) -> Optional[SkillDocument]:
    """Run the full four-stage validation pipeline (Appendix D.1).

    Returns a validated :class:`SkillDocument` on success or ``None`` if the
    input cannot be coerced into the schema even after truncation. The caller
    should fall back to :func:`trace_abstract_fallback` when this returns
    ``None``.
    """
    candidate = extract_json_object(raw)
    if candidate is None:
        return None

    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None

    skill_name = _normalise_skill_name(payload.get("skill_name", ""))
    if not skill_name or not _SNAKE_CASE_RE.match(skill_name):
        return None

    method_steps = _coerce_method(payload.get("method"))
    method_steps = [
        _truncate(step, FIELD_LIMITS["method_step"]) for step in method_steps
    ]
    if len(method_steps) > MAX_METHOD_STEPS:
        method_steps = method_steps[:MAX_METHOD_STEPS]
    if len(method_steps) < MIN_METHOD_STEPS:
        return None

    doc = SkillDocument(
        skill_name=skill_name,
        problem_type=_normalise_problem_type(payload.get("problem_type", "general")),
        key_insight=_truncate(
            payload.get("key_insight", ""), FIELD_LIMITS["key_insight"]
        ),
        method=method_steps,
        check=_truncate(payload.get("check", DEFAULT_CHECK), FIELD_LIMITS["check"]),
    )

    if not doc.key_insight:
        return None

    if doc.length() > MAX_DOCUMENT_CHARS:
        doc.check = _truncate(doc.check, FIELD_LIMITS["check"] // 2)
        if doc.length() > MAX_DOCUMENT_CHARS:
            doc.key_insight = _truncate(
                doc.key_insight, FIELD_LIMITS["key_insight"] // 2
            )
        if doc.length() > MAX_DOCUMENT_CHARS:
            return None

    return doc


def trace_abstract_fallback(positive_trace: str) -> SkillDocument:
    """Construct a minimal skill from a successful trace (Appendix D.1).

    Used when the primary summarisation rollout fails to produce a valid JSON
    object, to keep the library populated rather than discarding the step.
    """
    snippet = _truncate(positive_trace, FIELD_LIMITS["key_insight"] - len("Solve by: "))
    return SkillDocument(
        skill_name="trace_abstract",
        problem_type="general",
        key_insight=f"Solve by: {snippet}",
        method=["Reproduce the structure of a successful trace", "Verify each step"],
        check=DEFAULT_CHECK,
    )
