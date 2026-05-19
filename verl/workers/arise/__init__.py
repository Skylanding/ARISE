# Copyright 2026 ARISE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ARISE: hierarchical RL with intrinsic skill evolution.

Reference implementation of the modules described in Sections 3.1-3.4 and
Appendices C-D of the ARISE paper. The package exposes:

* :class:`SkillDocument` -- the five-field JSON schema (Table 6).
* :class:`SkillLibrary` -- the cache + reservoir store with the five
  management operations (Appendix C.2).
* :class:`SkillManager` -- policy-driven scoring and selection (Eq. 6, 10).
* :class:`SkillConcluder` -- the inference-only summary rollout (Appendix D.1).
* :class:`AriseReward` -- the four-level marginal quality reward and the
  hierarchical reward composition (Table 8, Appendix D.2).
* :class:`PhaseScheduler` -- the warm-up / skill-augmented training schedule
  (Section 3.4).
"""

from verl.workers.arise.phase_scheduler import Phase, PhaseDecision, PhaseScheduler
from verl.workers.arise.reward import AriseReward, HierarchicalReward, RsLevel
from verl.workers.arise.skill_conclusion import ConclusionResult, SkillConcluder
from verl.workers.arise.skill_document import (
    MAX_DOCUMENT_CHARS,
    PROBLEM_TYPES,
    SkillDocument,
    trace_abstract_fallback,
    validate_and_parse,
)
from verl.workers.arise.skill_library import LibraryEntry, SkillLibrary
from verl.workers.arise.skill_manager import SelectionResult, SkillManager

__all__ = [
    "AriseReward",
    "ConclusionResult",
    "HierarchicalReward",
    "LibraryEntry",
    "MAX_DOCUMENT_CHARS",
    "PROBLEM_TYPES",
    "Phase",
    "PhaseDecision",
    "PhaseScheduler",
    "RsLevel",
    "SelectionResult",
    "SkillConcluder",
    "SkillDocument",
    "SkillLibrary",
    "SkillManager",
    "trace_abstract_fallback",
    "validate_and_parse",
]
