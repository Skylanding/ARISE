# Copyright 2026 ARISE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Two-tier skill library with the five management operations.

Implements Appendix C.2 of the ARISE paper. The library state is partitioned
into an active *cache* (``C_c`` entries, default 10) that serves as the
selection pool inducing the option set ``\u03a9(C_t)`` and a *reservoir*
(``C_r`` entries, default 100) that archives displaced skills for potential
future promotion. Every entry carries a running utility ``r\u0304_k`` updated via
exponential moving average and a cumulative selection count ``n_k`` used to
guard against deleting never-tried skills.

Five operations execute in a fixed order at the end of each training step
(Appendix C.3):

``UPDATE`` -> ``ADD`` -> ``EVICT`` -> ``LOAD`` -> ``DELETE``

so that (i) the utility of the selected skill reflects the most recent reward
before any insertion or eviction, (ii) a newly added skill can immediately
trigger an eviction if the cache is full, and (iii) the ``LOAD`` check runs
after eviction so a just-evicted skill cannot immediately re-enter.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from verl.workers.arise.skill_document import SkillDocument

logger = logging.getLogger(__name__)


@dataclass
class LibraryEntry:
    """One row of the skill library.

    Attributes mirror the per-entry state described in Appendix C.2.
    """

    document: SkillDocument
    utility: float = 0.0
    usage_count: int = 0
    info_gain: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_selected_at: Optional[float] = None

    @property
    def name(self) -> str:
        return self.document.skill_name

    def to_dict(self) -> dict:
        return {
            "document": self.document.to_dict(),
            "utility": self.utility,
            "usage_count": self.usage_count,
            "info_gain": self.info_gain,
            "created_at": self.created_at,
            "last_selected_at": self.last_selected_at,
        }


class SkillLibrary:
    """Cache + reservoir skill store realising the library kernel ``K`` of Eq. 4.

    The class is intentionally framework-agnostic: it does no tensor work and
    no I/O beyond optional checkpointing via :meth:`save` / :meth:`load`. The
    skill_manager owns scoring, the skill_concluder owns inference, and this
    object owns *which* entries are alive at any point.

    Parameters
    ----------
    cache_capacity:
        ``C_c``. Default 10 (Table 5).
    reservoir_capacity:
        ``C_r``. Default 100 (Table 5).
    ema_beta:
        ``\u03b2`` for the utility EMA, default 0.9 (Table 5).
    delete_percentile:
        Reservoir entries whose utility falls below this percentile *and* have
        ``usage_count == 0`` are garbage-collected. Default 0.10 (the
        10th-percentile threshold ``q_{10}`` of Appendix C.2).
    """

    def __init__(
        self,
        cache_capacity: int = 10,
        reservoir_capacity: int = 100,
        ema_beta: float = 0.9,
        delete_percentile: float = 0.10,
    ) -> None:
        if cache_capacity <= 0:
            raise ValueError("cache_capacity must be > 0")
        if reservoir_capacity < 0:
            raise ValueError("reservoir_capacity must be >= 0")
        if not 0.0 < ema_beta < 1.0:
            raise ValueError("ema_beta must be in (0, 1)")
        if not 0.0 <= delete_percentile <= 1.0:
            raise ValueError("delete_percentile must be in [0, 1]")

        self.cache_capacity = cache_capacity
        self.reservoir_capacity = reservoir_capacity
        self.ema_beta = ema_beta
        self.delete_percentile = delete_percentile

        self._cache: Dict[str, LibraryEntry] = {}
        self._reservoir: Dict[str, LibraryEntry] = {}

    # -- Read-only views ------------------------------------------------------

    @property
    def cache(self) -> List[LibraryEntry]:
        """Active cache, sorted by descending utility for deterministic iteration."""
        return sorted(self._cache.values(), key=lambda e: e.utility, reverse=True)

    @property
    def reservoir(self) -> List[LibraryEntry]:
        return sorted(self._reservoir.values(), key=lambda e: e.utility, reverse=True)

    def cache_size(self) -> int:
        return len(self._cache)

    def reservoir_size(self) -> int:
        return len(self._reservoir)

    def total_size(self) -> int:
        return self.cache_size() + self.reservoir_size()

    def has(self, skill_name: str) -> bool:
        return skill_name in self._cache or skill_name in self._reservoir

    def get(self, skill_name: str) -> Optional[LibraryEntry]:
        return self._cache.get(skill_name) or self._reservoir.get(skill_name)

    def iter_candidates(self) -> Iterable[LibraryEntry]:
        """Iterate over cache entries (the selection pool ``\u03a9(C_t)``)."""
        return self.cache

    # -- Seeding --------------------------------------------------------------

    def seed(self, documents: Sequence[SkillDocument]) -> None:
        """Initialise the cache with seed skills (Appendix C.1 Table 7)."""
        for doc in documents:
            if self.cache_size() >= self.cache_capacity:
                break
            if not self.has(doc.skill_name):
                self._cache[doc.skill_name] = LibraryEntry(document=doc)

    # -- Five management operations (Appendix C.2) ---------------------------

    def update(self, skill_name: str, reward: float) -> bool:
        """``UPDATE``: EMA refresh of the selected entry's utility.

        ``r\u0304_{z_t} <- \u03b2 r\u0304_{z_t} + (1 - \u03b2) R_t``. Returns ``True`` if the
        entry was located, ``False`` otherwise. Skills not selected this step
        keep their previous utility.
        """
        entry = self._cache.get(skill_name) or self._reservoir.get(skill_name)
        if entry is None:
            return False
        entry.utility = self.ema_beta * entry.utility + (1.0 - self.ema_beta) * float(reward)
        entry.usage_count += 1
        entry.last_selected_at = time.time()
        return True

    def add(self, document: SkillDocument, initial_utility: float = 0.0) -> bool:
        """``ADD``: insert a freshly concluded skill into the cache.

        Returns ``False`` if a skill with the same name already exists in the
        library (the conclusion rollout would otherwise overwrite history).
        """
        if document is None or self.has(document.skill_name):
            return False
        self._cache[document.skill_name] = LibraryEntry(
            document=document, utility=float(initial_utility)
        )
        return True

    def evict(self) -> Optional[LibraryEntry]:
        """``EVICT``: spill the weakest cache entry into the reservoir.

        Triggered when ``len(cache) > C_c``. Returns the evicted entry or
        ``None`` if no eviction was needed.
        """
        if self.cache_size() <= self.cache_capacity:
            return None
        weakest = min(self._cache.values(), key=lambda e: e.utility)
        self._cache.pop(weakest.name)
        self._reservoir[weakest.name] = weakest
        return weakest

    def load(self) -> Optional[LibraryEntry]:
        """``LOAD``: promote a reservoir entry that outperforms the weakest cache slot.

        Returns the swapped-in entry, or ``None`` if no swap occurred.
        """
        if not self._reservoir or self.cache_size() == 0:
            return None
        best_reservoir = max(self._reservoir.values(), key=lambda e: e.utility)
        weakest_cache = min(self._cache.values(), key=lambda e: e.utility)
        if best_reservoir.utility <= weakest_cache.utility:
            return None
        # Swap them.
        self._reservoir.pop(best_reservoir.name)
        self._cache.pop(weakest_cache.name)
        self._cache[best_reservoir.name] = best_reservoir
        self._reservoir[weakest_cache.name] = weakest_cache
        return best_reservoir

    def delete(self) -> List[LibraryEntry]:
        """``DELETE``: garbage-collect stale reservoir entries.

        Removes reservoir skills satisfying *both* (i) utility below the
        ``delete_percentile`` quantile of the reservoir and (ii) zero
        cumulative selection count, mirroring the ``q_{10}`` rule in
        Appendix C.2.
        """
        if not self._reservoir:
            return []
        utilities = sorted(e.utility for e in self._reservoir.values())
        if not utilities:
            return []
        cutoff_idx = max(0, int(len(utilities) * self.delete_percentile) - 1)
        cutoff = utilities[cutoff_idx]
        removed: List[LibraryEntry] = []
        for name, entry in list(self._reservoir.items()):
            if entry.utility < cutoff and entry.usage_count == 0:
                self._reservoir.pop(name)
                removed.append(entry)
        # Hard cap on reservoir size: drop lowest-utility surplus entries.
        if len(self._reservoir) > self.reservoir_capacity:
            ordered = sorted(self._reservoir.values(), key=lambda e: e.utility)
            for entry in ordered[: len(self._reservoir) - self.reservoir_capacity]:
                self._reservoir.pop(entry.name)
                removed.append(entry)
        return removed

    # -- Composed step -------------------------------------------------------

    def step(
        self,
        selected_name: Optional[str],
        selected_reward: Optional[float],
        new_document: Optional[SkillDocument],
        new_initial_utility: float = 0.0,
    ) -> dict:
        """Execute the fixed sequence ``UPDATE -> ADD -> EVICT -> LOAD -> DELETE``.

        Convenience for the trainer; returns a small dict of bookkeeping
        information useful for logging.
        """
        info = {"updated": False, "added": False, "evicted": None, "loaded": None, "deleted": []}
        if selected_name is not None and selected_reward is not None:
            info["updated"] = self.update(selected_name, selected_reward)
        if new_document is not None:
            info["added"] = self.add(new_document, initial_utility=new_initial_utility)
        evicted = self.evict()
        if evicted is not None:
            info["evicted"] = evicted.name
        loaded = self.load()
        if loaded is not None:
            info["loaded"] = loaded.name
        deleted = self.delete()
        if deleted:
            info["deleted"] = [e.name for e in deleted]
        return info

    # -- Checkpointing -------------------------------------------------------

    def save(self, path: str) -> None:
        payload = {
            "cache_capacity": self.cache_capacity,
            "reservoir_capacity": self.reservoir_capacity,
            "ema_beta": self.ema_beta,
            "delete_percentile": self.delete_percentile,
            "cache": [e.to_dict() for e in self.cache],
            "reservoir": [e.to_dict() for e in self.reservoir],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    @classmethod
    def load_from(cls, path: str) -> "SkillLibrary":
        payload = json.loads(Path(path).read_text())
        lib = cls(
            cache_capacity=payload["cache_capacity"],
            reservoir_capacity=payload["reservoir_capacity"],
            ema_beta=payload["ema_beta"],
            delete_percentile=payload["delete_percentile"],
        )
        for record in payload["cache"]:
            doc = SkillDocument(**record["document"])
            lib._cache[doc.skill_name] = LibraryEntry(
                document=doc,
                utility=record["utility"],
                usage_count=record["usage_count"],
                info_gain=record.get("info_gain", 0.0),
                created_at=record.get("created_at", time.time()),
                last_selected_at=record.get("last_selected_at"),
            )
        for record in payload["reservoir"]:
            doc = SkillDocument(**record["document"])
            lib._reservoir[doc.skill_name] = LibraryEntry(
                document=doc,
                utility=record["utility"],
                usage_count=record["usage_count"],
                info_gain=record.get("info_gain", 0.0),
                created_at=record.get("created_at", time.time()),
                last_selected_at=record.get("last_selected_at"),
            )
        return lib
