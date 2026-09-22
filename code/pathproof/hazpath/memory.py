"""Verified path memory, and the contextual-bandit policy over it.

This is the part that separates the design from caching. A cache is keyed on the
question and returns the answer. This is keyed on a *fingerprint of the problem*
and returns a *method*, which is then adapted to nouns it has never seen.

Nothing here trains a model. The policy is a similarity- and reward-weighted
vote over stored records, and for any decision it can name the priors that
produced it -- which is why it can be shown to an evaluator.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path as FsPath
from typing import Any, Dict, List, Optional, Tuple

from .path import Observables, Path


# ---------------------------------------------------------------- fingerprint --
@dataclass(frozen=True)
class Fingerprint:
    """The context c(x). Structural only: it describes the SHAPE of a problem
    and never its literal nouns, or retrieval would degenerate into lookup.

    Fixed before any split is scored. Not fitted, not tuned.
    """

    metric_kind: str          # money | count | rate | share
    entity_classes: Tuple[str, ...]
    source_classes: Tuple[str, ...]
    temporal_grain: str       # none | fy | quarter | month
    needs_join: bool
    needs_normalization: bool
    needs_dedup: bool

    def similarity(self, other: "Fingerprint") -> float:
        """Registered similarity: Jaccard index over the feature sets.

            sim(c, c') = |F(c) INTERSECT F(c')| / |F(c) UNION F(c')|

        F(c) is the set built by _features(): one token per categorical field
        plus one per entity and source class. Unweighted and unparameterised on
        purpose -- a similarity function with knobs is a similarity function
        that can be turned until the result appears."""
        mine = self._features()
        theirs = other._features()
        inter = len(mine & theirs)
        union = len(mine | theirs)
        return inter / union if union else 0.0

    def _features(self) -> set:
        f = {f"metric={self.metric_kind}", f"grain={self.temporal_grain}",
             f"join={self.needs_join}", f"norm={self.needs_normalization}",
             f"dedup={self.needs_dedup}"}
        f |= {f"entity={e}" for e in self.entity_classes}
        f |= {f"source={s}" for s in self.source_classes}
        return f

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entity_classes"] = list(self.entity_classes)
        d["source_classes"] = list(self.source_classes)
        return d


# -------------------------------------------------------------------- record --
@dataclass
class Record:
    """One (problem, method, observables, reward). The unified pair the topic
    asks for, with the outcome attached."""

    question: str
    fingerprint: Fingerprint
    path: Path
    observables: Observables
    reward: float
    database: str

    def as_dict(self) -> Dict[str, Any]:
        return dict(
            question=self.question,
            fingerprint=self.fingerprint.as_dict(),
            path=[{"verb": s.verb, "args": s.args} for s in self.path.steps],
            path_signature=list(self.path.signature),
            observables=self.observables.as_dict(),
            reward=round(self.reward, 6),
            database=self.database)


# -------------------------------------------------------------------- memory --
class PathMemory:
    """Append-only store with a freeze. Once frozen, writes raise.

    The freeze is not a nicety. The transfer claim is only meaningful if the
    memory could not have seen the problem it is being scored on, and a store
    that can still be written to during scoring cannot support that claim.
    """

    def __init__(self) -> None:
        self._records: List[Record] = []
        self._frozen = False
        self._composites: Dict[str, Tuple[str, ...]] = {}

    # ----------------------------------------------------------- lifecycle --
    def add(self, rec: Record) -> None:
        if self._frozen:
            raise RuntimeError(
                "path memory is frozen; scoring must not write to it")
        self._records.append(rec)

    def freeze(self) -> None:
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    def __len__(self) -> int:
        return len(self._records)

    @property
    def databases(self) -> set:
        return {r.database for r in self._records}

    @property
    def composites(self) -> Dict[str, Tuple[str, ...]]:
        return dict(self._composites)

    def register_composite(self, name: str, signature: Tuple[str, ...]) -> None:
        if self._frozen:
            raise RuntimeError("cannot promote a composite after the freeze")
        self._composites[name] = signature

    # ------------------------------------------------------------- policy --
    # The store is partitioned by the sign of the utility. This is not a
    # convenience: a retrieval weight must be non-negative or the normalised
    # policy is not a probability distribution. Successful paths BIAS
    # selection; failed paths SUPPRESS materially equivalent retries. They are
    # two different mechanisms and they are kept apart.

    TAU_EXCLUDE = 0.60          # similarity above which a failure suppresses

    def successes(self) -> List[Record]:
        return [r for r in self._records if r.reward > 0]

    def failures(self) -> List[Record]:
        return [r for r in self._records if r.reward <= 0]

    def suppressed(self, fp: Fingerprint) -> List[Tuple[Tuple, Record]]:
        """Material signatures a prior failure rules out for this context.

        A path that produced a wrong answer on a sufficiently similar problem is
        removed from the candidate set rather than being ranked low, so it is
        not retried by accident.
        """
        out = []
        for rec in self.failures():
            if fp.similarity(rec.fingerprint) >= self.TAU_EXCLUDE:
                out.append((rec.path.material_signature, rec))
        return out

    def retrieve(self, fp: Fingerprint, *, k: int = 5,
                 exclude_databases: Optional[set] = None
                 ) -> List[Tuple[float, Record]]:
        """Top-k prior successes by sim(c, c_m) * u_m, with u_m > 0.

        Weights are non-negative by construction. Failures never appear here;
        their effect is exclusion, applied in policy().
        """
        exclude = exclude_databases or set()
        blocked = {sig for sig, _ in self.suppressed(fp)}
        scored = []
        for rec in self.successes():
            if rec.database in exclude:
                continue
            if rec.path.material_signature in blocked:
                continue
            sim = fp.similarity(rec.fingerprint)
            if sim <= 0:
                continue
            scored.append((sim * rec.reward, rec))
        scored.sort(key=lambda t: -t[0])
        return scored[:k]

    def policy(self, fp: Fingerprint, *, k: int = 5,
               exclude_databases: Optional[set] = None
               ) -> List[Tuple[Tuple[str, ...], float, List[str]]]:
        """P(method signature | context), with the evidence for each entry.

        Returns (signature, weight, [questions that support it]) so a reviewer
        can be shown WHY a method was preferred, which a policy gradient cannot
        provide.
        """
        votes: Dict[Tuple[str, ...], float] = {}
        support: Dict[Tuple[str, ...], List[str]] = {}
        for weight, rec in self.retrieve(fp, k=k, exclude_databases=exclude_databases):
            sig = rec.path.signature
            votes[sig] = votes.get(sig, 0.0) + weight
            support.setdefault(sig, []).append(rec.question)
        total = sum(votes.values()) or 1.0
        ranked = sorted(votes.items(), key=lambda t: -t[1])
        return [(sig, w / total, support[sig]) for sig, w in ranked]

    # --------------------------------------------------------------- io ----
    def to_json(self, path: FsPath) -> None:
        payload = {
            "frozen": self._frozen,
            "composites": {k: list(v) for k, v in self._composites.items()},
            "records": [r.as_dict() for r in self._records],
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
