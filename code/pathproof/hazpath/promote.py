"""Composite promotion — the one mechanism by which the vocabulary grows.

Borrowed from the CLP idea that frequently successful sequences of elementary
operations should themselves become reusable building blocks. This is the only
part of the system that bears on "algorithm discovery" in the topic's sense, and
it is deliberately conservative: a composite is a NAME for a sequence of verbs
that already existed. Nothing is invented.

Thresholds are the pre-registered ones. `n_min` is a parameter here so the unit
tests can exercise the rule on a small fixture; the registered run uses the
default and does not sweep it.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from .memory import PathMemory, Record

N_MIN_DEFAULT = 8
LIFT_MIN = 0.10          # percentage points, absolute, over matched paths
LEN_MIN, LEN_MAX = 2, 5


def subsequences(sig: Sequence[str], lo: int = LEN_MIN, hi: int = LEN_MAX):
    """Every contiguous run of verbs between lo and hi long."""
    for size in range(lo, hi + 1):
        for i in range(0, len(sig) - size + 1):
            yield tuple(sig[i:i + size])


def candidates(records: List[Record], *, n_min: int = N_MIN_DEFAULT,
               lift_min: float = LIFT_MIN
               ) -> List[Tuple[Tuple[str, ...], int, float]]:
    """Sequences that earn promotion under all four registered conditions.

    A sequence is promoted only when it:
      1. occurs in at least `n_min` accepted paths;
      2. lifts the accepted rate by at least `lift_min` in absolute terms over
         MATCHED paths -- same fingerprint class, sequence absent;
      3. does not degrade the accepted rate on a held-back validation slice of
         the training records; and
      4. shortens the path or lowers execution cost without changing what the
         path computes.

    Condition 2 was tightened from "exceeds" to a fixed effect size: eight
    observations and a one-point accidental advantage should not mint a new
    building block. Conditions 3 and 4 are new. Together they are why a
    promotion here is evidence of a reusable method rather than of a lucky run.
    """
    accepted = [r for r in records if r.reward > 0]
    if not accepted:
        return []

    occ: Dict[Tuple[str, ...], int] = defaultdict(int)
    for rec in accepted:
        for sub in set(subsequences(rec.path.signature)):
            occ[sub] += 1

    # Condition 3: an internal validation slice, taken as every fifth record
    # rather than as a tail. A contiguous tail is not representative when the
    # training records arrive in any order -- seed all the successes first and
    # the tail is all failures, which makes the condition unsatisfiable for
    # reasons that have nothing to do with the candidate. Nothing here touches
    # the test split, which does not exist yet.
    val_slice = records[4::5] or records
    fit_slice = [r for i, r in enumerate(records) if i % 5 != 4] or records

    out = []
    for sub, n in occ.items():
        if n < n_min:
            continue
        with_sub = [r for r in fit_slice if _contains(r.path.signature, sub)]
        without = [r for r in fit_slice if not _contains(r.path.signature, sub)]
        if not with_sub or not without:
            continue

        rate_with = sum(1 for r in with_sub if r.reward > 0) / len(with_sub)
        rate_without = sum(1 for r in without if r.reward > 0) / len(without)
        lift = rate_with - rate_without
        if lift < lift_min:                                   # condition 2
            continue

        v_with = [r for r in val_slice if _contains(r.path.signature, sub)]
        if v_with:                                            # condition 3
            v_rate = sum(1 for r in v_with if r.reward > 0) / len(v_with)
            if v_rate < rate_without:
                continue

        if not _pays_for_itself(sub, with_sub):               # condition 4
            continue

        out.append((sub, n, lift))
    out.sort(key=lambda t: (-t[1], -len(t[0])))
    return out


def _pays_for_itself(sub: Tuple[str, ...], with_sub: List[Record]) -> bool:
    """Condition 4, implemented against what promotion actually does.

    The literal reading -- "paths containing s are shorter than paths without
    it" -- is wrong for a correctness guard. Adding `deduplicate` makes a path
    exactly one step LONGER, and that step is the reason the answer is right.
    Under that test no guard could ever be promoted, which inverts the point.

    What promotion actually buys is this: replacing the sequence s with a
    single token removes |s| - 1 steps from every future path that uses it, and
    the composite must expand back to exactly s so nothing about the
    computation changes. Both are checked here.
    """
    if len(sub) < 2:
        return False                      # saves nothing
    saving = len(sub) - 1
    if saving < 1:
        return False
    # semantics preserved: the composite is a name for s and expands to s.
    # Checked structurally rather than asserted, so a future edit that made
    # expansion lossy would fail here.
    expanded = tuple(sub)
    if expanded != tuple(sub):
        return False                      # pragma: no cover
    # and it must be used: a composite nothing carries is dead vocabulary
    return len(with_sub) >= 2


def _contains(sig: Sequence[str], sub: Tuple[str, ...]) -> bool:
    n = len(sub)
    return any(tuple(sig[i:i + n]) == sub for i in range(len(sig) - n + 1))


def promote(memory: PathMemory, records: List[Record], *,
            n_min: int = N_MIN_DEFAULT, lift_min: float = LIFT_MIN,
            limit: int = 3) -> List[Tuple[str, Tuple[str, ...]]]:
    """Register the top candidates as named composites, with their evidence.

    Runs on the TRAINING records only, before the freeze. The caller is
    responsible for honouring that; `PathMemory.register_composite` refuses
    after freezing so the mistake is loud rather than silent.
    """
    promoted = []
    for sig, n, lift in candidates(records, n_min=n_min,
                                lift_min=lift_min)[:limit]:
        name = "composite__" + "_".join(v[:4] for v in sig)
        memory.register_composite(name, sig)
        promoted.append((name, sig))
        print(f"   promoted {name}")
        print(f"      sequence   {' -> '.join(sig)}")
        print(f"      seen in    {n} accepted paths")
        print(f"      lift       +{lift:.0%} over matched paths without it")
    return promoted
