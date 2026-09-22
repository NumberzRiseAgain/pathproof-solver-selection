"""The Path: an executable program over the primitive vocabulary.

A Path is the computational object the SPEED DIAL argument rests on. It is not
a SQL string and it is not a prompt. It is an ordered list of named steps that a
domain expert can read, that a machine can execute deterministically, and that
can be stored, retrieved, compared and edited as data.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .primitives import PRIMITIVES, State, step_digest

# Database-specific identifiers -> semantic classes. Fixed before scoring and
# part of the registered equivalence relation (pre-registration section 5).
# No entry on either side is a database name.
SEMANTIC_CLASS = {
    # sources
    "disposal_dtid": "disposal_event",
    "supply_transactions": "supply_event",
    "shops": "org_unit",
    "materials_hmid": "material_master",
    # fields
    "writeoff_usd": "money", "disposal_cost_usd": "money",
    "unit_cost": "money", "qty": "quantity",
    "nsn": "material_key", "shop_id": "org_key", "supply_id": "event_key",
    "fy": "fiscal_period", "state": "geo_name",
}


@dataclass(frozen=True)
class Step:
    verb: str
    args: Dict[str, Any]

    @property
    def digest(self) -> str:
        return step_digest(self.verb, self.args)

    def arg_class(self) -> str:
        """Arguments reduced to semantic classes, so that two paths over
        different schemas can be compared. `select_source(disposal_dtid)` and
        `select_source(writeoff_fact)` both reduce to `source=disposal_event`
        when the ontology says so; if it does not, they stay distinct.

        The map is fixed before scoring and is part of the registered
        equivalence relation. It never contains a database name.
        """
        parts = []
        for k in sorted(self.args):
            v = self.args[k]
            if k in ("definition", "parts", "reason"):
                continue
            if k == "table":
                parts.append(f"source={SEMANTIC_CLASS.get(str(v), 'other')}")
            elif k in ("column", "key"):
                parts.append(f"field={SEMANTIC_CLASS.get(str(v), 'other')}")
            elif k in ("name", "check", "how", "op"):
                parts.append(f"{k}={v}")
            # literal values (nsn codes, shop ids, fiscal years) are dropped:
            # they are the nouns, and the nouns are what must NOT carry over
        return ",".join(parts)

    def render(self) -> str:
        """One readable line. Long value lists are summarised, because a path a
        human cannot scan is not an interpretability artifact."""
        shown = {k: v for k, v in self.args.items() if k != "definition"}
        parts = []
        for k, v in shown.items():
            if isinstance(v, list) and len(v) > 3:
                parts.append(f"{k}=[{len(v)} values, sum={sum(v):,.2f}]")
            else:
                parts.append(f"{k}={v!r}")
        return f"{self.verb}({', '.join(parts)})"


@dataclass
class Observables:
    """What execution reveals. These are the quantities the reward is built
    from, and every one of them is measured rather than asserted."""

    correct: Optional[bool] = None
    result: Optional[float] = None
    rows_out: int = 0
    latency_s: float = 0.0
    steps: int = 0
    abstained: bool = False
    abstain_reason: str = ""
    validation_failures: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return dict(
            correct=self.correct, result=self.result, rows_out=self.rows_out,
            latency_s=round(self.latency_s, 6), steps=self.steps,
            abstained=self.abstained, abstain_reason=self.abstain_reason,
            validation_failures=self.validation_failures)


@dataclass
class Path:
    name: str
    steps: List[Step] = field(default_factory=list)
    provenance: str = "authored"     # authored | retrieved | adapted | promoted

    # ---------------------------------------------------------------- shape --
    @property
    def signature(self) -> Tuple[str, ...]:
        """The verb sequence, ignoring arguments. Two paths with the same
        signature solve the problem the same way on different nouns, which is
        exactly what should transfer between databases."""
        return tuple(s.verb for s in self.steps)

    @property
    def material_signature(self) -> Tuple[Tuple[str, str], ...]:
        """The registered equivalence relation, pi_m ~= pi.

        Two paths are materially equivalent when their ordered primitive
        sequence is identical after every argument that names a
        database-specific identifier is replaced by its semantic class. This
        is what makes "the same method" a testable statement across databases
        rather than a judgement call made after seeing results.
        """
        return tuple((s.verb, s.arg_class()) for s in self.steps)

    def equivalent_to(self, other: "Path") -> bool:
        return self.material_signature == other.material_signature

    @property
    def digest(self) -> str:
        return step_digest("path", {"steps": [s.digest for s in self.steps]})

    def render(self) -> str:
        """The interpretability exhibit. One line per decision, in order."""
        width = len(str(len(self.steps)))
        lines = [f"  {i:>{width}}. {s.render()}" for i, s in enumerate(self.steps, 1)]
        return "\n".join(lines)

    # ------------------------------------------------------------- execution --
    def execute(self, store) -> Tuple[State, Observables]:
        st = State()
        t0 = time.perf_counter()
        failures = 0
        for step in self.steps:
            fn = PRIMITIVES[step.verb]
            st = fn(st, store, **step.args)
            if st.abstained:
                break
            if step.verb == "validate" and st.notes[-1].endswith("FAIL"):
                failures += 1
        obs = Observables(
            result=st.scalar, rows_out=len(st.rows),
            latency_s=time.perf_counter() - t0, steps=len(self.steps),
            abstained=st.abstained, abstain_reason=st.abstain_reason,
            validation_failures=failures)
        return st, obs

    # --------------------------------------------------------------- editing --
    def substitute(self, *, name: str, **replacements: Dict[str, Any]) -> "Path":
        """Adapt a retrieved path to a new problem by changing arguments only.

        The verb sequence is preserved. This is the operation that makes
        transfer meaningful: what carries across problems is the METHOD, and
        only the nouns change.
        """
        new_steps = []
        for s in self.steps:
            args = dict(s.args)
            for key, val in replacements.get(s.verb, {}).items():
                if key in args:
                    args[key] = val
            new_steps.append(Step(s.verb, args))
        return Path(name=name, steps=new_steps, provenance="adapted")


def reward(obs: Observables, *, w_correct: float = 1.0,
           lam_latency: float = 1e-3, lam_steps: float = 1e-3) -> float:
    """Reward, with the weights fixed in the pre-registration.

    Correctness dominates; cost is a tie-breaker between paths that are both
    right. An honest abstention scores zero rather than negative -- refusing to
    answer is not the same kind of event as answering wrongly, and collapsing
    the two would teach the policy to guess.
    """
    if obs.abstained:
        return 0.0
    if not obs.correct:
        return -1.0
    return (w_correct
            - lam_latency * obs.latency_s
            - lam_steps * obs.steps)
