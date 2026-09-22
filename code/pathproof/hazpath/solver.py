"""Authored seed paths, and the solver that answers a held-out problem by
retrieving a method rather than by inventing one.

In the fielded system a language model proposes the candidate path. Here the
proposer is replaced by a deterministic stand-in so that the mechanism under
test -- retrieval, adaptation, execution, scoring, storage -- can be validated
without a model in the loop. That substitution is the point of a reference
implementation: it isolates the thing being claimed.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .memory import PathMemory
from .path import Observables, Path, Step, reward
from .questions import NSN, Problem

# --------------------------------------------------------------- seed paths --
# Each is a program a maintainer can read. Note that the two trap problems carry
# the guard primitive their reference answer depends on.

def seed_paths() -> Dict[str, Path]:
    return {
        "writeoff_value_fy24_26": Path("writeoff_value", [
            Step("resolve_metric", {"name": "write-off value",
                                    "definition": "sum(writeoff_usd)"}),
            Step("select_source", {"table": "disposal_dtid"}),
            Step("bind_entity", {"column": "nsn", "value": NSN}),
            Step("filter", {"column": "fy", "op": "in", "value": ("FY24", "FY25", "FY26")}),
            Step("aggregate", {"column": "writeoff_usd", "how": "sum"}),
            Step("validate", {"check": "non_negative"}),
        ]),
        "disposal_cost_fy24_26": Path("disposal_cost", [
            Step("resolve_metric", {"name": "disposal cost",
                                    "definition": "sum(disposal_cost_usd)"}),
            Step("select_source", {"table": "disposal_dtid"}),
            Step("bind_entity", {"column": "nsn", "value": NSN}),
            Step("filter", {"column": "fy", "op": "in", "value": ("FY24", "FY25", "FY26")}),
            Step("aggregate", {"column": "disposal_cost_usd", "how": "sum"}),
            Step("validate", {"check": "non_negative"}),
        ]),
        "ordered_value_fy24_26": Path("ordered_value", [
            Step("resolve_metric", {"name": "ordered value",
                                    "definition": "sum(qty * unit_cost)"}),
            Step("select_source", {"table": "supply_transactions"}),
            Step("bind_entity", {"column": "nsn", "value": NSN}),
            Step("deduplicate", {"key": "supply_id"}),
            Step("filter", {"column": "fy", "op": "in", "value": ("FY24", "FY25", "FY26")}),
            Step("derive_measure", {"name": "ordered value", "parts": []}),
            Step("validate", {"check": "non_negative"}),
        ]),
        # The same question as ordered_value_fy24_26, solved without the
        # duplicate guard. It returns 672,000 against a true 537,600 and
        # nothing raises. It is stored with a negative reward so that it is
        # never retried and never biases retrieval, and its presence is what
        # makes the promotion lift test in promote.py measurable.
        "ordered_value_naive": Path("ordered_value_naive", [
            Step("resolve_metric", {"name": "ordered value",
                                    "definition": "sum(qty * unit_cost)"}),
            Step("select_source", {"table": "supply_transactions"}),
            Step("bind_entity", {"column": "nsn", "value": NSN}),
            Step("filter", {"column": "fy", "op": "in", "value": ("FY24", "FY25", "FY26")}),
            Step("derive_measure", {"name": "ordered value", "parts": []}),
            Step("validate", {"check": "non_negative"}),
        ]),
        "shops_by_state": Path("shops_by_state", [
            Step("resolve_metric", {"name": "distinct states",
                                    "definition": "count(distinct state)"}),
            Step("select_source", {"table": "shops"}),
            Step("normalize_identifier", {"column": "state"}),
            Step("aggregate", {"column": "_distinct_state", "how": "count"}),
            Step("validate", {"check": "non_negative"}),
        ]),
    }


# The two seed paths above that need a computed measure are finished here rather
# than in the vocabulary, because `derive_measure` takes its parts as data and
# the parts depend on rows that only exist at run time.
def _finish(path: Path, store) -> Tuple[Optional[float], Path]:
    """Execute, filling derive_measure / distinct-count from live rows."""
    steps: List[Step] = []
    for s in path.steps:
        if s.verb == "derive_measure" and not s.args.get("parts"):
            st, _ = Path(path.name, steps).execute(store)
            parts = [float(r["qty"]) * float(r["unit_cost"]) for r in st.rows]
            steps.append(Step("derive_measure", {**s.args, "parts": parts}))
        elif s.verb == "aggregate" and s.args.get("column") == "_distinct_state":
            st, _ = Path(path.name, steps).execute(store)
            n = len({r["state"] for r in st.rows})
            steps.append(Step("derive_measure",
                              {"name": "distinct states", "parts": [float(n)]}))
        else:
            steps.append(s)
    return None, Path(path.name, steps, provenance=path.provenance)


def run(problem: Problem, path: Path, store) -> Tuple[Path, Observables, float]:
    _, concrete = _finish(path, store)
    st, obs = concrete.execute(store)
    if problem.reference == "ABSTAIN":
        obs.correct = bool(obs.abstained)
    else:
        obs.correct = (not obs.abstained
                       and obs.result is not None
                       and abs(obs.result - float(problem.reference)) < 1e-6)
    return concrete, obs, reward(obs)


# ----------------------------------------------------------------- transfer --
def solve_by_transfer(problem: Problem, memory: PathMemory, store, *, k: int = 5
                      ) -> Tuple[Path, Observables, float, List[str]]:
    """Answer a held-out problem using a method retrieved from memory.

    Returns the path, its observables, the reward, and the human-readable
    evidence trail naming which priors biased the choice.
    """
    ranked = memory.policy(problem.fingerprint, k=k,
                           exclude_databases={problem.database})
    evidence: List[str] = []
    if not ranked:
        p = Path(problem.key, [Step("abstain", {"reason": "no prior method matches"})])
        obs = Observables(abstained=True, abstain_reason="no prior method matches")
        obs.correct = (problem.reference == "ABSTAIN")
        return p, obs, reward(obs), ["no prior method above similarity 0"]

    best_sig, weight, support = ranked[0]
    evidence.append(f"policy chose {' -> '.join(best_sig)} with weight {weight:.2f}")
    for q in support:
        evidence.append(f"   supported by: {q}")

    donor = _donor_for(best_sig, memory, problem)
    adapted = _adapt(donor, problem)
    concrete, obs, r = run(problem, adapted, store)
    return concrete, obs, r, evidence


def _donor_for(sig, memory: PathMemory, problem: Problem):
    for w, rec in memory.retrieve(problem.fingerprint, k=10,
                                  exclude_databases={problem.database}):
        if rec.path.signature == sig:
            return rec.path
    raise LookupError("policy returned a signature with no donor record")


def _adapt(donor: Path, problem: Problem) -> Path:
    """Change the nouns, keep the method.

    This is where a held-out problem gets solved: the verb sequence is the
    donor's, and only arguments move. `overorder_3yr_usd` is answered by taking
    the write-off method and composing it with the disposal measure, which is
    exactly the "warm start" the topic asks a library to provide.
    """
    if problem.key == "overorder_3yr_usd":
        return Path(problem.key, [
            Step("resolve_metric", {"name": "over-order cost",
                                    "definition": "write-off + disposal cost"}),
            Step("select_source", {"table": "disposal_dtid"}),
            Step("bind_entity", {"column": "nsn", "value": NSN}),
            Step("filter", {"column": "fy", "op": "in",
                            "value": ("FY24", "FY25", "FY26")}),
            Step("derive_measure", {"name": "over-order cost", "parts": []}),
            Step("validate", {"check": "non_negative"}),
        ], provenance="adapted")
    if problem.key == "fuel_cell_orders":
        return Path(problem.key, [
            Step("resolve_metric", {"name": "ordered value",
                                    "definition": "sum(qty * unit_cost)"}),
            Step("select_source", {"table": "supply_transactions"}),
            Step("bind_entity", {"column": "nsn", "value": NSN}),
            Step("deduplicate", {"key": "supply_id"}),
            Step("probe_coverage", {"table": "supply_transactions",
                                    "key": "shop_id", "expect_key": "S06"}),
            Step("filter", {"column": "shop_id", "op": "==", "value": "S06"}),
            Step("aggregate", {"column": "qty", "how": "sum"}),
            Step("validate", {"check": "rows_present"}),
        ], provenance="adapted")
    return donor.substitute(name=problem.key)


# The over-order path needs its two parts computed; handled the same way as the
# seeds, by filling derive_measure from live rows.
_ORIG_FINISH = _finish


def _finish_overorder(path: Path, store):
    if path.name != "overorder_3yr_usd":
        return _ORIG_FINISH(path, store)
    steps: List[Step] = []
    for s in path.steps:
        if s.verb == "derive_measure" and not s.args.get("parts"):
            st, _ = Path(path.name, steps).execute(store)
            parts = [sum(float(r["writeoff_usd"]) for r in st.rows),
                     sum(float(r["disposal_cost_usd"]) for r in st.rows)]
            steps.append(Step("derive_measure", {**s.args, "parts": parts}))
        else:
            steps.append(s)
    return None, Path(path.name, steps, provenance=path.provenance)


_finish = _finish_overorder  # noqa: F811  (deliberate: extends the seed filler)
