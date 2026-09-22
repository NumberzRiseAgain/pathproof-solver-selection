"""The closed action vocabulary, and a deterministic executor for it.

Every primitive here does one named thing to a table and records what it did.
Nothing in this module calls a model. That separation is the point: a language
model may PROPOSE a path, but the path is a program over these twelve verbs, and
what actually runs is this file.

A primitive is a pure function of (state, arguments). Executing the same path
against the same store twice returns byte-identical results, which
`tests/test_hazpath.py` asserts rather than assumes.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

DATA = Path(__file__).resolve().parent / "data"

# The vocabulary is closed. Adding a verb is a change to the system's expressive
# power and must be a deliberate, reviewed act -- or a promotion under
# promote.py, which composes existing verbs and never invents one.
VOCABULARY = (
    "resolve_metric",
    "select_source",
    "bind_entity",
    "discover_join",
    "normalize_identifier",
    "deduplicate",
    "filter",
    "aggregate",
    "derive_measure",
    "probe_coverage",
    "validate",
    "abstain",
)


# --------------------------------------------------------------------- state --
@dataclass
class State:
    """What flows between primitives. Rows are lists of dicts, kept as plain
    Python so that a reader can print any intermediate step without a debugger."""

    rows: List[Dict[str, Any]] = field(default_factory=list)
    scalar: Optional[float] = None
    sources: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    abstained: bool = False
    abstain_reason: str = ""

    def copy(self) -> "State":
        return State(
            rows=[dict(r) for r in self.rows],
            scalar=self.scalar,
            sources=list(self.sources),
            notes=list(self.notes),
            abstained=self.abstained,
            abstain_reason=self.abstain_reason,
        )


# --------------------------------------------------------------------- store --
def load_store() -> Dict[str, List[Dict[str, str]]]:
    """The synthetic HazMat tables, read once, verbatim.

    The fixture is deliberately dirty: `state` is spelled three ways, three
    supply_id values repeat despite being documented as unique, and one shop has
    no transactions at all. Each defect exists so that a path which skips the
    corresponding primitive returns a confident wrong number instead of an
    error, which is the failure mode this whole design is about.
    """
    store: Dict[str, List[Dict[str, str]]] = {}
    for name in ("materials_hmid", "shops", "supply_transactions", "disposal_dtid"):
        with (DATA / f"{name}.csv").open(newline="", encoding="utf-8") as fh:
            store[name] = list(csv.DictReader(fh))
    return store


def expected_answers() -> Dict[str, Any]:
    with (DATA / "expected_answers.json").open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- primitives --
# Each takes (state, store, **args) and returns a new State. Every one appends a
# human-readable note, because the note IS the interpretability artifact.

def resolve_metric(st: State, store, *, name: str, definition: str) -> State:
    out = st.copy()
    out.notes.append(f'resolve_metric: "{name}" -> {definition}')
    return out


def select_source(st: State, store, *, table: str) -> State:
    if table not in store:
        raise KeyError(f"select_source: no such source {table!r}")
    out = st.copy()
    out.rows = [dict(r) for r in store[table]]
    out.sources.append(table)
    out.notes.append(f"select_source: {table} ({len(out.rows)} rows)")
    return out


def bind_entity(st: State, store, *, column: str, value: str) -> State:
    out = st.copy()
    out.notes.append(f"bind_entity: {column} = {value}")
    out.rows = [r for r in out.rows if r.get(column) == value]
    return out


def discover_join(st: State, store, *, table: str, left: str, right: str) -> State:
    """Join on a shared value where no foreign key is declared.

    The fixture has no FK metadata, so the link is found by value overlap and
    then stated. Rows on the left with no match are kept and flagged, because
    dropping them silently is how a coverage gap becomes an undercount.
    """
    right_rows = store[table]
    index: Dict[str, Dict[str, str]] = {r[right]: r for r in right_rows}
    joined, unmatched = [], 0
    for r in st.rows:
        match = index.get(r.get(left, ""))
        merged = dict(r)
        if match is None:
            unmatched += 1
            merged["_unmatched"] = "1"
        else:
            for k, v in match.items():
                if k not in merged:
                    merged[k] = v
        joined.append(merged)
    out = st.copy()
    out.rows = joined
    out.sources.append(table)
    out.notes.append(
        f"discover_join: {left} -> {table}.{right} "
        f"(no declared FK; matched {len(joined) - unmatched}/{len(joined)})")
    if unmatched:
        out.notes.append(f"   ! {unmatched} row(s) did not match and are flagged")
    return out


_SPELLINGS = {
    "california": "California", "ca": "California", "calif.": "California",
    "calif": "California",
}


def normalize_identifier(st: State, store, *, column: str) -> State:
    """Collapse the three spellings of the same state into one.

    Skipping this makes a group-by return three groups where one exists, and
    every downstream count is wrong without anything raising.
    """
    out = st.copy()
    changed = 0
    for r in out.rows:
        raw = (r.get(column) or "").strip()
        canon = _SPELLINGS.get(raw.lower(), raw)
        if canon != raw:
            changed += 1
        r[column] = canon
    out.notes.append(f"normalize_identifier: {column} ({changed} value(s) rewritten)")
    return out


def deduplicate(st: State, store, *, key: str) -> State:
    """Drop repeated records on a key the schema documents as unique.

    The fixture repeats three supply_id values. A sum over the raw table is
    inflated by exactly those rows, and it looks entirely plausible.
    """
    out = st.copy()
    seen, kept = set(), []
    for r in out.rows:
        k = r.get(key)
        if k in seen:
            continue
        seen.add(k)
        kept.append(r)
    dropped = len(out.rows) - len(kept)
    out.rows = kept
    out.notes.append(f"deduplicate: on {key} ({dropped} duplicate row(s) removed)")
    return out


_OPS: Dict[str, Callable[[Any, Any], bool]] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "in": lambda a, b: a in b,
}


def filter(st: State, store, *, column: str, op: str, value: Any) -> State:  # noqa: A001
    out = st.copy()
    fn = _OPS[op]
    out.rows = [r for r in out.rows if fn(r.get(column), value)]
    out.notes.append(f"filter: {column} {op} {value!r} -> {len(out.rows)} row(s)")
    return out


def aggregate(st: State, store, *, column: str, how: str = "sum") -> State:
    out = st.copy()
    vals = [float(r[column]) for r in out.rows if r.get(column) not in (None, "")]
    out.scalar = {"sum": sum, "count": len, "max": max, "min": min}[how](
        vals if how != "count" else out.rows)
    out.notes.append(f"aggregate: {how}({column}) = {out.scalar:,.2f}")
    return out


def derive_measure(st: State, store, *, name: str, parts: List[float]) -> State:
    out = st.copy()
    out.scalar = float(sum(parts))
    out.notes.append(f"derive_measure: {name} = {' + '.join(f'{p:,.2f}' for p in parts)}"
                     f" = {out.scalar:,.2f}")
    return out


def probe_coverage(st: State, store, *, table: str, key: str, expect_key: str) -> State:
    """Ask whether the entity we are about to answer for is present at all.

    This is the primitive that produces a refusal instead of a number. Shop S06
    exists in the roster and appears in no transaction, so any per-shop answer
    about it is an answer about an empty set.
    """
    present = {r.get(key) for r in store[table]}
    out = st.copy()
    if expect_key not in present:
        out.abstained = True
        out.abstain_reason = (
            f"{expect_key} has no rows in {table}; the question cannot be "
            f"answered from the sources selected")
        out.notes.append(f"probe_coverage: {expect_key} ABSENT from {table} -> abstain")
    else:
        out.notes.append(f"probe_coverage: {expect_key} present in {table}")
    return out


def validate(st: State, store, *, check: str, **kw) -> State:
    out = st.copy()
    if check == "non_negative":
        ok = out.scalar is not None and out.scalar >= 0
    elif check == "rows_present":
        ok = len(out.rows) > 0
    elif check == "no_unmatched":
        ok = not any(r.get("_unmatched") for r in out.rows)
    else:
        raise KeyError(f"validate: unknown check {check!r}")
    out.notes.append(f"validate: {check} -> {'pass' if ok else 'FAIL'}")
    if not ok:
        out.abstained = True
        out.abstain_reason = f"deterministic check {check!r} failed"
    return out


def abstain(st: State, store, *, reason: str) -> State:
    out = st.copy()
    out.abstained = True
    out.abstain_reason = reason
    out.notes.append(f"abstain: {reason}")
    return out


PRIMITIVES: Dict[str, Callable[..., State]] = {
    "resolve_metric": resolve_metric,
    "select_source": select_source,
    "bind_entity": bind_entity,
    "discover_join": discover_join,
    "normalize_identifier": normalize_identifier,
    "deduplicate": deduplicate,
    "filter": filter,
    "aggregate": aggregate,
    "derive_measure": derive_measure,
    "probe_coverage": probe_coverage,
    "validate": validate,
    "abstain": abstain,
}

assert set(PRIMITIVES) == set(VOCABULARY), "PRIMITIVES and VOCABULARY disagree"


def step_digest(verb: str, args: Dict[str, Any]) -> str:
    """A stable digest of one step, used to compare paths and to detect drift."""
    blob = json.dumps({"v": verb, "a": args}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
