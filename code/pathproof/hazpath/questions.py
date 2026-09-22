"""The worked problems, their fingerprints, and the reference answers.

Six problems over the synthetic HazMat store. Four are authored to seed the
memory; two are held out and are the ones that matter -- they are solved by
retrieving a METHOD from the seeds and changing only the nouns.

Two of the six are traps, and they are the reason this fixture is worth using:

  * `ordered_value_fy24_26` is inflated by exactly the three duplicated
    supply_id rows unless `deduplicate` runs. The wrong answer is $672,000
    against a true $537,600, and nothing raises.
  * `shops_by_state` returns three groups instead of one unless
    `normalize_identifier` runs, because the fixture spells California three
    ways.

A path that skips the guard primitive produces a confident wrong number. That is
the failure the reward is measuring, and it is why correctness has to be scored
against execution and not against whether the SQL looked reasonable.
"""
from __future__ import annotations

from typing import Any, List, NamedTuple

from .memory import Fingerprint

NSN = "8010-01-555-1234"


class Problem(NamedTuple):
    key: str
    question: str
    fingerprint: Fingerprint
    reference: Any
    database: str
    note: str = ""


# The four seeds. These populate memory.
SEEDS: List[Problem] = [
    Problem(
        key="writeoff_value_fy24_26",
        question="What did we write off on the chromate primer across FY24 to FY26?",
        fingerprint=Fingerprint(
            metric_kind="money", entity_classes=("material",),
            source_classes=("disposal",), temporal_grain="fy",
            needs_join=False, needs_normalization=False, needs_dedup=False),
        reference=403200.0,
        database="hazmat_alpha"),
    Problem(
        key="disposal_cost_fy24_26",
        question="What did disposal of that primer cost us over the same period?",
        fingerprint=Fingerprint(
            metric_kind="money", entity_classes=("material",),
            source_classes=("disposal",), temporal_grain="fy",
            needs_join=False, needs_normalization=False, needs_dedup=False),
        reference=236250.0,
        database="hazmat_alpha"),
    Problem(
        key="ordered_value_fy24_26",
        question="What did we spend ordering the primer across FY24 to FY26?",
        fingerprint=Fingerprint(
            metric_kind="money", entity_classes=("material", "shop"),
            source_classes=("supply",), temporal_grain="fy",
            needs_join=False, needs_normalization=False, needs_dedup=True),
        reference=537600.0,
        database="hazmat_alpha",
        note="inflated to 672000.0 if deduplicate is skipped"),
    Problem(
        key="shops_by_state",
        question="How many distinct states do the shops sit in?",
        fingerprint=Fingerprint(
            metric_kind="count", entity_classes=("shop",),
            source_classes=("roster",), temporal_grain="none",
            needs_join=False, needs_normalization=True, needs_dedup=False),
        reference=1.0,
        database="hazmat_alpha",
        note="returns 3 if normalize_identifier is skipped"),
    Problem(
        key="ordered_value_naive",
        question=("What did we spend ordering the primer across FY24 to FY26? "
                  "(asked again, answered without the duplicate guard)"),
        fingerprint=Fingerprint(
            metric_kind="money", entity_classes=("material", "shop"),
            source_classes=("supply",), temporal_grain="fy",
            needs_join=False, needs_normalization=False, needs_dedup=True),
        reference=537600.0,
        database="hazmat_alpha",
        note="the same question solved by a path that skips deduplicate"),
]

# The two held out. Never seen when memory is built.
HELD_OUT: List[Problem] = [
    Problem(
        key="overorder_3yr_usd",
        question=("Across FY24 to FY26, what did the over-ordering of this primer "
                  "cost us in write-off and disposal together?"),
        fingerprint=Fingerprint(
            metric_kind="money", entity_classes=("material",),
            source_classes=("disposal",), temporal_grain="fy",
            needs_join=False, needs_normalization=False, needs_dedup=False),
        reference=639450.0,
        database="hazmat_beta",
        note="the deck's headline number; composed from two seed measures"),
    Problem(
        key="fuel_cell_orders",
        question="How much primer did the Fuel Cell shop order in FY26?",
        fingerprint=Fingerprint(
            metric_kind="money", entity_classes=("shop", "material"),
            source_classes=("supply", "roster"), temporal_grain="fy",
            needs_join=True, needs_normalization=False, needs_dedup=True),
        reference="ABSTAIN",
        database="hazmat_beta",
        note="S06 exists in the roster and appears in no transaction"),
]


def all_problems() -> List[Problem]:
    return SEEDS + HELD_OUT
