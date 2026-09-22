"""Validation suite.

Each test pins one property the SPEED DIAL argument depends on. If a claim in
the volume rests on a behaviour, there is a test here that fails when that
behaviour stops holding. That is the standard the reviewer should hold us to,
and it is the standard we hold ourselves to.

    python3 -m pytest tests/ -q
"""
from __future__ import annotations

import pytest

from hazpath.memory import PathMemory, Record
from hazpath.path import Observables, Path, Step, reward
from hazpath.primitives import PRIMITIVES, VOCABULARY, load_store
from hazpath.promote import candidates, promote, subsequences
from hazpath.questions import HELD_OUT, SEEDS
from hazpath.solver import run, seed_paths, solve_by_transfer


@pytest.fixture(scope="module")
def store():
    return load_store()


@pytest.fixture(scope="module")
def paths():
    return seed_paths()


def _seeded_memory(store, paths) -> PathMemory:
    mem = PathMemory()
    for prob in SEEDS:
        concrete, obs, r = run(prob, paths[prob.key], store)
        mem.add(Record(prob.question, prob.fingerprint, concrete, obs, r,
                       prob.database))
    return mem


# ═══════════════════════════════════════════════════ the vocabulary is closed ═

def test_vocabulary_and_implementations_agree():
    """A verb without an implementation is a path that cannot be executed, and
    an implementation without a verb is a capability nobody declared."""
    assert set(PRIMITIVES) == set(VOCABULARY)


def test_every_seed_path_uses_only_declared_verbs(paths):
    for name, p in paths.items():
        for step in p.steps:
            assert step.verb in VOCABULARY, f"{name} uses undeclared {step.verb}"


# ════════════════════════════════════════════════════════════════ determinism ═

def test_execution_is_deterministic(store, paths):
    """The reward is computed from execution. A non-deterministic executor makes
    every reward in the memory meaningless, so this is checked, not assumed."""
    p = paths["ordered_value_fy24_26"]
    first, _, _ = run(SEEDS[2], p, store)
    second, _, _ = run(SEEDS[2], p, store)
    assert first.digest == second.digest
    st1, o1 = first.execute(store)
    st2, o2 = second.execute(store)
    assert o1.result == o2.result
    assert st1.notes == st2.notes


# ══════════════════════════════════════════════════════════════ the two traps ═

def test_skipping_deduplicate_returns_a_confident_wrong_number(store, paths):
    """The whole argument for path-level governance rests on this: the wrong
    path does not raise. It returns a plausible number that is 25% too high."""
    guarded, obs_ok, r_ok = run(SEEDS[2], paths["ordered_value_fy24_26"], store)
    naive, obs_bad, r_bad = run(SEEDS[4], paths["ordered_value_naive"], store)

    assert obs_ok.result == pytest.approx(537600.0)
    assert obs_bad.result == pytest.approx(672000.0)
    assert obs_bad.abstained is False          # nothing failed
    assert obs_bad.validation_failures == 0    # the check it ran still passed
    assert obs_ok.correct and not obs_bad.correct
    assert r_ok > 0 > r_bad


def test_skipping_normalization_would_split_one_state_into_three(store):
    """California is spelled three ways in the fixture. Without the guard the
    distinct count is 3; with it, 1."""
    raw = load_store()["shops"]
    assert len({r["state"] for r in raw}) == 3
    p = Path("norm", [Step("select_source", {"table": "shops"}),
                      Step("normalize_identifier", {"column": "state"})])
    st, _ = p.execute(store)
    assert len({r["state"] for r in st.rows}) == 1


# ═════════════════════════════════════════════════════════════════════ reward ═

def test_reward_orders_correct_above_abstain_above_wrong():
    """An honest refusal must score above a confident error, or the policy
    learns to guess."""
    good = Observables(correct=True, steps=5, latency_s=0.001)
    held = Observables(correct=True, steps=5, abstained=True)
    bad = Observables(correct=False, steps=5, latency_s=0.001)
    assert reward(good) > reward(held) > reward(bad)


def test_cheaper_path_wins_a_tie_between_two_correct_paths():
    short = Observables(correct=True, steps=4, latency_s=0.001)
    long_ = Observables(correct=True, steps=9, latency_s=0.001)
    assert reward(short) > reward(long_)


# ═════════════════════════════════════════════════════════════════════ memory ═

def test_memory_refuses_writes_after_the_freeze(store, paths):
    """The transfer claim is only meaningful if the memory could not have been
    written to while the held-out problems were scored."""
    mem = _seeded_memory(store, paths)
    rec = mem._records[0]
    mem.freeze()
    with pytest.raises(RuntimeError):
        mem.add(rec)
    with pytest.raises(RuntimeError):
        mem.register_composite("x", ("filter",))


def test_retrieval_never_returns_a_failed_path(store, paths):
    """The naive path is kept so it is not retried and can be reported. It must
    never bias selection upward."""
    mem = _seeded_memory(store, paths)
    assert any(r.reward < 0 for r in mem._records), "fixture should hold a failure"
    fp = SEEDS[2].fingerprint
    for _, rec in mem.retrieve(fp, k=10):
        assert rec.reward > 0


def test_retrieval_excludes_the_held_out_database(store, paths):
    mem = _seeded_memory(store, paths)
    mem.freeze()
    prob = HELD_OUT[0]
    got = mem.retrieve(prob.fingerprint, k=10,
                       exclude_databases={prob.database})
    assert got, "there should be usable priors from the training database"
    assert all(rec.database != prob.database for _, rec in got)


def test_policy_names_the_priors_that_produced_it(store, paths):
    """Interpretability of the policy itself: for any decision the system can
    say which prior problems supported it. A policy gradient cannot."""
    mem = _seeded_memory(store, paths)
    mem.freeze()
    ranked = mem.policy(HELD_OUT[0].fingerprint, k=5,
                        exclude_databases={HELD_OUT[0].database})
    assert ranked
    sig, weight, support = ranked[0]
    assert 0 < weight <= 1.0
    assert support and all(isinstance(q, str) for q in support)


# ═══════════════════════════════════════════════════════════════════ transfer ═

def test_held_out_problems_are_solved_from_a_retrieved_method(store, paths):
    mem = _seeded_memory(store, paths)
    mem.freeze()
    assert "hazmat_beta" not in mem.databases

    for prob in HELD_OUT:
        path, obs, r, evidence = solve_by_transfer(prob, mem, store)
        assert obs.correct, f"{prob.key} was answered wrongly"
        assert evidence, "a transfer with no evidence trail is not inspectable"
        assert path.provenance in ("adapted", "retrieved")


def test_the_headline_number_is_reproduced_exactly(store, paths):
    """639,450 is the figure in the deck. It is write-off plus disposal cost
    over FY24-26, and it is composed from two seed measures the memory holds."""
    mem = _seeded_memory(store, paths)
    mem.freeze()
    prob = HELD_OUT[0]
    _, obs, _, _ = solve_by_transfer(prob, mem, store)
    assert obs.result == pytest.approx(639450.0)


def test_the_coverage_gap_produces_a_refusal_not_a_zero(store, paths):
    """S06 is in the roster and in no transaction. Returning 0 would be a
    confident wrong answer; the correct output is a refusal that names why."""
    mem = _seeded_memory(store, paths)
    mem.freeze()
    prob = HELD_OUT[1]
    _, obs, _, _ = solve_by_transfer(prob, mem, store)
    assert obs.abstained
    assert "S06" in obs.abstain_reason
    assert obs.result is None
    assert obs.correct, "abstaining was the right answer here"


# ════════════════════════════════════════════════════════════════ promotion ═══

def test_subsequences_respect_the_declared_length_bounds():
    sig = tuple("abcdefgh")
    got = list(subsequences(sig))
    assert all(2 <= len(s) <= 5 for s in got)
    assert ("a", "b") in got and tuple("abcdef") not in got


def test_promotion_rejects_a_weak_candidate_on_a_small_fixture(store, paths):
    """The four-condition rule is strict on purpose. Five seed records do not
    justify minting a new building block, and the rule saying so is the
    behaviour we want -- an empty result here is a pass, not a gap."""
    mem = _seeded_memory(store, paths)
    got = candidates(mem._records, n_min=2)          # deliberately permissive n
    assert got == [], "a 5-record fixture must not mint composites"


def test_promotion_fires_when_the_signal_is_genuine(store, paths):
    """The rule must be capable of saying yes, or it proves nothing.

    Twenty synthetic records: those carrying the guard sequence succeed, those
    without it fail. That is a real effect, and the rule should find it.
    """
    mem = _seeded_memory(store, paths)
    good = next(r for r in mem._records if r.reward > 0
                and "deduplicate" in r.path.signature)
    bad = next(r for r in mem._records if r.reward <= 0)

    synthetic = []
    for i in range(10):
        synthetic.append(Record(f"good {i}", good.fingerprint, good.path,
                                good.observables, 1.0, "train"))
    for i in range(10):
        synthetic.append(Record(f"bad {i}", bad.fingerprint, bad.path,
                                bad.observables, -1.0, "train"))

    got = candidates(synthetic, n_min=8)
    assert got, "a clean 100-point separation must produce a candidate"
    for sig, n, lift in got:
        assert n >= 8                    # condition 1
        assert lift >= 0.10              # condition 2
        assert 2 <= len(sig) <= 5        # length bounds
        assert all(v in VOCABULARY for v in sig)


def test_promoted_composites_invent_no_new_verb(store, paths):
    mem = _seeded_memory(store, paths)
    good = next(r for r in mem._records if r.reward > 0
                and "deduplicate" in r.path.signature)
    bad = next(r for r in mem._records if r.reward <= 0)
    synthetic = ([Record(f"g{i}", good.fingerprint, good.path, good.observables,
                         1.0, "train") for i in range(10)]
                 + [Record(f"b{i}", bad.fingerprint, bad.path, bad.observables,
                           -1.0, "train") for i in range(10)])
    promote(mem, synthetic, n_min=8, limit=3)
    assert mem.composites
    for name, sig in mem.composites.items():
        assert all(v in VOCABULARY for v in sig), f"{name} invents a verb"


# ═════════════════════════════════════════════════════════════ fingerprints ═══

def test_fingerprint_similarity_is_bounded_and_reflexive():
    fp = SEEDS[0].fingerprint
    assert fp.similarity(fp) == pytest.approx(1.0)
    for other in (p.fingerprint for p in SEEDS + HELD_OUT):
        assert 0.0 <= fp.similarity(other) <= 1.0


def test_fingerprint_carries_no_literal_nouns():
    """If the fingerprint contained the question text or the table names,
    retrieval would be lookup and the transfer claim would be empty."""
    fp = SEEDS[0].fingerprint
    blob = repr(fp).lower()
    for noun in ("8010-01-555-1234", "primer", "disposal_dtid", "chromate"):
        assert noun not in blob
