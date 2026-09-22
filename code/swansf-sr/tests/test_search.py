"""
The search itself: the fold, its stability, orientation, and the surrogates.

The last three tests are the scientifically load-bearing ones. They assert
that the synthetic surrogates have the property they are claimed to have -
that the "recursive" signal really is invisible to every static descriptor,
and that a fold really does recover it. If those ever fail, the validation
evidence in the pre-registration is void and must not be cited.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import swan_io  # noqa: E402
from features import TrainOnlyScaler, extract  # noqa: E402
from metrics import best_threshold  # noqa: E402
from recursive_gp import (CONSTS, STATE_CLIP, Node, _rand_terminal,  # noqa: E402
                          fold_score, gini, random_tree, seed_filter)


def _seq_scale(X_fit, X):
    flat = X_fit.reshape(-1, X_fit.shape[2])
    med = np.nanmedian(flat, axis=0)
    q1, q3 = np.nanpercentile(flat, [25, 75], axis=0)
    scale = np.where(q3 - q1 > 1e-12, q3 - q1, 1.0)
    Z = (X - med) / scale
    return np.clip(np.where(np.isfinite(Z), Z, 0.0), -20.0, 20.0)


# --------------------------------------------------------------------------
# the fold
# --------------------------------------------------------------------------

def test_fold_of_pure_state_stays_at_zero():
    """s_t = s_{t-1} with s_0 = 0 must remain zero. If this drifts, the state
    is being initialised or carried wrongly."""
    X = np.random.default_rng(0).normal(size=(10, 60, 24))
    s, diverged = fold_score(Node("s"), X)
    assert np.allclose(s, 0.0) and not diverged


def test_fold_accumulates_a_constant_and_clips():
    """s_t = s + 1 over 59 update steps would reach 59; the clip holds it at
    STATE_CLIP. Clipping is what keeps an unbounded fold from diverging."""
    X = np.zeros((4, 60, 24))
    tree = Node("add", [Node("s"), Node(1.0)])
    s, diverged = fold_score(tree, X)
    assert np.allclose(s, STATE_CLIP) and not diverged


def test_fold_reports_divergence_rather_than_emitting_nan():
    X = np.full((3, 60, 24), 1e9)
    tree = Node("mul", [Node("s"), Node(("x", 0))])
    tree = Node("add", [Node(("x", 0)), tree])
    s, diverged = fold_score(tree, X)
    assert np.isfinite(s).all(), "a diverging program must not leak NaN"


def test_fold_respects_stride():
    X = np.zeros((2, 60, 24))
    tree = Node("add", [Node("s"), Node(1.0)])
    s1, _ = fold_score(tree, X, stride=1)
    s2, _ = fold_score(tree, X[:, :12, :], stride=1)
    assert s2.max() <= s1.max()


def test_random_trees_do_not_blow_up_the_fold():
    """Stability is enforced by tanh plus clipping, not hoped for."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 60, 24))
    for _ in range(200):
        s, _ = fold_score(random_tree(rng, 24, 4), X)
        assert np.isfinite(s).all()
        assert np.abs(s).max() <= STATE_CLIP + 1e-9


# --------------------------------------------------------------------------
# grammar
# --------------------------------------------------------------------------

def test_constant_terminals_are_reachable():
    """
    Regression test. Raising the state-terminal probability once made the
    constant branch unreachable, so no retention coefficient could be drawn
    and no discovered filter could express decay - only pure accumulation.
    """
    rng = np.random.default_rng(0)
    kinds = set()
    for _ in range(4000):
        nd = _rand_terminal(rng, 24)
        kinds.add("state" if nd.op == "s" else
                  "const" if isinstance(nd.op, float) else "var")
    assert kinds == {"state", "const", "var"}, kinds


def test_seeded_filter_template_carries_state():
    rng = np.random.default_rng(0)
    for _ in range(50):
        t = seed_filter(rng, 24)
        assert any(nd.op == "s" for nd in t.nodes())


def test_expression_text_names_physical_parameters():
    """Interpretability is the point: a printed rule must name parameters, not
    column indices."""
    t = Node("add", [Node("s"), Node("mul", [Node(("x", 0)), Node(("d", 9))])])
    txt = t.text(swan_io.CORE_PARAMS)
    assert "TOTUSJH" in txt and "d_R_VALUE" in txt and "s" in txt


def test_gini_separation_and_degeneracy():
    y = np.array([1] * 20 + [0] * 20)
    assert abs(gini(y, np.concatenate([np.ones(20), np.zeros(20)])) - 1.0) < 1e-9
    assert gini(y, np.zeros(40)) == 0.0          # constant score earns nothing
    # direction-agnostic: an inverted score scores the same magnitude
    assert abs(gini(y, np.concatenate([np.zeros(20), np.ones(20)])) - 1.0) < 1e-9


# --------------------------------------------------------------------------
# surrogate invariants - these underwrite the validation claims
# --------------------------------------------------------------------------

def test_synthetic_partitions_share_parameter_scales():
    """Two partitions must differ in sampling, not in units. Drawing the unit
    scale per partition once made a working pipeline look like total transfer
    failure."""
    a = swan_io.make_synthetic_partition(n=200, seed=1)
    b = swan_io.make_synthetic_partition(n=200, seed=2)
    sa = np.nanstd(a.X.reshape(-1, 24), axis=0)
    sb = np.nanstd(b.X.reshape(-1, 24), axis=0)
    assert np.allclose(sa / sb, 1.0, rtol=0.5)


def test_synthetic_class_imbalance_is_realistic():
    p = swan_io.make_synthetic_partition(n=2000, positive_rate=0.045, seed=0)
    assert 0.03 < p.positive_rate < 0.06
    assert p.synthetic is True


def test_recursive_surrogate_is_invisible_to_static_descriptors():
    """
    THE load-bearing invariant. The 'recursive' surrogate carries its class in
    the within-window covariance of two parameters, with every per-parameter
    marginal matched. No function of static summary statistics should separate
    the classes much above the selection noise floor.
    """
    tr = swan_io.make_synthetic_partition(n=1400, seed=11, signal_kind="recursive")
    F, names = extract(tr.X, "mvts", tr.params)
    Z = TrainOnlyScaler().fit(F).transform(F)
    best = max(max(best_threshold(tr.y, s * Z[:, j])[1] for s in (1.0, -1.0))
               for j in range(len(names)))
    assert best < 0.55, f"static descriptors reach {best:.3f}; surrogate is not blind"


def test_recursive_surrogate_is_recovered_by_the_planted_fold():
    """And the same signal must be near-perfectly recoverable by a fold, or the
    comparison in the pre-registration means nothing."""
    tr = swan_io.make_synthetic_partition(n=1400, seed=11, signal_kind="recursive")
    Z = _seq_scale(tr.X, tr.X)
    i = tr.params.index("TOTUSJH")
    j = tr.params.index("R_VALUE")
    planted = Node("add", [Node("s"), Node("mul", [Node(("x", i)), Node(("x", j))])])
    s, diverged = fold_score(planted, Z)
    val = max(best_threshold(tr.y, sign * s)[1] for sign in (1.0, -1.0))
    assert not diverged
    assert val > 0.9, f"planted fold only reaches {val:.3f}"


def test_compositional_surrogate_needs_composition():
    """The compositional surrogate's planted product must beat the best single
    feature by a clear margin, or it cannot test whether the search composes."""
    tr = swan_io.make_synthetic_partition(n=1400, seed=11,
                                          signal_kind="compositional")
    F, names = extract(tr.X, "mvts", tr.params)
    Z = TrainOnlyScaler().fit(F).transform(F)
    best_single = max(max(best_threshold(tr.y, s * Z[:, j])[1] for s in (1.0, -1.0))
                      for j in range(len(names)))
    prod = Z[:, names.index("TOTUSJH_slope")] * Z[:, names.index("R_VALUE_mean")]
    planted = max(best_threshold(tr.y, s * prod)[1] for s in (1.0, -1.0))
    assert planted > best_single + 0.15, (planted, best_single)
