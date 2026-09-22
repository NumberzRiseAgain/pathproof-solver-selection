"""
Metrics: TSS, HSS2, threshold selection, bootstrap intervals.

These are the numbers the whole study reports, so they are tested against
confusion matrices computed by hand rather than against the implementation.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import (best_threshold, bootstrap_ci, confusion, evaluate,  # noqa: E402
                     hss2, tss)


def test_confusion_counts():
    y = np.array([1, 1, 0, 0, 0])
    yhat = np.array([1, 0, 1, 0, 0])
    tp, fn, fp, tn = confusion(y, yhat)
    assert (tp, fn, fp, tn) == (1, 1, 1, 2)


def test_tss_perfect_is_one():
    y = np.array([1, 1, 0, 0])
    assert tss(y, np.array([1, 1, 0, 0])) == 1.0


def test_tss_inverted_is_minus_one():
    y = np.array([1, 1, 0, 0])
    assert tss(y, np.array([0, 0, 1, 1])) == -1.0


def test_tss_all_negative_is_zero():
    """The degenerate forecast. TSS is designed to score it at zero, which is
    the whole reason it is the headline metric on a 2% positive rate: accuracy
    would score this at 98%."""
    y = np.array([1, 0, 0, 0, 0])
    assert tss(y, np.zeros(5, dtype=int)) == 0.0


def test_tss_all_positive_is_zero():
    y = np.array([1, 0, 0, 0, 0])
    assert tss(y, np.ones(5, dtype=int)) == 0.0


def test_tss_hand_computed():
    # 10 positives, 90 negatives; predict 8 of the positives and 9 negatives
    # wrongly.  recall 0.8, false alarm 0.1, TSS 0.7
    y = np.array([1] * 10 + [0] * 90)
    yhat = np.array([1] * 8 + [0] * 2 + [1] * 9 + [0] * 81)
    assert abs(tss(y, yhat) - 0.7) < 1e-12


def test_hss2_perfect_is_one():
    y = np.array([1, 1, 0, 0])
    assert abs(hss2(y, np.array([1, 1, 0, 0])) - 1.0) < 1e-12


def test_hss2_penalises_crying_wolf_where_tss_does_not():
    """
    The reason HSS2 is reported alongside TSS. Predicting positive for
    everything scores TSS 0 and HSS2 0, but a forecast that catches every
    positive at the cost of many false alarms keeps a high TSS while HSS2
    collapses. A method that moves TSS while destroying HSS2 has learned to
    say 'flare' more often, not to predict flares.
    """
    y = np.array([1] * 5 + [0] * 95)
    greedy = np.array([1] * 5 + [1] * 40 + [0] * 55)   # all 5 caught, 40 false
    assert tss(y, greedy) > 0.5
    assert hss2(y, greedy) < 0.25


def test_best_threshold_finds_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.0, 0.1, 0.2, 5.0, 5.1, 5.2])
    thr, val = best_threshold(y, score)
    assert abs(val - 1.0) < 1e-12
    assert 0.2 < thr <= 5.0


def test_best_threshold_ignores_non_finite():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.0, np.nan, 5.0, np.inf])
    thr, val = best_threshold(y, score)
    assert np.isfinite(thr)


def test_evaluate_reports_consistent_counts():
    y = np.array([1, 1, 0, 0])
    res = evaluate(y, np.array([9.0, 9.0, 1.0, 1.0]), threshold=5.0)
    assert res["tp"] == 2 and res["tn"] == 2 and res["fp"] == 0 and res["fn"] == 0
    assert abs(res["tss"] - 1.0) < 1e-12
    assert abs(res["recall"] - 1.0) < 1e-12


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = np.array([1] * 50 + [0] * 450)
    score = np.concatenate([rng.normal(2.0, 1.0, 50), rng.normal(0.0, 1.0, 450)])
    thr, _ = best_threshold(y, score)
    point = tss(y, (score >= thr).astype(int))
    lo, hi = bootstrap_ci(y, score, thr, n=400, seed=0)
    assert lo <= point <= hi
    assert hi > lo


def test_bootstrap_ci_is_deterministic_for_a_seed():
    y = np.array([1] * 20 + [0] * 80)
    score = np.linspace(0, 1, 100)
    a = bootstrap_ci(y, score, 0.5, n=200, seed=3)
    b = bootstrap_ci(y, score, 0.5, n=200, seed=3)
    assert a == b
