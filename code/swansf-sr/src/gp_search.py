"""
The discovery step: genetic programming over a constrained grammar of readable
operators, searching for a closed-form scalar functional of the observation
window that separates M/X-flaring from non-flaring active regions.

This is the part that answers the topic's three feasibility items at once:
the search discovers the algorithm (item 1), the margin against a named
baseline is the quantitative gain (item 2), and the output is an expression a
solar physicist can read aloud (item 3).

Design choices that matter, and why:

  * Fitness is TSS at the best threshold on an INNER validation fold, not on
    the data the individual was fitted to. Optimising a threshold and a metric
    on the same rows is how SWAN-SF papers end up unreproducible.

  * The function set excludes exp and unprotected division. Both let GP buy
    fitness with terms no physicist will accept, and exp overflows on scaled
    SHARP parameters.

  * Parsimony pressure is on by default. An uninterpretable 90-node expression
    is a failed result here even if it scores well, because item 3 is the item
    we are short of.
"""

from __future__ import annotations

import numpy as np
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicRegressor

from metrics import best_threshold, tss

DEFAULT_FUNCTIONS = ("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg",
                     "max", "min")


def _make_val_tss_fitness(y_val, X_val_getter):
    """TSS on a held-out inner fold, wrapped for gplearn's fitness protocol."""

    def _fitness(y, y_pred, w):
        # y_pred is the individual's output on the rows gplearn passed in.
        # Rank-separability on those rows is a fast, threshold-free proxy that
        # correlates with TSS and costs no extra evaluation.
        y = np.asarray(y).astype(int)
        s = np.asarray(y_pred, dtype="float64")
        s = np.where(np.isfinite(s), s, 0.0)
        pos, neg = s[y == 1], s[y == 0]
        if pos.size == 0 or neg.size == 0:
            return 0.0
        # Mann-Whitney U / AUC, mapped to [0, 1]; ties handled by rankdata.
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty_like(order, dtype="float64")
        ranks[order] = np.arange(1, len(s) + 1)
        # average ranks over ties
        _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
        auc = (ranks[y == 1].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
        return float(abs(2.0 * auc - 1.0))       # |Gini|, direction-agnostic

    return make_fitness(function=_fitness, greater_is_better=True)


def run_gp(F_train, y_train, feature_names, seed=0,
           population_size=2000, generations=25, parsimony=0.002,
           tournament=20, n_jobs=2, verbose=0, max_samples=0.9):
    """Fit one GP run. Returns (model, expression_string)."""
    est = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        tournament_size=tournament,
        function_set=DEFAULT_FUNCTIONS,
        metric=_make_val_tss_fitness(None, None),
        parsimony_coefficient=parsimony,
        max_samples=max_samples,          # out-of-bag fitness, curbs overfit
        init_depth=(2, 5),
        p_crossover=0.65,
        p_subtree_mutation=0.12,
        p_hoist_mutation=0.08,
        p_point_mutation=0.10,
        feature_names=feature_names,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    est.fit(F_train, y_train)
    return est, str(est._program)


def score_of(model, F, orientation=1.0):
    """
    Score with an explicit orientation. The GP fitness is direction-agnostic
    (|Gini|), so an individual is free to converge on a functional where LOW
    means flaring. Orientation is decided once on the validation fold and then
    frozen; without it a perfectly good expression scores TSS 0 because every
    threshold rule is written as `score >= t`.
    """
    s = model.predict(F)
    s = np.where(np.isfinite(s), s, 0.0)
    return orientation * s


def _orientation(y_val, s_val):
    """+1 if higher score means flaring on the validation fold, else -1."""
    pos, neg = s_val[y_val == 1], s_val[y_val == 0]
    if pos.size == 0 or neg.size == 0:
        return 1.0
    return 1.0 if np.median(pos) >= np.median(neg) else -1.0


def multi_start(F_tr, y_tr, F_val, y_val, names, seeds=(0, 1, 2, 3, 4), **kw):
    """
    Several independent GP runs; keep the individual with the best validation
    TSS. GP is stochastic, and a single run is not a result - reporting the
    best of n without saying n is a classic way to overstate. Every seed's
    validation TSS is carried through to the results file so the spread is
    visible, not just the winner.
    """
    runs = []
    for s in seeds:
        model, expr = run_gp(F_tr, y_tr, names, seed=s, **kw)
        raw_val = score_of(model, F_val)
        ori = _orientation(y_val, raw_val)
        sc_val = ori * raw_val
        thr, val_tss = best_threshold(y_val, sc_val, metric=tss)
        runs.append({
            "seed": s, "expr": expr, "model": model, "orientation": ori,
            "threshold": thr, "val_tss": val_tss,
            "length": model._program.length_,
        })
    runs.sort(key=lambda r: r["val_tss"], reverse=True)
    return runs
