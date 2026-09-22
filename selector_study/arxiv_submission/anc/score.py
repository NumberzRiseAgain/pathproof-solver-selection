"""Score the selector under leave-one-family-out, exactly as pre-registered.

Reads results/raw.jsonl. Writes results/scores.json and results/tables.md.
Fits only on the training partition of each fold. The test families are read once.
"""
import collections
import json
import hashlib
import numpy as np

from probe import FEATURE_NAMES

RAW = "results/raw.jsonl"
SEED = 0
N_TREES, DEPTH, LR = 200, 3, 0.05
CAL_FRAC = 0.25
MAHAL_Q = 0.99
PROBE_METHOD = "M5_chebyshev8"      # the only method whose t_total already carries the probe

METHOD_ORDER = ["M0_none", "M1_jacobi", "M2_ssor", "M3_ic0",
                "M4_neumann4", "M5_chebyshev8"]


def load():
    rows = [json.loads(l) for l in open(RAW)]
    ops = {}
    for r in rows:
        ops.setdefault(r["op_id"], {})[r["method"]] = r
    return rows, ops


def oracle_and_matrix(ops):
    """Per operator: feature vector, family, per-method total time, oracle."""
    recs = []
    for op_id, by_m in sorted(ops.items()):
        if len(by_m) < len(METHOD_ORDER):
            continue
        any_rec = next(iter(by_m.values()))
        times, conv = {}, {}
        for m in METHOD_ORDER:
            r = by_m[m]
            conv[m] = bool(r["converged"])
            times[m] = r["t_total"] if r["converged"] else np.inf
        t_probe = float(any_rec["t_probe"])
        if not any(conv.values()):
            recs.append(dict(op_id=op_id, family=any_rec["family"], usable=False,
                             x=None, times=times, conv=conv, oracle=None,
                             t_probe=t_probe))
            continue
        oracle = min((m for m in METHOD_ORDER if conv[m]), key=lambda m: times[m])
        x = np.array([any_rec["features"][k] for k in FEATURE_NAMES], float)
        recs.append(dict(op_id=op_id, family=any_rec["family"], usable=True,
                         x=x, times=times, conv=conv, oracle=oracle,
                         t_probe=t_probe))
    return recs


def outcome_of(rec):
    """Why a selected run ended. 'Did not converge' is four different events and they are
    reported apart: the declared wall-clock cap, the declared iteration cap, a non-positive
    curvature breakdown, and a preconditioner setup that could not be built."""
    if rec.get("setup_failed"):
        return "setup_failed"
    if rec.get("converged"):
        return "converged"
    if rec.get("breakdown"):
        return "breakdown"
    if rec.get("timeout"):
        return "wall_cap_30s"
    return "iteration_cap_20000"


def fit_predictors(train, seed=SEED, shuffle=False):
    from sklearn.ensemble import GradientBoostingRegressor
    rng = np.random.default_rng(seed)
    X = np.vstack([t["x"] for t in train])
    models = {}
    for m in METHOD_ORDER:
        y, keep = [], []
        for i, t in enumerate(train):
            if np.isfinite(t["times"][m]):
                y.append(np.log10(t["times"][m])); keep.append(i)
        if len(keep) < 8:
            models[m] = None
            continue
        yv = np.array(y)
        if shuffle:
            yv = rng.permutation(yv)
        g = GradientBoostingRegressor(n_estimators=N_TREES, max_depth=DEPTH,
                                      learning_rate=LR, random_state=seed)
        g.fit(X[keep], yv)
        models[m] = g
    return models


def choose(models, x):
    best, best_t = None, np.inf
    for m in METHOD_ORDER:
        if models[m] is None:
            continue
        p = float(models[m].predict(x.reshape(1, -1))[0])
        if p < best_t:
            best, best_t = m, p
    return best


def regret(t_choice, t_oracle):
    if not np.isfinite(t_choice):
        return np.inf
    return (t_choice - t_oracle) / t_oracle


def deployment_time(rec, m):
    """End-to-end time an engineer pays when the SELECTOR picks m on this operator.

    The selector consumes the probe on every operator, whatever it goes on to choose, so the
    probe is charged once to every choice. M5's t_total already carries it and is not charged
    twice. The oracle is clairvoyant by definition and is never charged: it needs no probe.
    Post-hoc, recorded in addendum 2; the pre-registered score is unchanged and reported beside it.
    """
    t = rec["times"][m]
    if not np.isfinite(t):
        return t
    return t if m == PROBE_METHOD else t + rec["t_probe"]


def describe(v):
    """Both readings of the same regret vector, with the denominators shown.

    CONDITIONAL statistics are taken over the selections that converged, which is what a
    percentile over a vector containing infinities silently computes. UNCONDITIONAL statistics
    keep the non-converging selections at infinity and are taken by nearest rank, so a
    population with more than ten per cent failures honestly reports an infinite p90 instead of
    a finite number that excluded them. Both are reported everywhere; neither is the default.
    """
    v = np.array(v, float)
    fin = v[np.isfinite(v)]
    n = int(v.size)

    def nearest_rank(q):
        if n == 0:
            return None
        x = np.sort(v)                       # +inf sorts last
        val = x[min(n - 1, max(0, int(np.ceil(q * n)) - 1))]
        return "inf" if not np.isfinite(val) else float(val)

    return dict(
        n=n,
        n_converged=int(fin.size),
        n_not_converged=int(n - fin.size),
        median_conditional=float(np.median(fin)) if fin.size else None,
        p90_conditional=float(np.quantile(fin, 0.9)) if fin.size else None,
        median_unconditional=nearest_rank(0.50),
        p90_unconditional=nearest_rank(0.90),
    )


def mahalanobis_threshold(cal_X):
    mu = cal_X.mean(axis=0)
    C = np.cov(cal_X.T) + 1e-6 * np.eye(cal_X.shape[1])
    Ci = np.linalg.inv(C)
    d = np.array([np.sqrt((v - mu) @ Ci @ (v - mu)) for v in cal_X])
    return mu, Ci, float(np.quantile(d, MAHAL_Q))


def environment():
    """Every library whose floating point can move a near-tie, recorded with the scores.

    The median and the winner accuracy of this study are decided by argmin over predicted
    times that are often within noise of each other, so they move with the numeric stack.
    The version table in the paper is generated from this block.
    """
    import platform
    import numpy, scipy, sklearn
    return dict(python=platform.python_version(), platform=platform.platform(),
                numpy=numpy.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__)


def probe_economics(ops):
    """Probe cost against two denominators, because the paper defines T end to end.

    'end_to_end' compares the probe against the fastest converging T, which is the quantity
    the score of section 7 uses. 'solve_only' compares it against solve time with setup and
    probe removed. Both are reported; the first is the one the regret definition implies.
    """
    out = {}
    for key, name in (("t_total", "end_to_end"), ("t_solve", "solve_only")):
        n_over, worst, ratios = 0, 0.0, []
        for op_id, by_m in ops.items():
            conv = [r for r in by_m.values() if r["converged"]]
            if not conv:
                continue
            base = min(r[key] for r in conv)
            probe = float(next(iter(by_m.values()))["t_probe"])
            if base <= 0:
                continue
            ratios.append(probe / base)
            if probe > base:
                n_over += 1
                worst = max(worst, probe / base)
        out[name] = dict(n=len(ratios), n_probe_exceeds_solve=n_over,
                         max_ratio=round(worst, 2),
                         median_ratio=round(float(np.median(ratios)), 4) if ratios else None)
    pr = [float(next(iter(b.values()))["t_probe"]) for b in ops.values()]
    out["probe_seconds"] = dict(min=round(min(pr), 4), max=round(max(pr), 4),
                                median=round(float(np.median(pr)), 4), n=len(pr))
    return out


def miss_statistics(regs, hits):
    """The spread behind the accuracy column, which one proportion hides.

    Reported as counts with their denominators: exact picks, misses, how many misses never
    converged, and over the converging misses how many cost under twenty per cent and what
    the median costs. Section 8.3 of the paper is generated from this block.
    """
    r = np.array(regs, float)
    h = np.array(hits, float).astype(bool)
    miss = r[~h]
    fin = miss[np.isfinite(miss)]
    return dict(n=int(r.size), n_exact=int(h.sum()), n_miss=int((~h).sum()),
                n_miss_not_converged=int((~np.isfinite(miss)).sum()),
                n_miss_converged=int(fin.size),
                n_miss_under_0p20=int((fin < 0.20).sum()) if fin.size else 0,
                median_miss_converged=round(float(np.median(fin)), 4) if fin.size else None,
                frac_wrong=round(float((~h).mean()), 4) if r.size else None)


def run():
    rows, ops = load()
    recs = oracle_and_matrix(ops)
    usable = [r for r in recs if r["usable"]]
    families = sorted({r["family"] for r in recs})

    out = dict(environment=environment(),
               probe_economics=probe_economics(ops),
               n_operators=len(recs), n_usable=len(usable),
               n_no_method_converged=len(recs) - len(usable),
               folds={}, families=families)

    # how often the recurrence residual claimed a convergence it did not have
    lied = sum(1 for r in rows if r.get("rec_claimed_early"))
    out["rec_claimed_early_count"] = lied
    out["n_runs"] = len(rows)
    out["n_converged_runs"] = sum(1 for r in rows if r["converged"])
    out["n_timeout_runs"] = sum(1 for r in rows if r.get("timeout"))
    out["n_setup_failed"] = sum(1 for r in rows if r.get("setup_failed"))

    all_reg, all_reg_kappa, all_reg_null, all_hit = [], [], [], []
    all_reg_dep = []
    pooled_outcomes = collections.Counter()
    all_reg_cov, coverage = [], []
    for f in families:
        test = [r for r in usable if r["family"] == f]
        train = [r for r in usable if r["family"] != f]
        if not test or len(train) < 12:
            continue
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(train))
        ncal = max(4, int(CAL_FRAC * len(train)))
        cal = [train[i] for i in idx[:ncal]]
        fit = [train[i] for i in idx[ncal:]]

        models = fit_predictors(fit)
        null_models = fit_predictors(fit, shuffle=True)

        # condition-number-only baseline: same learner, one feature
        kappa_i = FEATURE_NAMES.index("log_kappa_ritz")
        fit_k = [dict(t, x=t["x"][[kappa_i]]) for t in fit]
        models_k = fit_predictors(fit_k)

        mu, Ci, thr = mahalanobis_threshold(np.vstack([c["x"] for c in cal]))
        # training fallback: best median rank on the fit partition
        ranks = {m: [] for m in METHOD_ORDER}
        for t in fit:
            order = sorted(METHOD_ORDER, key=lambda m: t["times"][m])
            for rk, m in enumerate(order):
                ranks[m].append(rk)
        fallback = min(METHOD_ORDER, key=lambda m: np.median(ranks[m]))

        regs, regs_k, regs_n, hits, covered, regs_cov = [], [], [], [], [], []
        regs_dep = []
        fold_outcomes = collections.Counter()
        for t in test:
            t_or = t["times"][t["oracle"]]
            m_hat = choose(models, t["x"])
            regs.append(regret(t["times"][m_hat], t_or))
            regs_dep.append(regret(deployment_time(t, m_hat), t_or))
            fold_outcomes[outcome_of(ops[t["op_id"]][m_hat])] += 1
            hits.append(1.0 if m_hat == t["oracle"] else 0.0)
            regs_k.append(regret(t["times"][choose(models_k, t["x"][[kappa_i]])], t_or))
            regs_n.append(regret(t["times"][choose(null_models, t["x"])], t_or))
            d = float(np.sqrt((t["x"] - mu) @ Ci @ (t["x"] - mu)))
            if d <= thr:
                covered.append(1.0); regs_cov.append(regs[-1])
            else:
                covered.append(0.0)
                regs_cov.append(regret(t["times"][fallback], t_or))

        out["folds"][f] = dict(
            n_test=len(test), fallback=fallback,
            misses=miss_statistics(regs, hits),
            regret=describe(regs), regret_deployment=describe(regs_dep),
            regret_kappa_only=describe(regs_k),
            regret_null=describe(regs_n),
            winner_accuracy=float(np.mean(hits)),
            coverage=float(np.mean(covered)),
            regret_with_abstention=describe(regs_cov),
            selected_outcomes=fold_outcomes,
            oracle_mix={m: sum(1 for t in test if t["oracle"] == m)
                        for m in METHOD_ORDER},
        )
        pooled_outcomes.update(fold_outcomes)
        fold_outcomes = dict(fold_outcomes)
        all_reg += regs
        all_reg_dep += regs_dep
        all_reg_kappa += regs_k
        all_reg_null += regs_n
        all_hit += hits
        all_reg_cov += regs_cov
        coverage += covered

    out["pooled"] = dict(
        misses=miss_statistics(all_reg, all_hit),
        regret=describe(all_reg), regret_deployment=describe(all_reg_dep),
        regret_kappa_only=describe(all_reg_kappa),
        regret_null=describe(all_reg_null),
        winner_accuracy=float(np.mean(all_hit)) if all_hit else None,
        coverage=float(np.mean(coverage)) if coverage else None,
        regret_with_abstention=describe(all_reg_cov),
        selected_outcomes=pooled_outcomes,
    )
    # the registered criteria are scored on the registered regret, read conditionally, which is
    # how the bars were written. The unconditional reading is reported beside them.
    crit = out["pooled"]["regret"]
    out["criteria"] = dict(
        primary_median_le_0p05=(crit["median_conditional"] is not None
                                and crit["median_conditional"] <= 0.05),
        primary_p90_le_0p20=(crit["p90_conditional"] is not None
                             and crit["p90_conditional"] <= 0.20),
        secondary_winner_acc_ge_0p80=(out["pooled"]["winner_accuracy"] or 0) >= 0.80,
    )
    with open("results/scores.json", "w") as fh:
        json.dump(out, fh, indent=2)
    out["pooled"]["selected_outcomes"] = dict(pooled_outcomes)
    with open("results/scores.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: out[k] for k in
                      ("n_operators", "n_usable", "n_no_method_converged",
                       "rec_claimed_early_count", "n_runs", "n_converged_runs",
                       "n_timeout_runs", "pooled", "criteria")}, indent=2))


if __name__ == "__main__":
    run()
