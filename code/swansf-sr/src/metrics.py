"""
The two metrics the solar flare forecasting community actually reports.

TSS  (True Skill Statistic, Hanssen-Kuipers)  = TP/(TP+FN) - FP/(FP+TN)
HSS2 (Heidke Skill Score, form 2)             = 2(TP*TN - FN*FP) /
                                                 ((TP+FN)(FN+TN) + (TP+FP)(FP+TN))

TSS is the headline because it is insensitive to the class-imbalance ratio,
which is the whole difficulty of SWAN-SF. HSS2 is reported alongside because
it is not, and a method that moves TSS while destroying HSS2 has usually just
learned to say "flare" more often.

Threshold selection is explicit and always fit on a split that the reported
number is not computed on.
"""

from __future__ import annotations

import numpy as np


def confusion(y, yhat):
    y = np.asarray(y).astype(int)
    yhat = np.asarray(yhat).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    return tp, fn, fp, tn


def tss(y, yhat):
    tp, fn, fp, tn = confusion(y, yhat)
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    return rec - far


def hss2(y, yhat):
    tp, fn, fp, tn = confusion(y, yhat)
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return 0.0 if denom == 0 else 2.0 * (tp * tn - fn * fp) / denom


def best_threshold(y, score, metric=tss, n_grid=200):
    """Threshold maximising `metric` on the data given. Fit on train/val only."""
    score = np.asarray(score, dtype="float64")
    score = np.where(np.isfinite(score), score, 0.0)
    lo, hi = np.percentile(score, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(score.min()), float(score.max() + 1e-9)
    grid = np.linspace(lo, hi, n_grid)
    vals = [metric(y, (score >= t).astype(int)) for t in grid]
    i = int(np.argmax(vals))
    return float(grid[i]), float(vals[i])


def evaluate(y, score, threshold):
    yhat = (np.asarray(score) >= threshold).astype(int)
    tp, fn, fp, tn = confusion(y, yhat)
    return {
        "tss": tss(y, yhat),
        "hss2": hss2(y, yhat),
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "far": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "threshold": float(threshold),
    }


def bootstrap_ci(y, score, threshold, metric=tss, n=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI over slices. Reported for every headline number."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    score = np.asarray(score)
    idx = np.arange(len(y))
    out = np.empty(n)
    for b in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        out[b] = metric(y[s], (score[s] >= threshold).astype(int))
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
