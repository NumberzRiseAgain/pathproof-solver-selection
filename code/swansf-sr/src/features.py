"""
Window -> descriptor inputs.

Two representations, both standard in the SWAN-SF literature so the baselines
and the GP see identical inputs:

  "vector"   last observed timestep per parameter (the closest-in-time vector
             representation; the usual strong-but-simple baseline)

  "mvts"     six temporal descriptors per parameter - last, mean, std, slope,
             delta (last minus first) and max-abs - which is the input the GP
             searches over. These are deliberately readable: every term in a
             discovered expression names a physical parameter and a plain
             temporal operation, which is what the interpretability
             requirement actually asks for.

Normalisation statistics are fit on TRAIN ONLY and applied to test. Getting
this wrong is the single most common source of inflated SWAN-SF numbers.
"""

from __future__ import annotations

import numpy as np

DESCRIPTORS = ("last", "mean", "std", "slope", "delta", "maxabs")


def _nan_slope(a):
    """Least-squares slope over time for each (slice, param), NaN-aware."""
    n, t, p = a.shape
    x = np.arange(t, dtype="float64")[None, :, None]
    m = ~np.isnan(a)
    cnt = m.sum(axis=1)
    safe = np.where(m, a, 0.0)
    xs = np.where(m, x, 0.0)
    sx = xs.sum(axis=1)
    sy = safe.sum(axis=1)
    sxx = (xs ** 2).sum(axis=1)
    sxy = (xs * safe).sum(axis=1)
    denom = cnt * sxx - sx ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (cnt * sxy - sx * sy) / denom
    return np.where(np.isfinite(slope), slope, 0.0)


def _first_last(a, last=True):
    n, t, p = a.shape
    out = np.full((n, p), np.nan)
    rng = range(t - 1, -1, -1) if last else range(t)
    for i in rng:
        fill = np.isnan(out) & ~np.isnan(a[:, i, :])
        out[fill] = a[:, i, :][fill]
        if not np.isnan(out).any():
            break
    return out


def extract(X, mode="mvts", params=None):
    """Return (F, names). F is [n_slices, n_features], finite, no NaNs."""
    params = params or [f"p{i}" for i in range(X.shape[2])]

    with np.errstate(all="ignore"):
        if mode == "vector":
            F = _first_last(X, last=True)
            names = [f"{p}_last" for p in params]
        elif mode == "mvts":
            last = _first_last(X, last=True)
            first = _first_last(X, last=False)
            mean = np.nanmean(X, axis=1)
            std = np.nanstd(X, axis=1)
            slope = _nan_slope(X)
            delta = last - first
            maxabs = np.nanmax(np.abs(X), axis=1)
            blocks = [last, mean, std, slope, delta, maxabs]
            F = np.concatenate(blocks, axis=1)
            names = [f"{p}_{d}" for d in DESCRIPTORS for p in params]
        else:
            raise ValueError(mode)

    F = np.where(np.isfinite(F), F, 0.0)
    return F, names


class TrainOnlyScaler:
    """Robust (median / IQR) scaling with statistics fit on train only."""

    def fit(self, F):
        self.med_ = np.median(F, axis=0)
        q1, q3 = np.percentile(F, [25, 75], axis=0)
        iqr = q3 - q1
        self.scale_ = np.where(iqr > 1e-12, iqr, 1.0)
        return self

    def transform(self, F):
        Z = (F - self.med_) / self.scale_
        return np.clip(Z, -20.0, 20.0)          # GP hates unbounded outliers

    def fit_transform(self, F):
        return self.fit(F).transform(F)
