"""
SWAN-SF I/O.

Two loaders with an identical output contract:

  load_partition(root, k)  -> (X, y, meta)
      X    : float64 [n_slices, n_timesteps, n_params]
      y    : int8    [n_slices]   1 = M/X class flare in prediction window, 0 = otherwise
      meta : DataFrame with one row per slice (filename, harpnum, label, timestamp)

  make_synthetic_partition(...) -> same contract, for pipeline validation only.

The real loader targets the SWAN-SF release on Harvard Dataverse
(doi:10.7910/DVN/EBCFKM): five chronological partitions, one CSV per MVTS
slice, 12-minute cadence, 51 columns of which we use the 24-parameter core
subset that the flare-prediction literature has converged on.

Nothing in this module invents data. make_synthetic_partition is clearly
named, is never used when a real partition is present, and every artifact it
produces is stamped synthetic=True so no synthetic number can be reported as
a result by accident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# The 24-parameter core subset. Order is fixed and load-bearing: discovered
# expressions refer to parameters by index, so this list is part of the result.
CORE_PARAMS = [
    "TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
    "USFLUX", "TOTFZ", "MEANPOT", "R_VALUE", "EPSZ", "SHRGT45",
    "MEANSHR", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZH",
    "TOTFY", "MEANJZD", "MEANALP", "TOTFX", "EPSY", "EPSX",
]

POSITIVE_CLASSES = ("X", "M")          # the >=M1.0 "strong flare" task
NEGATIVE_CLASSES = ("C", "B", "Q", "N")

# Bump when the parsed schema changes (new meta columns, different labelling,
# different windowing). It is part of the cache filename, so a stale cache is
# ignored rather than silently returned with the wrong columns - which is how
# a fix appears to have no effect.
CACHE_SCHEMA = 2

CADENCE_MIN = 12
DEFAULT_TOBS_HOURS = 12                # 12 h observation window -> 60 timesteps


@dataclass
class Partition:
    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    synthetic: bool
    params: list

    def __post_init__(self):
        assert self.X.ndim == 3, self.X.shape
        assert self.X.shape[0] == self.y.shape[0] == len(self.meta)
        assert self.X.shape[2] == len(self.params)

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def positive_rate(self):
        return float(self.y.mean())

    def describe(self):
        return (
            f"{'SYNTHETIC ' if self.synthetic else ''}partition: "
            f"n={self.n}  timesteps={self.X.shape[1]}  params={self.X.shape[2]}  "
            f"positives={int(self.y.sum())} ({100 * self.positive_rate:.2f}%)"
        )


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------

# SWAN-SF slice filenames lead with the flare class, e.g.
#   FQ_ar5742_s2015-07-05T08:36:00_e2015-07-05T20:24:00.csv
#   M1.0_ar1234_s2011-02-13T...csv
# and sit under partitionN/FL or partitionN/NF. The class is the leading token
# before the first underscore; its first character is the GOES class letter,
# with FQ meaning flare-quiet.
_CLASS_RE = re.compile(r"^(FQ|[XMCB])", re.IGNORECASE)


def _label_from_name(name: str) -> str:
    head = name.split("_", 1)[0].upper()
    m = _CLASS_RE.match(head)
    if not m:
        return "?"
    tok = m.group(1)
    return "Q" if tok == "FQ" else tok


def _sniff_sep(path):
    """SWAN-SF ships tab-separated files; sniff rather than assume."""
    with open(path, "r", errors="replace") as fh:
        line = fh.readline()
    return "\t" if line.count("\t") > line.count(",") else ","


def load_partition(root, k, params=None, tobs_hours=DEFAULT_TOBS_HOURS,
                   limit=None, cache=True, seed=0, verbose=True) -> Partition:
    """
    Load SWAN-SF partition k from `root` (the unzipped Dataverse tree).

    A partition holds tens of thousands of slices, so:

      * the separator is sniffed once, not per file, and pandas' C engine is
        used with an explicit sep (the Python engine sniffing per file turns a
        minutes-long load into an hours-long one);
      * `limit` takes a RANDOM sample, never the first N sorted paths. Sorted
        order puts every FL/ file before every NF/ file, so a sorted limit
        returns one class and reports 100% positives - a shakedown that looks
        like a triumph and means nothing;
      * the parsed partition is cached to an .npz beside the tree, so the
        second run costs seconds.
    """
    params = params or CORE_PARAMS
    n_steps = int(tobs_hours * 60 / CADENCE_MIN)

    pdir = Path(root) / f"partition{k}"
    if not pdir.is_dir():
        cands = [c for c in sorted(Path(root).glob(f"*artition*{k}*")) if c.is_dir()]
        if not cands:
            raise FileNotFoundError(
                f"no partition{k} DIRECTORY under {root}. If you see "
                f"partition{k}_instances.tar.gz, extract it first: "
                f"tar xzf partition{k}_instances.tar.gz"
            )
        pdir = cands[0]

    cache_f = (Path(root) /
               f".cache_v{CACHE_SCHEMA}_p{k}_t{tobs_hours}_l{limit or 0}_s{seed}.npz")
    if cache and cache_f.exists():
        z = np.load(cache_f, allow_pickle=True)
        if verbose:
            print(f"  (cached) {cache_f.name}")
        return Partition(X=z["X"], y=z["y"],
                         meta=pd.DataFrame(z["meta"].tolist()),
                         synthetic=False, params=list(params))

    files = sorted(pdir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV slices under {pdir}")
    if limit and limit < len(files):
        rng = np.random.default_rng(seed)
        files = [files[i] for i in sorted(rng.choice(len(files), limit, replace=False))]
    if verbose:
        print(f"  reading {len(files)} slices from {pdir.name} ...")

    sep = _sniff_sep(files[0])
    want = set(params)
    Xs, ys, rows, skipped = [], [], [], 0

    for n_done, f in enumerate(files, 1):
        try:
            df = pd.read_csv(f, sep=sep, engine="c",
                             usecols=lambda c: c in want)
        except Exception:
            skipped += 1
            continue
        if len(df.columns) < len(params):
            skipped += 1
            continue

        arr = df.reindex(columns=params).to_numpy(dtype="float64")
        if arr.shape[0] < n_steps:
            arr = np.vstack([np.full((n_steps - arr.shape[0], arr.shape[1]), np.nan), arr])
        arr = arr[-n_steps:]

        lab = _label_from_name(f.name)
        Xs.append(arr)
        ys.append(1 if lab in POSITIVE_CLASSES else 0)
        ar = re.search(r"ar(\d+)", f.name)
        rows.append({"file": f.name, "label": lab, "dir": f.parent.name,
                     "ar": int(ar.group(1)) if ar else -1,
                     "nan_frac": float(np.isnan(arr).mean())})
        if verbose and n_done % 5000 == 0:
            print(f"    {n_done}/{len(files)} ...", flush=True)

    if not Xs:
        raise RuntimeError(
            f"parsed 0 usable slices from {pdir} (skipped {skipped}). Check the "
            f"separator and that the CSVs carry {params[:3]} ..."
        )

    X = np.stack(Xs)
    y = np.array(ys, dtype="int8")
    meta = pd.DataFrame(rows)

    if verbose:
        counts = meta["label"].value_counts().to_dict()
        print(f"  label counts {counts}   skipped {skipped}")
        print(f"  active regions {meta['ar'].nunique()} distinct "
              f"({len(meta) / max(1, meta['ar'].nunique()):.1f} slices each)")
        # Missingness as a label proxy is a real leak channel: non-finite values
        # are replaced by 0.0 downstream, so if the classes differ in how much
        # data is absent, the zeros carry the label.
        mp = float(meta.loc[y == 1, "nan_frac"].mean())
        mn = float(meta.loc[y == 0, "nan_frac"].mean())
        flag = "  <-- CHECK: classes differ in missingness" if abs(mp - mn) > 0.02 else ""
        print(f"  NaN fraction  positives {mp:.4f}   negatives {mn:.4f}{flag}")
        # FL/NF directory is an independent check on the filename parse
        if "dir" in meta:
            bad = meta[(meta["dir"].str.upper() == "FL") & (y == 0)]
            if len(bad):
                print(f"  NOTE {len(bad)} slices sit under FL/ but parsed as "
                      f"non-positive (C and B class live there too - expected)")

    part = Partition(X=X, y=y, meta=meta, synthetic=False, params=list(params))
    if cache:
        np.savez_compressed(cache_f, X=X, y=y, meta=np.array(rows, dtype=object))
    return part


# --------------------------------------------------------------------------
# synthetic surrogate - pipeline validation only
# --------------------------------------------------------------------------

def make_synthetic_partition(n=1200, tobs_hours=DEFAULT_TOBS_HOURS,
                             positive_rate=0.045, params=None,
                             seed=0, signal=2.5,
                             signal_kind="compositional") -> Partition:
    """
    A surrogate with SWAN-SF's shape, scale, imbalance and missingness, built
    so the pipeline can be exercised end to end before the real partitions
    arrive. The planted signal is a known closed-form expression, which also
    gives the GP search a recoverable ground truth to be checked against.

    Planted rule (positives only): elevated TOTUSJH and TOTBSQ with a rising
    trend in TOTUSJH, modulated by R_VALUE. This mirrors the physics the
    literature reports as predictive but is NOT a claim about real data.
    """
    params = params or CORE_PARAMS
    rng = np.random.default_rng(seed)
    n_steps = int(tobs_hours * 60 / CADENCE_MIN)
    p = len(params)

    n_pos = max(1, int(round(n * positive_rate)))
    y = np.zeros(n, dtype="int8")
    y[rng.choice(n, size=n_pos, replace=False)] = 1

    # Log-normal-ish scales, as the SHARP parameters have. The scale vector is
    # a property of the instrument and the parameter, not of the partition, so
    # it is drawn from a FIXED seed shared by every synthetic partition. Real
    # SWAN-SF partitions differ by time period and flare distribution, not by
    # a factor of ten in units; drawing this per partition made the surrogate
    # unrealistically adversarial and masked a working pipeline as a failure.
    scale = np.exp(np.random.default_rng(20100501).normal(0.0, 1.6, size=p))
    base = rng.normal(0.0, 1.0, size=(n, 1, p)) * scale
    walk = np.cumsum(rng.normal(0.0, 0.12, size=(n, n_steps, p)), axis=1) * scale
    X = base + walk

    i_jh = params.index("TOTUSJH")
    i_bsq = params.index("TOTBSQ")
    i_rv = params.index("R_VALUE")
    t = np.linspace(0.0, 1.0, n_steps)[None, :]
    pos = y == 1
    n_p = int(pos.sum())

    if signal_kind == "univariate":
        # Each parameter is separately informative. A single terminal is the
        # optimum here, so this variant CANNOT test whether the search
        # composes - it only tests that the plumbing works.
        X[pos, :, i_jh] += signal * scale[i_jh] * (1.4 + 2.0 * t)
        X[pos, :, i_bsq] += signal * scale[i_bsq] * 1.1
        X[pos, :, i_rv] += signal * scale[i_rv] * 0.6

    elif signal_kind == "compositional":
        # No single feature separates the classes. The class is carried by the
        # PRODUCT of a trend in TOTUSJH and a level in R_VALUE, divided by the
        # volatility of TOTBSQ. Marginals are matched by construction: both
        # classes get the same distribution of each factor, and only their
        # combination differs. A search that cannot compose will score at
        # chance on this, which is exactly the property we need to verify
        # before trusting the pipeline on real data.
        a = rng.normal(0.0, 1.0, size=n_p)            # trend factor
        b = rng.normal(0.0, 1.0, size=n_p)            # level factor
        # force the product high for positives while leaving each marginal
        # standard normal in sign and spread
        flip = rng.random(n_p) < 0.5
        a = np.where(flip, -np.abs(a), np.abs(a))
        b = np.where(flip, -np.abs(b), np.abs(b))     # a*b > 0 for positives
        an = rng.normal(0.0, 1.0, size=n - n_p)
        bn = rng.normal(0.0, 1.0, size=n - n_p)
        bn = np.where(np.sign(an * bn) > 0, -bn, bn)  # a*b < 0 for negatives

        A = np.empty(n); B = np.empty(n)
        A[pos], B[pos] = a, b
        A[~pos], B[~pos] = an, bn

        X[:, :, i_jh] += signal * scale[i_jh] * A[:, None] * (2.0 * t)
        X[:, :, i_rv] += signal * scale[i_rv] * B[:, None]
        X[pos, :, i_bsq] *= 0.55                       # lower volatility

    elif signal_kind == "recursive":
        # A signal that is invisible to every static descriptor by
        # construction, and recoverable by a fold.
        #
        # Two zero-mean components u_t, v_t are added to TOTUSJH and R_VALUE.
        # For positives corr(u, v) = +rho within the window; for negatives it
        # is -rho. Each parameter's own mean, std, slope, delta, last and
        # maxabs therefore have the SAME distribution in both classes - only
        # the within-window covariance differs.
        #
        # No function of per-parameter summary statistics can separate these.
        # The fold s_t = s_{t-1} + TOTUSJH_t * R_VALUE_t separates them
        # exactly. This is the test that distinguishes discovering a procedure
        # from discovering a formula.
        rho = 0.85
        u = rng.normal(0.0, 1.0, size=(n, n_steps))
        w = rng.normal(0.0, 1.0, size=(n, n_steps))
        sign = np.where(y == 1, 1.0, -1.0)[:, None]
        v = sign * rho * u + np.sqrt(1.0 - rho ** 2) * w

        X[:, :, i_jh] += signal * scale[i_jh] * u
        X[:, :, i_rv] += signal * scale[i_rv] * v

    else:
        raise ValueError(signal_kind)

    # SWAN-SF has real missingness; keep the loader honest about NaNs
    mask = rng.random(X.shape) < 0.01
    X[mask] = np.nan

    meta = pd.DataFrame({
        "file": [f"synthetic_{i:05d}.csv" for i in range(n)],
        "label": np.where(y == 1, "M", "Q"),
        "path": "<synthetic>",
    })
    return Partition(X=X, y=y, meta=meta, synthetic=True, params=list(params))


PLANTED_RULE = (
    "positives carry elevated TOTUSJH (with rising slope), TOTBSQ and R_VALUE; "
    "a correct pipeline should recover an expression over those three."
)
