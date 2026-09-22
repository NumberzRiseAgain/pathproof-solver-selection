"""
Recursive-rule discovery, with the static formula search as the baseline.

The comparison this script exists to make:

  static GP    discovers f(last, mean, std, slope, delta, maxabs)   -> a FORMULA
  recursive GP discovers g(s_{t-1}, x_t, dx_t) folded over the window -> a PROCEDURE

Same data, same split, same metric, same threshold discipline. On the
"recursive" surrogate the static search should score at chance by
construction, and the recursive search should recover the planted rule. That
is the validation. On real SWAN-SF it is an open question, which is the point.

Usage
  python run_recursive.py --synthetic --signal recursive
  python run_recursive.py --data /path/to/SWAN-SF --train 1 --test 2
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).parent))

import swan_io                                              # noqa: E402
from features import TrainOnlyScaler, extract               # noqa: E402
from gp_search import multi_start, score_of                 # noqa: E402
from metrics import best_threshold, bootstrap_ci, evaluate, hss2, tss  # noqa: E402
from recursive_gp import describe, evolve, fold_score       # noqa: E402


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=Path(__file__).parent.parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


class SeqScaler:
    """Median/IQR per parameter over (sample, timestep). Train rows only."""

    def fit(self, X):
        flat = X.reshape(-1, X.shape[2])
        self.med_ = np.nanmedian(flat, axis=0)
        q1, q3 = np.nanpercentile(flat, [25, 75], axis=0)
        iqr = q3 - q1
        self.scale_ = np.where(iqr > 1e-12, iqr, 1.0)
        return self

    def transform(self, X):
        Z = (X - self.med_) / self.scale_
        Z = np.where(np.isfinite(Z), Z, 0.0)
        return np.clip(Z, -20.0, 20.0)


def _inner_split(part, val_frac=0.30, seed=7):
    """
    Split train into fit/validation by ACTIVE REGION, not by slice.

    SWAN-SF slices are sliding windows over the same active region, so
    consecutive slices are near-duplicates. A random slice-level split puts
    near-identical rows on both sides, inflating the validation score and
    therefore the threshold chosen from it. Grouping by the ar id in the
    filename removes that. Falls back to a stratified slice split only for the
    synthetic surrogate, which has no regions.
    """
    idx = np.arange(len(part.y))
    if "ar" in part.meta and part.meta["ar"].nunique() > 5:
        groups = part.meta["ar"].to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        a, b = next(gss.split(idx, part.y, groups))
        if part.y[a].sum() >= 5 and part.y[b].sum() >= 5:
            print(f"  inner split: GROUPED by active region "
                  f"({len(np.unique(groups[a]))} fit / "
                  f"{len(np.unique(groups[b]))} val regions, no region on both sides)")
            return idx[a], idx[b]
        print("  inner split: group split left too few positives on a side, "
              "FALLING BACK to a stratified slice split (validation will be "
              "optimistic - overlapping slices land on both sides)")
    elif not part.synthetic:
        print("  inner split: WARNING no active-region ids in meta, falling back "
              "to a stratified slice split. Delete the .npz caches and reload.")
    return train_test_split(idx, test_size=val_frac, stratify=part.y,
                            random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--signal", default="recursive",
                    choices=["recursive", "compositional", "univariate"])
    ap.add_argument("--train", type=int, default=1)
    ap.add_argument("--test", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--population", type=int, default=600)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="../results/recursive.json")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if not args.synthetic and not args.data:
        ap.error("give --data <SWAN-SF root> or --synthetic")

    if args.synthetic:
        tr = swan_io.make_synthetic_partition(n=1400, seed=11, signal_kind=args.signal)
        te = swan_io.make_synthetic_partition(n=1400, seed=22, signal_kind=args.signal)
    else:
        tr = swan_io.load_partition(args.data, args.train, limit=args.limit)
        te = swan_io.load_partition(args.data, args.test, limit=args.limit)

    print(f"train {tr.describe()}")
    print(f"test  {te.describe()}")
    if tr.synthetic:
        print(f"\n*** SYNTHETIC ({args.signal} signal) - validation only, "
              f"no number below is a scientific result. ***\n")

    idx_tr, idx_val = _inner_split(tr)
    y_tr, y_val, y_te = tr.y[idx_tr], tr.y[idx_val], te.y

    # ---- baseline: static formula search --------------------------------
    F_tr_raw, names = extract(tr.X, mode="mvts", params=tr.params)
    F_te_raw, _ = extract(te.X, mode="mvts", params=te.params)
    sc = TrainOnlyScaler().fit(F_tr_raw[idx_tr])
    runs = multi_start(sc.transform(F_tr_raw[idx_tr]), y_tr,
                       sc.transform(F_tr_raw[idx_val]), y_val, names,
                       seeds=tuple(range(args.seeds)),
                       generations=args.generations, population_size=2000)
    b = runs[0]
    s_te = score_of(b["model"], sc.transform(F_te_raw), b["orientation"])
    static = evaluate(y_te, s_te, b["threshold"])
    lo, hi = bootstrap_ci(y_te, s_te, b["threshold"], metric=tss, seed=0)
    static.update({"tss_ci95": [lo, hi], "expression": b["expr"],
                   "size": int(b["length"]), "val_tss": b["val_tss"],
                   "kind": "static formula"})

    # ---- reference floor: the best SINGLE static descriptor --------------
    # Both searches must beat this to have earned anything. A discovered rule
    # that ties it is a rediscovery of one feature dressed as a procedure -
    # e.g. the fold s_t = max(s, ABSNJZH) is exactly the ABSNJZH_maxabs
    # terminal the static search already had. Selection is on validation only.
    Ztr_s, Zval_s, Zte_s = (sc.transform(F_tr_raw[idx_tr]),
                            sc.transform(F_tr_raw[idx_val]),
                            sc.transform(F_te_raw))
    best_single = None
    for j, nm in enumerate(names):
        for sign in (1.0, -1.0):
            th, v = best_threshold(y_val, sign * Zval_s[:, j], metric=tss)
            if best_single is None or v > best_single["val_tss"]:
                best_single = {"feature": nm, "sign": sign, "threshold": th,
                               "val_tss": v, "index": j}
    bs_te = best_single["sign"] * Zte_s[:, best_single["index"]]
    single = evaluate(y_te, bs_te, best_single["threshold"])
    slo, shi = bootstrap_ci(y_te, bs_te, best_single["threshold"], metric=tss, seed=0)
    single.update({"tss_ci95": [slo, shi], "feature": best_single["feature"],
                   "sign": best_single["sign"], "val_tss": best_single["val_tss"],
                   "kind": "best single static descriptor",
                   "n_candidates": len(names),
                   "note": "selected on validation over all descriptors; the "
                           "selection itself is optimistic, which is the point - "
                           "a search must beat a cherry-picked single feature"})
    print(f"  reference floor: best single descriptor {best_single['feature']} "
          f"(sign {best_single['sign']:+.0f})  val {best_single['val_tss']:+.4f}")

    # ---- recursive rule search ------------------------------------------
    seq = SeqScaler().fit(tr.X[idx_tr])
    Z_tr, Z_val, Z_te = seq.transform(tr.X[idx_tr]), seq.transform(tr.X[idx_val]), seq.transform(te.X)

    rec_runs = []
    for s in range(args.seeds):
        tree, log = evolve(Z_tr, y_tr, tr.params, seed=s,
                           population=args.population,
                           generations=args.generations, stride=args.stride)
        sv, _ = fold_score(tree, Z_val, args.stride)
        ori = 1.0 if np.median(sv[y_val == 1]) >= np.median(sv[y_val == 0]) else -1.0
        thr, vt = best_threshold(y_val, ori * sv, metric=tss)
        rec_runs.append({"seed": s, "tree": tree, "orientation": ori,
                         "threshold": thr, "val_tss": vt,
                         **describe(tree, tr.params)})
        print(f"  seed {s}: val TSS {vt:+.4f}  size {tree.size()}  "
              f"{describe(tree, tr.params)['rule'][:90]}")

    rec_runs.sort(key=lambda r: r["val_tss"], reverse=True)
    rb = rec_runs[0]
    st, _ = fold_score(rb["tree"], Z_te, args.stride)
    st = rb["orientation"] * st
    rec = evaluate(y_te, st, rb["threshold"])
    rlo, rhi = bootstrap_ci(y_te, st, rb["threshold"], metric=tss, seed=0)
    rec.update({"tss_ci95": [rlo, rhi], "rule": rb["rule"], "size": rb["size"],
                "depth": rb["depth"], "terminals": rb["terminals"],
                "val_tss": rb["val_tss"], "kind": "recursive rule",
                "n_seeds": args.seeds,
                "all_seed_val_tss": [r["val_tss"] for r in rec_runs],
                "all_seed_rules": [r["rule"] for r in rec_runs],
                "all_seed_terminals": [r["terminals"] for r in rec_runs]})

    out = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": _git_rev(), "python": platform.python_version(),
        "synthetic": bool(tr.synthetic),
        "synthetic_signal": args.signal if tr.synthetic else None,
        "tag": args.tag,
        "protocol": {"train": "synthetic" if tr.synthetic else args.train,
                     "test": "synthetic" if te.synthetic else args.test,
                     "stride": args.stride, "seeds": args.seeds,
                     "threshold_fit_on": "inner validation split of train"},
        "best_single_descriptor": single,
        "static_formula": static,
        "recursive_rule": rec,
        "margin_recursive_minus_static": rec["tss"] - static["tss"],
        "margin_recursive_minus_single": rec["tss"] - single["tss"],
        "margin_static_minus_single": static["tss"] - single["tss"],
    }
    outp = Path(__file__).parent / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))

    print("\n" + "=" * 74)
    print(f"SINGLE FEATURE    TSS {single['tss']:+.4f}  [{slo:+.4f}, {shi:+.4f}]  "
          f"{single['feature']}")
    print(f"STATIC FORMULA    TSS {static['tss']:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"size {static['size']}")
    print(f"                  {static['expression'][:70]}")
    print(f"RECURSIVE RULE    TSS {rec['tss']:+.4f}  [{rlo:+.4f}, {rhi:+.4f}]  "
          f"size {rec['size']}")
    print(f"                  {rec['rule'][:70]}")
    print(f"margin  recursive - static {out['margin_recursive_minus_static']:+.4f}"
          f"   recursive - single feature {out['margin_recursive_minus_single']:+.4f}"
          f"   static - single feature {out['margin_static_minus_single']:+.4f}")
    print("=" * 74)
    print(f"\nterminals used: {', '.join(rec['terminals'])}")
    print(f"written to {outp}")


if __name__ == "__main__":
    main()
