"""
Driver. Runs the pre-registered protocol end to end and writes a results JSON.

Protocol (fixed before any test partition is scored):

  train on SWAN-SF partition k, tune on an inner stratified split of k,
  test once on partition k+1. No test data touches feature scaling, threshold
  selection, model selection or the GP fitness.

  headline metric TSS, secondary HSS2, both with percentile bootstrap CIs.
  baselines: Random Forest and RBF-SVM on the identical feature matrix, each
  with its own threshold tuned the same way. Comparing a tuned GP against an
  untuned baseline is the other classic way to overstate.

Usage
  python run_experiment.py --synthetic                 # pipeline validation
  python run_experiment.py --data /path/to/SWAN-SF --train 1 --test 2
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))

import swan_io                                    # noqa: E402
from features import TrainOnlyScaler, extract     # noqa: E402
from gp_search import multi_start, score_of       # noqa: E402
from metrics import best_threshold, bootstrap_ci, evaluate, hss2, tss  # noqa: E402


def _git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


def _baseline(name, F_tr, y_tr, F_val, y_val, F_te, y_te, seed):
    if name == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1)
    elif name == "svm":
        clf = SVC(kernel="rbf", C=2.0, gamma="scale",
                  class_weight="balanced", random_state=seed)
    else:
        raise ValueError(name)

    clf.fit(F_tr, y_tr)
    dec = (lambda F: clf.predict_proba(F)[:, 1]) if name == "rf" else clf.decision_function
    thr, val_tss = best_threshold(y_val, dec(F_val), metric=tss)
    res = evaluate(y_te, dec(F_te), thr)
    lo, hi = bootstrap_ci(y_te, dec(F_te), thr, metric=tss, seed=seed)
    res.update({"tss_ci95": [lo, hi], "val_tss": val_tss, "model": name})
    return res


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
    ap.add_argument("--data", default=None, help="unzipped SWAN-SF root")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--train", type=int, default=1)
    ap.add_argument("--test", type=int, default=2)
    ap.add_argument("--mode", default="mvts", choices=["mvts", "vector"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--population", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="../results/run.json")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if not args.synthetic and not args.data:
        ap.error("give --data <SWAN-SF root> or --synthetic")

    if args.synthetic:
        tr = swan_io.make_synthetic_partition(n=1400, seed=11)
        te = swan_io.make_synthetic_partition(n=1400, seed=22)
    else:
        tr = swan_io.load_partition(args.data, args.train, limit=args.limit)
        te = swan_io.load_partition(args.data, args.test, limit=args.limit)

    print(f"train {tr.describe()}")
    print(f"test  {te.describe()}")
    if tr.synthetic:
        print("\n*** SYNTHETIC DATA - pipeline validation only. "
              "No number below is a scientific result. ***\n")

    F_tr_raw, names = extract(tr.X, mode=args.mode, params=tr.params)
    F_te_raw, _ = extract(te.X, mode=args.mode, params=te.params)

    # inner split for threshold + model selection, stratified on the rare class
    idx_tr, idx_val = _inner_split(tr)

    scaler = TrainOnlyScaler().fit(F_tr_raw[idx_tr])
    F_tr = scaler.transform(F_tr_raw[idx_tr])
    F_val = scaler.transform(F_tr_raw[idx_val])
    F_te = scaler.transform(F_te_raw)
    y_tr, y_val, y_te = tr.y[idx_tr], tr.y[idx_val], te.y

    print(f"features: {F_tr.shape[1]}  train {len(y_tr)}  val {len(y_val)}  test {len(y_te)}")

    # ---- discovery -------------------------------------------------------
    runs = multi_start(
        F_tr, y_tr, F_val, y_val, names,
        seeds=tuple(range(args.seeds)),
        generations=args.generations, population_size=args.population)

    best = runs[0]
    sc_te = score_of(best["model"], F_te, best["orientation"])
    gp_res = evaluate(y_te, sc_te, best["threshold"])
    lo, hi = bootstrap_ci(y_te, sc_te, best["threshold"], metric=tss, seed=0)
    h_lo, h_hi = bootstrap_ci(y_te, sc_te, best["threshold"], metric=hss2, seed=0)
    gp_res.update({
        "tss_ci95": [lo, hi], "hss2_ci95": [h_lo, h_hi],
        "expression": best["expr"], "expression_length": best["length"],
        "val_tss": best["val_tss"], "model": "gp",
        "orientation": best["orientation"],
        "n_seeds": args.seeds,
        "all_seed_val_tss": [r["val_tss"] for r in runs],
        "all_seed_lengths": [r["length"] for r in runs],
    })

    # ---- baselines -------------------------------------------------------
    bl = [_baseline(n, F_tr, y_tr, F_val, y_val, F_te, y_te, seed=0)
          for n in ("rf", "svm")]

    out = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": _git_rev(),
        "python": platform.python_version(),
        "synthetic": bool(tr.synthetic),
        "tag": args.tag,
        "protocol": {
            "train_partition": "synthetic" if tr.synthetic else args.train,
            "test_partition": "synthetic" if te.synthetic else args.test,
            "representation": args.mode,
            "n_features": int(F_tr.shape[1]),
            "inner_val_fraction": 0.30,
            "threshold_fit_on": "inner validation split of train partition",
            "headline_metric": "TSS",
        },
        "class_balance": {
            "train_positive_rate": tr.positive_rate,
            "test_positive_rate": te.positive_rate,
            "train_n": tr.n, "test_n": te.n,
        },
        "gp": gp_res,
        "baselines": bl,
        "margin_tss_vs_best_baseline": gp_res["tss"] - max(b["tss"] for b in bl),
    }

    outp = Path(__file__).parent / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 72)
    print(f"GP   TSS {gp_res['tss']:+.4f}  [{lo:+.4f}, {hi:+.4f}]   "
          f"HSS2 {gp_res['hss2']:+.4f}   nodes {gp_res['expression_length']}")
    for b in bl:
        print(f"{b['model'].upper():<4} TSS {b['tss']:+.4f}  "
              f"[{b['tss_ci95'][0]:+.4f}, {b['tss_ci95'][1]:+.4f}]   "
              f"HSS2 {b['hss2']:+.4f}")
    print(f"margin vs best baseline: {out['margin_tss_vs_best_baseline']:+.4f}")
    print("=" * 72)
    print(f"\ndiscovered expression:\n  {gp_res['expression']}\n")
    print(f"written to {outp}")


if __name__ == "__main__":
    main()
