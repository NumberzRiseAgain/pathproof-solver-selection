"""Run every declared operator against every declared method, once.

Appends one JSON record per (operator, method) to results/raw.jsonl so the run
resumes if interrupted. Nothing is re-run once written.
"""
import json
import os
import platform
import sys
import time

import numpy as np

import operators as O
import probe as P
import solvers as S

OUT = "results/raw.jsonl"
os.makedirs("results", exist_ok=True)


def done_keys():
    keys = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    keys.add((r["op_id"], r["method"]))
                except Exception:
                    pass
    return keys


def main():
    done = done_keys()
    ops = O.enumerate_operators()
    t_all = time.time()
    with open(OUT, "a") as fh:
        for oi, (fam, op_id, size, param, seed) in enumerate(ops):
            if all((op_id, m[0]) in done for m in S.METHODS):
                continue
            A = O.build(fam, size, param, seed)
            n = A.shape[0]
            feats, t_probe, w = P.features(A, seed=seed)
            pr = {"ritz_min": float(w[0]), "ritz_max": float(w[-1])}
            b = np.random.default_rng(seed).standard_normal(n)
            for mname, mk, needs_probe in S.METHODS:
                if (op_id, mname) in done:
                    continue
                c = S.Counter()
                t0 = time.perf_counter()
                try:
                    M = mk(A, pr) if needs_probe else mk(A)
                    setup_failed = (mname == "M3_ic0" and M is None)
                except Exception as exc:            # a setup that cannot run is a result
                    M, setup_failed = None, True
                    print(f"setup error {op_id} {mname}: {exc}", file=sys.stderr)
                t_setup = time.perf_counter() - t0
                if setup_failed:
                    rec = dict(op_id=op_id, family=fam, n=n, nnz=int(A.nnz),
                               size_param=size, sweep_param=param, method=mname,
                               setup_failed=True, converged=False, timeout=False,
                               breakdown=False, rec_claimed_early=False,
                               iters=0, matvec=0, precond=0, t_setup=t_setup,
                               t_solve=0.0, t_probe=t_probe,
                               uses_probe=bool(needs_probe), true_res=None,
                               t_total=None, features=feats)
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    continue
                t0 = time.perf_counter()
                x, o = S.pcg(A, b, M, c)
                t_solve = time.perf_counter() - t0
                t_total = t_setup + t_solve + (t_probe if needs_probe else 0.0)
                rec = dict(op_id=op_id, family=fam, n=n, nnz=int(A.nnz),
                           size_param=size, sweep_param=param, method=mname,
                           setup_failed=False,
                           converged=bool(o["converged"]), timeout=bool(o["timeout"]),
                           breakdown=bool(o["breakdown"]),
                           rec_claimed_early=bool(o["rec_claimed_early"]),
                           iters=int(o["iters"]), matvec=int(c.matvec),
                           precond=int(c.precond), t_setup=t_setup,
                           t_solve=t_solve, t_probe=t_probe,
                           uses_probe=bool(needs_probe),
                           true_res=float(o["true_res"]), t_total=t_total,
                           features=feats)
                fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"[{oi+1}/{len(ops)}] {op_id} n={n} done  "
                  f"elapsed={time.time()-t_all:.0f}s", flush=True)

    import scipy
    try:
        import sklearn; skl = sklearn.__version__
    except Exception:
        skl = None
    meta = dict(python=sys.version.split()[0], platform=platform.platform(),
                numpy=np.__version__, scipy=scipy.__version__, sklearn=skl,
                tau=S.TAU, maxit=S.MAXIT,
                wall_cap=S.WALL_CAP, probe_steps=P.K_STEPS,
                finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open("results/run_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print("SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
