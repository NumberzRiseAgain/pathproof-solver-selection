#!/usr/bin/env python3
"""Write results/e4_swansf.json for the SWAN-SF genetic-programming study (swansf-sr).

The study is pre-registered (docs/PREREGISTRATION.md) and has been run on a
synthetic surrogate only; the confirmatory run on real SWAN-SF partitions has
not happened. This script runs the study's own test runner if it has one,
hashes its pre-registration and whatever result files it shipped, and records
that status. Nothing from this file is quoted in Volume 2 as a result; Part One
describes the study as pre-registered and not run.

Usage: python3 check_swansf.py <swansf-sr dir> <results/e4_swansf.json>
"""
import sys, os, json, hashlib, subprocess, datetime, glob

d, out = sys.argv[1], sys.argv[2]
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

res = {"artifact": "e4_swansf", "what": "swansf-sr: pre-registered GP study of readable flare predictors on SWAN-SF",
       "run_utc": datetime.datetime.utcnow().isoformat() + "Z", "real_data_run": False,
       "status_in_volume": "pre-registered, not run; listed so a reviewer knows it exists; changes nothing in the feasibility determination"}
prereg = os.path.join(d, "docs", "PREREGISTRATION.md")
res["preregistration_sha256"] = sha(prereg) if os.path.exists(prereg) else None
res["shipped_results"] = {os.path.relpath(p, d): sha(p) for p in sorted(glob.glob(os.path.join(d, "results", "*.json")))}
try:
    head = subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:
    head = ""
res["git_head"] = head or "no repository"

runner = os.path.join(d, "tests", "run_tests.py")
if os.path.exists(runner):
    p = subprocess.run([sys.executable, runner], cwd=d, capture_output=True, text=True)
    tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-5:])
    res["tests"] = {"command": "python3 tests/run_tests.py", "exit": p.returncode, "tail": tail}
    ok = p.returncode == 0
else:
    res["tests"] = {"command": None, "note": "no tests/run_tests.py in the study"}
    ok = True
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w"), indent=2)
print(f"e4_swansf: tests exit {res['tests'].get('exit', 'n/a')}; {len(res['shipped_results'])} shipped result file(s) hashed; real-data run: no")
sys.exit(0 if ok else 1)
