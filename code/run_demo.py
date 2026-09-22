#!/usr/bin/env python3
"""Run the pathproof walkthrough and write results/e2_demo.json.

Runs `python3 -m hazpath.demo` in the pathproof directory, compares the output
byte for byte with the DEMO_OUTPUT.txt shipped in the locked commit, and reads
the numbers the volume quotes out of the live output (never out of the shipped
file), so a change in the implementation shows up as a mismatch here and not
as a stale number in a volume.

Usage: python3 run_demo.py <pathproof dir> <results/e2_demo.json>
"""
import sys, os, re, json, subprocess, hashlib, datetime

pathproof, out = sys.argv[1], sys.argv[2]
p = subprocess.run([sys.executable, "-m", "hazpath.demo"], cwd=pathproof, capture_output=True, text=True)
live = p.stdout
shipped = open(os.path.join(pathproof, "DEMO_OUTPUT.txt"), encoding="utf-8").read()

def grab(pattern, text=live, flags=0, cast=str):
    m = re.search(pattern, text, flags)
    return cast(m.group(1)) if m else None

def money(s):
    return float(s.replace(",", ""))

# The seed problem that carries the guard, and the same problem with the guard skipped.
seed_ok = grab(r"What did we spend ordering the primer across FY24 to FY26\?\n\s+got ([\d,\.]+)", cast=money)
seed_ok_reward = grab(r"What did we spend ordering the primer across FY24 to FY26\?\n\s+got [\d,\.]+\s+expected [\d\.]+\s+->\s+correct\s+reward ([+-][\d\.]+)", cast=float)
trap = grab(r"trap: inflated to ([\d\.]+) if deduplicate is skipped", cast=float)
noguard = grab(r"answered without the duplicate guard\)\n\s+got ([\d,\.]+)\s+expected [\d\.]+\s+->\s+WRONG\s+reward ([+-][\d\.]+)", cast=str)
noguard_val = money(noguard) if noguard else None
noguard_reward = grab(r"answered without the duplicate guard\)\n\s+got [\d,\.]+\s+expected [\d\.]+\s+->\s+WRONG\s+reward ([+-][\d\.]+)", cast=float)

# The path as a maintainer reads it: count the numbered steps in section 2.
sec2 = live.split("2. THE PATH, AS A MAINTAINER READS IT", 1)[1].split("3. COMPOSITE PROMOTION", 1)[0]
steps = re.findall(r"^\s+(\d+)\. (\w+)\(", sec2, re.M)
dedup_step = next((int(n) for n, v in steps if v == "deduplicate"), None)

# Promotion on the five-record fixture, the freeze, and the held-out problems.
promoted_nothing = "nothing met the promotion rule" in live
records = grab(r"(\d+) records over databases", cast=int)
frozen = "memory is frozen; scoring cannot write to it" in live
headline = grab(r"got ([\d,\.]+)\n\s+expected 639450\.0", cast=money)
headline_reward = grab(r"got [\d,\.]+\n\s+expected 639450\.0\s+->\s+correct\s+reward ([+-][\d\.]+)", cast=float)
abstain = "got ABSTAIN (S06 has no rows" in live
m_ho = re.search(r"held-out problems answered correctly: (\d+) of (\d+)", live)
held_out_correct, held_out_total = (int(m_ho.group(1)), int(m_ho.group(2))) if m_ho else (None, None)

result = {
    "artifact": "e2_demo",
    "what": "python3 -m hazpath.demo on the synthetic HazMat fixture (33 rows, 4 tables, 6 problems)",
    "run_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "exit": p.returncode,
    "output_sha256": hashlib.sha256(live.encode()).hexdigest(),
    "matches_shipped_DEMO_OUTPUT": live == shipped,
    "primitive_vocabulary_size": 12,
    "path_with_guard": {"steps": len(steps), "deduplicate_is_step": dedup_step,
                        "answer": seed_ok, "reward": seed_ok_reward},
    "same_question_guard_skipped": {"answer": noguard_val, "reward": noguard_reward,
                                    "error_vs_true_pct": round(100 * (noguard_val / seed_ok - 1), 1) if noguard_val and seed_ok else None,
                                    "declared_trap": trap},
    "promotion_on_five_record_fixture": {"records": records, "promoted": 0 if promoted_nothing else None,
                                         "refused_as_designed": promoted_nothing},
    "freeze_enforced": frozen,
    "held_out": {"headline_answer": headline, "headline_reward": headline_reward,
                 "coverage_gap_refused": abstain, "correct": held_out_correct, "total": held_out_total},
    "measures_no_effect_size": True,
    "note": "Validates the mechanism on a synthetic fixture. Not a statistical result; no percentage from this file belongs in a volume.",
}
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(result, open(out, "w"), indent=2)
ok = (p.returncode == 0 and result["matches_shipped_DEMO_OUTPUT"] and seed_ok == 537600.0
      and noguard_val == 672000.0 and promoted_nothing and frozen and headline == 639450.0 and abstain)
print(f"e2_demo: exit {p.returncode}; matches shipped output: {result['matches_shipped_DEMO_OUTPUT']}; "
      f"guard {seed_ok:,.0f} vs no-guard {noguard_val:,.0f}; promoted 0: {promoted_nothing}; "
      f"held-out {held_out_correct} of {held_out_total}")
sys.exit(0 if ok else 1)
