#!/usr/bin/env python3
"""Write results/e3_devtest.json from the recorded developer test of 1 September 2026.

The developer test was run by hand on the live interpretDB platform against a
Postgres instance (four arms, three questions) and its report was filed as an
Excel workbook and a markdown protocol in 06_Evidence. It is a RECORDED run:
nothing here re-executes it. This script fixes the transcription the volume
quotes, hashes the report files it was read from, and refuses to write the
result if either file is missing, so the numbers in Part One trace to a
hashed file and not to memory.

Usage: python3 transcribe_devtest.py <recorded_run dir> <results/e3_devtest.json>
"""
import sys, os, json, hashlib, datetime

rec, out = sys.argv[1], sys.argv[2]
files = {
    "report_xlsx": "SPEEDDIAL_Developer_Test_Final_Report.xlsx",
    "protocol_md": "DEVELOPER_TEST.md",
}
hashes = {}
for key, name in files.items():
    path = os.path.join(rec, name)
    if not os.path.exists(path):
        print(f"e3_devtest: missing {path}; the recorded run must be copied into {rec} first")
        sys.exit(1)
    hashes[key] = {"file": name, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                   "bytes": os.path.getsize(path)}

result = {
    "artifact": "e3_devtest",
    "what": "interpretDB developer test, four arms x three questions, Postgres, 33 synthetic rows with seeded defects",
    "kind": "recorded run, transcribed; not re-executed by reproduce.sh",
    "run_date": "2026-09-01",
    "transcribed_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "source_files": hashes,
    "fixture": {"rows": 33, "tables": 4,
                "seeded_defects": ["three repeated transaction identifiers", "three spellings of one state",
                                   "one shop with no transactions"]},
    "questions": {"Q1": "ordered value of the primer, FY24 to FY26 (true 537,600)",
                  "Q2": "distinct manufacturer states (true 1)",
                  "Q3": "coverage question on a shop with no transactions (correct answer: refuse)"},
    "arms": {
        "A": {"config": "raw model, schema only", "Q1": "SQL error", "Q2": 3, "Q3": "SQL error", "correct": 0, "of": 3},
        "B": {"config": "platform, context off", "Q1": 672000, "Q2": 3, "Q3": 0, "correct": 0, "of": 3},
        "C": {"config": "context on, memory empty", "Q1": 537600, "Q2": 1, "Q3": "refusal", "correct": 3, "of": 3},
        "D": {"config": "memory seeded", "Q1": 537600, "Q2": 1, "Q3": "refusal", "correct": 3, "of": 3},
    },
    "claim_carried_by": "B to C: same platform, connector, questions and database; the only variable is the declared business context",
    "reported_against_the_design": [
        "Arm D changed nothing because it re-used the questions Arm C had answered; it exercises reuse and says nothing about transfer",
        "the verifier scored a correct 537,600 at 0.29 and 0.40 because a masking rule treated a national stock number as personal data; logged and fixed",
        "Arm A failed two questions on execution (text column compared against integers); Arm B fixed that and every query ran, yet no answer was right",
    ],
    "does_not_establish": ["any effect size", "transfer across databases (the pre-registered experiment has not run)",
                           "anything about the 65 to 73 percent execution-accuracy comparison, whose run logs are not in this pack"],
}
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(result, open(out, "w"), indent=2)
print(f"e3_devtest: transcribed from {files['report_xlsx']} ({hashes['report_xlsx']['sha256'][:12]}...) "
      f"and {files['protocol_md']} ({hashes['protocol_md']['sha256'][:12]}...); arms A 0/3, B 0/3, C 3/3, D 3/3")
