# recorded_run_2026-09-01 — the developer test on the live platform

A **recorded** run, not a reproducible one. On 1 September 2026 a developer ran four
configurations through the same three questions against a Postgres instance holding the
33-row synthetic HazMat fixture (seeded defects: three repeated transaction identifiers,
three spellings of one state, one shop with no transactions), following the protocol in
`DEVELOPER_TEST.md`. Every generated query, result, latency and token count was recorded in
`SPEEDDIAL_Developer_Test_Final_Report.xlsx`.

The two files here are byte copies of the ones filed in `06_Evidence/` (which is where
Volume 5 points). `code/transcribe_devtest.py` hashes both and writes the transcription
Part One quotes to `results/e3_devtest.json`; if either file is missing the transcription
is refused, so the arms table in the volume never rests on memory alone.

| Arm | Q1 ordered value | Q2 distinct states | Q3 coverage | Correct |
|---|---|---|---|---|
| A, raw model, schema only | SQL error | 3 | SQL error | 0 of 3 |
| B, platform, context off | 672,000 | 3 | 0 | 0 of 3 |
| C, context on, memory empty | 537,600 | 1 | refusal | 3 of 3 |
| D, memory seeded | 537,600 | 1 | refusal | 3 of 3 |

Reported against the design, in the volume as here: Arm D changed nothing because it
re-used Arm C's questions (reuse, not transfer); the verifier scored a correct 537,600 at
0.29 and 0.40 because a masking rule had treated a national stock number as personal data
(logged and fixed); Arm A failed two questions on execution and Arm B fixed execution
without fixing a single answer.
