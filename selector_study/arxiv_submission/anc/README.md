# Ancillary files

Everything the paper rests on. Generated operators, classical methods, no network access
required and no data download.

    shasum -a 256 -c ANC.sha256           # check the set before trusting it
    python3 run_sweep.py                  # ~28 min, one core, writes results/raw.jsonl
    python3 score.py                      # writes results/scores.json, prints the headline

`ANC.sha256` covers exactly the files in this directory and resolves completely.
`ARTIFACTS.sha256` is the study's own manifest and additionally covers the paper source and
PDF; those two lines will not resolve here, because arXiv stores the source outside `anc/`.
Use it to confirm that the PDF you are reading is the one these results were computed for.

| file | what it is |
|---|---|
| `PREREGISTRATION.md` | the protocol, locked before the sweep, with three dated addenda. Addenda 2 and 3 are post-hoc and say so on their face |
| `PREREG.sha256` | the hash and UTC timestamp at which the protocol and each addendum were locked, appended in order |
| `ARTIFACTS.sha256` | SHA-256 of every released file |
| `operators.py` | the six families and their swept parameters; 72 operators, n from 14,400 to 120,000 |
| `solvers.py` | the six candidate methods and one preconditioned CG. Convergence is on the TRUE residual, per addendum 1 |
| `probe.py` | 30 Lanczos steps with full reorthogonalisation, and the twelve features |
| `run_sweep.py` | one record per operator and method, appended, resumable |
| `score.py` | leave-one-family-out scoring. Reads `results/raw.jsonl` and nothing else |
| `results/raw.jsonl` | 432 records, including every failure |
| `results/scores.json` | what the paper prints |
| `results/run_meta.json` | interpreter, platform, tolerance, caps, finish time |

Install from `requirements.txt`. Seeds are fixed throughout, and on the pinned stack
`score.py` reproduces `results/scores.json` byte for byte. On a different stack every
number reproduces except the median regret and the winner accuracy, which are decided by
an argmin over near-tied predictions and move with library arithmetic. The five stacks we
ran are in `results/ENVIRONMENT_SWEEP.md`.
