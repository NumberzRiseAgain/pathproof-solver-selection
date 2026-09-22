#!/usr/bin/env bash
# Independent verification. No dataset required.
#
#   bash verify.sh
#
# Runs the test suite, then the three synthetic validations that underwrite the
# claims in docs/PREREGISTRATION.md. Roughly 10 minutes on a laptop.
#
# Every number printed here is from SYNTHETIC data and is a check on the
# METHOD, not a scientific result. Results on real SWAN-SF require the dataset
# and take hours - see docs/FOR_REVIEWER.md section 3.

set -u
cd "$(dirname "$0")"
mkdir -p results

line() { printf '\n%s\n' "======================================================================"; }

line
echo "STEP 1 of 4   test suite"
line
python3 tests/run_tests.py || { echo "TESTS FAILED - stop here and report this"; exit 1; }

line
echo "STEP 2 of 4   univariate surrogate"
echo "Expect: the search finds the planted parameter. Checks the plumbing only -"
echo "a single terminal is the true optimum here, so this cannot show composition."
line
python3 src/run_experiment.py --synthetic --seeds 3 --generations 15 \
    --population 800 --out ../results/verify_1_univariate.json \
    --tag "verify: univariate surrogate" 2>&1 | grep -vE "Warning|warn"

line
echo "STEP 3 of 4   compositional surrogate"
echo "Signal is carried only by a PRODUCT of two parameters, marginals matched."
echo "Expect: the discovered expression names TOTUSJH and R_VALUE together."
line
python3 src/run_experiment.py --synthetic --seeds 3 --generations 20 \
    --population 1500 --out ../results/verify_2_compositional.json \
    --tag "verify: compositional surrogate" 2>&1 | grep -vE "Warning|warn"

line
echo "STEP 4 of 4   recursive surrogate  (the load-bearing one)"
echo "Signal lives in within-window covariance, invisible to every static"
echo "descriptor by construction. Expect the static formula near the noise floor"
echo "and the recursive rule near 1.0, with a large positive margin."
line
python3 src/run_recursive.py --synthetic --signal recursive --seeds 5 \
    --generations 25 --population 800 \
    --out ../results/verify_3_recursive.json \
    --tag "verify: recursive surrogate" 2>&1 | grep -vE "Warning|warn"

line
echo "Done. JSON for each step is in results/."
echo
echo "What a PASS looks like:"
echo "  - 39 tests passed"
echo "  - step 4: RECURSIVE RULE well above STATIC FORMULA, margin > +0.5"
echo "  - step 4: the winning rule contains 's' and names TOTUSJH / R_VALUE"
echo
echo "If step 4's margin is small or negative, check the seed lines first:"
echo "per-seed success is about 60%, so 5 seeds are needed. Report what you see"
echo "either way - a failure here is information, not something to re-run until"
echo "it passes."
line
