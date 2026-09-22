#!/bin/sh
# DPA26TZ05-DV003 SPEED DIAL — reproduce every number Part One quotes from the
# Numberz.ai side, from the sources in this folder, on a bare machine.
#
#   sh 08_Analysis/code/reproduce.sh          (from the pursuit folder, or anywhere)
#
# Run by _Shared_Templates/scripts/evidence_pack.sh (Gate 0), which tees the
# log to logs/run_<UTC>.log, hashes results/*.json into _result_digests.txt
# and appends RUN_RECORD.md. Standard library only; pytest is used when it is
# installed and a shim runner otherwise (see run_tests.py).
#
# Steps, each writing one JSON under results/:
#   e1_tests.json     the reference implementation's 20 tests
#   e2_demo.json      the walkthrough, byte-compared with the shipped DEMO_OUTPUT.txt;
#                     the 537,600 / 672,000 paths, the refused promotion, the freeze,
#                     the 639,450 held-out answer and the S06 refusal
#   e3_devtest.json   the RECORDED developer test of 1 Sep 2026, transcribed and
#                     hashed (not re-executed: it was a hand-run test on the live platform)
#   e4_swansf.json    the SWAN-SF genetic-programming study's own test suite and result
#                     hashes, if code/swansf-sr is present (synthetic only; not run on
#                     real partitions; nothing from it is quoted as a result)
#   manifest.json     every input file hashed, and the pre-registration lock check
#
# Exit status is non-zero if any of e1, e2, e3 or the lock check fails.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)          # 08_Analysis
PP="$HERE/pathproof"
RES="$ROOT/results"
mkdir -p "$RES"
PY=${PYTHON:-python3}
status=0

echo "== SPEED DIAL evidence pack: reproduce.sh  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "== python: $($PY --version 2>&1)   root: $ROOT"

if [ ! -d "$PP/hazpath" ]; then
  echo "!! code/pathproof is not unpacked here (expected $PP/hazpath). Unpack 06_Evidence/pathproof.tar.gz into code/pathproof first."
  exit 2
fi

echo "-- e1: tests"
$PY "$HERE/run_tests.py" "$PP" "$RES/e1_tests.json" || status=1

echo "-- e2: walkthrough"
$PY "$HERE/run_demo.py" "$PP" "$RES/e2_demo.json" || status=1

echo "-- e3: recorded developer test (transcription + hashes)"
$PY "$HERE/transcribe_devtest.py" "$ROOT/recorded_run_2026-09-01" "$RES/e3_devtest.json" || status=1

echo "-- e4: SWAN-SF study (optional)"
if [ -d "$HERE/swansf-sr" ]; then
  $PY "$HERE/check_swansf.py" "$HERE/swansf-sr" "$RES/e4_swansf.json" || echo "   (e4 recorded a failure; it does not gate the pack, nothing from it is quoted in Volume 2)"
else
  echo "   code/swansf-sr not present; skipped"
fi

echo "-- manifest and lock check"
$PY "$HERE/manifest.py" "$ROOT" "$RES/manifest.json" || status=1

# memory_dump.json is a by-product of the walkthrough; it is not a result file and the
# manifest does not hash it. On a mount that refuses deletes it simply stays behind.
rm -f "$PP/memory_dump.json" 2>/dev/null || :

echo "== done, status $status  (0 = every gated step passed)"
exit $status
