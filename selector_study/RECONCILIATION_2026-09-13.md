# Paper / data / volume reconciliation, 13 September 2026

> **Superseded in part, 14 September 2026.** The diagnosis below, that `scores.json` was
> written by a stale scorer, is wrong. The scorer never changed. The two readings came from
> two numeric stacks: the original file was produced under scikit-learn 1.8.0 and the rerun
> under 1.7.2 with numpy 2.2.6. See `results/ENVIRONMENT_SWEEP.md`. The corrected numbers
> below are still correct for the stack named in `requirements.txt`; the cause was not.

Every file in `ARTIFACTS.sha256` verifies OK. The defect is not a changed file.
Running the recorded `score.py` on the recorded `results/raw.jsonl` does not reproduce the
recorded `results/scores.json`. `scores.json` was written by an earlier version of the scorer
and then hashed into the artifact set as if it were current.

## What does not reproduce

| quantity | scores.json (and the paper) | score.py on raw.jsonl | verdict |
|---|---|---|---|
| median regret | **0.00** | **0.1034** | differs, and crosses the 0.05 bar |
| p90 regret | 0.3519 | 0.3519 | same |
| infinite-regret picks | 6 of 65 | 6 of 65 | same |
| winner accuracy | **0.4923 (32 of 65)** | **0.4154 (27 of 65)** | differs |
| coverage | 0.2154 | 0.2154 | same |
| abstention p90 | 0.3469 | 0.3469 | same |
| criteria | median PASS, tail FAIL, accuracy FAIL | **all three FAIL** | differs |

Everything else in the paper reproduces exactly, including the probe accounting
(9 of 65 at up to 12.3x, median ratio 0.31, probe 0.027 to 0.485 s, median 0.056 s),
the recurrence-residual count (25 of 432), the oracle mix (unpreconditioned CG on 23 of 65,
all twelve of F3, eleven of twelve of F6), and the per-fold table.

## Corrected baselines, from the rerun

| series | median | p90 | non-converging |
|---|---|---|---|
| selector, eleven features | 0.1034 | 0.3519 | 6 |
| Ritz condition estimate alone | 0.0000 | 2.1779 | 8 |
| null control, shuffled labels | 0.1433 | 1.2795 | 1 |
| selector with abstention and fallback | 0.0000 | 0.3469 | 12 |

Three consequences the paper has to absorb.

1. **The median no longer passes.** The abstract's "median regret of $0.00$ and passes that
   bar" is wrong. All three registered criteria fail. The shape of the argument survives, and
   the sentence "it looks fine at the median" is the part that goes.
2. **The separation from the null control at the median is gone.** 0.1034 against 0.1433, not
   0.00 against 0.13. The tail separation, 0.35 against 1.28, is what actually carries the
   claim that the probe contains signal. Say only that.
3. **The pre-registered baseline-to-beat is not beaten at the median.** Section 9.3 of the
   pre-registration says: if the eleven-feature selector does not beat the Ritz-condition-only
   selector on median regret, we say so. It does not: 0.1034 against 0.0000. The paper
   currently says the two "match at the median". They do not; the baseline wins there and
   loses catastrophically in the tail.

## Claims in the paper that are wrong independently of the stale file

- **"On F4 ... coverage is 0.33 and the tail is unchanged."** F4 p90 falls from 7.955 to 1.732
  under abstention. The tail changes a great deal. What is true is that four F4 operators still
  carry unbounded regret.
- **The abstention section reports the finite p90 only.** Abstention takes non-converging picks
  from 6 to 12 while lowering the finite p90 from 0.3519 to 0.3469 and the median from 0.1034 to
  0.0000. Comparing two p90s computed over different numbers of excluded infinities understates
  the failure. The table already carries a "non-converging" column; the abstention row needs one.
- **"wrong on fifty-eight per cent of operators for a median cost of eleven per cent"** (F6) and
  **"seventy-five per cent for ... five hundred and fifty-seven per cent"** (F4) use the fold
  median over ALL operators, not over the misses. Over the misses: F6 25.2 per cent, F4 597 per
  cent. The contrast the sentence is drawing gets stronger, not weaker.
- **The misses paragraph** follows the stale accuracy. Reproducible: 27 exact, 38 misses, 19 of
  them under twenty per cent, median miss 16.87 per cent, six non-converging.

## The probe is not charged to the selector

Confirmed against `run_sweep.py`: `t_total = t_setup + t_solve + t_probe` only when
`needs_probe`, which is M5 alone. The selector consumes the probe on every operator and is not
charged for it. Charging it, with no double count on M5 and none against the clairvoyant oracle:

| | median | p90 | non-converging |
|---|---|---|---|
| choice regret, as reported | 0.1034 | 0.3519 | 6 |
| deployment regret, probe charged | **0.4103** | **3.4008** | 6 |

The tail moves by an order of magnitude. This is a real result and it belongs in the paper.

## The volume

`Part2_body.tex` line 430: "refused 78\% of held-out operators and did not improve the tail
regret on the ones it kept."

- "refused 78 per cent" is right, 78.46 per cent.
- "on the ones it kept" is wrong. 0.347 is the blended series, the kept picks plus the fallback
  on the refused ones, not the kept subset.
- The sentence omits the two facts that matter: the fallback doubled the non-converging picks
  from 6 to 12, and abstention improved the median from 0.10 to 0.00.

Nothing else in either body cites this study. The "120 held-out operators" on Part2 line 673 is
inside the passage explicitly labelled "An illustrative scenario and a design target" and is not
a measurement claim.

## What has to happen, in order

1. Re-run `score.py`, overwrite `results/scores.json`, re-hash `ARTIFACTS.sha256`.
2. Add the deployment-regret series to `score.py` so both regrets are computed in one place.
3. Correct the paper: abstract, criteria table, null-separation sentence, baseline-to-beat
   sentence, the misses paragraph, the F4 abstention sentence, the two median-over-misses
   figures, and a non-converging column on the abstention row.
4. Correct the one sentence in `Part2_body.tex`, then re-run census, voice and audit.
