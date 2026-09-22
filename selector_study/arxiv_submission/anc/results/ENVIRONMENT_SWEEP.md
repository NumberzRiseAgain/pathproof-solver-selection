# Environment sweep, 14 September 2026

Same `score.py`, same `results/raw.jsonl`, five numeric stacks.

| numpy | scipy | scikit-learn | median rho | accuracy | p90 | non-converging | coverage | kappa med/p90 | abstention p90 |
|---|---|---|---|---|---|---|---|---|---|
| 2.2.6 | 1.15.3 | 1.7.2 | 0.1034 | 0.4154 | 0.3519 | 6 | 0.2154 | 0.0000 / 2.1779 | 0.3469 |
| 2.4.6 | 1.17.1 | 1.7.2 | 0.1174 | 0.4000 | 0.3519 | 6 | 0.2154 | 0.0000 / 2.1779 | 0.3469 |
| 2.4.6 | 1.17.1 | 1.6.1 | 0.1174 | 0.4000 | 0.3519 | 6 | 0.2154 | 0.0000 / 2.1779 | 0.3469 |
| 2.4.4 | 1.17.1 | 1.8.0 | 0.0000 | 0.4923 | 0.3519 | 6 | 0.2154 | 0.0000 / 2.1779 | 0.3469 |
| 2.4.6 | 1.17.1 | 1.5.2 | 0.0000 | 0.4923 | 0.3519 | 6 | 0.2154 | 0.0000 / 2.1779 | 0.3469 |

Rows one and two share a scikit-learn version and disagree, so pinning scikit-learn alone
does not fix this. The whole stack is pinned in `requirements.txt`.

Only the median and the winner accuracy move. Both are decided by an argmin over predicted
times; on many operators two candidates sit closer together than the arithmetic noise
between library builds, so the argmin flips. The ninetieth percentile, the six
non-converging picks, the coverage, the Ritz-only row and the abstention row are identical
in all five, because those are decided by operators where one method finishes and another
does not.

**This supersedes the 13 September reconciliation note.** That note recorded the mismatch
between the first `scores.json` and a later rerun as a stale scorer. It was not. The first
`scores.json` was produced under scikit-learn 1.8.0, which is row four, and the rerun was
on scikit-learn 1.7.2 with numpy 2.2.6, which is row one. The scorer never changed.
