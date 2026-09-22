# Submittability review: "What a cheap spectral probe can and cannot tell you" (arxiv_selector_study.tar.gz, .tex of 14 Sep 13:20)

Built in the cloud from the tarball, every hash in anc/ checked, score.py re-run on the shipped raw.jsonl, and the paper's raw-derived numbers recomputed from raw.jsonl directly.

## What passed

- anc/ANC.sha256 and results/RESULTS.sha256 resolve completely. PREREG.sha256's final hash matches the shipped PREREGISTRATION.md, and the timeline holds: protocol locked 15:09Z, addendum 1 at 15:15Z, sweep finished 15:47Z, addenda 2 and 3 post-hoc at 17:13Z and 22:14Z and labelled so.
- Recomputed from raw.jsonl and correct: 432 runs, 7 operators with no converging method, 65 usable, 25 recurrence-early claims, 19 timeouts, 1 breakdown, unpreconditioned CG the oracle on 23 of 65 (all 12 of F3, 11 of 12 of F6), F4 at jump 1e6 and largest size converging only under Chebyshev, probe times 0.027 to 0.485 s with median 0.056 s.
- Stable across every scoring environment I tried: p90 conditional regret 0.3519, unconditional 10.66, 6 non-converging picks, coverage 0.2154, abstention p90 0.3469 with 12 non-converging, kappa-only baseline 0.00 / 2.18 / 8, deployment p90 3.40. Table 2's F1 to F4 and F6 rows reproduce exactly.
- References are real and correct.

## Blocker: the headline number depends on the scikit-learn version, and the artifacts do not record it

Running the shipped score.py on the shipped raw.jsonl:

| environment | median regret | winner accuracy | median criterion |
|---|---|---|---|
| scores.json as shipped (paper) | 0.1034 | 0.4154 (27/65) | FAIL |
| scikit-learn 1.8.0, numpy 2.4.4, Python 3.11.15 | 0.0000 | 0.4923 (32/65) | PASS |
| scikit-learn 1.7.2 | 0.1174 | 0.4000 (26/65) | FAIL |
| scikit-learn 1.6.1 | 0.1174 | 0.4000 (26/65) | FAIL |
| scikit-learn 1.5.2 | 0.0000 | 0.4923 (32/65) | PASS |

p90, the non-converging count, coverage and the abstention and kappa-only rows are identical in all five. Only the median and the accuracy move, and the pre-registered median criterion flips with the library version. The shipped scores.json corresponds to none of the four versions I ran, and run_meta.json records Python, numpy and platform but not scikit-learn, so nobody can reproduce the paper's exact 0.1034 from the pack. The 13 Sep RECONCILIATION note diagnosed the earlier mismatch as "an earlier version of the scorer"; the table above says it was the environment, since one unchanged score.py produces the old 0.00 / 0.4923 pair under two sklearn versions and the new pair under none.

What this breaks in the text: the abstract's "All three of our pre-registered criteria fail" and "Median regret is 0.10 against a registered bar of 0.05"; Section 8.1's first sentence; the baseline-to-beat comparison (0.10 against 0.00 holds under 1.6 and 1.7, not under 1.5 or 1.8); Section 10's "reproduces every number in this paper from scratch"; and the anc README's "score.py is deterministic and reproduces results/scores.json byte for byte".

What to do. Do not pick a version and hide the rest; the paper's own thesis is that the median is uninformative and the tail is where the signal lives, and this is the cleanest evidence for it in the whole study. Record scikit-learn in run_meta.json, re-run score.py under a pinned version and ship that scores.json, add the table above to Section 8.1 (or a short 8.1a), and rewrite the criteria sentence to: the tail and accuracy criteria fail in every environment tested; the median criterion fails in two library versions and passes in two, with the swing between 0.00 and 0.12 entirely inside learner-implementation noise, which is a fourth reason the median should not be the reported quantity. The baseline-to-beat sentence becomes "the twelve-feature selector does not beat the condition-estimate baseline at the median under any version, and ties it under two".

## Must fix

1. Section 8.7's probe accounting uses solve-only time. "9 of 65, up to 12.3x, median ratio 0.31" is the probe against the minimum t_solve across methods, excluding setup. Against the fastest end-to-end time, which is the T the paper defines in Section 7 and uses everywhere else, it is 3 of 65, up to 6.9x, median 0.24. Either state the solve-only definition in 8.7 and the abstract, or switch to the end-to-end figures; the qualitative point survives either way.
2. Section 10 says score.py reproduces every number. The misses paragraph in 8.3 (27 exact, 19 of 32 under twenty per cent, median miss 16.9 per cent, the F6 and F4 per-fold miss costs) and the 8.7 probe ratios are not in scores.json and are not printed by score.py. Emit them, or say which numbers come from a separate pass.
3. ARTIFACTS.sha256 carries paper/selector.tex at 1cc1518a... and the tarball's selector.tex hashes c80a1080..., so the manifest the README says will "confirm that the PDF you are reading is the one these results were computed for" fails on the shipped paper. Regenerate it after the final edit, or drop the two paper/ lines.
4. Conclusion, first sentence: "fails all three criteria we registered in advance, and the tail fails on one family out of six" contradicts 8.2, which says the tail bar is missed on four folds; one family is where the unconditional tail is infinite. Reword.
5. Section 8.8: "a true value near 2 x 10^-7" is 8.3e-8 to 1.8e-7 across the four methods on that operator. Say "between 8 x 10^-8 and 2 x 10^-7".
6. Build: the tarball fails on a TeX install without cm-super (microtype font expansion on bitmap Computer Modern). arXiv has cm-super, so it will probably build there, but adding \usepackage{lmodern} before microtype costs nothing and removes the risk.

## Minor

- Abstract is about 3,000 characters; arXiv's field takes 1,920. Prepare the short version before you are at the form.
- "Nine and three is twelve" belongs in the addendum, not the paper; Section 5's paragraph on the miscount can be one sentence.
- The paper says "seventy-two generated" operators in words and 72 in figures in different places; pick one.

## Verdict

Not submittable as shipped: the primary pre-registered criterion's verdict depends on an unrecorded library version, and the paper states it as a fixed fact. The repair is small and makes the paper better, because the instability is on the median and nowhere else, which is the argument the paper is already making. After the blocker, items 1 to 4, and a regenerated manifest, it is ready.

---

# Addendum: revised build (selector.tex of 14 Sep 13:50, 10 pp)

## Closed

- The blocker is handled better than I asked. Table 2 now gives five numeric stacks including the one that produced the paper's own numbers (numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, Python 3.10.12 on the Linux aarch64 box), shows that pinning scikit-learn alone is not enough (rows one and two share 1.7.2 and disagree), and the abstract and 8.1 state the criteria correctly: tail and accuracy fail everywhere, the median misses in three stacks and meets in two, everything else identical. The mechanism paragraph (argmin over near-ties versus which methods finish) is the right reading and strengthens the thesis. scores.json now carries an environment block, requirements.txt pins row one, Section 10 says which numbers a different stack will not reproduce. My own runs match rows two and four to the digit.
- 8.7 is on end-to-end time: 3 of 65, up to 6.9x, median 0.24, in body and abstract.
- Conclusion no longer says the tail fails on one family; it says four folds miss the bars.
- ARTIFACTS.sha256's paper/selector.tex line now matches the shipped selector.tex (bb74b17f).
- Fonts are Latin Modern, so the microtype expansion failure on a TeX without cm-super is gone.
- "honest" is out.

## Left, none blocking

1. 8.8 still says "a true value near 2 x 10^-7"; the four values are 8.3e-8 to 1.8e-7.
2. Section 10 says score.py "emits the probe figures of 8.7"; the 8.3 miss statistics (27 exact, 19 of 32 under twenty per cent, median miss 16.9 per cent, the F6 and F4 per-fold miss costs) are still not named as emitted. Either they are in scores.json now, in which case say so, or add one clause saying they come from a separate pass over the choices.
3. The date line still reads "Draft of 13 September 2026"; the build is 14 September.
4. Abstract is about 3,100 characters; prepare the 1,920-character version before you are at the form.

## Verdict

Ready to upload after the three one-line edits (items 1 to 3). The paper is stronger than the version I reviewed this morning, and the environment table is now its best paragraph.
