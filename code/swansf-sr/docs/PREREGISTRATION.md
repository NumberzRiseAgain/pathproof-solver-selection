# Pre-registration: GP-discovered interpretable descriptors for solar flare prediction on SWAN-SF

**Written 2026-09-01 (UTC), before any SWAN-SF partition has been loaded, scored, or inspected.**
Committed to git on the same date. Any change after the first real-data run is
recorded as an amendment below, with its own date and reason, and never by
editing the text above it.

The point of this document is narrow and mechanical. Symbolic regression on an
extremely imbalanced benchmark has many free choices, and choosing them after
seeing the test scores is how a result becomes unreproducible. Everything that
could be tuned to flatter the outcome is fixed here first.

---

## 1. Question

Can genetic programming discover a **closed-form, human-readable scalar
functional** of a SWAN-SF observation window that predicts ≥M1.0 solar flares
at least as well as a strong opaque baseline on the same features and the same
split?

The result is allowed to come back **no**. A negative result is reportable and
will be reported: it would say that on this benchmark interpretability costs
measurable skill, which is itself a finding worth publishing and worth knowing
before building a platform on the opposite assumption.

## 2. Hypotheses

- **H1 (skill).** The best GP-discovered expression achieves test TSS within
  0.05 of the best baseline (Random Forest, RBF-SVM), i.e. `TSS_gp ≥ TSS_baseline − 0.05`.
- **H2 (parsimony).** That expression has ≤ 25 nodes and references ≤ 6
  distinct physical parameters, so it can be printed in a paper and read aloud.
- **H3 (stability).** The parameters appearing in the winning expression recur
  in ≥ 3 of 5 independent seeds.

H1 is the headline. H2 is what makes the result relevant to the interpretability
question. H3 is what separates a discovery from a lucky seed.

## 3. Data

SWAN-SF, Harvard Dataverse `doi:10.7910/DVN/EBCFKM`
(Angryk, Martens, Aydin, Kempton et al., *Scientific Data* 7, 227, 2020).

- Five chronological partitions, CSV per MVTS slice, 12-minute cadence.
- Observation window `T_obs` = 12 h → 60 timesteps, taken as the window
  closest in time to the event.
- 24-parameter core subset, order fixed in `src/swan_io.py::CORE_PARAMS`.
- Binary task: **positive** = X or M class in the prediction window;
  **negative** = C, B, or flare-quiet.

## 4. Protocol — fixed

| Choice | Value | Fixed because |
|---|---|---|
| Split | train on partition *k*, test once on partition *k+1* | chronological, avoids temporal leakage between neighbouring slices of the same active region |
| Primary pair | train p1 → test p2 | declared now; other pairs are reported as secondary, never substituted for it |
| Inner validation | 30% stratified split of the train partition | threshold and model selection happen here and nowhere else |
| Feature scaling | robust median/IQR, **fit on the inner-train rows only** | the most common source of inflated SWAN-SF numbers |
| Headline metric | TSS | insensitive to the imbalance ratio, which is the whole difficulty |
| Secondary metric | HSS2 | is imbalance-sensitive; a method that lifts TSS while destroying HSS2 has only learned to cry wolf |
| Uncertainty | percentile bootstrap over test slices, 2000 resamples, 95% | reported for every headline number, no exceptions |
| Seeds | 5 independent GP runs | all five validation scores are published, not just the winner |
| Model selection | highest **validation** TSS | test partition is scored exactly once per configuration |
| Baselines | Random Forest, RBF-SVM | identical feature matrix, identical split, each with its own threshold tuned the same way |

**Stopping rule.** The test partition is scored once per configuration. If a
configuration is changed after seeing a test score, the change and its
motivation are logged in §7 and the run is reported as exploratory, not
confirmatory.

**No test data** touches feature scaling, threshold selection, GP fitness, or
model selection.

## 5. Search space

Function set: `add, sub, mul, div (protected), sqrt, log, abs, neg, max, min`.
`exp` is excluded — it overflows on scaled SHARP parameters and buys fitness
with terms no physicist will accept.

Terminals: six temporal descriptors per parameter — `last, mean, std, slope,
delta, maxabs` — so every leaf names a physical quantity and a plain temporal
operation. 24 parameters × 6 = 144 terminals.

Fitness: |Gini| (rank separability) on out-of-bag rows, with parsimony
pressure. Orientation is resolved once on the validation fold and frozen.

## 6. What will be reported regardless of outcome

- Test TSS and HSS2 with CIs for GP and both baselines.
- The winning expression verbatim, its node count, and its parameter set.
- Validation TSS for all five seeds, and the parameter overlap across seeds.
- Every configuration run, including those that failed.
- Class balance of both partitions.
- Code, seeds, and the exact partition pair, so the run can be repeated.

## 7. Amendments

**A1 — 2026-09-01, before any real data.** `parsimony_coefficient` was swept on
the synthetic surrogate over {0, 5e-5, 2e-4, 8e-4, 2e-3}. All five values gave
an identical winner, so parsimony is **not** the binding constraint at this
fitness scale and is fixed at the default 2e-3. The sweep instead revealed that
the first surrogate carried a signal a single feature captured on its own,
which meant it could not test whether the search composes. A compositional
surrogate was added (§8) and the search was re-validated against it. No real
partition was loaded at any point in this calibration.

**A2 - 2026-09-01, before any real data. Search space extended from formulas to procedures.**

The static search discovers `f(last, mean, std, slope, delta, maxabs)` - a closed-form
combination of precomputed summary statistics, evaluated once. That is a FORMULA. Every
example the DARPA topic cites (Transformers rediscovering the Kalman filter, GP
rediscovering wavelets, MCTS rediscovering optimisation algorithms) is a PROCEDURE:
something with state that iterates over a sequence. The original design did not reach
that category and should have said so.

A recursive search is therefore added as the **primary** method:

    s_0 = 0 ;  s_t = clip( g(s_{t-1}, x_t, dx_t) ) ;  score = s_T

with GP searching over `g`. The static search is retained unchanged as the **baseline**,
which makes "does a discovered procedure buy anything over a discovered formula?" a
measured question rather than an assumption. Added as **H4**: on the primary partition
pair, the recursive rule achieves higher test TSS than the static formula.

Three things must be stated plainly because they bound the claim:

1. **The filter shape is seeded, not discovered.** 30% of the initial population is
   built as the affine recursion `s_t = a*s_{t-1} + <innovation>`, the general form of a
   linear filter. Which parameters enter the innovation, how they combine, and the
   retention coefficient are discovered. The decision to carry state forward at all is
   given. Unseeded runs found the correct interacting variables but almost never
   discovered accumulation, so this is domain knowledge supplied by us and it will be
   reported as such, not folded into the discovery claim.
2. **Seed variance is high, and the budget is load-bearing.** On the validation
   surrogate 3 of 5 seeds recovered the planted rule exactly (val TSS 1.000), one
   reached 0.666, one 0.059 - a per-seed success rate near 60%. All five are reported.
   A single successful seed is not a result, and neither is a single failed one: at
   2 seeds / 12 generations / population 400 the same experiment returned 0.034 and
   looked like a refutation. Success rate as a function of search budget is therefore
   reported as its own curve, not assumed. Any run of the recursive search uses at
   least 5 seeds.
3. **Stability is enforced, not emergent.** The grammar carries tanh and the state is
   clipped each step, because an unbounded fold diverges within a few timesteps.
   Diverging programs are penalised, not silently dropped.

No real partition was loaded at any point in this extension.

**A3 - 2026-09-01, after the first real-data shakedown, before the confirmatory run.**
Two protocol changes and one added reference, all prompted by diagnostics rather than
by a test score. The confirmatory run has still not been made.

1. **The inner split is grouped by active region.** SWAN-SF slices are sliding windows
   over the same region - measured here at 580 regions across 3000 partition-1 slices,
   about 5 slices each - so a slice-level split puts near-duplicates on both sides. The
   effect was large: the static search scored test TSS 0.79 under a random inner split
   and 0.54 under a grouped one. Roughly a quarter of the original figure was overlap.
   Reported numbers use the grouped split.

2. **Missingness is audited per class at load.** Non-finite values are replaced by zero
   downstream, so a class difference in missingness is a label proxy. Partition 1 is
   clean (0.0018 positives vs 0.0021 negatives). Partition 2 is not symmetric: 0.0017
   against 0.0127, a factor of seven, in the partition used for test. Both are small in
   absolute terms and the effect is expected to be minor, but it is measured and
   reported rather than assumed away.

3. **A best-single-descriptor reference floor is now reported alongside both searches.**
   The shakedown's discovered rule was `s_t = max(s, ABSNJZH)` - a running maximum,
   which folded over the window is exactly `ABSNJZH_maxabs`, a terminal already present
   in the static search's own space. The recursive rule therefore beat the static
   formula by 0.252 without expressing anything a formula could not. The margin measured
   search efficiency, not expressiveness, and reporting it as evidence for H4 would have
   been wrong.

   Both searches are now scored against the best single descriptor selected on
   validation across all 144 candidates. That selection is itself optimistic, which is
   the point: a discovered rule that merely ties a cherry-picked single feature has
   rediscovered one feature and dressed it as a procedure. **H4 is amended**: the
   recursive rule must beat both the static formula and the single-descriptor floor.

---

### Pipeline validation preceding this

Before any real data, the pipeline was exercised on a clearly-labelled
synthetic surrogate with SWAN-SF's schema, imbalance and missingness, carrying
a **planted** closed-form signal. Two defects were found and fixed this way:

1. A direction-agnostic fitness combined with a `score ≥ threshold` decision
   rule silently scored every discovered expression at TSS 0.
2. The surrogate drew per-parameter unit scales independently per partition,
   which does not reflect SWAN-SF and made a working pipeline look like a
   failure.

3. The surrogate's planted signal was univariate, so a single terminal was the
   true optimum and the test could not distinguish a search that composes from
   one that does not.

### 8. Compositional validation

A second surrogate plants a signal carried only by the **product** of a trend
in TOTUSJH and a level in R_VALUE, with matched marginals: no single feature
separates the classes. Calibration on it, before any real data:

| quantity | validation TSS |
|---|---|
| best of 144 single features (selection-on-validation noise floor) | 0.397 |
| the planted product | 0.635 |

The GP recovered `mul(TOTUSJH_slope, R_VALUE_last)` - the planted structure -
scoring test TSS 0.497 [0.374, 0.603] with a 3-node expression, against Random
Forest 0.539 and RBF-SVM 0.343 on identical features and splits. That is the
H1/H2 profile this study is looking for: within 0.05 of the best opaque
baseline, in an expression short enough to print.

### 9. Recursive validation

A third surrogate plants a signal **no static descriptor can see by construction**: two
components are added to TOTUSJH and R_VALUE whose within-window correlation is +rho for
positives and -rho for negatives, leaving every per-parameter mean, std, slope, delta,
last and maxabs identically distributed across classes. Only the covariance differs.

| method | validation / test TSS |
|---|---|
| best of 144 single static descriptors | 0.397 (the selection-on-validation noise floor) |
| planted fold, written by hand | 1.000 |
| **static formula search** (baseline) | test **0.243** [0.117, 0.370], size 5 |
| **recursive rule search** (seeded) | test **0.9985** [0.9962, 1.0000], size 5 |

Discovered rule: `s_t = s + d_R_VALUE * TOTUSJH`, terminals {TOTUSJH, d_R_VALUE, s}.
Margin +0.756. Unseeded runs reached 0.585 and did not use the state terminal at all.

A defect was found and fixed during this work: raising the state-terminal probability to
0.30 made the constant-terminal branch unreachable, so no retention coefficient could
ever be drawn and no filter could express decay.

**Cross-platform reproduction.** The full-budget run was reproduced bit-for-bit on a
second machine - Python 3.9 with numpy 2.0 on Apple ARM, against Python 3.11 with
numpy 2.4 on Linux x86 - returning identical test TSS (0.9985), identical bootstrap
interval, identical discovered rule and identical static baseline (0.2428). Seeding is
therefore deterministic across platform and library version, and a reader can reproduce
the exact numbers rather than the distribution.

Synthetic runs are stamped `"synthetic": true` in every results file and no
synthetic number is a result.
