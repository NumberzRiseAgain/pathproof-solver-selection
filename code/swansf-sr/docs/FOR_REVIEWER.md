# Independent verification brief

**For a reviewer who was not present when this was built.**
Numberz.ai Inc. · SWAN-SF symbolic regression · September 2026

You are being asked to run this yourself and report what you get. Please do not
take any number in this repository on trust, including the ones quoted below.

---

## 1. Why this code exists

DARPA STTR topic **DPA26TZ05-DV003 (SPEED DIAL)** is Direct-to-Phase-II only.
That means a proposal is only responsive if it documents, from work already
performed, three things:

| | The topic's requirement, in its own words |
|---|---|
| 1 | "Successfully developed and employed methods (e.g., Transformers, Genetic Programming, Reinforcement Learning) to **discover novel algorithms that outperform existing state-of-the-art** methods in domains such as time-series analysis, data compression, and solving complex partial differential equations" |
| 2 | "Quantitatively shown the benefits of these discovered algorithms" |
| 3 | "The discovered algorithms are not black boxes. They are composed of **transparent building blocks** that are understandable to domain experts" |

Our existing evidence covered these badly. The team's published record is in
learned neural operators and time-series benchmarks, which is adjacent but is
not algorithm *discovery*, and item 3 was openly unmet in the working draft.

This experiment is the attempt to produce the missing evidence ourselves, at
private expense, and to document it honestly. **It is not backdated and it is
not described as older than it is.** The git history is the date record.

## 2. What the experiment actually does

SWAN-SF is a published benchmark (Angryk, Martens, Aydin, Kempton et al.,
*Scientific Data* 7, 227, 2020) of solar active regions. Each sample is 24
magnetic-field parameters sampled every 12 minutes for 12 hours - 60 timesteps.
About 2% of samples are followed by a major (M or X class) solar flare.

Two searches run over that data, and the difference between them is the point:

**Static search (the baseline).** Genetic programming looks for a closed-form
formula over summary statistics - `f(last, mean, std, slope, delta, maxabs)` of
each parameter. It evaluates once per sample. That discovers a **formula**.

**Recursive search (the primary method).** Genetic programming looks for an
update rule folded across the window:

```
s_0 = 0
s_t = clip( g(s_{t-1}, x_t, dx_t) )      for t = 1 … 60
score = s_60
```

It searches over `g`. What comes out has state and iterates over the sequence -
it is a **filter**, which is what the Kalman filter, wavelets and meta-solvers
all are, and what the topic's own examples all are. A formula is not.

Both produce something printable. A typical discovered rule looks like:

```
s_t = s + d_R_VALUE × TOTUSJH
```

That is a running total of one parameter's increment times another's level.
A solar physicist can read it, argue with it, and say whether it is sensible.
That is item 3.

## 3. What to run

```bash
cd swansf-sr
pip install numpy pandas scikit-learn gplearn

# 1. the test suite - about 5 seconds, no data needed
python3 tests/run_tests.py

# 2. the validation suite - about 10 minutes, no data needed
bash verify.sh

# 3. the real-data run - needs SWAN-SF, takes hours (see §6)
python3 src/run_recursive.py --data /path/to/SWAN-SF --train 1 --test 2 \
    --seeds 5 --generations 25 --population 800
```

`verify.sh` is the part we most want checked independently. It does not need
the dataset and it is where the method's claims are actually tested.

## 4. How to read the output

Every run prints three rows plus margins:

```
SINGLE FEATURE    TSS +0.7747  [+0.6807, +0.8573]  ABSNJZH_last
STATIC FORMULA    TSS +0.5377  [+0.4061, +0.6698]  size 3
RECURSIVE RULE    TSS +0.7895  [+0.7054, +0.8605]  size 3
margin  recursive - static +0.2518   recursive - single feature +0.0148
```

**TSS** (True Skill Statistic) = recall − false-alarm rate. Range −1 to +1.
It is the headline because it scores a degenerate "never predict a flare"
forecast at exactly 0, whereas accuracy would score that at 98%. **HSS2** is
reported alongside because it is imbalance-sensitive: a method that raises TSS
while collapsing HSS2 has learned to cry wolf, not to forecast.

**The bracket is a 95% bootstrap confidence interval.** Two rows whose
intervals overlap heavily are not distinguishable, whatever the point estimates
say. In the example above the recursive rule beats the single feature by 0.015
with almost total interval overlap - that is a tie, not a win, and it should be
reported as a tie.

**SINGLE FEATURE is the bar that matters.** It is the best of all 144
descriptors, chosen on validation. A discovered rule that only ties it has
rediscovered one known parameter and dressed it up as a procedure. We added
this row after a discovered rule `s_t = max(s, ABSNJZH)` turned out to be
exactly the `ABSNJZH_maxabs` descriptor the baseline already had.

Also check:
- **`inner split: GROUPED by active region`** must appear. If it says it is
  falling back to a stratified split, the number is inflated and unusable.
- **`NaN fraction positives / negatives`** should be close. A large gap means
  missing data is acting as a label proxy.
- **All five seed lines.** Genetic programming is stochastic; roughly 60% of
  seeds find a given structure. One good seed is not a result.

## 5. What is claimed, and what is not

**Claimed.** Genetic programming, run under a pre-registered protocol, searches
over recursive update rules and returns a short, readable expression over named
physical quantities, with its skill measured against a single-feature floor, a
formula-search baseline, Random Forest and RBF-SVM on identical splits.

**Not claimed, and please hold us to this:**

1. **The filter shape is supplied by us, not discovered.** 30% of the initial
   population is seeded with the affine recursion `s_t = a·s_{t-1} + innovation`.
   The search fills in which parameters enter, how they combine, and the
   retention coefficient. The decision to carry state forward at all is ours.
   The honest phrasing is "GP-discovered update rule, given the affine
   recursion form" - never "GP discovered a filter from nothing".
2. **No claim that this beats the SWAN-SF state of the art.** Published TSS on
   this benchmark ranges roughly 0.5-0.9 depending on preprocessing, and much
   of that spread is methodological rather than real.
3. **No claim of a positive result before the confirmatory run reports one.**
   As of this writing the shakedown suggests the searches tie the single-feature
   floor. If that holds, we say so.

## 6. Known limitations

- **Runtime.** The confirmatory run on full partitions takes hours - a first
  estimate of 8-12 minutes per seed proved roughly an order of magnitude wrong.
  The fold evaluates a tree 60 times per program per generation through a
  recursive Python walk. Compiling expressions to a flat array is the fix and
  is not yet done.
- **No per-seed checkpointing.** Results are written only at the end, so a
  crash loses the whole run.
- **Seed variance.** Per-seed success on the validation surrogate is about 60%.
  Runs below 5 seeds / 25 generations / population 800 are unreliable in both
  directions: at 2 seeds and 12 generations the same experiment once returned a
  number that looked like a refutation.
- **Missingness asymmetry.** Partition 2 shows roughly a sevenfold class
  difference in missing data (0.0017 vs 0.0121). Both are small in absolute
  terms and we expect little effect, but it is measured, not assumed away.
- **The single-feature floor is itself optimistic**, being the best of 288
  feature/sign combinations selected on validation. That is the conservative
  direction for our claim, and it is stated rather than hidden.

## 7. What would falsify this

- `verify.sh` fails, or the surrogate invariant tests fail - then the validation
  evidence in `docs/PREREGISTRATION.md` is void and must not be cited anywhere.
- The recursive rule does not beat the single-feature floor with separated
  confidence intervals - then H4 fails and the finding is that on this benchmark
  discovered composition buys nothing over one known parameter. **That is a
  reportable result and we will report it.**
- The discovered parameters do not recur across seeds - then H3 fails and what
  we have is five unrelated lucky guesses, not a discovery.

## 8. The question this cannot answer

If a rule survives all of the above, someone still has to say **why it is
physically sensible** - what it means that accumulating a particular product of
magnetic-field quantities predicts a flare. That is a solar physics judgement,
it belongs with Dr. Kempton and Prof. Angryk at Georgia State, and no amount of
compute substitutes for it. A discovered expression with no physical reading is
not a result worth publishing.

---

Everything here is reproducible from `git log`. Protocol and every amendment
are in `docs/PREREGISTRATION.md`, written before the data was touched and
amended only with dated entries that say what changed and why.
