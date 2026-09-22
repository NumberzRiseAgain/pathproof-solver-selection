# SWAN-SF symbolic regression

Genetic programming for closed-form, human-readable solar flare predictors on
the SWAN-SF benchmark.

Two purposes, in this order:

1. **A paper.** Interpretable algorithm discovery on a benchmark the community
   already trusts, with a pre-registered protocol. Target venue EuroGP 2027
   (deadline 1 November 2026), extended version to GPEM or MLST.
2. **Documented feasibility evidence** for DARPA STTR topic DPA26TZ05-DV003
   (SPEED DIAL), whose Direct-to-Phase-II bar asks for exactly three things:
   an algorithm discovered by GP-class methods that outperforms existing
   approaches, a quantitative margin, and transparent building blocks a domain
   expert can read. One experiment speaks to all three.

Nothing here is backdated or framed as older than it is. The git history is
the date record.

---

## Status

- [x] Pipeline built and validated end to end on a labelled synthetic surrogate
- [x] Protocol pre-registered (`docs/PREREGISTRATION.md`) before any real data
- [x] Compositional validation: the search recovers a planted product structure
- [x] SWAN-SF partitions downloaded and loader validated against real filenames
- [x] Test suite (39 tests) and independent-verification brief (`docs/FOR_REVIEWER.md`)
- [ ] Confirmatory run, partition 1 -> partition 2 (in progress, ~15 h)

- [ ] Secondary partition pairs, seed-stability analysis
- [ ] Technical report → arXiv → EuroGP

## Getting the data

The cloud workspace this was built in cannot reach Harvard Dataverse (egress
allowlist), so the partitions have to be fetched on a machine that can.

```bash
# ~10 GB, five partition zips. Any machine with normal internet.
mkdir -p SWAN-SF && cd SWAN-SF
curl -L -o swansf.zip \
  "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/EBCFKM"
unzip swansf.zip && unzip -o 'partition*.zip'
```

Expected layout after unzipping:

```
SWAN-SF/
  partition1/  *.csv     partition2/  *.csv     ...  partition5/
```

The loader is tolerant of the exact folder naming and will glob for
`*artition*<k>*` if `partition<k>` is not found.

## Independent verification

A reviewer who was not present should start with `docs/FOR_REVIEWER.md` — why
this exists, what is claimed, what is explicitly NOT claimed, and how to read
the output. Then:

```bash
python3 tests/run_tests.py     # 39 tests, ~5 seconds, no dataset needed
bash verify.sh                 # the three synthetic validations, ~10 minutes
```

`verify.sh` is the part worth checking independently: it tests the method
without needing the 10 GB dataset, and it is where the claims in
`docs/PREREGISTRATION.md` are actually earned.

## Running

```bash
cd src
pip install numpy pandas scikit-learn gplearn sympy

# pipeline validation, no real data needed
python3 run_experiment.py --synthetic --seeds 5 --generations 30 --population 3000

# the confirmatory run
python3 run_experiment.py --data /path/to/SWAN-SF --train 1 --test 2 \
    --seeds 5 --generations 30 --population 3000 --out ../results/p1_p2.json
```

Runtime scales with `population × generations × seeds`. On two cores the full
budget takes roughly half an hour; on an 8-core laptop, a few minutes.

`--limit N` caps slices per partition for a fast shakedown against real data
before committing to the full run.

## Layout

```
src/swan_io.py         loaders: real SWAN-SF, and the synthetic surrogate
src/features.py        window -> descriptors, train-only scaling
src/metrics.py         TSS, HSS2, threshold selection, bootstrap CIs
src/gp_search.py       the GP search and multi-seed driver
src/run_experiment.py  the pre-registered protocol, end to end
docs/PREREGISTRATION.md
results/               one JSON per run, every run kept
```

## Things that will bite

- **Scaling leakage.** Statistics are fit on inner-train rows only. This is the
  single most common reason SWAN-SF numbers do not reproduce.
- **Threshold leakage.** The decision threshold is chosen on the inner
  validation fold, never on test. Choosing it on test inflates TSS a lot.
- **Orientation.** The GP fitness is direction-agnostic, so an expression may
  score low-means-flaring. Orientation is resolved on validation and frozen.
  Missing this silently produces TSS 0.
- **Seed spread.** GP is stochastic. All five seeds' validation scores are
  published; reporting best-of-n without saying n is overstating.
- **Class imbalance.** Roughly 4-5% positives for the ≥M1.0 task. Accuracy is
  meaningless here; that is why the headline is TSS.

## Provenance

SWAN-SF: Angryk, R. A., Martens, P. C., Aydin, B., Kempton, D., et al.,
*Multivariate time series dataset for space weather data analytics*,
Scientific Data 7, 227 (2020). `doi:10.7910/DVN/EBCFKM`.
