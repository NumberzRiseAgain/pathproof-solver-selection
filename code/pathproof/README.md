# Transparent solution paths, path memory, and transfer — reference implementation

A runnable demonstration of the mechanism behind the SPEED DIAL feasibility
argument, on the synthetic HazMat fixture. Built so a developer can read it,
run it, break it and validate it in an afternoon.

```bash
python3 -m hazpath.demo          # the walkthrough
python3 -m pytest tests/ -q      # 19 tests
```

No dependencies beyond the standard library. No model is called.

---

## Read this before quoting anything from it

**This validates the mechanism. It measures no effect size.**

The fixture is 33 rows across four tables and six problems. There is no
statistical claim available here and none is made. The powered comparison —
four arms, held-out databases, McNemar, a success threshold declared in advance
— is specified in `PATH_TRANSFER_PREREGISTRATION.md` and runs on BIRD.

Anyone who puts a percentage from this directory into a proposal has misread it.

---

## What it demonstrates

**1. A solution is a readable program, not a SQL string.**
Twelve primitives, closed vocabulary, each individually executable and
inspectable. What a maintainer sees:

```
1. resolve_metric(name='ordered value')
2. select_source(table='supply_transactions')
3. bind_entity(column='nsn', value='8010-01-555-1234')
4. deduplicate(key='supply_id')
5. filter(column='fy', op='in', value=('FY24', 'FY25', 'FY26'))
6. derive_measure(name='ordered value', parts=[12 values, sum=537,600.00])
7. validate(check='non_negative')
```

**2. The wrong path does not raise. It returns a plausible number.**
The fixture repeats three `supply_id` values despite the schema documenting
them as unique. Drop step 4 and the same question returns **$672,000** against a
true **$537,600**, with every validation still passing. That is the failure
governance exists to catch, and `test_skipping_deduplicate_returns_a_confident_wrong_number`
pins it.

**3. Memory stores methods, not answers.**
Each record is *(problem fingerprint, path, observables, reward)*. The
fingerprint is structural — metric kind, entity classes, source classes,
temporal grain, and which guards the problem needs — and carries no literal
nouns, which `test_fingerprint_carries_no_literal_nouns` enforces. A memory
keyed on the question text would be a cache; this is not that.

**4. The freeze is real.**
Memory is populated from `hazmat_alpha`, frozen, then held-out problems over
`hazmat_beta` are scored. Writing after the freeze raises. The transfer claim
means nothing without this, so it is a runtime error and not a convention.

**5. Transfer changes the nouns and keeps the method.**
The held-out headline question is answered by retrieving the write-off method
and composing it with the disposal measure, reproducing **$639,450** exactly —
the figure in the deck. The system prints which prior problems supported the
choice and at what weight, so the *policy itself* is inspectable.

**6. A coverage gap produces a refusal, not a zero.**
Shop S06 is in the roster and in no transaction. `probe_coverage` detects it and
the system declines, naming the reason. Returning 0 would have been a confident
wrong answer, and the reward function scores an honest refusal above a
confident error so the policy never learns to guess.

**7. The vocabulary grows by promotion.**
Sequences that recur in accepted paths and beat the base accepted rate are
promoted into named composites, with the evidence that triggered each one. This
is the only mechanism here that composes new building blocks, and it composes
only verbs that already exist.

---

## Layout

```
hazpath/
  primitives.py   the twelve verbs, and a deterministic executor
  path.py         Path, Step, Observables, the reward function
  memory.py       Fingerprint, Record, PathMemory, the retrieval policy
  promote.py      composite promotion and its lift test
  questions.py    six problems, their fingerprints, reference answers
  solver.py       authored seed paths, and transfer to held-out problems
  demo.py         the walkthrough
  data/           the synthetic HazMat fixture, copied unmodified
tests/            19 tests, one per property the argument depends on
```

---

## For the developer validating this

Things worth trying, in order:

1. **Break the guard.** Delete step 4 from `ordered_value_fy24_26` in
   `solver.py`. The demo should report `672,000.00 ... WRONG` and the reward
   should go to −1.000. If it silently stays correct, the oracle is broken.
2. **Break the freeze.** Move `memory.freeze()` in `demo.py` to after the
   held-out section. `test_memory_refuses_writes_after_the_freeze` should fail.
3. **Leak a database.** Remove `exclude_databases` from the `solve_by_transfer`
   call. `test_retrieval_excludes_the_held_out_database` should fail. This is
   the exact leak the pre-registration's arm C is designed to detect at scale.
4. **Weaken the fingerprint.** Make `similarity` return a constant. Retrieval
   will start choosing irrelevant priors and the held-out answers will drift.
5. **Add a seventh problem** with a new fingerprint and no matching prior. The
   system should abstain with `no prior method matches`, not guess.

If any of 1–4 passes when it should fail, the harness is not measuring what it
claims and nothing built on it should go into a proposal.

---

## Provenance

The fixture in `hazpath/data/` is copied unmodified from the HazThread demo
build. It is **entirely synthetic** and carries seeded data-health defects on
purpose: three spellings of one state, three duplicated transaction ids, one
orphan shop code, and one shop with no transactions. Its own README documents
those seeds. No customer data is present in this directory and none should be
added to it.
