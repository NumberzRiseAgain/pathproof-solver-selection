# Pre-registration: experience-weighted transfer of transparent solution paths

**Status:** declared before any test-split result is observed.
**Registered by:** Raghu Venkat, Co-Founder & CPTO, Numberz.ai Inc.
**Date fixed:** 1 September 2026
**Supports:** DPA26TZ05-DV003 (SPEED DIAL) Direct-to-Phase-II feasibility, items 2 and 3.
**Integrity:** the git commit and the SHA-256 of this file are recorded in
`LOCK.txt`, written at the moment of locking. The hash cannot appear inside the
document it hashes, which is why it lives in a sidecar. Verify with
`shasum -a 256 PATH_TRANSFER_PREREGISTRATION.md` and compare.

**This document is locked. Do not edit it after the first test record is read.**
A correction discovered later is recorded as a dated addendum below the lock
line, never as an edit to the text above it.

Everything in this document is written before the held-out split is scored.
Nothing below has been informed by looking at a test-set result, which is the
only reason it is worth writing down. The corresponding artifact for our prior
work is `PROGNOSIS_STOPPING_RULE.md`, where a pre-registered criterion led us to
**withdraw** a claim; that file and this one are offered together, because a
stopping rule only means something if it has been honoured at least once when it
cost something.

---

## 1. The claim under test

We assert, and will test, exactly one thing:

> **H1.** A persistent memory of *(problem fingerprint, solution path, observables,
> reward)* records, used to bias path selection on **new problems over databases
> the memory has never contained**, produces a higher execution-accuracy rate
> than the same system with the memory disabled.

H1 is a claim about **transfer**, not about recall. The null hypothesis that
matters is not "memory does nothing"; it is "memory only helps by returning
answers it has already seen." Arm C below exists to make that null testable
rather than assumed.

We do **not** claim, and this experiment cannot support:

- discovery of a novel *algorithm* in the sense of the topic's items 1 (new
  scientific or numerical methods). What is composed here is a new *program over
  a fixed primitive vocabulary*, and where the vocabulary itself grows, it grows
  by promotion of observed sequences (§7), not by invention;
- any comparison against a text-to-SQL leaderboard. Absolute accuracy figures
  from this experiment are **not** state-of-the-art claims and must never appear
  in the same paragraph as the phrase. Every number here is an architecture
  delta with the underlying model held fixed.

---

## 2. The computational object

A solution is represented as a **path**: an ordered sequence of named primitives
drawn from a closed vocabulary,

```
V = { resolve_metric, select_source, bind_entity, discover_join,
      normalize_identifier, deduplicate, filter, aggregate,
      derive_measure, probe_coverage, validate, abstain }
```

each carrying typed arguments and each individually executable and inspectable.
A path is therefore a program a domain expert can inspect line by line. This
demonstrates the transparent computational representation and provenance
mechanism needed to support feasibility item 3; the proposing team separately
provides prior evidence of interpretable **discovered algorithms**, which is
what item 3 asks for in the topic's own terms. The two are offered together and
neither is asked to carry the other.

**Formally.** For a problem $x$ the system emits a path
$\pi = (v_1, a_1), \ldots, (v_k, a_k)$ with $v_i \in V$. Execution
$E(\pi, D)$ against store $D$ is deterministic in the sense that matters:
**canonicalised execution results must be identical**, where canonicalisation
normalises column ordering, row ordering wherever order is not semantically
specified, numeric representation and null encoding. Byte equality of raw
results would be a claim about database serialisation rather than about the
semantics of the path. This determinism is a precondition of the experiment and
is asserted by test (§9), not assumed.

---

## 3. The learning mechanism, named accurately

We describe the mechanism as a **contextual bandit with reward-weighted path
retrieval**. We deliberately do **not** call it reinforcement learning. There is
no sequential credit assignment across a trajectory of environment states, no
value function, and no policy gradient through a model. Calling it RL would
overstate it, and a reviewer who checks would be right to discount everything
else in the volume.

| Element | Definition in this system |
|---|---|
| Context $c(x)$ | Problem fingerprint: resolved metric type, entity classes, required source classes, temporal grain, constraint set. Computed by a function fixed before the split (§5). **Database-independent by construction — see §5.2** |
| Action | A candidate path $\pi$ over $V$ |
| Utility $u$ | $u_m = \mathbb{1}[\text{execution result} \equiv \text{reference}] \cdot w_c - \lambda_1 \cdot \text{tokens} - \lambda_2 \cdot \text{latency}$. **Signed**; a wrong answer scores below zero |
| Policy $\rho$ | See §3.1. Retrieval weights are non-negative by construction |
| Update | Append $(c, \pi, o, u)$ to $M$. No model weights change. The learning is entirely in a small transparent layer over known primitives |

### 3.1 The policy, with non-negative weights

The utility $u_m$ is signed, so it cannot be used directly as a retrieval
weight: a normalised sum containing negative terms is not a probability
distribution. The memory is therefore **partitioned by the sign of the
utility**, and successes and failures act through two different mechanisms.

Let $M^{+} = \{m : u_m > 0\}$ and $M^{-} = \{m : u_m \le 0\}$.

**Successes bias selection.** Over candidate methods $\pi$,

$$\rho(\pi \mid c) \;=\; \frac{\sum_{m \in M^{+},\ \pi_m \simeq \pi} \mathrm{sim}(c, c_m)\, u_m}
{\sum_{m \in M^{+}} \mathrm{sim}(c, c_m)\, u_m}$$

Every term is non-negative, the denominator is the same sum over all retained
methods, and $\rho$ is a proper distribution over the distinct method classes
in $M^{+}$. Where the denominator is zero, no prior applies and the system
falls through to proposal without a prior, which is recorded as such.

**Failures suppress retries.** $M^{-}$ is an *exclusion memory*, not a
negatively weighted term. A candidate $\pi$ is removed from the candidate set
when there exists $m \in M^{-}$ with $\pi_m \simeq \pi$ and
$\mathrm{sim}(c, c_m) \ge \tau_{\text{excl}} = 0.60$. A method that produced a
wrong answer on a sufficiently similar problem is not ranked low; it is not
offered.

Keeping the two apart is what makes the mechanism explicable in one sentence:
*successful prior paths bias selection; failed paths suppress materially
equivalent retries.*

**Why this framing is stronger than RL for this proposal.** The policy is
inspectable: for any decision the system can name which prior problems, at which
similarity, with which observed rewards, produced the bias. A policy gradient
cannot do that, and interpretability is a scored feasibility item.

---

## 4. Arms

All four arms use an **identical model and model version, prompt template and
decoding parameters, with the same seed where the provider supports one**.
Request and response artifacts are retained for every call. A seed does not make
a hosted model deterministic, so this is stated as reproducibility of the
configuration and not of the sampling. **Any provider-side model-version change
during the run invalidates the paired comparison**, and the run restarts. The
only intended variable is architecture.

| Arm | Architecture | Purpose |
|---|---|---|
| **A** | Model-only text-to-SQL. No grounding, no validation, no memory | Floor |
| **B** | A + problem grounding (schema, entity binding, join discovery) + deterministic validation | The comparison that matters |
| **C** | B + **exact** verified-query cache, keyed on `(database/schema signature, normalized question)` | **Predicted-null control** |
| **D** | B + reward-weighted path memory with similarity retrieval across problems | The claim |

**Arm C is a control and its result is predicted in advance.** The cache key
includes the schema signature deliberately: the same question text against two
different databases requires different SQL, so a cache keyed on text alone
would be unsound as an engineering artifact and would muddy the control. With
the schema in the key, C **cannot** fire on the held-out split, because no
held-out database signature exists anywhere in the cache. We therefore predict
**C = B on the held-out split, to within ±1 record.** Any gap larger than that
is not a finding; it is evidence of leakage, and the run is void until the leak
is found and the split is rebuilt. C is the instrument that certifies the
harness, and reporting it is not optional.

---

## 5. What is fixed before the split is scored

Fixed now, in this document, and recorded in code with a commit hash before the
first test record is read:

1. **The primitive vocabulary $V$** (§2). No primitive is added after scoring
   begins, except by promotion under the rule in §7, which operates on the
   training split only.
2. **The fingerprint function $c(\cdot)$** (§5.2). It is **not** fitted, tuned
   or selected on any split.
3. **The utility weights** $w_c = 1.0$, $\lambda_1 = 10^{-6}$ per token,
   $\lambda_2 = 10^{-3}$ per second. These make correctness dominant and cost a
   tie-breaker. They are not swept.
4. **The similarity function** (§5.3), the **method-equivalence relation**
   $\pi_m \simeq \pi$ (§5.4), the retrieval depth $k = 5$ and the exclusion
   threshold $\tau_{\text{excl}} = 0.60$.
5. **The acceptance oracle** (§6).
6. **The promotion rule and its four conditions** (§7).
7. **The success criterion** (§8).

### 5.2 The fingerprint is database-independent

> The fingerprint contains no database identifier, table name, column name,
> literal schema identifier, or other feature from which database identity can
> be recovered. Database-specific names are normalized to semantic classes
> before fingerprinting.

Without this sentence a reviewer can reasonably ask whether the "transfer" we
measure is lexical similarity between schema vocabularies. With it, the question
is closed by construction and by test: the released implementation carries
`test_fingerprint_carries_no_literal_nouns`, which fails if any literal from the
fixture reaches the fingerprint.

The fingerprint is the tuple

$$c(x) = \big(\text{metric kind},\ \text{entity classes},\ \text{source classes},\
\text{temporal grain},\ \text{needs join},\ \text{needs normalization},\
\text{needs dedup}\big)$$

every element of which is a semantic class drawn from a fixed ontology.

### 5.3 The similarity function, stated

$$\mathrm{sim}(c, c') = \frac{|F(c) \cap F(c')|}{|F(c) \cup F(c')|}$$

the Jaccard index over feature sets $F(\cdot)$, where $F$ emits one token per
categorical field of the fingerprint plus one token per entity class and per
source class. Unweighted, unparameterised, and therefore not tunable toward a
result. Range $[0, 1]$; reflexive at 1.

### 5.4 Method equivalence, $\pi_m \simeq \pi$, defined

> Two paths are **materially equivalent** when their ordered primitive sequence
> is identical after every typed argument that refers to a database-specific
> identifier is replaced by its semantic argument class.

So `select_source(disposal_dtid)` and `select_source(writeoff_fact)` both reduce
to `select_source(source=disposal_event)` where the ontology says they are the
same class, and stay distinct where it does not. Literal values — material
codes, organisation ids, fiscal years — are dropped entirely, because they are
the nouns and the nouns are exactly what must not carry across databases.

The relation is implemented as `Path.material_signature` in the released code
and is fixed with the rest of this document. Leaving it undefined would have
been an unregistered degree of freedom, and the temptation to settle it after
seeing which definition helps is precisely what pre-registration is for.

---

## 6. The oracle, stated precisely

Offline, the oracle is **execution-result equivalence against the reference
query**: multiset equality of returned rows after column-order normalization and
numeric tolerance $10^{-6}$. It is not string equality of SQL, and it is not a
model judging another model.

**User acceptance is the deployed oracle and is not used here.** The fielded
system takes analyst approval or correction as reward. That signal is not
available at benchmark scale, its absence is a limitation of this experiment,
and substituting a model-graded proxy for it would make the reward circular.
This paragraph exists so that no reviewer has to ask where the labels came from.

---

## 7. Composite promotion

A contiguous subsequence $s$ of primitives is promoted to a named composite
primitive when, **on the training split only**, all four of the following hold:

1. $s$ occurs in at least $n_{\min} = 8$ **accepted** paths;
2. the accepted-path rate of paths containing $s$ exceeds that of **matched**
   paths — same fingerprint class, $s$ absent — by at least
   $\delta_{\min} = 10$ percentage points, in absolute terms;
3. promotion does not degrade the accepted rate on an internal validation slice
   of the training records, taken as every fifth record rather than as a tail;
4. the composite pays for itself: replacing $s$ with a single token removes
   $|s| - 1$ steps from every future path that uses it, the composite expands
   back to exactly $s$ so the computation is unchanged, and at least two
   distinct accepted paths carry it.

with $2 \le |s| \le 5$. Promotion is logged with the evidence that triggered it.
Promoted composites are frozen with the rest of the memory before the test split
is scored.

**Condition 2 was tightened deliberately.** "Exceeds the rate without it" is too
weak: eight observations and a one-point accidental advantage would mint a
building block. A fixed effect size makes that impossible to do by luck.

**A note on condition 4 as implemented.** The literal reading — that paths
containing $s$ should be shorter than paths without it — is wrong for a
correctness guard. Adding a deduplication step makes a path exactly one step
*longer*, and that step is the reason the answer is right; under that test no
guard could ever be promoted, which inverts the intent. What promotion actually
buys is the $|s| - 1$ steps removed from every future use, and that is what is
checked, together with lossless expansion. The change is recorded here rather
than made silently.

**Condition 3 is taken as a stride, not a tail.** A contiguous final fifth is
not representative when training records arrive in any order; seed the successes
first and the tail is all failures, which makes the condition unsatisfiable for
reasons unrelated to the candidate.

This is the one mechanism in the system by which the *vocabulary itself* grows
with experience. We report the count of promotions and the reuse rate of
promoted composites on held-out problems.

**If zero composites are promoted, we report zero and make no item-1 argument
from this experiment.** And even when composites are promoted, this is
**supporting** evidence that the system learns reusable higher-level building
blocks. It does not demonstrate prior discovery of a novel scientific algorithm
outperforming the state of the art, the disclaimer in §1 stands unchanged, and
proposal-writing pressure is not a reason to weaken it.

---

## 8. Split, endpoint and success criterion

### Split

The split is **by database, not by query**. Every database appearing in any
memory record is excluded from the test set entirely. A query-level split is
insufficient and we will not use one: schema-specific join paths and identifier
normalizations learned on a database transfer to other questions over that same
database, and a gain measured that way is recall wearing the costume of transfer.

Memory is populated from the training databases, then **frozen**. The freeze
point is a commit. No record is added, reweighted or removed while the test
split is scored.

### Primary endpoint

Execution accuracy of **D minus B** on held-out databases, paired by query,
tested with the **exact McNemar test** on discordant pairs, two-sided,
$\alpha = 0.05$.

### Powered criterion

Monte-Carlo power under the paired design, assuming a base rate near 0.73 and
10% symmetric discordance between arms:

| Held-out queries | +3 pts | +4 pts | +5 pts | +6 pts |
|---|---|---|---|---|
| 300 | 0.25 | 0.40 | 0.55 | 0.71 |
| **500** | 0.42 | **0.63** | **0.81** | 0.91 |

Therefore:

> **Success is declared if and only if D exceeds B by at least 5 percentage
> points of execution accuracy on at least 500 held-out queries drawn from
> databases absent from memory, with McNemar $p < 0.05$, and C is within
> ±1 record of B.**

Five points is chosen because it is the smallest effect this design can detect
with power above 0.8. It is not chosen because it is attainable.

### Secondary endpoints, reported whatever the primary shows

- path reuse rate: fraction of held-out problems whose selected path was
  materially derived from a retrieved prior path;
- transfer distance: similarity between the held-out fingerprint and the prior
  it was biased by;
- composites promoted, and their held-out reuse rate;
- corrections required, tokens, latency;
- abstention rate and its precision.

---

## 9. What will not be done

Stated in advance because each is a way this result could be manufactured.

- **The success threshold will not be lowered after seeing a number.** Five
  points is five points.
- **The split will not be redrawn.** The database partition is fixed by seed
  before scoring. A disappointing first split is a result, not a reason to
  reshuffle.
- **No database will be dropped**, including any on which every arm does badly.
- **Reward weights, retrieval depth $k$, and $n_{\min}$ will not be swept.**
  Sweeping them and reporting the best is fitting, and it is the specific
  failure this document exists to prevent.
- **Arm C will be reported even when it is boring**, because a boring C is the
  evidence that the transfer in D is real.
- **Determinism is asserted by test**: every path is executed twice and
  equality of the *canonicalised* results is required. A non-deterministic
  executor invalidates the reward and the run stops.

**Legitimate to change, and disclosed as a change if it happens:** a defect in
the harness found before scoring begins; an oracle bug that misjudges result
equivalence; the addition of a database to the *training* pool. Each is recorded
with a date and a reason in the run log.

---

## 10. Disposition

| Outcome | What the volume says |
|---|---|
| D beats B by ≥5 pts, $p<0.05$, C ≈ B | Feasibility item 2 is evidenced by a controlled architecture delta with the model held constant, and the mechanism is experience-weighted transfer of inspectable paths. Item 3 is evidenced independently by the path representation |
| D beats B but below 5 pts or $p \ge 0.05$ | **No transfer claim.** We report the observed delta with its interval and state that the design was powered for 5 points and did not reach it. Item 2 rests on the prior controlled result; item 3 is unaffected |
| D ≈ B | We report that experience-weighted retrieval did not transfer across databases in this design, and say so in the volume. Item 3 is unaffected and is the argument we make |
| C ≠ B | Run void, leak reported, split rebuilt, experiment re-run once. The second run is disclosed as a second run |

**Item 3 does not depend on this experiment.** The transparent path
representation, deterministic execution and stored provenance already exist and
are demonstrated by the reference implementation released with this
document. That argument is written and submitted whatever the primary endpoint
shows. This paragraph is here so that no reader — and no author — mistakes a
failed transfer result for a failed feasibility case.

---

## 11. Threats to validity, acknowledged now

1. **Benchmark ceiling.** Absolute accuracy on any public text-to-SQL benchmark
   is not comparable to production performance on a governed enterprise store,
   in either direction.
2. **Single model.** Holding the model fixed is what makes the delta clean; it
   also means the result is a statement about one model's behaviour under this
   architecture.
3. **Fingerprint expressiveness.** If $c(\cdot)$ is too coarse, retrieval will
   bias toward irrelevant priors and D will underperform for a reason that is
   about the fingerprint and not about the hypothesis. We report the transfer
   distance distribution so this failure mode is visible.
4. **Homogeneous stores.** Public benchmarks are single-engine. The
   cross-engine case — a relational store and a document store with no shared
   schema and no foreign keys — is where the join-discovery primitive earns its
   place, and it is **not** measured by this experiment.
5. **Reward is offline.** See §6.
