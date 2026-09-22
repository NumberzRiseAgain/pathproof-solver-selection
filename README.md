# 08_Analysis — the SPEED DIAL evidence pack

Built 5 September 2026 for the Release 5 rebuild of Volume 2 (`04_Volumes/latex_v2/`).
Every number Part One quotes from the Numberz.ai side comes out of a JSON under
`results/`, and every result file is hashed by `evidence_pack.sh` into
`results/_result_digests.txt` with a row in `RUN_RECORD.md`. The written claim the volume
cites is `06_Evidence/SPEEDDIAL_FEASIBILITY_EVIDENCE.md`, which traces each number to the
file and key it came from.

## What the study is

Three pieces of evidence, two of them re-executed here and one recorded:

| Result file | What it is | Re-run here? |
|---|---|---|
| `results/e1_tests.json` | The reference implementation `code/pathproof` (transparent solution paths over a closed vocabulary of twelve primitives, a path memory keyed on structural fingerprints, reward-weighted retrieval, composite promotion under four registered conditions) and its **20 tests** | yes, first |
| `results/e2_demo.json` | The walkthrough `python3 -m hazpath.demo` on the synthetic HazMat fixture, compared byte for byte with the `DEMO_OUTPUT.txt` shipped in the locked commit: the seven-step path that answers **537,600** with the deduplicate guard, the same question answering **672,000** (25% high, every validation passing, reward −1.000) with the guard omitted, promotion **refusing** to mint a composite on five records, the freeze, and the two held-out problems (**639,450** by method transfer; a **refusal** on shop S06, which has no rows) | yes |
| `results/e3_devtest.json` | The developer test of **1 September 2026** on the live interpretDB platform: four arms, three questions, Postgres, the same 33-row fixture. Arm A (raw model, schema only) 0 of 3; B (platform, context off) 0 of 3; C (declared context on, memory empty) 3 of 3; D (memory seeded) 3 of 3 | **no** — a hand-run test; transcribed from the report in `recorded_run_2026-09-01/` and hashed |
| `results/e4_swansf.json` | The SWAN-SF genetic-programming study `code/swansf-sr` (pre-registered; synthetic surrogate only; not run on real partitions). Its own tests and shipped result files, hashed | its tests only; optional |
| `results/manifest.json` | Every input hashed, the run identifier, and the pre-registration **lock check**: `PATH_TRANSFER_PREREGISTRATION.md` SHA-256 `06bc8e0a…40b8`, commit `42a70b7d`, HEAD `9abeddc` | yes |

## The run of record

`RUN_RECORD.md` carries three rows for 6 September 2026 (UTC; the evening of 5 September in
Georgia). Row 1 (`run_20260906T003533Z`, identifier `eb8bf74abd2ba9bf`) hashed the walkthrough's
by-product `code/pathproof/memory_dump.json` into the manifest; that file carries wall-clock
latencies, so the identifier would have moved on every run. `manifest.py` now excludes it. Rows 2
and 3 (`run_20260906T003955Z`, `run_20260906T004000Z`) both give **`6fe38d1f66c26212`** over the
same 66 input files: the identifier is a function of the sources alone. Volume 2 cites that
identifier. The by-product is moved to `_to_delete/` after a run on this mount, which refuses
deletes; on any other machine `reproduce.sh` removes it.

## How to run it

```bash
# from the pursuit folder, the house way (logs, digests, run record):
../_Shared_Templates/scripts/evidence_pack.sh .
# or directly:
sh 08_Analysis/code/reproduce.sh
```

Standard library only; no network, no GPU, no model call. `pytest` is used when
installed, otherwise `run_tests.py` runs the same 20 tests under a small shim. Runs in
under two seconds. `code/pathproof` is the unpacked `06_Evidence/pathproof.tar.gz` with its
git history (the lock commit `42a70b7d` and the recording commit `9abeddc`); if it is ever
missing, unpack that tarball here.

## What it does not establish

- **No effect size.** The fixture is 33 rows and six problems. There is no statistical claim
  available here and none is made; anyone who puts a percentage from `e2_demo.json` into a
  volume has misread it. The powered comparison is the pre-registered transfer experiment,
  which **has not run**.
- **No transfer result.** The held-out answers in the walkthrough show the mechanism
  (retrieve a method, change the nouns) on two problems. Arm D of the developer test re-used
  the questions Arm C had answered; it exercises reuse and says nothing about transfer.
- **Not the 65 → 73 percent comparison.** The controlled execution-accuracy comparison on a
  500-case split is quoted in Part One with its run logs attached to Volume 5; those logs are
  not in this pack, and Part One marks the figure `\confirm{500-case run logs to Volume 5}`
  until they are.
- **Not the 0 of 400 false alarms.** That number is the ARRESTLINE diagnostics codebase's,
  hashed in `../DON26BZ05-DV087_ARRESTLINE/08_Analysis/results/prior_runs_2026-08/_prior_run_digests.txt`
  and carried forward there; this pack does not re-run it.
- **Nothing about the discovery half.** Item 1 of the feasibility determination (a discovered
  novel algorithm beating a named baseline) is the research institution's and is open.
- **Synthetic data only.** No customer data is in this folder and none may be added.

## Ownership

Company-funded, performed in house by Numberz.ai personnel, on Numberz.ai's own
reference implementation and platform. No federal SBIR or STTR funds. The pre-registration,
the reference implementation with its tests and the developer test report are filed in
Volume 5; the STTR Allocation of Rights agreement with GSU governs the discovery side.

## Repository

Public copy of the `08_Analysis/` evidence pack behind the SPEEDDIAL submission to DPA26TZ05-DV003, released 19 September 2026 by Numberz.ai Inc. under the MIT License (`LICENSE`). The study was company-funded; no Government funding. No third-party data is redistributed. swansf-sr was never run on real SWAN-SF partitions; the synthetic surrogate it uses is generated by the code. Run identifiers are the first 16 hex digits of a SHA-256 over the source digests, so the tree as committed reproduces the identifier in `results/manifest.json`; `RUN_RECORD.md` lists every run, including the ones that failed. Cite with `CITATION.cff`. The other packs from the same month are listed at https://github.com/NumberzRiseAgain/september-2026-evidence-packs.
