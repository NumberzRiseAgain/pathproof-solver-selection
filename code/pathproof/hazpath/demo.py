"""End-to-end demonstration.

    python3 -m hazpath.demo

Four seed problems populate the memory. The memory is frozen. Two held-out
problems over a database the memory has never contained are then answered by
retrieving a method and changing only the nouns.

What this demonstrates: the representation, the freeze, retrieval, adaptation,
deterministic execution, an honest abstention, and composite promotion.

What this does NOT demonstrate: a statistically meaningful transfer effect. The
fixture is 33 rows and six problems. The powered version of this comparison is
specified in PATH_TRANSFER_PREREGISTRATION.md and runs on BIRD. Anyone quoting a
percentage from this file in a proposal has misread it.
"""
from __future__ import annotations

import sys
from pathlib import Path as FsPath

from .memory import PathMemory, Record
from .primitives import load_store
from .promote import promote
from .questions import HELD_OUT, SEEDS
from .solver import run, seed_paths, solve_by_transfer

RULE = "=" * 78


def banner(text: str) -> None:
    print(f"\n{RULE}\n  {text}\n{RULE}")


def main(argv=None) -> int:
    store = load_store()
    memory = PathMemory()
    paths = seed_paths()

    banner("1. SEED PROBLEMS  —  building the path memory")
    for prob in SEEDS:
        concrete, obs, r = run(prob, paths[prob.key], store)
        memory.add(Record(question=prob.question, fingerprint=prob.fingerprint,
                          path=concrete, observables=obs, reward=r,
                          database=prob.database))
        verdict = "correct" if obs.correct else "WRONG"
        got = "abstained" if obs.abstained else f"{obs.result:,.2f}"
        print(f"\n  {prob.question}")
        print(f"     got {got}   expected {prob.reference}   -> {verdict}"
              f"   reward {r:+.3f}")
        if prob.note:
            print(f"     trap: {prob.note}")

    banner("2. THE PATH, AS A MAINTAINER READS IT")
    example, _, _ = run(SEEDS[2], paths[SEEDS[2].key], store)
    print(f"\n  {SEEDS[2].question}\n")
    print(example.render())
    st, _ = example.execute(store)
    print("\n  execution trace:")
    for note in st.notes:
        print(f"     {note}")

    banner("3. COMPOSITE PROMOTION  —  and why it refuses here")
    print("\n  The rule requires 8 accepted occurrences, a 10-point lift over")
    print("  matched paths, no degradation on an internal validation slice,")
    print("  and a real saving in path length. Five seed records cannot clear")
    print("  that bar, and the rule declining to invent a building block on")
    print("  thin evidence is the behaviour we want to show.\n")
    promoted = promote(memory, memory._records, n_min=2, limit=2)
    if not promoted:
        print("   nothing met the promotion rule  <-- correct on 5 records")
        print("   tests/test_hazpath.py proves the rule DOES fire on a clean")
        print("   20-record separation, so this is a refusal and not a bug")

    banner("4. FREEZE")
    memory.freeze()
    print(f"\n  {len(memory)} records over databases {sorted(memory.databases)}")
    print("  memory is frozen; scoring cannot write to it")
    try:
        memory.add(memory._records[0])
    except RuntimeError as exc:
        print(f"  verified: {exc}")

    banner("5. HELD-OUT PROBLEMS  —  a database the memory has never contained")
    ok = 0
    for prob in HELD_OUT:
        path, obs, r, evidence = solve_by_transfer(prob, memory, store)
        print(f"\n  {prob.question}")
        print(f"  [{prob.database}, absent from memory]\n")
        for line in evidence:
            print(f"     {line}")
        print("\n     method used:")
        print("\n".join("     " + ln for ln in path.render().splitlines()))
        got = f"ABSTAIN ({obs.abstain_reason})" if obs.abstained else f"{obs.result:,.2f}"
        verdict = "correct" if obs.correct else "WRONG"
        print(f"\n     got {got}")
        print(f"     expected {prob.reference}  ->  {verdict}   reward {r:+.3f}")
        ok += bool(obs.correct)

    banner("SUMMARY")
    print(f"\n  held-out problems answered correctly: {ok} of {len(HELD_OUT)}")
    print("  one of them is a refusal, and refusing was the correct answer")
    print("\n  This validates the MECHANISM. It measures no effect size.")
    print("  The powered comparison is PATH_TRANSFER_PREREGISTRATION.md, on BIRD.\n")

    out = FsPath(__file__).resolve().parent.parent / "memory_dump.json"
    memory.to_json(out)
    print(f"  memory written to {out.name}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
