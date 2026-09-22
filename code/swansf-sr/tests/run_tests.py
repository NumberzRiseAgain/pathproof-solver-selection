"""
Zero-dependency test runner.

Use pytest if it is installed:      python3 -m pytest tests -q
Otherwise this runs the same tests: python3 tests/run_tests.py

Kept dependency-free deliberately, so a reviewer can verify the suite on a
stock Python without installing anything beyond numpy, pandas and sklearn.
"""

import importlib.util
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ["test_metrics", "test_io_and_features", "test_search"]


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    passed = failed = 0
    failures = []
    t0 = time.time()

    for mod_name in MODULES:
        mod = load(mod_name)
        tests = sorted(n for n in dir(mod) if n.startswith("test_"))
        print(f"\n{mod_name}  ({len(tests)} tests)")
        for name in tests:
            fn = getattr(mod, name)
            try:
                fn()
            except Exception as exc:                       # noqa: BLE001
                failed += 1
                failures.append((mod_name, name, exc, traceback.format_exc()))
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            else:
                passed += 1
                print(f"  ok    {name}")

    dt = time.time() - t0
    print("\n" + "=" * 68)
    print(f"{passed} passed, {failed} failed in {dt:.1f}s")
    print("=" * 68)

    if failures:
        print("\nDetail:\n")
        for mod_name, name, _exc, tb in failures:
            print(f"--- {mod_name}.{name} ---\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
