#!/usr/bin/env python3
"""Run the pathproof test suite and write results/e1_tests.json.

Uses pytest when it is installed. When it is not (the Cowork VM has no network
and no pytest), a small shim supplies the four pytest names the suite uses
(fixture, approx, raises, importorskip is unused) and runs every test_*
function in tests/test_hazpath.py with module-scoped fixtures resolved by
name. The shim is a fallback for reproducibility on a bare machine; the count
it reports is the same 20 tests the pytest run reports.

Usage: python3 run_tests.py <pathproof dir> <results/e1_tests.json>
"""
import sys, os, json, subprocess, platform, importlib, inspect, traceback, math, datetime

pathproof, out = sys.argv[1], sys.argv[2]
result = {"artifact": "e1_tests", "what": "pathproof reference implementation, full test suite",
          "python": platform.python_version(), "run_utc": datetime.datetime.utcnow().isoformat() + "Z"}

def with_pytest():
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=pathproof,
                       capture_output=True, text=True)
    tail = (p.stdout.strip().splitlines() or [""])[-1]
    passed = failed = 0
    for tok in tail.replace(",", " ").split():
        pass
    import re
    m = re.search(r"(\d+) passed", tail); passed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) failed", tail); failed = int(m.group(1)) if m else 0
    return {"runner": "pytest", "passed": passed, "failed": failed, "summary": tail, "exit": p.returncode}

def with_shim():
    import types, contextlib
    shim = types.ModuleType("pytest")
    def fixture(*a, **k):
        def deco(f):
            f._is_fixture = True
            return f
        return deco(a[0]) if a and callable(a[0]) else deco
    class _Approx:
        def __init__(self, v, rel=1e-6, abs_=1e-12): self.v, self.rel, self.abs = v, rel, abs_
        def __eq__(self, o):
            return math.isclose(float(o), float(self.v), rel_tol=self.rel, abs_tol=self.abs)
        __req__ = __eq__
    def approx(v, rel=1e-6, abs=1e-12): return _Approx(v, rel, abs)
    @contextlib.contextmanager
    def raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"{exc.__name__} not raised")
    shim.fixture, shim.approx, shim.raises = fixture, approx, raises
    sys.modules["pytest"] = shim
    sys.path.insert(0, pathproof)
    mod = importlib.import_module("tests.test_hazpath")
    fixtures = {n: f for n, f in vars(mod).items() if getattr(f, "_is_fixture", False)}
    cache = {}
    def resolve(name):
        if name not in cache:
            f = fixtures[name]
            args = [resolve(a) for a in inspect.signature(f).parameters]
            cache[name] = f(*args)
        return cache[name]
    tests = [(n, f) for n, f in vars(mod).items() if n.startswith("test_") and callable(f)]
    passed, failed, failures = 0, 0, []
    for n, f in tests:
        try:
            f(*[resolve(a) for a in inspect.signature(f).parameters])
            passed += 1
        except Exception:
            failed += 1; failures.append({"test": n, "trace": traceback.format_exc()[-800:]})
    return {"runner": "shim (pytest not installed)", "passed": passed, "failed": failed,
            "summary": f"{passed} passed, {failed} failed of {len(tests)}", "failures": failures}

try:
    import pytest  # noqa
    result.update(with_pytest())
except ImportError:
    result.update(with_shim())

os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(result, open(out, "w"), indent=2)
print(f"e1_tests: {result['summary']} [{result['runner']}]")
sys.exit(0 if result["failed"] == 0 and result["passed"] > 0 else 1)
