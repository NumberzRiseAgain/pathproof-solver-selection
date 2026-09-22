#!/usr/bin/env python3
"""Write results/manifest.json: the inputs of this evidence pack, hashed, plus the
pre-registration lock check.

Usage: python3 manifest.py <08_Analysis dir> <results/manifest.json>

Lock check (LOCK.txt in the reference implementation, 1 September 2026):
  PATH_TRANSFER_PREREGISTRATION.md  SHA-256 06bc8e0af19402bc1094179e3a99513b04a4765710f540be9668b80a6dfe40b8
  git commit 42a70b7d4478cbf0c7667be58b09c5716b76b55f (document + implementation + tests)
  git HEAD   9abeddc (records the lock)
The check fails hard if the document hash has moved: the pre-registration is
only worth quoting while the text above its lock line is the text that was locked.
"""
import sys, os, json, hashlib, subprocess, datetime

root, out = sys.argv[1], sys.argv[2]
LOCK_SHA = "06bc8e0af19402bc1094179e3a99513b04a4765710f540be9668b80a6dfe40b8"
LOCK_COMMIT = "42a70b7d4478cbf0c7667be58b09c5716b76b55f"
SKIP = {".git", "__pycache__", ".pytest_cache"}
# By-products of a run are not sources. memory_dump.json is written by the walkthrough
# and carries wall-clock latencies, so hashing it would change the run identifier on
# every run; it is excluded here and removed by reproduce.sh where the filesystem allows.
SKIP_FILES = {"memory_dump.json"}

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def walk(sub):
    base = os.path.join(root, sub)
    files = {}
    if not os.path.isdir(base):
        return files
    for d, dirs, fs in os.walk(base):
        dirs[:] = [x for x in dirs if x not in SKIP]
        for f in sorted(fs):
            if f in SKIP_FILES:
                continue
            p = os.path.join(d, f)
            files[os.path.relpath(p, root)] = sha(p)
    return files

pp = os.path.join(root, "code", "pathproof")
prereg = os.path.join(pp, "PATH_TRANSFER_PREREGISTRATION.md")
prereg_sha = sha(prereg) if os.path.exists(prereg) else None
def git(*args):
    try:
        return subprocess.run(["git", "-C", pp, *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:  # git absent or not a repository
        return f"unavailable: {e.__class__.__name__}"
head = git("rev-parse", "HEAD")
has_lock_commit = git("cat-file", "-t", LOCK_COMMIT) == "commit"

inputs = {
    "code/reproduce.sh and runners": {k: v for k, v in walk("code").items() if k.count("/") == 1},
    "code/pathproof": walk("code/pathproof"),
    "code/swansf-sr": walk("code/swansf-sr"),
    "recorded_run_2026-09-01": walk("recorded_run_2026-09-01"),
}
# The study is deterministic (no seed, no sampling): the run identifier is the digest of
# the input digests, so the same sources always produce the same identifier.
run_id = hashlib.sha256("\n".join(f"{k} {v}" for grp in inputs.values() for k, v in sorted(grp.items())).encode()).hexdigest()[:16]
manifest = {
    "artifact": "manifest",
    "run_identifier": run_id,
    "seed": None,
    "deterministic": "no randomness anywhere in the reference implementation; the demo output is byte-stable",
    "written_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "python": sys.version.split()[0],
    "platform": __import__("platform").platform(),
    "lock": {"document": "code/pathproof/PATH_TRANSFER_PREREGISTRATION.md", "expected_sha256": LOCK_SHA,
             "observed_sha256": prereg_sha, "matches": prereg_sha == LOCK_SHA,
             "expected_commit": LOCK_COMMIT, "commit_present": has_lock_commit, "git_head": head},
    "inputs": inputs,
}
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(manifest, open(out, "w"), indent=2)
n = sum(len(v) for v in manifest["inputs"].values())
print(f"manifest: run {run_id}; {n} input files hashed; lock {'MATCHES' if manifest['lock']['matches'] else 'MISMATCH'} "
      f"({(prereg_sha or 'missing')[:12]}...); git HEAD {head[:7]}; lock commit present: {has_lock_commit}")
sys.exit(0 if manifest["lock"]["matches"] else 1)
