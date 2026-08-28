"""Shared helpers for the NucLogic test scripts.

Each test is a standalone script: run it directly, or run them all with
`python tests/run_all.py`. Tests that need real microscopy files look the path
up here and skip cleanly when it is not available, so the suite still runs on a
machine without the sample data.

Paths come from an environment variable, or from a local (gitignored)
tests/local_data.py holding the same names:

    NUCLOGIC_TEST_IMS          a multi-timepoint .ims
    NUCLOGIC_TEST_IMS_CROPPED  a _cropped.ims written by the pipeline
    NUCLOGIC_TEST_ND2          any Nikon .nd2
    NUCLOGIC_TEST_LIF          a Leica .lif holding several scenes
    NUCLOGIC_TEST_ND           a MetaMorph .nd, beside its .STK stacks
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# The pixi environment needs its DLL directories on PATH before h5py or torch
# load; running the app through `pixi run` does this, a bare python.exe does not.
_ENV = os.path.join(PROJECT_ROOT, ".pixi", "envs", "default")
if os.path.isdir(_ENV):
    for _sub in ("", "Library/bin", "Library/usr/bin", "Library/mingw-w64/bin",
                 "Scripts"):
        _dir = os.path.join(_ENV, *_sub.split("/")) if _sub else _ENV
        if os.path.isdir(_dir) and _dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _dir + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import local_data  # noqa: F401
except ImportError:
    local_data = None


def sample_path(name):
    """Return the configured path for `name`, or None if unset/missing."""
    path = os.environ.get(name) or getattr(local_data, name, None)
    return path if path and os.path.exists(path) else None


def skip(name):
    """Exit cleanly, telling the reader which sample file is missing."""
    print(f"SKIPPED: no sample data. Set {name} (env var, or add it to "
          f"tests/local_data.py) to run this test.")
    sys.exit(0)


def require(name):
    """Return the path for `name`, or skip the whole test if unavailable."""
    path = sample_path(name)
    if path is None:
        skip(name)
    return path


class Results:
    """Collects pass/fail lines and sets the exit code."""

    def __init__(self):
        self.failures = []

    def check(self, label, condition, detail=""):
        if not condition:
            self.failures.append(f"{label}: {detail}")
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {label} {detail if not condition else ''}", flush=True)
        return bool(condition)

    def finish(self):
        print("\n" + ("ALL PASS" if not self.failures
                      else f"{len(self.failures)} FAILURES:"), flush=True)
        for failure in self.failures:
            print("  -", failure)
        sys.exit(1 if self.failures else 0)
