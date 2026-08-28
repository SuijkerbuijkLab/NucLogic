"""Run every test script in this folder.

    pixi run python tests/run_all.py

Each test is a separate process, so one crash cannot take the rest down and
readers that hold file handles are always released. Exits non-zero if any
test fails.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _harness import PROJECT_ROOT  # noqa: E402


def main():
    tests = sorted(
        f for f in os.listdir(HERE)
        if f.startswith("test_") and f.endswith(".py")
    )
    if not tests:
        print("No tests found.")
        return 1

    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    results = []
    for name in tests:
        started = time.time()
        finished = subprocess.run(
            [sys.executable, "-u", os.path.join(HERE, name)],
            cwd=PROJECT_ROOT, capture_output=True, text=True, env=env,
        )
        output = finished.stdout + finished.stderr
        if "SKIPPED:" in output:
            status = "SKIP"
        elif finished.returncode == 0:
            status = "PASS"
        else:
            status = "FAIL"
        results.append((name, status, time.time() - started, output))

    print(f"\n{'test':<32} {'status':<7} {'seconds':>8}")
    print("-" * 50)
    for name, status, seconds, _ in results:
        print(f"{name:<32} {status:<7} {seconds:>8.1f}")

    failed = [r for r in results if r[1] == "FAIL"]
    for name, _, _, output in failed:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        print(output.strip()[-4000:])

    passed = sum(1 for r in results if r[1] == "PASS")
    skipped = sum(1 for r in results if r[1] == "SKIP")
    print(f"\n{passed} passed, {len(failed)} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
