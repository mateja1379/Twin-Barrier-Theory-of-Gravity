#!/usr/bin/env python3
"""Run all 13 verification stages of the Twin-Barrier Theory.

Each stage writes its full output to results/stageN.log.

Usage:
    python run_all.py          # run all 13 stages
    python run_all.py 3 7 10   # run only selected stages
"""
import subprocess
import sys
import time
import os

STAGES = list(range(1, 14))


def run_stage(n, results_dir):
    script = f"stage{n}.py"
    log = os.path.join(results_dir, f"stage{n}.log")
    if not os.path.exists(script):
        return False, f"{script} not found", 0.0

    t0 = time.time()
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["JAX_PLATFORMS"] = "cpu"

    result = subprocess.run(
        [sys.executable, "-u", script],
        capture_output=True, text=True, env=env,
        timeout=600,
    )
    elapsed = time.time() - t0
    output = result.stdout + result.stderr

    with open(log, "w") as f:
        f.write(output)

    passed = result.returncode == 0 and ("PASS" in output or "OK" in output)
    failed_lines = [l for l in output.splitlines()
                    if "FAIL" in l.upper() and "PASS" not in l.upper()]
    if failed_lines:
        passed = False
    return passed, log, elapsed


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    if len(sys.argv) > 1:
        stages = [int(x) for x in sys.argv[1:]]
    else:
        stages = STAGES

    w = 60
    print("=" * w)
    print("  Twin-Barrier Theory of Gravity")
    print("  Full Verification Suite  (13 stages)")
    print("=" * w)

    results = {}
    t_total = time.time()

    for n in stages:
        print(f"\n  Stage {n:2d} ... ", end="", flush=True)
        try:
            ok, log, dt = run_stage(n, results_dir)
        except subprocess.TimeoutExpired:
            ok, log, dt = False, "", 0.0
            print(f"FAIL  [TIMEOUT >600s]")
            results[n] = False
            continue
        tag = "PASS" if ok else "FAIL"
        print(f"{tag}  [{dt:.1f}s]  -> {log}")
        results[n] = ok

    elapsed_total = time.time() - t_total
    n_pass = sum(results.values())
    n_total = len(results)

    print("\n" + "=" * w)
    print(f"  RESULTS: {n_pass}/{n_total} stages passed   "
          f"({elapsed_total:.1f}s total)")
    print("=" * w)
    for n in stages:
        tag = "PASS" if results[n] else "FAIL"
        print(f"    Stage {n:2d}:  {tag}")

    if n_pass == n_total:
        print(f"\n  ALL {n_total} STAGES PASS\n")
    else:
        failed = [n for n in stages if not results[n]]
        print(f"\n  FAILED stages: {failed}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
