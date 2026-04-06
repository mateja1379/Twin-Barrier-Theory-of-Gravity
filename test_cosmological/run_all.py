#!/usr/bin/env python3
"""
run_all.py — Run all TBES Extended Cosmological tests
======================================================

Runs every test script in the test_cosmological/ folder sequentially
and reports a summary table of PASS / FAIL / ERROR results.

Usage:
    cd test_cosmological
    python run_all.py

Author: Mateja Radojičić / Twin Barrier Theory
Date:   April 2026
"""

import subprocess
import sys
import os
import time
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ordered test list — matches the proof numbering in twin-theory-extended.md
TESTS = [
    ("Proof 1:  5D → TBES profile (SPARC 30+30)",        "tb_dm_extended_support.py"),
    ("Proof 2:  Jeans → η₀ = 2.163 derivation",          "tb_dm_derivation_ell.py"),
    ("Proof 3:  η₀ dynamical attractor",                  "tb_eta_attractor.py"),
    ("Proof 4:  Generalized Jeans η(μ) — SLACS",          "tb_dm_generalized_jeans.py"),
    ("Proof 5:  LITTLE THINGS independent validation",     "tb_dm_little_things.py"),
    ("Proof 6:  Strong lensing — SLACS Einstein radii",    "tb_dm_strong_lensing.py"),
    ("Proof 7:  Galaxy clusters — CLASH",                  "tb_dm_cluster_lensing.py"),
    ("Proof 8:  ΔN_eff warp-factor decoupling",           "tb_neff_resolution.py"),
    ("Proof 9:  Radion resolution — UV-brane coupling",    "tb_radion_resolution.py"),
    ("Proof 10: DM self-interaction σ/m",                  "tb_dm_self_interaction.py"),
    ("Extra:    DM halo profile comparison",               "tb_dm_halo_test.py"),
    ("Extra:    DM abundance (exploratory)",                "tb_dm_abundance.py"),
]


def run_test(name, script):
    """Run a single test script and return (status, elapsed, last_lines)."""
    path = os.path.join(SCRIPT_DIR, script)
    if not os.path.isfile(path):
        return "MISSING", 0.0, f"File not found: {path}"

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=SCRIPT_DIR,
        )
        elapsed = time.time() - t0
        output = result.stdout + result.stderr

        # Detect PASS/FAIL from output
        lines = output.strip().split('\n')
        last_lines = '\n'.join(lines[-5:]) if lines else ''

        if result.returncode != 0:
            status = "ERROR"
        elif re.search(r'(FAIL|FAILED|REJECTED)', output, re.IGNORECASE):
            status = "FAIL"
        elif re.search(r'(PASS|PROLAZI|VERDICT.*PASS)', output, re.IGNORECASE):
            status = "PASS"
        else:
            status = "DONE"

        return status, elapsed, last_lines

    except subprocess.TimeoutExpired:
        return "TIMEOUT", 600.0, "Exceeded 600s timeout"
    except Exception as e:
        return "ERROR", time.time() - t0, str(e)


def main():
    print("=" * 78)
    print("  TWIN BARRIER EXTENDED COSMOLOGICAL — FULL TEST SUITE")
    print("=" * 78)
    print(f"  Python:    {sys.executable}")
    print(f"  Directory: {SCRIPT_DIR}")
    print(f"  Tests:     {len(TESTS)}")
    print("=" * 78)

    results = []
    total_t0 = time.time()

    for i, (name, script) in enumerate(TESTS, 1):
        print(f"\n{'─' * 78}")
        print(f"  [{i}/{len(TESTS)}] {name}")
        print(f"  Script: {script}")
        print(f"{'─' * 78}\n")

        status, elapsed, last_lines = run_test(name, script)
        results.append((name, script, status, elapsed))

        marker = {"PASS": "✅", "FAIL": "❌", "ERROR": "⚠️",
                  "TIMEOUT": "⏰", "DONE": "🔵", "MISSING": "❓"}.get(status, "?")
        print(f"\n  → {marker} {status} ({elapsed:.1f}s)")

    total_time = time.time() - total_t0

    # Summary table
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"\n  {'#':<4} {'Test':<50} {'Status':<8} {'Time':>6}")
    print(f"  {'─'*4} {'─'*50} {'─'*8} {'─'*6}")

    n_pass = n_fail = n_error = 0
    for i, (name, script, status, elapsed) in enumerate(results, 1):
        marker = {"PASS": "✅", "FAIL": "❌", "ERROR": "⚠️",
                  "TIMEOUT": "⏰", "DONE": "🔵", "MISSING": "❓"}.get(status, "?")
        short_name = name[:50]
        print(f"  {i:<4} {short_name:<50} {marker} {status:<6} {elapsed:>5.1f}s")
        if status == "PASS":
            n_pass += 1
        elif status == "FAIL":
            n_fail += 1
        else:
            n_error += 1

    print(f"\n  {'─' * 70}")
    print(f"  Total: {n_pass} PASS, {n_fail} FAIL, {n_error} other "
          f"({total_time:.0f}s)")
    print("=" * 78)

    # Exit code: 0 if all pass, 1 otherwise
    sys.exit(0 if n_fail == 0 and n_error == 0 else 1)


if __name__ == "__main__":
    main()
