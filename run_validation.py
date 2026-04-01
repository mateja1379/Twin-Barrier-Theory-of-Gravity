#!/usr/bin/env python3
"""
run_validation.py – Master runner for Stages 3-7 of the bulk gravity validation suite.

Executes each stage sequentially, collects verdicts, and determines final outcome:
  STRONG PASS : all 5 stages PASS
  MEDIUM PASS : Stages 3-5 PASS, but 6 or 7 partial/FAIL
  FAIL        : any of Stages 3, 4, or 5 FAIL
"""
import json
import time
import sys
import traceback

import jax
jax.config.update("jax_enable_x64", True)

# ─── Import stages ────────────────────────────────────────────────
from stage3_zero_mode import run_stage3
from stage4_kk_spectrum import run_stage4
from stage5_ghost_tachyon import run_stage5
from stage6_time_evolution import run_stage6
from stage7_ppn import run_stage7


def run_all(verbose=True):
    verdicts = {}
    results_all = {}
    lines = []

    def log(msg):
        if verbose:
            print(msg)
        lines.append(msg)

    log("=" * 70)
    log("  BULK GRAVITY VALIDATION SUITE")
    log("  Stages 3-7")
    log("=" * 70)
    log(f"  JAX {jax.__version__} | {jax.default_backend()} | {jax.devices()}")
    log("")

    t_start = time.perf_counter()

    # ── Stage 3: Zero Mode ────────────────────────────────────────
    log("\n" + "#" * 70)
    log("  RUNNING STAGE 3: Zero Mode Validation")
    log("#" * 70)
    try:
        r3 = run_stage3(k=1.0, Ny=200, y_max=10.0, verbose=verbose)
        verdicts["stage3"] = r3.get("verdict", "ERROR")
        results_all["stage3"] = r3
    except Exception as e:
        verdicts["stage3"] = "ERROR"
        log(f"  *** Stage 3 ERROR: {e}")
        traceback.print_exc()

    # ── Stage 4: KK Spectrum ──────────────────────────────────────
    log("\n" + "#" * 70)
    log("  RUNNING STAGE 4: KK Spectrum")
    log("#" * 70)
    try:
        r4 = run_stage4(k=1.0, Ny_base=200, y_max=10.0, verbose=verbose)
        verdicts["stage4"] = r4.get("verdict", "ERROR")
        results_all["stage4"] = r4
    except Exception as e:
        verdicts["stage4"] = "ERROR"
        log(f"  *** Stage 4 ERROR: {e}")
        traceback.print_exc()

    # ── Stage 5: Ghost / Tachyon Gate ─────────────────────────────
    log("\n" + "#" * 70)
    log("  RUNNING STAGE 5: Ghost / Tachyon Gate")
    log("#" * 70)
    try:
        r5 = run_stage5(k=1.0, Ny=200, y_max=10.0, verbose=verbose)
        verdicts["stage5"] = r5.get("verdict", "ERROR")
        results_all["stage5"] = r5
    except Exception as e:
        verdicts["stage5"] = "ERROR"
        log(f"  *** Stage 5 ERROR: {e}")
        traceback.print_exc()

    # ── Stage 6: Time Evolution ───────────────────────────────────
    log("\n" + "#" * 70)
    log("  RUNNING STAGE 6: Time Evolution Stability")
    log("#" * 70)
    try:
        r6 = run_stage6(k=1.0, Ny=200, y_max=10.0, T_final=50.0,
                         verbose=verbose)
        verdicts["stage6"] = r6.get("verdict", "ERROR")
        results_all["stage6"] = r6
    except Exception as e:
        verdicts["stage6"] = "ERROR"
        log(f"  *** Stage 6 ERROR: {e}")
        traceback.print_exc()

    # ── Stage 7: PPN Sanity ───────────────────────────────────────
    log("\n" + "#" * 70)
    log("  RUNNING STAGE 7: PPN Relativistic Sanity")
    log("#" * 70)
    try:
        r7 = run_stage7(k=1.0, Nr=120, Ny=50, r_max=200.0, y_max=5.0,
                         verbose=verbose)
        verdicts["stage7"] = r7.get("verdict", "ERROR")
        results_all["stage7"] = r7
    except Exception as e:
        verdicts["stage7"] = "ERROR"
        log(f"  *** Stage 7 ERROR: {e}")
        traceback.print_exc()

    dt_total = time.perf_counter() - t_start

    # ── Final Verdict ─────────────────────────────────────────────
    log("\n\n" + "=" * 70)
    log("  FINAL VALIDATION SUMMARY")
    log("=" * 70)
    for stage, v in sorted(verdicts.items()):
        log(f"    {stage}: {v}")

    core_pass = all(verdicts.get(s) == "PASS" for s in ["stage3", "stage4", "stage5"])
    all_pass = all(v == "PASS" for v in verdicts.values())

    if all_pass:
        final = "STRONG PASS"
    elif core_pass:
        final = "MEDIUM PASS"
    else:
        final = "FAIL"

    log(f"\n  ╔══════════════════════════════════╗")
    log(f"  ║  FINAL VERDICT:  {final:^15s} ║")
    log(f"  ╚══════════════════════════════════╝")
    log(f"\n  Total run time: {dt_total:.1f}s")

    # Save
    summary = {
        "verdicts": verdicts,
        "final_verdict": final,
        "total_time_s": round(dt_total, 2),
    }
    with open("validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("validation_report.txt", "w") as f:
        f.write("\n".join(lines))

    log(f"\nSaved: validation_summary.json, validation_report.txt")
    return summary


if __name__ == "__main__":
    run_all(verbose=True)
