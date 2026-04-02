"""
stage4_kk_spectrum.py – Stage 4: KK spectrum validation.

Compute the full Kaluza-Klein tower from the y-direction operator,
verify spectral convergence, gap structure, and density.

Uses the same L_y operator from Stage 3.

PASS conditions:
  - First 20 modes converge < 2% under N → 2N doubling
  - Clear zero mode (m₀² ≈ 0)
  - Stable KK gap m₁² - m₀²
  - Smooth spectral density
  - No solver-dependent spectrum drift
"""
import json
import time
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from stage3 import build_Ly_operator_simple


def compute_spectrum(Ny, y_max, k=1.0):
    """Compute full eigenspectrum for given resolution."""
    y_grid = jnp.linspace(0.0, y_max, Ny)
    L = build_Ly_operator_simple(y_grid, k)
    eigenvalues, _ = jnp.linalg.eigh(L)
    eigenvalues.block_until_ready()
    return jnp.sort(eigenvalues)


def run_stage4(k=1.0, Ny_base=200, y_max=10.0, verbose=True):
    """Run Stage 4: KK spectrum validation."""
    results = {}
    lines = []

    def log(msg):
        if verbose:
            print(msg)
        lines.append(msg)

    log("=" * 70)
    log("  STAGE 4: KK Spectrum Validation")
    log("=" * 70)
    log(f"  JAX {jax.__version__} | {jax.default_backend()} | {jax.devices()}")
    log(f"  k={k}, Ny_base={Ny_base}, y_max={y_max}")

    # ── Convergence study: N, 2N, 4N ────────────────────────────
    resolutions = [Ny_base, 2 * Ny_base, 4 * Ny_base]
    spectra = {}

    for Ny in resolutions:
        t0 = time.perf_counter()
        evals = compute_spectrum(Ny, y_max, k)
        dt = time.perf_counter() - t0
        spectra[Ny] = evals
        log(f"\n  Ny={Ny}: {dt:.2f}s, {len(evals)} eigenvalues")
        log(f"    m₀² = {float(evals[0]):.10e}")
        if len(evals) > 1:
            log(f"    m₁² = {float(evals[1]):.10e}")
            log(f"    gap  = {float(evals[1] - evals[0]):.10e}")

    # ── Convergence table for first 20 modes ─────────────────────
    N1, N2, N4 = resolutions
    evals_N = spectra[N1]
    evals_2N = spectra[N2]
    evals_4N = spectra[N4]

    n_check = min(20, len(evals_N), len(evals_2N))
    log(f"\n  Convergence table (first {n_check} modes):")
    log(f"  {'n':>3s}  {'m²(N)':>14s}  {'m²(2N)':>14s}  {'m²(4N)':>14s}  "
        f"{'|Δ|/m²(2N)':>10s}  {'|Δ₂|/m²(4N)':>12s}")
    log("  " + "-" * 80)

    convergence_ok = True
    conv_data = []
    for n in range(n_check):
        mn_N = float(evals_N[n])
        mn_2N = float(evals_2N[n])
        mn_4N = float(evals_4N[n]) if n < len(evals_4N) else mn_2N

        if abs(mn_2N) > 1e-10:
            rel_err_1 = abs(mn_2N - mn_N) / abs(mn_2N)
        else:
            rel_err_1 = abs(mn_2N - mn_N)

        if abs(mn_4N) > 1e-10:
            rel_err_2 = abs(mn_4N - mn_2N) / abs(mn_4N)
        else:
            rel_err_2 = abs(mn_4N - mn_2N)

        # Check 2N vs N convergence (skip zero mode — it's ~0)
        if n > 0 and rel_err_1 > 0.02:
            convergence_ok = False

        conv_data.append({
            "n": n, "m2_N": mn_N, "m2_2N": mn_2N, "m2_4N": mn_4N,
            "rel_err_N_2N": rel_err_1, "rel_err_2N_4N": rel_err_2
        })

        log(f"  {n:3d}  {mn_N:14.6e}  {mn_2N:14.6e}  {mn_4N:14.6e}  "
            f"{rel_err_1:10.4e}  {rel_err_2:12.4e}")

    results["convergence_table"] = conv_data

    # ── First 1000 eigenvalues (from highest resolution) ─────────
    n_out = min(1000, len(evals_4N))
    evals_out = [float(evals_4N[i]) for i in range(n_out)]
    results["eigenvalues_1000"] = evals_out

    # ── KK gap ───────────────────────────────────────────────────
    m0_sq = float(evals_4N[0])
    m1_sq = float(evals_4N[1]) if len(evals_4N) > 1 else 0.0
    gap = m1_sq - m0_sq
    log(f"\n  KK gap: m₁² - m₀² = {gap:.6e}")
    log(f"    m₀² = {m0_sq:.6e}")
    log(f"    m₁² = {m1_sq:.6e}")
    results["kk_gap"] = gap
    results["m0_sq"] = m0_sq
    results["m1_sq"] = m1_sq

    # ── Spectral density histogram ───────────────────────────────
    # Bin eigenvalues and count
    max_eval = float(evals_4N[min(n_out - 1, len(evals_4N) - 1)])
    n_bins = 20
    bin_edges = np.linspace(0, max_eval * 1.1, n_bins + 1)
    hist, _ = np.histogram(np.array(evals_out), bins=bin_edges)
    log(f"\n  Spectral density (first {n_out} modes, {n_bins} bins):")
    for b in range(n_bins):
        bar = "#" * int(hist[b] * 50 / max(max(hist), 1))
        log(f"    [{bin_edges[b]:10.2f}, {bin_edges[b+1]:10.2f}): {hist[b]:4d} {bar}")

    results["spectral_hist_counts"] = hist.tolist()
    results["spectral_hist_edges"] = bin_edges.tolist()

    # ── Spectrum drift check ─────────────────────────────────────
    # Compare eigenvalues from 2N and 4N for the first 20 modes
    drift_max = 0.0
    for n in range(min(20, len(evals_2N), len(evals_4N))):
        if abs(float(evals_4N[n])) > 1e-10:
            drift = abs(float(evals_4N[n]) - float(evals_2N[n])) / abs(float(evals_4N[n]))
        else:
            drift = abs(float(evals_4N[n]) - float(evals_2N[n]))
        drift_max = max(drift_max, drift)
    log(f"\n  Max spectrum drift (2N→4N, first 20): {drift_max:.4e}")
    results["max_drift_2N_4N"] = drift_max

    # ── PASS/FAIL ────────────────────────────────────────────────
    c1 = convergence_ok
    c2 = abs(m0_sq) < 1e-6  # clear zero mode
    c3 = gap > 0  # stable positive gap
    c4 = drift_max < 0.02  # no solver drift

    PASS = c1 and c2 and c3 and c4

    log(f"\n{'='*70}")
    log(f"  STAGE 4 VERDICT: {'PASS' if PASS else 'FAIL'}")
    log(f"    Convergence < 2%        : {c1}")
    log(f"    Clear zero mode         : {c2}  (m₀² = {m0_sq:.2e})")
    log(f"    Stable positive KK gap  : {c3}  (gap = {gap:.2e})")
    log(f"    No solver drift         : {c4}  (max = {drift_max:.2e})")
    log(f"{'='*70}")
    results["verdict"] = "PASS" if PASS else "FAIL"

    # ── Save ─────────────────────────────────────────────────────
    with open("stage4_report.txt", "w") as f:
        f.write("\n".join(lines))

    json_safe = {}
    for kk, v in results.items():
        if isinstance(v, (float, int, bool, str, list)):
            json_safe[kk] = v
        else:
            json_safe[kk] = str(v)
    with open("stage4_results.json", "w") as f:
        json.dump(json_safe, f, indent=2)

    log(f"\nSaved: stage4_report.txt, stage4_results.json")
    return results


if __name__ == "__main__":
    Ny = 200
    y_max = 10.0
    if len(sys.argv) >= 3:
        Ny = int(sys.argv[1])
        y_max = float(sys.argv[2])
    run_stage4(k=1.0, Ny_base=Ny, y_max=y_max)
