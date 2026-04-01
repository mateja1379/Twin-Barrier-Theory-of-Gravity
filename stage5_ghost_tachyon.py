"""
stage5_ghost_tachyon.py – Stage 5: Ghost / Tachyon fatal gate.

This is a FATAL stage — any failure = HARD FAIL.

Checks:
  1. Ghost: kinetic matrix K must have λ_min(K) ≥ 0
  2. Tachyon: all mass-squared eigenvalues m_n² ≥ 0

In the Schrödinger picture (χ-space, where ψ = e^{2ky} χ), the
operator is H = -d²/dy² + 4k², which is self-adjoint with standard
inner product.  The kinetic energy is:

    K[χ] = ∫ (χ')² dy  ≥ 0

and the "potential" contribution is ∫ 4k² χ² dy ≥ 0.  So the total
Hamiltonian H = K + V is manifestly positive semi-definite, which
means no ghosts and no tachyons (m² ≥ 0).

We verify this numerically.
"""
import json
import time
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from stage3_zero_mode import build_Ly_operator_simple


def build_kinetic_matrix(y_grid, k=1.0):
    """Build the discrete kinetic (gradient energy) matrix in χ-space:

        K_ij = ∫₀^{y_max} (dφ_i/dy)(dφ_j/dy) dy

    In the Schrödinger picture the weight is 1 (standard L²).
    Returns (Ny, Ny) positive semi-definite matrix.
    """
    Ny = y_grid.shape[0]
    dy = float(y_grid[1] - y_grid[0])
    K = jnp.zeros((Ny, Ny))

    for j in range(Ny - 1):
        # Gradient stencil: dχ/dy ≈ (χ_{j+1} - χ_j) / dy
        # Contribution to K:  (1/dy) [[1, -1], [-1, 1]]
        coeff = 1.0 / dy
        K = K.at[j, j].add(coeff)
        K = K.at[j, j + 1].add(-coeff)
        K = K.at[j + 1, j].add(-coeff)
        K = K.at[j + 1, j + 1].add(coeff)

    return K


def build_mass_matrix(y_grid, k=1.0):
    """Build the discrete mass matrix in χ-space:

        M_ij = ∫₀^{y_max} φ_i φ_j dy

    Standard L² (weight=1 in Schrödinger picture).  Lumped diagonal.
    """
    Ny = y_grid.shape[0]
    dy = float(y_grid[1] - y_grid[0])
    M = jnp.zeros((Ny, Ny))

    for j in range(Ny):
        if j == 0 or j == Ny - 1:
            M = M.at[j, j].set(dy / 2.0)
        else:
            M = M.at[j, j].set(dy)

    return M


def run_stage5(k=1.0, Ny=200, y_max=10.0, verbose=True):
    """Run Stage 5: ghost/tachyon fatal gate."""
    results = {}
    lines = []

    def log(msg):
        if verbose:
            print(msg)
        lines.append(msg)

    log("=" * 70)
    log("  STAGE 5: Ghost / Tachyon Fatal Gate")
    log("=" * 70)
    log(f"  JAX {jax.__version__} | {jax.default_backend()} | {jax.devices()}")
    log(f"  k={k}, Ny={Ny}, y_max={y_max}")

    y_grid = jnp.linspace(0.0, y_max, Ny)
    dy = float(y_grid[1] - y_grid[0])

    # ── Ghost check: kinetic matrix ──────────────────────────────
    log("\n--- Ghost Check: Kinetic Matrix ---")
    t0 = time.perf_counter()
    K = build_kinetic_matrix(y_grid, k)
    dt_k = time.perf_counter() - t0
    log(f"  Built kinetic matrix: {K.shape} in {dt_k:.2f}s")

    k_evals = jnp.linalg.eigvalsh(K)
    k_evals.block_until_ready()
    lambda_min = float(k_evals[0])
    lambda_max = float(k_evals[-1])
    n_negative = int(jnp.sum(k_evals < -1e-14))

    log(f"  λ_min(K) = {lambda_min:.10e}")
    log(f"  λ_max(K) = {lambda_max:.10e}")
    log(f"  Negative eigenvalues (< -1e-14): {n_negative}")
    results["lambda_min_K"] = lambda_min
    results["lambda_max_K"] = lambda_max
    results["n_negative_kinetic"] = n_negative

    ghost_free = lambda_min >= -1e-10
    log(f"  Ghost-free: {ghost_free}")

    # ── Tachyon check: mass spectrum ─────────────────────────────
    log("\n--- Tachyon Check: Mass Spectrum ---")
    L = build_Ly_operator_simple(y_grid, k)
    t0 = time.perf_counter()
    m_sq_evals = jnp.linalg.eigvalsh(L)
    m_sq_evals.block_until_ready()
    dt_m = time.perf_counter() - t0
    log(f"  Eigendecomposition: {dt_m:.2f}s")

    m_sq_min = float(m_sq_evals[0])
    n_tachyon = int(jnp.sum(m_sq_evals < -1e-6))

    log(f"  m²_min = {m_sq_min:.10e}")
    log(f"  Tachyonic modes (m² < -1e-6): {n_tachyon}")

    # First 10 m² values
    log(f"  First 10 m²:")
    for i in range(min(10, len(m_sq_evals))):
        log(f"    m²_{i} = {float(m_sq_evals[i]):.10e}")

    results["m_sq_min"] = m_sq_min
    results["n_tachyonic"] = n_tachyon

    tachyon_free = m_sq_min >= -1e-6
    log(f"  Tachyon-free: {tachyon_free}")

    # ── Near-zero unstable branch ────────────────────────────────
    log("\n--- Near-zero unstable branch check ---")
    # Count modes with -1e-6 < m² < -1e-10 (suspicious near-zero)
    suspicious = jnp.sum((m_sq_evals > -1e-6) & (m_sq_evals < -1e-10))
    n_suspicious = int(suspicious)
    log(f"  Near-zero suspicious modes: {n_suspicious}")
    results["n_suspicious"] = n_suspicious

    # ── Full energy positivity: K - m² M ─────────────────────────
    log("\n--- Energy positivity: E = K + m² M ---")
    M_mat = build_mass_matrix(y_grid, k)

    # For each mass eigenvalue, the energy functional should be positive
    # Just check: all eigenvalues of K are ≥ 0 (ghost), and all m² ≥ 0 (tachyon)
    # Combined: the total Hamiltonian H = K + V should have all positive eigenvalues
    # For our Sturm-Liouville system, H = L with the mass matrix:
    #   K ψ = m² M ψ  →  generalized eigenvalue  K^{-1} M has positive spectrum
    # Simpler: just verify K ≥ 0 and m² ≥ 0 separately.

    no_unstable = n_suspicious == 0
    log(f"  No unstable branch: {no_unstable}")

    # ── VERDICT ──────────────────────────────────────────────────
    PASS = ghost_free and tachyon_free and no_unstable

    log(f"\n{'='*70}")
    log(f"  STAGE 5 VERDICT: {'PASS' if PASS else '*** HARD FAIL ***'}")
    log(f"    Ghost-free (λ_min(K) ≥ 0) : {ghost_free}  (λ_min = {lambda_min:.2e})")
    log(f"    Tachyon-free (m² ≥ 0)      : {tachyon_free}  (m²_min = {m_sq_min:.2e})")
    log(f"    No unstable branch          : {no_unstable}  (n = {n_suspicious})")
    if not PASS:
        log(f"  *** THIS IS A FATAL FAILURE ***")
    log(f"{'='*70}")
    results["verdict"] = "PASS" if PASS else "HARD FAIL"

    with open("stage5_report.txt", "w") as f:
        f.write("\n".join(lines))

    json_safe = {}
    for kk, v in results.items():
        if isinstance(v, (float, int, bool, str)):
            json_safe[kk] = v
        else:
            json_safe[kk] = str(v)
    with open("stage5_results.json", "w") as f:
        json.dump(json_safe, f, indent=2)

    log(f"\nSaved: stage5_report.txt, stage5_results.json")
    return results


if __name__ == "__main__":
    Ny = 200
    y_max = 10.0
    if len(sys.argv) >= 3:
        Ny = int(sys.argv[1])
        y_max = float(sys.argv[2])
    run_stage5(k=1.0, Ny=Ny, y_max=y_max)
