"""
background_test.py – Stage 1 integration test.

Evaluates the warped RS background on a 2-D grid and verifies:
  1)  DeTurck vector ξ² ≈ 0
  2)  Einstein-DeTurck residual E_{AB} ≈ 0
  3)  Ricci scalar ≈ -20k²

Outputs a text report and a JSON summary.
"""
import json
import time
import sys
import os

import jax
import jax.numpy as jnp

# ── Project imports ──────────────────────────────────────────────────────────
from grid import make_grid
from metric import (
    ricci_scalar,
    ricci_scalar_analytic,
    ricci_tensor,
    ricci_tensor_analytic,
    christoffel_analytic,
    christoffel_all,
    background_metric,
    background_metric_inv,
)
from deturck import (
    deturck_vector,
    deturck_vector_norm_sq,
    einstein_deturck_residual,
    residual_on_grid,
)


def run_tests(k: float = 1.0, Nx: int = 20, Nz: int = 20, verbose: bool = True):
    """Run all Stage 1 verification tests.

    Returns a dict with all results (also saved to disk).
    """
    results = {}
    lines = []

    def log(msg):
        if verbose:
            print(msg)
        lines.append(msg)

    log("=" * 60)
    log("  Stage 1: 5D Warped-Brane Background Verification")
    log("=" * 60)

    # ── 0. Environment ───────────────────────────────────────────
    log(f"\nJAX version    : {jax.__version__}")
    log(f"Devices        : {jax.devices()}")
    log(f"Backend        : {jax.default_backend()}")
    results["jax_version"] = jax.__version__
    results["devices"] = str(jax.devices())
    results["backend"] = jax.default_backend()

    # ── 1. Grid ──────────────────────────────────────────────────
    log(f"\nGrid: Nx={Nx}, Nz={Nz}")
    grid = make_grid(Nx=Nx, Nz=Nz, r_h=1.0, r_scale=1.0, y_scale=1.0)
    log(f"  R range : [{float(grid['R'].min()):.3f}, {float(grid['R'].max()):.3f}]")
    log(f"  Y range : [{float(grid['Y'].min()):.3f}, {float(grid['Y'].max()):.3f}]")

    # ── 2. Single-point metric sanity ────────────────────────────
    log("\n--- Single-point checks at (r=5, θ=π/4, y=0.5) ---")
    coords0 = jnp.array([0.0, 5.0, jnp.pi / 4, jnp.pi / 3, 0.5])

    g0 = background_metric(coords0[1], coords0[2], coords0[4], k=k)
    gi0 = background_metric_inv(coords0[1], coords0[2], coords0[4], k=k)
    identity_err = float(jnp.max(jnp.abs(g0 @ gi0 - jnp.eye(5))))
    log(f"  max|g·g^-1 - I| = {identity_err:.2e}")
    results["metric_inverse_error"] = identity_err

    # Christoffel: analytic vs autodiff
    log("\n  Christoffel comparison (analytic vs autodiff):")
    Ga = christoffel_analytic(coords0, k=k)
    Gn = christoffel_all(coords0, k=k)
    christoffel_err = float(jnp.max(jnp.abs(Ga - Gn)))
    log(f"  max|Γ_analytic - Γ_autodiff| = {christoffel_err:.2e}")
    results["christoffel_error"] = christoffel_err

    # Ricci single point
    R_num = ricci_tensor(coords0, k=k)
    R_exact = ricci_tensor_analytic(coords0, k=k)
    ricci_err = float(jnp.max(jnp.abs(R_num - R_exact)))
    Rs_num = float(ricci_scalar(coords0, k=k))
    Rs_exact = float(ricci_scalar_analytic(k=k))
    log(f"\n  max|R_AB(num) - R_AB(exact)| = {ricci_err:.2e}")
    log(f"  Ricci scalar (num)   = {Rs_num:.6f}")
    log(f"  Ricci scalar (exact) = {Rs_exact:.6f}")
    results["ricci_tensor_error"] = ricci_err
    results["ricci_scalar_numerical"] = Rs_num
    results["ricci_scalar_exact"] = Rs_exact

    # ── 3. DeTurck at single point ───────────────────────────────
    xi0 = deturck_vector(coords0, k=k)
    xi2_0 = float(deturck_vector_norm_sq(coords0, k=k))
    log(f"\n  ξ^A  = {xi0}")
    log(f"  ξ²   = {xi2_0:.2e}")
    results["xi2_single"] = xi2_0

    E0 = einstein_deturck_residual(coords0, k=k)
    E0_max = float(jnp.max(jnp.abs(E0)))
    log(f"  max|E_AB| (single) = {E0_max:.2e}")
    results["E_AB_max_single"] = E0_max

    # ── 4. Full grid evaluation ──────────────────────────────────
    log(f"\n--- Full grid evaluation ({Nx}×{Nz} = {Nx*Nz} points) ---")
    t0 = time.perf_counter()
    res = residual_on_grid(grid["R"], grid["Y"], k=k)
    dt_grid = time.perf_counter() - t0
    log(f"  Time (incl JIT) : {dt_grid:.2f} s")

    # Second run (JIT-warm)
    t1 = time.perf_counter()
    res2 = residual_on_grid(grid["R"], grid["Y"], k=k)
    dt_warm = time.perf_counter() - t1
    log(f"  Time (JIT-warm) : {dt_warm:.2f} s")

    log(f"\n  max |ξ²|    = {res['xi2_max']:.2e}")
    log(f"  max |E_AB|  = {res['E_AB_max']:.2e}")
    log(f"  mean R      = {res['R_scalar_mean']:.6f}  (exact = {Rs_exact:.6f})")

    results["xi2_max_grid"] = res["xi2_max"]
    results["E_AB_max_grid"] = res["E_AB_max"]
    results["R_scalar_mean"] = res["R_scalar_mean"]
    results["grid_time_cold_s"] = dt_grid
    results["grid_time_warm_s"] = dt_warm
    results["grid_Nx"] = Nx
    results["grid_Nz"] = Nz

    # ── 5. Find worst component ──────────────────────────────────
    E_flat = jnp.abs(res["E_AB"]).reshape(-1, 5, 5)
    # max over grid points for each component
    E_comp_max = jnp.max(E_flat, axis=0)
    worst_idx = jnp.unravel_index(jnp.argmax(E_comp_max), (5, 5))
    worst_val = float(E_comp_max[worst_idx[0], worst_idx[1]])
    comp_names = ["t", "r", "θ", "φ", "y"]
    worst_name = f"E_{comp_names[int(worst_idx[0])]}{comp_names[int(worst_idx[1])]}"
    log(f"  Worst component : {worst_name} = {worst_val:.2e}")
    results["worst_component"] = worst_name
    results["worst_component_value"] = worst_val

    # ── 6. Verdict ───────────────────────────────────────────────
    PASS = (res["xi2_max"] < 1e-8 and
            res["E_AB_max"] < 1e-6 and
            abs(res["R_scalar_mean"] - Rs_exact) / abs(Rs_exact) < 1e-4)

    status = "PASS" if PASS else "FAIL"
    log(f"\n{'='*60}")
    log(f"  VERDICT: {status}")
    if not PASS:
        if res["xi2_max"] >= 1e-8:
            log(f"    ξ² too large: {res['xi2_max']:.2e} (threshold 1e-8)")
        if res["E_AB_max"] >= 1e-6:
            log(f"    E_AB too large: {res['E_AB_max']:.2e} (threshold 1e-6)")
        if abs(res["R_scalar_mean"] - Rs_exact) / abs(Rs_exact) >= 1e-4:
            log(f"    Ricci scalar off: {res['R_scalar_mean']:.6f} vs {Rs_exact:.6f}")
    log(f"{'='*60}")
    results["verdict"] = status

    # ── Save outputs ─────────────────────────────────────────────
    report_txt = "\n".join(lines)
    with open("stage1_report.txt", "w") as f:
        f.write(report_txt)

    # Convert any non-serializable values
    json_safe = {}
    for kk, v in results.items():
        if isinstance(v, float):
            json_safe[kk] = v
        else:
            json_safe[kk] = str(v)

    with open("stage1_results.json", "w") as f:
        json.dump(json_safe, f, indent=2)

    log(f"\nSaved: stage1_report.txt, stage1_results.json")
    return results


if __name__ == "__main__":
    k = 1.0
    Nx = 20
    Nz = 20
    # Allow overriding grid size from command line
    if len(sys.argv) >= 3:
        Nx = int(sys.argv[1])
        Nz = int(sys.argv[2])

    run_tests(k=k, Nx=Nx, Nz=Nz)
