"""
stage6_time_evolution.py – Stage 6: Time evolution stability.

Evolve a small localized perturbation under:
    ∂²_t h + L_y h = 0

using a symplectic (leapfrog/Störmer-Verlet) integrator.

The perturbation h(y,t) is decomposed in eigen-modes:
    h(y,t) = Σ_n  a_n(t) ψ_n(y)

where  a_n'' + m_n² a_n = 0, giving bounded oscillatory solutions
a_n(t) = A_n cos(m_n t) + B_n sin(m_n t)  for m_n² > 0.

PASS conditions:
  - Bounded oscillatory evolution
  - No exponential growth
  - Stable energy envelope
  - No mode blow-up
"""
import json
import time
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from stage3 import build_Ly_raw, build_Ly_operator


def initial_perturbation(y_grid, y0=0.0, sigma=0.5):
    """Localized Gaussian perturbation centered at y0."""
    return jnp.exp(-0.5 * ((y_grid - y0) / sigma) ** 2)


def leapfrog_step(h, v, A, dt):
    """One Störmer-Verlet (leapfrog) step for  h'' = -A h.

    A = M^{-1} H  is the effective operator.

    h_{n+1} = h_n + dt * v_{n+1/2}
    v_{n+3/2} = v_{n+1/2} - dt * A h_{n+1}
    """
    h_new = h + dt * v
    acc = -A @ h_new
    v_new = v + dt * acc
    return h_new, v_new


def compute_energy(h, v, H, M_diag):
    """Total energy = (1/2) v^T M v + (1/2) h^T H h."""
    kinetic = 0.5 * jnp.dot(v * M_diag, v)
    potential = 0.5 * jnp.dot(h, H @ h)
    return float(kinetic + potential)


def compute_norm(h, M_diag):
    """Weighted L2 norm: sqrt(h^T M h)."""
    return float(jnp.sqrt(jnp.dot(h * M_diag, h)))


def run_stage6(k=1.0, Ny=200, y_max=10.0, T_final=50.0, dt=0.005,
               verbose=True):
    """Run Stage 6: time evolution stability check."""
    results = {}
    lines = []

    def log(msg):
        if verbose:
            print(msg)
        lines.append(msg)

    log("=" * 70)
    log("  STAGE 6: Time Evolution Stability")
    log("=" * 70)
    log(f"  JAX {jax.__version__} | {jax.default_backend()} | {jax.devices()}")
    log(f"  k={k}, Ny={Ny}, y_max={y_max}")
    log(f"  T_final={T_final}, dt={dt}")

    y_grid = jnp.linspace(0.0, y_max, Ny)
    N = Ny - 1  # reduced dimension (Dirichlet node removed)

    # Build Sturm-Liouville stiffness H and mass M (raw)
    H_mat, M_diag = build_Ly_raw(y_grid, k)
    # Effective operator for time evolution: A = M^{-1} H
    M_inv = 1.0 / M_diag
    A = H_mat * M_inv[:, None]  # A[i,j] = H[i,j] / M[i,i]

    # For eigenvalue check and mode tracking, use the symmetric form
    S, M_inv_sqrt = build_Ly_operator(y_grid, k)

    # Check CFL-like condition: dt < 2/sqrt(λ_max(A))
    # Eigenvalues of A = eigenvalues of S (same generalized problem)
    evals = jnp.linalg.eigvalsh(S)
    lambda_max = float(evals[-1])
    dt_cfl = 2.0 / jnp.sqrt(lambda_max)
    log(f"  N = {N} (reduced), λ_max = {lambda_max:.4e}, CFL dt_max = {float(dt_cfl):.6f}")
    if dt > float(dt_cfl):
        dt = float(dt_cfl) * 0.9
        log(f"  *** Reducing dt to {dt:.6f} for stability ***")
    results["lambda_max"] = lambda_max
    results["dt_used"] = dt

    Nt = int(T_final / dt)
    log(f"  Nt = {Nt} steps")

    # Initial condition: Gaussian at brane (reduced space, exclude Dirichlet node)
    h = initial_perturbation(y_grid[:N], y0=0.0, sigma=0.5)
    v = jnp.zeros(N)  # start from rest

    # Initialize leapfrog: half-step for velocity
    v = v - 0.5 * dt * (A @ h)

    # Energy and norm tracking
    n_samples = min(500, Nt)
    sample_interval = max(Nt // n_samples, 1)
    energies = []
    norms = []
    times = []
    mode_amps = []

    # Project onto first few eigenmodes (in ψ-space) for tracking
    _, evecs_S = jnp.linalg.eigh(S)
    evecs_psi = evecs_S * M_inv_sqrt[:, None]  # ψ_n = M^{-1/2} v_n
    n_modes_track = min(5, N)

    E0 = compute_energy(h, v + 0.5 * dt * (A @ h), H_mat, M_diag)
    h0_norm = compute_norm(h, M_diag)

    log(f"\n  Initial energy: {E0:.6e}")
    log(f"  Initial norm:   {h0_norm:.6e}")

    # ── Time evolution loop ──────────────────────────────────────
    t0 = time.perf_counter()

    for step in range(Nt):
        h, v = leapfrog_step(h, v, A, dt)
        # Dirichlet BC is implicit (node removed from system)

        if step % sample_interval == 0:
            # Correct velocity for energy computation
            v_full = v - 0.5 * dt * (A @ h)
            E = compute_energy(h, v_full, H_mat, M_diag)
            norm_h = compute_norm(h, M_diag)
            t_curr = (step + 1) * dt

            energies.append(E)
            norms.append(norm_h)
            times.append(t_curr)

            # Mode amplitudes: project h onto ψ-space eigenvectors
            # Using weighted inner product: a_m = <ψ_m, h>_w / <ψ_m, ψ_m>_w
            # For lumped mass: a_m = Σ_j M_j ψ_m(j) h(j) / Σ_j M_j ψ_m²(j)
            amps = []
            for m in range(n_modes_track):
                psi_m = evecs_psi[:, m]
                a_m = float(jnp.dot(M_diag * psi_m, h) / jnp.dot(M_diag * psi_m, psi_m))
                amps.append(a_m)
            mode_amps.append(amps)

    dt_run = time.perf_counter() - t0
    log(f"\n  Evolution completed: {dt_run:.2f}s for {Nt} steps")

    energies = np.array(energies)
    norms = np.array(norms)
    times = np.array(times)
    mode_amps = np.array(mode_amps)

    # ── Analysis ─────────────────────────────────────────────────
    E_mean = float(np.mean(energies))
    E_std = float(np.std(energies))
    E_rel_var = E_std / max(abs(E_mean), 1e-30)

    norm_max = float(np.max(norms))
    norm_min = float(np.min(norms))
    norm_ratio = norm_max / max(norm_min, 1e-30)

    log(f"\n  Energy: mean={E_mean:.6e}, std={E_std:.6e}, rel_var={E_rel_var:.4e}")
    log(f"  Norm: min={norm_min:.6e}, max={norm_max:.6e}, ratio={norm_ratio:.4f}")
    results["E_mean"] = E_mean
    results["E_std"] = E_std
    results["E_rel_var"] = E_rel_var
    results["norm_max"] = norm_max
    results["norm_min"] = norm_min
    results["norm_ratio"] = norm_ratio

    # ── Exponential growth fit ───────────────────────────────────
    # Fit log(norm) = a + ω*t.  If ω > 0, there's growth.
    log_norms = np.log(np.maximum(norms, 1e-30))
    if len(times) > 2:
        coeffs = np.polyfit(times, log_norms, 1)
        omega = float(coeffs[0])  # growth rate
        log(f"\n  Growth rate ω from log-norm fit: {omega:.6e}")
        log(f"    (ω > 0 means exponential growth)")
        results["growth_rate_omega"] = omega
    else:
        omega = 0.0
        results["growth_rate_omega"] = 0.0

    # ── Mode amplitudes ──────────────────────────────────────────
    log(f"\n  Mode amplitudes at t=0 and t=T_final:")
    for m in range(n_modes_track):
        a0 = float(mode_amps[0, m])
        af = float(mode_amps[-1, m])
        a_max = float(np.max(np.abs(mode_amps[:, m])))
        log(f"    mode {m}: a(0)={a0:.4e}, a(T)={af:.4e}, max|a|={a_max:.4e}")

    # Check mode blow-up
    mode_blowup = False
    for m in range(n_modes_track):
        a_max = float(np.max(np.abs(mode_amps[:, m])))
        a_init = max(float(np.abs(mode_amps[0, m])), 1e-30)
        if a_max / a_init > 100:
            mode_blowup = True
            log(f"    *** Mode {m} blow-up detected: ratio {a_max/a_init:.1f}")

    results["mode_blowup"] = mode_blowup

    # ── Energy samples at key times ──────────────────────────────
    log(f"\n  Energy at selected times:")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        idx = min(int(frac * (len(energies) - 1)), len(energies) - 1)
        log(f"    t={times[idx]:.1f}: E={energies[idx]:.6e}")

    # ── VERDICT ──────────────────────────────────────────────────
    c1 = E_rel_var < 0.01  # energy conserved to 1%
    c2 = omega < 0.01  # no exponential growth
    c3 = norm_ratio < 5.0  # bounded oscillation
    c4 = not mode_blowup

    PASS = c1 and c2 and c3 and c4

    log(f"\n{'='*70}")
    log(f"  STAGE 6 VERDICT: {'PASS' if PASS else 'FAIL'}")
    log(f"    Energy stable (rel_var < 1%)  : {c1}  ({E_rel_var:.2e})")
    log(f"    No exp growth (ω < 0.01)      : {c2}  (ω = {omega:.2e})")
    log(f"    Bounded norm (ratio < 5)       : {c3}  ({norm_ratio:.4f})")
    log(f"    No mode blow-up                : {c4}")
    log(f"{'='*70}")
    results["verdict"] = "PASS" if PASS else "FAIL"

    with open("stage6_report.txt", "w") as f:
        f.write("\n".join(lines))

    json_safe = {}
    for kk, v in results.items():
        if isinstance(v, (float, int, bool, str)):
            json_safe[kk] = v
        else:
            json_safe[kk] = str(v)
    with open("stage6_results.json", "w") as f:
        json.dump(json_safe, f, indent=2)

    log(f"\nSaved: stage6_report.txt, stage6_results.json")
    return results


if __name__ == "__main__":
    Ny = 200
    y_max = 10.0
    T = 50.0
    if len(sys.argv) >= 3:
        Ny = int(sys.argv[1])
        y_max = float(sys.argv[2])
    if len(sys.argv) >= 4:
        T = float(sys.argv[3])
    run_stage6(k=1.0, Ny=Ny, y_max=y_max, T_final=T)
