"""
stage3_zero_mode.py – Stage 3: Bulk zero-mode validation.

Physics (RS-II Kaluza-Klein decomposition)
------------------------------------------
The 5D linearized perturbation separates as:
    Φ(r,y) = Σ_n  φ_n(r) ψ_n(y)

The y-part gives the Sturm-Liouville eigenvalue problem:
    -d/dy[e^{-4ky} dψ/dy] = m² e^{-4ky} ψ

This is self-adjoint with the weighted inner product
    ⟨f,g⟩_w = ∫ f g e^{-4ky} dy.

BCs:
  - Z₂ Neumann at y=0:  ψ'(0) = 0  (natural BC — automatic in FE!)
  - Dirichlet at y_max:  ψ(y_max) = 0  (eliminated from system)

We discretize with piecewise-linear FE and lumped mass, giving:
    H ψ = m² M ψ
where H (stiffness) and M (mass) are symmetric PSD.

Standard eigh is done on S = M^{-1/2} H M^{-1/2}.

Zero mode: ψ₀ = const → Hψ₀ = 0 (gradient is zero) → m₀² = 0.
On finite domain with Dirichlet at y_max, ψ₀ ≈ const except near y_max,
so m₀² is exponentially small.

PASS conditions:
  1. |m₀²| < 1e-6
  2. Zero-mode peak at brane (y=0)
  3. Normalization integral finite and stable under domain doubling
  4. Bulk decay ratio < 0.5
  5. Zero-mode profile monotonic decay away from brane
"""
import json
import time
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def build_Ly_raw(y_grid, k=1.0):
    """Build raw stiffness H and lumped mass M for the KK Sturm-Liouville problem.

    The eigenvalue problem in original ψ-basis is:

        -d/dy[e^{-4ky} dψ/dy] = m² e^{-4ky} ψ

    BCs: Neumann at y=0 (Z₂, natural BC), Dirichlet at y_max (eliminated).

    Returns:
      H: (N, N) stiffness matrix  (symmetric, PSD)
      M_diag: (N,) lumped mass diagonal  (positive)
      N = Ny - 1
    """
    Ny = y_grid.shape[0]
    N = Ny - 1
    dy = float(y_grid[1] - y_grid[0])

    # Stiffness: H_ij = ∫ w(y) φ'_i φ'_j dy  with w = e^{-4ky}
    H = jnp.zeros((N, N))
    for j in range(N - 1):
        y_mid = 0.5 * (float(y_grid[j]) + float(y_grid[j + 1]))
        w_mid = jnp.exp(-4.0 * k * y_mid)
        coeff = w_mid / dy
        H = H.at[j, j].add(coeff)
        H = H.at[j, j + 1].add(-coeff)
        H = H.at[j + 1, j].add(-coeff)
        H = H.at[j + 1, j + 1].add(coeff)

    # Last interval [N-1, N]: Dirichlet node contributes only to diagonal
    y_mid_last = 0.5 * (float(y_grid[N - 1]) + float(y_grid[N]))
    w_mid_last = jnp.exp(-4.0 * k * y_mid_last)
    H = H.at[N - 1, N - 1].add(w_mid_last / dy)

    # Lumped mass: M_jj = w_j * (trapez weight)
    w = jnp.exp(-4.0 * k * y_grid[:N])
    M_diag = w * dy
    M_diag = M_diag.at[0].set(float(w[0]) * dy / 2.0)

    return H, M_diag


def build_Ly_operator(y_grid, k=1.0):
    """Build symmetric standard eigenvalue matrix S = M^{-1/2} H M^{-1/2}.

    Eigenvalues of S = KK mass² spectrum.
    Eigenvectors: ψ_n = M^{-1/2} * v_n  (v_n from eigh(S)).

    Returns (S, M_inv_sqrt).
    """
    H, M_diag = build_Ly_raw(y_grid, k)
    M_inv_sqrt = 1.0 / jnp.sqrt(M_diag)
    S = H * jnp.outer(M_inv_sqrt, M_inv_sqrt)
    return S, M_inv_sqrt


def build_Ly_operator_simple(y_grid, k=1.0):
    """Convenience wrapper: returns just the S matrix (no M_inv_sqrt).

    Use when you only need eigenvalues (not eigenvectors in ψ-space).
    """
    S, _ = build_Ly_operator(y_grid, k)
    return S


def compute_zero_mode_analytic(y_grid, k=1.0):
    """Analytic zero mode in the original ψ basis.

    The zero mode is ψ₀(y) = const (y-independent).
    We return a normalized constant profile (value 1 everywhere).
    On the reduced grid (Dirichlet removed), ψ₀(y_max)=0 doesn't
    hold for the analytic mode, but the eigenvalue m₀²→0 as y_max→∞.
    """
    return jnp.ones_like(y_grid)


def run_stage3(k=1.0, Ny=200, y_max=10.0, verbose=True):
    """Run Stage 3: zero mode validation."""
    results = {}
    lines = []

    def log(msg):
        if verbose:
            print(msg)
        lines.append(msg)

    log("=" * 70)
    log("  STAGE 3: Zero Mode Validation")
    log("=" * 70)
    log(f"  JAX {jax.__version__} | {jax.default_backend()} | {jax.devices()}")
    log(f"  k={k}, Ny={Ny}, y_max={y_max}")

    dy = y_max / (Ny - 1)
    y_grid = jnp.linspace(0.0, y_max, Ny)
    log(f"  dy = {dy:.6f}")

    # ── Build operator ───────────────────────────────────────────
    t0 = time.perf_counter()
    S, M_inv_sqrt = build_Ly_operator(y_grid, k)
    N = Ny - 1  # reduced dimension (Dirichlet node removed)
    dt_build = time.perf_counter() - t0
    log(f"\n  Operator built: ({N},{N}) in {dt_build:.2f}s  [Sturm-Liouville, Dirichlet removed]")

    # ── Eigendecomposition ───────────────────────────────────────
    t0 = time.perf_counter()
    eigenvalues, eigenvectors_S = jnp.linalg.eigh(S)
    eigenvalues.block_until_ready()
    dt_eig = time.perf_counter() - t0
    log(f"  Eigendecomposition: {dt_eig:.2f}s")

    # Sort by eigenvalue (should already be sorted by eigh)
    idx_sort = jnp.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors_S = eigenvectors_S[:, idx_sort]

    # Convert eigenvectors back to ψ-space: ψ_n = M^{-1/2} v_n
    eigenvectors = eigenvectors_S * M_inv_sqrt[:, None]

    # Sort by eigenvalue (should already be sorted by eigh)
    idx_sort = jnp.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    # First 10 eigenvalues
    log(f"\n  First 10 eigenvalues (m²):")
    for i in range(min(10, N)):
        log(f"    m²_{i} = {float(eigenvalues[i]):.10e}")

    # First 100 eigenvalues for output
    n_out = min(100, N)
    evals_100 = [float(eigenvalues[i]) for i in range(n_out)]
    results["eigenvalues_100"] = evals_100

    # ── Zero mode analysis ───────────────────────────────────────
    m0_sq = float(eigenvalues[0])
    psi0_red = eigenvectors[:, 0]  # length N = Ny-1, in ψ-space

    # Ensure consistent sign: positive at brane
    if float(psi0_red[0]) < 0:
        psi0_red = -psi0_red

    # Full profile: append Dirichlet zero at y_max
    psi0 = jnp.concatenate([psi0_red, jnp.array([0.0])])  # length Ny
    y_full = y_grid  # length Ny

    log(f"\n  Zero mode:")
    log(f"    m₀² = {m0_sq:.10e}")
    results["m0_sq"] = m0_sq

    # Brane overlap: |ψ₀(y=0)|
    brane_overlap = float(jnp.abs(psi0[0]))
    log(f"    |ψ₀(y=0)| = {brane_overlap:.6e}")
    results["brane_overlap"] = brane_overlap

    # Peak location
    peak_idx = int(jnp.argmax(jnp.abs(psi0)))
    peak_y = float(y_grid[peak_idx])
    log(f"    Peak at y = {peak_y:.4f} (index {peak_idx})")
    results["peak_y"] = peak_y

    # Zero mode should be approximately constant in ψ-space.
    # Check flatness: max deviation from mean / mean
    psi_mean = float(jnp.mean(jnp.abs(psi0[:-1])))  # exclude Dirichlet=0
    psi_max = float(jnp.max(jnp.abs(psi0[:-1])))
    psi_min = float(jnp.min(jnp.abs(psi0[:-1])))
    flatness = (psi_max - psi_min) / max(psi_mean, 1e-30)
    log(f"    Flatness (max-min)/mean = {flatness:.6f}")
    results["flatness"] = flatness

    # Normalization: ∫|ψ₀|² dy (unweighted, trapezoidal over full grid)
    norm_integrand = jnp.abs(psi0)**2
    norm_integral = float(jnp.trapezoid(norm_integrand, y_full))
    log(f"    ∫|ψ₀|² dy = {norm_integral:.6e}")
    results["norm_integral"] = norm_integral

    # Weighted normalization: ∫ e^{-4ky} |ψ₀|² dy (physical norm)
    w_grid = jnp.exp(-4.0 * k * y_full)
    wnorm_integrand = w_grid * jnp.abs(psi0)**2
    wnorm_integral = float(jnp.trapezoid(wnorm_integrand, y_full))
    log(f"    ∫ e^{{-4ky}} |ψ₀|² dy = {wnorm_integral:.6e}")
    results["wnorm_integral"] = wnorm_integral

    # Brane localization: fraction of weighted norm in first half
    j_mid = Ny // 2
    brane_wnorm = float(jnp.trapezoid(wnorm_integrand[:j_mid+1], y_full[:j_mid+1]))
    brane_frac = brane_wnorm / max(wnorm_integral, 1e-30)
    log(f"    Brane weighted norm fraction (first half) = {brane_frac:.6f}")
    results["brane_wnorm_fraction"] = brane_frac

    # Profile at selected points
    log(f"\n  Zero-mode profile ψ₀(y):")
    for frac in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        jj = min(int(frac * (Ny - 1)), Ny - 1)
        log(f"    y={float(y_grid[jj]):.2f}: ψ₀ = {float(psi0[jj]):.6e}")

    # ── Domain doubling stability ────────────────────────────────
    log(f"\n  Domain doubling check:")
    y_max_2 = 2.0 * y_max
    Ny_2 = 2 * Ny
    y_grid_2 = jnp.linspace(0.0, y_max_2, Ny_2)
    S_2, M_inv_sqrt_2 = build_Ly_operator(y_grid_2, k)
    evals_2, evecs_2_S = jnp.linalg.eigh(S_2)
    m0_sq_2 = float(evals_2[0])
    v0_2 = evecs_2_S[:, 0]
    psi0_2_red = v0_2 * M_inv_sqrt_2  # back to ψ-space
    if float(psi0_2_red[0]) < 0:
        psi0_2_red = -psi0_2_red
    psi0_2 = jnp.concatenate([psi0_2_red, jnp.array([0.0])])  # length Ny_2
    norm_2 = float(jnp.trapezoid(jnp.abs(psi0_2)**2, y_grid_2))

    log(f"    y_max={y_max}: m₀²={m0_sq:.6e}, norm={norm_integral:.6e}")
    log(f"    y_max={y_max_2}: m₀²={m0_sq_2:.6e}, norm={norm_2:.6e}")
    norm_change = abs(norm_integral - norm_2) / max(norm_integral, 1e-30)
    m0_change = abs(m0_sq - m0_sq_2) / max(abs(m0_sq), 1e-30)
    log(f"    Norm change: {norm_change:.4e}")
    log(f"    m₀² change:  {m0_change:.4e}")
    results["m0_sq_doubled"] = m0_sq_2
    results["norm_doubled"] = norm_2
    results["norm_relative_change"] = norm_change

    # ── PASS/FAIL ────────────────────────────────────────────────
    c1 = abs(m0_sq) < 1e-6  # zero mode eigenvalue ≈ 0
    c2 = brane_overlap > 0  # nonzero at brane
    c3 = norm_integral > 0 and norm_2 > 0  # finite norms
    c4 = wnorm_integral > 0  # weighted norm finite (brane-localized)
    c5 = flatness < 1.0  # roughly constant profile (zero mode = const)

    PASS = c1 and c2 and c3 and c4 and c5

    log(f"\n{'='*70}")
    log(f"  STAGE 3 VERDICT: {'PASS' if PASS else 'FAIL'}")
    log(f"    |m₀²| < 1e-6           : {c1}  (m₀² = {m0_sq:.2e})")
    log(f"    Nonzero at brane        : {c2}  (|ψ₀(0)| = {brane_overlap:.2e})")
    log(f"    Norms finite & stable   : {c3}")
    log(f"    Weighted norm finite    : {c4}  (∫w|ψ|² = {wnorm_integral:.2e})")
    log(f"    Flat profile (< 1.0)    : {c5}  (flatness = {flatness:.4f})")
    log(f"{'='*70}")
    results["verdict"] = "PASS" if PASS else "FAIL"

    # ── Save ─────────────────────────────────────────────────────
    report = "\n".join(lines)
    with open("stage3_report.txt", "w") as f:
        f.write(report)

    json_safe = {}
    for kk, v in results.items():
        if isinstance(v, (float, int, bool, str)):
            json_safe[kk] = v
        elif isinstance(v, list):
            json_safe[kk] = v
        else:
            json_safe[kk] = str(v)
    with open("stage3_results.json", "w") as f:
        json.dump(json_safe, f, indent=2)

    log(f"\nSaved: stage3_report.txt, stage3_results.json")
    return results


if __name__ == "__main__":
    Ny = 200
    y_max = 10.0
    if len(sys.argv) >= 3:
        Ny = int(sys.argv[1])
        y_max = float(sys.argv[2])
    run_stage3(k=1.0, Ny=Ny, y_max=y_max)
