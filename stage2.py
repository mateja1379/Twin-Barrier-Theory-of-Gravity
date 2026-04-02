"""
stage2_source.py – Stage 2: Linearized graviton on the warped brane.

Physics
-------
For a static point mass M on the brane (y=0), the linearized equation
for the Newtonian potential Ψ(r,y) in the RS-II background is:

    [∂²_r + (2/r)∂_r + e^{2ky}(∂²_y - 4k ∂_y)] Ψ = S(r,y)

We solve the RS graviton equation in (r,y):

    [∂²_r + (2/r)∂_r + ∂²_y - 4k ∂_y] Φ = S_eff(r,y)

and reconstruct the brane potential:  V(r) = -Φ(r, y=0) / 2.

Boundary conditions:
    r → 0   : regularity  ∂_r Φ = 0   (Neumann)
    r → r_max: Robin BC  ∂_r(rΦ) = 0  (consistent with Φ ~ A/r decay)
    y = 0   : Z₂ symmetry
    y → y_max: Φ → 0  (Dirichlet, deep bulk)
"""
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

jax.config.update("jax_enable_x64", True)


# ── Regularized brane source ────────────────────────────────────────────────

def brane_source_gaussian(r: jnp.ndarray, y: jnp.ndarray,
                          M: float = 1.0, sigma_r: float = 0.5,
                          sigma_y: float = 0.1, k: float = 1.0,
                          M5: float = 1.0) -> jnp.ndarray:
    """Regularized point-mass source on the brane.

    S(r,y) = -(M / (4π M₅³ σ_r³ (2π)^{3/2})) exp(-r²/(2σ_r²))
             × (1 / (σ_y √(2π))) exp(-y²/(2σ_y²))

    The y-profile approximates δ(y) as σ_y → 0.
    The r-profile approximates a point mass as σ_r → 0.

    Parameters
    ----------
    r, y  : arrays (same shape)
    M     : source mass
    sigma_r : radial regularization width
    sigma_y : bulk (y) regularization width
    k     : AdS curvature
    M5    : 5D Planck mass

    Returns
    -------
    S : array, same shape as r
    """
    # Radial profile: spherically symmetric Gaussian in 3D
    rho_r = jnp.exp(-0.5 * (r / sigma_r) ** 2) / (sigma_r ** 3 * (2 * jnp.pi) ** 1.5)

    # Bulk profile: Gaussian approximation of delta(y)
    rho_y = jnp.exp(-0.5 * (y / sigma_y) ** 2) / (sigma_y * jnp.sqrt(2 * jnp.pi))

    # Overall normalization: -(M / (4π M₅³))
    prefactor = -M / (4.0 * jnp.pi * M5 ** 3)

    return prefactor * rho_r * rho_y


# ── 2D operator assembly (Robin BC at outer r) ──────────────────────────────

def build_operator_matrix(r_grid: jnp.ndarray, y_grid: jnp.ndarray,
                          k: float = 1.0) -> jnp.ndarray:
    """Build the 2D finite-difference operator matrix for:

        L[Φ] = ∂²_r Φ + (2/r) ∂_r Φ + ∂²_y Φ - 4k ∂_y Φ

    on a uniform (r,y) grid.  Returns dense matrix L of shape (N, N)
    where N = Nr × Ny and the solution vector is Φ flattened in row-major.

    Boundary conditions:
        r = r_min : regularity ∂_r Φ = 0  (Neumann via ghost point)
        r = r_max : Robin  ∂_r(rΦ) = 0  →  ∂_rΦ = -Φ/r
                    (ghost: Φ_{Nr} = Φ_{Nr-2} - (2dr/r)Φ_{Nr-1})
        y = 0     : Z₂ symmetry  (ghost: Φ_{i,-1} = Φ_{i,+1})
        y = y_max : Φ = 0  (Dirichlet, deep bulk)
    """
    import numpy as _np
    r_arr = _np.asarray(r_grid)
    y_arr = _np.asarray(y_grid)
    Nr = len(r_arr)
    Ny = len(y_arr)
    N = Nr * Ny

    dr = float(r_arr[1] - r_arr[0])
    dy = float(y_arr[1] - y_arr[0])

    L = _np.zeros((N, N), dtype=_np.float64)

    def idx(i, j):
        return i * Ny + j

    for i in range(Nr):
        r = float(r_arr[i])
        r_safe = max(r, 1e-14)

        for j in range(Ny):
            row = idx(i, j)

            # ── r-direction ──────────────────────────────────────
            if i == 0:
                cr_center = -2.0 / dr**2
                cr_plus = 2.0 / dr**2
                L[row, idx(i, j)] += cr_center
                if i + 1 < Nr:
                    L[row, idx(i + 1, j)] += cr_plus

            elif i == Nr - 1:
                c_im1 = 2.0 / dr**2
                c_i = -(2.0 + 2.0 * dr / r_safe) / dr**2
                c_i += -2.0 / r_safe**2

                L[row, idx(i - 1, j)] += c_im1
                L[row, idx(i,     j)] += c_i

            else:
                cr_minus = 1.0 / dr**2 - 1.0 / (r_safe * dr)
                cr_center = -2.0 / dr**2
                cr_plus = 1.0 / dr**2 + 1.0 / (r_safe * dr)
                L[row, idx(i - 1, j)] += cr_minus
                L[row, idx(i, j)] += cr_center
                L[row, idx(i + 1, j)] += cr_plus

            # ── y-direction ──────────────────────────────────────
            if j == 0:
                cy_center = -2.0 / dy**2
                cy_plus = 2.0 / dy**2
                L[row, idx(i, j)] += cy_center
                if j + 1 < Ny:
                    L[row, idx(i, j + 1)] += cy_plus

            elif j == Ny - 1:
                L[row, :] = 0.0
                L[row, row] = 1.0
                continue

            else:
                cy_minus = 1.0 / dy**2 + 4.0 * k / (2.0 * dy)
                cy_center = -2.0 / dy**2
                cy_plus = 1.0 / dy**2 - 4.0 * k / (2.0 * dy)
                L[row, idx(i, j - 1)] += cy_minus
                L[row, idx(i, j)] += cy_center
                L[row, idx(i, j + 1)] += cy_plus

    return jnp.array(L)


def build_rhs(r_grid: jnp.ndarray, y_grid: jnp.ndarray,
              k: float = 1.0, M: float = 1.0,
              sigma_r: float = 0.5, sigma_y: float = 0.1,
              M5: float = 1.0) -> jnp.ndarray:
    """Build RHS vector for the linear system L Φ = S.

    Only the y=y_max boundary row is zeroed (Dirichlet in y).
    The outer r boundary uses Robin BC (PDE row, source passes through).

    Returns
    -------
    rhs : (Nr*Ny,) array
    """
    Nr = r_grid.shape[0]
    Ny = y_grid.shape[0]

    R, Y = jnp.meshgrid(r_grid, y_grid, indexing='ij')
    S = brane_source_gaussian(R, Y, M=M, sigma_r=sigma_r, sigma_y=sigma_y,
                              k=k, M5=M5)

    rhs = S.ravel()

    # Zero out rows corresponding to Dirichlet BC (y=y_max only)
    import numpy as _np
    rhs_np = _np.array(rhs, copy=True)
    for i in range(Nr):
        row = i * Ny + (Ny - 1)
        rhs_np[row] = 0.0

    return jnp.array(rhs_np)


# ── Solver ───────────────────────────────────────────────────────────────────

def solve_linearized(r_grid: jnp.ndarray, y_grid: jnp.ndarray,
                     k: float = 1.0, M: float = 1.0,
                     M5: float = 1.0,
                     sigma_r: float = 0.5,
                     sigma_y: float = 0.1) -> jnp.ndarray:
    """Solve the linearized graviton equation and return Φ(r,y).

    Returns
    -------
    Phi : (Nr, Ny) array
    """
    Nr = r_grid.shape[0]
    Ny = y_grid.shape[0]

    # Can't JIT the loop-based build, so call outside
    L = build_operator_matrix(r_grid, y_grid, k=k)
    rhs = build_rhs(r_grid, y_grid, k=k, M=M,
                    sigma_r=sigma_r, sigma_y=sigma_y, M5=M5)

    # Dense solve
    Phi_flat = jnp.linalg.solve(L, rhs)
    return Phi_flat.reshape(Nr, Ny)


# ── Extract brane potential ──────────────────────────────────────────────────

def brane_potential(Phi: jnp.ndarray) -> jnp.ndarray:
    """Newtonian potential on the brane: V(r) = -Φ(r, y=0) / 2.

    The y=0 slice is the first column (j=0).
    """
    return -Phi[:, 0] / 2.0


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("stage2_source.py self-test")

    # Small uniform grid for testing
    Nr, Ny = 40, 30
    r_max = 20.0
    y_max = 5.0
    k = 1.0

    r_grid = jnp.linspace(0.1, r_max, Nr)
    y_grid = jnp.linspace(0.0, y_max, Ny)

    print(f"  Grid: Nr={Nr}, Ny={Ny}")
    print(f"  r ∈ [{float(r_grid[0]):.2f}, {float(r_grid[-1]):.2f}]")
    print(f"  y ∈ [{float(y_grid[0]):.2f}, {float(y_grid[-1]):.2f}]")

    # Build and solve
    t0 = time.perf_counter()
    L = build_operator_matrix(r_grid, y_grid, k=k)
    rhs = build_rhs(r_grid, y_grid, k=k, M=1.0, sigma_r=0.5, sigma_y=0.2)
    Phi_flat = jnp.linalg.solve(L, rhs)
    Phi = Phi_flat.reshape(Nr, Ny)
    dt = time.perf_counter() - t0

    V_brane = brane_potential(Phi)
    print(f"  Solve time: {dt:.2f} s")
    print(f"  Phi(r_min, y=0) = {float(Phi[0, 0]):.6e}")
    print(f"  V_brane max     = {float(jnp.max(jnp.abs(V_brane))):.6e}")
    print(f"  V_brane[5]      = {float(V_brane[5]):.6e}")
    print("  OK")
