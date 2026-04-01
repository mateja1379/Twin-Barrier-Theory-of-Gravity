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
    Nr = r_grid.shape[0]
    Ny = y_grid.shape[0]
    N = Nr * Ny

    dr = r_grid[1] - r_grid[0]
    dy = y_grid[1] - y_grid[0]

    L = jnp.zeros((N, N))

    def idx(i, j):
        return i * Ny + j

    for i in range(Nr):
        r = r_grid[i]
        r_safe = jnp.maximum(r, 1e-14)

        for j in range(Ny):
            row = idx(i, j)

            # ── r-direction ──────────────────────────────────────
            if i == 0:
                # Neumann at r_min: ∂_rΦ=0 → ghost Φ_{-1}=Φ_{+1}
                # ∂²_r ≈ (2Φ_{1,j} - 2Φ_{0,j}) / dr²
                cr_center = -2.0 / dr**2
                cr_plus = 2.0 / dr**2
                L = L.at[row, idx(i, j)].add(cr_center)
                if i + 1 < Nr:
                    L = L.at[row, idx(i + 1, j)].add(cr_plus)
                # (2/r) ∂_r = 0 by symmetry

            elif i == Nr - 1:
                # Robin BC: ∂_r(rΦ) = 0  →  ∂_rΦ = -Φ/r
                # Ghost point: Φ_{Nr} = Φ_{Nr-2} - (2dr/r) Φ_{Nr-1}
                #
                # ∂²_r = (Φ_Nr - 2Φ_{i} + Φ_{i-1}) / dr²
                #       = (Φ_{i-1} - (2dr/r)Φ_i - 2Φ_i + Φ_{i-1}) / dr²
                #       = (2Φ_{i-1} - (2 + 2dr/r)Φ_i) / dr²
                c_im1 = 2.0 / dr**2
                c_i = -(2.0 + 2.0 * dr / r_safe) / dr**2

                # (2/r)∂_r = (2/r)(Φ_Nr - Φ_{i-1})/(2dr)
                #           = (2/r)(-(2dr/r)Φ_i)/(2dr) = -2/r² Φ_i
                c_i += -2.0 / r_safe**2

                L = L.at[row, idx(i - 1, j)].add(c_im1)
                L = L.at[row, idx(i,     j)].add(c_i)

            else:
                # Interior
                cr_minus = 1.0 / dr**2 - 1.0 / (r_safe * dr)
                cr_center = -2.0 / dr**2
                cr_plus = 1.0 / dr**2 + 1.0 / (r_safe * dr)
                L = L.at[row, idx(i - 1, j)].add(cr_minus)
                L = L.at[row, idx(i, j)].add(cr_center)
                L = L.at[row, idx(i + 1, j)].add(cr_plus)

            # ── y-direction ──────────────────────────────────────
            if j == 0:
                # Z₂ at brane: ghost Φ_{i,-1} = Φ_{i,+1}
                cy_center = -2.0 / dy**2
                cy_plus = 2.0 / dy**2
                L = L.at[row, idx(i, j)].add(cy_center)
                if j + 1 < Ny:
                    L = L.at[row, idx(i, j + 1)].add(cy_plus)
                # -4k ∂_y = 0 by Z₂

            elif j == Ny - 1:
                # Dirichlet: Φ = 0 at y-boundary
                L = L.at[row, row].set(1.0)
                continue

            else:
                cy_minus = 1.0 / dy**2 + 4.0 * k / (2.0 * dy)
                cy_center = -2.0 / dy**2
                cy_plus = 1.0 / dy**2 - 4.0 * k / (2.0 * dy)
                L = L.at[row, idx(i, j - 1)].add(cy_minus)
                L = L.at[row, idx(i, j)].add(cy_center)
                L = L.at[row, idx(i, j + 1)].add(cy_plus)

    return L


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
    for i in range(Nr):
        row = i * Ny + (Ny - 1)
        rhs = rhs.at[row].set(0.0)

    return rhs


# ── Solver ───────────────────────────────────────────────────────────────────

@partial(jit, static_argnums=(2, 3, 4))
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
