"""
deturck.py – Einstein-DeTurck residuals for the 5-D warped-brane system.

The DeTurck trick:
    ξ^A  = g^{BC} (Γ^A_{BC} − Γ̄^A_{BC})
where Γ̄ are Christoffel symbols of the *reference* metric ḡ.

Einstein-DeTurck residual:
    E_{AB} = R_{AB} − ∇_{(A} ξ_{B)} − (2/3) Λ₅ g_{AB}

with Λ₅ = −6 k² (in units where M₅ = 1).

For Stage 1 the physical metric equals the reference metric (pure background),
so ξ^A = 0 identically and E_{AB} = R_{AB} − (2/3)Λ₅ g_{AB}.
"""
import jax
import jax.numpy as jnp
from functools import partial

from metric import (
    NDIM,
    background_metric,
    background_metric_inv,
    christoffel_analytic,
    ricci_tensor,
)


# ── DeTurck vector ξ^A ──────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(1,))
def deturck_vector(coords: jnp.ndarray, k: float = 1.0) -> jnp.ndarray:
    """DeTurck gauge vector  ξ^A = g^{BC}(Γ^A_{BC} − Γ̄^A_{BC}).

    For Stage 1 the physical metric IS the reference metric, so ξ ≡ 0
    up to numerical precision.  This function computes it from scratch
    as a correctness check.

    Parameters
    ----------
    coords : (5,) array  [t, r, θ, φ, y]
    k      : warp parameter

    Returns
    -------
    xi : (5,) array
    """
    r, theta, y = coords[1], coords[2], coords[4]
    ginv = background_metric_inv(r, theta, y, k=k)

    Gamma = christoffel_analytic(coords, k=k)          # physical
    GammaBar = christoffel_analytic(coords, k=k)        # reference = same for Stage 1

    diff = Gamma - GammaBar                              # should be 0

    # ξ^A = g^{BC} diff^A_{BC}
    xi = jnp.einsum('bc,abc->a', ginv, diff)
    return xi


@partial(jax.jit, static_argnums=(1,))
def deturck_vector_norm_sq(coords: jnp.ndarray, k: float = 1.0) -> jnp.ndarray:
    """ξ² = g_{AB} ξ^A ξ^B  (should be ~0 for Stage 1)."""
    r, theta, y = coords[1], coords[2], coords[4]
    g = background_metric(r, theta, y, k=k)
    xi = deturck_vector(coords, k=k)
    return jnp.einsum('a,ab,b->', xi, g, xi)


# ── Covariant derivative of ξ ────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(1,))
def nabla_xi_symmetrized(coords: jnp.ndarray, k: float = 1.0) -> jnp.ndarray:
    """∇_{(A} ξ_{B)} = ½ (∇_A ξ_B + ∇_B ξ_A).

    Uses autodiff to compute ∂_A ξ^C and then lowers.
    """
    r, theta, y = coords[1], coords[2], coords[4]
    g = background_metric(r, theta, y, k=k)
    Gamma = christoffel_analytic(coords, k=k)

    # ξ^C(coords)
    xi = deturck_vector(coords, k=k)

    # ∂_A ξ^C  via jacfwd
    dxi = jax.jacfwd(lambda c: deturck_vector(c, k=k))(coords)
    # dxi shape: (5, 5)  →  dxi[C, A] = ∂_A ξ^C

    # ∇_A ξ^C = ∂_A ξ^C + Γ^C_{AD} ξ^D
    nabla_xi_up = dxi + jnp.einsum('cad,d->ca', Gamma, xi)
    # nabla_xi_up[C, A] = ∇_A ξ^C

    # Lower: ∇_A ξ_B = g_{BC} ∇_A ξ^C
    nabla_xi_low = jnp.einsum('bc,ca->ab', g, nabla_xi_up)
    # nabla_xi_low[A, B] = ∇_A ξ_B

    # Symmetrize
    return 0.5 * (nabla_xi_low + nabla_xi_low.T)


# ── Einstein-DeTurck residual E_{AB} ────────────────────────────────────────

@partial(jax.jit, static_argnums=(1,))
def einstein_deturck_residual(coords: jnp.ndarray,
                              k: float = 1.0) -> jnp.ndarray:
    """E_{AB} = R_{AB} − ∇_{(A}ξ_{B)} − (2/3) Λ₅ g_{AB}

    where Λ₅ = −6k²  ⟹  −(2/3)Λ₅ = 4k².

    For the exact background solution E_{AB} should vanish:
        R_{AB} = −4k² g_{AB}   and  ξ = 0
        E_{AB} = −4k² g_{AB} − 0 + 4k² g_{AB} = 0  ✓

    Parameters
    ----------
    coords : (5,) array
    k      : warp parameter

    Returns
    -------
    E : (5, 5) array
    """
    r, theta, y = coords[1], coords[2], coords[4]
    g = background_metric(r, theta, y, k=k)

    Rab = ricci_tensor(coords, k=k)
    nxi = nabla_xi_symmetrized(coords, k=k)

    Lambda5 = -6.0 * k**2
    # -(2/3)Λ₅ = -(2/3)(-6k²) = +4k²  →  E = R - ∇ξ + 4k² g
    cosmological = -(2.0 / 3.0) * Lambda5   # = +4k²

    E = Rab - nxi + cosmological * g
    return E


# ── Convenience: evaluate on a grid (vmap) ──────────────────────────────────

def residual_on_grid(R_grid: jnp.ndarray,
                     Y_grid: jnp.ndarray,
                     k: float = 1.0,
                     theta: float = None,
                     t: float = 0.0) -> dict:
    """Evaluate all DeTurck quantities over a 2-D grid of (r, y).

    Parameters
    ----------
    R_grid, Y_grid : (Nx, Nz) arrays of physical r, y values
    k              : warp parameter
    theta          : polar angle  (defaults to π/4 to avoid axis singularity)

    Returns
    -------
    dict with keys:  xi2, E_AB_max, E_AB, R_scalar
    """
    if theta is None:
        theta = jnp.pi / 4.0

    Nx, Nz = R_grid.shape

    # Flatten
    r_flat = R_grid.ravel()
    y_flat = Y_grid.ravel()
    N = r_flat.shape[0]

    # Build coords array: (N, 5)
    coords_all = jnp.stack([
        jnp.full(N, t),
        r_flat,
        jnp.full(N, theta),
        jnp.full(N, 0.0),       # φ = 0
        y_flat,
    ], axis=-1)

    # vmap over batch dimension
    xi_batch = jax.vmap(lambda c: deturck_vector(c, k=k))(coords_all)
    xi2_batch = jax.vmap(lambda c: deturck_vector_norm_sq(c, k=k))(coords_all)
    E_batch = jax.vmap(lambda c: einstein_deturck_residual(c, k=k))(coords_all)

    # Also compute Ricci scalar for reporting
    from metric import ricci_scalar
    R_scal = jax.vmap(lambda c: ricci_scalar(c, k=k))(coords_all)

    xi2_grid = xi2_batch.reshape(Nx, Nz)
    E_grid = E_batch.reshape(Nx, Nz, NDIM, NDIM)
    R_scal_grid = R_scal.reshape(Nx, Nz)

    return dict(
        xi2=xi2_grid,
        E_AB=E_grid,
        E_AB_max=float(jnp.max(jnp.abs(E_grid))),
        xi2_max=float(jnp.max(jnp.abs(xi2_grid))),
        R_scalar=R_scal_grid,
        R_scalar_mean=float(jnp.mean(R_scal_grid)),
    )


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    coords = jnp.array([0.0, 5.0, jnp.pi / 4, jnp.pi / 3, 0.5])
    k = 1.0

    print("deturck.py self-test")
    xi = deturck_vector(coords, k=k)
    xi2 = deturck_vector_norm_sq(coords, k=k)
    print(f"  ξ^A       = {xi}")
    print(f"  ξ²        = {float(xi2):.2e}")

    E = einstein_deturck_residual(coords, k=k)
    print(f"  E diag    = {jnp.diag(E)}")
    print(f"  max|E_AB| = {float(jnp.max(jnp.abs(E))):.2e}")
    print("  OK")
