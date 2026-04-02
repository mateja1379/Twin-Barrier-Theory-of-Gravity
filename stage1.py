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
from functools import partial

# ======================================================================
#  Inlined: metric.py
# ======================================================================

"""
metric.py – 5-D warped-brane background metric and curvature tensors.

Background metric (Randall-Sundrum-like):

    ds² = e^{-2ky} (-dt² + dr² + r² dΩ₂²) + dy²

Index convention:  A, B ∈ {0,1,2,3,4} = (t, r, θ, φ, y)

All functions are JAX-traceable so they can be JIT-compiled, differentiated,
and vmap-ed over grid points.
"""

# Enable float64 for numerical precision in curvature computations
jax.config.update("jax_enable_x64", True)

# Dimension
NDIM = 5

# ── Helpers ──────────────────────────────────────────────────────────────────

_EPS_R = 1e-12     # floor for r² to avoid 1/r² blow-up
_EPS_SIN = 1e-12   # floor for sin²θ


# ── Background metric ───────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=())
def background_metric(r: jnp.ndarray,
                      theta: jnp.ndarray,
                      y: jnp.ndarray,
                      k: float = 1.0) -> jnp.ndarray:
    """Return the 5×5 background metric g_{AB} at a single point.

    Parameters
    ----------
    r, theta, y : scalars (or 0-d arrays)
    k           : AdS₅ curvature scale  (Λ₅ = -6k²)

    Returns
    -------
    g : (5, 5) array
    """
    w = jnp.exp(-2.0 * k * y)          # warp factor squared
    r2 = jnp.maximum(r**2, _EPS_R)     # safeguard
    sin2 = jnp.maximum(jnp.sin(theta)**2, _EPS_SIN)

    g = jnp.zeros((NDIM, NDIM))
    g = g.at[0, 0].set(-w)             # g_tt
    g = g.at[1, 1].set(w)              # g_rr
    g = g.at[2, 2].set(w * r2)         # g_θθ
    g = g.at[3, 3].set(w * r2 * sin2)  # g_φφ
    g = g.at[4, 4].set(1.0)            # g_yy
    return g


@partial(jax.jit, static_argnums=())
def background_metric_inv(r: jnp.ndarray,
                          theta: jnp.ndarray,
                          y: jnp.ndarray,
                          k: float = 1.0) -> jnp.ndarray:
    """Return g^{AB} (inverse metric) at a single point."""
    w = jnp.exp(-2.0 * k * y)
    winv = jnp.exp(2.0 * k * y)
    r2 = jnp.maximum(r**2, _EPS_R)
    sin2 = jnp.maximum(jnp.sin(theta)**2, _EPS_SIN)

    gi = jnp.zeros((NDIM, NDIM))
    gi = gi.at[0, 0].set(-winv)
    gi = gi.at[1, 1].set(winv)
    gi = gi.at[2, 2].set(winv / r2)
    gi = gi.at[3, 3].set(winv / (r2 * sin2))
    gi = gi.at[4, 4].set(1.0)
    return gi


# ── Christoffel symbols via autodiff ────────────────────────────────────────

def _metric_component(coords, k, A, B):
    """g_{AB}(coords) where coords = (t, r, θ, φ, y).

    Used as a scalar function for JAX autodiff.
    """
    r, theta, y = coords[1], coords[2], coords[4]
    g = background_metric(r, theta, y, k=k)
    return g[A, B]


@partial(jax.jit, static_argnums=(1,))
def christoffel_all(coords: jnp.ndarray, k: float = 1.0) -> jnp.ndarray:
    """Christoffel symbols Γ^A_{BC} at a single point.

    Parameters
    ----------
    coords : (5,) array  [t, r, θ, φ, y]
    k      : warp parameter

    Returns
    -------
    Gamma : (5, 5, 5) array  Γ^A_{BC}
    """
    r, theta, y = coords[1], coords[2], coords[4]
    ginv = background_metric_inv(r, theta, y, k=k)

    # dg_{AB}/dx^C  via autodiff
    dg = jnp.zeros((NDIM, NDIM, NDIM))
    for A in range(NDIM):
        for B in range(A, NDIM):
            grad_fn = jax.grad(lambda c: _metric_component(c, k, A, B))
            dg_AB = grad_fn(coords)                        # (5,)
            dg = dg.at[A, B, :].set(dg_AB)
            if B != A:
                dg = dg.at[B, A, :].set(dg_AB)            # symmetry

    # Γ^A_{BC} = ½ g^{AD} (∂_B g_{DC} + ∂_C g_{DB} - ∂_D g_{BC})
    Gamma = jnp.zeros((NDIM, NDIM, NDIM))
    for A in range(NDIM):
        for B in range(NDIM):
            for C in range(B, NDIM):
                val = 0.0
                for D in range(NDIM):
                    val = val + 0.5 * ginv[A, D] * (
                        dg[D, C, B] + dg[D, B, C] - dg[B, C, D]
                    )
                Gamma = Gamma.at[A, B, C].set(val)
                if C != B:
                    Gamma = Gamma.at[A, C, B].set(val)     # symmetry in lower indices
    return Gamma


# ── Analytic Christoffel (fast, for validation & production) ─────────────────

@partial(jax.jit, static_argnums=(1,))
def christoffel_analytic(coords: jnp.ndarray, k: float = 1.0) -> jnp.ndarray:
    """Christoffel symbols computed analytically for the warped background.

    Much faster than autodiff version; use this in production loops.

    Non-zero components for ds² = e^{-2ky}(-dt² + dr² + r²dΩ₂²) + dy²:

    Γ^t_{ty} = Γ^t_{yt} = -k
    Γ^r_{ry} = Γ^r_{yr} = -k
    Γ^r_{θθ} = -r
    Γ^r_{φφ} = -r sin²θ
    Γ^θ_{rθ} = Γ^θ_{θr} = 1/r
    Γ^θ_{θy} = Γ^θ_{yθ} = -k
    Γ^θ_{φφ} = -sinθ cosθ
    Γ^φ_{rφ} = Γ^φ_{φr} = 1/r
    Γ^φ_{θφ} = Γ^φ_{φθ} = cosθ/sinθ
    Γ^φ_{φy} = Γ^φ_{yφ} = -k
    Γ^y_{tt} = -k e^{-2ky}
    Γ^y_{rr} = -k e^{-2ky}
    Γ^y_{θθ} = -k e^{-2ky} r²
    Γ^y_{φφ} = -k e^{-2ky} r² sin²θ
    """
    r = coords[1]
    theta = coords[2]
    y_coord = coords[4]

    r_safe = jnp.maximum(jnp.abs(r), 1e-12)
    sin_th = jnp.sin(theta)
    cos_th = jnp.cos(theta)
    sin_th_safe = jnp.where(jnp.abs(sin_th) < 1e-12,
                            jnp.sign(sin_th + 1e-30) * 1e-12, sin_th)
    sin2 = sin_th**2
    w = jnp.exp(-2.0 * k * y_coord)

    G = jnp.zeros((NDIM, NDIM, NDIM))

    # Γ^t_{ty}
    G = G.at[0, 0, 4].set(-k)
    G = G.at[0, 4, 0].set(-k)

    # Γ^r_{ry}
    G = G.at[1, 1, 4].set(-k)
    G = G.at[1, 4, 1].set(-k)

    # Γ^r_{θθ}
    G = G.at[1, 2, 2].set(-r_safe)

    # Γ^r_{φφ}
    G = G.at[1, 3, 3].set(-r_safe * sin2)

    # Γ^θ_{rθ}
    G = G.at[2, 1, 2].set(1.0 / r_safe)
    G = G.at[2, 2, 1].set(1.0 / r_safe)

    # Γ^θ_{θy}
    G = G.at[2, 2, 4].set(-k)
    G = G.at[2, 4, 2].set(-k)

    # Γ^θ_{φφ}
    G = G.at[2, 3, 3].set(-sin_th * cos_th)

    # Γ^φ_{rφ}
    G = G.at[3, 1, 3].set(1.0 / r_safe)
    G = G.at[3, 3, 1].set(1.0 / r_safe)

    # Γ^φ_{θφ}
    cot = cos_th / sin_th_safe
    G = G.at[3, 2, 3].set(cot)
    G = G.at[3, 3, 2].set(cot)

    # Γ^φ_{φy}
    G = G.at[3, 3, 4].set(-k)
    G = G.at[3, 4, 3].set(-k)

    # Γ^y_{tt}
    G = G.at[4, 0, 0].set(-k * w)

    # Γ^y_{rr}   (positive: -½ ∂_y(+e^{-2ky}) = +kw)
    G = G.at[4, 1, 1].set(k * w)

    # Γ^y_{θθ}   (positive)
    G = G.at[4, 2, 2].set(k * w * r_safe**2)

    # Γ^y_{φφ}   (positive)
    G = G.at[4, 3, 3].set(k * w * r_safe**2 * sin2)

    return G


# ── Riemann → Ricci via pure double-autodiff from smooth metric ──────────────

def _smooth_metric_flat(coords: jnp.ndarray, k: float) -> jnp.ndarray:
    """Smooth 5×5 metric g_{AB} without safeguards – for autodiff only.

    coords = [t, r, θ, φ, y].  No jnp.maximum/jnp.where so gradients
    are exact everywhere the coordinates are non-singular.
    """
    r = coords[1]
    theta = coords[2]
    y_coord = coords[4]
    w = jnp.exp(-2.0 * k * y_coord)
    r2 = r ** 2
    sin2 = jnp.sin(theta) ** 2

    g = jnp.zeros((NDIM, NDIM))
    g = g.at[0, 0].set(-w)
    g = g.at[1, 1].set(w)
    g = g.at[2, 2].set(w * r2)
    g = g.at[3, 3].set(w * r2 * sin2)
    g = g.at[4, 4].set(1.0)
    return g


def _christoffel_from_metric(coords: jnp.ndarray, k: float) -> jnp.ndarray:
    """Γ^A_{BC} computed by differentiating the smooth metric via autodiff."""
    g = _smooth_metric_flat(coords, k)
    ginv = jnp.linalg.inv(g)

    # dg[A,B,C] = ∂g_{AB}/∂x^C
    dg = jax.jacfwd(lambda c: _smooth_metric_flat(c, k))(coords)
    # dg shape: (5,5,5) → dg[A,B,C] = ∂g_{AB}/∂x^C

    # Γ^A_{BC} = ½ g^{AD} (∂_B g_{DC} + ∂_C g_{DB} - ∂_D g_{BC})
    Gamma = 0.5 * jnp.einsum('ad,dcb->abc', ginv, dg) \
          + 0.5 * jnp.einsum('ad,dbc->abc', ginv, dg) \
          - 0.5 * jnp.einsum('ad,bcd->abc', ginv, dg)
    return Gamma


@partial(jax.jit, static_argnums=(1,))
def ricci_tensor(coords: jnp.ndarray, k: float = 1.0) -> jnp.ndarray:
    """Ricci tensor R_{AB} via double-autodiff of the smooth metric.

    Computes Christoffel from metric (1st deriv), then differentiates
    Christoffel (2nd deriv) to get Ricci.  No analytic shortcuts,
    so this works for any metric deformation in later stages.
    """
    Gamma = _christoffel_from_metric(coords, k)

    dGamma = jax.jacfwd(lambda c: _christoffel_from_metric(c, k))(coords)
    # dGamma[A, B, C, D] = ∂Γ^A_{BC}/∂x^D

    # R_{AB} = ∂_C Γ^C_{AB} - ∂_B Γ^C_{AC} + Γ^C_{CD} Γ^D_{AB} - Γ^C_{BD} Γ^D_{AC}
    term1 = jnp.einsum('cabc->ab', dGamma)     # ∂_C Γ^C_{AB}
    term2 = jnp.einsum('cacb->ab', dGamma)     # ∂_B Γ^C_{AC}
    term3 = jnp.einsum('ccd,dab->ab', Gamma, Gamma)
    term4 = jnp.einsum('cbd,dac->ab', Gamma, Gamma)

    return term1 - term2 + term3 - term4


@partial(jax.jit, static_argnums=(1,))
def ricci_scalar(coords: jnp.ndarray, k: float = 1.0) -> jnp.ndarray:
    """Ricci scalar R = g^{AB} R_{AB}."""
    r, theta, y = coords[1], coords[2], coords[4]
    ginv = background_metric_inv(r, theta, y, k=k)
    Rab = ricci_tensor(coords, k=k)
    return jnp.einsum('ab,ab->', ginv, Rab)


# ── Analytic Ricci for validation ────────────────────────────────────────────

def ricci_tensor_analytic(coords: jnp.ndarray, k: float = 1.0) -> jnp.ndarray:
    """Analytic Ricci tensor for the RS warped background.

    For ds² = e^{-2ky}(-dt²+dr²+r²dΩ₂²)+dy²  the space is locally AdS₅:
        R_{AB} = -4k² g_{AB}    for A,B ∈ {t,r,θ,φ}  (warp-factor piece)
        R_{yy} = -4k²
    Overall: R_{AB} = -4k² g_{AB}   (since g_{yy}=1).
    And R = -20k².
    """
    r, theta, y = coords[1], coords[2], coords[4]
    g = background_metric(r, theta, y, k=k)
    return -4.0 * k**2 * g


def ricci_scalar_analytic(k: float = 1.0) -> float:
    """Analytic Ricci scalar: R = g^{AB}(-4k² g_{AB}) = -4k² × 5 = -20k²."""
    return -20.0 * k**2


# ── Quick self-test ──────────────────────────────────────────────────────────


# ======================================================================
#  Inlined: grid.py
# ======================================================================

"""
grid.py – Compactified 2D coordinate grid for the Einstein-DeTurck solver.

Coordinate maps
----------------
x ∈ [0, 1]  →  radial coordinate r
z ∈ [0, 1]  →  bulk (extra-dimension) coordinate y

    r(x) = r_h + (1 - x) / (x + eps) * r_scale
    y(z) = y_scale * arctanh(beta * z) / beta

The compactification puts spatial infinity at x = 0 and the horizon at x = 1,
while z = 0 is the brane and z → 1 maps to the deep bulk.
"""


# ── Coordinate maps ─────────────────────────────────────────────────────────

@jax.jit
def x_to_r(x: jnp.ndarray,
           r_h: float = 1.0,
           r_scale: float = 1.0,
           eps: float = 1e-10) -> jnp.ndarray:
    """Map compactified x ∈ (0, 1] to radial r ∈ [r_h, ∞)."""
    return r_h + (1.0 - x) / (x + eps) * r_scale


@jax.jit
def z_to_y(z: jnp.ndarray,
           y_scale: float = 1.0,
           beta: float = 0.95) -> jnp.ndarray:
    """Map compactified z ∈ [0, 1) to bulk coordinate y ∈ [0, ∞).

    Uses arctanh stretched by y_scale.  beta < 1 keeps the argument
    of arctanh safely away from ±1.
    """
    return y_scale * jnp.arctanh(beta * z) / beta


# ── Jacobians (needed for PDE in compactified coords) ───────────────────────

@jax.jit
def dr_dx(x: jnp.ndarray,
          r_scale: float = 1.0,
          eps: float = 1e-10) -> jnp.ndarray:
    """dr/dx  (always negative: r decreases as x grows toward the horizon)."""
    return -r_scale / (x + eps)**2


@jax.jit
def dy_dz(z: jnp.ndarray,
          y_scale: float = 1.0,
          beta: float = 0.95) -> jnp.ndarray:
    """dy/dz  (positive)."""
    return y_scale / (1.0 - (beta * z)**2)


# ── Grid construction ───────────────────────────────────────────────────────

def make_grid(Nx: int = 30,
              Nz: int = 30,
              r_h: float = 1.0,
              r_scale: float = 1.0,
              y_scale: float = 1.0,
              beta: float = 0.95,
              eps: float = 1e-10):
    """Build a 2D Chebyshev-like grid in compactified coordinates.

    Returns
    -------
    dict with keys:
        x, z        – 1-D arrays of compactified coords (interior points)
        X, Z        – 2-D meshgrid arrays  (Nx, Nz)
        R, Y        – 2-D arrays of physical coords r, y
        dR_dX, dY_dZ – 2-D Jacobian arrays
    """
    # Chebyshev-Lobatto nodes mapped to (0,1) – avoids endpoints exactly
    # to sidestep coordinate singularities.
    i_x = jnp.arange(1, Nx + 1)
    i_z = jnp.arange(1, Nz + 1)
    x = 0.5 * (1.0 - jnp.cos(jnp.pi * i_x / (Nx + 1)))   # ∈ (0, 1)
    z = 0.5 * (1.0 - jnp.cos(jnp.pi * i_z / (Nz + 1)))   # ∈ (0, 1)

    X, Z = jnp.meshgrid(x, z, indexing="ij")  # shape (Nx, Nz)

    R = x_to_r(X, r_h=r_h, r_scale=r_scale, eps=eps)
    Y = z_to_y(Z, y_scale=y_scale, beta=beta)

    dR_dX = dr_dx(X, r_scale=r_scale, eps=eps)
    dY_dZ = dy_dz(Z, y_scale=y_scale, beta=beta)

    return dict(x=x, z=z, X=X, Z=Z, R=R, Y=Y, dR_dX=dR_dX, dY_dZ=dY_dZ,
                Nx=Nx, Nz=Nz, r_h=r_h)


# ── Quick self-test ─────────────────────────────────────────────────────────


# ======================================================================
#  Inlined: deturck.py
# ======================================================================

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
