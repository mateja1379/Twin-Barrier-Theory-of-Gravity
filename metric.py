"""
metric.py – 5-D warped-brane background metric and curvature tensors.

Background metric (Randall-Sundrum-like):

    ds² = e^{-2ky} (-dt² + dr² + r² dΩ₂²) + dy²

Index convention:  A, B ∈ {0,1,2,3,4} = (t, r, θ, φ, y)

All functions are JAX-traceable so they can be JIT-compiled, differentiated,
and vmap-ed over grid points.
"""
import jax
import jax.numpy as jnp
from functools import partial

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

if __name__ == "__main__":
    import time
    k = 1.0
    coords = jnp.array([0.0, 5.0, jnp.pi / 4, jnp.pi / 3, 0.5])

    print("metric.py self-test")
    print(f"  coords = {coords}")

    g = background_metric(coords[1], coords[2], coords[4], k=k)
    gi = background_metric_inv(coords[1], coords[2], coords[4], k=k)
    print(f"  g diag = {jnp.diag(g)}")
    print(f"  g^-1 diag = {jnp.diag(gi)}")
    print(f"  g @ g^-1 diag = {jnp.diag(g @ gi)}")

    # Analytic Christoffel
    Ga = christoffel_analytic(coords, k=k)
    print(f"  Γ^t_ty = {float(Ga[0,0,4]):.6f}  (expect -1.0)")
    print(f"  Γ^y_rr = {float(Ga[4,1,1]):.6f}  (expect {-k*jnp.exp(-2*k*0.5):.6f})")

    # Ricci
    t0 = time.perf_counter()
    R_ab = ricci_tensor(coords, k=k)
    dt = time.perf_counter() - t0
    R_exact = ricci_tensor_analytic(coords, k=k)
    print(f"\n  R_ab (numerical) diag  = {jnp.diag(R_ab)}")
    print(f"  R_ab (analytic) diag   = {jnp.diag(R_exact)}")
    print(f"  max |R_num - R_exact|  = {float(jnp.max(jnp.abs(R_ab - R_exact))):.2e}")

    Rs = ricci_scalar(coords, k=k)
    Rs_ex = ricci_scalar_analytic(k=k)
    print(f"\n  Ricci scalar (num)     = {float(Rs):.6f}")
    print(f"  Ricci scalar (exact)   = {Rs_ex:.6f}")
    print(f"  Ricci time (incl JIT)  = {dt*1000:.1f} ms")
    print("  OK")
