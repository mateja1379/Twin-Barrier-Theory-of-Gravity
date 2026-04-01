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
import jax
import jax.numpy as jnp
from functools import partial


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

if __name__ == "__main__":
    g = make_grid(Nx=10, Nz=10)
    print("Grid self-test")
    print(f"  x range  : [{float(g['x'].min()):.4f}, {float(g['x'].max()):.4f}]")
    print(f"  z range  : [{float(g['z'].min()):.4f}, {float(g['z'].max()):.4f}]")
    print(f"  R range  : [{float(g['R'].min()):.4f}, {float(g['R'].max()):.4f}]")
    print(f"  Y range  : [{float(g['Y'].min()):.4f}, {float(g['Y'].max()):.4f}]")
    print(f"  R shape  : {g['R'].shape}")
    print("  OK")
