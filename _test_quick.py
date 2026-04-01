"""Quick sanity test for the bounce solver."""
from instanton_bounce import *
import numpy as np

# Test vacua
lam, u, eta = 0.08, 1.0, 0.06
t, f = find_vacua(lam, u, eta)
print(f'Vacua: true={t:.6f}, false={f:.6f}')
print(f'V(true)={V_potential(t,lam,u,eta):.6e}, V(false)={V_potential(f,lam,u,eta):.6e}')
print(f'DV = {V_potential(f,lam,u,eta)-V_potential(t,lam,u,eta):.6e}')
print(f'Thin-wall est: {thin_wall_estimate(lam,eta):.2f}')
print()

# Quick small solve
params = {'lam': lam, 'u': u, 'eta': eta, 'k': 1.0, 'v0': t, 'vL': t}
gp = {'Nr': 40, 'Ny': 16, 'Rmax': 20.0, 'L': 20.0}
sp = {'r0': 7.0, 'delta': 2.0, 'max_iter': 50, 'tol': 1e-8}
res = solve_bounce(params, gp, sp, verbose=True)
print(f'\nSB = {res["SB"]:.4f}, converged={res["converged"]}')
