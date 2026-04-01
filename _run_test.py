#!/usr/bin/env python3
"""Quick test of bounce solver with corrected parameters."""
import sys
sys.path.insert(0, '.')
from instanton_bounce import *
import numpy as np

lam, u, eta = 0.30, 1.0, 0.11
print(f'eta/lam = {eta/lam:.4f} (critical = {2/(3*np.sqrt(3)):.4f})')
Phi_true, Phi_false = find_vacua(lam, u, eta)
print(f'Phi_true = {Phi_true:.6f}, Phi_false = {Phi_false:.6f}')
print(f'V(true)  = {V_potential(Phi_true, lam, u, eta):.6e}')
print(f'V(false) = {V_potential(Phi_false, lam, u, eta):.6e}')
print(f'DeltaV   = {V_potential(Phi_false, lam, u, eta) - V_potential(Phi_true, lam, u, eta):.6e}')

SB_est = thin_wall_estimate(lam, u, eta, k=1.0, L=20.0)
print(f'Thin-wall S_B estimate = {SB_est:.2f}')

params = {'lam': lam, 'u': u, 'eta': eta, 'k': 1.0, 'v0': Phi_true, 'vL': Phi_true}
grid = {'Nr': 40, 'Ny': 16, 'Rmax': 30.0, 'L': 20.0}
solver = {'r0': 10.0, 'delta': 2.0, 'max_iter': 80, 'tol': 1e-8}
print('\nSolving bounce (Nr=40, Ny=16)...')
res = solve_bounce(params, grid, solver, verbose=True)
print(f'\nS_B = {res["SB"]:.6f}')
print(f'Converged: {res["converged"]}')
print(f'Time: {res["time_s"]:.2f}s')
