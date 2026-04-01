#!/usr/bin/env python3
"""Quick test: does the solver converge with L_comp=3?"""
from instanton_bounce import *

lam, eta, u, k = 0.10, 0.037, 1.0, 1.0
Phi_true, Phi_false = find_vacua(lam, u, eta)
print(f"Phi_true={Phi_true:.6f}, Phi_false={Phi_false:.6f}")
print(f"eta/lam={eta/lam:.4f}")

params = {"lam": lam, "u": u, "eta": eta, "k": k, "v0": Phi_true}
grid = {"Nr": 60, "Ny": 30, "Rmax": 20.0, "L": 3.0}
solver = {"r0": 7.0, "delta": 1.5, "max_iter": 60, "tol": 1e-10}

res = solve_bounce(params, grid, solver, verbose=True)
print(f"\nS_B = {res['SB']:.6f}, converged={res['converged']}")
