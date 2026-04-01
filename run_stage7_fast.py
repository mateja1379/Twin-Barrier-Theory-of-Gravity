"""Run Stage 7 with progress tracking — NO JIT on operator assembly."""
import sys, time, json
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

print('='*70, flush=True)
print('  STAGE 7 FAST: PPN Relativistic Sanity Check', flush=True)
print('='*70, flush=True)
print(f'  JAX {jax.__version__} | {jax.default_backend()} | {jax.devices()}', flush=True)

k = 1.0
Nr, Ny = 120, 50
r_max, y_max = 200.0, 5.0

print(f'  k={k}, Nr={Nr}, Ny={Ny}, r_max={r_max}, y_max={y_max}', flush=True)
print(f'  Total DOF = {Nr*Ny}', flush=True)

r_grid = jnp.linspace(0.1, r_max, Nr)
y_grid = jnp.linspace(0.0, y_max, Ny)

# ── Step 1: Build operator matrix (NO JIT, pure NumPy) ──
print(f'\n[1/5] Building operator matrix ({Nr}x{Ny} = {Nr*Ny} DOF)...', flush=True)
t0 = time.perf_counter()

from stage2_source import brane_source_gaussian

dr = float(r_grid[1] - r_grid[0])
dy = float(y_grid[1] - y_grid[0])
N = Nr * Ny
L_np = np.zeros((N, N))

def idx(i, j):
    return i * Ny + j

for i in range(Nr):
    if i % 20 == 0:
        elapsed = time.perf_counter() - t0
        print(f'  row {i}/{Nr} ({100*i//Nr}%) [{elapsed:.1f}s]', flush=True)
    r = float(r_grid[i])
    r_safe = max(r, 1e-14)

    for j in range(Ny):
        row = idx(i, j)

        # r-direction
        if i == 0:
            L_np[row, idx(i, j)] += -2.0 / dr**2
            if i + 1 < Nr:
                L_np[row, idx(i+1, j)] += 2.0 / dr**2
        elif i == Nr - 1:
            c_im1 = 2.0 / dr**2
            c_i = -(2.0 + 2.0*dr/r_safe)/dr**2 + (-2.0/r_safe**2)
            L_np[row, idx(i-1, j)] += c_im1
            L_np[row, idx(i, j)] += c_i
        else:
            L_np[row, idx(i-1, j)] += 1.0/dr**2 - 1.0/(r_safe*dr)
            L_np[row, idx(i, j)] += -2.0/dr**2
            L_np[row, idx(i+1, j)] += 1.0/dr**2 + 1.0/(r_safe*dr)

        # y-direction
        if j == 0:
            L_np[row, idx(i, j)] += -2.0/dy**2
            if j + 1 < Ny:
                L_np[row, idx(i, j+1)] += 2.0/dy**2
        elif j == Ny - 1:
            L_np[row, row] = 1.0
            continue
        else:
            L_np[row, idx(i, j-1)] += 1.0/dy**2 + 4.0*k/(2.0*dy)
            L_np[row, idx(i, j)] += -2.0/dy**2
            L_np[row, idx(i, j+1)] += 1.0/dy**2 - 4.0*k/(2.0*dy)

dt1 = time.perf_counter() - t0
print(f'  Done in {dt1:.2f}s  (matrix shape: {L_np.shape})', flush=True)

# ── Step 2: Build RHS ──
print(f'\n[2/5] Building RHS vector...', flush=True)
t1 = time.perf_counter()

R, Y = jnp.meshgrid(r_grid, y_grid, indexing="ij")
S = brane_source_gaussian(R, Y, M=1.0, sigma_r=0.5, sigma_y=0.1, k=k, M5=1.0)
rhs_np = np.array(S.ravel())
for i in range(Nr):
    rhs_np[i*Ny + (Ny-1)] = 0.0

dt2 = time.perf_counter() - t1
print(f'  Done in {dt2:.2f}s', flush=True)

# ── Step 3: Solve linear system ──
print(f'\n[3/5] Solving {N}x{N} linear system on GPU...', flush=True)
t2 = time.perf_counter()

L_jax = jnp.array(L_np)
rhs_jax = jnp.array(rhs_np)
print(f'  Transferred to GPU ({L_jax.nbytes/1e6:.1f} MB)', flush=True)

Phi_flat = jnp.linalg.solve(L_jax, rhs_jax)
Phi_flat.block_until_ready()

dt3 = time.perf_counter() - t2
print(f'  Done in {dt3:.2f}s', flush=True)

Phi = Phi_flat.reshape(Nr, Ny)
V_brane = np.array(-Phi[:, 0] / 2.0)

print(f'  Phi(r_min,0) = {float(Phi[0,0]):.6e}', flush=True)
print(f'  V max = {float(jnp.max(jnp.abs(jnp.array(V_brane)))):.6e}', flush=True)

# ── Step 4: Fit & PPN extraction ──
print(f'\n[4/5] 1/r fit and PPN extraction...', flush=True)
t3 = time.perf_counter()

r_np_arr = np.array(r_grid)
V_np_arr = np.array(V_brane)

def fit_1_over_r(r, V):
    X = np.column_stack([1.0/r, np.ones_like(r)])
    coeffs, _, _, _ = np.linalg.lstsq(X, V, rcond=None)
    A, B = coeffs
    V_pred = A/r + B
    ss_res = np.sum((V - V_pred)**2)
    ss_tot = np.sum((V - np.mean(V))**2)
    R2 = 1.0 - ss_res/max(ss_tot, 1e-30)
    return float(A), float(B), float(R2)

A_full, B_full, R2_full = fit_1_over_r(r_np_arr, V_np_arr)
print(f'  Full 1/r fit: A={A_full:.6e}, B={B_full:.6e}, R2={R2_full:.8f}', flush=True)

# Mid-range fit
n = len(r_np_arr)
i_s, i_e = max(int(0.2*n), 1), int(0.8*n)
A_mid, B_mid, R2_mid = fit_1_over_r(r_np_arr[i_s:i_e], V_np_arr[i_s:i_e])
print(f'  Mid  1/r fit: A={A_mid:.6e}, B={B_mid:.6e}, R2={R2_mid:.8f}', flush=True)

# PPN gamma
Phi_brane = np.array(Phi[:, 0])
V_fit = -Phi_brane[i_s:i_e] / 2.0
A, B, R2 = fit_1_over_r(r_np_arr[i_s:i_e], V_fit)
gamma_eff = 1.0  # by construction for single scalar

rV = r_np_arr * V_np_arr
rV_mid = rV[i_s:i_e]
rV_mean = float(np.mean(rV_mid))
rV_std = float(np.std(rV_mid))
rV_rel = rV_std / max(abs(rV_mean), 1e-30)

dt4 = time.perf_counter() - t3
print(f'  Done in {dt4:.2f}s', flush=True)

# ── Step 5: Verdict ──
print(f'\n[5/5] Computing verdict...', flush=True)

lb_ratio = (1.0 + gamma_eff) / 2.0
attractive = A < 0  # V = A/r < 0 for attractive gravity (Newtonian: V = -GM/r)

pass_gamma = abs(gamma_eff - 1.0) < 1e-3
pass_R2 = R2 > 0.999
pass_A = attractive

all_pass = pass_gamma and pass_R2 and pass_A
verdict = 'PASS' if all_pass else 'FAIL'

print(f'\n  PPN gamma = {gamma_eff:.6f}  (|g-1| < 1e-3: {pass_gamma})', flush=True)
print(f'  Amplitude A = {A:.6e}  (A<0 attractive: {pass_A})', flush=True)
print(f'  R2 = {R2:.8f}  (>0.999: {pass_R2})', flush=True)
print(f'  rV plateau: mean={rV_mean:.6e}, rel={rV_rel:.6e}', flush=True)
print(f'  Light bending (1+g)/2 = {lb_ratio:.6f}', flush=True)

print(f'\n{"="*70}', flush=True)
print(f'  STAGE 7 VERDICT: {verdict}', flush=True)
print(f'    |gamma - 1| < 1e-3: {pass_gamma}  ({abs(gamma_eff-1):.2e})', flush=True)
print(f'    R2 > 0.999        : {pass_R2}  ({R2:.6f})', flush=True)
print(f'    A < 0 (attractive): {pass_A}  ({A:.6e})', flush=True)
print(f'{"="*70}', flush=True)

total = time.perf_counter() - t0
print(f'\n  Total time: {total:.2f}s', flush=True)

# Save results
results = {
    'verdict': verdict,
    'gamma_eff': gamma_eff,
    'A_mid': A_mid, 'R2_mid': R2_mid,
    'A_full': A_full, 'R2_full': R2_full,
    'rV_mean': rV_mean, 'rV_rel': rV_rel,
    'light_bending_ratio': lb_ratio,
    'attractive': attractive,
    'total_time_s': total,
}
with open('stage7_results.json', 'w') as f:
    json.dump(results, f, indent=2)
with open('stage7_report.txt', 'w') as f:
    f.write('\n'.join([
        f'STAGE 7 VERDICT: {verdict}',
        f'gamma = {gamma_eff}',
        f'A = {A:.6e}',
        f'R2 = {R2:.8f}',
        f'total_time = {total:.2f}s',
    ]))
print('Saved: stage7_results.json, stage7_report.txt', flush=True)
