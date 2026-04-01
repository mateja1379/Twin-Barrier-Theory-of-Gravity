"""Scan for parameters where the twin-barrier potential has two real minima."""
import numpy as np

def V(Phi, lam, u, eta):
    return (lam/4)*(Phi**2 - u**2)**2 - eta*u**3*Phi

def dV(Phi, lam, u, eta):
    return lam*Phi*(Phi**2 - u**2) - eta*u**3

u = 1.0

# V'(Φ) = λΦ³ - λu²Φ - ηu³ = 0
# For two minima need discriminant > 0
# Cubic: Φ³ - u²Φ - (η/λ)u³ = 0 (divide by λ)
# Standard depressed cubic: t³ + pt + q = 0
# with p = -u², q = -(η/λ)u³
# Discriminant: Δ = -4p³ - 27q²
# For THREE real roots: Δ > 0
# -4(-u²)³ - 27(-(η/λ)u³)² > 0
# 4u⁶ - 27(η/λ)²u⁶ > 0
# 4 > 27(η/λ)²
# η/λ < 2/(3√3) ≈ 0.3849

print("Condition for two minima: η/λ < 2/(3√3) ≈", 2/(3*np.sqrt(3)))
print()

print(f"{'lam':>6} {'eta':>6} {'eta/lam':>8} {'roots':>40} {'#real':>6} {'Phi_t':>10} {'Phi_f':>10} {'DV':>12}")
print("-"*100)

for lam in [0.03, 0.05, 0.08, 0.10, 0.15]:
    for eta in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
        ratio = eta/lam
        coeffs = [lam, 0.0, -lam*u**2, -eta*u**3]
        roots = np.roots(coeffs)
        real_roots = np.sort(roots[np.abs(roots.imag) < 1e-10].real)
        n_real = len(real_roots)
        
        if n_real >= 3:
            # Potential at the extrema
            Vs = [V(r, lam, u, eta) for r in real_roots]
            # Minima are at root[0] and root[2], max at root[1]
            Phi_t = real_roots[0] if Vs[0] < Vs[2] else real_roots[2]
            Phi_f = real_roots[2] if Vs[0] < Vs[2] else real_roots[0]
            # Actually: these are extrema of V'. The minima of V are where V''> 0
            d2V_vals = [lam*(3*r**2 - u**2) for r in real_roots]
            minima = [real_roots[k] for k in range(3) if d2V_vals[k] > 0]
            if len(minima) >= 2:
                V_min = [V(m, lam, u, eta) for m in minima]
                i_t = np.argmin(V_min)
                i_f = np.argmax(V_min)
                DV = V_min[i_f] - V_min[i_t]
                print(f"{lam:>6.3f} {eta:>6.3f} {ratio:>8.4f} {str(np.round(real_roots,4)):>40} {n_real:>6} {minima[i_t]:>10.4f} {minima[i_f]:>10.4f} {DV:>12.6e}")
                
                # Thin-wall estimate
                SB_est = 16.65 * lam**2 / eta**3
                print(f"       S_B^est = {SB_est:.2f}")
            else:
                print(f"{lam:>6.3f} {eta:>6.3f} {ratio:>8.4f} {str(np.round(real_roots,4)):>40} {n_real:>6} {'<2 minima':>10}")
        else:
            print(f"{lam:>6.3f} {eta:>6.3f} {ratio:>8.4f} {str(np.round(real_roots,4)):>40} {n_real:>6} {'no barrier':>10}")
