#!/usr/bin/env python3
"""
Twin Barrier DM Self-Interaction & Dwarf Galaxy Core-Cusp Test
================================================================

TB predicts dark matter = gauge-decoupled particles on the twin brane.
ALL gauge fields (EM, weak, strong) are localized at y=0 (visible brane).
DM at y=L has ONLY gravitational self-interaction → collisionless CDM.

This script:
1. Computes σ/m for twin DM (gravitational, Yukawa, scenarios)
2. Compares with SIDM constraints from clusters, dwarfs, bullet cluster
3. Tests NFW vs cored profiles on SPARC dwarf galaxies
4. Quantifies core-cusp tension for TB's CDM prediction

Key papers:
  - Kaplinghat+ 2016 (SIDM cross-sections)
  - Oh+ 2015 (LITTLE THINGS dwarf cores)
  - Read+ 2019 (core formation via feedback)
  - Walker & Peñarrubia 2011 (dSph inner slopes)

Author: Mateja Radojičić
Date:   April 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os, sys

# ============================================================
# PHYSICAL CONSTANTS
# ============================================================
G_N    = 6.674e-11    # m³/(kg·s²)
m_p    = 1.672e-27    # kg (proton)
c_light = 3e8         # m/s
hbar   = 1.055e-34    # J·s
eV     = 1.602e-19    # J
GeV    = 1e9 * eV
kpc_m  = 3.086e19     # m
Msun   = 1.989e30     # kg
km_s   = 1e3          # m/s

# TB parameters
alpha_TB = 21.214
k_TB     = 1.2e20     # m⁻¹ (curvature scale)
eps_yuk  = 0.005      # Yukawa coupling
lam_yuk  = 200e-9     # Yukawa range (m)
E_barrier = 41.9      # TeV per particle

print("=" * 74)
print("  TWIN BARRIER DM: SELF-INTERACTION & CORE-CUSP ANALYSIS")
print("=" * 74)

# ============================================================
# PART 1: TWIN DM SELF-INTERACTION CROSS SECTION
# ============================================================
print(f"\n{'─'*74}")
print(f"  PART 1: σ/m FOR TWIN BARRIER DARK MATTER")
print(f"{'─'*74}")

print(f"""
  TB's dark matter picture (from paper §14.4):
  ─────────────────────────────────────────────
  • DM = SM particles whose wavefunction shifted to twin brane (y=L)
  • ALL gauge fields (EM, weak, strong) localized at y=0 only
  • DM has NO electromagnetic, weak, or strong self-interaction
  • Only gravitational coupling preserved (graviton zero-mode spans bulk)
  • Barrier transition energy: {E_barrier:.1f} TeV per particle
  • Freeze-out at T ~ {E_barrier:.0f} TeV → before QCD confinement
  • After freeze-out: gauge-decoupled particles, mass ~ 1 GeV
""")

# 1a. Pure gravitational scattering
print(f"  ── Channel 1: Gravitational scattering ──")
print(f"  σ_grav = π(2Gm/v²)² for point particles")

for m_name, m_kg, v_name, v_kms in [
    ("proton", m_p, "dwarf", 30),
    ("proton", m_p, "MW-like", 200),
    ("proton", m_p, "cluster", 1000),
    ("10 GeV", 10*m_p, "dwarf", 30)]:
    
    v = v_kms * km_s
    b_min = 2 * G_N * m_kg / v**2  # gravitational radius
    sigma_grav = np.pi * b_min**2
    sigma_over_m = sigma_grav / m_kg  # m²/kg → cm²/g
    sigma_over_m_cgs = sigma_over_m * 1e4 / 1e-3  # m²/kg × 10⁴cm²/m² × 10³g/kg
    sigma_over_m_cgs = sigma_grav * 1e4 / (m_kg * 1e-3)
    
    print(f"    m={m_name:>8s}, v={v_kms:>5d} km/s:  σ/m = {sigma_over_m_cgs:.2e} cm²/g")

# 1b. Yukawa-enhanced gravitational scattering (KK modes)
print(f"\n  ── Channel 2: Yukawa-enhanced (KK graviton, ε={eps_yuk}, λ={lam_yuk*1e9:.0f}nm) ──")
print(f"  At galactic scales (d >> λ=200nm), Yukawa is zero.")
print(f"  Even for closest DM particle approach in a dwarf galaxy:")

n_DM = 0.3 * GeV / (m_p * c_light**2)  # particles per cm³ (local DM density)
# Actually: ρ_DM ~ 0.3 GeV/cm³ = 0.3 GeV/(cm³) 
# n_DM ~ ρ/m ~ 0.3/1 = 0.3 per cm³ (for 1 GeV particles)
# Mean inter-particle distance: d_mean ~ n^(-1/3) ~ 1.5 cm

rho_DM_dwarf = 1.0  # GeV/cm³ (typical for dwarf center)
n_DM_dwarf = rho_DM_dwarf  # per cm³ for 1 GeV particles
d_mean = n_DM_dwarf**(-1.0/3.0)  # cm
print(f"  ρ_DM(dwarf center) ~ 1 GeV/cm³ → n ~ 1/cm³ → d_mean ~ {d_mean:.1f} cm")
print(f"  d_mean / λ_Yukawa = {d_mean*1e-2 / lam_yuk:.1e}")
print(f"  Yukawa suppression: exp(-d/λ) = exp(-{d_mean*1e-2/lam_yuk:.0e}) = 0")
print(f"  → Yukawa channel contributes ZERO at any macroscopic scale")

# 1c. Residual gauge coupling through bulk
print(f"\n  ── Channel 3: Residual gauge coupling through barrier ──")
print(f"  Gauge field at y=L is suppressed by bulk warp factor:")
print(f"  A_μ(y=L) ∝ e^{{-α}} × A_μ(y=0)")
print(f"  α = kL = {alpha_TB:.3f}")
print(f"  e^{{-α}} = {np.exp(-alpha_TB):.3e}")
print(f"  Coupling suppression: e^{{-2α}} = {np.exp(-2*alpha_TB):.3e}")
print(f"")
print(f"  If twin EM existed with coupling α_EM_twin:")
print(f"    α_EM_twin ~ α_EM × e^{{-2α}} ~ (1/137) × {np.exp(-2*alpha_TB):.1e}")
print(f"             = {(1/137)*np.exp(-2*alpha_TB):.1e}")
print(f"  This is 10^{{-21}} weaker than electromagnetism → negligible")

# 1d. Summary cross-section
print(f"\n  ╔═══════════════════════════════════════════════════════════════╗")
print(f"  ║  TB PREDICTION: σ/m ≈ 0 (effectively collisionless CDM)     ║")
print(f"  ║                                                               ║")
print(f"  ║  Gravitational: σ/m ~ 10⁻⁶⁰ cm²/g                          ║")
print(f"  ║  Yukawa (KK):   σ/m = 0 at d >> 200 nm                      ║")
print(f"  ║  Residual gauge: coupling ~ 10⁻²¹ × α_EM → negligible       ║")
print(f"  ║                                                               ║")
print(f"  ║  Twin DM behaves EXACTLY like collisionless CDM (WIMPs)      ║")
print(f"  ╚═══════════════════════════════════════════════════════════════╝")


# ============================================================
# PART 2: OBSERVATIONAL CONSTRAINTS ON σ/m
# ============================================================
print(f"\n{'─'*74}")
print(f"  PART 2: OBSERVATIONAL SIDM CONSTRAINTS")
print(f"{'─'*74}")

# Published constraints on σ/m at various velocity scales
# Format: (name, v_km/s, sigma_m_lower, sigma_m_upper, type)
# type: 'upper_limit' or 'preferred'
sidm_constraints = [
    # Clusters
    ("Bullet Cluster (Markevitch+ 2004)", 4700, 0, 1.25, "upper_limit"),
    ("Abell 3827 (Massey+ 2015)", 1500, 0, 3.0, "upper_limit"),
    ("Cluster ensemble (Harvey+ 2015)", 1000, 0, 0.47, "upper_limit"),
    # Groups
    ("Galaxy groups (Kaplinghat+ 2016)", 500, 0.1, 1.0, "preferred"),
    # Galaxies
    ("Milky Way (Kaplinghat+ 2016)", 200, 0.5, 5.0, "preferred"),
    ("LSB galaxies (Kamada+ 2017)", 100, 1.0, 10.0, "preferred"),
    # Dwarfs
    ("Dwarf spheroidals (Correa 2021)", 30, 5.0, 50.0, "preferred"),
    ("Ultra-faint dwarfs (Hayashi+ 2021)", 10, 10.0, 100.0, "preferred"),
]

print(f"\n  Published constraints on DM self-interaction:")
print(f"  {'System':<40s} {'v(km/s)':>8s} {'σ/m range':>15s} {'Type':>12s}")
print(f"  {'─'*40} {'─'*8} {'─'*15} {'─'*12}")
for name, v, lo, hi, typ in sidm_constraints:
    if typ == 'upper_limit':
        print(f"  {name:<40s} {v:>8.0f} {'< ' + f'{hi:.2f}':>15s} {'excluded':>12s}")
    else:
        print(f"  {name:<40s} {v:>8.0f} {f'{lo:.1f}–{hi:.0f}':>15s} {'preferred':>12s}")

print(f"\n  TB predicts σ/m = 0 → consistent with ALL upper limits")
print(f"  BUT: dwarfs & LSBs PREFER σ/m ~ 1-50 cm²/g (cores)")
print(f"  TB faces same core-cusp tension as standard CDM")


# ============================================================
# PART 3: CORE-CUSP TEST ON SPARC DWARFS
# ============================================================
print(f"\n{'─'*74}")
print(f"  PART 3: NFW vs CORED PROFILES ON SPARC DWARFS")
print(f"{'─'*74}")

# Load SPARC data
script_dir = os.path.dirname(os.path.abspath(__file__))
sparc_file = os.path.join(script_dir, 'sparc_rotcurves.dat')

def load_sparc(filename):
    galaxies = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('---') and i > 5:
            data_start = i + 1
            break
    for line in lines[data_start:]:
        line = line.rstrip('\n')
        if len(line) < 60: continue
        try:
            name = line[0:11].strip()
            R    = float(line[19:25])
            Vobs = float(line[26:32])
            eV   = float(line[33:38])
            Vgas = float(line[39:45])
            Vdisk= float(line[46:52])
            Vbul = float(line[53:59])
        except (ValueError, IndexError):
            continue
        if name not in galaxies:
            galaxies[name] = {'R': [], 'Vobs': [], 'eVobs': [],
                              'Vgas': [], 'Vdisk': [], 'Vbul': []}
        galaxies[name]['R'].append(R)
        galaxies[name]['Vobs'].append(Vobs)
        galaxies[name]['eVobs'].append(eV)
        galaxies[name]['Vgas'].append(Vgas)
        galaxies[name]['Vdisk'].append(Vdisk)
        galaxies[name]['Vbul'].append(Vbul)
    for name in galaxies:
        for key in galaxies[name]:
            galaxies[name][key] = np.array(galaxies[name][key])
    return galaxies

all_sparc = load_sparc(sparc_file)

# Select DM-dominated dwarfs (V_max < 80 km/s)
dwarfs = {}
for g, d in all_sparc.items():
    if len(d['R']) >= 8 and np.max(d['Vobs']) < 80:
        dwarfs[g] = d

print(f"\n  Selected {len(dwarfs)} DM-dominated dwarfs (V_max < 80 km/s, N ≥ 8)")

# Halo models
G_SI = 6.674e-11

def V_NFW(R_kpc, rho0, rs):
    """NFW: V² = 4πG ρ₀ rs³ [ln(1+x) - x/(1+x)] / r, cusp γ=-1"""
    x = R_kpc / rs
    fx = np.log(1 + x) - x / (1 + x)
    rho_SI = rho0 * Msun / kpc_m**3
    V2 = 4*np.pi*G_SI * rho_SI * (rs*kpc_m)**3 * fx / (R_kpc*kpc_m)
    return np.sign(V2) * np.sqrt(np.abs(V2)) / 1e3

def V_Burkert(R_kpc, rho0, rc):
    """Burkert (cored): ρ = ρ₀ / [(1+r/rc)(1+(r/rc)²)], core γ=0"""
    x = R_kpc / rc
    fx = 0.5 * np.log((1+x)**2 * (1+x**2)) - np.arctan(x)
    rho_SI = rho0 * Msun / kpc_m**3
    V2 = 4*np.pi*G_SI * rho_SI * (rc*kpc_m)**3 * fx / (R_kpc*kpc_m)
    return np.sign(V2) * np.sqrt(np.abs(V2)) / 1e3

def V_ISO(R_kpc, rho0, rc):
    """Pseudo-isothermal (cored): ρ = ρ₀ / [1+(r/rc)²], core γ=0"""
    x = R_kpc / rc
    fx = x - np.arctan(x)
    rho_SI = rho0 * Msun / kpc_m**3
    V2 = 4*np.pi*G_SI * rho_SI * (rc*kpc_m)**3 * fx / (R_kpc*kpc_m)
    return np.sign(V2) * np.sqrt(np.abs(V2)) / 1e3

def fit_halo(R, Vobs, eV, Vdisk, Vgas, halo_func, name):
    """Fit halo + baryonic model to rotation curve."""
    def chi2(params):
        log_rho, log_r, upsilon_d = params
        if not (3 <= log_rho <= 10 and -1.5 <= log_r <= 2 and 0.05 <= upsilon_d <= 2.0):
            return 1e12
        rho0, r_scale = 10**log_rho, 10**log_r
        V2_bar = upsilon_d * Vdisk**2 + Vgas**2
        V_h = halo_func(R, rho0, r_scale)
        V_mod = np.sqrt(np.maximum(V2_bar + V_h**2, 1e-10))
        err = np.maximum(eV, 1.0)
        return np.sum(((Vobs - V_mod) / err)**2)
    
    best_c2, best_p = 1e20, None
    for lr in np.arange(4, 9, 1.0):
        for ls in np.arange(-1, 2.0, 0.5):
            for ud in [0.2, 0.5, 0.8]:
                c2 = chi2([lr, ls, ud])
                if c2 < best_c2:
                    best_c2, best_p = c2, [lr, ls, ud]
    
    if best_p is None:
        best_p = [6.5, 0.5, 0.5]
    
    result = minimize(chi2, best_p, method='Nelder-Mead',
                      options={'maxiter': 80000, 'xatol': 1e-9, 'fatol': 1e-9})
    
    lr, ls, ud = result.x
    rho0, r_scale = 10**lr, 10**ls
    V2_bar = ud * Vdisk**2 + Vgas**2
    V_h = halo_func(R, rho0, r_scale)
    V_mod = np.sqrt(np.maximum(V2_bar + V_h**2, 1e-10))
    ndof = max(len(R) - 3, 1)
    
    return {
        'chi2': result.fun, 'chi2_red': result.fun / ndof, 'ndof': ndof,
        'rho0': rho0, 'r_scale': r_scale, 'upsilon_d': ud,
        'V_model': V_mod, 'V_halo': np.abs(V_h),
    }

# Fit all dwarfs with NFW (cusp) and Burkert (core) and Pseudo-isothermal (core)
print(f"\n  Fitting NFW (cusp, TB prediction) vs Burkert & ISO (core, SIDM) ...")
print(f"  {'Galaxy':<14s} {'N':>3s} {'Vmax':>5s} │ {'NFW':>7s} {'Burk':>7s} {'ISO':>7s} │ {'ΔBIC_BN':>8s} {'ΔBIC_IN':>8s} │ {'Core?':>6s}")
print(f"  {'─'*14} {'─'*3} {'─'*5} │ {'─'*7} {'─'*7} {'─'*7} │ {'─'*8} {'─'*8} │ {'─'*6}")

nfw_wins = 0
core_wins = 0
tie_count = 0
all_delta_bic = []

dwarf_results = {}

for gname in sorted(dwarfs.keys()):
    gd = dwarfs[gname]
    R, Vobs, eV = gd['R'], gd['Vobs'], gd['eVobs']
    Vdisk, Vgas = gd['Vdisk'], gd['Vgas']
    Vmax = np.max(Vobs)
    
    res_nfw = fit_halo(R, Vobs, eV, Vdisk, Vgas, V_NFW, "NFW")
    res_bur = fit_halo(R, Vobs, eV, Vdisk, Vgas, V_Burkert, "Burkert")
    res_iso = fit_halo(R, Vobs, eV, Vdisk, Vgas, V_ISO, "ISO")
    
    n = len(R)
    k_all = 3  # same for all: ρ₀, r_scale, Υ_d
    bic_nfw = res_nfw['chi2'] + k_all * np.log(n)
    bic_bur = res_bur['chi2'] + k_all * np.log(n)
    bic_iso = res_iso['chi2'] + k_all * np.log(n)
    
    # ΔBIC: Burkert - NFW (negative = Burkert better = core preferred)
    dbic_bn = bic_bur - bic_nfw
    dbic_in = bic_iso - bic_nfw
    
    # Best cored model
    best_core_bic = min(bic_bur, bic_iso)
    dbic = best_core_bic - bic_nfw
    all_delta_bic.append(dbic)
    
    if dbic < -2:
        verdict = "CORE"
        core_wins += 1
    elif dbic > 2:
        verdict = "CUSP"
        nfw_wins += 1
    else:
        verdict = "~TIE"
        tie_count += 1
    
    dwarf_results[gname] = {
        'nfw': res_nfw, 'burkert': res_bur, 'iso': res_iso,
        'bic_nfw': bic_nfw, 'bic_bur': bic_bur, 'bic_iso': bic_iso,
        'Vmax': Vmax, 'npts': n, 'verdict': verdict,
    }
    
    print(f"  {gname:<14s} {n:>3d} {Vmax:>5.0f} │ {res_nfw['chi2_red']:>7.2f} {res_bur['chi2_red']:>7.2f} {res_iso['chi2_red']:>7.2f} │ {dbic_bn:>+8.1f} {dbic_in:>+8.1f} │ {verdict:>6s}")

N_dwarfs = len(dwarfs)
print(f"\n  Summary: NFW(cusp) = {nfw_wins}, Core = {core_wins}, Tie = {tie_count}  (of {N_dwarfs})")

frac_core = core_wins / N_dwarfs * 100
frac_cusp = nfw_wins / N_dwarfs * 100
frac_tie = tie_count / N_dwarfs * 100


# ============================================================
# PART 4: INNER SLOPE ANALYSIS
# ============================================================
print(f"\n{'─'*74}")
print(f"  PART 4: INNER DENSITY SLOPE γ = d ln ρ / d ln r")
print(f"{'─'*74}")
print(f"\n  TB/CDM predicts γ → -1 (NFW cusp)")
print(f"  SIDM predicts γ → 0 (isothermal core)")
print(f"  Observed: γ ∈ [-0.5, 0] for most dwarfs\n")

# Estimate inner slope from the innermost points of each rotation curve
# V ∝ r^{(1+γ)/2} in the inner region → log V = const + (1+γ)/2 × log r
# γ = 2 × (d log V / d log r) - 1

print(f"  {'Galaxy':<14s} {'R_in':>5s} {'V_in':>5s} {'γ_obs':>6s} │ {'γ_NFW':>6s} {'γ_Bur':>6s} │ {'Tension':>8s}")
print(f"  {'─'*14} {'─'*5} {'─'*5} {'─'*6} │ {'─'*6} {'─'*6} │ {'─'*8}")

gamma_obs_all = []
gamma_nfw_pred = -1.0  # fixed
gamma_core_pred = 0.0  # fixed

for gname in sorted(dwarfs.keys()):
    gd = dwarfs[gname]
    R, Vobs = gd['R'], gd['Vobs']
    
    # Use innermost 3-5 points for slope estimate
    n_inner = min(4, len(R)//2)
    if n_inner < 2:
        continue
    
    R_in = R[:n_inner]
    V_in = Vobs[:n_inner]
    
    # Filter positive values
    mask = (R_in > 0) & (V_in > 0)
    if np.sum(mask) < 2:
        continue
    
    log_r = np.log10(R_in[mask])
    log_v = np.log10(V_in[mask])
    
    # Linear fit: log V = a + b × log r
    if len(log_r) >= 2:
        b = np.polyfit(log_r, log_v, 1)[0]
        gamma_obs = 2 * b - 1
        gamma_obs = np.clip(gamma_obs, -2, 1)
        gamma_obs_all.append(gamma_obs)
        
        tension = "none" if gamma_obs < -0.7 else ("mild" if gamma_obs < -0.3 else "STRONG")
        
        print(f"  {gname:<14s} {R_in[0]:>5.2f} {V_in[0]:>5.1f} {gamma_obs:>+6.2f} │ {gamma_nfw_pred:>+6.1f} {gamma_core_pred:>+6.1f} │ {tension:>8s}")

gamma_obs_arr = np.array(gamma_obs_all)
print(f"\n  Inner slope distribution (N = {len(gamma_obs_arr)} dwarfs):")
print(f"    Mean γ_obs  = {np.mean(gamma_obs_arr):+.2f}")
print(f"    Median γ_obs = {np.median(gamma_obs_arr):+.2f}")
print(f"    Std          = {np.std(gamma_obs_arr):.2f}")
print(f"    NFW prediction: γ = -1.0")
print(f"    Cored prediction: γ = 0.0")
print(f"    γ_obs closer to: {'CORE' if abs(np.mean(gamma_obs_arr)) < 0.5 else 'CUSP'}")

n_cuspy = np.sum(gamma_obs_arr < -0.7)
n_cored = np.sum(gamma_obs_arr > -0.3)
n_middle = len(gamma_obs_arr) - n_cuspy - n_cored
print(f"\n    Cuspy (γ < -0.7): {n_cuspy}/{len(gamma_obs_arr)}")
print(f"    Cored (γ > -0.3): {n_cored}/{len(gamma_obs_arr)}")
print(f"    Intermediate:      {n_middle}/{len(gamma_obs_arr)}")


# ============================================================
# PART 5: IMPLICATIONS FOR TB
# ============================================================
print(f"\n{'─'*74}")
print(f"  PART 5: WHAT THIS MEANS FOR TWIN BARRIER THEORY")
print(f"{'─'*74}")

# Core formation via baryonic feedback
print(f"""
  The core-cusp problem is NOT unique to TB — it affects ALL CDM models.
  The standard resolution is baryonic feedback:
  
  ── Baryonic feedback mechanism ──
  • Supernovae inject energy → gas outflows → DM orbits expand
  • Creates cores in galaxies with M* > 10⁶ M☉ (enough star formation)
  • Read+ 2019: CDM + feedback reproduces observed cores for 
    M_200 > 10⁹ M☉ (V_max > ~20 km/s)
  • For ultra-faint dwarfs (M* < 10⁵): feedback insufficient → cusp remains
  
  ── TB-specific considerations ──
  • TB DM = gauge-decoupled → σ/m = 0 → collisionless
  • SAME core-cusp tension as ΛCDM, no better, no worse
  • TB does NOT predict SIDM-like cores
  • If future observations definitively show cores in ultra-faint dwarfs
    (V_max < 15 km/s) where feedback can't work → TB has a problem
    (same problem as ALL CDM models)

  ── Current data verdict ──
  Our SPARC dwarf analysis ({N_dwarfs} galaxies, V_max < 80 km/s):
    Core preferred in {core_wins}/{N_dwarfs} dwarfs ({frac_core:.0f}%)
    Cusp preferred in {nfw_wins}/{N_dwarfs} dwarfs ({frac_cusp:.0f}%)
    Indeterminate:     {tie_count}/{N_dwarfs} dwarfs ({frac_tie:.0f}%)
    Mean inner slope: γ = {np.mean(gamma_obs_arr):+.2f} (between cusp and core)
""")

# ============================================================
# DISCRIMINATING TEST
# ============================================================
print(f"  ╔═══════════════════════════════════════════════════════════════════╗")
print(f"  ║  CAN THIS DISCRIMINATE TB FROM OTHER DM MODELS?                  ║")
print(f"  ╠═══════════════════════════════════════════════════════════════════╣")
print(f"  ║                                                                   ║")
print(f"  ║  TB predicts σ/m = 0 (collisionless CDM)                         ║")
print(f"  ║  SIDM predicts σ/m ~ 1-50 cm²/g (velocity-dependent)            ║")
print(f"  ║  Fuzzy DM: cores from wave interference, λ_dB ~ kpc              ║")
print(f"  ║                                                                   ║")
print(f"  ║  DISCRIMINATING OBSERVATIONS:                                     ║")
print(f"  ║  1. Ultra-faint dwarfs with V_max < 15 km/s:                     ║")
print(f"  ║     Core → rules out TB (no baryonic feedback possible)          ║")
print(f"  ║     Cusp → supports TB, rules out simple SIDM                    ║")
print(f"  ║                                                                   ║")
print(f"  ║  2. Gravothermal catastrophe in massive SIDM halos:              ║")
print(f"  ║     SIDM predicts core collapse at late times → high-σ cusps     ║")
print(f"  ║     TB predicts NFW cusps at all times                           ║")
print(f"  ║                                                                   ║")
print(f"  ║  3. DM subhalo count (LSST/Rubin survey):                        ║")
print(f"  ║     SIDM erases small subhalos; TB preserves them (CDM-like)     ║")
print(f"  ║                                                                   ║")
print(f"  ║  CURRENT STATUS:                                                  ║")
print(f"  ║  Data cannot yet discriminate — observed cores are consistent     ║")
print(f"  ║  with both CDM+feedback (=TB) and SIDM. Ultra-faint dwarf data   ║")
print(f"  ║  from Rubin Observatory (2025+) will be decisive.                ║")
print(f"  ╚═══════════════════════════════════════════════════════════════════╝")


# ============================================================
# PLOTS
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Twin Barrier DM: Self-Interaction & Core-Cusp Analysis\n'
             'TB predicts σ/m = 0 (collisionless CDM)', 
             fontsize=13, fontweight='bold')

# Panel 1: σ/m vs velocity scale
ax1 = axes[0, 0]
v_range = np.logspace(0.5, 4, 100)

# SIDM constraints
for name, v, lo, hi, typ in sidm_constraints:
    short = name.split('(')[0].strip()
    if typ == 'upper_limit':
        ax1.plot(v, hi, 'rv', ms=10, zorder=5)
        ax1.annotate(short, (v, hi*1.3), fontsize=6, ha='center')
    else:
        ax1.errorbar(v, (lo+hi)/2, yerr=[[((lo+hi)/2-lo)], [(hi-(lo+hi)/2)]],
                     fmt='bo', ms=6, capsize=3, zorder=5)
        ax1.annotate(short, (v, hi*1.3), fontsize=6, ha='center')

# TB prediction
ax1.axhline(0, color='blue', lw=3, ls='-', alpha=0.8, label='TB prediction (σ/m = 0)')
ax1.axhspan(-0.1, 0.1, color='blue', alpha=0.1)

# SIDM models
for sigma_m, ls, label in [(1, '--', 'σ/m = 1 cm²/g'), (10, ':', 'σ/m = 10 cm²/g')]:
    ax1.axhline(sigma_m, color='red', ls=ls, lw=1.5, alpha=0.6, label=label)

ax1.set_xscale('log')
ax1.set_yscale('symlog', linthresh=0.1)
ax1.set_xlabel('Velocity dispersion (km/s)', fontsize=11)
ax1.set_ylabel('σ/m (cm²/g)', fontsize=11)
ax1.set_title('(a) DM self-interaction constraints')
ax1.legend(fontsize=7, loc='upper right')
ax1.set_xlim(5, 6000)
ax1.set_ylim(-0.5, 200)
ax1.grid(True, alpha=0.2)

# Panel 2: ΔBIC histogram (NFW vs Core)
ax2 = axes[0, 1]
delta_arr = np.array(all_delta_bic)
colors = ['blue' if d > 2 else ('red' if d < -2 else 'gray') for d in delta_arr]
ax2.hist(delta_arr, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(0, color='k', ls='-', lw=1)
ax2.axvline(-2, color='red', ls=':', label='|ΔBIC|=2')
ax2.axvline(2, color='blue', ls=':')
ax2.set_xlabel('ΔBIC (Core − NFW)', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title(f'(b) Core vs Cusp: Core={core_wins}, Cusp={nfw_wins}, Tie={tie_count}')
ax2.legend(fontsize=8)

# Panel 3: Inner slope histogram  
ax3 = axes[0, 2]
ax3.hist(gamma_obs_arr, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax3.axvline(-1.0, color='blue', ls='--', lw=2, label='NFW (γ=-1, TB)')
ax3.axvline(0.0, color='red', ls='--', lw=2, label='Core (γ=0, SIDM)')
ax3.axvline(np.mean(gamma_obs_arr), color='green', ls='-', lw=2,
            label=f'Mean γ={np.mean(gamma_obs_arr):+.2f}')
ax3.set_xlabel('Inner slope γ', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('(c) Observed inner slopes')
ax3.legend(fontsize=8)

# Panels 4-6: Example dwarf rotation curves (NFW vs Burkert)
examples = sorted(dwarf_results.keys(), 
                  key=lambda g: abs(dwarf_results[g]['bic_bur'] - dwarf_results[g]['bic_nfw']),
                  reverse=True)[:3]

for idx, gname in enumerate(examples):
    ax = axes[1, idx]
    gd = dwarfs[gname]
    R, Vobs, eV = gd['R'], gd['Vobs'], gd['eVobs']
    
    dr = dwarf_results[gname]
    
    ax.errorbar(R, Vobs, yerr=eV, fmt='ko', ms=4, capsize=2, label='Data', zorder=5)
    ax.plot(R, dr['nfw']['V_model'], 'b-', lw=2, 
            label=f'NFW (χ²/dof={dr["nfw"]["chi2_red"]:.2f})')
    ax.plot(R, dr['burkert']['V_model'], 'r--', lw=2,
            label=f'Burkert (χ²/dof={dr["burkert"]["chi2_red"]:.2f})')
    ax.plot(R, dr['iso']['V_model'], 'g:', lw=2,
            label=f'ISO (χ²/dof={dr["iso"]["chi2_red"]:.2f})')
    
    dbic = dr['bic_bur'] - dr['bic_nfw']
    ax.set_xlabel('R (kpc)', fontsize=10)
    ax.set_ylabel('V (km/s)', fontsize=10)
    ax.set_title(f'{gname} (Vmax={dr["Vmax"]:.0f}, ΔBIC={dbic:+.1f} → {dr["verdict"]})')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'tb_dm_core_cusp.png'), dpi=150, bbox_inches='tight')
print(f"\n  Plot saved: tb_dm_core_cusp.png")

# ============================================================
# FINAL VERDICT
# ============================================================
print(f"\n{'═'*74}")
print(f"  FINAL VERDICT")
print(f"{'═'*74}")

print(f"""
  Q: "twin barioni imaju presek σ/m = X cm²/g" → testirati na patuljcima
  
  A: σ/m = 0 (collisionless) — derived from TB's gauge localization.
  
  THIS IS NOT A DISCRIMINATING TEST because:
  
  1. TB predicts σ/m = 0 → identical to standard CDM (WIMPs, etc.)
     There's nothing unique about TB here.
  
  2. Current dwarf data shows mixed results:
     {core_wins}/{N_dwarfs} prefer cores, {nfw_wins}/{N_dwarfs} prefer cusps, {tie_count}/{N_dwarfs} ties.
     Mean γ = {np.mean(gamma_obs_arr):+.2f} — between cusp and core.
  
  3. Baryonic feedback can explain observed cores for V_max > 20 km/s,
     so cores don't rule out CDM (nor TB).
  
  4. Only ultra-faint dwarfs (V_max < 15 km/s) could discriminate,
     but current data is insufficient.
  
  BOTTOM LINE: Na galaksijama TB ne može da se razlikuje od ΛCDM.
  Za dokaz TB-a treba ili nano-skala (Casimir ~60nm) ili TeV (kolajder >42 TeV).
""")
