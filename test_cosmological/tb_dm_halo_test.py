#!/usr/bin/env python3
"""
Twin Barrier Dark-Matter Halo Profiles vs Standard Profiles
============================================================
Comprehensive, skeptical test of TB dark-matter halo profiles against
NFW, Burkert, and pseudo-isothermal on SPARC dwarf/LSB rotation curves.

Tests performed:
  1. Individual galaxy fits (chi2, reduced chi2, AIC, BIC)
  2. Global parameter distributions & scaling relations
  3. Core-cusp discrimination (inner log-slope)
  4. Rotation-curve diversity for matched Vmax
  5. Leave-one-out robustness

Author: Mateja Radojičić / Twin Barrier Theory validation suite
Date:   April 2026
"""

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings, os, sys, json, time
from collections import defaultdict

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(SCRIPT_DIR, 'sparc_rotcurves.dat')
PROP_FILE  = os.path.join(SCRIPT_DIR, 'sparc_table2.dat')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'tb_dm_results')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# CONSTANTS
# ============================================================
G_SI   = 6.67430e-11   # m^3 kg^-1 s^-2
kpc_m  = 3.0857e19     # m per kpc
Msun   = 1.989e30      # kg
MIN_PTS = 5

# ============================================================
# DATA LOADING
# ============================================================

def load_sparc_rotcurves(filename):
    """Parse SPARC MassModels -> dict of galaxy data."""
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
        if len(line) < 60:
            continue
        try:
            name = line[0:11].strip()
            D    = float(line[12:18])
            R    = float(line[19:25])
            Vobs = float(line[26:32])
            eV   = float(line[33:38])
            Vgas = float(line[39:45])
            Vdisk= float(line[46:52])
            Vbul = float(line[53:59])
            SBdisk = float(line[60:67]) if len(line) >= 67 else 0.0
        except (ValueError, IndexError):
            continue
        if eV <= 0:
            eV = max(abs(Vobs) * 0.1, 1.0)
        if name not in galaxies:
            galaxies[name] = {'D': D, 'R': [], 'Vobs': [], 'eVobs': [],
                              'Vgas': [], 'Vdisk': [], 'Vbul': [], 'SBdisk': []}
        galaxies[name]['R'].append(R)
        galaxies[name]['Vobs'].append(Vobs)
        galaxies[name]['eVobs'].append(eV)
        galaxies[name]['Vgas'].append(Vgas)
        galaxies[name]['Vdisk'].append(Vdisk)
        galaxies[name]['Vbul'].append(Vbul)
        galaxies[name]['SBdisk'].append(SBdisk)
    for name in galaxies:
        for key in ['R', 'Vobs', 'eVobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk']:
            galaxies[name][key] = np.array(galaxies[name][key])
    return galaxies

def load_sparc_props(filename):
    """Parse SPARC Table1 -> dict of galaxy properties."""
    props = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('---') and i > 10:
            data_start = i + 1
            break
    for line in lines[data_start:]:
        if len(line.strip()) < 20:
            continue
        try:
            name = line[0:11].strip()
            T    = int(line[11:13])
            D    = float(line[13:19])
            Inc  = float(line[26:30])
            L36  = float(line[34:41])
            Reff = float(line[48:53])
            SBeff= float(line[53:61])
            MHI  = float(line[74:81])
            Vflat= float(line[86:91])
            Q    = int(line[96:99])
        except (ValueError, IndexError):
            continue
        props[name] = {
            'T': T, 'D': D, 'Inc': Inc, 'L36': L36,
            'Reff': Reff, 'SBeff': SBeff, 'MHI': MHI,
            'Vflat': Vflat, 'Q': Q
        }
    return props

def select_dwarf_lsb(galaxies, props, max_vflat=80.0, min_pts=MIN_PTS, max_quality=2):
    """Select dwarf/LSB subset."""
    selected = []
    for name, gd in galaxies.items():
        if len(gd['R']) < min_pts:
            continue
        if name in props:
            p = props[name]
            if p['Q'] > max_quality:
                continue
            if p['Vflat'] > max_vflat:
                continue
            selected.append(name)
        else:
            vmax = np.max(np.abs(gd['Vobs']))
            if vmax > max_vflat:
                continue
            selected.append(name)
    return sorted(selected)

# ============================================================
# HALO PROFILE MODELS
# ============================================================

def _mass_enclosed_numerical(rho_func, R_kpc, params, n_grid=300):
    """Compute V_DM(R) in km/s via cumulative integration on a fine grid.
    Much faster than per-radius integration — O(n_grid) instead of O(n_grid * n_data)."""
    R_max = np.max(R_kpc) * 1.01
    r_grid = np.linspace(0, R_max, n_grid)  # kpc
    r_grid_m = r_grid * kpc_m
    rho_grid = rho_func(r_grid, params)  # Msun/kpc^3
    rho_SI = rho_grid * Msun / kpc_m**3
    integrand = 4.0 * np.pi * r_grid_m**2 * rho_SI
    M_cum = np.cumsum(integrand) * (r_grid_m[1] - r_grid_m[0]) if n_grid > 1 else np.zeros(n_grid)
    # V = sqrt(G*M/r)
    with np.errstate(divide='ignore', invalid='ignore'):
        V_grid = np.where(r_grid_m > 0, np.sqrt(np.maximum(G_SI * M_cum / r_grid_m, 0)) * 1e-3, 0.0)
    # Interpolate to data radii
    return np.interp(R_kpc, r_grid, V_grid)

# --- NFW ---
def rho_NFW(r_kpc, params):
    rho0, rs = params
    x = np.maximum(r_kpc / rs, 1e-10)
    return rho0 / (x * (1 + x)**2)

def V_NFW(R_kpc, rho0, rs):
    R_m = R_kpc * kpc_m
    x = np.maximum(R_kpc / rs, 1e-10)
    fx = np.log(1 + x) - x / (1 + x)
    rho_SI = rho0 * Msun / kpc_m**3
    rs_m = rs * kpc_m
    M = 4.0 * np.pi * rho_SI * rs_m**3 * fx
    V2 = np.maximum(G_SI * M / R_m, 0)
    return np.sqrt(V2) * 1e-3

# --- Burkert ---
def rho_Burkert(r_kpc, params):
    rho0, r0 = params
    x = np.maximum(r_kpc / r0, 1e-10)
    return rho0 / ((1 + x) * (1 + x**2))

def V_Burkert(R_kpc, rho0, r0):
    """Burkert circular velocity via analytic enclosed mass."""
    x = R_kpc / r0
    r0_m = r0 * kpc_m
    rho_SI = rho0 * Msun / kpc_m**3
    # Analytic M(<r) for Burkert: M = pi*rho0*r0^3 * [ln(1+x^2) + 2*ln(1+x) - 2*arctan(x)]
    M = np.pi * rho_SI * r0_m**3 * (np.log(1 + x**2) + 2*np.log(1 + x) - 2*np.arctan(x))
    R_m = R_kpc * kpc_m
    V2 = np.maximum(G_SI * M / R_m, 0)
    return np.sqrt(V2) * 1e-3

# --- Pseudo-isothermal ---
def rho_ISO(r_kpc, params):
    rho0, rc = params
    return rho0 / (1 + (r_kpc / rc)**2)

def V_ISO(R_kpc, rho0, rc):
    R_m = R_kpc * kpc_m
    rc_m = rc * kpc_m
    rho_SI = rho0 * Msun / kpc_m**3
    x = R_m / rc_m
    M = 4 * np.pi * rho_SI * rc_m**3 * (x - np.arctan(x))
    V2 = np.maximum(G_SI * M / R_m, 0)
    return np.sqrt(V2) * 1e-3

# --- TB-1: rho = rho0 / (1 + r/rc)^beta ---
def rho_TB1(r_kpc, params):
    rho0, rc, beta = params
    x = np.maximum(r_kpc / rc, 1e-10)
    return rho0 / (1 + x)**beta

def V_TB1(R_kpc, rho0, rc, beta):
    return _mass_enclosed_numerical(rho_TB1, R_kpc, [rho0, rc, beta])

# --- TB-2: rho = rho0 / [1 + (r/rc)^2]^(beta/2) ---
def rho_TB2(r_kpc, params):
    rho0, rc, beta = params
    x = r_kpc / rc
    return rho0 / (1 + x**2)**(beta / 2)

def V_TB2(R_kpc, rho0, rc, beta):
    return _mass_enclosed_numerical(rho_TB2, R_kpc, [rho0, rc, beta])

# --- TB-3: rho = rho0 * exp[-(r/rc)^nu] ---
def rho_TB3(r_kpc, params):
    rho0, rc, nu = params
    x = r_kpc / rc
    return rho0 * np.exp(-x**nu)

def V_TB3(R_kpc, rho0, rc, nu):
    return _mass_enclosed_numerical(rho_TB3, R_kpc, [rho0, rc, nu])

# ============================================================
# MODEL DEFINITIONS
# ============================================================

MODELS = {
    'NFW': {
        'npar_halo': 2,
        'labels': ['log10_rho0', 'rs'],
        'bounds': [(4, 10), (0.1, 50)],
        'Vfunc': lambda R, p: V_NFW(R, 10**p[0], p[1]),
        'rho_func': rho_NFW,
        'rho_unpack': lambda p: [10**p[0], p[1]],
    },
    'Burkert': {
        'npar_halo': 2,
        'labels': ['log10_rho0', 'r0'],
        'bounds': [(4, 10), (0.1, 50)],
        'Vfunc': lambda R, p: V_Burkert(R, 10**p[0], p[1]),
        'rho_func': rho_Burkert,
        'rho_unpack': lambda p: [10**p[0], p[1]],
    },
    'ISO': {
        'npar_halo': 2,
        'labels': ['log10_rho0', 'rc'],
        'bounds': [(4, 10), (0.1, 50)],
        'Vfunc': lambda R, p: V_ISO(R, 10**p[0], p[1]),
        'rho_func': rho_ISO,
        'rho_unpack': lambda p: [10**p[0], p[1]],
    },
    'TB1': {
        'npar_halo': 3,
        'labels': ['log10_rho0', 'rc', 'beta'],
        'bounds': [(4, 10), (0.1, 50), (0.5, 6.0)],
        'Vfunc': lambda R, p: V_TB1(R, 10**p[0], p[1], p[2]),
        'rho_func': rho_TB1,
        'rho_unpack': lambda p: [10**p[0], p[1], p[2]],
    },
    'TB2': {
        'npar_halo': 3,
        'labels': ['log10_rho0', 'rc', 'beta'],
        'bounds': [(4, 10), (0.1, 50), (0.5, 6.0)],
        'Vfunc': lambda R, p: V_TB2(R, 10**p[0], p[1], p[2]),
        'rho_func': rho_TB2,
        'rho_unpack': lambda p: [10**p[0], p[1], p[2]],
    },
    'TB3': {
        'npar_halo': 3,
        'labels': ['log10_rho0', 'rc', 'nu'],
        'bounds': [(4, 10), (0.1, 50), (0.3, 4.0)],
        'Vfunc': lambda R, p: V_TB3(R, 10**p[0], p[1], p[2]),
        'rho_func': rho_TB3,
        'rho_unpack': lambda p: [10**p[0], p[1], p[2]],
    },
}

# ============================================================
# FITTING INFRASTRUCTURE
# ============================================================

def V_total(R, Vgas, Vdisk, Vbul, V_DM, ML_disk):
    """V^2 = Vgas^2 + ML*Vdisk^2 + 0.7*Vbul^2 + V_DM^2"""
    V2 = (Vgas**2
          + ML_disk * np.sign(Vdisk) * Vdisk**2
          + 0.7 * np.sign(Vbul) * Vbul**2
          + np.sign(V_DM) * V_DM**2)
    return np.sqrt(np.maximum(V2, 0))

def chi2_func(params, gal_data, model_info):
    ML_disk = params[0]
    halo_params = params[1:]
    R = gal_data['R']
    try:
        V_DM = model_info['Vfunc'](R, halo_params)
    except Exception:
        return 1e30
    if np.any(np.isnan(V_DM)) or np.any(np.isinf(V_DM)):
        return 1e30
    Vtot = V_total(R, gal_data['Vgas'], gal_data['Vdisk'], gal_data['Vbul'], V_DM, ML_disk)
    residuals = (gal_data['Vobs'] - Vtot) / gal_data['eVobs']
    return np.sum(residuals**2)

def fit_galaxy(gal_data, model_name, model_info, n_restarts=2):
    """Fit a single galaxy with differential evolution + polish."""
    ml_bounds = [(0.1, 2.0)]
    halo_bounds = model_info['bounds']
    all_bounds = ml_bounds + list(halo_bounds)
    n_data = len(gal_data['R'])
    n_params = 1 + model_info['npar_halo']
    best_result = None
    best_chi2 = 1e30
    for trial in range(n_restarts):
        try:
            result = differential_evolution(
                chi2_func, all_bounds,
                args=(gal_data, model_info),
                seed=42 + trial,
                maxiter=200,
                tol=1e-6,
                polish=True,
                popsize=10,
                init='sobol' if trial == 0 else 'latinhypercube'
            )
            if result.fun < best_chi2:
                best_chi2 = result.fun
                best_result = result
        except Exception:
            continue
    if best_result is None:
        return None
    chi2_val = best_result.fun
    dof = max(n_data - n_params, 1)
    red_chi2 = chi2_val / dof
    aic = chi2_val + 2 * n_params
    bic = chi2_val + n_params * np.log(n_data)
    return {
        'model': model_name,
        'params': best_result.x.tolist(),
        'ML_disk': best_result.x[0],
        'halo_params': best_result.x[1:].tolist(),
        'chi2': chi2_val,
        'dof': dof,
        'red_chi2': red_chi2,
        'aic': aic,
        'bic': bic,
        'n_data': n_data,
        'n_params': n_params,
    }

# ============================================================
# INNER SLOPE COMPUTATION (Test 3)
# ============================================================

def inner_log_slope(rho_func, params, r_eval_kpc=0.3, dr_frac=0.1):
    r1 = r_eval_kpc * (1 - dr_frac)
    r2 = r_eval_kpc * (1 + dr_frac)
    rho1 = rho_func(np.array([r1]), params)[0]
    rho2 = rho_func(np.array([r2]), params)[0]
    if rho1 <= 0 or rho2 <= 0:
        return np.nan
    return (np.log(rho2) - np.log(rho1)) / (np.log(r2) - np.log(r1))

# ============================================================
# BARYONIC MASS ESTIMATION
# ============================================================

def estimate_baryonic_mass(gal_data, ML_disk=0.5):
    R_max_m = gal_data['R'][-1] * kpc_m
    Vgas_max = gal_data['Vgas'][-1] * 1e3
    Vdisk_max = gal_data['Vdisk'][-1] * 1e3
    M_gas = Vgas_max**2 * R_max_m / G_SI
    M_disk = ML_disk * Vdisk_max**2 * R_max_m / G_SI
    return (M_gas + M_disk) / Msun

# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_analysis():
    print("=" * 78)
    print("  TWIN BARRIER DM HALO PROFILES — COMPREHENSIVE SKEPTICAL TEST")
    print("=" * 78)
    t0 = time.time()

    # Load data
    print("\n[1] Loading SPARC data...")
    galaxies = load_sparc_rotcurves(DATA_FILE)
    props = load_sparc_props(PROP_FILE)
    print(f"    Loaded {len(galaxies)} galaxies, {len(props)} with properties")

    # Select dwarf/LSB subset — try strict first, then relax
    dwarf_names = select_dwarf_lsb(galaxies, props, max_vflat=80, min_pts=MIN_PTS, max_quality=2)
    print(f"    Dwarf/LSB subset (Vflat<80, Q<=2): {len(dwarf_names)} galaxies")
    if len(dwarf_names) < 8:
        dwarf_names = select_dwarf_lsb(galaxies, props, max_vflat=120, min_pts=MIN_PTS, max_quality=3)
        print(f"    Relaxed (Vflat<120, Q<=3): {len(dwarf_names)} galaxies")

    for n in dwarf_names[:5]:
        nd = len(galaxies[n]['R'])
        vmax = np.max(np.abs(galaxies[n]['Vobs']))
        print(f"      {n:12s}  N={nd:3d}  Vmax={vmax:.1f} km/s")
    if len(dwarf_names) > 5:
        print(f"      ... and {len(dwarf_names)-5} more")

    # ============================================================
    # TEST 1: Individual galaxy fits
    # ============================================================
    print(f"\n{'='*78}")
    print("  TEST 1: INDIVIDUAL GALAXY FITS")
    print(f"{'='*78}")

    all_results = {}
    model_names = ['NFW', 'Burkert', 'ISO', 'TB1', 'TB2', 'TB3']

    for ig, gname in enumerate(dwarf_names):
        gd = galaxies[gname]
        print(f"\n  [{ig+1}/{len(dwarf_names)}] {gname} (N={len(gd['R'])})", end='', flush=True)
        all_results[gname] = {}
        for mname in model_names:
            minfo = MODELS[mname]
            result = fit_galaxy(gd, mname, minfo, n_restarts=2)
            if result:
                all_results[gname][mname] = result
                print(f"  {mname}:{result['red_chi2']:.2f}", end='', flush=True)
            else:
                print(f"  {mname}:FAIL", end='', flush=True)
        print()

    # ============================================================
    # RESULTS TABLE
    # ============================================================
    print(f"\n{'='*78}")
    print("  RESULTS TABLE — PER GALAXY")
    print(f"{'='*78}")
    header = f"{'Galaxy':12s} {'Model':8s} {'ML_d':5s} {'chi2':8s} {'rchi2':8s} {'AIC':8s} {'BIC':8s} {'Halo params'}"
    print(header)
    print("-" * 90)

    for gname in dwarf_names:
        if gname not in all_results:
            continue
        for mname in model_names:
            if mname not in all_results[gname]:
                continue
            r = all_results[gname][mname]
            hp_str = ', '.join(f'{p:.3g}' for p in r['halo_params'])
            print(f"{gname:12s} {mname:8s} {r['ML_disk']:5.2f} {r['chi2']:8.2f} {r['red_chi2']:8.3f} {r['aic']:8.2f} {r['bic']:8.2f}  [{hp_str}]")
        print()

    # ============================================================
    # GLOBAL SUMMARY — TEST 2
    # ============================================================
    print(f"\n{'='*78}")
    print("  TEST 2: GLOBAL SUMMARY BY MODEL")
    print(f"{'='*78}")

    model_stats = {m: {'red_chi2': [], 'aic': [], 'bic': [], 'daic_nfw': [], 'daic_burkert': [],
                       'dbic_nfw': [], 'dbic_burkert': [],
                       'halo_params': [], 'ML_disk': [], 'Mbar': []} for m in model_names}
    best_count = {m: 0 for m in model_names}
    indist_count = {m: 0 for m in model_names}
    loses_count = {m: 0 for m in model_names}

    for gname in dwarf_names:
        if gname not in all_results:
            continue
        gr = all_results[gname]
        aics = {m: gr[m]['aic'] for m in model_names if m in gr}
        bics = {m: gr[m]['bic'] for m in model_names if m in gr}
        if not aics:
            continue
        best_aic = min(aics.values())
        best_model = min(aics, key=aics.get)
        nfw_aic = aics.get('NFW', np.nan)
        bur_aic = aics.get('Burkert', np.nan)
        nfw_bic = bics.get('NFW', np.nan)
        bur_bic = bics.get('Burkert', np.nan)

        for m in model_names:
            if m not in gr:
                continue
            r = gr[m]
            model_stats[m]['red_chi2'].append(r['red_chi2'])
            model_stats[m]['aic'].append(r['aic'])
            model_stats[m]['bic'].append(r['bic'])
            model_stats[m]['daic_nfw'].append(r['aic'] - nfw_aic if not np.isnan(nfw_aic) else np.nan)
            model_stats[m]['daic_burkert'].append(r['aic'] - bur_aic if not np.isnan(bur_aic) else np.nan)
            model_stats[m]['dbic_nfw'].append(r['bic'] - nfw_bic if not np.isnan(nfw_bic) else np.nan)
            model_stats[m]['dbic_burkert'].append(r['bic'] - bur_bic if not np.isnan(bur_bic) else np.nan)
            model_stats[m]['halo_params'].append(r['halo_params'])
            model_stats[m]['ML_disk'].append(r['ML_disk'])
            Mbar = estimate_baryonic_mass(galaxies[gname], ML_disk=r['ML_disk'])
            model_stats[m]['Mbar'].append(Mbar)

            delta_aic = r['aic'] - best_aic
            if m == best_model:
                best_count[m] += 1
            elif delta_aic < 2:
                indist_count[m] += 1
            else:
                loses_count[m] += 1

    print(f"\n{'Model':8s} {'MedRchi2':9s} {'MeanDAIC':9s} {'MeanDAIC':9s} {'MeanDBIC':9s} {'MeanDBIC':9s} {'Best':5s} {'~Eq':4s} {'Lose':4s}")
    print(f"{'':8s} {'':9s} {'vsNFW':9s} {'vsBurk':9s} {'vsNFW':9s} {'vsBurk':9s}")
    print("-" * 80)
    for m in model_names:
        if not model_stats[m]['red_chi2']:
            continue
        med_rchi2 = np.nanmedian(model_stats[m]['red_chi2'])
        mean_daic_nfw = np.nanmean(model_stats[m]['daic_nfw'])
        mean_daic_bur = np.nanmean(model_stats[m]['daic_burkert'])
        mean_dbic_nfw = np.nanmean(model_stats[m]['dbic_nfw'])
        mean_dbic_bur = np.nanmean(model_stats[m]['dbic_burkert'])
        print(f"{m:8s} {med_rchi2:9.3f} {mean_daic_nfw:+9.2f} {mean_daic_bur:+9.2f} {mean_dbic_nfw:+9.2f} {mean_dbic_bur:+9.2f} {best_count[m]:5d} {indist_count[m]:4d} {loses_count[m]:4d}")

    # Parameter distributions
    print(f"\n  Parameter distributions for TB models:")
    for m in ['TB1', 'TB2', 'TB3']:
        hp = np.array(model_stats[m]['halo_params'])
        if len(hp) == 0:
            continue
        print(f"\n  {m}:")
        labels = MODELS[m]['labels']
        for j, lab in enumerate(labels):
            vals = hp[:, j]
            print(f"    {lab:12s}: median={np.median(vals):.3f}, std={np.std(vals):.3f}, "
                  f"range=[{np.min(vals):.3f}, {np.max(vals):.3f}]")
        if m in ['TB1', 'TB2']:
            betas = hp[:, 2]
            cv = np.std(betas) / max(np.mean(betas), 1e-10)
            print(f"    beta CV: {cv:.3f} {'(universal <0.3)' if cv < 0.3 else '(NOT universal)'}")
        elif m == 'TB3':
            nus = hp[:, 2]
            cv = np.std(nus) / max(np.mean(nus), 1e-10)
            print(f"    nu CV: {cv:.3f} {'(universal <0.3)' if cv < 0.3 else '(NOT universal)'}")

    # Scaling: rc vs Mbar
    print(f"\n  Scaling relations (rc vs Mbar):")
    for m in ['TB1', 'TB2', 'TB3']:
        hp = np.array(model_stats[m]['halo_params'])
        Mbar = np.array(model_stats[m]['Mbar'])
        if len(hp) < 3:
            continue
        rc_vals = hp[:, 1]
        mask = (rc_vals > 0) & (Mbar > 0)
        if np.sum(mask) < 3:
            continue
        log_rc = np.log10(rc_vals[mask])
        log_Mbar = np.log10(Mbar[mask])
        if np.std(log_Mbar) > 0.01:
            slope, intercept = np.polyfit(log_Mbar, log_rc, 1)
            corr = np.corrcoef(log_Mbar, log_rc)[0, 1]
            print(f"    {m}: rc ~ Mbar^{slope:.2f}  (r={corr:.3f})")

    # ============================================================
    # TEST 3: CORE-CUSP DISCRIMINATION
    # ============================================================
    print(f"\n{'='*78}")
    print("  TEST 3: CORE-CUSP DISCRIMINATION (INNER LOG-SLOPE)")
    print(f"{'='*78}")

    slopes_by_model = {m: [] for m in model_names}
    for gname in dwarf_names:
        if gname not in all_results:
            continue
        r_inner = max(galaxies[gname]['R'][0], 0.1)
        for m in model_names:
            if m not in all_results[gname]:
                continue
            minfo = MODELS[m]
            halo_p = minfo['rho_unpack'](all_results[gname][m]['halo_params'])
            alpha_val = inner_log_slope(minfo['rho_func'], halo_p, r_eval_kpc=r_inner)
            slopes_by_model[m].append(alpha_val)

    print(f"\n  {'Model':8s} {'Median a':10s} {'Mean a':10s} {'Std':8s} {'%core(a>-0.5)':15s}")
    print("-" * 60)
    for m in model_names:
        vals = np.array(slopes_by_model[m])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        frac_core = 100 * np.mean(vals > -0.5)
        print(f"  {m:8s} {np.median(vals):+10.3f} {np.mean(vals):+10.3f} {np.std(vals):8.3f} {frac_core:15.1f}%")

    # ============================================================
    # TEST 4: DIVERSITY
    # ============================================================
    print(f"\n{'='*78}")
    print("  TEST 4: ROTATION-CURVE DIVERSITY")
    print(f"{'='*78}")

    vmax_data = {}
    for gname in dwarf_names:
        vmax_data[gname] = np.max(np.abs(galaxies[gname]['Vobs']))

    bins = [(20, 40), (40, 60), (60, 80), (80, 120)]
    for vlo, vhi in bins:
        group = [n for n in dwarf_names if vlo <= vmax_data.get(n, 0) < vhi]
        if len(group) < 2:
            continue
        print(f"\n  Vmax bin [{vlo},{vhi}) km/s: {len(group)} galaxies")
        for m in model_names:
            v_inner = []
            for gname in group:
                if gname not in all_results or m not in all_results[gname]:
                    continue
                gd = galaxies[gname]
                r = all_results[gname][m]
                minfo = MODELS[m]
                R_eval = min(2.0, gd['R'][-1] * 0.5)
                try:
                    V_DM = minfo['Vfunc'](np.array([R_eval]), r['halo_params'])[0]
                    Vtot = V_total(
                        np.array([R_eval]),
                        np.array([np.interp(R_eval, gd['R'], gd['Vgas'])]),
                        np.array([np.interp(R_eval, gd['R'], gd['Vdisk'])]),
                        np.array([np.interp(R_eval, gd['R'], gd['Vbul'])]),
                        np.array([V_DM]),
                        r['ML_disk']
                    )[0]
                    if vmax_data[gname] > 0:
                        v_inner.append(Vtot / vmax_data[gname])
                except Exception:
                    pass
            if len(v_inner) >= 2:
                v_inner = np.array(v_inner)
                print(f"    {m:8s}: V(2kpc)/Vmax spread = {np.std(v_inner):.3f} "
                      f"(range {np.min(v_inner):.2f}-{np.max(v_inner):.2f})")

    # ============================================================
    # TEST 5: LEAVE-ONE-OUT
    # ============================================================
    print(f"\n{'='*78}")
    print("  TEST 5: LEAVE-ONE-OUT ROBUSTNESS")
    print(f"{'='*78}")

    full_ranking = {}
    for m in model_names:
        med = np.nanmedian(model_stats[m]['red_chi2']) if model_stats[m]['red_chi2'] else 999
        full_ranking[m] = med
    overall_best = min(full_ranking, key=full_ranking.get)
    print(f"\n  Full-sample best (median rchi2): {overall_best} ({full_ranking[overall_best]:.3f})")

    rank_changes = 0
    for gname_leave in dwarf_names:
        loo_ranking = {}
        for m in model_names:
            rchi2_vals = []
            for gname in dwarf_names:
                if gname == gname_leave:
                    continue
                if gname in all_results and m in all_results[gname]:
                    rchi2_vals.append(all_results[gname][m]['red_chi2'])
            loo_ranking[m] = np.nanmedian(rchi2_vals) if rchi2_vals else 999
        loo_best = min(loo_ranking, key=loo_ranking.get)
        if loo_best != overall_best:
            rank_changes += 1
            print(f"    Removing {gname_leave}: best -> {loo_best}")

    print(f"\n  Ranking changes: {rank_changes}/{len(dwarf_names)} "
          f"({'STABLE' if rank_changes <= 2 else 'UNSTABLE'})")

    # ============================================================
    # GENERATE PLOTS
    # ============================================================
    print(f"\n{'='*78}")
    print("  GENERATING PLOTS")
    print(f"{'='*78}")

    pdf_path = os.path.join(OUT_DIR, 'tb_dm_rotation_curves.pdf')
    with PdfPages(pdf_path) as pdf:
        # Rotation curves for each galaxy
        for gname in dwarf_names[:20]:
            if gname not in all_results:
                continue
            gd = galaxies[gname]
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.errorbar(gd['R'], gd['Vobs'], yerr=gd['eVobs'], fmt='ko', ms=4, label='Observed', zorder=10)
            colors = {'NFW': 'red', 'Burkert': 'blue', 'ISO': 'green',
                      'TB1': 'purple', 'TB2': 'orange', 'TB3': 'brown'}
            R_fine = np.linspace(max(gd['R'][0], 0.01), gd['R'][-1], 200)
            for mn in model_names:
                if mn not in all_results[gname]:
                    continue
                r = all_results[gname][mn]
                minfo = MODELS[mn]
                try:
                    V_DM = minfo['Vfunc'](R_fine, r['halo_params'])
                    Vg = np.interp(R_fine, gd['R'], gd['Vgas'])
                    Vd = np.interp(R_fine, gd['R'], gd['Vdisk'])
                    Vb = np.interp(R_fine, gd['R'], gd['Vbul'])
                    Vtot = V_total(R_fine, Vg, Vd, Vb, V_DM, r['ML_disk'])
                    ax.plot(R_fine, Vtot, color=colors.get(mn, 'gray'),
                            label=f"{mn} (rchi2={r['red_chi2']:.2f})", alpha=0.8, lw=1.5)
                except Exception:
                    pass
            ax.set_xlabel('R [kpc]'); ax.set_ylabel('V [km/s]')
            ax.set_title(gname); ax.legend(fontsize=7, loc='lower right')
            ax.set_xlim(0, None); ax.set_ylim(0, None)
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Inner slope boxplot
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, m in enumerate(model_names):
            vals = np.array(slopes_by_model[m]); vals = vals[~np.isnan(vals)]
            if len(vals) == 0: continue
            bp = ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True)
            c = colors.get(m, 'gray')
            for patch in bp['boxes']: patch.set_facecolor(c); patch.set_alpha(0.5)
        ax.axhline(-1, color='red', ls='--', alpha=0.5, label='cusp (a=-1)')
        ax.axhline(0, color='blue', ls='--', alpha=0.5, label='core (a=0)')
        ax.set_xticks(range(len(model_names))); ax.set_xticklabels(model_names)
        ax.set_ylabel('Inner log-slope'); ax.set_title('Core-Cusp Comparison')
        ax.legend(); plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # AIC comparison histograms
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for j, ref in enumerate(['NFW', 'Burkert']):
            ax = axes[j]
            dkey = 'daic_nfw' if ref == 'NFW' else 'daic_burkert'
            for m in model_names:
                if m == ref: continue
                vals = np.array(model_stats[m][dkey]); vals = vals[~np.isnan(vals)]
                if len(vals) == 0: continue
                ax.hist(vals, bins=15, alpha=0.4, label=m, color=colors.get(m, 'gray'))
            ax.axvline(0, color='k', ls='--'); ax.axvline(-2, color='gray', ls=':', alpha=0.5)
            ax.axvline(2, color='gray', ls=':', alpha=0.5)
            ax.set_xlabel(f'dAIC (model - {ref})'); ax.set_ylabel('Count')
            ax.set_title(f'AIC vs {ref}'); ax.legend(fontsize=7)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Density profiles for first galaxy
        if dwarf_names and dwarf_names[0] in all_results:
            gname = dwarf_names[0]
            fig, ax = plt.subplots(figsize=(8, 6))
            r_arr = np.logspace(-2, 1.5, 200)
            for m in model_names:
                if m not in all_results[gname]: continue
                minfo = MODELS[m]
                hp = minfo['rho_unpack'](all_results[gname][m]['halo_params'])
                rho_vals = minfo['rho_func'](r_arr, hp)
                ax.loglog(r_arr, rho_vals, color=colors.get(m, 'gray'), label=m, lw=1.5)
            ax.set_xlabel('r [kpc]'); ax.set_ylabel(r'$\rho$ [M$_\odot$/kpc$^3$]')
            ax.set_title(f'Density profiles - {gname}'); ax.legend()
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # rc vs Mbar scaling
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for j, m in enumerate(['TB1', 'TB2', 'TB3']):
            ax = axes[j]
            hp = np.array(model_stats[m]['halo_params'])
            Mbar = np.array(model_stats[m]['Mbar'])
            if len(hp) < 3: continue
            rc_vals = hp[:, 1]; mask = (rc_vals > 0) & (Mbar > 0)
            if np.sum(mask) < 3: continue
            lrc = np.log10(rc_vals[mask]); lMb = np.log10(Mbar[mask])
            ax.scatter(lMb, lrc, alpha=0.6)
            if np.std(lMb) > 0.01:
                sl, ic = np.polyfit(lMb, lrc, 1)
                xf = np.linspace(lMb.min(), lMb.max(), 50)
                ax.plot(xf, sl*xf+ic, 'r-', label=f'slope={sl:.2f}')
            ax.set_xlabel('log10(Mbar/Msun)'); ax.set_ylabel('log10(rc/kpc)')
            ax.set_title(f'{m}: rc vs Mbar'); ax.legend()
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Reduced chi2 distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        for m in model_names:
            vals = model_stats[m]['red_chi2']
            if not vals: continue
            ax.hist(vals, bins=15, alpha=0.4, color=colors.get(m, 'gray'),
                    label=f'{m} (med={np.median(vals):.2f})')
        ax.set_xlabel(r'Reduced $\chi^2$'); ax.set_ylabel('Count')
        ax.set_title(r'Reduced $\chi^2$ distribution'); ax.legend()
        ax.axvline(1, color='k', ls='--', alpha=0.5)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print(f"  Plots saved to: {pdf_path}")

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print(f"\n{'='*78}")
    print("  FINAL VERDICT")
    print(f"{'='*78}")

    tb_models = ['TB1', 'TB2', 'TB3']
    best_tb = None; best_tb_med = 999
    for m in tb_models:
        if model_stats[m]['red_chi2']:
            med = np.nanmedian(model_stats[m]['red_chi2'])
            if med < best_tb_med:
                best_tb_med = med; best_tb = m

    burkert_med = np.nanmedian(model_stats['Burkert']['red_chi2']) if model_stats['Burkert']['red_chi2'] else 999
    nfw_med = np.nanmedian(model_stats['NFW']['red_chi2']) if model_stats['NFW']['red_chi2'] else 999
    iso_med = np.nanmedian(model_stats['ISO']['red_chi2']) if model_stats['ISO']['red_chi2'] else 999

    print(f"\n  Median reduced chi2:")
    print(f"    NFW:     {nfw_med:.4f}")
    print(f"    Burkert: {burkert_med:.4f}")
    print(f"    ISO:     {iso_med:.4f}")
    if best_tb:
        print(f"    Best TB ({best_tb}): {best_tb_med:.4f}")

    tb_best_daic_bur = np.nanmean(model_stats[best_tb]['daic_burkert']) if best_tb else 999
    tb_best_daic_nfw = np.nanmean(model_stats[best_tb]['daic_nfw']) if best_tb else 999
    tb_best_dbic_bur = np.nanmean(model_stats[best_tb]['dbic_burkert']) if best_tb else 999
    tb_best_dbic_nfw = np.nanmean(model_stats[best_tb]['dbic_nfw']) if best_tb else 999

    print(f"\n  Mean delta-AIC (best TB vs references):")
    print(f"    vs NFW:     {tb_best_daic_nfw:+.2f}")
    print(f"    vs Burkert: {tb_best_daic_bur:+.2f}")
    print(f"  Mean delta-BIC (best TB vs references):")
    print(f"    vs NFW:     {tb_best_dbic_nfw:+.2f}")
    print(f"    vs Burkert: {tb_best_dbic_bur:+.2f}")

    # Universality
    if best_tb and model_stats[best_tb]['halo_params']:
        hp = np.array(model_stats[best_tb]['halo_params'])
        beta_vals = hp[:, 2]
        beta_cv = np.std(beta_vals) / max(np.mean(beta_vals), 1e-10)
        universal = beta_cv < 0.3
    else:
        beta_cv = np.inf; universal = False

    # Core fractions
    tb_slopes = np.array(slopes_by_model.get(best_tb, []))
    tb_slopes = tb_slopes[~np.isnan(tb_slopes)]
    nfw_slopes = np.array(slopes_by_model.get('NFW', []))
    nfw_slopes = nfw_slopes[~np.isnan(nfw_slopes)]
    tb_core_frac = np.mean(tb_slopes > -0.5) if len(tb_slopes) > 0 else 0
    nfw_core_frac = np.mean(nfw_slopes > -0.5) if len(nfw_slopes) > 0 else 0

    stable = rank_changes <= 2

    print(f"\n  Criteria check:")
    print(f"    TB beats NFW in median rchi2?           {'YES' if best_tb_med < nfw_med else 'NO'}")
    print(f"    TB beats Burkert in median rchi2?       {'YES' if best_tb_med < burkert_med else 'NO'}")
    print(f"    TB beats Burkert in mean AIC?           {'YES' if tb_best_daic_bur < -2 else 'NO' if tb_best_daic_bur > 2 else 'MARGINAL'}")
    print(f"    TB beats Burkert in mean BIC?           {'YES' if tb_best_dbic_bur < -2 else 'NO' if tb_best_dbic_bur > 2 else 'MARGINAL'}")
    print(f"    TB beta/nu is universal (CV<0.3)?       {'YES' if universal else 'NO'} (CV={beta_cv:.3f})")
    print(f"    TB produces cores (a>-0.5) >80%?        {'YES' if tb_core_frac > 0.8 else 'NO'} ({100*tb_core_frac:.0f}%)")
    print(f"    NFW cusp fraction (a<=-0.5):            {100*(1-nfw_core_frac):.0f}%")
    print(f"    Results robust (leave-one-out)?          {'YES' if stable else 'NO'}")

    strong = (best_tb_med < nfw_med and
              tb_best_daic_bur < -2 and
              universal and
              tb_core_frac > 0.8 and
              stable)
    moderate = (best_tb_med < nfw_med and
                tb_best_daic_bur < 2 and
                tb_core_frac > 0.5)

    if strong:
        verdict = "STRONG SUPPORT"
        expl = ("TB robustly outperforms NFW and is competitive with/better than "
                "Burkert with similar effective freedom. Parameters show universality.")
    elif moderate:
        verdict = "MODERATE SUPPORT"
        expl = ("TB works well and resolves core-cusp for most galaxies, but is not "
                "clearly better than existing core profiles (Burkert/ISO) after AIC/BIC penalty.")
    else:
        verdict = "NO SUPPORT"
        expl = ("TB does not provide meaningful advantage over standard profiles, "
                "or requires too much fine-tuning per galaxy.")

    print(f"\n  +{'='*60}+")
    print(f"  | VERDICT: {verdict:48s} |")
    print(f"  +{'='*60}+")
    print(f"\n  {expl}")
    print(f"\n  Final question: Does TB transition from concept to")
    print(f"  predictive model of dwarf galaxy halo structure?")
    if strong:
        print(f"  -> YES: predictive, robust, parsimonious.")
    elif moderate:
        print(f"  -> PARTIALLY: viable alternative, not clearly superior.")
    else:
        print(f"  -> NOT YET: remains conceptual for halo profiles.")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # Save JSON
    json_path = os.path.join(OUT_DIR, 'tb_dm_results.json')
    save_data = {
        'dwarf_names': dwarf_names, 'model_names': model_names,
        'verdict': verdict, 'full_ranking': {m: float(v) for m, v in full_ranking.items()},
        'best_tb': best_tb, 'best_tb_med_rchi2': float(best_tb_med),
        'nfw_med_rchi2': float(nfw_med), 'burkert_med_rchi2': float(burkert_med),
        'results': {}
    }
    for gname in dwarf_names:
        if gname in all_results:
            save_data['results'][gname] = {}
            for m in model_names:
                if m in all_results[gname]:
                    r = all_results[gname][m]
                    save_data['results'][gname][m] = {
                        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in r.items()
                    }
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved to: {json_path}")
    print(f"\n{'='*78}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*78}")

if __name__ == '__main__':
    run_analysis()
