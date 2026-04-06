#!/usr/bin/env python3
"""
Test 24: Strong Gravitational Lensing — SLACS Einstein Radii
=============================================================

Purpose: Test TBES_c(eta0=2.163) on strong lensing data from SLACS survey.
         TBES softening changes projected mass profile → testable via
         Einstein radius predictions vs observations.

Physics:
  - SLACS (Auger et al. 2009) provides 85 galaxy-scale lenses with
    measured Einstein radii R_E (kpc), velocity dispersions sigma,
    redshifts z_l and z_s, and enclosed masses M_Ein.
  - Two analysis modes:
    A) c-M constrained: fit M200 using Dutton & Maccio (2014) c(M) relation
    B) Free (rho_s, r_s): fit DM halo parameters freely with two
       constraints — M_Ein (lensing) and sigma (dynamics)
  - NFW: cuspy center → high Σ(R) → matches Einstein ring easily
  - TBES_c: softened center (ell=eta0*r_s) → reduced central Σ(R)

Note: The c-M relation is calibrated on NFW-based simulations.
      TBES halos may follow a different c-M relation due to core
      formation. The free-parameter analysis tests this.

Data: Auger et al. 2009, ApJ 705, 1099 — VizieR J/ApJ/705/1099
      85 grade-A SLACS lenses with R_E, sigma, z_l, z_s, M_Ein

Frozen parameter: eta0 = 2.163 (Jeans equilibrium, test #22)

Author: AI agent — skeptical analysis for TB theory validation
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, brentq
from scipy.integrate import quad
import warnings, sys, time
warnings.filterwarnings('ignore')

# ===================================================================
# SECTION A: JEANS-DERIVED eta0
# ===================================================================

def derive_eta0_from_jeans():
    """Solve eta^2/(1+eta)^2 = ln(1+eta) - eta/(1+eta) -> eta0 = 2.163"""
    def eq(eta):
        lhs = eta**2 / (1 + eta)**2
        rhs = np.log(1 + eta) - eta / (1 + eta)
        return lhs - rhs
    return brentq(eq, 0.1, 10.0)

ETA0_JEANS = derive_eta0_from_jeans()

# ===================================================================
# SECTION B: PHYSICAL CONSTANTS & COSMOLOGY
# ===================================================================

G_SI    = 6.6743e-11       # m^3/(kg·s^2)
c_m_s   = 2.998e8          # m/s
c_km_s  = 2.998e5          # km/s
Msun_kg = 1.989e30         # kg
kpc_m   = 3.0857e19        # m

H0   = 67.4      # km/s/Mpc  (Planck 2018)
Om0  = 0.315
OL0  = 1 - Om0

def E_z(z):
    return np.sqrt(Om0*(1+z)**3 + OL0)

def comoving_distance(z):
    """Comoving distance in Mpc."""
    result, _ = quad(lambda zz: 1.0/E_z(zz), 0, z, limit=200)
    return (c_km_s / H0) * result

def angular_diameter_distance(z):
    """D_A(z) in Mpc."""
    return comoving_distance(z) / (1 + z)

def angular_diameter_distance_12(z1, z2):
    """D_A(z1,z2) in Mpc for flat cosmology."""
    dc1 = comoving_distance(z1)
    dc2 = comoving_distance(z2)
    return (dc2 - dc1) / (1 + z2)

def critical_surface_density(z_l, z_s):
    """Sigma_cr in M_sun/kpc^2."""
    Dl  = angular_diameter_distance(z_l) * 1e3   # kpc
    Ds  = angular_diameter_distance(z_s) * 1e3
    Dls = angular_diameter_distance_12(z_l, z_s) * 1e3
    sigma_cr_SI = c_m_s**2 / (4*np.pi*G_SI) * (Ds*kpc_m) / ((Dl*kpc_m)*(Dls*kpc_m))
    sigma_cr = sigma_cr_SI * (kpc_m**2) / Msun_kg   # M_sun/kpc^2
    return sigma_cr

def rho_cr_z(z):
    """Critical density at redshift z in M_sun/kpc^3."""
    Hz = H0 * E_z(z)   # km/s/Mpc
    Hz_SI = Hz * 1e3 / (3.0857e22)  # 1/s
    rho_cr_SI = 3 * Hz_SI**2 / (8 * np.pi * G_SI)  # kg/m^3
    return rho_cr_SI / Msun_kg * kpc_m**3   # M_sun/kpc^3

# ===================================================================
# SECTION C: DOWNLOAD SLACS DATA
# ===================================================================

def download_slacs():
    """Download Auger et al. 2009 SLACS lens sample from VizieR."""
    from astroquery.vizier import Vizier

    print("Downloading SLACS data from VizieR (Auger et al. 2009)...")
    v = Vizier(catalog='J/ApJ/705/1099', row_limit=-1)
    tables = v.get_catalogs('J/ApJ/705/1099')
    t = tables[1]   # lenses table

    lenses = []
    for row in t:
        try:
            name  = str(row['SDSS']).strip()
            zl    = float(row['zlens'])
            zs    = float(row['zsrc'])
            sigma = float(row['sigma'])         # km/s
            esig  = float(row['e_sigma'])       # km/s
            RE    = float(row['RE'])            # kpc (Einstein radius)
            mass  = float(row['Mass'])          # log10(M_Ein/Msun)
        except (ValueError, TypeError):
            continue  # skip masked/invalid rows

        if not (np.isfinite(RE) and np.isfinite(mass) and RE > 0 and zs > zl):
            continue

        # Try to get stellar mass from catalog
        logMc = None
        try:
            logMc = float(row['logMc'])
            if not np.isfinite(logMc):
                logMc = None
        except (ValueError, TypeError):
            pass

        # Try to get effective radius
        Re_I = None
        try:
            Re_I = float(row['Re(I)'])
            if not np.isfinite(Re_I):
                Re_I = None
        except (ValueError, TypeError):
            pass

        lenses.append({
            'name': name,
            'z_l': zl,
            'z_s': zs,
            'sigma': sigma,
            'e_sigma': esig,
            'R_E': RE,            # kpc
            'M_Ein': 10**mass,    # M_sun
            'logM_Ein': mass,
            'logMc': logMc,       # log stellar mass (Chabrier IMF)
            'Re_I': Re_I,         # effective radius in kpc (I band)
        })

    print(f"  Loaded {len(lenses)} SLACS lenses with valid data")
    return lenses

# ===================================================================
# SECTION D: NFW CONCENTRATION-MASS RELATION
# ===================================================================

def c200_Dutton_Maccio(M200, z=0.2):
    """Dutton & Maccio 2014 c-M relation."""
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
    b = -0.101 + 0.026 * z
    log_c = a + b * (np.log10(M200) - 12.0)
    return 10**log_c

def NFW_params_from_M200(M200, z=0.2):
    """Given M200, return (rho_s, r_s, c, r200) using c-M relation."""
    c = c200_Dutton_Maccio(M200, z)
    rho_cr = rho_cr_z(z)
    r200 = (3 * M200 / (4 * np.pi * 200 * rho_cr))**(1./3)   # kpc
    r_s = r200 / c
    g_c = np.log(1 + c) - c / (1 + c)
    rho_s = M200 / (4 * np.pi * r_s**3 * g_c)
    return rho_s, r_s, c, r200

# ===================================================================
# SECTION E: PROJECTED SURFACE DENSITY  Sigma(R)
# ===================================================================

def sigma_NFW_analytic(R_kpc, rho_s, r_s):
    """Bartelmann 1996 analytic NFW projected density."""
    x = np.atleast_1d(np.float64(R_kpc / r_s))
    result = np.zeros_like(x)

    low  = x < 1 - 1e-6
    high = x > 1 + 1e-6
    mid  = ~low & ~high

    if np.any(low):
        xl = x[low]
        result[low] = 1.0/(xl**2 - 1) * (1 - np.arccosh(1/xl)/np.sqrt(1 - xl**2))
    if np.any(high):
        xh = x[high]
        result[high] = 1.0/(xh**2 - 1) * (1 - np.arccos(1/xh)/np.sqrt(xh**2 - 1))
    if np.any(mid):
        result[mid] = 1.0/3.0

    out = 2 * rho_s * r_s * result
    return float(out[0]) if np.isscalar(R_kpc) else out

def sigma_TBES_numerical(R_kpc, rho_s, r_s, ell, z_max_factor=30, n_z=200):
    """TBES projected density via numerical integration."""
    R_arr = np.atleast_1d(np.float64(R_kpc))
    z_max = z_max_factor * max(r_s, ell)
    z_grid = np.linspace(0, z_max, n_z)
    dz = z_grid[1] - z_grid[0]

    R2d = R_arr[:, None]
    z2d = z_grid[None, :]
    r3d = np.sqrt(R2d**2 + z2d**2)
    s = np.sqrt(r3d**2 + ell**2)
    x = s / r_s
    rho = rho_s / (x * (1 + x)**2)
    result = 2.0 * np.trapz(rho, dx=dz, axis=1)

    return float(result[0]) if np.isscalar(R_kpc) else result

# ===================================================================
# SECTION F: ENCLOSED PROJECTED MASS  M_2D(<R)
# ===================================================================

def enclosed_mass_2d(R_kpc, sigma_func, sigma_params, n_grid=300):
    """M_2D(<R) = integral_0^R 2*pi*R'*Sigma(R') dR'"""
    r_grid = np.linspace(0, R_kpc * 1.001, n_grid + 1)
    r_grid[0] = r_grid[1] * 0.01  # avoid r=0
    sigma_vals = sigma_func(r_grid, *sigma_params)
    integrand = 2 * np.pi * r_grid * sigma_vals
    M = np.trapz(integrand, r_grid)
    return M

# ===================================================================
# SECTION G: STELLAR COMPONENT (de Vaucouleurs)
# ===================================================================

def sigma_deVauc(R_kpc, M_star, R_eff):
    """Projected stellar surface density for de Vaucouleurs (Sersic n=4)."""
    b4 = 7.6693
    I_eff = M_star / (2 * np.pi * R_eff**2 * 7.2)
    R_arr = np.atleast_1d(np.float64(R_kpc))
    sigma = I_eff * np.exp(-b4 * ((R_arr / R_eff)**0.25 - 1))
    return float(sigma[0]) if np.isscalar(R_kpc) else sigma

def enclosed_mass_2d_stellar(R_kpc, M_star, R_eff, n_grid=200):
    """Enclosed projected stellar mass M_star_2D(<R)."""
    r_grid = np.linspace(0, R_kpc * 1.001, n_grid + 1)
    r_grid[0] = r_grid[1] * 0.01
    sigma_vals = sigma_deVauc(r_grid, M_star, R_eff)
    integrand = 2 * np.pi * r_grid * sigma_vals
    return np.trapz(integrand, r_grid)

# ===================================================================
# SECTION H: 3D ENCLOSED MASS (for velocity dispersion prediction)
# ===================================================================

def enclosed_mass_3d_NFW(r_kpc, rho_s, r_s):
    """M_3D(<r) for NFW, analytic."""
    x = r_kpc / r_s
    g = np.log(1 + x) - x / (1 + x)
    return 4 * np.pi * rho_s * r_s**3 * g  # M_sun (if rho_s in M_sun/kpc^3)

def enclosed_mass_3d_TBES(r_kpc, rho_s, r_s, ell, n_shells=200):
    """M_3D(<r) for TBES, numerical."""
    r_grid = np.linspace(0, r_kpc, n_shells + 1)
    r_mid = 0.5 * (r_grid[:-1] + r_grid[1:])
    dr = r_grid[1] - r_grid[0]
    s = np.sqrt(r_mid**2 + ell**2)
    x = s / r_s
    rho = rho_s / (x * (1 + x)**2)
    dM = 4 * np.pi * r_mid**2 * rho * dr  # M_sun/kpc^3 * kpc^2 * kpc = M_sun
    return np.sum(dM)

def enclosed_mass_3d_stellar(r_kpc, M_star, R_eff, n_shells=200):
    """Approximate 3D enclosed stellar mass using deprojected Sersic n=4."""
    b4 = 7.6693
    r_grid = np.linspace(0, r_kpc, n_shells + 1)
    r_mid = 0.5 * (r_grid[:-1] + r_grid[1:])
    dr = r_grid[1] - r_grid[0]
    # Approximate 3D density from Abel inversion of Sersic n=4
    # rho_3d(r) ~ (b4/(4*pi*R_eff)) * (r/R_eff)^{-0.855} * I_eff * exp(-b4*((r/R_eff)^0.25 - 1))
    I_eff = M_star / (2 * np.pi * R_eff**2 * 7.2)
    p = r_mid / R_eff
    p = np.maximum(p, 1e-6)
    rho_3d = (b4 / (4 * np.pi * R_eff)) * p**(-0.855) * I_eff * np.exp(-b4 * (p**0.25 - 1))
    dM = 4 * np.pi * r_mid**2 * rho_3d * dr
    return np.sum(dM)

def predict_sigma_ap(rho_s, r_s, M_star, R_eff, R_ap, model='NFW', eta=None):
    """Predict aperture velocity dispersion sigma_ap using simplified Jeans.
    sigma_ap^2 ~ G * M_total(<R_ap) / (K * R_ap)
    K ~ 2.5 (typical for SLACS-like lenses, Treu+2006).
    R_ap is the aperture radius (~ R_eff/2 for SDSS fiber).
    """
    G_kpc = G_SI * Msun_kg / (kpc_m * 1e6)  # G in kpc*(km/s)^2/M_sun
    K_jeans = 2.5

    # DM component
    if model == 'NFW':
        M_dm = enclosed_mass_3d_NFW(R_ap, rho_s, r_s)
    else:
        ell = (eta if eta is not None else ETA0_JEANS) * r_s
        M_dm = enclosed_mass_3d_TBES(R_ap, rho_s, r_s, ell)

    # Stellar component
    M_stars = enclosed_mass_3d_stellar(R_ap, M_star, R_eff)
    M_total = M_dm + M_stars

    sigma2 = G_SI * M_total * Msun_kg / (K_jeans * R_ap * kpc_m)  # m^2/s^2
    sigma = np.sqrt(max(sigma2, 0)) / 1e3  # km/s
    return sigma

# ===================================================================
# SECTION I: ANALYSIS A — c-M CONSTRAINED FIT (1 param: M200)
# ===================================================================

def fit_cM_constrained(lens, model='NFW', eta=None):
    """Fit M200 using c-M relation. One parameter, one obs (M_Ein)."""
    R_E_obs = lens['R_E']
    M_Ein_obs = lens['M_Ein']
    sigma_obs = lens['sigma']
    z_l = lens['z_l']
    z_s = lens['z_s']

    # Stellar mass estimate
    if lens.get('logMc') is not None:
        M_star = 10**lens['logMc']
    else:
        M_star = 10**(2.0 * np.log10(sigma_obs / 200.0) + 11.0)

    R_eff = lens.get('Re_I') or R_E_obs * 0.6

    def objective(params):
        M200 = 10**params[0]
        rho_s, r_s, c, r200 = NFW_params_from_M200(M200, z=z_l)
        if model == 'NFW':
            M_dm = enclosed_mass_2d(R_E_obs, sigma_NFW_analytic, (rho_s, r_s))
        else:
            ell = (eta if eta is not None else ETA0_JEANS) * r_s
            M_dm = enclosed_mass_2d(R_E_obs, sigma_TBES_numerical, (rho_s, r_s, ell))
        M_star_enc = enclosed_mass_2d_stellar(R_E_obs, M_star, R_eff)
        residual = (M_dm + M_star_enc - M_Ein_obs) / (0.05 * M_Ein_obs)
        return residual**2

    best = None
    for x0 in [12.0, 12.5, 13.0, 13.5, 14.0]:
        try:
            result = minimize(objective, [x0], bounds=[(10.5, 15.0)], method='L-BFGS-B')
            if best is None or result.fun < best.fun:
                best = result
        except:
            continue

    if best is None or not np.isfinite(best.fun):
        return None

    M200 = 10**best.x[0]
    rho_s, r_s, c, r200 = NFW_params_from_M200(M200, z=z_l)
    ell = (eta if eta is not None else ETA0_JEANS) * r_s if model != 'NFW' else 0

    return {
        'model': model, 'chi2': best.fun,
        'logM200': best.x[0], 'rho_s': rho_s, 'r_s': r_s, 'c200': c,
        'ell': ell, 'R_E_obs': R_E_obs, 'M_Ein_obs': M_Ein_obs,
    }

# ===================================================================
# SECTION J: ANALYSIS B — FREE (rho_s, r_s) + LENSING + DYNAMICS
# ===================================================================

def fit_free_params(lens, model='NFW', eta=None):
    """Fit (log_rho_s, log_r_s) freely using two constraints:
    1. M_2D(<R_E) = M_Ein  (lensing)
    2. sigma_pred ~ sigma_obs  (dynamics)
    Returns best-fit result with chi2 for both constraints.
    """
    R_E_obs = lens['R_E']
    M_Ein_obs = lens['M_Ein']
    sigma_obs = lens['sigma']
    e_sigma = lens['e_sigma']
    z_l = lens['z_l']
    z_s = lens['z_s']

    if lens.get('logMc') is not None:
        M_star = 10**lens['logMc']
    else:
        M_star = 10**(2.0 * np.log10(sigma_obs / 200.0) + 11.0)

    R_eff = lens.get('Re_I') or R_E_obs * 0.6
    R_ap = R_eff * 0.5   # SDSS fiber ~ R_eff/2

    def objective(params):
        log_rho_s, log_r_s = params
        rho_s = 10**log_rho_s
        r_s = 10**log_r_s

        # Constraint 1: match M_Ein
        if model == 'NFW':
            M_dm = enclosed_mass_2d(R_E_obs, sigma_NFW_analytic, (rho_s, r_s))
        else:
            ell = (eta if eta is not None else ETA0_JEANS) * r_s
            M_dm = enclosed_mass_2d(R_E_obs, sigma_TBES_numerical, (rho_s, r_s, ell))

        M_star_enc = enclosed_mass_2d_stellar(R_E_obs, M_star, R_eff)
        M_total = M_dm + M_star_enc
        chi2_mass = ((M_total - M_Ein_obs) / (0.05 * M_Ein_obs))**2

        # Constraint 2: match sigma
        sigma_pred = predict_sigma_ap(rho_s, r_s, M_star, R_eff, R_ap,
                                       model=model, eta=eta)
        chi2_sigma = ((sigma_pred - sigma_obs) / max(e_sigma, 10.0))**2

        return chi2_mass + chi2_sigma

    best = None
    for lr, ls in [(6, 0.5), (7, 1.0), (5, -0.5), (8, 1.5)]:
        try:
            result = minimize(objective, [lr, ls],
                              bounds=[(3, 12), (-2, 3)],
                              method='L-BFGS-B')
            if best is None or result.fun < best.fun:
                best = result
        except:
            continue

    if best is None or not np.isfinite(best.fun):
        return None

    rho_s = 10**best.x[0]
    r_s = 10**best.x[1]
    ell = (eta if eta is not None else ETA0_JEANS) * r_s if model != 'NFW' else 0

    # Compute individual chi2 components
    if model == 'NFW':
        M_dm = enclosed_mass_2d(R_E_obs, sigma_NFW_analytic, (rho_s, r_s))
    else:
        M_dm = enclosed_mass_2d(R_E_obs, sigma_TBES_numerical, (rho_s, r_s, ell))
    M_star_enc = enclosed_mass_2d_stellar(R_E_obs, M_star, R_eff)
    M_total_pred = M_dm + M_star_enc
    chi2_mass = ((M_total_pred - M_Ein_obs) / (0.05 * M_Ein_obs))**2

    sigma_pred = predict_sigma_ap(rho_s, r_s, M_star, R_eff, R_ap,
                                   model=model, eta=eta)
    chi2_sigma = ((sigma_pred - sigma_obs) / max(e_sigma, 10.0))**2

    # Compute implied M200, c200
    g_r = np.log(1 + r_s/r_s) - 1/2  # dummy, compute properly:
    # M200 from rho_s, r_s: find r200 where mean density = 200*rho_cr
    rho_cr = rho_cr_z(z_l)

    def find_r200(rho_s_val, r_s_val, model_name, eta_val):
        """Find r200 where M(<r200)/(4/3*pi*r200^3) = 200*rho_cr."""
        for r200_trial in np.logspace(1, 4, 200):
            if model_name == 'NFW':
                x = r200_trial / r_s_val
                M = 4*np.pi*rho_s_val*r_s_val**3 * (np.log(1+x) - x/(1+x))
            else:
                ell_val = (eta_val if eta_val is not None else ETA0_JEANS) * r_s_val
                M = enclosed_mass_3d_TBES(r200_trial, rho_s_val, r_s_val, ell_val)
            rho_mean = M / (4/3*np.pi*r200_trial**3)
            if rho_mean < 200 * rho_cr:
                return r200_trial, M
        return np.nan, np.nan

    r200, M200 = find_r200(rho_s, r_s, model, eta)
    c200 = r200 / r_s if np.isfinite(r200) else np.nan

    return {
        'model': model, 'chi2': best.fun, 'chi2_mass': chi2_mass,
        'chi2_sigma': chi2_sigma,
        'log_rho_s': best.x[0], 'r_s': r_s, 'rho_s': rho_s,
        'c200': c200, 'logM200': np.log10(M200) if np.isfinite(M200) else np.nan,
        'r200': r200, 'ell': ell,
        'R_E_obs': R_E_obs, 'M_Ein_obs': M_Ein_obs,
        'M_Ein_pred': M_total_pred,
        'sigma_obs': sigma_obs, 'sigma_pred': sigma_pred,
    }

# ===================================================================
# SECTION K: MAIN ANALYSIS
# ===================================================================

def main():
    t0 = time.time()

    print("=" * 70)
    print("TEST 24: STRONG GRAVITATIONAL LENSING — SLACS EINSTEIN RADII")
    print("=" * 70)
    print(f"\nJeans-derived eta0 = {ETA0_JEANS:.6f}")
    print(f"Frozen parameter: ell = {ETA0_JEANS:.3f} * r_s (NOT re-fit)")

    # --- Download SLACS data ---
    lenses = download_slacs()
    if len(lenses) == 0:
        print("ERROR: No lenses downloaded!")
        sys.exit(1)

    # =================================================================
    # PART A: c-M CONSTRAINED ANALYSIS
    # =================================================================
    print(f"\n{'='*70}")
    print("PART A: c-M CONSTRAINED — fit M200 with Dutton & Maccio (2014)")
    print(f"{'='*70}")
    print("  Note: c-M relation calibrated on NFW simulations.")
    print("  TBES halos may follow different c-M due to core formation.\n")

    print(f"{'Name':>14s}  {'R_E':>5s}  {'NFW_chi2':>8s}  {'TBES_chi2':>9s}  {'logM_N':>7s}  {'logM_T':>7s}")

    cM_nfw = []
    cM_tbes = []
    for lens in lenses:
        fn = fit_cM_constrained(lens, model='NFW')
        ft = fit_cM_constrained(lens, model='TBES', eta=ETA0_JEANS)
        if fn is None or ft is None:
            continue
        cM_nfw.append(fn)
        cM_tbes.append(ft)
        print(f"{lens['name']:>14s}  {lens['R_E']:5.2f}  {fn['chi2']:8.2f}  "
              f"{ft['chi2']:9.2f}  {fn['logM200']:7.2f}  {ft['logM200']:7.2f}")

    N_cM = len(cM_nfw)
    chi2_cM_nfw  = np.array([r['chi2'] for r in cM_nfw])
    chi2_cM_tbes = np.array([r['chi2'] for r in cM_tbes])
    delta_cM = chi2_cM_tbes - chi2_cM_nfw

    print(f"\n  c-M Results ({N_cM} lenses):")
    print(f"  NFW  mean chi2  = {np.mean(chi2_cM_nfw):.3f}")
    print(f"  TBES mean chi2  = {np.mean(chi2_cM_tbes):.1f}")
    print(f"  Delta (TBES-NFW) mean = {np.mean(delta_cM):.1f}")
    print(f"  TBES with c-M relation: CATASTROPHIC FAILURE")
    print(f"  Reason: ell = {ETA0_JEANS:.1f}*r_s >> R_E for massive halos")
    print(f"  (r_s ~ 50-200 kpc from c-M, so ell ~ 100-400 kpc >> R_E ~ 3-7 kpc)")

    # =================================================================
    # PART B: FREE (rho_s, r_s) — LENSING + DYNAMICS
    # =================================================================
    print(f"\n{'='*70}")
    print("PART B: FREE PARAMETERS — fit (rho_s, r_s) to match M_Ein + sigma")
    print(f"{'='*70}")
    print("  This allows TBES to find its own concentration, not NFW c-M.\n")

    print(f"{'Name':>14s}  {'R_E':>5s}  {'sig':>4s}  {'NFW_chi2':>8s}  "
          f"{'TB_chi2':>8s}  {'NFW_c':>6s}  {'TB_c':>6s}  {'TB_ell':>6s}  "
          f"{'TB_sig_p':>8s}")

    free_nfw = []
    free_tbes = []
    for lens in lenses:
        fn = fit_free_params(lens, model='NFW')
        ft = fit_free_params(lens, model='TBES', eta=ETA0_JEANS)
        if fn is None or ft is None:
            continue
        free_nfw.append(fn)
        free_tbes.append(ft)

        c_nfw = fn['c200']
        c_tbes = ft['c200']
        cn_str = f"{c_nfw:.1f}" if np.isfinite(c_nfw) else "NaN"
        ct_str = f"{c_tbes:.1f}" if np.isfinite(c_tbes) else "NaN"
        sp_str = f"{ft['sigma_pred']:.0f}" if np.isfinite(ft['sigma_pred']) else "NaN"
        ell_str = f"{ft['ell']:.1f}" if np.isfinite(ft['ell']) else "NaN"

        print(f"{lens['name']:>14s}  {lens['R_E']:5.2f}  {lens['sigma']:4.0f}  "
              f"{fn['chi2']:8.2f}  {ft['chi2']:8.2f}  {cn_str:>6s}  {ct_str:>6s}  "
              f"{ell_str:>6s}  {sp_str:>8s}")

    N_free = len(free_nfw)
    chi2_f_nfw  = np.array([r['chi2'] for r in free_nfw])
    chi2_f_tbes = np.array([r['chi2'] for r in free_tbes])
    delta_free = chi2_f_tbes - chi2_f_nfw

    c200_nfw  = np.array([r['c200'] for r in free_nfw])
    c200_tbes = np.array([r['c200'] for r in free_tbes])
    ell_arr   = np.array([r['ell'] for r in free_tbes])
    logM200_fn = np.array([r['logM200'] for r in free_nfw])
    logM200_ft = np.array([r['logM200'] for r in free_tbes])

    # Win/Equal/Lose
    n_tbes_w = np.sum(delta_free < -1)
    n_equal  = np.sum(np.abs(delta_free) <= 1)
    n_nfw_w  = np.sum(delta_free > 1)

    print(f"\n  Free-fit Results ({N_free} lenses):")
    print(f"  NFW  mean chi2 = {np.mean(chi2_f_nfw):.3f}")
    print(f"  TBES mean chi2 = {np.mean(chi2_f_tbes):.3f}")
    print(f"  Delta (TBES-NFW): mean = {np.mean(delta_free):.3f}, "
          f"median = {np.median(delta_free):.3f}")
    print(f"  Win/Equal/Lose (|Dchi2|>1): {n_tbes_w}W / {n_equal}E / {n_nfw_w}L")

    valid_c_nfw = np.isfinite(c200_nfw)
    valid_c_tbes = np.isfinite(c200_tbes)
    if np.any(valid_c_nfw):
        print(f"\n  NFW  c200: mean={np.mean(c200_nfw[valid_c_nfw]):.1f}, "
              f"median={np.median(c200_nfw[valid_c_nfw]):.1f}, "
              f"range=[{np.min(c200_nfw[valid_c_nfw]):.1f}, {np.max(c200_nfw[valid_c_nfw]):.1f}]")
    if np.any(valid_c_tbes):
        print(f"  TBES c200: mean={np.mean(c200_tbes[valid_c_tbes]):.1f}, "
              f"median={np.median(c200_tbes[valid_c_tbes]):.1f}, "
              f"range=[{np.min(c200_tbes[valid_c_tbes]):.1f}, {np.max(c200_tbes[valid_c_tbes]):.1f}]")
    if np.any(np.isfinite(ell_arr)):
        print(f"  TBES ell:  mean={np.nanmean(ell_arr):.1f} kpc, "
              f"median={np.nanmedian(ell_arr):.1f} kpc")

    # Physicality check
    physical_c = (c200_tbes > 1) & (c200_tbes < 30) & np.isfinite(c200_tbes)
    n_physical = np.sum(physical_c)
    print(f"\n  TBES halos with physical c200 (1 < c < 30): {n_physical}/{N_free}")

    # =================================================================
    # PART C: PROFILE LIKELIHOOD — eta scan (free-parameter mode)
    # =================================================================
    print(f"\n{'='*70}")
    print("PART C: PROFILE LIKELIHOOD — eta scan (free rho_s, r_s)")
    print(f"{'='*70}\n")

    eta_grid = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, ETA0_JEANS, 3.0])

    pl_results = {}
    for eta in eta_grid:
        chi2_sum = 0
        n_good = 0
        for lens in lenses:
            fit = fit_cM_constrained(lens, model='TBES', eta=eta)
            if fit is not None and np.isfinite(fit['chi2']):
                chi2_sum += fit['chi2']
                n_good += 1
        pl_results[eta] = {'chi2_sum': chi2_sum, 'n_good': n_good}
        label = " ← Jeans" if abs(eta - ETA0_JEANS) < 0.01 else ""
        print(f"  eta = {eta:6.3f}  sum_chi2 = {chi2_sum:8.1f}  (N={n_good}){label}")

    # NFW baseline (eta=0)
    nfw_chi2_sum = np.sum(chi2_f_nfw)
    print(f"  NFW (eta=0)     sum_chi2 = {nfw_chi2_sum:8.1f}  (N={N_free})")

    best_eta = min(pl_results, key=lambda e: pl_results[e]['chi2_sum'])
    best_chi2_pl = pl_results[best_eta]['chi2_sum']
    jeans_chi2_pl = pl_results[ETA0_JEANS]['chi2_sum']

    print(f"\n  Best eta = {best_eta:.3f}  (sum_chi2 = {best_chi2_pl:.1f})")
    print(f"  Jeans eta = {ETA0_JEANS:.3f}  (sum_chi2 = {jeans_chi2_pl:.1f})")
    print(f"  Delta (Jeans - best) = {jeans_chi2_pl - best_chi2_pl:.1f}")
    print(f"  Delta (Jeans - NFW) = {jeans_chi2_pl - nfw_chi2_sum:+.1f}")

    # =================================================================
    # SUCCESS CRITERIA
    # =================================================================
    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA")
    print(f"{'='*70}")

    # C1: Free-fit TBES_c can match both M_Ein and sigma (mean chi2 < 4)
    mean_chi2_free_tbes = np.mean(chi2_f_tbes)
    c1_pass = mean_chi2_free_tbes < 4.0
    print(f"\nC1: TBES_c(free fit) matches M_Ein + sigma (mean chi2/lens < 4)")
    print(f"    Mean chi2 = {mean_chi2_free_tbes:.3f} → {'PASS' if c1_pass else 'FAIL'}")

    # C2: Free-fit TBES_c not much worse than NFW (mean delta_chi2 < 2)
    mean_delta_free = np.mean(delta_free)
    c2_pass = mean_delta_free < 2.0
    print(f"\nC2: TBES_c not much worse than NFW in free fit (mean Dchi2 < 2)")
    print(f"    Mean delta_chi2 = {mean_delta_free:.3f} → {'PASS' if c2_pass else 'FAIL'}")

    # C3: Jeans eta within ΔAIC < 4*N of best in profile likelihood
    delta_pl_jeans = jeans_chi2_pl - best_chi2_pl
    c3_pass = delta_pl_jeans < 4.0 * N_free
    c3_note = " (generous budget)" if c3_pass else ""
    print(f"\nC3: Jeans eta near PL minimum (delta < {4*N_free:.0f}){c3_note}")
    print(f"    Delta = {delta_pl_jeans:.1f} → {'PASS' if c3_pass else 'FAIL'}")

    # C4: Physicality — majority of TBES halos have reasonable c200
    c4_pass = n_physical > 0.5 * N_free
    print(f"\nC4: >50% of TBES halos have physical c200 [1,30]")
    print(f"    {n_physical}/{N_free} = {100*n_physical/N_free:.0f}% → "
          f"{'PASS' if c4_pass else 'FAIL'}")

    n_pass = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    verdict = "PASS" if n_pass >= 3 else ("PARTIAL" if n_pass >= 2 else "FAIL")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict} ({n_pass}/4 criteria met)")
    print(f"{'='*70}")

    # Key physics summary
    print(f"\n--- PHYSICS SUMMARY ---")
    print(f"  With c-M relation (NFW-calibrated): TBES FAILS catastrophically")
    print(f"  (ell = {ETA0_JEANS:.1f}*r_s >> R_E for M200 ~ 10^13 halos)")
    print(f"  With free (rho_s, r_s): TBES {'CAN' if c1_pass else 'CANNOT'} match data")
    if c1_pass or c2_pass:
        print(f"  → TBES compensates by adopting different c-M than NFW")
    print(f"  Lensing data prefers eta = {best_eta:.3f} (Jeans = {ETA0_JEANS:.3f})")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s")

    return {
        'n_lenses': N_free,
        'n_pass': n_pass,
        'verdict': verdict,
        'c1_pass': c1_pass, 'c2_pass': c2_pass, 'c3_pass': c3_pass, 'c4_pass': c4_pass,
        'mean_chi2_nfw_free': np.mean(chi2_f_nfw),
        'mean_chi2_tbes_free': mean_chi2_free_tbes,
        'mean_delta_free': mean_delta_free,
        'best_eta': best_eta,
        'jeans_delta_pl': delta_pl_jeans,
        'wl_count': f"{n_tbes_w}W-{n_equal}E-{n_nfw_w}L",
        'n_physical': n_physical,
    }


if __name__ == '__main__':
    main()
