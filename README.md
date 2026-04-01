# Twin-Barrier-Theory-of-Gravity

## 📄 Twin-Barrier-Theory.pdf

The complete theoretical framework is contained in the file **`Twin-Barrier-Theory.pdf`** included in this repository. The PDF contains the full Twin Barrier Theory of Gravity — a 5D braneworld model with twin-barrier scalar potential, Einstein-DeTurck formulation, and numerical validation.

### Document Integrity — SHA-256 Hash

The following cryptographic hash serves as a timestamped proof of the document's contents at the time of publication:

```
SHA-256: 2eab3ce32d2c9827305f4c9d7fec54f0a09ef623264d0d00972e8f93db6b56a8
File:    Twin-Barrier-Theory.pdf
```

To verify: `sha256sum Twin-Barrier-Theory.pdf`

---

# Einstein-DeTurck Braneworld Validation Suite

**9-stage numerical validation of the Randall-Sundrum braneworld — graviton spectrum (Stages 1–7, JAX/GPU), vacuum tunneling instanton (Stage 8, SciPy), and microscopic closure derivation (Stage 9, SciPy).**

This repository provides a complete, self-contained validation pipeline for the 5D Einstein-DeTurck formulation of the Randall-Sundrum braneworld model. Stages 1–7 validate the RS-II graviton spectrum on GPU via [JAX](https://github.com/jax-ml/jax) with 64-bit precision. Stage 8 solves the 5D Euclidean bounce instanton for the RS-I twin-barrier scalar potential, confirming that $S_B \approx 160$ reproduces the brane hierarchy $\alpha \sim 21$.

---

## Physics Background

The Randall-Sundrum II model places our 4D universe on a brane embedded in a 5D anti-de Sitter (AdS₅) bulk. The background metric is:

$$ds^2 = e^{-2ky}\bigl(-dt^2 + dr^2 + r^2\,d\Omega_2^2\bigr) + dy^2$$

where $k$ is the AdS curvature scale and $y \geq 0$ is the extra-dimensional coordinate. Gravity is localized near the brane ($y = 0$) via the exponential warp factor $e^{-2ky}$.

The **Einstein-DeTurck trick** modifies the Einstein equations into a well-posed elliptic system by subtracting a gauge-fixing term involving a reference connection:

$$E_{AB} = R_{AB} - \nabla_{(A}\xi_{B)} + \frac{2}{3}\,|\Lambda_5|\,g_{AB} = 0$$

This formulation is standard in numerical general relativity for static/stationary spacetimes.

---

## Validation Stages

### Stage 1 — Background Metric Verification
**File:** `background_test.py`  
**Dependencies:** `metric.py`, `grid.py`, `deturck.py`

Validates the analytical 5D warped background:
- DeTurck vector norm: $|\xi|^2 \approx 0$ (gauge consistency)
- Einstein-DeTurck residual: $\max|E_{AB}| \approx 0$ (field equations satisfied)
- Ricci scalar: $R = -20k^2$ (correct AdS₅ curvature)

**PASS criteria:** All quantities at machine precision ($< 10^{-10}$).

---

### Stage 2 — Linearized Brane Graviton
**File:** `stage2_source.py`

Solves the linearized graviton equation for a static point mass on the brane:

$$\left[\partial_r^2 + \frac{2}{r}\partial_r + \partial_y^2 - 4k\,\partial_y\right]\Phi = S(r,y)$$

using 2D finite differences on an $(r, y)$ grid with:
- **Neumann** at $r = 0$ (spherical regularity)
- **Robin BC** at $r = r_\text{max}$: $\partial_r(r\Phi) = 0$ (Coulomb decay)
- **Z₂ symmetry** at $y = 0$ (brane orbifold)
- **Dirichlet** at $y = y_\text{max}$ (bulk suppression)

The source is a regularized Gaussian approximation to the brane-localized delta function. The solution is obtained via dense `jnp.linalg.solve()` on GPU.

---

### Stage 3 — Kaluza-Klein Zero Mode
**File:** `stage3_zero_mode.py`

Verifies that the graviton zero mode ($m_0^2 = 0$) exists and has the correct profile. The 1D eigenvalue problem in the extra dimension is the Sturm-Liouville system:

$$-\frac{d}{dy}\!\left[e^{-4ky}\frac{d\psi}{dy}\right] = m^2\,e^{-4ky}\,\psi$$

Discretized using **piecewise-linear finite elements** with:
- **Stiffness matrix** $H$: tridiagonal, symmetric positive semi-definite
- **Lumped mass matrix** $M$: diagonal, with warp-factor weights $w_j = e^{-4ky_j}$
- **Standard eigenvalue form**: $S = M^{-1/2}HM^{-1/2}$, solved by `jnp.linalg.eigh`

The Neumann BC at the brane ($y = 0$) is the **natural boundary condition** of the variational form — it requires no special treatment. This is the key insight that made the discretization correct.

**PASS criteria:** $|m_0^2| < 10^{-6}$, flat profile ($\psi_0 \approx \text{const}$), stable under domain doubling.

---

### Stage 4 — KK Spectrum Convergence
**File:** `stage4_kk_spectrum.py`

Computes the full KK mass spectrum at three resolutions ($N$, $2N$, $4N$) and verifies:
- **Convergence:** eigenvalue relative change $< 2\%$ between resolutions
- **Spectral gap:** $m_1^2 - m_0^2 > 0$ and stable
- **No drift:** physical modes persist across resolutions
- **Smooth density:** no spurious clustering or gaps

**PASS criteria:** All first 20 eigenvalues converge within 2%.

---

### Stage 5 — Ghost & Tachyon Exclusion (Fatal Gate)
**File:** `stage5_ghost_tachyon.py`

This is a **hard veto** — any failure here means the theory is pathological:

- **Ghost check:** kinetic matrix $K$ must have $\lambda_\text{min}(K) \geq 0$  
  (negative kinetic energy → infinite-energy decay, physically unacceptable)
- **Tachyon check:** mass spectrum must have $m_\text{min}^2 \geq -10^{-6}$  
  (negative $m^2$ → exponential time growth, vacuum instability)

In the Schrödinger picture $\psi = e^{2ky}\chi$, the Hamiltonian $H = -\partial_y^2 + 4k^2$ is manifestly positive, guaranteeing both conditions analytically. This stage confirms it numerically.

**PASS criteria:** $\lambda_\text{min}(K) \geq 0$, $m_\text{min}^2 \geq -10^{-6}$.

---

### Stage 6 — Time Evolution Stability
**File:** `stage6_time_evolution.py`

Evolves a Gaussian brane perturbation under the wave equation:

$$\partial_t^2\,h + L_y\,h = 0$$

using a **symplectic leapfrog (Störmer-Verlet)** integrator with:
- $A = M^{-1}H$ as the evolution operator
- Energy $E = \tfrac{1}{2}v^T M v + \tfrac{1}{2}h^T H h$ (conserved quantity)
- Weighted norm $\|h\|_M = \sqrt{h^T M h}$ (physically meaningful L² norm)

The CFL condition $\Delta t < 2/\sqrt{\lambda_\text{max}(A)}$ is enforced automatically.

**PASS criteria:** Energy relative variance $< 1\%$, growth rate $\omega < 0.01$, norm ratio $< 5$, no individual mode blow-up.

---

### Stage 7 — PPN Relativistic Consistency
**File:** `stage7_ppn.py` (or `run_stage7_fast.py` for optimized version)

Verifies that brane gravity recovers General Relativity by:
1. Solving Stage 2 for the Newtonian potential $V(r) = -\Phi(r, 0)/2$
2. Fitting $V(r) = A/r + B$ to extract amplitude and shape
3. Computing the PPN parameter $\gamma$ (should be 1 for GR)
4. Checking light-bending ratio $(1+\gamma)/2 = 1$

In the linearized single-field regime, $\gamma = 1$ by construction since the same potential enters both $g_{tt}$ and $g_{rr}$. The meaningful check is that the solution actually produces the correct $1/r$ fall-off (not $1/r^3$ from extra dimensions).

**PASS criteria:** $|\gamma - 1| < 10^{-3}$, $R^2 > 0.999$, $A < 0$ (attractive gravity).

---

### Stage 8 — 5D Euclidean Bounce Instanton
**File:** `instanton_bounce.py`  
**Supporting file:** `pde_derivation.py`

Solves the 5D warped Euclidean bounce for the twin-barrier scalar potential:

$$V(\Phi) = \frac{\lambda}{4}(\Phi^2 - u^2)^2 - \eta\,u^3\,\Phi$$

in the Randall-Sundrum warped background:

$$ds_E^2 = e^{-2ky}(d\rho^2 + \rho^2\,d\Omega_3^2) + dy^2$$

The 5D Euler-Lagrange equation (derived independently in `pde_derivation.py`):

$$\partial_y^2\Phi - 4k\,\partial_y\Phi + e^{2ky}\!\left[\partial_\rho^2\Phi + \frac{3}{\rho}\,\partial_\rho\Phi\right] = V'(\Phi)$$

**Method:** For UV-localized bounce ($\partial_y\Phi \approx 0$), the problem reduces to a 4D O(4)-symmetric BVP:

$$\phi'' + \frac{3}{\rho}\,\phi' = V'(\phi), \quad \phi'(0) = 0,\quad \phi(\rho_{\max}) = \phi_{\text{false}}$$

solved via `scipy.integrate.solve_bvp` with a tanh initial guess (wall position/width automatically varied). The 5D action uses warp-integrated effective lengths:

$$S_B^{5D} = S_{\text{kin}} \cdot L_{\text{eff}}^{\text{kin}} + S_{\text{pot}} \cdot L_{\text{eff}}^{\text{pot}}$$

where $L_{\text{eff}}^{\text{kin}} = \frac{1-e^{-2kL}}{2k}$ and $L_{\text{eff}}^{\text{pot}} = \frac{1-e^{-4kL}}{4k}$ arise from the different warp weights of kinetic ($e^{-2ky}$) and potential ($e^{-4ky}$) terms.

**Result:** For $\lambda = 0.1$, $\eta/\eta_{\text{crit}} = 0.98$ (with $\eta_{\text{crit}} = 2\lambda u / 3\sqrt{3}$):

| Quantity | Value |
|:--|:--|
| $S_B^{4D}$ | 219.90 |
| $S_B^{5D}$ | **164.94** |
| Virial $S_{\text{kin}}/|S_{\text{pot}}|$ | 2.000 |
| $\alpha(\nu=3) = S_B/(2\nu+2)$ | **20.62 ≈ 21** |

Grid convergence: $\delta S_B/S_B < 0.005\%$ (N=100→800). Domain convergence: $< 0.01\%$ (R=50→200).

Parameter scan (10×14 grid over $\lambda \in [0.05, 0.15]$, $\eta/\eta_c \in [0.92, 0.995]$) finds 7 points with $S_B \in [150, 170]$, confirming $S_B \approx 160$ is robust.

**PASS A** (regularity): finite action, $\phi'(0)=0$, no NaN/Inf, reaches both vacua ✓  
**PASS B** (thin-wall): $S_B^{\text{num}}/S_B^{\text{tw}} = 0.083$ — correct for thick-wall regime ✓  
**PASS C** (hierarchy): $S_B^{5D} \in [150, 170]$ → $\alpha \approx 21$ for $\nu = 3$ ✓

### Stage 8b — Converged Bounce (dual-solver validation)
**File:** `instanton_bounce_converged.py`

Full convergence analysis of the 5D bounce instanton using **two independent solvers** — BVP collocation (`scipy.solve_bvp`) and Newton finite-difference relaxation with tridiagonal banded solver — plus adaptive domain growth, mesh refinement, and $\eta$-parameter scan.

**Four convergence tests:**

| Test | Method | Result |
|:--|:--|:--|
| A) Mesh convergence | $N = 500 \to 4000$, $\Delta S/S$ | $< 10^{-5}$ ✓ |
| B) Domain convergence | $\rho_{\max} = 200 \to 400$ | $< 0.1\%$ ✓ |
| C) Solver consistency | BVP vs Relaxation | $0.07\%$ ✓ |
| D) Tail stability | $|\delta\phi_{\text{end}}| \sim 10^{-16}$ | Machine precision ✓ |

**Converged result** at $\eta/\eta_{\text{crit}} = 0.9817$:

| Quantity | Value |
|:--|:--|
| $S_B^{4D}$ (BVP) | 207.30 |
| $S_B^{4D}$ (Relaxation) | 207.36 |
| $S_B^{5D}$ | **155.47** |
| Virial $S_{\text{kin}}/|S_{\text{pot}}|$ | 2.0000 (exact) |
| BVP–Relax agreement | $0.07\%$ |

Parameter scan confirms $S_B^{5D} \in [140, 200]$ for $\eta/\eta_c \in [0.975, 0.983]$.

**ALL PASS** — mesh ✓, domain ✓, solver consistency ✓, tail ✓, target range ✓

---

### Stage 9 — Microscopic Closure Derivation
**File:** `stage9_closure.py`

Derives — rather than assumes — the two closure relations of the Randall-Sundrum braneworld from the single 5D warped action with Goldberger-Wise (GW) stabilization:

$$S = \int d^5x\,\sqrt{-g}\left[\frac{M_5^3}{2}R - \frac{1}{2}(\partial\Phi)^2 - \frac{1}{2}m^2\Phi^2\right] - \sum_{i=0,L}\int d^4x\,\sqrt{-g_i}\;\lambda_i(\Phi - v_i)^2$$

The bulk scalar satisfies $\Phi'' - 4k\Phi' - m^2\Phi = 0$ with general solution $\Phi(y) = Ae^{\alpha_+ y} + Be^{\alpha_- y}$, where $\alpha_\pm = 2k \pm \nu$ and $\nu = \sqrt{4k^2 + m^2}$. Robin boundary conditions from brane-localized potentials determine $(A,B)$ via a $2\times 2$ linear system.

**Three modules:**

- **Module A** — Effective potential $V_\text{eff}(L)$ has a stable minimum at $L^* = \beta/m$. A 1000-point Latin hypercube scan over $(m/k, \lambda, v_0, v_L)$ yields $\beta_\text{median} = 1.14$ with 64.2% in $[0.3, 3.0]$, confirming $\beta = \mathcal{O}(1)$ without fine-tuning.

- **Module B** — The GW scalar backreacts on the warp function at $\mathcal{O}(\kappa^2)$:
  $$\sigma(y) = ky + \frac{\kappa^2}{12}\int_0^y (\Phi')^2\,dy'$$
  giving $\mathcal{O}(L) = e^{-\sigma(L)}$ with effective decay constant $c_\text{eff} = \sigma(L)/(kL) = 1 + \mathcal{O}(\kappa^2)$. Fitted value: $c = 0.999993 \pm 10^{-6}$ ($R^2 = 1.0$). A 200-point parameter scan confirms $c_\text{median} = 0.999998$.

- **Module C** — Since $M_\text{Pl}^2 = M_5^3/(2k)(1 - e^{-2kL}) \approx M_5^3/(2k)$ for $kL \gg 1$, Newton's constant is $L$-independent. Result: $\Delta G/G < 10^{-6}$.

**Summary of results:**

| Module | Relation | Key result | Status |
|:--|:--|:--|:--|
| A | $L^* = \beta/m$, $\beta = \mathcal{O}(1)$ | $\beta_\text{median} = 1.14$; 64.2% in $[0.3, 3.0]$ | **PASS** |
| B | $\mathcal{O}(L) = Ae^{-ckL}$, $c \approx 1$ | $c = 0.999993 \pm 10^{-6}$; $R^2 = 1.0$ | **PASS** |
| C | $\Delta G/G \approx 0$ | $\Delta G/G < 10^{-6}$ | **PASS** |
| T1–T3 | Numerical robustness | Mesh, solver, BC convergence | **PASS** |
| T4–T5 | Parameter robustness | Tail stability, 200-pt $c$ scan | **PASS** |

**PASS criteria:** $\beta \in [0.3, 3.0]$ for $> 50\%$ of scan, $|c - 1| < 10^{-3}$, $\Delta G/G < 10^{-3}$, all 5 validation tests pass.

**Verdict: STRONG MICROSCOPIC SUPPORT** — all closure relations derived from first principles (12/12 checks passed).

---

## Repository Structure

```
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_all.sh                   # One-command full validation
│
├── metric.py                    # 5D warped RS-II metric, Christoffels, Ricci
├── grid.py                      # Compactified Chebyshev-Lobatto coordinate grid
├── deturck.py                   # DeTurck gauge vector & Einstein-DeTurck residual
├── env_check.py                 # JAX/GPU environment verification
│
├── background_test.py           # Stage 1: background metric validation
├── stage2_source.py             # Stage 2: linearized graviton solver
├── stage3_zero_mode.py          # Stage 3: KK zero mode verification
├── stage4_kk_spectrum.py        # Stage 4: KK spectrum convergence
├── stage5_ghost_tachyon.py      # Stage 5: ghost/tachyon exclusion
├── stage6_time_evolution.py     # Stage 6: time evolution stability
├── stage7_ppn.py                # Stage 7: PPN relativistic check
│
├── run_validation.py            # Master runner for stages 3–7
├── run_stage7_fast.py           # Optimized stage 7 (builds FD matrix in NumPy)
│
├── instanton_bounce.py          # Stage 8: 5D Euclidean bounce instanton (BVP)
├── instanton_bounce_converged.py # Stage 8b: Converged bounce (BVP + Relaxation)
├── pde_derivation.py            # Independent PDE derivation & sign verification
│
├── stage9_closure.py            # Stage 9: Microscopic closure derivation (GW)
├── closure_derivation.py        # Stage 9 alias (same file)
└── results/                     # Output: plots, convergence tables, JSON report
```

## Requirements

- **Python** ≥ 3.10
- **JAX** ≥ 0.4.20 with CUDA support (GPU required for Stages 1–7)
- **NumPy**, **SciPy**, **Matplotlib** (standard scientific stack)
- NVIDIA GPU with ≥ 8 GB VRAM for Stages 1–7 (tested on RTX 5090)
- Stages 8–9 run on CPU only (SciPy BVP solver, no GPU needed)

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install JAX with CUDA support (adjust CUDA version as needed)
pip install -U "jax[cuda12]"

# Install remaining dependencies
pip install -r requirements.txt
```

## Running the Validation

### Quick Start — Full Pipeline

```bash
# Run all 7 stages sequentially
bash run_all.sh
```

### Individual Stages

```bash
# Stage 1: Background metric (standalone, ~10s)
python background_test.py

# Stages 3–7: Full KK validation suite (~30s total on GPU)
python run_validation.py

# Stage 7 only (optimized, ~3s)
python run_stage7_fast.py

# Stage 8: Euclidean bounce instanton (~3 min quick, ~17 min full)
python instanton_bounce.py --quick    # reduced parameter scan
python instanton_bounce.py            # full 10×14 scan
python instanton_bounce.py --lam 0.12 --eta-frac 0.976  # custom parameters

# Stage 8b: Converged bounce with dual-solver validation (~2 min)
python instanton_bounce_converged.py  # BVP + Relaxation, mesh/domain/tail tests

# Stage 9: Microscopic closure derivation (~13s, CPU only)
python stage9_closure.py              # GW stabilization → β, c, ΔG derivation
```

### Environment Check

```bash
# Verify JAX sees the GPU
python env_check.py
```

## Expected Output

A successful run produces:

```
======================================================================
  FINAL VALIDATION VERDICT: STRONG PASS
    Stage 3 (Zero mode)       : PASS  (m₀² = 1.51e-13)
    Stage 4 (KK spectrum)     : PASS  (convergence < 2%)
    Stage 5 (Ghost/tachyon)   : PASS  (ghost-free, tachyon-free)
    Stage 6 (Time evolution)  : PASS  (E rel_var = 0.33%)
    Stage 7 (PPN)             : PASS  (γ = 1.0, R² = 1.0)
======================================================================
```

Stage 8 (instanton bounce) produces:

```
  S_B^{5D} =         164.938042
  PASS A (regular bounce):  ✓
  PASS B (thin-wall):       ✓
  PASS C (hierarchy α≈21):  ✓
  Overall: ✓ ALL PASSED
```

and saves plots + JSON report to `results/`.

Stage 9 (closure derivation) produces:

```
  MODULE A  β median = 1.1361   (64.2% in [0.3, 3.0])   ✓
  MODULE B  c = 0.999993 ± 0.000000   R² = 1.000000     ✓
  MODULE C  ΔG/G = 0.000000                              ✓
  Validation: 5/5 tests PASS
  ══════════════════════════════════════════════════════
   VERDICT: STRONG MICROSCOPIC SUPPORT
  ══════════════════════════════════════════════════════
```

and saves `results/closure_report.json` and `results/closure_derivation.png`.

Each stage also generates `stage{N}_results.json` and `stage{N}_report.txt`.

## Key Technical Decisions

### Why Sturm-Liouville FE instead of Schrödinger Substitution?

The KK eigenvalue equation in the extra dimension can be cast either as:

1. **Schrödinger form:** $-\chi'' + V_\text{eff}\,\chi = m^2\chi$ with $\psi = e^{2ky}\chi$
2. **Sturm-Liouville form:** $-(e^{-4ky}\psi')' = m^2 e^{-4ky}\psi$

We use the S-L form because:
- The **Neumann BC** ($\psi'(0) = 0$ from Z₂ symmetry) is the **natural BC** of the variational formulation — no ghost points or special stencils needed
- The stiffness matrix $H$ is **symmetric by construction** (Galerkin property)
- The zero mode $\psi_0 = \text{const}$ satisfies $H\psi_0 = 0$ **exactly** at the discrete level

The Schrödinger approach requires Robin BCs ($\chi'(0) = -2k\chi(0)$) that break matrix symmetry and introduce O($\Delta y$) errors in the zero mode eigenvalue.

### Why Build the FD Matrix in NumPy for Stage 7?

The original `solve_linearized()` was decorated with `@jax.jit`, which forced JAX to trace through a Python double loop (120×50 = 6000 iterations of `.at[].add()`). JAX's XLA compiler would spend 20+ minutes building the computation graph for a 3-second actual solve.

The fix: build the matrix with plain NumPy, transfer to GPU, then call `jnp.linalg.solve()`. Total time: ~3 seconds.

## Citation

If you use this code in academic work, please cite:

```
Randall-Sundrum II Braneworld Validation Suite
Einstein-DeTurck formulation with JAX/CUDA
https://github.com/[your-username]/einstein-deturck-validation
```

## References

- L. Randall, R. Sundrum, "An Alternative to Compactification," *Phys. Rev. Lett.* **83** (1999) 4690
- M. Headrick, S. Kitchen, T. Wiseman, "A new approach to static numerical relativity and its application to Kaluza-Klein black holes," *Class. Quant. Grav.* **27** (2010) 035002
- P. Figueras, T. Wiseman, "On the existence of stationary Ricci solitons," *Class. Quant. Grav.* **34** (2017) 145007

## License

MIT License. See individual files for attribution.
