# Twin Barrier Theory of Gravity

**Author:** Mateja Radojicic
**Affiliation:** Independent Researcher, Belgrade, Serbia
**Release:** 1.0 — April 2026

---

## Twin-Barrier-Theory.pdf

The complete theoretical framework is contained in the file **Twin-Barrier-Theory.pdf** included in this repository. The PDF contains the full Twin Barrier Theory of Gravity — a 5D braneworld model that derives Newton's constant from Standard Model parameters with zero free parameters.

### Document Integrity — SHA-256 Hash

The following cryptographic hash serves as a timestamped proof of the document's contents at the time of publication:

```
SHA-256: 3916c4875d1302a2b3f7fed30abb9001aefd3ba63c9057380fe956a301d90d49
File:    Twin-Barrier-Theory.pdf
```

To verify: `sha256sum Twin-Barrier-Theory.pdf`

---

## Thirteen-Stage Computational Validation Suite

This repository provides a complete, self-contained validation pipeline for the 5D Einstein-DeTurck formulation of the Randall-Sundrum braneworld model.

- **Stages 1-7** (JAX, CPU/GPU): Graviton spectrum — background metric, linearized graviton, KK decomposition, spectrum convergence, ghost/tachyon exclusion, time evolution stability, PPN relativistic check.
- **Stage 8** (SciPy): 5D Euclidean bounce instanton for the twin-barrier scalar potential.
- **Stage 9** (SciPy): Microscopic derivation of RS closure relations from Goldberger-Wise stabilization.
- **Stage 10** (NumPy): QCD route to Newton's constant — derives the warp exponent from QCD dimensional transmutation.
- **Stage 11** (NumPy): Bootstrap proof that the bulk scalar mass equals the first QCD beta coefficient times the electroweak VEV.
- **Stage 12** (NumPy/SciPy): Coleman-Weinberg proof that the proportionality constant c = 1, closing the last hypothesis.
- **Stage 13** (NumPy/SciPy): NLO precision and error budget — maps the 1.88% G error to 0.05-sigma tension in the strong coupling constant.

---

## Validation Stages

### Stage 1 — Background Metric Verification (`stage1.py`)

Validates the analytical 5D warped background. Self-contained — all metric, grid, and DeTurck functions are inlined.
- DeTurck vector norm: |xi|^2 ~ 0 (gauge consistency)
- Einstein-DeTurck residual: max|E_AB| ~ 0 (field equations satisfied)
- Ricci scalar: R = -20k^2 (correct AdS5 curvature)

PASS criteria: all quantities at machine precision (< 10^-10).

### Stage 2 — Linearized Brane Graviton (`stage2.py`)

Solves the linearized graviton equation for a static point mass on the brane using 2D finite differences. Produces the brane potential V(r) = -Phi(r,0)/2 and verifies 1/r fall-off.

PASS criteria: solution bounded, attractive potential, 1/r behavior, BCs satisfied.

### Stage 3 — Kaluza-Klein Zero Mode (`stage3.py`)

Verifies that the graviton zero mode (m_0^2 = 0) exists and has the correct profile from the Sturm-Liouville eigenvalue problem in the extra dimension. Used by Stages 4-6.

PASS criteria: |m_0^2| < 10^-6, flat profile, stable under domain doubling.

### Stage 4 — KK Spectrum Convergence (`stage4.py`)

Computes the full KK mass spectrum at three resolutions (N, 2N, 4N) and checks convergence.

PASS criteria: all first 20 eigenvalues converge within 2%.

### Stage 5 — Ghost and Tachyon Exclusion (`stage5.py`)

Hard veto gate — checks that the kinetic matrix has no negative eigenvalues (no ghosts) and the mass spectrum has no tachyons.

PASS criteria: lambda_min(K) >= 0, m_min^2 >= -10^-6.

### Stage 6 — Newtonian Limit Recovery (`stage6.py`)

Evolves a Gaussian brane perturbation under the wave equation using symplectic leapfrog integration and verifies energy conservation and stability.

PASS criteria: energy variance < 1%, growth rate < 0.01, norm ratio < 5.

### Stage 7 — PPN Relativistic Check (`stage7.py`)

Verifies brane gravity recovers GR by extracting the PPN parameter gamma from the Stage 2 solution and checking light-bending consistency.

PASS criteria: |gamma - 1| < 10^-3, R^2 > 0.999, A < 0.

### Stage 8 — 5D Bounce Instanton (`stage8.py`)

Solves the 5D warped Euclidean bounce for the twin-barrier scalar potential using BVP collocation. Confirms S_B ~ 160 reproducing the hierarchy alpha ~ 21.

PASS criteria: finite action, virial ratio = 2.000, S_B in [150, 170].

### Stage 9 — Microscopic Derivation of RS Closure Relations (`stage9.py`)

Derives (rather than assumes) the two closure relations from the 5D warped action with Goldberger-Wise stabilization: L* = beta/m with beta = O(1), and c = 0.999993.

PASS criteria: beta in [0.3, 3.0] for > 50% of scan, |c - 1| < 10^-3, delta_G/G < 10^-3.

### Stage 10 — QCD Route to Newton's Constant (`stage10.py`)

Derives the warp exponent alpha from QCD dimensional transmutation using 1-loop running of alpha_s from M_Z through the top threshold. Predicts the baryon asymmetry eta_B to 0.32%.

PASS criteria: alpha agreement < 2%, G error < 1%, eta_B prediction within 1%.

### Stage 11 — Bootstrap Proof: m_Phi = b_0 v_EW (`stage11.py`)

Proves the bulk scalar mass relation m = b_0 * v_EW via self-consistency: only N_f = 6 (b_0 = 7) passes the sub-percent G accuracy test across 20 candidate mass hypotheses.

PASS criteria: unique N_f = 6 solution with G error < 1%.

### Stage 12 — Coleman-Weinberg Proof: c = 1 (`stage12.py`)

Closes the last remaining hypothesis by proving c = 1 via three independent routes: (1) RG fixed point, (2) 1-loop Coleman-Weinberg correction delta_c ~ 10^-4, (3) implied c from observational G within NLO uncertainty.

PASS criteria: all 8 checks pass, delta_c < 0.5%.

### Stage 13 — NLO Precision and Error Budget (`stage13.py`)

Quantifies the theoretical precision: the 1.88% G error maps to 0.05-sigma tension in alpha_s(M_Z) (287 ppm in alpha). Verifies QCD 3-loop/2-loop convergence ratio = 0.0018.

PASS criteria: all 8 checks pass, alpha precision < 300 ppm.

---

## Repository Structure

```
README.md                  # This file
Twin-Barrier-Theory.pdf    # Complete theory (PDF)
Twin-Barrier-Theory.md     # Complete theory (Markdown source)
run_all.py                 # One-command full validation (all 13 stages)
stage1.py                  # Stage 1: Background metric verification
stage2.py                  # Stage 2: Linearized brane graviton
stage3.py                  # Stage 3: KK zero mode verification
stage4.py                  # Stage 4: KK spectrum convergence
stage5.py                  # Stage 5: Ghost and tachyon exclusion
stage6.py                  # Stage 6: Newtonian limit recovery
stage7.py                  # Stage 7: PPN relativistic check
stage8.py                  # Stage 8: 5D bounce instanton
stage9.py                  # Stage 9: Microscopic closure derivation
stage10.py                 # Stage 10: QCD route to Newton's constant
stage11.py                 # Stage 11: Bootstrap proof m = b0 * vEW
stage12.py                 # Stage 12: Coleman-Weinberg proof c = 1
stage13.py                 # Stage 13: NLO precision and error budget
```

## Requirements

- **Python** >= 3.10
- **JAX** >= 0.4.20 (Stages 1-7; works on CPU, GPU optional)
- **NumPy**, **SciPy**, **Matplotlib** (standard scientific stack)
- Stages 8-13 run on CPU only (no GPU needed)

## Running the Validation

### Full Pipeline

```bash
# Run all 13 stages sequentially
python run_all.py
```

### Individual Stages

```bash
# Any single stage
python stage1.py
python stage2.py
# ...
python stage13.py

# Specific stages via run_all.py
python run_all.py 1 2 3
```

Output logs are saved to `results/stageN.log`.
