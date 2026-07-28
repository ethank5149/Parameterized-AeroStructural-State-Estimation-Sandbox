![Paper Status](https://img.shields.io/badge/Paper-Drafting-blue?style=for-the-badge)
![Implementation](https://img.shields.io/badge/Code-Upcoming-lightgrey?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

# PASSES: Parameterized AeroStructural State-Estimation Sandbox

A unified, GPU-accelerated spectral framework for coupled aeroelastic, thermodynamic, and stochastic GNC flight simulation. PASSES is designed to analyze multi-body continuum mechanics, non-linear filter convergence, and optimal guidance laws for civilian launch vehicles through a global $C^\infty$ continuous state vector.

## Project Overview

PASSES is an integrated computational suite developed to model the full atmospheric, orbital, and atmospheric entry life cycle of multi-stage vehicles. To eliminate numerical friction and interpolation errors common in moving-mesh simulations, the framework unifies five distinct domains of aerospace mathematics into a singular, fixed-grid software sandbox:

1. **Fixed-Grid Multi-Physics (Spectral Collocation):** Modeling of non-linear structural flexing via Chebyshev Spectral Collocation and fluid-structure interactions (propellant slosh) projected onto spectral grids.
2. **Phase-Change Thermodynamics (Enthalpy Method):** Simulation of moving-boundary ablation physics without mesh regeneration, utilizing a smoothed Enthalpy Method for $C^\infty$ continuity.
3. **Multi-Body Separation & Plume Dynamics:** Modeling of discrete-event staging discontinuities, plume impingement reflections (Roberts continuum model), and the "tail-wags-dog" (TWD) inertial reaction effect.
4. **Resilient GNC (IAE-EKF):** Implementation of an Innovation-Based Adaptive Estimation (IAE) Extended Kalman Filter that uses Mahalanobis distance thresholds to scale process noise during catastrophic shocks.
5. **Terminal Guidance (AC-APN):** Implementation of Aerodynamically-Compensated Augmented Proportional Navigation (AC-APN) featuring quadratic time-to-go ($t_{go}$) prediction for non-linear atmospheric deceleration.

---

## Architectural Breakdown & Core Mathematics

### 1. Aeroelasticity & Spectral Collocation
The flexible fuselage is modeled as a free-free continuous beam. Instead of discrete Finite Element Methods (FEM), PASSES utilizes **Chebyshev Spectral Collocation** to map the Euler-Bernoulli PDE into a discrete system of ODEs:

$$\mathbf{M} \mathbf{\ddot{w}} + \mathbf{K} \mathbf{w} = \mathbf{Q}$$

Where $\mathbf{K} = \mathbf{D}^2 \text{diag}(EI(x)) \mathbf{D}^2$ and $\mathbf{D}$ is the global spectral differentiation matrix. This preserves $C^\infty$ spatial continuity, allowing for quasi-static RHS matrices that are ideal for GPU offloading.

### 2. Thermodynamics & The Enthalpy Method
Ablation is treated as a classic Stefan problem but solved on a fixed grid to avoid front-tracking complexities. The framework introduces a smoothed ablation fraction $\phi(T)$ to absorb phase-change physics into a singular volumetric enthalpy state $H$:

$$\frac{\partial H}{\partial t} = \frac{\partial}{\partial x} \left( k(T) \frac{\partial T}{\partial x} \right), \quad \phi(T) = \frac{1}{2} \left[ 1 + \tanh\left(\frac{T - T_m}{\Delta T}\right) \right]$$

### 3. Innovation-Based Adaptive Estimation (IAE)
To prevent filter divergence during stage separation shocks or plume impingement, the GNC core monitors the **Squared Mahalanobis Distance** ($\gamma_k$) of the EKF innovation sequence:

$$\gamma_k = \boldsymbol{\nu}_k^T \mathbf{S}_k^{-1} \boldsymbol{\nu}_k$$

If $\gamma_k$ breaches a Chi-square threshold, the filter dynamically scales the process noise matrix $\mathbf{Q}_k$ using a moving-window trace-ratio calculation, allowing the state estimate to absorb the transient without losing track of the vehicle.

### 4. Terminal Guidance (AC-APN)
For terminal recovery, the guidance suite utilizes **Aerodynamically-Compensated APN** with a **Quadratic Time-to-Go** prediction to account for non-linear atmospheric drag:

$$\frac{1}{2}\hat{A}_c t_{go}^2 + \hat{V}_c t_{go} - \Vert{}\mathbf{r}_{LOS}\Vert{} = 0$$

$$\mathbf{a}_n = N' \cdot \mathbf{V}_r \times \boldsymbol{\dot{\lambda}}_{LOS} + \text{Gravity Compensation}$$

---

## Geometric Transcription (CAD to $C^\infty$)
PASSES includes a pre-processing pipeline that transcribes discrete CAD meshes (STL/OBJ) into continuous analytical domains.
* **Chebyshev Projection:** Discrete geometric slices are projected onto truncated Chebyshev polynomials to filter out $C^0$ faceting.
* **Hyperbolic Blending:** Multi-material interfaces (e.g., carbon-to-titanium joints) are smoothed via hyperbolic tangent functions to maintain differentiability at structural boundaries.

### 5. Statistical Error Analysis (Circular Error Probable)
To measure the end-to-end performance and accuracy of the coupled GNC framework under aerodynamic disturbances and sensor anomalies, a Monte Carlo simulation engine executes batch runs. The terminal touchdown coordinates are passed to a bivariate normal spatial distribution processor to calculate the **Circular Error Probable (CEP)** at a 50% confidence radius:

$$CEP \approx 0.562\sigma_x + 1.177\sigma_y \quad (\text{for non-symmetric down-range/cross-range variances})$$

---

## Proposed Directory Structure

```text
├── PASSES/
│   ├── src/
│   │   ├── physics_engine/    # Multi-body dynamics, J2 gravity, slosh pendulums, ablation PDEs
│   │   ├── sensor_models/     # IMU synthesis, white noise injection, random-walk bias drift
│   │   ├── gnc_core/          # Adaptive EKF, measurement Jacobians, TVC notch filters
│   │   ├── guidance_laws/     # Proportional Navigation (PN) loops, optimal control solvers
│   │   └── analytical_tools/  # Monte Carlo batch runners, bivariate normal CEP processors
│   ├── data/
│   │   └── parameters/        # TEXTBOOK CONSTANTS ONLY (Normalized civilian rocket profiles)
│   ├── tests/                 # Unit tests for RKF45 integration stability and filter convergence
│   └── README.md              # Documentation
```

## Running the Simulation Sandbox

### Prerequisites
* Python 3.10+ or C++17 Compiler
* NumPy, SciPy, Matplotlib

### Planned Execution
To execute the holistically coupled flight profile, initialize the terminal guidance loop, and output the statistical CEP matrix, run:
```bash
python src/main.py --config config/profile.json --monte-carlo --runs 100
```

---

## Compliance and Academic Disclaimers
* **Fundamental Research Exemption:** This software suite is built strictly for academic, public-domain research purposes under the **Fundamental Research Exception (22 CFR 125.4)**. 
* **Data Sanitization:** Absolutely zero proprietary, classified, or export-controlled (ITAR/EAR) military hardware specifications are contained within this repository.
* **Intended Use:** This framework is designed to evaluate the mathematical coupling of structural mechanics and statistical estimation theory.
