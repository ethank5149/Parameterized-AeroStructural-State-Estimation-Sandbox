![Paper Status](https://img.shields.io/badge/Paper-Drafting-blue?style=for-the-badge)
![Implementation](https://img.shields.io/badge/Code-Upcoming-lightgrey?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

# PASSES: Parameterized AeroStructural State-Estimation Sandbox

A unified, GPU-accelerated spectral framework for coupled aeroelastic, thermodynamic, and stochastic GNC flight simulation. PASSES is designed to analyze multi-body continuum mechanics, non-linear filter convergence, and optimal guidance laws for civilian launch vehicles, **Hypersonic Glide Vehicles (HGV)**, and **Fractional Orbital Bombardment Systems (FOBS)** through a global $C^\infty$ continuous state vector.

## Project Overview

PASSES is an integrated computational suite developed to model the full atmospheric, orbital, and atmospheric entry life cycle of multi-stage vehicles. To eliminate numerical friction and interpolation errors common in moving-mesh simulations, the framework unifies distinct domains of aerospace mathematics into a singular, fixed-grid software sandbox:

1. **6-DOF Multi-Physics (Bivariate Spectral Collocation):** Modeling of non-linear structural flexing via 2D tensor-product Chebyshev grids. Kinematics are tracked in 6-DOF using quaternions to avoid gimbal lock during extreme atmospheric maneuvering.
2. **Hypersonic Aerothermodynamics:** Analytical closure of aerodynamic loops using **Modified Newtonian Theory** ($C_p = C_{p,max} \sin^2 \delta_c$) and **Sutton-Graves Convective Heating** for stagnation point heat flux.
3. **Phase-Change Thermodynamics (Enthalpy Method):** Simulation of moving-boundary ablation physics without mesh regeneration, utilizing a smoothed Enthalpy Method for $C^\infty$ continuity.
4. **Resilient GNC & Predictor-Corrector Guidance:** Implementation of an **Innovation-Based Adaptive Estimation (IAE)** EKF for state estimation, coupled with a GPU-accelerated **Predictor-Corrector** guidance loop featuring **Exterior Penalty Functions** for autonomous thermal boundary management.
5. **Orbital Mechanics & "Physics Idle":** Integration of $J_2$-perturbed Keplerian orbital coast phases with a continuous transition from atmospheric flight to exo-atmospheric orbital mechanics within a single, unbroken integration run.

---

## Architectural Breakdown & Core Mathematics

### 1. 2D Spectral Plate Dynamics (Aeroelasticity)
For lifting bodies and waveriders, PASSES elevates structural dynamics from 1D beams to **2D Spectral Plates** using bivariate Chebyshev grids $(\xi, \eta) \in [-1, 1]$. Using **Kronecker tensor products ($\otimes$)**, the framework constructs global differentiation matrices that govern longitudinal bending and torsional twisting simultaneously:

$$\mathbf{K} = \mathbf{D}_x^2 \mathbf{D}_{11} \mathbf{D}_x^2 + 2 \mathbf{D}_{xy} \mathbf{D}_{66} \mathbf{D}_{xy} + \mathbf{D}_y^2 \mathbf{D}_{22} \mathbf{D}_y^2$$

Where $\mathbf{D}_{11}$, $\mathbf{D}_{22}$, and $\mathbf{D}_{66}$ represent the anisotropic flexural rigidity tensors (Knox-Xi tensors).

### 2. Hypersonic Forcing & Sutton-Graves Heating
Aerodynamic and thermal loads are calculated analytically at every spectral node:
*   **Pressure Distribution:** $C_p(\xi, \eta) = C_{p,max} \sin^2(\delta_c(\xi, \eta))$
*   **Thermal Flux:** $\dot{q}_s = K \sqrt{\rho_\infty / R_{eff}} \vert{}\mathbf{v}_\infty\vert{}^3$ (Sutton-Graves Stagnation Heating)

Local incidence angles $\delta_c$ are mapped to the deformed surface normal $\mathbf{n}(\xi, \eta, t)$ to capture the coupling between structural flexing and aerothermal loading.

### 3. Predictor-Corrector Guidance & Thermal Penalties
The HGV guidance suite utilizes a **Shadow Physics Predictor** (a 3-DOF version of the GPU-accelerated engine) to project the footprint forward. To enforce thermal limits ($\dot{q}_{max}$), an **Exterior Penalty Function** ($\Phi_{heat}$) is mapped into the downrange error space:

$$\tilde{R}_{pred}(\mathbf{u}) = R_{pred}(\mathbf{u}) - W_{heat} \Phi_{heat}(\mathbf{u})$$

This "virtual undershoot" tricks the Secant Corrector into aggressively reducing bank angles to steer the vehicle toward cooler, lower-density atmospheric regimes.

### 4. Orbital Coast & Reentry (The "Physics Idle")
During the transition to LEO (Karman line), the RKF45 solver detects the decay of dynamic pressure and automatically expands its integration step size ($\Delta t$). The framework propagates $J_2$-perturbed orbital mechanics until the de-orbit burn:

$$\mathbf{g}(\mathbf{r}) = -\frac{\mu}{\vert{}\mathbf{r}\vert{}^3} \mathbf{r} + \mathbf{g}_{J_2}(\mathbf{r})$$

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
