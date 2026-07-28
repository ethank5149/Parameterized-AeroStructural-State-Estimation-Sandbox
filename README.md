![Paper Status](https://img.shields.io/badge/Paper-Drafting-blue?style=for-the-badge)
![Implementation](https://img.shields.io/badge/Code-Upcoming-lightgrey?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

# PASSES: Parameterized AeroStructural State-Estimation Sandbox

A unified, GPU-accelerated spectral framework for coupled aeroelastic, thermodynamic, and stochastic GNC flight simulation. PASSES is designed to analyze multi-body continuum mechanics, non-linear filter convergence, and optimal guidance laws for civilian launch vehicles, **Hypersonic Glide Vehicles (HGV)**, and **Fractional Orbital Bombardment Systems (FOBS)** through a global $C^\infty$ continuous state vector.

## Project Overview

PASSES is an integrated computational suite developed to model the full atmospheric, orbital, and atmospheric entry life cycle of multi-stage vehicles. To eliminate numerical friction and interpolation errors common in moving-mesh simulations, the framework unifies distinct domains of aerospace mathematics into a singular, fixed-grid software sandbox:

1. **6-DOF Fixed-Grid Multi-Physics:** Modeling of non-linear structural flexing via **Chebyshev Spectral Collocation** for 1D beams and 2D plates. Kinematics are tracked in 6-DOF using quaternions to avoid gimbal lock during extreme atmospheric maneuvering.
2. **Hypersonic Aerothermodynamics:** Analytical closure of aerodynamic loops using **Modified Newtonian Theory** ($C_p \propto \sin^2 \delta_c$) and **Sutton-Graves Convective Heating** for stagnation point heat flux.
3. **Phase-Change Thermodynamics (Enthalpy Method):** Simulation of moving-boundary ablation physics without mesh regeneration, utilizing a smoothed Enthalpy Method for $C^\infty$ continuity.
4. **Resilient GNC & Predictor-Corrector Guidance:** Implementation of an **Innovation-Based Adaptive Estimation (IAE)** EKF for state estimation, coupled with a GPU-accelerated **Predictor-Corrector** guidance loop for energy management during long-duration hypersonic glides.
5. **Orbital Perturbations & FOBS:** Integration of high-fidelity planetary gravitational harmonics ($J_2$ through $J_4$) and non-inertial reference frame kinematics for partial-orbit trajectories.

---

## Architectural Breakdown & Core Mathematics

### 1. Expanded 6-DOF Global State Vector
To support waveriders and skipping trajectories, the global state vector $\mathbf{X}_{global}$ incorporates full rigid-body kinematics alongside spectral structural modes and nodal enthalpies:
$$\mathbf{X}_{global} = [\mathbf{r}_E, \mathbf{v}_E, \mathbf{q}, \boldsymbol{\omega}_B, m_{bulk}, \mathbf{w}, \mathbf{\dot{w}}, \mathbf{H}]^T$$
Where $\mathbf{q}$ is the attitude quaternion and $\boldsymbol{\omega}_B$ is the body-frame angular velocity.

### 2. Spectral Plate Expansion (2D Aeroelasticity)
For lifting bodies and HGVs, PASSES elevates structural dynamics from 1D beams to **2D Spectral Plates** (Kirchhoff-Love theory) using bivariate Chebyshev grids ($\xi, \eta \in [-1, 1]$). This captures complex torsional modes and asymmetric flutter induced by bank-angle modulation:
$$\mathbf{M} \mathbf{\ddot{w}} + \mathbf{K} \mathbf{w} = \mathbf{Q}_{aero} + \mathbf{Q}_{thrust}$$

### 3. Hypersonic Forcing & Sutton-Graves Heating
Aerodynamic and thermal loads are calculated analytically at every spectral node, ensuring global differentiability:
*   **Pressure Distribution:** $C_p = C_{p,max} \sin^2 \delta_c$ (Modified Newtonian Flow)
*   **Thermal Flux:** $\dot{q}_{conv} = k \sqrt{\rho} V_r^3$ (Sutton-Graves Stagnation Heating)

### 4. GNC: Adaptive EKF & Predictor-Corrector Guidance
The GNC architecture combines high-frequency state estimation with real-time trajectory optimization:
*   **IAE-EKF:** Monitors the **Squared Mahalanobis Distance** ($\gamma_k$) to detect separation shocks or plume impingement, dynamically scaling process noise $\mathbf{Q}_k$ via trace-ratio calculations.
*   **Shadow Physics Predictor:** For HGV glide phases, the flight computer calls a "shadow" version of the GPU-accelerated PASSES engine to project the vehicle footprint forward and iteratively adjust bank-angle commands for optimal energy management.

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
