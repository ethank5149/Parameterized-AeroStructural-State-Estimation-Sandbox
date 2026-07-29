![Paper Status](https://img.shields.io/badge/Paper-Drafting-blue?style=for-the-badge)
![Implementation](https://img.shields.io/badge/Code-Upcoming-lightgrey?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

# PASSES: Parameterized AeroStructural State-Estimation Sandbox

A unified, GPU-accelerated spectral framework for coupled aeroelastic, thermodynamic, and stochastic GNC flight simulation. PASSES is designed to analyze multi-body continuum mechanics, non-linear filter convergence, and optimal guidance laws for civilian launch vehicles, **Hypersonic Glide Vehicles (HGV)**, and **Fractional Orbital Bombardment Systems (FOBS)** through a global $C^\infty$ continuous state vector.

## Project Overview

PASSES is an integrated computational suite developed to model the full atmospheric, orbital, and atmospheric entry life cycle of multi-stage vehicles. To eliminate numerical friction and interpolation errors common in moving-mesh simulations, the framework unifies distinct domains of aerospace mathematics into a singular, fixed-grid software sandbox:

1. **6-DOF Multi-Physics (Mindlin-Reissner Spectral Plates):** Modeling of non-linear structural flexing via bivariate Chebyshev grids. It accounts for transverse shear deformation and anisotropic flexural rigidity, tracking kinematics in 6-DOF using quaternions.
2. **High-Fidelity Hypersonic Aerothermodynamics:** Analytical closure of real-gas thermochemistry using **Fay-Riddell** (convective heating with surface catalysis) and **Tauber-Sutton** (volumetric radiative heating) formulations.
3. **Successive Convexification (SOCP) Guidance:** Autonomous trajectory optimization using Second-Order Cone Programming to enforce hard thermal ($\dot{q}_{max}$) and structural ($q_{max}$) constraints during aggressive maneuvering.
4. **Ionization-Aware GNC:** Modeling of **Plasma Sheath Blackout** via the Saha equation, coupled to an adaptive EKF that dynamically gates GNSS measurements during high-velocity atmospheric entry.
5. **Orbital Mechanics & "Physics Idle":** Integration of $J_2$-perturbed Keplerian orbital coast phases with a continuous transition from atmospheric flight to exo-atmospheric orbital mechanics within a single, unbroken integration run.

---

## Architectural Breakdown & Core Mathematics

### 1. Mindlin-Reissner Spectral Plate Dynamics
For lifting bodies and waveriders, PASSES utilizes **Mindlin-Reissner Anisotropic Plate Theory** to capture transverse shear deformations and torsional flutter. The structural state vector $\mathbf{U}$ expands to include transverse deflection ($w$) and independent normal rotations ($\phi_x, \phi_y$). Using **Kronecker tensor products ($\otimes$)**, the framework constructs global block-sparse differentiation matrices:

$$\mathbf{K} = \begin{bmatrix} \mathbf{K}_{ww} & \mathbf{K}_{w\phi_x} & \mathbf{K}_{w\phi_y} \\ \mathbf{K}_{\phi_x w} & \mathbf{K}_{\phi_x \phi_x} & \mathbf{K}_{\phi_x \phi_y} \\ \mathbf{K}_{\phi_y w} & \mathbf{K}_{\phi_y \phi_x} & \mathbf{K}_{\phi_y \phi_y} \end{bmatrix}$$

### 2. Fay-Riddell & Tauber-Sutton Heating
Aerodynamic and thermal loads incorporate real-gas effects and radiative transfer:
*   **Convective Flux:** $\dot{q}_{s, conv} \propto (h_{0e} - h_w) [ 1 + (Le^\beta - 1)\frac{h_D}{h_{0e}} ]$ (Fay-Riddell with surface catalysis)
*   **Radiative Flux:** $\dot{q}_{s, rad} \propto V_\infty^n R_{eff}^a$ (Tauber-Sutton, scaling up to $V^{10}$)

These high-fidelity correlations ensure that molecular dissociation and volumetric heating are evaluated dynamically across the bivariate spectral grid.

### 3. Successive Convexification (SOCP) Guidance
To guarantee constraint satisfaction, the guidance suite utilizes **Successive Convexification**. The non-linear OCP is linearized and solved as a series of **Second-Order Cone Programs (SOCP)**, allowing the vehicle to ride the edge of the thermal boundary:

$$\min_{\mathbf{u}} J \quad \text{s.t.} \quad \mathbf{x}_{i+1} = \mathbf{A}_i \mathbf{x}_i + \mathbf{B}_i \mathbf{u}_i + \mathbf{z}_i, \quad g(\mathbf{x}, \mathbf{u}) \le 0$$

### 4. Plasma Blackout & Ionization-Aware EKF
PASSES models the plasma sheath electron density ($n_e$) and critical frequency ($\omega_p$). The **Ionization-Aware EKF** dynamically zeroes out the measurement Jacobian $\mathbf{H}_k$ for external GNSS sensors when $\omega_p \ge \omega_{GNSS}$:

$$\omega_p = \sqrt{\frac{n_e e^2}{m_e \epsilon_0}}, \quad \mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}_k^T (\mathbf{H}_k \mathbf{P}_k^- \mathbf{H}_k^T + \mathbf{R}_k)^{-1}$$

---

## Geometric Transcription (CAD to $C^\infty$)
PASSES includes a pre-processing pipeline that transcribes discrete CAD meshes (STL/OBJ) into continuous analytical domains.
* **Chebyshev Projection:** Discrete geometric slices are projected onto truncated Chebyshev polynomials to filter out $C^0$ faceting.
* **Hyperbolic Blending:** Multi-material interfaces (e.g., carbon-to-titanium joints) are smoothed via hyperbolic tangent functions to maintain differentiability at structural boundaries.

### 5. Batched-Tensor CUDA Monte Carlo Validation
To measure end-to-end performance under stochastic disturbances, PASSES leverages a **Batched-Tensor CUDA Architecture**. This allows the parallel execution of 10,000+ trajectories as a singular rank-3 tensor operation, culminating in the extraction of the terminal covariance matrix $\boldsymbol{\Sigma}_{impact}$ and Circular Error Probable (CEP).

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
