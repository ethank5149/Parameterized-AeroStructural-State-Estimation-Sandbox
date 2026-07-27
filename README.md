![Paper Status](https://img.shields.io/badge/Paper-Drafting-blue?style=for-the-badge)
![Implementation](https://img.shields.io/badge/Code-Upcoming-lightgrey?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

# PASSES: Parameterized AeroStructural State-Estimation Sandbox

A holistically coupled, end-to-end multidisciplinary flight-dynamics simulation, navigation, and closed-loop control framework designed to analyze multi-body continuum mechanics, non-linear filter convergence, and optimal guidance laws for civilian launch vehicles.

## Project Overview

PASSES is an integrated computational suite developed to model the full atmospheric, orbital, and atmospheric entry life cycle of multi-stage vehicles within non-inertial reference frames. Written to bridge the gap between high-fidelity structural physics and advanced statistical estimation theory, the framework unifies five distinct domains of aerospace mathematics into a singular, parameterized software sandbox:

1. **Coupled Continuum Mechanics:** Modeling of non-linear structural flexing (Euler-Bernoulli beam theory), fluid-structure interactions (liquid propellant sloshing pendulums), and moving-boundary aerothermodynamics (Stefan-problem ablation physics).
2. **Advanced Multi-Body Kinematics:** Simulation of discrete-event staging discontinuities, engine inertial reaction forces (the tail-wags-dog effect), and non-spherical planetary gravitational harmonics (J₂ anomalies).
3. **Inertial Measurement Unit (IMU) Realism & Sensor Degradation:** Synthesizing of raw IMU telemetry injected with high-frequency white noise jitter, scale factor errors, and time-correlated random-walk sensor drift biases.
4. **Non-Linear Guidance, Navigation, and Control (GNC):** Implementation of a dual-rate adaptive Extended Kalman Filter (EKF) for sensor fusion, coupled to automated Thrust Vector Control (TVC) actuator models with hard rate saturation boundaries.
5. **Terminal Guidance & Statistical Error Assessment:** Implementation of optimal closed-loop Proportional Navigation (PN) terminal trajectories paired with bivariate normal stochastic processors to calculate Circular Error Probable (CEP).

---

## Architectural Breakdown & Core Mathematics

### 1. Non-Inertial Reference Frame Kinematics & Gravity
To preserve tracking accuracy relative to a rotating planet, translational vehicle acceleration is mapped within the rotating Earth-Centered, Earth-Fixed (ECEF) frame, fully accounting for Coriolis and centrifugal accelerations relative to the Earth-Centered Inertial (ECI) frame:

$$\mathbf{\ddot{x}}_E = \mathbf{\ddot{x}}_I - 2(\boldsymbol{\Omega}_E \times \mathbf{\dot{x}}_E) - \boldsymbol{\Omega}_E \times (\boldsymbol{\Omega}_E \times \mathbf{x}_E)$$

True forces are integrated continuously in the inertial frame utilizing an adaptive Runge-Kutta-Fehlberg 4th and 5th order method (RKF45) scheme, incorporating planetary equatorial bulge via the gradient of the J₂ gravitational potential harmonic expansion:

$$\mathcal{V}_{J2}(\mathbf{x}_I) = -\frac{\mu_E}{\Vert{}\mathbf{x}_I\Vert{}} \left[ 1 - \frac{J_2}{2} \left( \frac{R_E}{\Vert{}\mathbf{x}_I\Vert{}} \right)^2 \left( 3\left(\frac{z_I}{\Vert{}\mathbf{x}_I\Vert{}}\right)^2 - 1 \right) \right]$$

### 2. Aeroelasticity & Tail-Wags-Dog (TWD) Actuation
The flexible fuselage is modeled as a free-free continuous beam using Euler-Bernoulli theory. Rapidly pivoting a heavy engine bell creates a dramatic lateral inertial reaction force (\(F_{TWD}\)) at the gimbal bearing that feeds back into the aft structure. This creates a non-linear feedback loop capable of exciting destructive resonance modes during rapid transonic Mach transitions:

$$F_{TWD} = -m_e x_g \ddot{\delta} + m_e \dot{x}_g \dot{\delta}$$

$$\frac{\partial^2}{\partial x^2}\left( EI(x)\frac{\partial^2 y_f}{\partial x^2} \right) + \mu(x)\frac{\partial^2 y_f}{\partial t^2} = f_{aero}(x, t, M) + \left[F_{thrust}\sin(\delta) + F_{TWD}\right]\delta_D(x - x_g)$$

Digital notch filters are hardcoded into the TVC actuator controller to attenuate commands at the fuselage's natural structural frequencies, dampening cross-talk between the steering mechanism and the elastic body.

### 3. IMU Modeling & Adaptive EKF State Estimation
The navigation core receives data from a simulated IMU. The software generates true kinematic acceleration and angular velocity, then corrupts them using a time-correlated stochastic differential equation tracking accelerometer and gyroscope random-walk bias drift ($\mathbf{b}_{IMU}$):

$$\dot{\mathbf{b}}_{IMU}(t) = \mathbf{w}_{bias}(t), \quad E[\mathbf{w}_{bias}(t)\mathbf{w}_{bias}^T(\tau)] = \mathbf{Q}_{bias}\delta_D(t-\tau)$$

The dual-rate Extended Kalman Filter (EKF) recalculates the state Jacobian ($\mathbf{F}_k$) and measurement Jacobian ($\mathbf{H}_k$) to isolate actual vehicle motion from high-frequency structural noise:

$$\mathbf{K}_k = \mathbf{P}_{k\vert{}k-1}\mathbf{H}_k^T \left( \mathbf{H}_k \mathbf{P}_{k\vert{}k-1}\mathbf{H}_k^T + \mathbf{R}_k \right)^{-1}$$

$$\hat{\mathbf{x}}_{k\vert{}k} = \hat{\mathbf{x}}_{k\vert{}k-1} + \mathbf{K}_k \left( \mathbf{z}_k - \mathbf{h}(\hat{\mathbf{x}}_{k\vert{}k-1}) \right)$$

During stage separation discontinuities, a discrete-event handler automatically scales up the process noise covariance matrix (\(\mathbf{Q}_k\)) to prevent filter divergence caused by staging shocks.

### 4. Terminal Closed-Loop Guidance (Proportional Navigation)
Upon atmospheric entry or recovery phase initialization, the guidance suite transitions to a closed-loop terminal steering routine. The framework utilizes **Proportional Navigation (PN)**, driving lateral acceleration commands ($\mathbf{a}_n$) proportionally to the rotation rate of the Line-of-Sight vector ($\boldsymbol{\lambda}_{LOS}$) between the vehicle and the destination coordinates:

$$\mathbf{a}_n = N \cdot \mathbf{V}_r \times \boldsymbol{\dot{\lambda}}_{LOS}$$

Where N is the navigation constant (tuned between 3.0 and 5.0) and $\mathbf{V}_r$ is the relative velocity vector. Commands are continuously checked against TVC actuator saturation thresholds ($\delta_{max}$, $\dot{\delta}_{max}$).

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
