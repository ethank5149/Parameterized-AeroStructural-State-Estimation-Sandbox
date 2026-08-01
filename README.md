![Status](https://img.shields.io/badge/status-formulation%20complete-blue?style=flat-square)
![Implementation](https://img.shields.io/badge/implementation-roadmap%20complete-brightgreen?style=flat-square)
![Verification](https://img.shields.io/badge/verification-14%2F16%20tasks%20%2B%202%20partial-brightgreen?style=flat-square)
![Papers](https://img.shields.io/badge/papers-2%20preprints-informational?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

# PASSES — Parameterized AeroStructural State-Estimation Sandbox

A fixed-grid spectral formulation for coupled aeroelastic, thermal, and stochastic GNC flight simulation.

**This repository contains two research manuscripts, their bibliographies, and a complete implementation of the roadmap — every kernel of both papers, assembled into a coupled single-trajectory simulator. Fourteen of the sixteen verification tasks are executed and passing; the remaining two (I-V4, II-V8) pass every leg executable from this repository, with only their published-reference-data comparisons outstanding.** The papers present a mathematical formulation and a falsifiable verification plan written *before* any implementation existed. **Both manuscripts have since been updated in place**: measured values are quoted inline where a claim has been verified, and where implementation showed a claim to be wrong it is corrected with the correction identified as such — most substantially the Tauber–Sutton exponent in Paper II §4.2 and the rigid/elastic split of the structural coordinate in Paper I §3.5. Measured results for every executed task live in [`results/`](results/). Please read the [status table](#status) before drawing conclusions about what works.

---

## The papers

| | Manuscript | Source | PDF |
|---|---|---|---|
| **I** | PASSES: A Unified, GPU-Accelerated Spectral Framework for Coupled Aeroelastic, Thermodynamic, and Stochastic GNC Flight Simulation | [`passes-updated.tex`](passes-updated.tex) | [`passes-updated.pdf`](passes-updated.pdf) |
| **II** | PASSES-HGV: Extending a Fixed-Grid Spectral Framework to 6-DOF Hypersonic Glide and Fractional Orbital Trajectories | [`passes-hgv-updated.tex`](passes-hgv-updated.tex) | [`passes-hgv-updated.pdf`](passes-hgv-updated.pdf) |

Paper II depends on Paper I for the fixed-grid rationale, the Landau transformation used for ablation, the adaptive filtering architecture, and the Monte Carlo dispersion estimators. Read them in order.

Every bibliography entry in both papers has been verified against publisher records. The audit — including nine entries that were wrong or unverifiable in earlier drafts — is in [`CITATION-AUDIT.md`](CITATION-AUDIT.md). Working from the archived sources rather than from recollection also caught a substantive error in Paper II's statement of the Tauber–Sutton correlation; see the [findings](#status).

---

## The idea in one section

Coupled multi-physics flight simulation conventionally partitions the problem: a structural solver, a thermal solver, and a trajectory propagator, each on its own mesh, exchanging boundary data. This costs you twice.

**Interpolation error at every interface.** Surface pressure, heat flux, and displacement must be interpolated between meshes. The interpolated field is generally no smoother than $C^0$ across element boundaries, and the aerothermal closures take several derivatives of it.

**Remeshing.** Ablative recession and large deformation are handled by moving-boundary formulations with periodic mesh regeneration. That is expensive, hard to vectorize, and introduces a projection step between pre- and post-remesh states. Tolerable for one deterministic trajectory; prohibitive for the $10^3$–$10^4$ replicates needed to resolve a dispersion footprint.

PASSES discretizes every spatial domain **once**, on a fixed grid, and holds it fixed for the entire trajectory. Three formulations make that possible:

- **Structural.** Free-free boundary conditions are imposed by projecting onto the null space of the constraint operator rather than by replacing rows. Row replacement destroys operator symmetry and produces spurious growing modes; projection does not.
- **Thermal.** A Landau coordinate transformation maps the receding physical domain onto a fixed computational domain. The ablation front is not eliminated from the physics — it is rendered *stationary in computational coordinates*, so no node is ever created, destroyed, or interpolated. The cost is one explicit advection term.
- **Coupling.** Localized slosh forces are regularized by a Gaussian kernel normalized against the collocation quadrature rule, so total force transfers exactly rather than to within quadrature error.

The result is one system of ODEs with fixed dimension and fixed sparsity. **That is the whole computational argument:** a Monte Carlo batch becomes a rank-3 tensor operation (replicate × state × stage) with no per-replicate remesh, so the batch never decoheres. A moving-mesh formulation cannot offer this, because each replicate's mesh diverges from every other's after its first remesh event.

---

## Status

Distinguishing what is derived from what is measured, because the distinction matters.

| Component | Formulated | Implemented | Verified |
|---|:---:|:---:|:---:|
| Chebyshev collocation, variable-$EI$ beam | ✅ | ✅ | ✅ V1 |
| Null-space free-free boundary projection | ✅ | ✅ | ✅ V1 |
| Temporal integration strategies (explicit / modal / IMEX) | ✅ | ✅ | ✅ V3 |
| Quadrature-normalized slosh regularization | ✅ | ✅ | ✅ V2 |
| Two-phase charring ablation (CMA-style) | ✅ | ✅ | ◐ V4 ‡ |
| Landau transformation to fixed thermal grid | ✅ | ✅ | ◐ V4 ‡ |
| Mahalanobis $\chi^2$ anomaly detection | ✅ | ✅ | ✅ V5 |
| Innovation-Based Adaptive Estimation | ✅ | ✅ | ✅ V5 |
| AC-APN terminal guidance | ✅ | ✅ | ✅ V6 † |
| CEP / $R_{95}$ dispersion estimators | ✅ | ✅ | ✅ V7 |
| Ultraspherical spectral discretization | ✅ | ✅ | ✅ II-V1 |
| Mindlin–Reissner anisotropic plates | ✅ | ✅ | ✅ II-V1/V2/V3 § |
| Fay–Riddell / Lees / Tauber–Sutton heating | ✅ | ✅ | ◐ II-V8 ‖ |
| 6-DOF quaternion state, deformed-surface incidence | ✅ | ✅ | ✅ II-V4 |
| Blended Newtonian / Prandtl–Meyer closure | ✅ | ✅ | ✅ II-V4 |
| Successive convexification guidance | ✅ | ✅ | ✅ II-V7 |
| Plasma blackout gating | ✅ | ✅ | ✅ II-V6 |
| $J_2$-perturbed orbital propagation | ✅ | ✅ | ✅ II-V5 |
| CUDA batched Monte Carlo | ✅ | ✅ | ✅ V8 ¶ |

† V6 verifies the $t_{go}$ precision claim and the non-intercept guard; the command law (Eq. 4.18) is implemented and unit-tested but has no closed-loop verification task until the batch layer exists.

‡ V4 is **partially complete**: the method-of-manufactured-solutions leg passes (spectral convergence of the coupled energy/kinetics/gas-flux system on the Landau grid, error contracting $5\times 10^{8}$-fold from $N_T=6$ to $20$), but the stated failure criterion — recession within 5% of a FIAT reference case — requires the external FIAT code or its published reference data, neither of which is in this repository. It is not counted as finished.

¶ V8's failure criterion (sublinear throughput scaling below device saturation) is evaluated on real CUDA hardware (RTX 3090, CuPy backend) and passes. **Theoretical** occupancy is computed exactly from the compiled kernel's register and shared-memory footprint against the device's per-SM limits. The **achieved** counter is wired to Nsight Compute (now installed) but blocked at the driver: `ERR_NVGPUCTRPERM`, because the NVIDIA module restricts performance counters to administrators. Enabling it means loading the module with `NVreg_RestrictProfilingToAdminUsers=0` — a host-wide security setting, so it is reported rather than changed. The code path runs the moment the permission is granted.

§ II-V3's stated reference is *published Rayleigh–Ritz values* for the free-free plate. This repository cannot audit those figures against a publisher record, so the frequency criterion is instead evaluated against the **closed-form Mindlin solution for the simply-supported plate**, derived in the code rather than transcribed and exercising the identical operator, boundary machinery, projection and eigensolve. The free-free comparison against the commonly circulated values is reported alongside, clearly labelled, and carries no verdict.

‖ II-V8 is **partially complete**. The implementation leg passes, and the **radiative half is now verified against the published source**: Tauber & Sutton's Table 1 and Eqs. (1)–(4) are transcribed from the PDF in [`reference/`](reference/), reproduced exactly at every tabulated node, and the blunting trade is demonstrated on published data rather than a surrogate. The convective **coefficients** are now checked against the Fay & Riddell original in [`reference/`](reference/): the Lewis exponents are confirmed (0.52 equilibrium, Eq. 60/62; 0.63 frozen/catalytic, Eq. 65 and Fig. 4), and the exponent split is *derived* from Eqs. (45) and (58)/(62) rather than assumed, and the leading constant turns out to be less settled than its usual three-figure presentation suggests — see the findings. What remains is the convective **reference case**: the 5% criterion is stated against published Fay–Riddell reference conditions, which need transcribed equilibrium-air properties ($\rho_e\mu_e$, $\rho_w\mu_w$, $h_D$) the repository does not carry, and the source presents its own results in figures rather than tables. The FIAT recession leg shares I-V4's pending status.

**Verification tasks: 14 of 16 complete, plus 2 partial** (Paper I V1, V2, V3, V5, V6, V7, V8 and Paper II II-V1 through II-V7 complete; I-V4 and II-V8 partial — reports with measured numbers in [`results/`](results/)). V1–V8 are tabulated in §8 of Paper I and §8 of Paper II, each with a stated reference and an explicit failure criterion; Paper II's tasks carry the **II-** prefix throughout this repository to keep the two numbering schemes distinct. They were written before any results existed, so the plan is falsifiable in advance rather than reportable selectively afterward. Measured findings worth flagging against the papers' expectations:

- **Conditioning (V1).** The raw $\kappa_2(\hat{\mathbf{K}})$ is pinned at $\sim 1/\varepsilon$ at every $N$, because the free-free operator retains its two *physical* rigid-body null directions. The informative measurand is the elastic condition number $\sigma_1/\sigma_{n-2}$, whose fitted slope is $\approx N^{8.0}$ for both uniform and stepped profiles — the projection removes the constraint-violating extremal modes (and passes the $10^{-6}$ frequency criterion at $N=32$ with $10^{-9}$ to spare) but does not flatten the asymptotic growth rate, which Remark 3 of Paper I deliberately declined to predict.
- **Rigid-mode floor.** The two rigid eigenvalues compute to $\sim 10^{-15}$ relative to $\lambda_{\max}$, occasionally negative. Any long-horizon integration of the *unmodified* reduced operator therefore drifts at rate $\sqrt{|\lambda_{\text{rigid}}|}$ regardless of integrator; modal truncation or rigid-mode deflation handles it, and the V3 comparison accounts for it explicitly.
- **Slosh moment error (V2).** The interior first-moment error for a *resolved* kernel sits near the rounding floor — far below the $\mathcal{O}(\sigma^2)$ allowance of the remark after Prop. 1 — because Clenshaw–Curtis integrates a resolved Gaussian's first moment spectrally. The $\mathcal{O}(\sigma^2)$-scale bias appears exactly where the paper localizes it: stations within $\sim 2\sigma$ of an endpoint (measured $2.2\times 10^{-3}$ relative at $x_s/\sigma = 2$, growing monotonically as the station approaches the end). Force transfer is exact everywhere, worst case $1.5\times 10^{-16}$ relative.
- **Density-rate convention (V4).** Eqs. (3.17)–(3.18) as printed source the gas continuity and pyrolysis enthalpy with $\partial\rho/\partial t|_\eta$ — the full computational-frame rate including grid advection — where the CMA lineage uses the material-frame Arrhenius rate alone; the two differ at $\mathcal{O}(\dot s\,\rho_\eta/\ell)$. The solver implements the paper's letter by default and exposes the choice as an explicit option (`density_rate_convention`) rather than deciding silently; the MMS verifies the default.
- **Filter calibration and recovery (V5).** The χ² gate fires at $9.8\times 10^{-4}$ against the design $p = 10^{-3}$ over $5\times 10^5$ nominal gate evaluations, and no replicate diverges in any of 27 $(N_w, \alpha_{\max}, p)$ configurations — the structural claim of Remark 8 that scalar inflation cannot destabilize the filter. IAE recovers from a 30 m/s separation transient in a median 1.30 s versus 2.15 s for the fixed-Q filter on identical measurements.
- **Batch scaling (V8).** GPU throughput ramps with log–log slope 1.00 in $N_{\mathrm{MC}}$ below saturation and peaks at ~85k replicates/s on the entry workload — 5× the vectorized CPU batch and ~380× a per-replicate Python loop (the decohered execution model a moving-mesh formulation forces). The batched IMEX Newmark structural block sustains millions of replicate-steps/s through one shared LU factorization, which is the §5.2 batching argument made measurable.
- **The conditioning claim, quantified (II-V1).** On the *same* fourth-order variable-$EI$ operator, the ultraspherical banded interior conditions as $\mathcal{O}(N^{1.01})$ where dense collocation gives $\mathcal{O}(N^{8})$ — a ratio of $7.8\times 10^{7}$ at $N = 64$. Under the leading-diagonal right preconditioner the interior $\kappa \le 13$ at every $N$ tested, reproducing the Olver–Townsend $\mathcal{O}(1)$ statement. The assembled *bivariate* Mindlin–Reissner block operator is likewise essentially $\mathcal{O}(1)$: $\kappa$ moves only from 101 to 108 while the problem grows five-fold. The caveat Paper II's Remark raises is confirmed on both: the dense boundary rows — not the interior, and not the block coupling — are what costs conditioning.
- **Conditioning is a statement about scaling, not just discretization (II-V1).** Measured with column equilibration alone the block operator appears to grow as $N^{1.6}$, which would fail the stated criterion. It does not: a block operator mixes entries in different physical units — transverse shear stiffness in N/m against bending rigidity in N·m, several decades apart — and one-sided scaling leaves that imbalance in the matrix. Standard two-sided (Ruiz) equilibration removes it and recovers $N^{0.09}$. The verification report prints both, so the preprocessing choice is visible rather than buried.
- **Free-free is not a BVP (II-V1).** Free-free boundary conditions leave the rigid-body null space, so the bordered free-free system is *singular by construction* — the configuration is a generalized eigenproblem, not a boundary-value problem. The solver detects this and says so rather than returning a factorization of a singular matrix; the bordered-conditioning sweep therefore uses clamped conditions.
- **An exact reference beats a tabulated one (II-V3).** Paper II specifies *published Rayleigh–Ritz values* for the free-free plate as V3's reference, but this repository cannot audit those numbers. Deriving the closed-form Mindlin solution for the **simply-supported** plate instead — separable in $(p,q)$, reducing to a symmetric $3\times 3$ pencil per mode — gives a reference that is exact, self-contained, and exercises the identical operator, boundary machinery, projection and eigensolve. Against it the error contracts exponentially to $3.7\times 10^{-10}$ at $N = 20$, clearing the $10^{-5}$ criterion by five orders of magnitude.
- **Free and supported perimeters fail differently at the corners (II-V1/V3).** The free-edge condition set is rank deficient by exactly **four** — the twist condition $M_{xy} = 0$ belongs to the free-edge triple of *both* edges meeting at each corner, while $M_x \ne M_y$ and $Q_x \ne Q_y$ are independent. A simply-supported perimeter is deficient by **twelve**. Neither number was assumed: the code measures the rank and fails loudly if it lands outside what four corners can produce, so a wrong boundary assembly surfaces immediately instead of as quietly wrong frequencies.
- **Free-free plates converge algebraically, and that is physical (II-V3).** The simply-supported spectrum converges exponentially; the free-free spectrum does not, because free-edge plates carry weak corner singularities. This is a property of the problem, not the discretization — which is exactly why the verdict rests on the exact reference and the free-free comparison against circulated values carries no verdict.
- **No shear locking across three decades (II-V2).** The fundamental frequency parameter falls from 11.71 at $h/L = 0.2$ (genuine thick-section shear softening) to a plateau of 13.307, and at $h/L = 10^{-3}$ differs from that plateau by $+0.018\%$ — against a 1% criterion. Paper II's Remark claims high-order spectral discretizations are "markedly less susceptible" to locking but explicitly declines to claim immunity; this measures it. The cost that *is* paid as sections thin is conditioning: the shear-to-bending stiffness ratio grows as $h^{-2}$, and the rigid-body modes separate from the elastic spectrum by correspondingly fewer decades.
- **The blend is load-neutral (II-V4).** Trim incidence moves by $0.027°$ across the full blend-width sweep — from the unblended $C^0$ closure out to a $2.3°$ half-width — against a $0.1°$ criterion, and the offset grows monotonically and vanishes as $\delta_{\mathrm{blend}} \to 0$. The seam the blend exists to fix is confirmed first: the unblended closure is continuous to $10^{-16}$ but its slope jumps, the Newtonian branch having zero slope at $\delta_c = 0$ where the expansion branch does not.
- **Freezing the structural block wins, and by how much (II-V5).** Carrying a 100 rad/s structural mode through a 300 s coast costs **15× more right-hand-side evaluations** than freezing it once its modal energy has decayed to $10^{-6}$ of initial — with both strategies holding energy to better than $10^{-8}$. Paper II's Remark 5 declined to call the trade in advance; the measurement calls it for freezing, and notes the saving *grows* with coast duration because the ring-down time is fixed while the coast is not. The switch is defensible only because the freeze condition is checked rather than assumed.
- **A quadratic blackout model under-predicts by 13× (II-V6).** All three covariance channels reproduce their $t^3$, $t^4$ and $t^6$ exponents to better than $10^{-6}$, measured from a Lyapunov propagation that carries the biases and tilt as *states* and so shares no code with the closed form. The operational consequence the paper's Remark asserts is now quantified: at two minutes of blackout the gyro channel carries 88% of the variance and a $t^2$ model under-predicts position uncertainty **13-fold**. A pull-up trigger sized on the quadratic model fires far too late.
- **Chattering is structural, not a tuning problem (II-V6).** A bare threshold on $\omega_p \ge \omega_{\mathrm{GNSS}}$ produces 966 transitions on a signal that merely hovers at the boundary. No choice of threshold fixes that — only hysteresis does, and a 10% reacquisition margin reduces it to the single legitimate latch. The gate is therefore hysteretic by construction.
- **The $\ell_1$ penalty is exact; $\ell_2$ is not (II-V7).** On the *same* linearized subproblem, the $\ell_1$ penalty drives $\|\bm{\nu}\|_1$ to **exactly 0.0** at every weight from $w_\nu = 3$ upward — a finite threshold, visible between 1 and 3 — while a quadratic penalty gives 92 → 12 → 1.3 over three decades of weight and then *grows again* as the solver loses the ill-conditioned problem. Both halves are the paper's argument: $\ell_2$ approaches zero only asymptotically, and the weight that makes it small is the weight that wrecks conditioning.
- **A modeling bug the coupled run caught (item 13).** The free-free spectrum's three zero-frequency modes are rigid translation and rotation, which the 6-DOF block already carries. Integrating them in the *modal* block too double-counts rigid motion, and because they have no restoring term the modal coordinate grows without bound under any steady load — it reached $10^7$ before the integrator failed. The structural state of Eq. (3.20) is the *elastic* deformation; only the coupled assembly exposes this, because each kernel is correct on its own.
- **Implicit beats adaptive, decisively (item 13).** With per-component tolerances the coupled system costs ~1970 right-hand-side evaluations under BDF, **flat across a 19× range of $\omega_{\max}$** (265 → 4973 rad/s) — the Prop. 2 constraint removed on the full assembly, not just the structural block. LSODA, nominally a stiffness-switching method, does not switch here and needs $10^5$–$10^6$ evaluations for the same arc. A single scalar `atol` cannot serve a state spanning nine orders of magnitude in units, and using one costs three orders of magnitude in work.
- **Blunting is a genuine trade, on published data (II-V8).** With convective heating falling as $R_{\mathrm{eff}}^{-1/2}$ and the published Tauber–Sutton correlation rising as $R_{\mathrm{eff}}^{+a}$, the total heating has an *interior* optimum **inside the source's own 0.3–3 m nose-radius envelope**, at a condition where radiation exceeds convection — the regime the correlation was written for. A convection-only framework drives the radius to the top of the range, which is the over-blunting bias Paper II predicts. The Lewis-exponent choice moves the Fay–Riddell bracket by 1.07% between equilibrium and frozen/catalytic, confirming the paper's "several percent" statement.
- **Fay–Riddell's exponents are derivable; its leading constant is not (II-V8).** Working from the original paper rather than its restatements gave one reassuring result and one uncomfortable one. The *exponents* are recoverable even though the archived scan of Eq. (63) is too degraded to read them off: Eq. (45) writes $q = (Nu/\sqrt{Re})\sqrt{\rho_w\mu_w (du_e/dx)_s}(h_s-h_w)/\Pr$ and Eq. (58)/(62) correlates $Nu/\sqrt{Re} = 0.67(\rho_s\mu_s/\rho_w\mu_w)^{0.4}\{\cdots\}$ — a **ratio**, since the parameter "was found to depend only upon the total variation in $\rho\mu$ across the boundary layer." Substituting gives 0.4 from the ratio plus a residual 0.1 from the $\sqrt{\rho_w\mu_w}$, i.e. exactly $(\rho_e\mu_e)^{0.4}(\rho_w\mu_w)^{0.1}$. The Lewis exponents check out directly (0.52 equilibrium, Eq. 60/62; 0.63 frozen/catalytic, Eq. 65 and Fig. 4). The *constant* does not survive as well: the same substitution shows every number in the correlation descending from **one fitted value at two significant figures**, the 0.67 of Eq. (58)/(62). The printed 0.94 of Eq. (63) is $0.67/0.71 = 0.9437$ rounded; the footnote's $\Pr^{-0.6}$ is the correlation's own $\Pr^{+0.4}$ against Eq. (45)'s $\Pr^{-1}$. Backing the coefficient out three ways gives **0.7684, 0.7654, and 0.76** — a 1.1% band, about 1% in heat flux. The **0.763** in general circulation, and in Paper II Eq. (4.1), appears *nowhere* in the source, though it sits inside that band and is as defensible as any of them. This is a **provenance** finding, not an accuracy finding — the spread is far inside the uncertainty of the edge properties that feed the correlation — recorded because a coefficient quoted to four figures implies a traceability the source supports at two. All four constants are named and selectable; 0.763 stays the default, for continuity with the literature rather than fidelity to the source.
- **Three things the published Tauber–Sutton source corrects (II-V8).** Transcribing Eqs. (1)–(4) from the archived PDF rather than paraphrasing them surfaced three discrepancies. (i) Paper II §4.2 describes the nose-radius exponent as $a \approx 1.0$; the source makes it $a = 1.072\times10^{6}V^{-1.88}\rho^{-0.325}$, *capped* at 0.6 for $1 \le r_n \le 2$ and 0.5 for $2 < r_n \le 3$, and states that $a < 1$ must always hold — it runs 0.25–0.6 across the envelope, never 1.0. (ii) The correlation returns **W/cm²**, so an SI framework needs a $10^4$ conversion; omitting it is a four-order-of-magnitude error that still resembles a heat flux. (iii) Table 1 tabulates $f_E$ from 9 km/s but Eqs. (1)–(2) are stated valid only from **10 to 16 km/s** — interpolating the table below 10 km/s is fine, applying the correlation there is not, and the implementation enforces the distinction. The band edges are honoured exactly: $r_n = 2$ m takes the 0.6 cap, not 0.5.

---

## Technical summary

What follows condenses the papers. Where the two disagree with each other, the papers say so explicitly and explain why.

### Structural discretization

Paper I discretizes the variable-rigidity Euler–Bernoulli beam by dense Chebyshev collocation, retaining the full product-rule expansion:

$$\mathbf{K} = \mathrm{diag}(\mathbf{EI})\,\mathbf{D}^4 + 2\,\mathrm{diag}(\mathbf{D}\mathbf{EI})\,\mathbf{D}^3 + \mathrm{diag}(\mathbf{D}^2\mathbf{EI})\,\mathbf{D}^2$$

Dropping the second and third terms is only valid where $EI$ is near-constant over a bending wavelength, which fails across stage joints and in regions of thermal softening.

Paper II replaces this. Dense collocation conditions as $\mathcal{O}(N^{2k})$ — that is $\mathcal{O}(N^8)$ for fourth-order bending — which is manageable in 1D but not for a bivariate tensor-product operator. Paper II uses the **ultraspherical spectral method**, where differentiation is banded and conditioning is $\mathcal{O}(1)$, with the bivariate system assembled as a Kronecker-structured generalized Sylvester equation.

Paper II also replaces Euler–Bernoulli kinematics with **Mindlin–Reissner** three-field plates $(w, \phi_x, \phi_y)$. Transverse shear is not a correction for thick hull sections — it is the dominant mechanism of the torsional modes that set flutter margins. This requires three independent free-edge conditions ($M_x = M_{xy} = Q_x = 0$) rather than the classical Kirchhoff effective-shear pair, which over-constrains the perimeter.

### The stiffness constraint

This is stated rather than assumed away, because it is the sharpest limit on the architecture. For the spectral structural operator, $\lambda_{\max}$ inherits the $\mathcal{O}(N^8)$ growth of the fourth-derivative operator, so $\omega_{\max} = \mathcal{O}(N^4)$, and any explicit Runge–Kutta method requires

$$\Delta t \le C_{\mathrm{RK}} / \omega_{\max} = \mathcal{O}(N^{-4})$$

At $N = 32$ that factor is $\sim 10^6$. Continuity does not rescue you from this. Paper I gives two mitigations — modal truncation and IMEX splitting, the latter preserving the batching argument because the factorization is shared across all replicates — and V3 measures which is preferable.

### Aerothermodynamics

Stagnation convective heating follows **Fay–Riddell**, with the Lewis exponent stated ($\beta = 0.52$ equilibrium, $0.63$ frozen/catalytic) and the stagnation velocity gradient supplied by the modified Newtonian estimate. **Tauber–Sutton** gives the radiative component, and **Lees** the distribution away from stagnation.

Note the trade the papers make explicit: recession increases $R_\mathrm{eff}$, which *reduces* convective heating as $R_\mathrm{eff}^{-1/2}$ but *increases* radiative heating. A framework modeling only convection will systematically favor over-blunted geometries.

Ablation is split by material class rather than forced into one model — charring pyrolysis for phenolic acreage, single-temperature oxidative recession for non-pyrolyzing refractory leading edges (C/C, ZrB₂–SiC). Applying either model to the other material class is wrong in a specific, stated way.

### Navigation and guidance

Anomaly detection gates on the normalized innovation squared, $d_k^2 = \bm{\nu}_k^\top \mathbf{S}_k^{-1} \bm{\nu}_k \sim \chi^2_m$, against a design false-alarm rate. On detection, IAE inflates process noise by a **bounded** scalar trace ratio $\alpha_k \in [1, \alpha_{\max}]$.

Paper I is explicit that positive-semidefiniteness of $\mathbf{Q}^*_k$ follows trivially from scaling a PSD matrix by a non-negative scalar — *not* from the trace clamp, which says nothing about matrix definiteness. What makes the scheme unconditionally well-posed is that it estimates a **scalar** rather than a matrix; entrywise estimators of $\mathbf{Q}$ routinely return indefinite matrices under short windows.

Terminal guidance uses the numerically stable time-to-go root:

$$t_{go} = \frac{2 R_\mathrm{LOS}}{\hat V_c + \sqrt{\hat V_c^2 + 2 \hat A_c R_\mathrm{LOS}}}$$

This is algebraically identical to the textbook quadratic form but does not lose precision as $\hat A_c \to 0$ — which is where terminal guidance spends most of its time. The textbook form evaluates $0/0$ by catastrophic cancellation there. A guard handles the negative-discriminant case (decelerating closure predicts no intercept), which is physically meaningful and must not propagate as NaN.

Paper II's SCvx layer uses **free** virtual controls under an exact $\ell_1$ penalty. Sign-constraining them makes the subproblem infeasible in exactly the cases they exist to rescue. The $\ell_1$ norm is chosen because it is *exact* — above a finite penalty weight the virtual controls reach zero exactly, where a quadratic penalty only drives them to zero as $w_\nu \to \infty$.

For plasma blackout, the unaided covariance grows as

$$\mathbf{P}_{rr}(t) \sim \tfrac{1}{3} q_a t^3 + \tfrac{1}{4}\sigma_{b_a}^2 t^4 + \tfrac{1}{36} g^2 \sigma_{b_g}^2 t^6$$

in the velocity-random-walk, accelerometer-bias, and gyro-bias-through-gravity channels. The $t^6$ term dominates past a few tens of seconds. Quadratic is the growth of position *error* from an accelerometer bias, not of the *covariance* — a guidance layer sizing its pull-up trigger on a quadratic model will under-predict badly at the durations that matter.

### Dispersion statistics

Terminal footprints are characterized by the eigendecomposition of the sample impact covariance. $R_{95}$ uses $\chi^2_{2,0.95} = 5.991$, giving semi-axes $2.4477\,\sigma_i$. CEP uses the linear approximation $0.5887(\sigma_1 + \sigma_2)$ — **but only where $0.25 \le \sigma_2/\sigma_1 \le 1$**, which lifting reentry footprints routinely violate. Outside that range the framework falls back to the direct order statistic rather than reporting a CEP the approximation does not support.

Relative standard error on each $\sigma_i$ is $\approx 1/\sqrt{2N_\mathrm{MC}}$ — 0.7% at $N_\mathrm{MC} = 10^4$. Any dispersion figure quoted without a sample size, or to more significant figures than that bound supports, is not meaningful.

---

## Repository layout

```text
├── passes-updated.tex          # Paper I  — source
├── passes-updated.pdf          # Paper I  — compiled
├── passes-references.bib       # Paper I  — verified bibliography
├── passes-hgv-updated.tex      # Paper II — source
├── passes-hgv-updated.pdf      # Paper II — compiled
├── passes-hgv-references.bib   # Paper II — verified bibliography
├── CITATION-AUDIT.md           # verification log for every reference
├── Makefile / .latexmkrc       # papers + code targets
├── pyproject.toml              # package metadata, ruff + mypy strict config
├── src/passes/                 # implementation (roadmap items 1–6)
│   ├── spectral/               #   CGL nodes, direct-recurrence D^(k),
│   │                           #   Clenshaw–Curtis, barycentric interpolation
│   ├── structures/             #   profiles, product-rule K assembly,
│   │                           #   null-space projection, modal solve,
│   │                           #   Newmark IMEX + exact modal propagator
│   ├── coupling/               #   bandwidth-adapted, quadrature-normalized
│   │                           #   slosh load regularization
│   ├── thermal/                #   Arrhenius kinetics, Landau frame,
│   │                           #   in-depth energy + gas-flux solver,
│   │                           #   surface energy balance + blowing
│   ├── guidance/               #   stable t_go with guard, AC-APN law
│   ├── estimation/             #   batched χ²-gated IAE Kalman filter
│   ├── batch/                  #   NumPy/CuPy backends, Philox sampling,
│   │                           #   rank-3 batched RK4, dispersion stats
│   │                           #   (R95, CEP + fallback, bootstrap, HZ)
│   ├── ultraspherical/         #   sparse D_k / S_λ / Jacobi operators,
│   │                           #   banded variable-coefficient assembly,
│   │                           #   bordered BVP solve  [Paper II]
│   ├── aerothermal/            #   Fay–Riddell, Sutton–Graves, Tauber–
│   │                           #   Sutton, Lees, leading-edge recession
│   ├── plates/                  #   Mindlin–Reissner 3-field kernel: Kronecker
│   │                           #   block assembly, free / simply-supported
│   │                           #   edges, exact SS reference  [Paper II]
│   ├── dynamics/               #   quaternion + Baumgarte, deformed normals,
│   │                           #   local incidence  [Paper II]
│   ├── aerodynamics/           #   Rayleigh–Pitot, Prandtl–Meyer, C² blend,
│   │                           #   panel loads and trim  [Paper II]
│   ├── orbital/                #   J₂ gravity, coast propagation, regime
│   │                           #   transition, strategy comparison  [Paper II]
│   ├── flight/                 #   coupled single-trajectory simulator: fixed
│   │                           #   global state, one integrator, all regimes
│   └── verification/           #   executable V1–V8, II-V1…II-V8, integration
├── tests/                      # 425 pytest cases
├── results/                    # verification reports and CSV data
└── passes.tex, passes-hgv.tex  # superseded earlier drafts (see note)
```

`passes.tex`, `passes-hgv.tex`, and `references.bib` are earlier drafts retained for history. They contain the citation errors documented in the audit and should not be built or cited.

## Building the papers

Requires a TeX Live distribution with `latexmk` and `biber`.

```bash
latexmk -pdf passes-updated.tex
latexmk -pdf passes-hgv-updated.tex
```

Output lands in `build/` and is copied to the repository root. Both compile with zero LaTeX warnings and zero biber warnings; if yours do not, something is wrong with the toolchain rather than the sources.

```bash
latexmk -C          # clean
```

## Running the code

Requires Python ≥ 3.10 with NumPy ≥ 1.26 and SciPy ≥ 1.11. From the repository root:

```bash
pip install -e .[dev]           # editable install with dev tooling
pip install -e .[cuda]          # optional: CuPy backend for the GPU batch layer
make test                       # 425 pytest cases (GPU tests skip without CUDA)
make verify                     # execute every task + the integration check
make check                      # ruff + mypy --strict + tests + verification
```

The verification runners are the authoritative record: each writes a markdown report stating the acceptance criterion from §8 of Paper I, the measured values, and a PASS/FAIL verdict, plus CSV files with the raw numbers. Sample results are committed under [`results/`](results/); regenerate them locally with `make verify`.

---

## Roadmap

Ordered by dependency, not ambition.

1. ~~**Structural kernel** — Chebyshev operators, null-space projection, free-free eigenvalue solve.~~ **Done.** V1 (conditioning vs. $N$, frequencies vs. analytic) and V3 (integrator strategy comparison) executed and passing.
2. ~~**Guidance numerics** — self-contained and quick.~~ **Done.** V6 (the $t_{go}$ precision comparison in single and double precision) executed and passing; the conjugate form holds the precision floor across the full $\hat A_c$ sweep while the textbook form degrades to under one significant digit.
3. ~~**Slosh regularization**~~ **Done.** V2 executed and passing: force transfer exact to $1.5\times 10^{-16}$ across $N$, $\gamma$, and station; interior moment error within (in fact far below) the $\mathcal{O}(\sigma^2)$ bound; the endpoint bias measured and localized to $x_s \lesssim 2\sigma$ as the paper predicts.
4. ~~**Thermal solver** — Arrhenius kinetics, Landau transform, surface energy balance.~~ **Done.** V4's MMS leg executed and passing (spectral convergence to the $10^{-15}$ time-integration floor); kinetics verified against closed-form isothermal solutions; gas-flux operator exact on polynomials; SEB solved by bracketed Brent with the log1p blowing correction. The FIAT comparison leg remains **pending** external reference data — the comparison harness is ready for a tabulated $(t, s, T)$ reference.
5. ~~**Filter**~~ **Done.** V5 executed and passing: false-alarm rate $9.8\times 10^{-4}$ against design $p = 10^{-3}$, zero divergence across all 27 sensitivity configurations, and IAE recovery measurably faster than the fixed-Q baseline on identical data.
6. ~~**Batch layer**~~ **Done.** V7 and V8 executed and passing: CEP sampling error converges at the $1/\sqrt{2N_{\mathrm{MC}}}$ rate (fitted slope −0.63), the $R_{95}$ ellipse empirically contains 95.0% of impacts, the elongated-footprint CEP fallback engages exactly at the validity boundary, and the CUDA batch scales linearly below saturation on real hardware. Occupancy counters await profiler instrumentation.

Paper I's roadmap is complete. The extension below covers Paper II's formulation and the remaining rows of the status table, again ordered by dependency: the ultraspherical core precedes the plates that need it, the correlations are self-contained, and the coupled integration comes last because it consumes everything else. Paper II's verification tasks are prefixed **II-V** to keep the two papers' numbering distinct.

7. ~~**Ultraspherical spectral core**~~ **Done.** II-V1's univariate leg executed and passing: banded interior conditioning measured at $\mathcal{O}(N^{1.01})$ against the O(N) criterion, $\kappa \le 13$ under the leading-diagonal preconditioner, and free-free beam eigenvalues matching the analytic solution to $1.5\times 10^{-8}$ — the same physical problem Paper I verified in collocation form, reproduced by an independent discretization. The block-operator measurement lands with item 9.
8. ~~**Aerothermal correlations**~~ **Done.** II-V8's implementation leg executed and passing: exact $R_{\mathrm{eff}}^{-1/2}$ scaling through the modified-Newtonian gradient, quantified Lewis-exponent sensitivity, Lees continuity at the stagnation-region boundary to $5\times 10^{-10}$, and the interior blunting optimum demonstrated. The published-reference-case comparison awaits transcribed reference data, as with I-V4's FIAT leg.
9. ~~**Mindlin–Reissner plate kernel**~~ **Done.** II-V1 (block leg), II-V2 and II-V3 executed and passing: the assembled block operator is $\mathcal{O}(1)$-conditioned under two-sided equilibration, no shear locking across three decades of $h/L$, MMS exact to $7\times 10^{-13}$, and frequencies matching the closed-form Mindlin simply-supported solution to $3.7\times 10^{-10}$.
10. ~~**6-DOF state and aerodynamic blending**~~ **Done.** II-V4 executed and passing: trim incidence shifts by $0.027°$ across the blend-width sweep against a $0.1°$ criterion, with the $C^0$-but-not-$C^1$ seam measured directly and the $C^2$ smoothstep shown to close it.
11. ~~**Orbital coast**~~ **Done.** II-V5 executed and passing: secular energy drift $1.7\times 10^{-14}$ per orbit against a $10^{-8}$ criterion, nodal regression matching the analytic first-order rate to 0.2%, and the frozen-structure strategy costing 15× fewer right-hand-side evaluations than carrying the block through coast.
12. ~~**SCvx and blackout gating**~~ **Done.** II-V6 and II-V7 executed and passing: all three covariance channels reproduce their $t^3/t^4/t^6$ exponents to 1e-6, the hysteretic gate eliminates boundary chattering, and the $\ell_1$ penalty drives the virtual controls to *exactly* zero at $w_\nu = 3$ where an $\ell_2$ penalty on the same subproblem never reaches zero at any weight.
13. ~~**Coupled flight integration**~~ **Done.** The simulator in [`src/passes/flight/`](src/passes/flight/) advances a 71-component global state — rigid-body translation and attitude, elastic modal coordinates, and the charring thermal block on its Landau grid — through entry as **one ODE with one integrator**, with the checks in [`results/int-coupled.md`](results/int-coupled.md): fixed dimension, no regime branch, the aerothermal loop closing inside a single right-hand side, and implicit cost flat in retained modes. I-V8's occupancy instrumentation is closed as far as it can be without a profiler.

**Items 1–13 are complete.** The extension below is not a wish list: every item is something the *implementation* uncovered, ordered by how much it constrains what the framework can currently claim. Both manuscripts have been updated in place to record these findings — corrections identified as corrections, measurements quoted inline — so the roadmap and the papers now say the same thing.

### Blocked on external material (not code)

14. **FIAT ablation reference (I-V4).** Needs the FIAT code or its published reference-case output. The comparison harness accepts any tabulated $(t, s, T)$ reference. [`docs/FIAT-reference-data.md`](docs/FIAT-reference-data.md) sets out exactly what a usable case must contain, where to look for one (NTRS, the Chen & Milos FIAT paper already in the bibliography, NASA STI, FIAT itself under export control), why PATO/CHAR are a cross-verification rather than this criterion, and what to do if none of it succeeds.
15. **Fay–Riddell reference conditions (II-V8, convective half).** Needs transcribed equilibrium-air properties ($\rho_e\mu_e$, $\rho_w\mu_w$, $h_D$) at the published reference states. The radiative half is closed against the Tauber–Sutton source and the convective *coefficients* are now checked against the Fay & Riddell original, so what is left is a tabulated reference *case*. The original presents its results in figures (Figs. 2, 3, 4, 7, 8), not tables, and cannot supply one at the precision a 5% criterion needs; a later source with tabulated reference conditions is required.
16. **Achieved occupancy and warp divergence (I-V8).** Nsight Compute is installed and the code path is wired, but the driver returns `ERR_NVGPUCTRPERM` for a non-root user. Enabling counters means loading the NVIDIA module with `NVreg_RestrictProfilingToAdminUsers=0` — a host-wide security setting, deliberately not changed from here.

### Uncovered by implementation, and currently limiting

17. **Boundary-row preconditioning.** II-V1 measured the banded ultraspherical interior at $\mathcal{O}(1)$ and the *bordered* system at $N^{1.4}$–$N^{3.3}$. The conditioning claim the method is sold on therefore belongs to the interior, not to what actually gets solved. Any statement about scaling this discretization to large $N$ is a statement about the boundary treatment, and nothing here preconditions it.
18. **Corner-resolving refinement for free edges.** Free-edge plates carry weak corner singularities, so the free-free spectrum converges *algebraically* where the simply-supported spectrum converges exponentially. Until this is addressed, II-V3's free-free leg cannot carry a pass criterion at attainable $N$, and the verdict rests on the closed-form simply-supported reference. Graded meshes or singular enrichment would close it.
19. **Variable-coefficient bivariate operators.** Paper II specifies $\mathbf{D}_{ij}(\xi,\eta)$ built by hyperbolic blending; the plate kernel currently uses constant coefficients. Separable fields drop straight into the Kronecker assembly; genuinely bivariate ones need a low-rank approximation of each coefficient for the Sylvester structure to survive.
20. **Enthalpy-form thermal state.** Paper I, Eq. (3.20) carries nodal *enthalpy* precisely because it stays continuous across the pyrolysis zone where $c_p$ varies sharply. The charring solver integrates temperature, which is what the coupled state currently carries. Converting is a caloric-equation-of-state inversion, and the argument for doing it is strongest exactly where the current form is weakest.
21. **Batched coupled trajectories.** The batch layer and the coupled simulator both exist but have not been composed: the fixed-dimension state makes a rank-3 ensemble of *full* trajectories available, which is the framework's headline claim at full strength rather than on the reduced entry model I-V7/V8 use.

### Numerical practice worth promoting to first-class support

22. **Constraint stabilization as a framework service.** Baumgarte stabilization is used for the quaternion norm (Paper I) and, as the coupled simulator forced, for the thermal boundary conditions. Both are hand-rolled. A single tested facility for "differentiate the constraint, stabilize the drift" would generalize to the plate edge conditions and the free-free projection.
23. **Per-block tolerance and scaling policy.** A scalar `atol` costs three orders of magnitude on a state spanning nine orders of magnitude in units, and a block condition number is meaningless without its scaling stated. Both are currently conventions inside two modules rather than a policy the framework enforces.

---

## Citing this work

Both manuscripts are unpublished preprints. Will update these entries once arXiv identifiers are assigned.

```bibtex
@misc{knox2026passes,
  author = {Knox, Ethan},
  title  = {{PASSES}: A Unified, {GPU}-Accelerated Spectral Framework for Coupled
            Aeroelastic, Thermodynamic, and Stochastic {GNC} Flight Simulation},
  year   = {2026},
  note   = {Preprint}
}

@misc{knox2026hgv,
  author = {Knox, Ethan},
  title  = {{PASSES-HGV}: Extending a Fixed-Grid Spectral Framework to 6-{DOF}
            Hypersonic Glide and Fractional Orbital Trajectories},
  year   = {2026},
  note   = {Preprint}
}
```

---

## Scope and compliance

This repository contains mathematical formulation only: published governing equations, engineering-level correlations drawn from the open literature (Fay–Riddell, Lees, Tauber–Sutton, Sutton–Graves, CMA), and standard estimation and optimal control methods. Every source is cited and independently verifiable — see [`CITATION-AUDIT.md`](CITATION-AUDIT.md).

It contains **no** proprietary, classified, or export-controlled data: no vehicle geometries, no material property databases, no performance specifications, no validated aerodynamic tables, and no parameters traceable to any specific fielded system. The intended use is study of the numerical coupling between continuum mechanics and statistical estimation theory.

Under ITAR, information already published and generally accessible to the public is not "technical data" (22 CFR §120.34, public domain). Note that the separate *fundamental research* provision is defined with reference to accredited U.S. institutions of higher learning and does not straightforwardly cover independent researchers — the two are frequently conflated and are not the same exemption. **This is a description of intent, not legal advice.** If you are extending this work with real vehicle data or fielded-system parameters, get an actual export-control determination first.

## License

MIT — see [`LICENSE`](LICENSE).
