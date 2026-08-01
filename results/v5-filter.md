# V5: Adaptive filter — recovery, sensitivity, false-alarm calibration

- **Failure criterion (stated in advance, Paper I §8):** divergence on any replicate; measured false-alarm rate above 2p (design p = 1e-3)
- **Verdict:** **PASS**
- **Generated:** 2026-08-01 01:23 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Scenario

Constant-velocity tracker (Δt = 0.05 s, position measurement, σ_z = 2 m), 500 replicates. Separation transient: an unmodeled 30 m/s velocity jump. Recovery is declared when the NIS holds below the gate for 10 consecutive steps. All runs share seeds, so configurations see identical measurement realizations.

## False-alarm rate (nominal flight, design p = 1e-3)

Measured **9.80e-04** over 500,000 gate evaluations (binomial 1σ ≈ 4.5e-05) against the criterion ≤ 2p = 2e-03 → **PASS**. Replicates diverged: 0.

## Recovery from the injected transient (identical measurements)

| configuration | median recovery (s) | 95th pct (s) | unrecovered | max α seen |
|---|---|---|---|---|
| IAE, α_max = 500 | 1.30 | 1.50 | 0 | 77.6 |
| fixed Q (α_max = 1) | 2.15 | 2.55 | 0 | 1.0 |

## Recovery acceptance

Every adaptive replicate recovers, none diverges, and the adaptive median recovery (1.30 s) beats the fixed-Q filter (2.15 s) on identical data → **PASS**.

## Sensitivity to (N_w, α_max, p) — 27 configurations × 500 replicates

| N_w | α_max | p | median recovery (s) | pre-jump false-alarm | diverged |
|---|---|---|---|---|---|
| 10 | 10 | 1e-02 | 1.60 | 1.0e-02 | 0 |
| 10 | 10 | 1e-03 | 1.55 | 1.2e-03 | 0 |
| 10 | 10 | 1e-04 | 1.50 | 1.6e-04 | 0 |
| 10 | 100 | 1e-02 | 1.30 | 1.0e-02 | 0 |
| 10 | 100 | 1e-03 | 1.25 | 1.2e-03 | 0 |
| 10 | 100 | 1e-04 | 1.20 | 1.6e-04 | 0 |
| 10 | 1000 | 1e-02 | 1.30 | 1.0e-02 | 0 |
| 10 | 1000 | 1e-03 | 1.25 | 1.2e-03 | 0 |
| 10 | 1000 | 1e-04 | 1.20 | 1.6e-04 | 0 |
| 20 | 10 | 1e-02 | 1.60 | 1.0e-02 | 0 |
| 20 | 10 | 1e-03 | 1.55 | 1.2e-03 | 0 |
| 20 | 10 | 1e-04 | 1.50 | 1.6e-04 | 0 |
| 20 | 100 | 1e-02 | 1.35 | 1.0e-02 | 0 |
| 20 | 100 | 1e-03 | 1.30 | 1.2e-03 | 0 |
| 20 | 100 | 1e-04 | 1.30 | 1.6e-04 | 0 |
| 20 | 1000 | 1e-02 | 1.35 | 1.0e-02 | 0 |
| 20 | 1000 | 1e-03 | 1.30 | 1.2e-03 | 0 |
| 20 | 1000 | 1e-04 | 1.30 | 1.6e-04 | 0 |
| 40 | 10 | 1e-02 | 1.65 | 1.0e-02 | 0 |
| 40 | 10 | 1e-03 | 1.55 | 1.2e-03 | 0 |
| 40 | 10 | 1e-04 | 1.50 | 1.6e-04 | 0 |
| 40 | 100 | 1e-02 | 1.45 | 1.0e-02 | 0 |
| 40 | 100 | 1e-03 | 1.40 | 1.2e-03 | 0 |
| 40 | 100 | 1e-04 | 1.35 | 1.6e-04 | 0 |
| 40 | 1000 | 1e-02 | 1.45 | 1.0e-02 | 0 |
| 40 | 1000 | 1e-03 | 1.40 | 1.2e-03 | 0 |
| 40 | 1000 | 1e-04 | 1.35 | 1.6e-04 | 0 |

## Sensitivity reading

Recovery time is insensitive to N_w over the 0.5–2 s window band and to α_max above ~10 — the inflation saturates the window statistics either way — and mildly sensitive to p through the gate re-arm behavior. No configuration diverges, which is the structural claim of Remark 8: scalar inflation cannot destabilize the filter because Q* stays within [Q_nom, α_max Q_nom].
