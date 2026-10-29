# Per-band background correction of reconstructed R_n: results

2026-10-29, branch `smash-train-20261028`. Protocol:
`reports/rn_background_correction_protocol_20261029.md` (revised after its
preflight review; validation only). Every number comes from
`results/rn_background_correction_20261029/summary.json`, written by
`scripts/rn_background_correction.py`; figure
`results/rn_background_correction_20261029/figures/background_coefficients.pdf`.
No report-set file was read and no number enters the manuscript.

<!-- table-provenance: validation only. SMASH: 286 truth-bound even-part files
     of corrected, deduplicated SMASH Xe+Cs 2.87 GeV/nucleon, each scored by the
     training fold that validates it, at its frozen 0.70 cut and calibration.
     DCM: the frozen A model's validation, DCM-QGSM-SMM Xe+CsI 3.26 AGeV, run
     988974, 41,808 events. Labels and background counts are MC truth; bands
     and selected counts are reconstructed. -->

## The quantity

The hard-cluster correction subtracts c·N_H from the low band, with
c = B_L/N_H the low-band background per selected high-band cluster, fitted
pooled over the potentials. To first order the corrected contrast is C4/π_L,
so on the test files it would add about C4·(1/π_L − 1) ≈ 1.5 points to B1
(C4 ≈ 1.9 %, π_L ≈ 0.56); the measured truth-label gap C3 − C4 is 1.1–1.3
points. The correction is useful only if c is known to better than that
scale.

## Within SMASH: c across the potential

Ratio 90 MeV / 0 MeV per training seed, three folds pooled (286 files). The
K bias is what a pooled c would leave in the corrected contrast,
−100·(ratio − 1)·(1 − π_L)/π_L. Spread: SD of 400 whole-file resamples.

| arm, seed | c ratio → 1 | K bias (points) → 0 | β_L ratio | β_H ratio | hard clusters per event, ratio | π_L |
|---|---|---|---|---|---|---|
| B1, 42 | 0.999 ± 0.017 | +0.07 ± 1.31 | 1.019 ± 0.014 | 1.027 ± 0.019 | 1.020 ± 0.009 | 0.557 |
| B1, 123 | 1.006 ± 0.015 | −0.44 ± 1.20 | 1.024 ± 0.013 | 1.022 ± 0.019 | 1.018 ± 0.008 | 0.557 |
| B1, 456 | 0.994 ± 0.017 | +0.51 ± 1.32 | 1.013 ± 0.015 | 1.021 ± 0.019 | 1.019 ± 0.008 | 0.559 |
| A1, 42 | 1.016 ± 0.021 | −1.33 ± 1.79 | 1.030 ± 0.020 | 1.003 ± 0.016 | 1.014 ± 0.009 | 0.544 |
| A1, 123 | 1.003 ± 0.020 | −0.24 ± 1.72 | 1.020 ± 0.019 | 1.000 ± 0.017 | 1.017 ± 0.009 | 0.534 |
| A1, 456 | 1.018 ± 0.019 | −1.76 ± 1.84 | 1.032 ± 0.018 | 0.994 ± 0.018 | 1.013 ± 0.009 | 0.508 |

In B1 the background per event and the hard clusters per event both rise by
about 2 % from 0 to 90 MeV (the latter at 2.2–2.5 SD), while c stays within
0.7 % of unity. That is the pattern the hard-cluster model assumes, and a
constant-per-event model would miss it. The test has little power. The K bias
is known to ±1.2–1.3 points per seed for B1 and ±1.7–1.8 for A1, as large as
the correction itself. Reaching ±0.5 points for B1 would need about
286·(1.26/0.5)² ≈ 1,650–2,000 files, 3 to 3.5 times the 568-file truth-bound
census.

## Between productions: c for the same A1 model

Per run (fold × seed), the DCM validation scored by the frozen A model with
that run's A1 cut; DCM energies with the run's SMASH calibration (the stated
premise) and raw (variant). Spread: DCM event bootstrap × SMASH file
bootstrap, 400 replicates.

| quantity (DCM / SMASH) | SMASH calibration | raw DCM energy |
|---|---|---|
| r = c ratio | 0.51–0.57 (SD 0.02) | 0.57–0.65 (SD 0.02) |
| low-band background per event, ratio | 0.89–0.98 (SD 0.035) | 0.93–1.06 (SD 0.035) |
| selected hard clusters per event, ratio | 1.73–1.76 (SD 0.02) | 1.52–1.80 (SD 0.02) |
| fraction of the correction removed, (1 − r)/(1 − r(1 − π_L)) | 0.59–0.66 (SD 0.02) | 0.50–0.59 (SD 0.025) |

For reference, half of the correction is removed at r = 1/(1 + π_L) =
0.65–0.67. A correction calibrated on one of these productions and applied to
the other would lose 59–66 % of its size, or 50–59 % without recalibrating the
DCM energy scale. Most of the change in c comes from the selected hard
clusters per event, 1.74 times higher in the DCM sample. The background per
event changes much less. The comparison changes generator, beam energy (3.26
against 2.87 AGeV) and target (CsI against Cs) together and does not separate
them. The DCM label "3.26 AGeV" comes from project records; without the run
card it is not verified whether it is the kinetic energy per nucleon or
√s_NN (which equals 3.26 GeV for E_kin/A = 3.8 GeV).

Per-event rates use events with at least one candidate cluster in both
samples. With the processed-event count for SMASH instead, both per-event
ratios rise by about 1.35. The ratio r does not depend on this choice.

## For the path-B decision

A simulation-calibrated per-band correction is not established as a remedy on
the present inputs. Within SMASH its coefficient is consistent with
potential-independence, but only to ±1.2–1.8 points of the contrast, which is
the size of the correction. Between two simulated productions the coefficient
moves enough to remove about 60 % of the correction. Establishing it would
need about three times more truth-bound files, plus a way to fix the
hard-cluster yield independently of the simulation, for example from data.
The report-set check (Part B of the draft) was not run, so the 282 odd-part
files remain unread for a later test.

## What this does not establish

The production comparison is A1 only; B1 would need inference on the DCM
validation (a GPU job). Generator, energy and target are not separated. The
three DCM seeds share the same events, and the DCM validation also fixed A's
epoch and DCM cut. The SMASH values were seen once while the protocol was
drafted, on the same files, and serve as reproduction guards. One detector
simulation is used, and nothing is measured on data.
