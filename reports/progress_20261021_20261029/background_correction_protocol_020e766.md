# Per-band background correction of reconstructed R_n: protocol

Draft, 2026-10-29, branch `smash-train-20261028`. Requested by the user on
2026-10-29 after the purity scan (`reports/rn_purity_scan_protocol_20261029.md`)
was withdrawn: a per-band background correction of C4 at the frozen 0.70 cut,
with its model dependence as the main question. No training or inference; the
analysis re-reads published, verified predictions. No number enters the
manuscript.

## Question and consumer

At the 0.70 cut the reconstructed contrast C4 keeps about 40 % of the truth
contrast (`reports/smash_train_results_20261029.md`), and the largest loss is
C4 − C3, the background admitted by the selection (B1 −1.1 to −1.3 points).
Can that loss be corrected by subtracting an estimated background per energy
band, and how much does the correction depend on the background model and on
the generator that calibrates it? The consumer is the path-B decision of
review v9: whether background can be handled by a simulation-calibrated
correction, or must be reduced or measured some other way.

## What a correction must do

The hardness contrast is K = R(90)/R(0) − 1 with R = high-band count /
low-band count (high: reconstructed E ≥ 2 GeV; low: [0.3, 1) GeV). A
correction that multiplies each band by a purity that is the same at every
potential cancels in K exactly, so it changes nothing. A purity measured
separately per potential does change K, but requires knowing the potential,
which a measurement on data does not. A usable correction must therefore model
how the background count depends on quantities observed in the event.

Design evidence, computed while drafting on validation data only (the
truth-bound files of the even split parts, scored by the fold that validates
them; no test number at a new setting was computed). The script recomputes
every value with its uncertainty.

- Band purity at the 0.70 cut, B1, 0 → 90 MeV: low band 0.5609 → 0.5557, high
  band 0.8101 → 0.8103. The 0.5-point drop in low-band purity alone shifts K by
  about 1 point, the size of C4 − C3.
- Background clusters per event, B1, 0 → 90 MeV: low band 27.1 → 27.7 per
  1000 events (+2.1 %), high band 16.1 → 16.5 (+2.7 %); low-band signal per
  event is flat (34.6). Background is not constant per event.
- Low-band background per high-band signal cluster, B1: 0.3946 at 0 MeV,
  0.3944 at 90 MeV. Background in the low band scales with hard clusters.
- Generator: for the same A1 model, cut and calibration, the low-band
  background per hard signal cluster κ_L is 0.148–0.156 on DCM validation
  (DCM-QGSM-SMM Xe+CsI 3.26 AGeV, run 988974, 101,107 clusters per seed) and
  0.274–0.313 on SMASH validation; the high-band coefficient κ_H is 0.27–0.32
  against 0.31–0.37.

## Background models

Fixed before any report-set number. Each is fitted per run (fold k, seed s,
arm) on that run's fold validation, truth-bound files only, pooled over the
three potentials, at the frozen cut, with the run's calibration applied to
e_pred. N_b is the selected count in band b, n_ev the processed events of the
file (span from its evaluation CSV), Ŝ_b the corrected count.

| model | correction | assumption |
|---|---|---|
| M_event | Ŝ_b = N_b − β_b·n_ev | background per event is the same at every potential |
| M_hard | Ŝ_H = N_H/(1+κ_H); Ŝ_L = N_L − κ_L·Ŝ_H | background per hard signal cluster is the same at every potential |
| M_hard, DCM (A1 only) | as M_hard with κ from DCM validation | the coefficients transfer from DCM to SMASH |
| oracle | Ŝ_b(U) = π_b(U)·N_b(U), π per potential from validation | knows the potential; reference only |

β_b = background clusters per event in band b; κ_b = background clusters in
band b per high-band signal cluster; π_b(U) = band purity at potential U.

## Measurements

**A. Coefficients (primary; validation, high statistics).** For B1 and A1 per
run: β_b, κ_b and π_b per potential, with a whole-file bootstrap over
validation files (400 replicates). Reported: the potential dependence of β_b
and κ_b (ratio 90 MeV / 0 MeV), and for A1 the generator ratio
κ_b(DCM)/κ_b(SMASH), its spread over runs, and the shift of the corrected
contrast it implies, evaluated on the report-set counts.

**B. Corrected ladder (secondary; report set).** Report set F: the 282
truth-bound files of the odd split parts, which are never validation; each
file is scored by the fold whose test it is and corrected with that fold's
coefficients. Per arm and seed: C4, C3, T1 and each model's corrected
contrast, with paired differences (model − C4, model − C3, M_hard DCM − M_hard
SMASH) on the 400 whole-file draws shared with `reco_rn_heldout.py`. All 568
files are reported as a secondary scope, flagged because the even-part files
supplied the design evidence above.

Power: on the even parts the bootstrap SD of the M_hard-corrected contrast is
2.0–2.5 points (B1, 286 files) against 1.1–1.4 uncorrected, the paired SD of
M_hard − C3 is 1.3–1.5 and of M_hard − C4 0.9–1.1; F is the same size. Part B
can therefore detect the correction (about +1 point) only at about 1 SD, and
cannot test closure at the ±0.5-point level. Its numbers are a consistency
check on Part A, and a null in Part B is uninformative.

## Predictions (committed before the measurement)

Ranges are centred on the design evidence and widened by about one bootstrap SD
where the value is a contrast. Each is checked per seed unless stated.

| quantity | predicted |
|---|---|
| B1 κ_L(90)/κ_L(0), validation | 0.97–1.03 |
| B1 β_L(90)/β_L(0) and β_H(90)/β_H(0), validation | each 1.01–1.05 |
| A1 κ_L(DCM)/κ_L(SMASH), validation | 0.45–0.60 |
| A1 κ_H(DCM)/κ_H(SMASH), validation | 0.80–0.92 |
| B1 M_event − C4, F | −1.0 to +0.3 |
| B1 M_hard − C4, F | +0.2 to +2.0; seed mean not resolved at 2 paired SD |
| B1 M_hard − C3, F | −1.5 to +1.5; not resolved |
| A1 M_hard DCM − M_hard SMASH, F | −1.5 to 0; removes 40–80 % of A1's M_hard − C4 where that exceeds 0.5 points |
| oracle − C3, F | within ±1.5 (limited by validation purity statistics) |

## Guards and stop rules

- A potential-independent multiplicative purity leaves K unchanged to 10⁻¹²
  (the identity above): otherwise the counting is wrong; stop.
- C4 and C3 per-file counts on F equal
  `results/smash_train_20261028/heldout_rn/per_file_counts.json` for A1 and
  B1: otherwise stop.
- The DCM predictions come from the checkpoint A1 uses: the hash of
  `student_sage_<seed>/checkpoints/model.pt` in the DCM `complete.json` equals
  `complete.json['dcm']['checkpoint_sha256']` of every run (checked while
  drafting); the three DCM `predictions.csv` hashes are recorded.
- Ŝ_L ≤ 0 in any replicate: stop (the model subtracts more than it counts).
- The F set and the fitting files are disjoint by construction (odd vs even
  parts); asserted.
- The same-sample null G5 at 0.70 passed in the published ladder; it is not
  recomputed per model.

## Fixed parameters

| parameter | value | reason |
|---|---|---|
| cut | frozen 0.70 per run | one variable, the correction; the purity scan showed changing the cut is not resolvable |
| bands | low [0.3, 1) GeV, high ≥ 2 GeV, reconstructed | inherited from the ladder |
| fit population | truth-bound validation files, potentials pooled | the files with a processed-event count; pooling is what a data analysis can do |
| model set | M_event, M_hard, oracle | M_event is the simplest data model, M_hard follows the design evidence, the oracle bounds both |
| bootstrap | 400 whole-file replicates, the ladder's seeds | inherited |

## Execution

Local only. New script `scripts/rn_background_correction.py` reusing the
ladder helpers and the per-file counting of `reco_rn_heldout.py`; output
`results/rn_background_correction_20261029/` (summary, coefficients, replicas,
one figure of the coefficients by potential and generator, one of the
corrected ladders, both marked MC truth/reconstructed). The three DCM
validation `predictions.csv` (about 4 MB each) are copied from the frozen
campaign into `results/rn_background_correction_20261029/inputs/` with their
hashes. Estimated under 30 min CPU. A preflight review of this protocol and of
the script precedes the run.

## How the result is used

- The A1 generator ratio of κ_L (Part A) sets the size of the systematic a
  simulation-calibrated correction carries. If it moves the corrected contrast
  by more than half of the correction itself, a correction calibrated on one
  generator is not a remedy for data; background must be reduced at the
  classifier or measured from data. The design evidence (ratio about 0.5)
  points this way.
- If κ is generator-stable to better than about 10 % and potential-independent
  within its SD, the correction is worth adopting, and the B1 generator check
  (B1 inference on DCM validation, a GPU job) comes next under its own
  protocol.
- Part B either agrees with Part A within its SD or is reported as a tension;
  it does not decide on its own.

## What this does not establish

The generator comparison is A1 only, and it changes generator, beam energy
(3.26 against 2.87 AGeV) and target (CsI against Cs) together; it measures how
much the background coefficients move between two simulated productions, not
which property moves them. B1's generator dependence is not measured. One
detector simulation is used throughout. The background is not decomposed into
its sources. M_hard's assumption holds on SMASH validation by the design
evidence, and the report-set check is under-powered. Nothing here is measured
on data.
