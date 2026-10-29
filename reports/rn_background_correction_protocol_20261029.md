# Per-band background correction of reconstructed R_n: protocol

Draft 2026-10-29 (020e766), revised the same day after its preflight review
(GO-with-fixes). Branch `smash-train-20261028`. Requested by the user on
2026-10-29 after the purity scan (`reports/rn_purity_scan_protocol_20261029.md`)
was withdrawn: a per-band background correction of C4 at the frozen 0.70 cut,
with its model dependence as the main question. No training, inference or
report-set number; no number enters the manuscript.

**Revision.** The review found that the draft's Part A predictions had already
been measured while drafting, on the same validation files, and could not be
falsified; that the potential-dependence test has the same power as the
report-set check; that most of the generator ratio comes from the hard-signal
normaliser; and that the report-set check (Part B) had no consumer. The revised
scope is Part A only, reported as a measurement with uncertainties in units of
the contrast, not as a test of predictions. Part B is dropped, so the 282
odd-part files stay unread for a later decisive test. The script also ran
Part A once on validation before the review finished; nothing on the report
set was computed.

## Question and consumer

At the 0.70 cut the reconstructed contrast C4 keeps about 40 % of the truth
contrast (`reports/smash_train_results_20261029.md`); the largest loss is
C4 − C3, the background admitted by the selection (B1 −1.1 to −1.3 points,
paired SD about 0.8). How well can a simulation-calibrated background
correction be determined: how stable are its coefficients across the potential
within SMASH, and how far do they move between two simulated productions? The
consumer is the path-B decision of review v9.

## The correction

K = R(90)/R(0) − 1 with R = high-band / low-band selected count (high:
reconstructed E ≥ 2 GeV; low: [0.3, 1) GeV). A purity that is the same at every
potential cancels in K, so a usable correction must model how the background
count depends on observed quantities. The hard-cluster model subtracts from the
low band a background proportional to the selected high-band count:

  Ŝ_L = N_L − c·N_H,  c = B_L / N_H,

fitted on validation pooled over the potentials (B_L: low-band selected
clusters with the MC label 0). The high band needs no correction in K: a
constant factor cancels. To first order the corrected contrast is K̂ ≈ C4/π_L,
where π_L is the low-band purity, so the correction amplifies both C4 and its
statistical error by about 1/π_L ≈ 1.8 (B1).

Two errors in c matter:

- **Potential dependence.** If c at 90 MeV differs from c at 0 MeV by a
  fraction ε, the pooled c leaves a bias in K̂ of about
  −100·ε·(1 − π_L)/π_L points (0.79·ε·100 for B1).
- **Calibration source.** If c is taken from another production, scaled by
  r = c_other/c_SMASH, the correction K̂ − C4 shrinks by the fraction
  (1 − r)/(1 − r·(1 − π_L)), independent of the counts.

Both are first-order expressions checked against explicit computation in
`tests/test_rn_background_correction.py`.

## Measurements (validation only)

SMASH: the 286 truth-bound files of the even split parts, each scored by the
fold that validates it, at that run's frozen cut and calibration. Per-event
rates use events with at least one candidate cluster (published per-file
counts), the definition the DCM validation shares, since its predictions list
only events with clusters; the processed span would also count empty events
and inflate the DCM/SMASH per-event ratios by about 1.35. DCM: the frozen A model's own
validation (DCM-QGSM-SMM Xe+CsI 3.26 AGeV, run 988974, 41,808 events,
101,107 clusters per seed), with the A1 cut and calibration of each SMASH run.

1. Per run and per seed (three folds pooled): c, κ_L = B_L/S_H, β_L, β_H
   (background per event), π_L and N_H per event, each per potential; their
   ratios 90/0 MeV with a 400-replicate whole-file bootstrap; the K bias of a
   pooled c in points with its SD.
2. A1 per run: r = c_DCM/c_SMASH with SMASH-calibrated DCM energies (the stated
   premise: the DCM median response is within 2 % of unity with it) and with
   raw DCM energies (variant); the decomposition of r into the ratio of
   background per event and the ratio of hard clusters per event; the removed
   fraction with its SD (DCM event bootstrap × SMASH file bootstrap).
3. The number of files at which the per-seed SD of the K bias would reach a
   stated size, scaling as the inverse square root of the file count.

## Design evidence already seen (reproduction guards)

These values were computed while drafting on the same files and are therefore
guards on the inputs, not predictions:

- A1 generator ratio per run: κ_L 0.49–0.55, κ_H 0.84–0.86 (the script stops
  outside 0.45–0.60 and 0.80–0.92).
- B1 potential ratios per run (SD 0.025–0.033): κ_L 0.96–1.03, β_L 0.99–1.04,
  π_L 0.98–1.01. Per seed, three folds pooled: κ_L(90)/κ_L(0) = 1.001 ± 0.017,
  1.006 ± 0.016, 0.994 ± 0.016 (preflight).
- DCM labels have the native signature (positives E_true ≥ 0.3 GeV, positive
  fraction 16.7 % against 17.9 % on SMASH); the DCM predictions come from the
  checkpoint A1 uses (hash equal for all nine runs).

## What would change the reading

No threshold decides here; the numbers go to the path-B consumer with these
reference sizes, which come from measured quantities:

- the statistical SD of C4 on all 568 test files, 0.9 points (B1), and the
  loss to be corrected, C4 − C3 = 1.1–1.3 points;
- a calibration-source shift that removes half of the correction corresponds
  to r = 1/(1 + π_L), about 0.65 for A1; it is drawn on the figure as a
  reference line, not a cut.

## Fixed parameters

| parameter | value | reason |
|---|---|---|
| cut | frozen 0.70 per run | one variable, the correction; changing the cut is not resolvable (withdrawn purity scan) |
| bands | low [0.3, 1) GeV, high ≥ 2 GeV, reconstructed | inherited from the ladder |
| fit population | truth-bound validation files, potentials pooled | the files with a processed-event count; pooling is what a data analysis can do |
| bootstrap | 400 replicates, seed 20261026 | inherited; DCM resampled by event (no file structure) |

## Execution

Local only: `scripts/rn_background_correction.py --dcm-validation <dir>`;
output `results/rn_background_correction_20261029/` (summary, one figure
marked MC truth + reconstructed, the three DCM inputs with their hashes). The
DCM inputs are copied from
`/scratch/vbocharnikov/hgnd/experiments/student_pipeline_20261020_56bcc48206/`.
Minutes of CPU.

## What this does not establish

The generator comparison is A1 only and changes generator, beam energy (3.26
against 2.87 AGeV) and target (CsI against Cs) together; about two thirds of
the change in c comes from the larger number of hard clusters per event in the
DCM sample (preflight decomposition), so it bounds the shift between these two
productions and does not isolate the generator. B1's generator dependence is
not measured (it needs B1 inference on DCM, a GPU job). The DCM seeds share
the same events, and that validation set also fixed A's epoch and DCM cut. One
detector simulation is used throughout. Nothing is measured on data, and no
corrected contrast on the report set is computed.
