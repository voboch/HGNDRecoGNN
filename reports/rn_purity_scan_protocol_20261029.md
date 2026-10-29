# Score-cut purity and R_n closure: protocol

Draft, 2026-10-29, branch `smash-train-20261028`, written after the SMASH
training results (`reports/smash_train_results_20261029.md`) and before any
number at a cut other than purity 0.70 has been computed. Requested by the user
on 2026-10-29 as the next step on R_n closure. No new training or inference;
the analysis re-reads published, verified predictions.

**Status: withdrawn 2026-10-29 by the user's decision; superseded by
`reports/rn_background_correction_protocol_20261029.md`. Not run.**
Preflight review 2026-10-29: NO-GO. The choice rule below rests on a false premise: file
sampling does not cancel between purities, because raising the cut removes
most of the low-band population (B1 on S loses about 74 % of its low-band
clusters from 0.70 to 0.90). Measured on S only (bootstrap SDs, no contrast
values looked at), the paired SD of C4(p) − C4(0.70), B1, seed mean, is 0.44,
0.50, 0.85, 1.17, 1.77 and 3.32 points at p = 0.65, 0.75, 0.80, 0.85, 0.90 and
0.95 (A1: 0.46, 0.68, 1.37, 2.10, 3.82, 8.95); the unpaired SD of C4 is 1.1–3.8
(B1) and 1.2–14.7 (A1). With the predicted ΔC4 of +0.3 to +1.2 points the
paired one-SE rule would return the lowest p, and the primary test has about
10–30 % power even on all 568 files (paired SD about 0.8 at 0.85), so a null
would be uninformative, not negative. The review also measured the count
purity per energy band at the 0.70 cut on S, B1: 0.56 in the low band and 0.81
in the high band (0.80 and 0.93 at 0.90), so the dilution acts mostly on the
denominator. The predictions table below is kept as committed in 61659b5.

## Question and consumer

The reconstructed hardness contrast C4 keeps about 40 % of the truth contrast
T1 in both arms (B1 +1.8 to +2.0 %, A1 +1.2 to +2.7 %, T1 +4.72 ± 0.57 %), and
the largest single loss is C4 − C3, the background admitted at the cut
(B1 −1.11 to −1.31 points, A1 −1.28 to −1.78). The cut purity 0.70 is an
inherited convention that was never examined for R_n. The question: how does
the contrast recovered at C4 change when the score cut is set to a different
count purity, and at the purity chosen on validation, how close does C4 come to
T1 on files not used for the choice? The consumer is the path-B decision of
review v9: whether the remaining gap is a cut setting, or must be attacked
through background correction, the label definition or the selection.

## Why background dilution should respond to the cut

If the admitted background carries no potential contrast, a selection of count
purity π dilutes the signal contrast to about π·C3. At the 0.70 cut the
measured C4/C3 is 0.58–0.64 for B1 (pooled, by seed) and 0.39–0.67 for A1,
near or below 0.70, which is what dilution by contrast-free or opposite-sign
background gives. Raising the purity should therefore raise C4 toward C3. C3
itself (B1 +3.05 to +3.14 %, C3/T1 ≈ 0.66) bounds what any cut can recover
unless the selection step C2 − C1 or the migration step C3 − C2 also change
with the cut; the label step C1 − T2 (−0.84 ± 0.92) is independent of the cut.

## Design

**One variable:** the count-purity target p of the score cut. For each run
(fold k, seed s, arm), cut_k,s(p) is the first of the 401 thresholds
`np.linspace(0.001, 0.999, 401)` whose count purity on that fold's validation
clusters reaches p; this is the rule that produced the 0.70 cuts. The index of
that threshold is read from the run's published
`validation/threshold_scan_{A,B}.csv` and the value taken from the grid,
because the CSV prints thresholds to nine digits (a 5.6 × 10⁻¹⁷ difference
from the frozen cut in 8 of 18 run-arm pairs). Grid: p = 0.60, 0.65, …, 0.95 (eight
values). Calibrations, epochs and models are unchanged.

**Arms:** B1 (primary) and A1 (secondary), each with its own cut_k,s(p) and its
validation-fitted calibration. A0, B0 and Bcomp are not scanned.

**Two disjoint file sets** from the split (`split_manifest.json`, 848ebf49):

| set | parts | truth-bound files | scored by | role |
|---|---|---|---|---|
| S, selection | 0, 2, 4 | 286 | the fold that validates the part (validation predictions in the training export) | choose p* |
| F, final | 1, 3, 5 | 282 | the fold whose test the part is (evaluation CSVs) | report |

Part 2k+2 (mod 6) is fold k's validation and fold k+1's test, so every S file
was scored on validation by a model that never trained on it. The odd parts
are never validation. No file is in both sets and no F file's truth enters
the choice.

**Inputs for S:** `validation/predictions_{A,B}.npz` of the nine training
exports (per cluster: content, score, raw energy, true energy, label, cluster
size), with the run's calibration from `complete.json` applied to e_pred. The
npz has no event index; T1 takes the processed event span of the file from
its evaluation CSV, as in `scripts/reco_rn_heldout.py`.

**Inputs for F:** the accepted evaluation CSVs (A1 and B1, calibrated e_pred),
with cut_k,s(p) in place of the frozen cut.

**Ladder:** T1 and C1–C4 per file, primary band ([0.3, 1) GeV denominator,
E ≥ 2 GeV numerator), whole-file bootstrap with 400 replicates shared by all
p and both arms within a set, the injection and the same-sample null, all from
`reco_rn_heldout.py` and `reco_rn_retrospective.py` unchanged.

**Choice of p\*** (per arm, on S only). Closure gap g_s(p) = C4_s(p) − T1, mean
over seeds ḡ(p). Let p_best minimise |ḡ(p)| over the eligible grid. p* is the
smallest eligible p whose |ḡ(p)| − |ḡ(p_best)| is within one bootstrap SD of
that same paired difference (computed per replicate on the shared draws). The
paired SD is used because file sampling is common to all p and cancels in the
difference; the unpaired SD of C4 (about 1.2–1.6 points on 286 files, scaled
from the pooled 0.86–1.11) is larger than the expected change over the grid
and would always select the lowest p.
A grid point is eligible only if the bootstrap SD of ḡ on S is at most twice
its value at p = 0.70; beyond that the closure is too noisy to compare.

**Reported on F:** the full ladder at p* and at 0.70, the paired difference
ΔC4 = C4(p*) − C4(0.70) and the step changes with their paired bootstrap SD,
gates G1, G2, G5, G6, and the descriptive scan over all p (marked as such: it
is not used for any choice). B1 − A1 at each arm's own p*, paired on shared
draws.

## Fixed parameters

| parameter | value | reason |
|---|---|---|
| grid | 0.60–0.95, step 0.05 | brackets 0.70 on both sides; every run reaches 0.95 on validation (A1 efficiency there 0.02–0.12, B1 0.38–0.40), and A1 already falls to 0.14–0.24 at 0.90; the step is a convention |
| choice rule | paired one-SE rule | convention (Breiman et al., CART, 1984), favouring the larger sample when closure is statistically tied; the objective is plotted over the whole grid |
| eligibility | bootstrap SD ≤ 2× its 0.70 value | convention; stops the rule from comparing points whose gap is mostly noise |
| bands, resamples, injection weight, null | as in the training protocol | inherited, unexamined here |
| selection/final split | even/odd parts | the only split of the truth-bound files into validation-scored and never-validated sets |

The objective g(p) on S with its paired bands is drawn as a figure over the
whole grid, so a spike, a boundary optimum or a tie is visible.

## Predictions (committed before the scan)

Scales from the 0.70 results; F has half the files, so its bootstrap SDs are
about √2 larger than the pooled ones.

| quantity | predicted |
|---|---|
| cut_k,s(0.70) from the threshold scans | equal to the frozen cuts in all 18 run-arm pairs (checked while drafting) |
| B1 C4 − C3 on F at p = 0.90 | −0.2 to −0.8 points (−1.1 to −1.3 at 0.70, pooled) |
| B1 p* | 0.85–0.95 |
| A1 p* | 0.75–0.90 (A1 efficiency falls to 0.14–0.24 at 0.90) |
| B1 ΔC4 = C4(p*) − C4(0.70) on F, paired | +0.3 to +1.2 points, paired SD 0.3–0.6 |
| B1 C4 on F at p* | +2.0 to +3.2 %, bootstrap SD 1.4–1.9 |
| B1 C4/T1 on F at p* | 0.45–0.70 |
| B1 G2 on F at p* | not resolvable |
| B1 C2 − C1 on F at p* | −1.2 to 0 points |
| A1 G6 on F at p* | fail (seed spread above the statistical SD) |
| B1 − A1 at each arm's p*, F | −0.5 to +1.5 points, not resolved |
| G5 null on S and F | pass or marginal in every arm |

## Guards and stop rules

- cut_k,s(0.70) differs from a frozen cut in `complete.json` (exact float
  equality): stop, the scan reader is wrong.
- For every S file, the validation npz rows (label, true energy, cluster size,
  in order) equal the evaluation CSV of the same file: otherwise stop.
- Validation metrics recomputed from the npz at 0.70 with the calibration
  applied equal `validation/metrics.json`: otherwise stop.
- The F ladder at 0.70 reproduces the corresponding per-file counts in
  `results/smash_train_20261028/heldout_rn/per_file_counts.json` exactly:
  otherwise stop.
- S and F disjoint, union 568 files.
- A same-sample null with |z| ≥ 3 on S or F stops interpretation of that set.
- p* is fixed and written to the output before any F number at p ≠ 0.70 is
  computed; the script computes S completely first.

## Execution

Local only, no cluster job. New script `scripts/rn_purity_scan.py` reusing the
ladder helpers; output `results/rn_purity_scan_20261029/` (summary, per-file
counts, replicas, figures marked MC truth/reconstructed). Estimated 30–60 min
CPU (the 0.70 ladder over five arms took 6 min; this is 2 arms × 8 cuts × 2
sets). A preflight review of this protocol and of the script precedes the run.

## How the result is used

- ΔC4 resolved at 2 paired SD and C4 − C3 shrinking as predicted: background
  dilution is confirmed as a recoverable loss; the R_n analysis adopts a
  validation-chosen purity, and the remaining gap (label, selection, migration)
  is the next target.
- ΔC4 not resolved, or C4 − C3 not shrinking with p: the background loss does
  not respond to count purity, and the next step is a background correction or
  a different background model, under its own protocol.
- In either case, a C4/T1 at p* below the C3/T1 ceiling of about 0.66 shows
  how much of the gap a cut cannot recover.

## What this does not establish

The motivation for scanning came from the full test results at 0.70, which
include F, so F is held out from the choice of p* but not from the decision to
look; this is a post-hoc analysis of a published test, not a new held-out
measurement. Everything listed in the training protocol still applies (one
generator, no blind holdout, native label, no centrality control). A cut
chosen to close R_n is not chosen for classification or energy; its purity and
efficiency are reported but not optimised. Background subtraction, a
different label or an energy-dependent cut are other arms and need their own
protocols.
