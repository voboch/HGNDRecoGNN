# Cross-scoring of the SMASH-trained networks on shared files

2026-10-29, branch `smash-train-20261028`. Requested by the user after critical
review v11 (methodology M3, "Next 1"). Local analysis of published predictions; no
training or inference.

## Question

In the held-out R_n ladder, the SMASH-trained network's selection step C2 − C1 changes
sign between folds (−2.7, +2.1, −1.2 points, seed 42; χ² ≈ 11–16 on 2 df), while the
transferred DCM network, with the same weights in every fold, shows no such spread.
Does the spread come from the trained networks (their training files) or from the test
files each fold happens to hold?

## Design

Split part q ∈ {0, 2, 4} is the validation part of fold (q/2 − 1) mod 3 and a test part
of fold q/2. Its 286 truth-bound files are therefore scored by two networks of the same
arm and seed, trained on different files: the validating fold's (training-export
validation predictions) and the testing fold's (accepted evaluation CSVs). Each network
uses its own frozen cut and calibration. On identical files and events, per part, seed
and arm (B1 and, as control, A1):

- steps C2 − C1 (selection), C3 − C2 (energy), C4 − C3 (background) of the 0–90 MeV
  neutron hardness contrast, primary band, as in `scripts/reco_rn_heldout.py`;
- band selection efficiency of MC-positive clusters, ε_b(90)/ε_b(0) − 1, for the low
  ([0.3, 1) GeV) and high (≥ 2 GeV) bands, by true energy;
- the paired difference between the two networks, with a 400-replicate whole-file
  bootstrap whose draws are shared by both networks.

A1's two scorings use the same weights; they differ only in cut and calibration, so A1
measures what the fold-specific operating point alone does.

## Predictions (written before the run)

- B1: on the same files, the two networks' selection steps differ by 1–4 points in at
  least one part, beyond 2 paired bootstrap SD; the low-band efficiency ratio differs
  between them by more than 1 percentage point in at least one part.
- A1: the two scorings' selection steps agree within 1 paired SD in every part.
- If B1's networks agree on the same files instead, the fold spread comes from the test
  files and is a file-sampling effect that the bootstrap underestimates.

## Results

Run 2026-10-29 with `scripts/smash_cross_scoring.py` (24 s); output
`results/smash_cross_scoring_20261029/summary.json`. Every even-part file's two
scorings align row by row with identical truth-label rung C1. Spreads below are
SDs of 400 whole-file replicas within a part (95–96 files); differences are
validating-fold minus testing-fold scoring on the same files, paired on shared draws.

**The two SMASH-trained networks agree on identical files.** Over the three parts and
three seeds:

| quantity | max \|difference\| | max \|difference\|/SD | beyond 2 SD |
|---|---|---|---|
| selection step C2 − C1 (points) | 0.89 | 1.54 | 0 of 9 |
| energy step C3 − C2 (points) | 1.16 | 2.06 | 1 of 9 |
| background step C4 − C3 (points) | 1.12 | 1.92 | 0 of 9 |
| C4 (points) | 1.30 | 2.10 | 1 of 9 |
| low-band efficiency ratio (%) | 0.82 | 1.45 | 0 of 9 |
| high-band efficiency ratio (%) | 0.39 | 2.17 | 1 of 9 |

3 of 54 comparisons pass 2 SD, about the 2.7 expected by chance. The DCM-trained
control (same weights, fold-specific cut and calibration) has 2 of 54.

**The step varies with the files.** On the same part both networks give a similar
selection step, and it differs between parts: B1 seed 42, testing-fold scoring,
−2.88 / +0.20 / −0.98 points for parts 0 / 2 / 4 (per-part SD 1.4–1.7). Across the
three even parts this spread matches the within-part bootstrap: χ² = 1.1–3.2 on 2 df
(p = 0.20–0.57) for both scorings, all seeds and both arms, for the selection step,
the background step and the low-band efficiency.

**Where the fold-level heterogeneity comes from.** The published per-fold selection
steps (−2.7 / +2.1 / −1.2, SD about 1.0, χ² ≈ 12) combine one even and one odd part
per fold. Fold 1's even part (part 2) gives +0.2, so its +2.1 comes from its odd part
(part 3). Odd parts have only one scoring and are kept unread at part level for a later
test, so whether part 3 is an unusual file set or fold 1's network scores it unusually
is not tested here; fold 1's network agrees with fold 0's on part 2.

**Predictions.** B1 networks differing by 1–4 points beyond 2 SD on the same files:
not observed (largest 0.89 points, 1.5 SD). Low-band efficiency differing by more than
1 point: not observed (0.82). A1 within 1 SD in every part: 8 of 9 for the selection
step; part 2, seed 123 is at 1.3 SD. The third branch holds: the variation comes from
the test files, and across the shared files the bootstrap describes it.

**Consequence.** The training files do not move the selection step or the band
efficiencies beyond the bootstrap SD. The fold dependence quoted in the manuscript is
a property of the test files; the pooled loss ordering still holds for the pooled
sample only.
