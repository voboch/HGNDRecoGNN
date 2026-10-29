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

(to be filled after the run)
