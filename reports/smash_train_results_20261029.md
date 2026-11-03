# Training on corrected SMASH: results

**Expanded report, 2026-11-03:** [full SMASH performance report](smash_performance_20261103.md)
and [PDF](../output/pdf/hgnd_smash_performance_20261103.pdf). It adds primary
n/p and cluster spectra, 23 diagnostic figure families, every frozen cut,
checkpoint/calibration selection, all potential-pair contrasts and the accepted
S286 label census. It also states the limits of fold attribution and calibration
improvements explicitly. The original 29 October results below remain preserved.

Results of the campaign defined in `reports/smash_train_protocol_20261028.md`,
2026-10-29, branch `smash-train-20261028`. Every number below comes from
`results/smash_train_20261028/results_table.json`, written by
`scripts/smash_train_results_table.py` from the verified analysis outputs
(`analysis/summary.json`, `heldout_rn/summary.json`). On the user's decision of
2026-10-29 the results enter the manuscript as Appendix B, through the paper's
canonical builder from a pinned copy of that table (protocol amendment of the
same date); the original rule that no number enters the manuscript is superseded.

<!-- table-provenance: scored on cross-fitted test files (568 truth-bound files of
     corrected, deduplicated SMASH Xe+Cs 2.87 GeV/nucleon, U_sym 0/18/90 MeV,
     split 848ebf49); each file scored only by models that never saw it; cuts and
     calibrations fitted on fold validation; the frozen DCM transfer had scored
     these files before, so this is not a blind holdout. Tables that differ say so. -->

## What ran

Nine GraphSAGE runs (three folds × seeds 42/123/456, 30 epochs, inherited
student recipe) trained on fold-train SMASH files, and three evaluations, one
per fold, each opening its test files once. The test sample is 568 files
(zeroSpot 195, defaultSpot 200, bigSpot 173), 3,572,710 candidate clusters,
640,277 of them MC-positive. Every training and evaluation export passed the
acceptance checkers (`scripts/smash_train_check_task.py`,
`scripts/smash_train_check_evaluation.py`); records are in
`results/smash_train_20261028/train_4466016/` and `evaluate_*/`. Three runs
were cancelled by a cluster process and completed by resume retries; see the
protocol amendment of 2026-10-29 and "Execution" below.

Arm A is the frozen DCM GraphSAGE (A0 with its DCM cut and raw energy, A1
with cut and energy calibration fitted on fold validation). Arm B is the
SMASH-trained network (B0 raw energy, B1 calibrated, Bcomp with the composite
epoch selector). The primary contrast is B1 against A1.

## Classification and energy response

Spread: SD over the three training seeds. Each seed's value is the mean of its
three per-fold values; AUC is computed per fold because pooling across folds
would mix three models' score scales. The score is reconstructed, the label is
MC truth; bias and MAE compare reconstructed energy with MC-truth energy on
all 640,277 MC-positive clusters.

| arm | ROC-AUC ↑ | AP ↑ | purity at cut (target 0.70) | efficiency at cut ↑ | energy bias (MeV) → 0 | MAE (MeV) ↓ |
|---|---|---|---|---|---|---|
| A0 | 0.900 ± 0.007 | 0.700 ± 0.009 | 0.699 ± 0.003 | 0.549 ± 0.010 | −49 ± 53 | 169 ± 15 |
| A1 | 0.900 ± 0.007 | 0.700 ± 0.009 | 0.701 ± 0.001 | 0.546 ± 0.015 | +20 ± 3 | 167 ± 13 |
| B0 | 0.9547 ± <0.0001 | 0.8470 ± 0.0001 | 0.702 ± 0.001 | 0.805 ± 0.001 | −8.4 ± 3.1 | 81.8 ± 0.4 |
| B1 | 0.9547 ± <0.0001 | 0.8470 ± 0.0001 | 0.702 ± 0.001 | 0.805 ± 0.001 | +7.8 ± 0.8 | 84.7 ± 0.2 |
| Bcomp | 0.9547 ± <0.0001 | 0.8470 ± 0.0001 | 0.702 ± <0.001 | 0.804 ± <0.001 | +7.8 ± 0.9 | 84.7 ± 0.2 |

B1 − A1, paired over the three seed labels (different weights: mean of the
per-seed differences, SE of the mean, t on 2 df, t_crit = 4.303): ROC-AUC
+0.055 ± 0.004 (t = 13.1), efficiency at the cut +0.259 ± 0.008 (t = 32.0),
MAE −82 ± 7 MeV (t = −11.0). All three are resolved. The seed SE leaves out
the file-sampling error the seeds share; for AUC that error is small against
the difference (about 1.2 million clusters per fold), and the per-fold
differences agree (+0.050 to +0.064 in every fold and seed).

The B arm does not depend on the potential: AUC 0.9544–0.9550 and efficiency
0.802–0.805 in every tag (fold 0). B's validation and test AUCs differ by at
most 0.0005 over the nine fold-seed pairs (A0: 0.0010), with test the higher;
the protocol reads a gap above 0.005 as a sign of leakage.

Calibration fixes B's median response, not its mean. The calibrated median
response of B1 on test stays within 0.84 % of unity in every 0.3–3 GeV bin and
seed (A1: 2.1–2.7 %, set by the 0.3–0.5 GeV bin). The same calibration moves
B's mean bias from −8.4 to +7.8 MeV and its MAE from 81.8 to 84.7 MeV, since
the inverse-median correction is not the MAE-optimal one.

## Potential-dependent energy bias

Δbias(90−0) is the difference of the mean e_pred − e_true between bigSpot and
zeroSpot MC-positive clusters in the reviewer's [1.75, 3) GeV window
(67,481 and 74,294 clusters), computed out of fold. The control is the
within-window difference of mean true energy, which is the same for every arm:
Δ⟨E_true⟩(90−0) = +3.8 MeV. Spreads: seed SD of the per-seed values, and the
SD of 400 whole-file bootstrap resamples per seed.

| arm | Δbias(90−0) (MeV) → 0 | boot SD per seed (MeV) | Δbias(18−0) (MeV) |
|---|---|---|---|
| A0 | −3.6 ± 0.5 | 1.7 | −1.3 ± 0.8 |
| A1 | −3.6 ± 0.6 | 1.8–2.0 | −1.2 ± 0.8 |
| B0 | −0.8 ± 0.1 | 0.8–1.2 | −0.3 ± 0.2 |
| B1 | −0.6 ± 0.1 | 0.9 | −0.1 ± 0.2 |

Δbias equals Δ⟨e_pred⟩ − Δ⟨E_true⟩. With Δ⟨E_true⟩ = +3.8 MeV, A's −3.6 MeV
means its mean prediction is nearly the same, +0.2 MeV, for the harder 90 MeV
spectrum inside the window; B's −0.6 MeV means its prediction follows
+3.2 MeV of the 3.8. A's Δbias is about 2 bootstrap SD from zero, B's below
1 SD. The B1 − A1 difference, +3.0 MeV, has no paired bootstrap in the
analysis; against A's 1.8–2.0 MeV it is not resolved by file resampling,
although the seeds agree closely.

The predicted A0 value, −10 ± 3 MeV reproducing review v9 M4, is not
reproduced: A0 gives −3.6 ± 1.7 MeV on these files. M4 is a critic's
measurement whose cluster population and file set are not recorded in this
repository, so the cause of the difference is not established here.

## Reconstructed hardness R_n

R_n contrast 100[R_n(90)/R_n(0) − 1] in percent, primary band
[0.3, 1) GeV denominator, pooled out of fold over 568 files. T1 is MC truth
(primary neutrons, events in the processed range); C4 is fully reconstructed
(clusters above the cut, reconstructed energy). Spread: SD of 400 shared
whole-file bootstrap resamples.

| arm | T1 (MC truth) | C4 (reconstructed), seeds 42/123/456 | C4 − T1 | C4/T1 | seed spread of C4 |
|---|---|---|---|---|---|
| A0 | +4.72 ± 0.57 | +1.97 / +1.92 / +3.20 (± 1.0–1.1) | −2.75 / −2.80 / −1.51 (± 1.1–1.2) | 0.42 / 0.41 / 0.68 | 1.28 |
| A1 | +4.72 ± 0.57 | +1.15 / +1.90 / +2.67 (± 1.0–1.1) | −3.57 / −2.82 / −2.04 (± 1.1) | 0.24 / 0.40 / 0.57 | 1.53 |
| B1 | +4.72 ± 0.57 | +1.94 / +2.01 / +1.82 (± 0.9) | −2.77 / −2.70 / −2.90 (± 0.9) | 0.41 / 0.43 / 0.39 | 0.20 |

Neither arm closes on truth: C4 − T1 exceeds 2 bootstrap SD in the pooled
sample for every arm (gate G2 "fail"). B1 retains 39–43 % of the truth
contrast, A1 24–57 % depending on the seed. B removes the seed dependence:
its C4 seed spread is 0.20 points against a statistical SD of 0.87 (G6 pass),
A1's 1.53 against 1.02 (G6 fail). The same-sample null passes in every arm
pooled (G5), so the ladder is interpretable. The E < 1 GeV denominator variant
gives C4 values within 0.02 points of these.

B1 − A1 at C4, paired on shared resamples: +0.80 ± 0.88, +0.12 ± 1.00,
−0.86 ± 1.03 points by seed (mean +0.02). Not resolved. Per fold, one of nine
fold-seed differences passes 2 SD (fold 0, seed 456, −3.75 ± 1.79), about
what nine tests produce by chance. In fold 0, C4 of every arm has the
opposite sign to T1 for at least one seed (G1 fails), and B1's C4 is
−0.57 to +0.11; folds 1 and 2 keep the sign.

Where the contrast is lost, pooled, by seed (points, with the bootstrap SD of
the step):

| step | what changes | B1 | A1 |
|---|---|---|---|
| C1 − T2 | primary neutrons → native-label positive clusters (MC truth) | −0.84 ± 0.92 | −0.84 ± 0.92 |
| C2 − C1 | score selection (MC-truth label, reconstructed selection) | −0.60 / −0.40 / −0.54 ± 0.64 | −0.62 / −0.01 / −0.21 ± 1.0–1.1 |
| C3 − C2 | reconstructed energy on the same clusters | −0.34 / −0.46 / −0.33 ± 0.43–0.48 | −0.44 / −0.51 / +0.18 ± 0.7–0.9 |
| C4 − C3 | background admitted by the selection | −1.11 / −1.12 / −1.31 ± 0.63 | −1.78 / −1.57 / −1.28 ± 1.0 |

In the pooled sample the largest step in both arms is the background admitted
at the cut. This ranking is a property of the average over folds, not of each
fold: B1's selection step C2 − C1 is −2.7, +2.1 and −1.2 points in folds 0, 1
and 2 (seed 42; the other seeds agree), heterogeneous at χ² ≈ 11–16 on 2 df,
and the background step moves the other way (−0.7, −2.5, −0.2), while A1,
with the same frozen weights in every fold, shows no such heterogeneity
(critical review v11, M3). The seeds within a fold share their training and
test files, so G6 measures initialisation stability only. C1 − T2
is a property of the native label, the same for every classifier. The 5 %
injection on true-hard clusters is recovered with k = 0.75 in B1 and
0.64–0.67 in A1.

## Predictions

Committed before training (protocol, "Predictions"). "Inside" uses the
predicted interval as written.

| quantity | predicted | measured | inside |
|---|---|---|---|
| A0 ROC-AUC | 0.90 ± 0.01 | 0.900 ± 0.007 | yes |
| B1 ROC-AUC | 0.915–0.95 | 0.9547 | no, higher |
| Δ(B1 − A1) ROC-AUC | +0.015 to +0.05 | +0.055 ± 0.004 (per seed +0.050 to +0.064) | no, higher |
| \|AUC_val − AUC_test\|, B | < 0.005 | ≤ 0.0005 | yes |
| B purity at the cut | 0.70 ± 0.01 | 0.702 | yes |
| B efficiency at purity 0.70 (A0 0.55) | 0.58–0.68 | 0.805 (A0 0.549) | no, higher |
| B0 energy bias, seed SD (A0 −48 ± 53 MeV) | \|bias\| ≤ 25 MeV, SD ≤ 20 MeV | −8.4 MeV, SD 3.1 MeV (A0 −49 ± 53) | yes |
| B0 MAE (A0 ≈ 168 MeV) | 120–160 MeV | 81.8 MeV (A0 169) | no, lower |
| B1 calibrated median response, 0.3–3 GeV | within ±2 % | within ±0.84 % | yes |
| Δbias(90−0), A0, [1.75, 3) GeV | −10 ± 3 MeV | −3.6 ± 1.7 MeV | no |
| Δbias(90−0), B | −6 to +4 MeV | B0 −0.8, B1 −0.6 MeV | yes |
| selected epochs | e_cls 2–20, epoch 0 in ≤ 1 seed; e_en 5–29 | e_cls 21–29, never 0; e_en 21–28 | e_cls no; e_en yes |
| fold-0 T1 endpoint | +2.5 to +7 % | +4.56 % | yes |
| C4/T1 retention | A 0.5–0.9; B 0.7–1.1 | A1 0.24/0.40/0.57; B1 0.39–0.43 | A1 one seed; B no |
| G2 closure | not resolvable in every arm | fail in every arm, pooled | no |
| C4(B1) − C4(A1) | 0 to +2.5 points | +0.80 / +0.12 / −0.86 | two of three seeds |

The classification and energy predictions were too cautious: B is better than
the upper end of every predicted range. The R_n predictions were too
optimistic: retention is lower than predicted for B, and the pooled sample was
large enough to show that the reconstruction does not close, where the
prediction expected the closure test to be unresolvable. The e_cls prediction
assumed B would behave like the DCM network, whose validation BCE was minimal
at epoch 0; on SMASH the validation BCE reaches its minimum at epochs 21–29,
after the learning-rate steps at epochs 10, 15 and 25, and two runs select
epoch 29 (below).

## Stop rules and checks

- No run selects e_cls = 0; the census is not sent back.
- Two runs are budget-limited, fold 0 seed 123 and fold 2 seed 42 (e_cls and
  e_comp at epoch 29). The epoch-29 minimum improves on epochs 0–24 by
  2.8 × 10⁻⁴ and 1.1 × 10⁻⁴ relative (classification) with the learning rate at
  10⁻⁶; recorded, not extended.
- A0 reproduces the stored frozen-transfer outputs on 1,287 files to
  2.3 × 10⁻⁶ in score and 1.9 × 10⁻⁶ GeV in energy.
- No test threshold or calibration was fitted on test; the evaluation checker
  confirms the cuts equal those frozen in each run's `complete.json`.

## Execution

32.0 GPU-hours on V100 (protocol budget about 30–37 GPU-hours): 19.4 h of completed training, 9.4 h of training cancelled by the
cluster and recovered through the scratch mirror, 0.85 h of three resume
retries, 2.35 h of evaluation (45–48 min per fold). A root process on the
cluster cancels jobs older than 3 h at xx:03–xx:04 each hour; it hit tasks 5,
6 and 7 at 3:06–3:12 elapsed, after 29 or 30 of their 30 epochs. The user
authorised one resume retry for each. The restored epoch checkpoints are
hash-equal to those of the cancelled runs; epoch 29 of tasks 6 and 7 was
retrained after the resume. Details: protocol amendment 2026-10-29, `HPC.md`
section 2, `results/smash_train_20261028/train_launch.json`.

## What this does not establish

The protocol's list stands: not a blind holdout, since the frozen transfer
scored these files before; not independent of the generator, since training
and test share the SMASH production, its potentials and its detector
simulation, so the B gains are an upper bound for data from another generator
or from the experiment; no centrality control; the native hit label, not the
primary neutron; efficiency relative to candidate clusters, not to generated
neutrons. The B arm's flat response across potentials is measured on the
potentials it trained on; a leave-one-potential-out control is the test of
whether it generalises in U_sym.

## For the path-B decision

Retraining on SMASH improves classification (ROC-AUC +0.055, efficiency at the
cut +0.26) and the energy MAE (−82 MeV) by resolved amounts, and removes A's
seed dependence of the reconstructed contrast. It moves the window
Δbias(90−0) from −3.6 ± 1.7 to −0.6 ± 0.9 MeV, a change not resolved by file
resampling. It does not change the reconstructed hardness contrast: B1 and A1
agree at C4 within the bootstrap SD, and both recover about 40 % of the truth
contrast. In B1 the largest loss is
the background admitted at the 0.70-purity cut (−1.1 to −1.3 points), followed
by the native-label definition (−0.84, untouched by any classifier) and the
selection (−0.4 to −0.6). The cut purity is an inherited convention, not
optimised for R_n.
