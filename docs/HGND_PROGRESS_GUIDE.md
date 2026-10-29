# HGND: current evidence and next gates

Status: 2026-10-29. The dated [21–29 October progress report](../reports/HGND_progress_20261021_20261029.md)
includes nine evidence figures and the current experiment status. Slurm was
checked at 13:47:57 MSK; the material cutoff is commit 020e766 at 14:09.
v10 corrections and the completed reach-window replay are recorded in
[the v10 action record](../reports/v10_actions_20261026.md). Previous finite work is recorded in
[the v9 decision record](../reports/v9_actions_20261024.md). Older conclusions
are preserved in [the dated guide](archive/HGND_PROGRESS_GUIDE_20260927.md)
and must not be treated as current acceptance decisions.

- Scope A: primary-truth spectral response plus bounded reconstruction
  diagnostics. No reconstructed symmetry-potential sensitivity or unfolded
  neutron spectrum is established.
- DCM: 197,027 represented retained-hit events, 79,324 distinct contents. Training
  has 2209 distinct contents out of 64138 events; validation 41808/41808;
  evaluation 35307/91081. Exact full retained-hit identity does not cross source
  directories. Generator collision lineage remains unverified.
- The student pickle is exactly the multiset union of these retained-hit
  contents.23,880/36,343 test candidate inputs/edges recur in training. AUC on
  seen candidates is 0.994–0.997, on unseen 0.934–0.947; neither stratum is a
  freshly designed independent test. Prior≈0.98 generalisation attribution is
  withdrawn. No numerical causal apportionment of the entire AUC gap is made.
- SMASH: one full CPU census 4440582 completed 0: 0 in 38: 16, after smoke 4440572.
  All 568 primary files and 2,345,993 represented events were checked. No exact
  primary repeats within/across files or potentials, including vertex-excluded
  identity. PRNG/configuration lineage remains an owner question.
- The rectangular window used for truth ratios omits module gaps and inactive
  regions. It is an analysis phase-space window, not full detector acceptance
  or charged-particle transport. For the certified original E<1GeV variant, its neutron90/0 response is+4.39±0.46%; double ratio+10.49±0.72% (whole-file bootstrap SD). The complete certified exact[0.3,1)GeV primary aggregate4453896 gives neutron90/0 +4.71±0.58%, proton−6.16±0.56% and double ratio+11.59±0.89%. These SDs take the larger of whole-file bootstrap and event-clustered compound-Poisson estimates. Neutron18/0 is+0.70±0.57%, unresolved. The disjoint outside-window multiplicity-proxy control shifts the central hardness contrasts by<0.01percentage points, with paired file-bootstrap SDs recorded separately. No original-window number is relabelled.
- Six diagnostic-network checkpoints are selected before the late step-rate
  decreases. Their schedule comparison does not test those decreases.
- The earlier corrected-SMASH frozen transfer remains a retrospective diagnostic
  without fitting on SMASH. Its 432 scored files differ from the later full
  training census. The archived four-epoch +2.87±6.92% result stays inconclusive.
- A separate corrected-SMASH training campaign is now complete: nine accepted
  outcomes (three folds × three seeds, 30 epochs), including three verified
  resume retries after root cancellations. Evaluations 4466339/4467086/4467091
  were accepted on 192/189/187 test files. All 568 truth-bound files were scored
  out of fold; the data had been seen in earlier transfer, so this is not a blind
  holdout. See [the results](../reports/smash_train_results_20261029.md).
- On that common cross-fitted population, A1 (DCM) has AUC 0.8995±0.0073
  (training-seed SD); B1 (SMASH) has 0.9547 with seed SD <0.0001. Their paired
  AUC difference is +0.0552±0.0042 (SE over three seed labels, not file-sampling
  or between-production error). Cluster efficiency and energy MAE improve too.
- Reconstructed hardness does not close: B1 C4 is +1.82 to +2.01% across seeds,
  with file-bootstrap SD 0.85–0.89 percentage points, against processed-range
  MC truth T1 +4.72±0.57%. Paired C4−T1 is −2.70 to −2.90 points with SD
  0.91–0.94; G2 fails in every pooled arm. B1−A1 at C4 is unresolved. B1
  passes seed stability, but fold 0 has near-zero or negative contrast.
  Better cluster classification has not established a reconstructed potential
  measurement, calibrated neutron-track response or unfolded spectrum.
- Scaler/candidate/cache changes in older comparisons do not isolate a pure
  normalisation effect. The old guide's attribution of two thirds to scaling
  and its claim that sensitivity is blocked only by statistics are withdrawn.
- All checked HGND campaign jobs are terminal. The next purity-scan draft
  (61659b5) received preflight NO-GO (5a49497) and is not adopted: its paired
  uncertainty does not cancel as assumed, low-band statistics fall sharply,
  and the proposed primary test has low expected power. Only selection-set
  uncertainty/band-purity checks were made; no alternative-cut primary
  contrast or final-set scan result is accepted. The draft is withdrawn.
- The next committed draft (020e766) studies per-band background correction at
  the frozen 0.70 cut, with model/generator dependence as its primary question.
  No accepted result of that study is included in this dated snapshot. Its
  ongoing work is separate; no new job was submitted for the progress report.

Next external gates: production run cards and generator seeds; exact active
geometry and field mapping; immutable weight/data deposit; co-author scope
and author/funding approval. Content-grouped DCM retraining and an independent
SMASH production remain separate possible experiments. Do not retune on
previously opened 988976. The manuscript remains scope A; the new SMASH
training results have not been inserted into its canonical inputs.
