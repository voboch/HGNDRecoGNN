# HGND: current evidence and next gates

Status: 2026-10-24. Current finite work is recorded in
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
  or charged-particle transport. Its neutron 90/0 response is+4.52±0.45%;
  neutron/proton hardness double ratio+10.48±0.76% (whole-file bootstrap SD).
- Six diagnostic-network checkpoints are selected before the late step-rate
  decreases. Their schedule comparison does not test those decreases.
- Corrected-SMASH transfer is retrospective, with reused data and no new SMASH
  fitting. C1–C4 of reviewv9 remain open. The four-epoch+2.87±6.92% result
  stays inconclusive; newer weights do not replace it.
- Scaler/candidate/cache changes in older comparisons do not isolate a pure
  normalisation effect. The old guide's attribution of two thirds to scaling
  and its claim that sensitivity is blocked only by statistics are withdrawn.
- Old campaign recovery is closed. No new training, sweep or retry is pending.

Next external gates: production run cards and generator seeds; exact active
geometry and field mapping; immutable weight/data deposit; co-author scope
and author/funding approval. Any content-grouped retraining is a separately
specified future experiment. Do not retune on previously opened 988976.
