# Methodological preflight: bounded saved-output response diagnostics

06.11.2026. Read-only inspection of `protocol.md`, `analyze.py`,
`input_pins.json`, the inherited correction report and the accepted split.
Only this requested review report was written; no calculation, test, model
call, cluster action or Copilot invocation was performed by the critic.

**Verdict: GO** for the registered local S286 reduction. No methodological
execution blocker was found. GO does not establish transfer to data or an
independent production, and does not authorise a window/model search.

## Progress gate

- Latest scientific result: ideal-proton hybrid diagnostic, 05.11.2026.
- Latest training: no-floor models finished 04.11.2026; parent supplied the
  current actual Slurm receipt check, all 18 segments COMPLETED/0:0 and an
  empty campaign queue. The critic did not independently contact Slurm.
- Subsequent work is scientific saved-output reduction and reporting. This
  phase is not an infrastructure-only continuation.

## Evidence and estimator

1. The sample is the 286 truth-bound even-part contents, scored by the
   original TEST fold. The accepted split has validation/test-even roles
   2/0, 4/2 and 0/4 for folds 0, 1 and 2. The same-fold coefficient fit and
   application sets are therefore disjoint. This remains retrospective
   internal cross-fitting; those files and that correction form were already
   inspected, so the result is not a new holdout.
2. Candidate comparisons use Row/ClusterID/cl_istop and then assert native
   labels, targets and cluster sizes. The original CSV SHA and C1–C4 counts
   are checked before the new reduction. A1/B1 disagreement is a comparison
   of two pipelines, not an isolated weights effect.
3. Raw A0 energies are evaluated with the retained A1 selection and raw B0
   with B1 selection. Identical scores and candidate keys are asserted.
   This is the appropriate saved-output calibration control; ordinary A0 at
   its different native cut would change two quantities at once.
4. The stage ordering is T1,C1,C2,C3,C4,raw_C3,raw_C4,M_hard. High/low counts
   are pooled before ratios. Adjacent C1–C4 contrast differences telescope
   to C4−T1 and are given paired file-bootstrap SDs.
5. M_hard retains one inherited form, c=(NL−SL)/NH, pooled across validation
   potentials per fold/seed/arm. Every central coefficient must reproduce
   the accepted 29 October coefficient. No per-potential correction is fit.
   The selected hard yield and high-band background are not corrected by
   this low-band subtraction.
6. A single 400×content weight matrix, stratified by potential×even part,
   is reused across seeds, arms, rungs, neutron/proton counts and both roles
   of a physical file. The coefficient is recomputed inside each replica
   using the same weights as its other-fold application. Thus covariance
   from reusing a physical content in different roles is retained rather
   than replaced with independent coefficient/application resampling.
7. Shapes and axes are consistent: validation sums are draw×seed×arm×count;
   application sums are draw×seed×arm×rung×high/low. File SD is over draws.
   The correction subtracts each fold's coefficient times that fold's hard
   application count. Negative file contributions are included linearly;
   no clipping is applied.
8. The ideal-proton hybrid uses the same physical files and joint weights.
   dK_D/dln(eta_p)=−(100+K_D) is the correct local response slope. The stored
   division by 100 expresses points per one-percent log-response change;
   it must be labelled as a unit conversion, not an assumed or measured
   one-percent proton systematic.

## Non-blocking minimum improvement

**M1 [S]: preserve the invalid-draw count if a corrected denominator fails.**
The current assertion aborts before publishing a result, which is safe.
If it fails, the protocol also asks for the number of invalid aggregate
denominators. Writing the affected indices/count before that assertion would
make this failure diagnostic complete. No redraw, clipping, selective draw
deletion or new recovery mechanism is justified.

## Limits for the resulting report

Show all three training seeds; they are different weights on one production,
not independent productions. File SD is conditional on these weights and
frozen settings. If a mean A1/B1 improvement across trained weights is claimed,
add the three paired training-seed differences and their SE/t(df2); per-seed
descriptive plots alone must not be called evidence of a resolved mean gain.

A correction with a closer central contrast is not automatically useful:
retain M_hard−C4 and M_hard−T1 with coefficient uncertainty and report the
remaining migration/selection contribution. Do not infer primary efficiency,
ancestry, material background, causal scaler effects or proton tracking/PID
from the available cluster labels and CSV fields. Those portions of the
broader plan remain explicit unavailable-input work.

Expected reviewed cost: one local CPU pass, no GPU allocation or new jobs,
bounded by the specified 20-minute execution cap. This result feeds the
response attribution and correction-feasibility sections of the status
report, without promoting M_hard to a transferable data correction.
