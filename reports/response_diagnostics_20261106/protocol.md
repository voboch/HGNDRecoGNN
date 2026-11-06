# Bounded response diagnostics before dE/dx production

2026-11-06. Direct human instruction: execute the proposed calculations.
This phase implements the available saved-output portions of plan1/2, one
already-defined background correction transfer, and an algebraic proton-response
requirement. No new models, inference, fitting of cuts/calibration, window scan,
odd report reads, DCM test988976 or raw detector job. No Copilot or extra unit
or test-queue jobs. All source/data assertions remain mandatory.

## Progress and execution

Latest scientific result: ideal-proton hybrid diagnostic5November2026.
Latest model jobs: accepted30-epoch no-floor models finished4November2026;
actual sacct on6November confirms all18segments4482848/4482849 COMPLETED0:0,
actual user queue empty. Work since model completion: scientific saved-output
reductions and report/policy work. The present execution adds science rather
than a recovery controller. Original source, predictions and receipts remain
read-only. No stageA or training submission is allowed by this launcher.

One local CPU process, one BLAS/OpenMP thread, wall cap20minutes, no GPU,
expected output below50MB. Cap is an execution bound, not a scientific cut.
Output result directory is new and refuses a completed-output overwrite.
Failures preserve traceback and partial evidence, with no automatic retries.

## Population and controls

Use accepted split848ebf49, only truth-bound even parts0/2/4: S286.
Read existing cross-fitted TEST predictions A0/A1/B0/B1 from evaluations
4466339/4467086/4467091. Each file uses its original test model; these are
retrospective diagnostics of already inspected files, not a fresh holdout.
Do not substitute validation-scored no-floor numbers or the full568 population.
Original frozen score cut perfold/seed, seeds42/123/456. Key is
(content,Row,ClusterID,cl_istop); native labels/targets are assertions after
key matching, never join keys. CSV SHA and unique keys are mandatory.
Replay C1..C4 exactly against saved per-file counts for every file/arm/seed.
No new threshold is selected. Low0.3<=E<1GeV, highE>=2GeV; no upper cap.
These are inherited diagnostic bands, not an optimum. True-energy display bins
are inherited from the accepted performance report with under/overflow kept;
MC-negative e_true=0 has no physical-energy interpretation.

## Outputs and predictions before execution

1. Joint A1/B1 decision counts, stratified by native label and observed cluster
   sizes1,2–3,4–7,8+ (accepted display strata). MC-positive energy-bin selection
   counts and selected true/reco band migration. No ancestry or detector
   efficiency inference from the labels. Expected: mixed pipeline changes can
   localise disagreement but cannot uniquely attribute weights/scaler effects.
2. Same-head calibration controls: raw A0 at frozen A1 cut, raw B0 at frozen
   B1 cut versus calibrated A1/B1, conditional on identical scores and keys.
   This changes only the saved energy map and keeps selection fixed. Ordinary
   A0 at its different original cut is not called a pure calibration control.
3. Paired T1,C1..C4 contrasts and adjacent-rung shifts on S286. Differences
   telescope exactly; labels identify count populations, not primary tracks.
   Expected: accepted totals replay and the stage contributions need not have
   the same sign across potential pairs. No dominant stage is assumed in advance.
4. Frozen M_hard only, inherited from29October: c=(NL−SL)/NH pooled over
   validation potentials, low_corrected=NL−c*NH; high remainsNH. Fit inputs are
   the same fold's even validation part, disjoint from its even test part.
   Coefficients must reproduce the accepted background per_run pooled c.
   No per-potential fit, choice among correction forms, or post-result tuning.
   Expected: background-only correction may move contrasts toward truth but
   cannot by construction remove all migration/acceptance/high-band background.
   No correction-size or closure target is assumed. Report M_hard−C4 and
   M_hard−T1 with paired uncertainties, including coefficient sampling.
5. Ideal-proton D on this same S286 population, then derivative
   dK_D/d ln(eta_p)=−(100+K_D), eta_p=kappa_p(a)/kappa_p(b).
   Report response sensitivity per unit fractional proton-response change.
   This is an algebraic requirement, not an assumed1% detector uncertainty,
   proton PID measurement or a strategy to cancel neutron bias deliberately.

## Bootstrap and acceptance

400whole-file draws, RNG20261106, stratified potential×evenpart. This reuses
accepted bootstrap resolution but does not reuse a previously claimed numerical
SD. Shared draws apply across arms, seeds, rungs, n/p AND the two roles of a
physical content (validation for one model, test for another). Refit the same
ratio c inside each replica to propagate its statistical uncertainty; no model
choice is refitted. Sum counts before ratios. Negative per-file subtractions
are retained; report invalid aggregate denominator draws, never clip or silently
redraw. Contrasts18/0,90/0,90/18 are correlated.

File SD is conditional on trained weights, frozen cuts/calibration and this
production. It does not include training, selection, geometry or generator
uncertainty. Show all3seeds without treating them as3independent productions.
No fitted response is promoted to transferable data correction.

A report/figure follows numerical acceptance and read-only review. Remaining
raw-primary efficiency, ancestry composition, feature/scaler projections and
actual proton tracking/PID require separate available-input inspection; this
phase must report those limits rather than fabricate missing information.
