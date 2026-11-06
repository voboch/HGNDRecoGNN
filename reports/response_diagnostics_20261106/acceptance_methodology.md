# Methodological acceptance: response diagnostics

06.11.2026. **Verdict: ACCEPT** for the registered S286 retrospective
saved-output diagnostic and its stated limitations. No new methodological
blocker was found. This verdict does not approve a transferable correction,
independent closure or a new detector-level proton measurement.

The critic inspected `results.json`, `counts.npz`, `README.md` and the
reviewed analysis path. The analysis was not rerun. Independent arithmetic
checks used only the saved arrays; no model, test, SSH or cluster call was
made. Only this designated report was written.

## Numerical and covariance checks

- The saved NPZ SHA equals `results.json:counts_sha256`. Shapes are consistent:
  ledger286×3×2×6×2; validation counts286×3×2×4; shared weights400×286;
  coefficients3×3×2; bootstrap coefficients400×3×3×2; Rn replicas400×3×2×8×3.
- All physical contents have different validation and TEST folds. Their
  shared file weights nevertheless enter both roles, retaining the intended
  cross-fold covariance. The joint decision tables sum exactly to each
  content's candidate count for every seed.
- An independent scalar-loop calculation reconstructed the B1 seed42
  M_hard central Rn and all 400 replicas directly from ledger, validation
  counts, content weights and fold mappings. Both maximum discrepancies
  against the saved arrays were exactly zero. Thus coefficient resampling
  and fold-specific application were numerically checked, not inferred
  from a central-value agreement alone.
- The corresponding seed42 corrected contrasts are −2.722614%, +2.171377%
  and +5.030965% for18/0,90/0,90/18, with file SD1.975868,2.299773,2.353627
  percentage points. Digits here identify the arithmetic check; the report
  correctly uses uncertainty-matched precision.
- The execution records zero invalid aggregate denominators. No clipping,
  redraw or selective replicate deletion entered the accepted result.

## Scientific interpretation accepted

For B1 at90/0, T1≈4.492%. C4 is approximately1.239,1.443,1.317% across the
three seeds; M_hard moves the central values to2.171,2.615,2.346%.
The corrected-minus-truth residuals are −2.3201,−1.8768,−2.1460 points,
with paired file SD2.3123,2.2518,2.2866 points. The central90/0 shifts are
toward truth, but neither a resolved correction benefit nor equivalence to
truth follows from these broad conditional uncertainties.

The other pairs prevent a universal-offset interpretation. At18/0 the B1
correction moves all three central contrasts more negative, away from the
positive truth contrast. At90/18 it passes the truth central value and leaves
a positive central residual. The three pair contrasts share data and are
correlated; they are not three independent successes or failures.

For90/0 the B1 central stage contributions are C1−T1=−1.3004 points,
C2−C1=−0.9152 to−1.2181, C3−C2=−0.0359 to+0.4672, and
C4−C3=−0.6981 to−1.2403. These are paired differences between declared
count populations. They telescope, but their errors correlate. They do not
identify a statistically dominant or unique physical cause. In particular,
T1→C1 is not primary detection efficiency and A1/B1 differences are not an
isolated weights/scaler intervention.

The same-head raw/calibrated control retains the selected set. It therefore
isolates the saved energy-map change conditional on the frozen selection;
it does not establish the accuracy of a calibration transferred to data.
The proton slope remains an algebraic sensitivity unit, not a measured
systematic or justification for choosing a proton response to cancel bias.

## Required scope retained in downstream summaries

This is **S286 even-part TEST-scored baseline**, 1,798,225 candidates.
It is neither the older full568 hybrid result nor the same-content
validation-scored no-floor comparison. Their fold-specific trained models,
cuts, energy policy and/or physical populations differ. The README states
that distinction and identifies missing ancestry, primary-to-hit efficiency,
raw feature/scaler and proton tracking/PID work.

File SD is conditional on trained weights, frozen selection/calibration,
this production and the fixed correction form. All three seeds must remain
visible. A claim about a resolved average difference between trained A1/B1
pipelines would require its separate three-paired-seed analysis; no such
claim is accepted here. M_hard is a conditional internal transfer scenario
whose uncertainty is propagated, not a validated remedy for the total bias.
