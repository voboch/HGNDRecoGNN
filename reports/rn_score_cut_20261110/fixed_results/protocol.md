# Fixed score policy and neutron hardness observables

Apply the nine frozen SMASH-trained score cuts recorded in
`../choice.json` without reopening selection. The reference is the
original 70% validation-purity policy. The selected policy targets 90%
validation purity. Each fold and repeat uses the same score threshold
for all three potential settings. Scores and both saved energy maps are
held fixed. Count candidates with score strictly above the recorded cut.
Use the existing bands [0.3, 1) GeV and [2, ∞) GeV, exact band counts,
nominal front-face window and processed event spans. No impact-parameter
selection is available.

The primary sample is the 282 odd files. The 286 even files are shown
separately as selection diagnostics. The union of 568 files is
descriptive because it includes the selection sample. The odd sample
was excluded from score-policy choice and has previously reported
baseline results. No threshold, energy band, checkpoint or calibration
is refit on any sample.

For each policy, tabulate hardness ratios by potential and signed
18/0, 90/0 and 90/18 contrasts. Include all three truth populations
and the four reconstructed populations in the accepted response ledger,
with raw and calibrated energy as separate views. The primary endpoint
counts all selected calibrated candidates. Report native cluster count
purity, recall, both band populations, and the ideal-proton diagnostic
using the certified primary proton counts from the same event spans.
For the spectra retain the accepted energy bins including explicit
underflow/overflow and normalize to the processed event spans, including
events without selected candidates. These are selected-cluster spectra.

Sum integer file counts before forming ratios. Use the same 400
whole-file resamples stratified by potential and file part, shared by
both cuts, views, rungs and training repeats. The even and odd scopes replay their archived draws. Use a separate
continuous RNG stream over all strata for the pooled descriptive scope.
Within each scope the folds restrict its draws to their parts. The quoted center is the equal-repeat mean of the
three ratios or contrasts calculated separately for each repeat.
Report file SD, empirical Q16–Q84 endpoints and half-width for each
contrast and each within-replica difference. The repeat range is a
separate descriptive diagnostic. Widths here refer to sampling
distributions of contrasts and to the stability of the maximum across
score policies. The energy-band widths are fixed.

On the even selection sample only, use the saved eight-point scan
counts and its shared bootstrap to quantify the width of the maximum.
Show the signed contrast and its Q16–Q84 width, and the paired
best-minus-policy interval. Mark eligible policies whose central paired gap from the maximum is
at most one file SD of that paired gap, as a descriptive plateau on this grid. It is not a simultaneous
confidence region after maximization. The 90% threshold remains the
registered winner. Contrast divided by its file SD can be listed as
a descriptive precision comparison, with no second threshold selected.

Require exact reproduction of archived file-level baseline truth/rung
counts, prediction digests, raw/calibrated score and candidate identity,
and both-cut counts from the frozen scan. Verify fixed truth counts
against the ideal-proton records and the shared bootstrap endpoint
against the existing odd and even scan. All changes and truth gaps are
formed within each draw, with no new model calls.
