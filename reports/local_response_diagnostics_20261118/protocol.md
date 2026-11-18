# Local response diagnostics on frozen SMASH exports

18 November 2026. Authorized by the user's request to check seed stability,
misses/splits/merges versus multiplicity, and energy errors versus y and pT.
This extends §2 of the neutron-yield plan while original ROOT access is
unavailable. Thesis use: reconstruction performance and uncertainty of the
geometric diagnostic. Audience: HGND analysis group. Output budget: three
figures, one short note, complete machine-readable tables and source bindings.
No manuscript changes, new fits/inference, threshold selection, extra tests,
HPC jobs or independent-closure claims.

## Frozen evidence and uncertainty

Use accepted even.npz/even.json, rapidity scan and joint migration ledgers,
and all 286 accepted kinematics sidecars with their published member SHA.
Re-read locally stored direct-primary associations and rules edges under the
existing hash guards. Keep all 1,180,595 processed events, geometric MC truth,
raw DynamicEdgeConv, seeds 42/123/456, 70% native-label validation-purity
target and candidate energy 0.3 < T(reco) < 4 GeV. Models, scalers, folds,
preprocessing and event spans are inherited from these accepted exports.
Files have been inspected previously and this is retrospective analysis.

All comparisons use the same 400 paired whole-file bootstrap draws within
potential/part strata. Compute ratios within seed/draw before equal-seed
averaging. Export file SD (ddof=1), both Q16/Q84 bounds, invalid-draw counts,
per-seed centers and per-seed file SD. Separately give descriptive SD and
min/max of the three seed centers. Three trained pipelines do not define a
well-measured training-uncertainty distribution; no seed significance test or
quadrature combination with file SD is planned. No numerical outcome of a
new conditional analysis is assumed in advance.

## 1. Multiplicity

Use inherited MC-event categories 0/1/2/>=3 from the existing event matrix,
over all true energies. They identify geometric primary-neutron multiplicity, not
measured occupancy or entrance multiplicity. Retain zero events. Per primary,
count no selected direct link (miss), one or more (coverage), and more than
one (split). Count excess direct edges from split geometric primaries
separately; these are an ambiguity diagnostic and may involve merged showers.
Per selected candidate, count links to more than one observed direct primary
(merge), including links outside the geometric reference as in the accepted
comparison. Denominators are geometric primaries for miss/split and selected
candidates for merge, within the same event category. The zero-truth category
has undefined primary rates. Descendants are not resolved; direct misses and
merges do not constitute physical efficiency or physical background. Publish
all potential settings and pooled. Sum categories to replay every relevant
accepted per-file count, including event and no-output matrices.

## 2. Energy errors in true y and pT

Use the same common selected isolated primary-neutron links as the accepted
energy/rapidity comparison, degree one at both ends in both full observed
graphs before candidate selection, geometric truth and 0.3 <= T(true) < 4
GeV. Pair energies of both methods for each seed. Reconcile the boundary
with the old strict-0.3 energy cohort; do not silently change its membership.
The primary analysis does not require native ToF qualification. Retain its
flag for the existing secondary cohort. Bins belong to MC truth: signed NN-CM
rapidity edges -1.1..1.4 in steps of 0.1 and pT below/from 0.5 GeV/c, inherited
from the registered rapidity scan. Show all counts and undefined cells;
binning is not changed after observing errors.

Report bias and MAE in MeV, median relative error and central relative width
(Q84-Q16)/2 in percent, plus paired method differences. The width summarizes
the existing central energy response and is distinct from MAE and statistical
uncertainty. Use the existing 25-pair-per-seed convention for reporting
quantiles; identify it as inherited, retain all count/mean-error rows and mark
unavailable quantiles. Quantiles are computed within each seed and draw,
then averaged equally over all three seeds. An undefined required seed/draw
remains undefined and is never dropped from the mean. Selected reconstructed energies are bounded for both
methods; edge bias and common-selection conditioning remain limitations.
Retain integrated and marginal y/pT summaries as well as joint cells.

For seed stability repeat integrated energy errors on the intersection of
selected primary keys across all three seeds, so selection variation is
visible separately from changes in energy estimates. No outcome is used to
change the production selection. Common-within-seed and common-across-seeds
cohorts are separately named and never pooled as one sample.

## 3. Seed stability

From accepted ledgers, report centers/spreads by seed for coverage, misses,
splits, merges, energy errors, rapidity MAE and same joint (y,pT) bin rate.
Check signs of paired method differences by seed; publish disagreements as
well as agreement. For Rn reuse already accepted odd-file count ledgers at
raw70 and the frozen raw75 example, in both existing energy windows, all
three potential contrasts. Show contrast, its fixed-model file SD and paired
shift from MC by seed. The 282 odd files are already inspected; this extends
uncertainty reporting of existing outcomes and performs no new tuning.
Three-seed agreement cannot establish domain transfer or independence.
The even and odd ledgers retain their own accepted file coordinates and draws;
400 shared draws means shared within each inventory, not matched even/odd files.

## Execution and review

One local sequential reduction, expected under ten CPU minutes and 1 GB RAM;
no training or model call. Log actual elapsed time and output hashes. Check
all input hashes, candidate/primary key alignment, event spans, exact
accounting and old energy/y/joint/Rn marginal replays. Save compact per-file
counts and selected energy/key rows so numerical outputs can be reproduced
without ROOT, predictions, or a new model invocation. Hash-bound graph
assembly remains reproducible with the accepted internal exports.

Rules: project AGENTS.md; thesis-kb index and §§10,20,25,30 (12 November
revision), read in this conversation. Standalone no-ai-slop skill unavailable;
apply the inline project checklist. Preflight and final technical/methodology
critics use GPT-6 Astra / medium, read-only, with the actual source and final
plots. Preserve unrelated dirty work. Commit only completed evidence from
this local analysis and required accepted dependencies/previous local result.
