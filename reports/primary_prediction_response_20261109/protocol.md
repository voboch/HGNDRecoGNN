# Direct-primary response after the frozen score cut

Registered on 9 November 2026 before calculating prediction outcomes. The
study uses corrected SMASH Xe+Cs at 2.87 GeV per nucleon, S286 truth-bound even
parts 0/2/4, the accepted nominal geometric window and T1 processed span.
The analysis applies no impact-parameter selection. It reads the accepted
4491592 direct edges and original TEST-fold predictions for A1 and B1,
with A0 and B0 supplying the corresponding raw energy from the same head.
One local CPU pass suffices. Score cuts, calibrations, split, seeds 42, 123
and 456 and the 400 existing stratified physical-file draws are frozen.

Join within content on (Row, ClusterID, cl_istop). Verify the complete candidate
key digest, native labels, native cluster targets and sizes against the accepted
native feature ledger. Verify all input file hashes and same-head scores. The
two arms share the candidate set. Use the strict original score comparison.

The score denominator consists of distinct in-window T1 primary neutrons with
at least one native qualifying direct edge. Count each primary once if any
such edge leads to a cluster passing the score cut. Preserve split and merge
multiplicity in separate counts, including linked objects outside the nominal
window when computing degrees. Descendant associations are unavailable.

For the energy comparison, require the qualifying edge to have degree one
at both endpoints in the full observed direct-neutron graph, before geometry
and score selection. This fixes a common structural subset and rules out
assigning an arbitrary cluster or primary in a split or merge. Report the
fraction of the native-linked population excluded by this rule, by band.
Energy residuals use initial MC primary kinetic energy. Compare calibrated
and raw energy on the same selected objects within each pipeline and compare
A/B energy errors on the intersection selected by both. This is a comparison
of the accepted pipelines with their respective preprocessing and calibrations.

Preserve the inherited low [0.3,1) and high [2,infinity) GeV bands. For the
isolated subset, compare true counts before score selection, true counts after
score selection, and predicted counts after raw and calibrated energy migration
on the same selected objects. Keep all five migration states, including
negative predictions and the middle band, and check row and total conservation.
Compare 18/0, 90/0 and 90/18 within the same replica at each rung, as an
internal diagnostic of this subset. Use paired file-bootstrap SD and report
seed spread separately. These associations cover the direct-neutron path
and the structurally isolated energy subset. A detector-wide correction
requires independent closure and coverage of the unresolved response.
