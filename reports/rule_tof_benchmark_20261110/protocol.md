# Paired event-level comparison with a rule-based ToF reference

## Input and roles

Use the 286 accepted even-part SMASH Xe+Cs files at 2.87 GeV/nucleon.
They form an already studied sample, with nominal potentials 0, 18 and
90 MeV and no impact-parameter cut. Use the frozen original SMASH GNN
evaluation-fold scores, 70% validation-purity cuts and calibrated
energies for seeds 42, 123 and 456. Count candidates from both arms of
the detector together. Compare both methods on the same exported Row
inventory in the accepted processed span, including zero-candidate
and zero-primary events. The common hit stream is the accepted
>3 MeV, finite-field, nonzero-y and 0.15 ns jitter view. Replay its
identity, including the inactive MC PDG=0 veto.

## Rule reference

Use the basic cuts from the publicly mirrored [Shabanov ICPPA 2026
presentation](https://indico.particle.mephi.ru/event/626/contributions/6331/attachments/3534/6648/Shabanov_StatusNeutronReconstructionHGND.pdf),
slide 7, corroborated by [NUCLEUS 2025](https://indico.spbu.ru/event/1/contributions/332/attachments/220/342/Shabanov_Nucleus2025.pdf),
slide 8. Build an independent spatial graph in each event and arm.
Implement the 26-neighbour spatial flood fill of
[BmnRoot FindClusters2](https://git.jinr.ru/nica/bmnroot/-/blob/d68dd57354ed9ccb6c1ea2ce8ffbe4c369d28bad/detectors/ndet/BmnNdetClusterFinder.cxx)
as connected components on the layer/row/column lattice. This variant
has no time or speed adjacency gate. Use the hit with the largest
positive time-of-flight speed from the origin as the head, with ties
resolved by hit ID. Reject clusters touching either of the front two
layers (CSV LayerId 1 and 2), with fewer than two distinct fired cells,
or with an undefined or nonphysical head. Use physical head beta < 1 and the common reconstructed-energy
interval 0.3–4 GeV (upper edge excluded) for the main comparison. The
rule's own lower cut is strict T > 0.3 GeV. The GNN applies its original
score cut followed by the same nominal energy interval, and its full
pre-window counts are checked against the archived baseline. This
retains a design-range comparison consistent with [the early detector
paper, section 4.1](https://arxiv.org/html/2412.00455v1#S4.SS1), which
uses beta < 1, and with the user's instruction after reviewing slide 6.

[XIV BM@N slides 5–6](https://indico.jinr.ru/event/5284/contributions/31175/attachments/22274/39328/Shabanov_XIV_BM_N.pdf)
show beta < 0.9 in the selection list and a populated reconstructed-energy
purity plot through 4 GeV. We do not infer the effective settings used for
that plot. Retain the literal beta < 0.9 cut as a nested descriptive
control, with an implied 1.216 GeV ToF ceiling. The reference combines
public BmnRoot connectivity and slide-inspired selections. Optional
border/remote veto and a 35 ns cut are outside this variant. The export
has at most one hit per event/arm/cell. The later ICPPA slide 6 labels its
beam 2.5 GeV/nucleon (sqrt(s_NN) = 2.87 GeV), a different setting.

The later spectrum method, ICPPA slides 8–10, uses a
separate absorber configuration, background/signal transfer,
Richardson–Lucy and a multiplicity correction. This experiment measures
candidate output before those corrections. There is no validated
paired absorber sample or independent unfolding fit in this input.

## Endpoints

Use the accepted exact direct-primary definition, PDG 2112, Mother=-1,
Side 5/6, with certified (Row,Id) and initial kinematics. Rule candidates
use the hit-to-track association after reconstruction. GNN candidates
use the accepted observed (rather than native label-conditioned) graph.
The in-window denominator is all geometric primary neutrons. Report
in-window truth degree zero and >1, candidate degree >1 using both
in- and out-of-window direct primaries, and mixed in/out merges.
Count selected candidates with an in-window direct edge, an out-only
edge, or no observed direct edge. The last category can contain
descendant-mediated responses.

Compute maximum bipartite cardinality M within the event graph using
in-window primary nodes. This gives exclusive deficits N_truth-M
and N_candidate-M, without selecting energy pairs. Report zero
multiplicity and the mean candidate count per event. Keep separate
counts for sequential, mutually exclusive rule rejection categories.
Invalid ToF is counted after the front-layer and singleton cuts. For a
secondary paired energy diagnostic require a degree-one link at both
ends of each complete preselection graph, then selection by both methods
and physical energy. Compute bias and MAE against the same initial
primary energy, and report the conditional sample count. Its denominator
differs from the end-to-end event totals.

Use the inherited truth states [0,0.3), [0.3,1), [1,2), [2,4), [4,infinity)
and calibrated candidate [0.3,1) and [2,4) counters. Sum file counts before
forming rates. Use the 400 accepted stratified whole-file draws, shared
across methods and repeats. Report the equal-seed mean, file SD and
same-replica difference. Results are conditional on a single production
and on direct associations. Candidate counts and no-link fractions do
not estimate physical purity or efficiencies at the active boundary.
