# Retrospective reconstructed R_n on corrected SMASH: results

Run 2026-10-26 under the protocol frozen in commit 5abd90e
(`reports/reco_rn_retrospective_protocol_20261026.md`). One pass of
`scripts/reco_rn_retrospective.py` from a clean tracked tree: local CPU, 17 s,
peak memory 0.75 GB, Python 3.11.14, numpy 2.3.5, pandas 3.0.2. Output:
`results/reco_rn_retrospective_20261026/` (`summary.json`,
`per_file_counts.json`, `replicas.npz`, `figures/ladder.{pdf,png}`). The
summary records the protocol and code SHA256, which match the committed files.
All input hashes verified: 1,296 prediction files, transfer summaries, the
binding inventory and the three certified truth proofs.

Scope: retrospective feasibility on reused corrected-SMASH files with frozen
DCM-trained weights, no holdout, no fitting. Rungs T0–C3 use MC truth (C2
with reconstructed selection, C3 with reconstructed energy on truth-labelled
clusters); only C4 is reconstructed. No impact-parameter selection (B
unavailable). Errors are SDs of 400 whole-file bootstrap resamples; seeds share
events, so their spread is a training systematic.

## Population

195 / 186 / 51 detector files at U_sym = 0 / 18 / 90 MeV, as specified.
Events with candidates 594,190 / 563,287 / 156,757; primary events in the
processed range 804,300 / 760,955 / 211,781 of 804,369 / 767,071 / 211,796.

## Ladder: endpoint contrast 100[R_n(90)/R_n(0) − 1] (%)

| rung | provenance | seed 42 | seed 123 | seed 456 |
|---|---|---:|---:|---:|
| T0 | MC truth | +5.07 ± 0.68 | +5.07 ± 0.68 | +5.07 ± 0.68 |
| T1 | MC truth | +5.08 ± 0.68 | +5.08 ± 0.68 | +5.08 ± 0.68 |
| T2 | MC truth | +4.62 ± 0.74 | +4.62 ± 0.74 | +4.62 ± 0.74 |
| C1 | MC truth | +4.1 ± 1.2 | +4.1 ± 1.2 | +4.1 ± 1.2 |
| C2 | MC truth label, reco selection | +5.5 ± 2.0 | +6.5 ± 2.2 | +7.3 ± 2.3 |
| C3 | MC truth label, reco energy | +6.4 ± 2.0 | +4.8 ± 2.3 | +6.9 ± 2.3 |
| C4 | reconstructed | +3.1 ± 1.4 | +2.8 ± 1.7 | +4.3 ± 1.6 |

## Paired steps (percentage points, paired SD)

| step | seed 42 | seed 123 | seed 456 |
|---|---:|---:|---:|
| T1 − T0 | +0.0122 ± 0.0059 | +0.0122 ± 0.0059 | +0.0122 ± 0.0059 |
| T2 − T1 | −0.46 ± 0.32 | −0.46 ± 0.32 | −0.46 ± 0.32 |
| C1 − T2 | −0.5 ± 1.3 | −0.5 ± 1.3 | −0.5 ± 1.3 |
| C2 − C1 | +1.4 ± 1.4 | +2.3 ± 1.6 | +3.2 ± 1.8 |
| C3 − C2 | +0.9 ± 1.1 | −1.7 ± 1.3 | −0.4 ± 1.4 |
| C4 − C3 | −3.3 ± 1.4 | −2.0 ± 1.5 | −2.6 ± 1.6 |
| C4 − T1 | −2.0 ± 1.5 | −2.3 ± 1.7 | −0.8 ± 1.6 |

Midpoint contrast 100[R_n(18)/R_n(0) − 1]: T1 +0.57 ± 0.50 (MC truth), C1
+0.77 ± 0.90 (MC truth), C4 +0.13 ± 0.95, −0.85 ± 0.93 and −0.4 ± 1.1
(reconstructed, seeds 42/123/456). Paired seed differences of the C4
endpoint: 42 − 123 +0.4 ± 1.3, 42 − 456 −1.2 ± 1.3, 123 − 456 −1.5 ± 1.1.

## Gates (inherited P1; all-seed rule)

| gate | seed 42 | seed 123 | seed 456 | all seeds |
|---|---|---|---|---|
| G1 same sign of C4 and T1 | pass | pass | pass | pass |
| G2 closure within 25 % of T1 at 2 SD | not resolvable | not resolvable | not resolvable | not resolvable |
| G3 energy-migration step consistent with zero | not resolvable | not resolvable | not resolvable | not resolvable |
| G4 ordering (descriptive) | ordered; 0.54 of replicates | not ordered; 0.18 | not ordered; 0.38 | — |
| G5 null, largest `|z|` 1.9 | — | — | — | pass |
| G6 seed spread 1.54 against mean SD 1.58 | — | — | — | pass |

G2: the threshold is 0.25 × 5.08 = 1.27 points; the paired SDs of C4 − T1
(1.5–1.7) are larger than that threshold, so no central value could pass.
G3: all three energy steps are within 2 SD, but 2 SD (2.1–2.9) exceeds the
1.4-point size implied by M4, so the step cannot exclude that response. G6
passes by 0.04 points.

## Controls

- Same-sample null: twelve tests, largest `|z|` = 1.94 (zeroSpot, C4, seed 456).
  The bootstrap is consistent with zero within one production.
- Injection: a 5.00 % change of true-hard MC-positive clusters raises C4 by
  3.31, 3.17 and 3.53 %; recovery k = 0.6619 ± 0.0021, 0.6343 ± 0.0020 and
  0.7067 ± 0.0023. About two thirds of a signal-path change at high energy
  reaches the reconstructed observable. The SD is small because the same draws
  enter both arms; it does not include background correlated with hard neutrons.
- File-over-event factor: 0.84 (T1) and 0.90 / 1.04 / 0.95 (C4). The file
  bootstrap is not wider than the event-level bootstrap; no clustering by
  production file is visible at this statistics.

## Predictions

45 of 53 checks fall inside their preregistered ranges. Outside:

| quantity | value | predicted |
|---|---|---|
| C4 − T1, seed 123 | −2.32 | −2.0 to +0.5 |
| paired SD of C4 − T1, seeds 123 / 456 | 1.67 / 1.64 | 1.0 to 1.5 |
| fraction of ordered replicates, seeds 123 / 456 | 0.18 / 0.38 | 0.5 to 0.8 |
| C4 file-over-event factor, seeds 42 / 456 | 0.90 / 0.95 | 1.0 to 1.6 |
| seed spread of the C4 endpoint | 1.54 | 0 to 1.5 |

The C1–C4 central values reproduce the critic's exploratory M6 numbers on these
files exactly; this run adds the truth rungs, paired errors and controls.

## Decision outputs (protocol rules 1–5)

1. Files per potential at which the 2-SD interval of a null closure difference
   equals the G2 threshold: 446, 556 and 538 (seeds 42 / 123 / 456). The
   corrected production has 195 bound files at 0 MeV and 51 unique non-empty
   files at 90 MeV. A G2-capable path B needs roughly 450–560 unique files per
   potential, consistent with the review v9 estimate of about 520; this is a
   lower bound, because it assumes a closure difference of zero.
2. Largest component step in absolute value: C4 − C3 (background, −3.3 ± 1.4,
   beyond 2 SD) for seed 42; C2 − C1 (selection, +2.3 ± 1.6 and +3.2 ± 1.8,
   within 2 SD) for seeds 123 and 456. The selected background is the only step
   that exceeds 2 SD in any seed, and it lowers the contrast in all three. No
   single component is established across seeds.
3. k = 0.63–0.71 is above 0.5; dilution before C4 removes about one third of a
   signal-path change, not most of it.
4. G5 passes; the run is interpretable.
5. Scope A of the manuscript revision is unchanged; no number from this run
   enters the manuscript.

## Reading

- The truth contrast on the 51 bound bigSpot files and their partners is
  +5.08 ± 0.68 %, against +4.39 ± 0.46 % for all 568 primary files. The
  difference, 0.69 points, is 1.4 times the expected SD of 0.48; the 51 files
  are an acceptable but slightly harder subset.
- The reconstructed contrast keeps the sign of the truth contrast in every
  seed and retains +2.8 to +4.3 %, 54 % to 85 % of T1. The closure difference
  −0.8 to −2.3 points is within 2 SD in every seed and cannot be resolved
  against the 25 % gate at this statistics.
- Along the ladder, events without a candidate remove 0.46 ± 0.32 points;
  the change from primary neutrons to native clusters costs 0.5 ± 1.3; the
  frozen selection raises the contrast by 1.4 to 3.2 points; the energy step
  has a seed-dependent sign (−1.7 to +0.9); the selected background lowers it
  by 2.0 to 3.3 points. The selection and background steps act in opposite
  directions; their sum, not a validated closure, brings C4 near T1.
- The reconstructed midpoint (18 MeV) is not resolved: C4 gives −0.85 to
  +0.13 % against truth +0.57 ± 0.50 %, and the 0/18/90 ordering survives in
  only one seed.

## Not established

A reconstructed measurement of the symmetry-potential response; holdout
performance; efficiency relative to generated neutrons; the proton arm;
centrality control; anything about the model described in the manuscript (the
transfer uses the student-pipeline GraphSAGE trained on DCM exports with 2,209
distinct training events); exclusion of the M4 differential energy response.

## v10 interpretation correction (2026-10-26)

This addendum supersedes the unqualified interpretation in “Reading” and the
production-budget inference in decision output1. The frozen protocol, recorded
measurements, predictions and gate definitions are unchanged. These values use
the original E<1GeV denominator; they are not results for the newly adopted
[0.3,1)GeV observable.

The C4/T1 response-retention ratio is uncertain. Its bootstrap intervals are
reported in the v10 action record from the saved joint replicas; a range of
central ratios alone cannot establish a retained fraction. Selection shifts are
positive in all seeds and background shifts negative, but only C4−C3 in seed42
exceeds2pairedSD. No component effect is resolved across all seeds. G6 is a
marginal pass:1.54 versus1.58percentage points, a0.04margin and outside the
predicted spread range. Passing it does not establish stability.

The injected weight acts only on true-hard positive clusters. Recoveryk therefore
tests the numerator signal path with fixed background, not an energy-spectrum
perturbation through the denominator or correlated background. About44–50% of
the selected low-energy counts are background. There is no no-selection C0
control. These untested paths limit the interpretation ofk.

The446/556/538file estimate used the harder matched-subset truth contrast5.08%.
Using the full-production4.39% contrast gives approximately597/745/720files
per potential under the same inverse-square, zero-bias assumptions. This is a
precision estimate for a hypothetical null closure, not a production request
that guarantees G2. Observed closure offsets−2.0 and−2.3points already exceed
the1.27point subset tolerance in seeds42/123; reducing uncertainty alone cannot
make those fixed offsets pass. A response correction and an independent closure
study are required before sizing a new production. PathB remains future work.

Bootstrap retention intervals from the saved joint replicas (no independent
resampling of T1 and C4):

| training seed | median C4/T1 | central68% interval | central95% interval |
|---|---:|---:|---:|
|42|0.58|[0.33,0.87]|[0.07,1.14]|
|123|0.50|[0.18,0.85]|[−0.09,1.15]|
|456|0.81|[0.51,1.14]|[0.17,1.48]|
