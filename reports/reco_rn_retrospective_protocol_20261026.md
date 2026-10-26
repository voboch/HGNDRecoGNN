# Retrospective reconstructed R_n on corrected SMASH: protocol

Written 2026-10-26, before any reconstructed or ladder value of R_n is computed
by this code, and revised the same day after an independent preflight review
(verdict GO-with-fixes); no outcome had been computed at either point. Scope A′
of review v9, requested by the user on 2026-10-26 ("подготовь протокол
ретроспективной оценки reco-R_n на существующем corrected SMASH"). It is a
retrospective feasibility check on data that has already been scored and partly
viewed. It cannot become a measurement, a holdout result or a manuscript claim
of symmetry-potential sensitivity, and it does not change scope A of the
manuscript revision recorded in `reports/v9_actions_20261024.md`.

## Question and consumer

How much of the generator-truth neutron hardness response survives the frozen
DCM-trained reconstruction on the corrected SMASH detector production, and at
which step is it lost? The consumer is the specification of path B (new
production and retraining) in review v9: how many files per potential a
closure test needs, and which reconstruction component to repair first.

## Inputs (all read-only, hash-verified before use)

- Cluster predictions of the frozen transfer: 1,296 CSV files
  `results/smash_full_transfer_20261023/full_transfer_inputs/transfer/<tag>/seed_<s>/<member>.csv`
  (3 seeds × 432 scored detector files). Each SHA256 must equal its entry in the
  committed `full_metrics_verified.json` `input_sha256`, and the set of files on
  disk must equal the bound population. Columns used: `Row`, `cl_score`,
  `e_pred`, `cl_label`, `e_true`.
- Per-seed score cut and checkpoint: `transfer/<tag>/summary.json` `per_seed`,
  whose SHA256 must equal `full_transfer_summary_bundle.json` `summary_sha256`
  (itself pinned in `full_metrics_verified.json`). Cuts 0.465070 / 0.357785 /
  0.397705 for seeds 42 / 123 / 456; they must agree across the three
  potentials. They were selected on DCM validation run 988974; nothing is fitted
  here.
- Detector → primary binding: committed `bound_inventory.json`
  (`primary_member_map`, `analysis_pair_names`, `empty_exports`,
  `duplicate_exports`). Within a bound pair, `Row` and MC `Id` are identical by
  the binding gate (`build_smash_bound_inventory.py`).
- Generator truth: certified per-event counters of the B-free replay
  (aggregate 4452505), `results/bfree_truth_20261025/aggregate_4452505_verified/result/<tag>/`,
  loaded with `analysis.truth_provenance.load_certified` (role `acceptance`,
  full coverage, zero counter mismatches, artifact hashes). The SHA256 of each
  `proof.json` must equal `proof_sha256` in the committed
  `aggregate_4452505_verified.summary.json`. Counters used: `acc_n_hi`
  (primary neutrons in the rectangle with `E_kin ≥ 2 GeV`) and `acc_n_lo`
  (`E_kin < 1 GeV`, no lower bound).

## Population

Analysis files per potential are `analysis_pair_names` minus header-only
exports, each with a non-null primary binding: 195 (U_sym = 0), 186 (18) and
51 (90 MeV) detector files, each bound to a distinct primary file; 36 byte
aliases at 90 MeV are excluded upstream. Clusters: 1,220,632 / 1,160,155 /
322,049; MC-positive 218,410 / 208,101 / 57,959; events with candidates
594,190 / 563,287 / 156,757.

Detector exports list only events with at least one candidate, and the number
of processed events per file is unknown. Measured while writing this protocol
and in the preflight (Row metadata only, no ratio computed): the highest
detector `Row` reaches the last primary event in 195/195, 182/186 and 51/51
files; four defaultSpot files stop before the last event, three of them below
0.99 coverage (0.251, 0.388, 0.884). The
fraction of events with a candidate is 0.716–0.760 per file, and the longest run
of events without a candidate inside the covered range is at most 10, as
expected for independent events with a 0.26 empty probability. The processed
range of a file is therefore taken as primary events with
`Row ≤ max(detector Row)`. This assumes sequential processing from `Row = 0`; a
file whose trailing events produced no candidate is covered to its last
exported event only.

## Observable

`R = N(E ≥ 2 GeV) / N(E < 1 GeV)`, the manuscript definition, with endpoint
contrast `100[R(90)/R(0) − 1]` and the midpoint contrast `100[R(18)/R(0) − 1]`.
The boundaries are an inherited convention, not optimised; the denominator lies
where cluster selection efficiency is lowest, which the ladder is meant to
expose. No second window is evaluated. The archived plateau window
(`[1.8, 2.4)` versus `≥ 2.6 GeV`) was chosen on development data that contained
the network's training split (review v9 S6), was already viewed on these files
(review v9 M6), and cannot be formed at truth level from the certified counters.

## Ladder (same files; rungs differ by one step)

| rung | population | energy | provenance |
|---|---|---|---|
| T0 | primary neutrons in the rectangle, all events of the bound primary files | true `E_kin` | MC truth |
| T1 | as T0, events in the processed range | true `E_kin` | MC truth |
| T2 | as T1, events with at least one detector candidate | true `E_kin` | MC truth |
| C1 | all MC-positive clusters (native label) | `e_true` (max qualifying-hit `E_kin`) | MC truth |
| C2 | C1 with `cl_score > cut_s` | `e_true` | MC truth label, reconstructed selection |
| C3 | the same clusters as C2 | `e_pred` | MC truth label, reconstructed energy |
| C4 | all clusters with `cl_score > cut_s` | `e_pred` | reconstructed |

Steps: T1−T0 processing coverage; T2−T1 event selection by detector response;
C1−T2 change from primary neutrons in the rectangle to native positive
clusters, a package of differences that this run does not decompose: the
label requires a neutron hit with `E_kin > 0.3 GeV` (so the C1 low band is
effectively (0.3, 1) GeV, while T counts every primary below 1 GeV), a
time-of-flight coincidence within ±0.3 ns, the detector sides 5 and 6, and any
neutron, not only a primary (review v9 M7); the cluster energy is the maximum
qualifying-hit `E_kin`. C2−C1 selection; C3−C2 energy migration on the same
clusters; C4−C3 selected background; C4−T1 the total. T rungs do not depend on
the seed; C rungs are evaluated for each training seed.

## Statistics

- Whole-file bootstrap: 400 replicates (as in the B-free campaign; the Monte
  Carlo error of a 400-replicate SD is about 3.5 %, below two-significant-figure
  rounding), `numpy.random.default_rng(20261026)` (an arbitrary seed fixed
  before the run), multinomial over detector files within each potential,
  independent between potentials. One draw per potential and replicate is
  shared by every rung, seed and the injection, so step, seed and injection
  differences are paired. A drawn detector file brings its bound primary file.
  The null test draws its own replicates within half-samples after the main
  draws. Error = SD with ddof = 1. The quoted error is always the file SD.
- Seeds: reported separately. They share events, so their spread
  (max − min) is a training systematic, not a statistical error, and is not
  averaged into one. Paired seed differences of the C4 endpoint are reported
  with their own bootstrap SD.
- File-over-event factor: the ratio of the file-bootstrap SD to an event-level
  Poisson-weight bootstrap SD (400 replicates, rng seed 20261027) for the T1 and
  C4 endpoint contrasts. This is not the per-particle factor of methodology §4.1
  and has no precedent in the project; the predicted range is an estimate.
- The replicate arrays (endpoint, midpoint, per-potential R, injection) are saved
  so that any later paired comparison can be made without a rerun.
- Precision: two significant figures in the SD.

## Controls

- Same-sample null: within U_sym = 0 and within 18 MeV, files ordered as in the
  analysed list (`analysis_pair_names` minus header-only exports) are split by
  index parity; `100[R(odd)/R(even) − 1]` for T1 and C4, with the file bootstrap
  applied within halves. Expected consistent with zero.
- Injected response: in U_sym = 0 only, every MC-positive cluster with
  `e_true ≥ 2 GeV` gets weight 1.05 and every other cluster weight 1. C1 then
  rises by exactly 5.00 %. The recovery `k = Δ(C4)/Δ(C1)` per seed, paired over
  replicates, measures how much of a high-energy change on the signal path
  reaches the reconstructed observable. Background correlated with hard neutrons
  is not scaled, so its contribution to C4−C3 is not modelled by `k`. The 5 %
  size matches the truth effect; it is a convention and does not test linearity.

## Predictions (committed before the run)

The methodology critic computed exploratory versions of C1–C4 on these files
for review v9 (M6); these predictions are informed by them and are therefore
not blind. The run's value is a frozen, hash-verified computation with the T
rungs, paired step errors, seeds, controls and their SDs. Ranges apply to each
seed; the code checks every row (`PREDICTED`, `PREDICTED_GLOBAL`).

| quantity (endpoint, %) | predicted range | basis |
|---|---|---|
| T0 | +3.4 to +5.4 | the zeroSpot part equals the full truth sample; only 51 of 173 bigSpot files differ, SD of T0 − 4.39 ≈ 0.31·√(173/51 − 1) ≈ 0.48. T0 against +4.39 is a representativeness test of the 51 files (review v9 C3, M5) |
| SD of T0 | 0.5 to 0.85 | ≈ 0.65 from the same scaling |
| T1 − T0, T2 − T1 | each −0.5 to +0.5 | three truncated files; uniform candidate fraction |
| C1 − T2 | −2.0 to +1.5 | M6 A (C1) +4.14 against T ≈ +4.4 implies ≈ −0.3 |
| C1 | +3.0 to +5.5 | M6 A: +4.14 ± 1.24 |
| SD of C1 | 0.9 to 1.5 | M6 A |
| C2 − C1 | +1.0 to +3.5 | M6: +1.4 to +3.2 |
| C3 − C2 | −3.0 to +1.0, sign not predicted | M4 Δbias ≈ −10 MeV implies ≈ −1.4 |
| C4 − C3 | −3.5 to −1.5 | M6: −2.0 to −3.3 |
| C4 | +2.5 to +4.5 | M6 D |
| SD of C4 | 1.2 to 2.0 | M6 D: 1.4 to 1.7 |
| C4 − T1 | −2.0 to +0.5 | M6 D − A, assuming C1 − T1 ≈ 0 |
| paired SD of C4 − T1 | 1.0 to 1.5 | M6 D − A |
| recovery `k` | 0.5 to 0.9 | selected-signal share of the C4 numerator |
| fraction of replicates with C4 ordered R(0) < R(18) < R(90) | 0.5 to 0.8 | truth midpoint +0.58 % against a reco SD near 1 % per potential |
| C4 file-over-event factor | 1.0 to 1.6 | estimate; file-level b structure (χ²/ndf 8.2–9.2) can raise it |
| seed spread of the C4 endpoint (all seeds) | 0 to 1.5 | M6 |
| largest null `|z|` (all tests) | 0 to 2.5 | same production |

## Evaluation (inherited P1 go/no-go gates)

Inherited from the proposed P1 gates in the archived guide
(`docs/HGND_PROGRESS_GUIDE.md` at c85a0df, lines 495–512); the 25 % fraction
and the 2-SD criterion are inherited conventions. Evaluated per seed as pass,
fail or not resolvable; a gate passes overall only if it passes for all three
seeds and fails overall if it fails for any.

- G1: reconstructed (C4) and truth (T1) endpoint contrasts have the same sign.
- G2: closure. Fail if `|C4 − T1| ≥ 2` paired SD; pass if
  `|C4 − T1| + 2·SD < 0.25·|T1|`; otherwise not resolvable. With a predicted
  paired SD of 1.0–1.5 against a threshold near 1.1, G2 is expected to be not
  resolvable: a pass would need a paired SD below about 0.55.
- G3: the energy-migration step C3 − C2 is consistent with zero: fail if
  `|C3 − C2| ≥ 2` paired SD; not resolvable if 2 SD ≥ 1.4 points, the size implied
  by M4; otherwise pass. C3 − C2 moves clusters across the 1 and 2 GeV
  boundaries through resolution and bias even without a potential-dependent
  response, so a nonzero step is not by itself a differential response. The
  archived gate on the differential response on blind files is not tested by
  this run; M4 remains a critic's measurement.
- G4: ordering of the reconstructed R at 0, 18 and 90 MeV, reported
  descriptively (central ordering and the fraction of ordered replicates); the
  midpoint is not expected to be resolvable, so G4 has no pass state.
- G5: the twelve null tests (two potentials, T1 and C4, three seeds) are
  correlated, so the gate judges the largest `|z|`: below 2.5 passes, which
  keeps the family-wise false-alarm rate near 5 % for about four effectively
  independent tests; 2.5 to 3 is marginal; `|z| ≥ 3` (about 1 % for those tests)
  stops interpretation of the run.
- G6: seed stability. The seed spread of the C4 endpoint must be below the mean
  statistical SD of that endpoint, i.e. the training systematic is smaller than
  the statistical error of the result. With a predicted spread up to 1.5 against
  an SD of 1.2–2.0 the outcome is open. The paired seed differences are
  reported but are not the gate, because seeds differing beyond their paired
  error on fixed events is expected. The archived gate's second model family
  is not available; that part is untested.

## Decision rules

The primary quantity is the paired closure difference C4 − T1 with its paired
SD. What each outcome feeds:

1. Files per potential for G2: `N = 2·SD² / ((1/n₀ + 1/n₉₀)·(0.25·|T1|/2)²)`
   with the measured paired SD and the analysed `n₀ = 195`, `n₉₀ = 51` files. It
   is the equal number of files per potential at which the 2-SD interval of a
   null closure difference equals the G2 threshold, a lower bound for the
   path-B production request (review v9 S4 estimated about 520).
2. The largest in absolute value, of either sign, of the steps C2 − C1
   (selection), C3 − C2 (energy) and C4 − C3 (background), and whether it
   exceeds 2 paired SD, names the component path B examines first; a positive
   step distorts the contrast as much as a negative one. If none exceeds 2 SD,
   no single component is named.
3. `k` below 0.5 means that most of a signal-path change is diluted before C4;
   a retrained model must then be judged on `k` as well as on AUC.
4. G5 = stop invalidates every other statement of the run.
5. No outcome changes scope A of the manuscript revision, and no number from
   this run enters the manuscript.

## Execution and budget

Local CPU only (`pyg` environment), no cluster, no GPU, no training or
inference. `scripts/reco_rn_retrospective.py` runs once after this protocol and
the code are committed: it refuses to start unless the whole tracked tree is
clean, and records the commit and library versions. Unit tests use synthetic
fixtures only. Measured in the preflight: input verification 4.6 s, one pass
over the 1,296 CSV files about 10 s; the event-level bootstrap dominates;
estimated total 1.5–3 minutes and below 2 GB. Stop rules: any hash mismatch,
missing or extra file, binding inconsistency, nonfinite value or empty
denominator stops the run; no retry without a written amendment. Outputs are
written to `results/reco_rn_retrospective_20261026.partial/` and renamed to
`results/reco_rn_retrospective_20261026/` only on success; an existing
directory of either name stops the run. Contents: `summary.json`,
`per_file_counts.json`, `replicas.npz`, `figures/ladder.{pdf,png}`. The figure
shows the endpoint per rung and, below it, the paired steps with paired SDs;
both panels carry the MC-truth/reconstructed stamp, and the header states the
system, energy, potentials, the absence of impact-parameter selection and the
retrospective status. The results report is written afterwards as a separate
file.

## Risks that decide the implementation

- Event matching assumes sequential processing; T1 and T2 bracket the effect.
- The native signal label is not the primary neutron (review v9 M7); C1 − T2
  measures that change but does not decompose it.
- At 90 MeV only 51 files remain; the endpoint SD is dominated by them.

## What this does not establish

A reconstructed measurement of the symmetry-potential response; holdout
performance; efficiency relative to generated neutrons; anything about the
proton arm; centrality control (the detector exports carry no b); the model
described in the manuscript (the transfer uses the student-pipeline GraphSAGE);
a cure for DCM training repetition (2,209 distinct of 64,138 training events).
