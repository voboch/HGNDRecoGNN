# Technical preflight: saved-output response diagnostics

06.11.2026. **GO** for one local CPU execution of `analyze.py` under the
registered 20-minute cap and one BLAS/OpenMP thread. No training, inference,
Slurm submission or additional unit tests are part of this approval.

## Progress gate

- Latest scientific result: ideal-proton hybrid diagnostic, 05.11.2026.
- Latest model jobs: accepted no-floor training completed 04.11.2026.
  `cluster_check.json` records the parent's actual 06.11 check: all 18
  segments of 4482848/4482849 COMPLETED/0:0, user queue empty.
- Work since then includes scientific saved-output reductions and reporting;
  the proposed calculation adds a bounded scientific diagnostic. No
  infrastructure-only drift or new recovery mechanism was found.

## Verified code and provenance

Reviewed `protocol.md`, `analyze.py`, `input_pins.json`, the accepted hybrid
counter layout, the inherited calibration reader and background-coefficient
reader. All nine current input pins match their files. Execution is confined
to S286 even-part CSVs; complete receipts and validation NPZ SHA are checked
transitively against pinned accepted evaluation summaries. The calibration
helper `scripts/smash_train.py` is pinned directly.

The file/seed/arm/rung/band indices are consistent. C1--C4 replay guards address
the original per-file counters. Raw A0 energy is selected with the A1 cut,
and raw B0 with the B1 cut; scores, keys and truth must match, so these are
same-selection calibration controls. Candidate keys exclude labels/targets;
the stricter ordered-row equality guard rejects a reordered or altered input.

M_hard coefficients come from the same model's disjoint validation part,
pooled across potentials, and must replay the old saved coefficient. Bootstrap
weights are shared for every physical content across validation/application,
arms, seeds, rungs and neutron/proton counters. Counts are summed before
ratios, and each replica recomputes the same coefficient without selecting a
new correction. Rung indices, telescoping differences, correction differences
and the ideal-proton derivative conversion have consistent dimensions.

The planned arrays are small relative to the 50 MB output expectation. The
read volume is finite: 286 contents times three seeds times four saved CSV
arms, plus the accepted validation NPZs. The built-in SIGALRM bounds execution;
the parent must set the one-thread environment before starting Python.

## Findings closed before execution

1. The entry point now rejects any existing `execution.json` or `results.json`
   before entering the failure handler. An accidental repeated invocation
   preserves previous completed or failed evidence.
2. The imported calibration helper now has an explicit input SHA pin.
3. Invalid corrected aggregate denominators now write their values and
   draw/seed/arm indices to `invalid_denominators.json` before failing.
   They are neither clipped nor silently redrawn. Negative individual
   contributions remain in the linear sums.

Successful completion still requires the runtime SHA/key/count/coefficient
assertions to pass. A scientific failure remains a failure with preserved
evidence; this GO does not authorize an automatic repeated calculation.
File SD remains conditional on fixed trained models and this production.

## Reviewed revision

- analyze.py SHA256: `c3b8f5fcbe43e7af5df9f8fffa98a3a5bd69131c52e3724c126d8fa11dc903c5`
- protocol.md SHA256: `d7691fe6a1607970b2d01ac755000da5ef05a0f851c177bf005b3be1277adb3a`
- input_pins.json SHA256: `5ece5a55d43b5ccafc56a07fb7d17de00f20cc20c42f37182a555164c4413854`

Review was read-only except for this requested review file. The calculation,
tests, models and cluster commands were not executed by this reviewer.
