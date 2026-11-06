# Native encoding comparison: numerical acceptance

06.11.2026. **GO** for the saved native-feature result and its current report.
This review independently reduces the accepted arrays; it does not repeat the
cluster job, run a model, or execute a test suite.

The parent accepted job4491447 and its archive/member hashes. The saved
summary records 286 paired files, 1,798,225 clusters, 5,833,940 hits and
32.918 seconds of analysis. Its source reports 286 exact non-x byte comparisons
and 286 candidate-key/native-target/size digest matches. The latest model
jobs completed 04.11; the current work adds scientific diagnostics on 06.11,
with no infrastructure-drift blocker.

## Independent reductions

- `features.npz` matches its saved summary SHA. Its file order matches both
  the frozen feature plan and the accepted response-diagnostic counter order;
  all 286 original TEST-fold assignments match as well.
- The reused weights have shape 400 by 286, are nonnegative integers, and
  preserve every potential by even-part stratum size. Their SHA matches the
  accepted response result and the new generated `draws.source_sha256` pin.
- Node-weighted means use total feature sums divided by total hit counts.
  B−A uncertainty is the SD of the paired difference inside the same 400
  physical-file draws, including the resampled hit-count denominator.
- Same-hit encoding RMS is the square root of summed squared A−B differences
  divided by the same hit count. Each bootstrap RMS uses those same draws
  before taking the SD. No hit is treated as an independent resampling unit.
- Independent matrix reductions reproduce every reported mean, B−A, paired
  file SD, RMS and RMS file SD; the largest absolute discrepancy is
  5.6e-17 from floating-point summation order.
- The 16-group axes agree with side/native-cluster-label/size semantics.
  All aggregate groups are populated. Their central means and within-group
  node population SD replay; the minimum variance is 0.021814, so the
  roundoff guard did not clip a negative variance. These group SDs describe
  feature spread, rather than errors on a mean.
- SurfaceHit has exactly zero same-hit squared/max difference, zero B−A,
  zero encoding RMS and zero paired/RMS bootstrap SD.

The observed eToF B−A is +0.32990585 and its encoding RMS is 0.36424537;
fTime B−A is −0.17143258 and RMS is 0.20546790. The report states that these
are pipeline-specific train coordinates, without feature-importance,
Gaussian/OOD, causal quality-gap or primary-efficiency claims.

## Presentation and provenance

The current PNG was inspected. Both panels and rotated feature labels are
readable; the legend sits above the panels without obscuring marks. The
footer now names Xe+Cs, Ekin/A=2.87 GeV, S286, no B selection and 400 whole-file
SD. The table retains the exact binary control, and prose explains that small
error bars can lie within markers. The observed eToF/fTime conclusion is
explicit and numerically supported. No presentation blocker remains.

Reviewed SHA256 values:

- features.npz: `083bf820182c8911d9366ed884eeea2fffabd8f03278b205c74e5d9960bf70be`
- response counts/weights: `f14d46c9c435a522f8e1519f594dd05c6d9f58e3f3c5f5e7b6f982264df12e9f`
- results.json: `0e31781911f1ace13de98107da1f930ae95a2851a7de7906bfef002513bab066`
- report.py: `e8f86002338dbd54e6ba49ea4d22f8deb06a4d7331b46b378f39940dbbb107fb`
- encoding_comparison.png: `b27346ff4776f5a66dbff9ac935e3bf289c0ecd143ec89b9736ae6bbdca5780e`

Only this requested acceptance note was written. Original arrays, reports,
source packages and cluster state were not modified by the reviewer.
