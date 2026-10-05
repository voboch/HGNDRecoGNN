# Existing-data decision for Spot reconstruction

## Decision

**DECLINE the current primary-nucleon sample for n/p-based Spot reconstruction.**

The plots are valid diagnostics of the exported sample, but the sample fails the
physics-use gates below:

- impact parameter B is unavailable or invalid.
- participant count NPrim is unavailable or invalid.
- generator weights are absent and unit weighting is not documented.
- sample is a small sequential export rather than the complete/random production.
- HGND alone does not provide a common-acceptance proton measurement.

## What remains usable

The existing full HGND reconstruction outputs may be used conditionally for a
simulation-only, three-point neutron-spectrum sensitivity/ordinal-classification
study. They are not sufficient for a calibrated continuous Spot estimator or an
experimental n/p measurement. Such a study must use balanced event-level splits,
match centrality, hold out independent productions, and use HGND observables only.
use