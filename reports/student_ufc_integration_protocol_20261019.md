# Student UFC compatibility branch — 19 October 2026

This continuation prepares a separate candidate classifier/regressor branch.
It does not replace `net_default` or the simplified `thesis_*` registrations.
No training or Slurm submission is part of this local compatibility step.

The six architecture definitions are ported from the audited notebook hashes.
Their state-dict names, four pooling operations, residual blocks and energy
heads are retained. Only the already recorded syntax repairs and distinct
Python names for the three spectral gate classes are allowed. Expected parity
on identical weights/graphs is absolute error below 1e-5, a float32 engineering
tolerance. This is not a trained performance target.

Observable graph construction keeps the student's inherited Edep >0.003 GeV,
radius 5 cells, edge time difference <=2.5 ns, at least two hits per component,
32 radius neighbours and four Laplacian coordinates. These are reproduction
settings, not choices optimised by this work. Geometry normalisers 8/11/11
are the detector index maxima recorded in the notebooks. Timing is an explicit
input column; `fTime_nonoise` reproduces the original recipe and `fTime` is a
separate, labelled deployment check. No AUC gain is predicted from this port.

MC columns are optional and used only by a separate target attachment step.
All input event IDs are retained in an event inventory even when no candidate
survives. Predictions are joined across batches before event calculations.
Object metrics use maximum-cardinality one-to-one matching against distinct
eligible (event, truth ID) objects. Their denominator is explicitly limited
to the supplied eligible truth objects; generated-neutron efficiency requires
an additional generator-level truth inventory.

The compatibility checks cover: exact candidate topology/features against the
notebook builder; state/forward parity for all six architectures; inference
with all MC columns removed; empty events; candidates split across batches;
merged and duplicate truth objects; strict per-seed CSV AUC recomputation;
and rejection of duplicate or inconsistent prediction populations. Test-source
988976 remains outside this work. Original trained AUC reproduction remains
pending the student's checkpoints or complete prediction CSVs.
