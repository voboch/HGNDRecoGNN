# Plan — neutron/proton ratio as a U_sym probe

## Why

The MC-truth study (`results/v3_mc_truth/`) showed that **neutron yield alone
barely distinguishes the symmetry potentials**: integrated multiplicity M_n
moves ~0.5 % and is *non-monotonic* in U_sym (18 MeV sits above 90 MeV), so it
cannot measure the parameter at any statistics. Only the spectral shape carries
signal — hardness R = N(E>2 GeV)/N(E<1 GeV) reaches 10σ for 90 vs 0 MeV.

The n/p ratio attacks the same physics from the side where the effect is
largest. The symmetry potential is *isovector*: it pushes neutrons and protons
in opposite directions. A neutron-only observable sees one arm of that and
partially cancels it against the isoscalar flow; the ratio sees the difference
of the two arms and adds them. This is why n/p (and its double ratio between
systems) is the standard symmetry-energy probe in heavy-ion work, rather than
a neutron spectrum on its own.

Concretely, the expectation to test: **the U_sym response of n/p should exceed
that of any neutron-only observable on the same events**, and — unlike M_n — it
should be *monotonic* in U_sym, because the isovector force has a definite sign.

## First look — the prediction holds (2026-09-17)

Measured on the `_v3` parquets (`results/v3_mc_truth/np_probe.txt`), with the
hit-level caveat of blocker 3 below:

| U_sym | n/p (integrated) |
|---|---|
| 0 MeV | 1.2427 ± 0.0012 |
| 18 MeV | 1.2479 ± 0.0013 |
| 90 MeV | 1.2493 ± 0.0014 |

**n/p is monotonic in U_sym** — the isovector signature, and the first observable
besides spectral hardness to order correctly where M_n turns over.

The amplification is the striking part. In E_kin ∈ [3,5) GeV, where the signal
lives:

| observable | ratio 90/0 | relative effect | significance |
|---|---|---|---|
| neutron yield alone | 1.0759 | +7.6 % | 11.6σ |
| **n/p** | 1.1641 | **+16.4 %** | 9.9σ |

n/p carries **2.16× the relative response** of the neutron spectrum — the two
arms adding rather than partly cancelling, as the isovector argument predicts.
Its significance is marginally lower (9.9σ vs 11.6σ) only because proton
counting adds statistical noise; the larger effect wins as statistics grow.

Same lesson as everywhere else in this study: *integrated* n/p is only +0.52 %
at 3.5σ, while the same ratio in one energy bin is +16.4 %. Integration destroys
this signal too — any n/p analysis must be differential.

This is measured on HGND-hit protons, so it is a detector-response ratio, not
the physics n/p (blocker 3). It is evidence the observable is worth the work,
not a result to quote.

## What blocks it today

Three separate problems, in increasing order of cost.

### 1. Charge sign is destroyed (`data/graph_dataset.py`)

```python
mcpdf['PDG'] = np.abs(mcpdf.PDG)
```

Protons and antiprotons fold together, as do π⁺/π⁻. Antiprotons are negligible
at 2.87 GeV so the proton count is nearly unaffected — but the operation is
wrong in principle and makes every charge-differential observable unavailable.

### 2. The MC columns needed for a differential ratio are dropped

The hits↔truth merge keeps only:

```python
mcpdf[['Row','Instance','Id','fMotherId','PDG','Ekin','Side','X','Y','Z']]
```

`Rapid`, `Weight`, and `Px,Py,Pz` exist in `*_vacs.csv` and are discarded. Without
rapidity and momentum there is no n/p versus (y, p_T) — which is the form the
observable is defined in, since the effect is strongly rapidity-dependent.

### 3. The parquet is hit-aligned, so "protons" are the wrong sample

`*_vacs.csv` records the particle that produced each *hit in the HGND*. The HGND
sits behind absorber and is built to reject charge, so protons appearing there
are a heavily biased, detector-response population — not emitted protons. A
ratio built from them measures the detector, not the source.

**This is the binding constraint.** Fixing 1 and 2 yields a legitimate
detector-level ratio and a useful systematic handle, but the physics n/p needs
primary-particle information that the current CSVs do not contain at all.

## Staged plan

### Stage 1 — preserve what the CSVs already hold *(no new simulation)*

In `load_hits()`:

- keep a signed `PDG_signed` column alongside the existing absolute `PDG`
  (backward compatible — nothing downstream changes meaning);
- carry `Rapid`, `Weight`, `Px`, `Py`, `Pz` through the merge into the parquet,
  and derive `pT = hypot(Px,Py)`;
- bump `LOADER_V` so existing caches invalidate and rebuild.

Cost: one rebuild (~8 h/dataset, already budgeted). Delivers hit-level n/p vs
(E_kin, y, p_T) — a detector-response ratio, worth having as a systematic
cross-check and a first look at whether the ordering is monotonic.

### Stage 2 — obtain primary-particle output *(needs the simulation team)*

The physics ratio needs one row per **primary particle per event**, not per hit.
Two routes:

- **`unigen` ROOT files.** The `defaultSpot` archive already ships these
  (`unigen/particles_*.root` plus `unigen_*.tar.gz`); `zeroSpot` and `bigSpot`
  do **not** — their manifests contain zero ROOT files. Request the matching
  unigen output for the other two potentials.
- **A primaries CSV**, if that is cheaper to produce. Requested schema, one row
  per primary particle:

  | column | meaning |
  |---|---|
  | `Row` | event index **within the file** (see note below) |
  | `Instance` | particle index within the event |
  | `PDG` | **signed** PDG code |
  | `E`, `Ekin` | total and kinetic energy (GeV) |
  | `Px`, `Py`, `Pz` | momentum components (GeV/c) |
  | `Rapid` | rapidity in the lab frame |
  | `Weight` | generator weight |
  | `b` | impact parameter (fm) |
  | `Npart` | participant nucleons |

  `b` / `Npart` are not optional: n/p depends strongly on centrality, so the
  three potentials must be compared in matched centrality bins or the
  comparison is confounded by any difference in the sampled impact-parameter
  distribution.

  **Note on `Row`:** it only needs to be unique *within a file* — the loader now
  assigns each file its own global block (`LOADER_V = 2`). Do not attempt a
  global numbering in the writer; that is what produced the cross-file collision
  fixed in `9b7c6ee`.

### Stage 3 — the measurement

- Single ratio `R_np(y, p_T) = N_n / N_p` per U_sym, in matched centrality.
- **Double ratio** `DR = R_np(U=90) / R_np(U=0)` — cancels generator-level and
  acceptance systematics common to the two samples, which is why the literature
  quotes it rather than the single ratio.
- Run the same significance machinery used for R, so the two observables are
  compared on equal footing: significance per event, and events needed for 5σ.
- Decision criterion: n/p replaces R as the headline observable only if it
  beats it in σ per event **and** is monotonic in U_sym.

### Stage 4 — reconstruction reality

The HGND reconstructs neutrons. Protons are measured by BM@N tracking
(GEM/TOF), not by this detector. So a *measured* n/p is a cross-detector
analysis outside the HGND pipeline's present scope. That does not devalue
Stages 1–3 — a generator-level n/p sensitivity number sets the physics reach
and tells you whether a cross-detector analysis is worth proposing — but the
report must not imply the HGND alone can deliver n/p.

## Order of work

1. Stage 1 now — cheap, reversible, and rides the rebuild already planned.
2. Request unigen for `zeroSpot` / `bigSpot` in parallel — it is a lead-time
   item and blocks the physics result.
3. Stage 3 once either arrives.
4. Keep R (spectral hardness) as the headline observable until n/p is
   demonstrated to beat it. It is measured, monotonic, and already at 10σ.
