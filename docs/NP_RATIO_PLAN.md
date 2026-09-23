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

## First look, and why it is not a physics ratio (2026-09-17)

n/p measured on the `_v3` parquets is **monotonic in U_sym** — the isovector
signature, and the first observable besides spectral hardness to order correctly
where M_n turns over. That much is robust. The magnitude is not, for a reason
that is specific to BM@N.

### The magnetic-field problem

BM@N's analysing magnet bends charged tracks. A **primary proton with kinematics
comparable to a neutron that reaches the HGND is swept out of acceptance** — the
two species are simply not sampled over the same phase space. Whatever is
recorded as a proton at the HGND is therefore either a magnetically-selected
survivor or something produced locally, and neither is comparable to the neutron
sample sitting in the numerator.

The data bear this out. Using `fMotherId == -1` for "primary" (note: **not**
`!= 0` — `0` is a valid track index; the primary flag is `-1`):

| species | primary | secondary |
|---|---|---|
| neutrons | 48.4 % | 51.2 % |
| protons | **20.3 %** | **79.2 %** |

and the contamination is **energy dependent**, with protons lagging neutrons at
every energy:

| E_kin (GeV) | 0–.25 | .25–.5 | .5–.75 | .75–1 | 1–1.5 | 1.5–2 | 2–3 | 3–5 |
|---|---|---|---|---|---|---|---|---|
| neutron primary % | 4 | 23 | 40 | 50 | 61 | 71 | 81 | 95 |
| proton primary % | 1 | 12 | 21 | 28 | 34 | 42 | 61 | 85 |
| gap (pp) | 3 | 11 | 19 | 22 | 27 | 29 | 20 | 10 |

An energy-dependent contamination distorts the *shape* of the ratio, not merely
its normalisation — which is fatal for a spectral observable.

### How much it matters

Restricting to primaries moves the ratio by a factor ~2.4 and more than doubles
the significance:

| | all recorded | primaries only |
|---|---|---|
| n/p at U = 0 | 1.2427 | 2.9672 |
| n/p at U = 18 | 1.2479 | 2.9900 |
| n/p at U = 90 | 1.2493 | 3.0334 |
| 90 vs 0 | +0.52 %, 3.5σ | +2.23 %, **7.6σ** |

**Withdrawn:** the earlier "n/p carries 2.16× the relative response of the
neutron spectrum" was computed on the all-recorded sample and is not a physics
result. The qualitative claim survives — the response is larger in n/p than in
neutrons alone, in both treatments, and monotonic in both — but no number from
HGND hits should be quoted, because numerator and denominator do not share an
acceptance.

## Measured on primaries (2026-09-23) — the signal is outside the HGND

Stage 2 delivered: `*_prim.csv` now exists, so n/p can be formed over
physics-defined phase space. Sample: 2,225 events per Spot, ~571k primaries each.

**Total production is identical across the three samples** — 4π yields agree to
0.17 % (n) and 0.24 % (p). U_sym does not change how many nucleons are made; it
changes where they go.

**n/p tilts strongly with U_sym, but only at forward angles:**

| θ (deg) | U=0 | U=18 | U=90 | 90−0 | σ | ordered |
|---|---|---|---|---|---|---|
| [0,1) | 1.1997 | 1.2937 | 1.3379 | +0.1383 | 5.8 | yes |
| [1,2) | 1.2262 | 1.2830 | 1.2947 | +0.0685 | 4.5 | yes |
| [3,4) | 1.3505 | 1.2814 | 1.2764 | −0.0741 | 4.0 | yes |
| [4,6) | 1.3837 | 1.2172 | 1.2165 | −0.1672 | **7.8** | yes |
| **[8.9,13.1) HGND** | 1.1644 | 1.1478 | 1.1811 | +0.0167 | **0.8** | **no** |

The ratio rises with U_sym below 3°, falls between 3° and 6°, crossing over near
2–3°. Integrating over a wide angular range therefore cancels it almost exactly:
4π n/p is 1.2955 → 1.2902 → 1.2940, a 0.3σ non-effect. Same lesson as the
neutron spectrum — the signal is differential, here in *angle* rather than
energy.

**Inside the HGND acceptance the ratio is flat and non-monotonic.** The band
(θ ∈ [8.9°,13.1°], derived from hit positions) sits beyond the signal region.

### Is the difference driven by protons? No.

`dR/R = dn/n − dp/p` attributes the change directly:

| region | pair | dR/R | dn/n | dp/p | proton share |
|---|---|---|---|---|---|
| 4π | 90 vs 0 | −0.12 % | −0.07 % | +0.04 % | 37 % |
| HGND band | 90 vs 0 | +1.43 % | +4.81 % | +3.32 % | 41 % |

Neither species dominates (37–59 % across pairs).

**The useful finding hides in the same table**: HGND-band *yields* change by
3–6.5 % between potentials, against 0.2 % over 4π — a twenty-fold
amplification. The band is genuinely sensitive to the angular redistribution;
the *ratio* is blind because n and p move into it together and divide out. So
the HGND acceptance is not useless for U_sym — n/p is simply the wrong
observable there. Single-species yield or shape keeps what the ratio cancels,
which is consistent with spectral hardness on neutrons alone reaching 6.4σ.

### Blocking caveat — centrality

The converter wrote `B = -1` and `NPrim = -1` for every event: the
`MCEventHeader.` branch was not found in this BmnRoot build. n/p depends
strongly on impact parameter, so a difference in the b-distribution between
samples is indistinguishable from a U_sym effect. **No n/p number here is
defensible until centrality can be matched.** The 0.2 % agreement of 4π yields
is reassuring about gross sample composition but is not a substitute.

The sample is also small and sequential (first 40 MB of the first two files per
dataset), so per-bin significances need re-measuring on a random draw.

Full output: `results/np_primaries/`.

## Staged plan

The magnetic-field argument changes the priority order: Stage 1 is no longer a
route to a physics ratio, only a diagnostic. The primaries dump (Stage 2) is now
the *only* path to the measurement, so it moves to the front.

### Stage 1 — preserve what the CSVs hold *(diagnostic value only)*

In `load_hits()`:

- keep a signed `PDG_signed` alongside the existing absolute `PDG`;
- carry `Rapid`, `Weight`, `Px`, `Py`, `Pz` through the merge, and derive
  `pT = hypot(Px, Py)`;
- **propagate `fMotherId` usefully** — it is already carried, but nothing
  downstream uses it. Every nucleon-level observable should be able to select
  primaries;
- bump `LOADER_V` so caches rebuild.

This buys rapidity- and p_T-differential *diagnostics* and a clean primary
selection. It does **not** buy an n/p measurement, for the acceptance reason
above. Cheap, and it rides a rebuild already planned.

### Stage 2 — all primary nucleons, regardless of acceptance *(the blocking item)*

This is the request to whoever runs the simulation. What is needed is **every
primary proton and neutron from the collision, written whether or not the
particle reaches any detector** — so that n/p can be formed over a phase-space
region defined by physics rather than by what survived the magnet.

`unigen` output already satisfies this in principle and exists for
`defaultSpot` (`unigen/particles_*.root`); `zeroSpot` and `bigSpot` archives
contain **zero** ROOT files. Either request the matching unigen for the other
two potentials, or produce a primaries CSV per event:

| column | meaning |
|---|---|
| `Row` | event index **within the file** |
| `Instance` | particle index within the event |
| `PDG` | **signed** PDG code |
| `E`, `Ekin` | total and kinetic energy (GeV) |
| `Px`, `Py`, `Pz` | momentum components (GeV/c) |
| `Rapid` | lab rapidity |
| `pT` | transverse momentum (GeV/c) |
| `b` | impact parameter (fm) |
| `Npart` | participant nucleons |
| `Weight` | generator weight |

Two constraints that are not optional:

- **All primaries, no acceptance filter.** The moment the writer applies a
  geometric or magnetic cut, the sample inherits exactly the bias this plan
  exists to remove. If a cut is unavoidable, it must be applied identically to
  protons and neutrons and recorded in the file.
- **`b` / `Npart` for centrality matching.** n/p depends strongly on impact
  parameter, so the three potentials must be compared in matched centrality
  bins or any difference in the sampled b distribution is confounded with the
  U_sym effect.

`Row` needs to be unique only *within a file* — the loader assigns each file its
own global block (`LOADER_V = 2`). A writer-side global numbering is what caused
the cross-file collision fixed in `9b7c6ee`.

### Stage 3 — the measurement, on a common fiducial region

- Define a **common fiducial window in (y, p_T)** and compute n/p inside it for
  both species. This is what makes the ratio meaningful: identical phase space,
  not identical detectors.
- Single ratio `R_np(y, p_T)` per U_sym, in matched centrality.
- **Double ratio** `R_np(U=90) / R_np(U=0)` — cancels generator-level and
  acceptance systematics common to both samples.
- Same significance machinery as spectral hardness, so the two observables are
  compared per event on equal footing.
- Differential, always: integrated n/p is +0.52 % / 3.5σ at hit level and
  +2.23 % / 7.6σ for primaries, while the [3,5) GeV bin alone reaches ~10σ.
  Integration destroys this signal exactly as it destroys the neutron one.

### Stage 4 — what a *measured* n/p would require

The HGND measures neutrons. Protons are measured by BM@N tracking (GEM / TOF /
DCH) **inside** the magnetic field, with an acceptance that is a strong function
of rigidity and charge sign. A measured n/p therefore needs:

1. neutron acceptance × efficiency from the HGND pipeline,
2. proton acceptance × efficiency from the tracking system,
3. a fiducial region where **both** are non-zero and well understood,
4. each corrected to the same (y, p_T) window before the ratio is formed.

That is a cross-detector analysis, outside the present HGND scope. Stages 2–3
remain worth doing regardless: a generator-level n/p sensitivity number tells you
whether that analysis is worth proposing, and is publishable on its own as a
feasibility statement.

## Order of work

1. **Request the primaries dump now** — it is the blocking item and has the
   longest lead time. Everything physics-facing waits on it.
2. Stage 1 alongside, riding the rebuild already queued; it gives primary
   selection and (y, p_T) diagnostics.
3. Stage 3 once the primaries arrive.
4. Keep spectral hardness R as the headline observable meanwhile. It is
   measurable by the HGND alone, monotonic, and at 6.4σ on the reconstruction
   target — no acceptance caveat attached.
