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

## Full-production HGND acceptance result (2026-09-25)

The acceptance study has now been repeated over **all available primary CSVs**
with file-local event identities and event-clustered uncertainties.  This
supersedes every numerical result from the earlier 2,225-event-per-Spot pilot.

| sample | files | events | particle rows | nucleons/event |
|---|---:|---:|---:|---:|
| 0 MeV (`zeroSpot`) | 195 | 804,369 | 206,615,286 | 256.8663 |
| 18 MeV (`defaultSpot`) | 200 | 824,338 | 211,744,636 | 256.8663 |
| 90 MeV (`bigSpot`) | 173 | 717,286 | 184,242,696 | 256.8609 |
| **total** | **568** | **2,345,993** | **602,602,618** | — |

The near-identical total nucleons/event confirms that these exports contain
almost the full Xe+Cs nucleon inventory, including spectators.  It is a sample-
composition check, not a produced-nucleon multiplicity measurement.

### Angular dependence and the HGND band

The full result sharpens the sign-changing angular redistribution.  Quoted
significances compare the independent 0 and 90 MeV productions using
event-clustered statistical errors only; they do **not** include centrality,
generator, or final-state-definition systematics.

| θ_lab (deg) | 0 MeV | 18 MeV | 90 MeV | 90/0 change | local σ | ordered |
|---|---:|---:|---:|---:|---:|:---:|
| [0,1) | 1.2230 | 1.2944 | 1.3099 | +7.10 % | 70.1 | yes |
| [1,2) | 1.2297 | 1.2892 | 1.3019 | +5.87 % | 97.5 | yes |
| [2,3) | 1.2613 | 1.2817 | 1.2922 | +2.45 % | 42.9 | yes |
| [3,4) | 1.3361 | 1.2749 | 1.2694 | −5.00 % | 72.0 | yes |
| [4,6) | 1.3727 | 1.2360 | 1.2061 | −12.14 % | 150.8 | yes |
| [6,8.9) | 1.1947 | 1.1939 | 1.1822 | −1.04 % | 10.1 | yes |
| **[8.9,13.1) HGND** | **1.17596 ± 0.00077** | **1.18247 ± 0.00076** | **1.17494 ± 0.00081** | **−0.087 %** | **0.92** | **no** |
| [13.1,18) | 1.1729 | 1.1783 | 1.1742 | +0.11 % | 1.13 | no |

![Full-production n/p versus polar angle with HGND acceptance highlighted](../results/np_ratio_acceptance_full/01_np_vs_theta_hgnd.png)

**Inside the HGND acceptance, n/p is flat and non-monotonic.**  The full
production gives a 90/0 double ratio of **0.99913 ± 0.00094**, consistent with
unity.  More importantly, the preliminary yield amplification does not survive:

| HGND band quantity | 0 MeV | 18 MeV | 90 MeV | 90/0 change | local σ |
|---|---:|---:|---:|---:|---:|
| neutrons/event | 5.98171 ± 0.00631 | 6.01704 ± 0.00623 | 5.99254 ± 0.00662 | +0.181 % | 1.18 |
| protons/event | 5.08668 ± 0.00557 | 5.08853 ± 0.00551 | 5.10031 ± 0.00594 | +0.268 % | 1.67 |

Thus neither the ratio nor either truth-level species yield has a significant,
monotonic Spot response in this angular band.  The pilot claim that HGND-band
yields changed by 3–6.5 % was a small, sequential-sample fluctuation and is
withdrawn.

![Full-production species-yield changes versus polar angle](../results/np_ratio_acceptance_full/02_yield_change_vs_theta_hgnd.png)

### Spectator proxy and 4π cross-checks

The same heuristic spectator proxy used in the pilot,
`pT < 0.25 GeV/c` and within 0.25 rapidity units of target or projectile
rapidity, remains strongly angle dependent.  Its fraction in the HGND band is
only 5.47 %, 5.48 %, and 5.27 % for 0, 18, and 90 MeV respectively.

![Full-production spectator-proxy fraction versus polar angle](../results/np_ratio_acceptance_full/03_spectator_fraction_vs_theta_hgnd.png)

Integrated over 4π, n/p is 1.29165 → 1.29231 → 1.29512.  The statistically
precise 90/0 increase (+0.268 %, 23.3 local σ) is not interpretable as a Spot
measurement because centrality and weights are missing.  Splitting with the
proxy makes the composition dependence explicit:

| component | 0 MeV | 18 MeV | 90 MeV | 90/0 | ordering |
|---|---:|---:|---:|---:|:---:|
| spectator-like | 1.32427 | 1.34754 | 1.35569 | 1.02372 | yes |
| remainder | 1.23431 | 1.19890 | 1.19966 | 0.97192 | no |

The monotonic forward signal is therefore still associated with the
spectator-like component under this proxy, while the remainder differs in sign
and is not monotonic.  This proxy is not a truth participant label; the large
statistical significances should not be confused with controlled physics
significances.

### Full-statistics cross-check against the reference slides

The reference remains `docs/n_p - smash check.pdf`; its 2.5A GeV panels match
this √s_NN ≈ 2.87 GeV production label, and `y_cm = Rapid − 0.9863`.  The table
compares 90/0 double ratios.  “Local pull” uses only this sample's event-
clustered statistical error; the slide values are approximate and their
uncertainties, centrality, and final-state definition are unavailable.

| region | full-production 90/0 | slides | local pull |
|---|---:|---:|---:|
| \|y_cm\|>0.5, Ekin 0.3–0.7 | 1.1133 ± 0.0025 | ~1.10 | 5.4σ |
| \|y_cm\|>0.5, Ekin 0.9–1.3 | 1.0128 ± 0.0090 | ~1.25 | 26.3σ |
| \|y_cm\|<0.5, Ekin 0.3–0.7 | 0.98735 ± 0.00079 | ~1.04 | 66.9σ |
| \|y_cm\|<0.5, Ekin 0.9–1.3 | 1.00072 ± 0.00099 | ~1.13 | 131.2σ |
| \|y_cm\|<0.5, Ekin 1.3–1.7 | 1.0310 ± 0.0016 | ~1.20 | 105.5σ |

The discrepancy is decisively not a finite-statistics effect.  The leading
unresolved explanations remain different final-state definitions (all exported
nucleons, including spectators and possibly fragment-bound nucleons, versus
free/emitted nucleons) and unmatched centrality.

### Physics-use decision

`B = -1` and `NPrim = -1` throughout the production, generator weights are
absent, and unit weighting is undocumented.  Therefore the full statistics
make the descriptive angular pattern precise but do not remove the confounding:

- **Decline primary-nucleon n/p for HGND Spot reconstruction.**
- **Decline the preliminary claim of a useful HGND-band single-species yield
  response in these primary exports.**
- Retain the forward-angle redistribution only as a generator diagnostic until
  centrality, weighting, and the free/emitted-nucleon definition are controlled.
- Do not transfer the result to experimental HGND n/p: HGND alone has no
  common-acceptance proton measurement.

Machine-readable output: [`acceptance_full.json`](../results/np_ratio_acceptance_full/acceptance_full.json)
and [`acceptance_theta_binned.csv`](../results/np_ratio_acceptance_full/acceptance_theta_binned.csv).
The authoritative reproduction plots remain in
[`results/np_ratio_smash_check_full/`](../results/np_ratio_smash_check_full/).

## Superseded pilot on primaries (2026-09-23) — historical record

> **Superseded by the 2026-09-25 full-production result above.**  This section
> used only 2,225 sequential events per Spot.  Its numerical significances and
> its 3–6.5 % HGND-band yield claim must not be quoted as final results.

### Measured on primaries — the signal appeared outside the HGND

Stage 2 delivered: `*_prim.csv` now exists, so n/p can be formed over
physics-defined phase space. Sample: 2,225 events per Spot, ~571k primaries each.

**The total exported nucleon count is nearly identical across the three
samples** — 4π counts agree to 0.17 % (n) and 0.24 % (p). Because the export
contains almost all 264 projectile and target nucleons, including spectators,
this is a sample-composition check rather than a nucleon-production result.

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

### Cross-check against the reference SMASH study (2026-09-23)

Compared against `docs/n_p - smash check.pdf`. Identifications established first:
the slides' **2.5A GeV is this dataset** (√s_NN = 2.866 GeV ≈ the "2.87gev"
label); slides at 3.8A GeV are a different production. `Rapid` is **lab**
rapidity — the distribution is bimodal with peaks at y ≈ 0 and y ≈ y_beam =
1.973 — so y_cm = y_lab − 0.9863. The three samples are distinct productions.

| region | this analysis (90/0) | slides | local pull* |
|---|---|---|---|
| \|y_cm\|>0.5, Ekin 0.3–0.7 | 1.133 ± 0.048 | ~1.10 | 0.7σ |
| \|y_cm\|>0.5, Ekin 0.9–1.3 | 1.125 ± 0.192 | ~1.25 | 0.7σ |
| \|y_cm\|<0.5, Ekin 0.9–1.3 | 0.996 ± 0.019 | ~1.13 | **7.0σ** |
| \|y_cm\|<0.5, Ekin 1.3–1.7 | 1.007 ± 0.030 | ~1.20 | **6.5σ** |

*The last column divides by this sample's statistical error only. It is not a
formal tension because the slide values are read approximately and the slide
uncertainties, centrality, and final-state definition are unavailable.

Statistics in the mid-rapidity cells (5–17k nucleons per species) are ample to
show that the two outputs differ numerically; they do not establish which
selection or production definition is responsible.

**Leading explanation — the primaries include spectators.** The file holds 256.7
nucleons per event against A(Xe+Cs) = 264: essentially every nucleon of both
nuclei, participants and spectators alike. The spectator fraction is strongly
angle dependent:

| θ_lab | spectator-like |
|---|---|
| [0,1) | 96.3 % |
| [2,3) | 96.1 % |
| [3,4) | 90.9 % |
| [4,6) | 32.1 % |
| [6,8.9) | 3.3 % |
| **[8.9,13.1) HGND** | **5.7 %** |

Splitting the sample is decisive:

| component | 0 MeV | 18 MeV | 90 MeV | 90/0 | ordered |
|---|---|---|---|---|---|
| participant | 1.2401 | 1.1975 | 1.2001 | 0.968 ± 0.006 | no |
| spectator | 1.3254 | 1.3460 | 1.3527 | 1.021 ± 0.005 | yes |

**Under this spectator proxy, the monotonic forward-angle signal reported above
is carried by spectator-like nucleons rather than the remainder.** The proxy is
heuristic and varies with its rapidity and pT thresholds; it is not a truth-level
participant label. The remainder disagrees with the slides in sign and ordering.
Unresolved.

What this pilot appeared to show was an HGND band dominated by the remainder
(5.7 % spectator-like under the proxy), no n/p dependence on S_pot, and
single-species yield changes of 3–6.5 %.  The full-production analysis above
refutes that yield claim: the corresponding changes are only +0.181 % for
neutrons and +0.268 % for protons and are not significant or monotonic.

Candidate explanations to settle: (i) the reference study may count only free or
emitted nucleons, excluding those bound in spectator fragments, whereas the
converter writes all of them; (ii) centrality is unmatched (below).

Full output: `results/np_primaries/xcheck_vs_slides.txt`.

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

## Staged plan — superseded 2026-09-25, historical record

*Stages 1–3 are complete: the primaries export was delivered and analysed, and
the outcome is the decision recorded above. This section is retained for the
reasoning that led to the request, not as outstanding work. Stage 4 (what a
measured n/p would require) remains the only forward-looking part, and is
subsumed by the "do not transfer to experimental HGND n/p" decision.*

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

## Order of work — closed 2026-09-25

Items 1–3 are done; the answer is the decision in "Physics-use decision" above.
Item 4 stands.

1. ~~Request the primaries dump~~ — delivered (568 files, 2.35 M events).
2. ~~Stage 1 diagnostics~~ — superseded by the primaries export itself.
3. ~~Stage 3 measurement~~ — performed; n/p declined for HGND Spot
   reconstruction.
4. **Keep spectral hardness R as the headline observable.** It is measurable by
   the HGND alone, monotonic, and at 6.4σ on the reconstruction target — no
   acceptance caveat attached. This is now the standing recommendation rather
   than an interim one.

### If n/p is revisited

The blockers are no longer data availability but definition and control:
centrality (`B = -1` throughout), generator weights (absent), and the
free/emitted-nucleon definition that separates this export from the reference
calculation. Until those three are settled, more statistics only sharpen an
uncontrolled comparison — as the 131σ local pull against the reference slides
demonstrates.
