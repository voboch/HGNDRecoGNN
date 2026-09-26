# Centrality convention for the HGND sensitivity analyses

## Why percentile classes rather than impact parameter

The impact parameter `b` is now recovered per event (`DstEventHeader.fB`, see
`ncx/README.md`), which makes it tempting to compare the symmetry-potential
samples in bins of `b` directly. That would not be a measurement.

**`b` is not an observable.** No detector measures it. What an experiment
measures is a quantity monotonically correlated with it — charged-particle
multiplicity in a reference region, or spectator energy in a forward calorimeter
(for BM@N, the FHCal) — and converts that into a *centrality percentile* from
where the event falls in the measured distribution.

An analysis binned in `b` therefore cannot be reproduced on data. It also hides
the dominant experimental systematic, which is the resolution of the centrality
estimator, not the width of a `b` bin.

## The convention

Events are ordered by the centrality estimator and divided into **percentile
classes of the inelastic cross section**, conventionally **ten bins of 10 %**:

| class | meaning |
|---|---|
| 0–10 % | most central decile |
| 10–20 % | … |
| … | |
| 90–100 % | most peripheral decile |

Finer binning (5 % or 2.5 %) is used where statistics allow and the observable
varies rapidly; coarser grouping (0–10 %, 10–40 %, 40–80 %) is common when
statistics are the limit. Ten bins is the default here: it resolves the strong
centrality dependence of nucleon observables without splitting the samples below
the point where the per-bin Spot comparison stops being meaningful.

## How the classes are defined in this analysis

In simulation, with minimum-bias sampling (`dσ/db ∝ b`), the percentile follows
directly from `b`:

```
centrality percentile of an event = (number of events with b' < b) / N_events
```

so sorting on `b` and cutting at deciles reproduces the classes an experiment
would obtain with a perfect estimator. **This is the MC-truth limit of the
experimental procedure, not a substitute for it.**

Two consequences are carried through every result:

1. **Quantiles are computed per sample, not from shared `b` edges.** That is what
   an experiment does — each dataset is calibrated against its own measured
   distribution. It also makes the comparison insensitive to any difference in
   the sampled `b` range between productions. Shared edges are used only as a
   cross-check; if the two disagree, the `b` distributions differ, and that is
   itself a finding.

2. **The result is an upper bound on centrality-resolved sensitivity.** A real
   estimator has finite resolution, which smears events across class boundaries
   and dilutes any centrality-dependent signal. Quoting a truth-`b` result as if
   it were achievable overstates the reach; the degradation has to be folded in
   before any number is presented as a projection.

## Why this matters for the symmetry-potential comparison

n/p and the neutron spectrum both depend strongly on centrality. Comparing the
three `S_pot` productions without matching centrality confounds a difference in
the sampled `b` distribution with a genuine potential effect — this was the
blocking systematic recorded in `NP_RATIO_PLAN.md` while `B` was unavailable.

Matching is therefore done **within percentile class**: the Spot comparison is
performed bin by bin, and a claimed sensitivity must survive in individual
classes, not only in the centrality-integrated sample, where a distribution
mismatch can masquerade as signal.

## Checks that accompany any centrality-binned result

- The `b` distributions of the three samples agree (they should: same system,
  same energy, same generator settings). A disagreement invalidates the
  centrality-integrated comparison regardless of what the binned one shows.
- `b` ranges and per-class event counts are reported, not only the class labels.
- Where a signal appears in the integrated sample but in no individual class, it
  is treated as a centrality artefact until shown otherwise.

---

## Finding: the productions do not share a b distribution (2026-09-26)

The first centrality-matched analysis found that the check above **fails**:

| sample | ⟨b⟩ (fm) | KS vs S_pot = 0 |
|---|---|---|
| 0 MeV | 8.705 | — |
| 18 MeV | 8.632 | D = 0.030, p = 7.6 × 10⁻⁴ |
| 90 MeV | 8.450 | D = 0.056, p = 1.5 × 10⁻¹² |

The 90 MeV sample is **2.93 % more central**. Since `S_pot` is applied after `b`
is drawn and cannot alter dσ/db, this is a generation-level sampling difference.

It matters because spectral hardness varies by a **factor 3.8** across centrality
(0.22 central → 0.83 peripheral), so a per-cent-level mean-b offset manufactures
a large apparent signal. The centrality-integrated R differs at 9.1σ between the
90 and 0 MeV samples — in exactly the direction the mismatch predicts.

**Percentile matching does not remove it.** When the underlying distributions
differ, equal percentiles select unequal `b`: the 10–20 % class spans
4.20–5.83 fm for the 0 MeV sample but 4.00–5.53 fm for 90 MeV. Under percentile
matching the artefact survives at up to 8.2σ per class; under common-`b` edges
it falls below 3σ everywhere.

This creates a tension with the convention above, and it should be resolved at
the source rather than by choosing a matching scheme:

- **For physics comparison between productions**, match on common `b` edges. It
  removes the sampling artefact, and is defensible precisely because `b` is
  MC truth being used to correct an MC generation problem.
- **For projecting experimental reach**, percentile classes remain the right
  frame — but only once the productions genuinely share a `b` distribution.
  Until then a percentile-matched projection inherits the artefact.

Full analysis: `results/centrality_feasibility/`.
