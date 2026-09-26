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

## Finding in the feasibility subset: the samples do not share a b distribution (2026-09-26)

The first centrality-matched analysis found that the check above **fails in the
analysed subset**.  This subset comprises four sequential, byte-limited primary
CSV extracts per production; the incomplete final event of each extract is
dropped.  It is approximately 1 % of the full production, not a random sample.

| sample | ⟨b⟩ (fm) | KS vs S_pot = 0 |
|---|---|---|
| 0 MeV | 8.705 | — |
| 18 MeV | 8.632 | D = 0.030, p = 7.6 × 10⁻⁴ |
| 90 MeV | 8.450 | D = 0.056, p = 1.5 × 10⁻¹² |

The direct normalized distributions, empirical cumulative distributions, and
density ratios are shown below.  They are normalized before taking ratios, so
the comparison is not driven by the two-to-four-event differences in sample
size.

![Impact-parameter distributions for the three centrality-feasibility samples](../results/centrality_feasibility/b_distributions.png)

The 90 MeV subset has a **2.93 % smaller mean b** than the 0 MeV subset.  This is
a relative shift in the sample mean, not a shift of 2.93 centrality-percentile
points.  Since `S_pot` is applied after `b` is drawn and cannot alter dσ/db, the
shift is not a symmetry-potential response.  It is either a generation-level
sampling difference or a consequence of taking non-random sequential extracts;
the full B-enabled production must distinguish those possibilities.

It matters because spectral hardness varies by a **factor 3.8** across centrality
(0.22 central → 0.83 peripheral), so a per-cent-level mean-b offset manufactures
a large apparent signal. The centrality-integrated R differs at 9.1σ between the
90 and 0 MeV samples — in exactly the direction the mismatch predicts.

**Percentile matching does not remove it in this subset.** When the underlying
distributions differ, equal percentiles select unequal `b`: the 10–20 % class spans
4.20–5.83 fm for the 0 MeV sample but 4.00–5.53 fm for 90 MeV. Under percentile
matching the artefact survives at up to 8.2σ per class; under common-`b` edges
it falls below 3σ everywhere.

Common `b` edges are a stratification check, not exact event-level matching or
reweighting.  Residual differences inside a wide bin can remain when an
observable changes rapidly with `b`.  Likewise, the integrated row in the
comparison is unchanged by either binning scheme because all events are summed;
it must not be described as an integrated matched result.

This creates a tension with the convention above, and it should be resolved at
the source rather than by choosing a matching scheme:

- **For physics comparison between productions**, match on common `b` edges. It
  removes the sampling artefact, and is defensible precisely because `b` is
  MC truth being used to correct an MC generation problem.
- **For projecting experimental reach**, percentile classes remain the right
  frame — but only once the productions genuinely share a `b` distribution.
  Until then a percentile-matched projection inherits the artefact.

### Review status and reproducibility

The numerical statements in the linked
[`Centrality Confound` draft](https://claude.ai/artifact/7XWP8dHu5Wv6iz8v6Bi8Pw)
are reproduced by the local event-level data: 8,839 / 8,841 / 8,843 complete
events, means 8.705 / 8.632 / 8.450 fm, and the two quoted KS tests.  The added
PDF/PNG makes the evidence for the distribution claim visible rather than
relying only on a ratio panel.

What is established is a statistically clear difference among these sequential
subsets.  What is **not yet established** is that the full productions have the
same difference.  Until the full B-enabled sample is analysed, the report must
retain this scope qualification and must not identify the cause as generator
configuration rather than sequential sample selection.

Reproduction:

- plot: [`b_distributions.pdf`](../results/centrality_feasibility/b_distributions.pdf)
  and [`b_distributions.png`](../results/centrality_feasibility/b_distributions.png)
- binned values: [`b_distribution_binned.csv`](../results/centrality_feasibility/b_distribution_binned.csv)
- exact summary: [`b_distribution_summary.json`](../results/centrality_feasibility/b_distribution_summary.json)
- script: [`scripts/plot_b_distributions.py`](../scripts/plot_b_distributions.py)

Full analysis: [`results/centrality_feasibility/`](../results/centrality_feasibility/).

---

## Finding: `b` carries job-file block structure (2026-09-26)

Before the full production could be compared, two properties of the exports had
to be established, because both change what an uncertainty on a b-dependent
quantity means.

### 1. The feasibility subset was biased by its own truncation

The four extracts per sample were byte-limited to 80 MiB, which is roughly the
first half of a ~4 130-event job file.  Reducing the *complete* first four
`zeroSpot` job files and comparing them with the truncated versions of the same
files gives

| | events | ⟨b⟩ (fm) |
|---|---|---|
| 80 MiB truncation | 8 839 | 8.705 |
| the same four files, complete | 16 315 | 8.589 |

KS D = 0.033, p = 6.6 × 10⁻⁶.  The truncation is therefore **not** a neutral
sample of its own files: b is serially correlated within a file (one file drifts
by 0.8 fm between its halves, Spearman ρ = −0.14, p = 3.5 × 10⁻¹¹), so cutting
each file at a fixed byte offset selects a b-biased subsample.  Any b result
from the feasibility subset is superseded by the full-production numbers.

### 2. Events within a job file are not independent in `b`

Within a *single* sample, where no `S_pot` difference can exist, job-file means
of `b` scatter far beyond their statistical errors:

| sample | files | χ²/ndf across files |
|---|---|---|
| 0 MeV (feasibility subset) | 4 | 27.2 |
| 18 MeV (feasibility subset) | 4 | 5.1 |
| 90 MeV (feasibility subset) | 4 | 15.1 |
| 0 MeV (first 30 complete files) | 30 | 6.68 |

χ²/ndf = 1 is what i.i.d. sampling of `b` within a job would give.  At 30
complete files the job-level error on ⟨b⟩ is 0.0223 fm against a per-event
error of 0.0086 fm — an inflation factor of **2.6**.

**Consequence.** The unit of resampling for any b-dependent quantity is the job
file, not the event.  This is the same class of error as the per-particle vs.
event-level distinction in `plotting_style_protocol.md` §3, one level further
out.  Per-event errors on `b` — including the KS tests quoted in the previous
section, which assume independent events — overstate significance.

---

## Method: reweighting rather than percentile matching

`scripts/b_reweight.py` implements the comparison as follows.

1. **Common reference density.** A reference `f_ref(b)` is formed as the mean of
   the three samples' own b densities in 0.25 fm bins.  Averaging densities
   rather than pooling events prevents the largest sample from defining the
   target it is then weighted towards.
2. **Event weights.** Each event gets `w = f_ref(b) / f_sample(b)`, normalised to
   unit mean.  Bins holding fewer than 20 events in a sample are unusable and
   those events get zero weight; the excluded fraction is reported.
3. **Centrality classes.** Ten 10 % classes with edges from the *pooled* b
   quantiles, so every sample's class *k* covers the same b interval, with
   reweighting applied again inside each class to remove the residual.
4. **Uncertainties.** Job files are resampled with replacement and the
   reweighting is rebuilt inside each replicate, so the weight-estimation error
   is propagated along with the sampling error.

Reweighting is what percentile matching and common-b stratification are not: an
event-level correction. Stratification leaves a residual whenever the observable
varies inside a bin, and equal percentiles select unequal b whenever the
underlying distributions differ.

### The null test is what makes a result believable

A reweighted difference between samples is only meaningful if the procedure
produces no difference when there is nothing to find.  `b_reweight.py null`
splits **one** sample's job files into two pseudo-samples and runs the identical
pipeline; both halves have the same `S_pot`, so any apparent signal is
manufactured by the method.  The distribution of |σ| over many random splits is
the procedure's false-positive rate; a calibrated procedure gives a median |σ|
near 0.7 and exceeds 3σ in about 0.3 % of splits.

On the four-file feasibility subset the null test returns a median |σ| of
0.8–2.2 depending on observable and exceeds 3σ in up to 33 % of splits, and two
halves of the *same* sample differ by a median 1.9 % in the HGND-band neutron
hardness.  With four job files the bootstrap cannot estimate its own error, so
**no result from that subset is interpretable** — including a nominally 3.9 %,
3.9σ reweighted `S_pot` difference in the same observable.  This is why the full
production is required rather than merely desirable.
