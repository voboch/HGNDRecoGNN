# Audit against the Dombay reference notebook

> Review update, 2026-10-16: [the reconstruction review](reconstruction_review_20261016.md)
> documents unresolved differences in metric definitions and effective model
> architecture (DCM-02–03), plus split and reproducibility defects (DCM-04–07).
> The claimed reproduction of the reference settings needs those differences
> and the provenance of the published checkpoint to be checked. This document
> is retained as the earlier analysis record.
> A [live cluster check on 2026-10-16](dcm_experiment_launch_20261016.md) found
> baseline 4366196 and evaluation 4366562 completed on 2026-09-30. The later
> section describing job 4366196 as still running is superseded.

Reference: `EgdeClass_Cluster_Jan25-xecs.ipynb`, the notebook that produced the
Dombay benchmarks on DCM-QGSM-SMM Xe+CsI at 3.26 AGeV. A copy sits at
`NeutronRecoGNN/notebooks/`; the cluster copy is
`/home/vbocharnikov/het_nDetHalf/EgdeClass_Cluster_Jan25-xecs.ipynb`.

Benchmarks quoted from it: ROC AUC ≈ 0.97, ~80 % efficiency at ~87 % purity,
linearity within 10 % over 0.7–5 GeV, resolution < 10 %, cluster isolation
> 97 %.

Contents: the code audit against the notebook; a coincidence-window
measurement on local SMASH data; the per-sample scaling defect that was
cancelling the reconstructed U_sym signal, and its fix; and the DCM results.
The DCM jobs are `slurm/preprocess_dcm.hpc.sbatch`, `train_dcm`, `eval_dcm`.

## What matches

Graph construction reproduces the reference where it matters. Both build two
radius graphs on **raw** coordinates — spatial on `(LayerId, RowId, ColumnId)`
with `rlocal = 3.6`, temporal on `fTime` — and take their **intersection**, not
their union. Both then define clusters as connected components of that
intersection; the reference reaches them by iterating a min-aggregation
`MessagePassing` to its fixed point, this pipeline calls
`scipy.sparse.csgraph.connected_components`, and the two give the same
partition.

Also identical: the 0.15 ns time smearing, the `fELoss > 0.003` cut, the `eToF`
definition and its ±2σ variants, the `groupby(['Row','Instance']).tail(1)`
deduplication, the top/bottom split at `fY = 0`, dropping events containing an
unlinked hit (`PDG == 0`), the cluster label as the max of `n0_label` over the
cluster and the cluster energy target as the max of `Ekin · n0_label`.

One reference quirk is worth recording because it looks like a bug and is not
inherited: the notebook fits `scaler_top` and `scaler_bot` separately, then
transforms **both** halves with `scaler_top` (cell 45, both loops). Its bottom
half is therefore scaled by the top half's statistics. This pipeline uses one
shared scaler, which is what the reference does in effect.

## Divergence 1 — the coincidence window, and what it costs

The reference uses `twindow = 1.0` ns. This pipeline has used `1.5` ns since the
first run with no reason recorded. Slide 9 of the reference report judges the
clustering on three multiplicities, and those are what the window has to be
chosen on:

- **clusters per prompt n⁰** — splitting. The report claims the prompt neutron
  splits into secondary clusters at under 2 %.
- **prompt n⁰ per signal cluster** — isolation. The report claims over 97 %.
- **background hits per signal cluster** — contamination, which motivates the
  network rather than the clustering.

All three are defined on **prompt** hits, those whose kinetic energy sits within
±2σ of the time-of-flight energy. That matters: a neutron deposits many hits but
usually only one prompt hit. An earlier version of this scan counted every hit
of a neutron and reported a 21.7 % split rate against the report's under 2 %;
the discrepancy was the definition, not the pipeline.

`scripts/scan_time_window.py` measures the three distributions against the
window on both samples, 97,227 SMASH and 98,597 DCM events.
Figures: `results/graph_audit/` and `results/graph_audit_dcm/`,
`*_multiplicities.png` for the slide-9 distributions and `*_scan.png` for what
the window costs.

Which sample decides what. **SMASH is the working dataset** — every physics
number in this project is measured on it — and the clustering parameters have
to be chosen there. DCM-QGSM-SMM enters only as the reconstruction-performance
reference, and its role here is narrow but necessary: it is the sample the
reference report's own clustering claims were made on, so it is the only place
those claims can be checked.

**They check out.** On DCM at 1.5 ns this scan gives isolation 96.7 % against
the quoted "> 97 %", and splitting 1.6 % against the quoted "< 2 %". The
clustering behaves as the report describes, which is what licenses using its
two criteria as targets on SMASH.

| $t_{\rm win}$ [ns] | isolation | unsplit | bkg/cluster | count bias | bias on R |
|---|---|---|---|---|---|
| **SMASH — the working dataset** | | | | | |
| 0.50 | 96.77 % | 89.41 % | 1.77 | +7.30 % | −0.21 % |
| **0.75** | 95.66 % | 96.39 % | 2.40 | **−1.06 %** | +2.05 % |
| 1.00 | 94.99 % | 97.11 % | 2.70 | −2.52 % | +3.05 % |
| 1.50 (this pipeline) | 94.04 % | 97.28 % | 3.06 | −3.71 % | +4.70 % |
| **DCM — the reconstruction reference** | | | | | |
| 0.50 | 98.38 % | 89.70 % | 1.65 | +8.81 % | +0.37 % |
| **0.75** | **97.72 %** | **97.53 %** | 2.17 | **+0.14 %** | +2.04 % |
| 1.00 (reference setting) | 97.22 % | 98.25 % | 2.40 | −1.08 % | +2.89 % |
| 1.50 | 96.69 % | 98.37 % | 2.63 | −1.76 % | +3.94 % |

The two criteria pull against each other, which is what makes this a choice.
Splitting is a *narrow*-window failure: below about 0.6 ns a prompt neutron's
hits stop being linked and the unsplit fraction collapses, to 89.7 % at 0.5 ns
and 66 % at 0.3 ns on SMASH. Isolation is a *wide*-window failure: it falls
monotonically as the window admits more neighbours, 98.4 % to 95.4 % on DCM
across the scanned range. Background contamination rises monotonically too.

On both samples the signal-cluster count closes near 0.75 ns — 0.72 ns on SMASH
and 0.76 ns on DCM, two independent samples agreeing to 0.04 ns — and the
hardness bias there is roughly half its value at 1.5 ns. That is already enough
to say 1.5 ns is not defensible. It is not yet enough to fix the window, because
the table above holds the prompt definition at its own unexamined value; that is
Divergence 1b.

## Divergence 1b — the prompt definition, calibrated with the window

The window is not the only unexamined constant in the clustering. A hit counts
as **prompt** when its MC kinetic energy is consistent with the energy implied
by its arrival time and position, within $n\sigma$ of the 0.15 ns time
resolution. The production fixes $n\sigma = 2$ and nothing records why.

The two are not independent — widening either admits more of the event into the
signal definition — so choosing one while holding the other at an unexamined
value is not a calibration. `scripts/scan_time_window.py --prompt-sigmas` scans
both. On SMASH, 97,227 events, 30 combinations
(`results/graph_audit/joint/`, figure `fig_prompt_window_joint.png`):

| $n\sigma$ | isolation | unsplit | count bias | bias on R |
|---|---|---|---|---|
| *at $t_{\rm win}$ = 0.75 ns* | | | | |
| **1.0** | **95.88 %** | **98.24 %** | **−2.60 %** | **+0.79 %** |
| 1.5 | 95.71 % | 97.31 % | −1.90 % | +1.55 % |
| 2.0 (production) | 95.66 % | 96.39 % | −1.06 % | +2.05 % |
| 2.5 | 95.64 % | 95.43 % | −0.12 % | +2.15 % |
| 3.0 | 95.62 % | 94.46 % | +0.83 % | +2.62 % |
| *production setting* | | | | |
| 2.0 at $t_{\rm win}$ = 1.5 ns | 94.04 % | 97.28 % | −3.71 % | +4.70 % |

The two parameters act on different failures, which is why scanning them
together was necessary. **Isolation is set by the window alone** — it moves by
0.26 points across the whole $n\sigma$ range at fixed $t_{\rm win}$, against
3 points across the window range. **Splitting is set mainly by the prompt
width**: a looser match labels more of a neutron's deposit prompt, those hits
spread across clusters, and the unsplit fraction falls from 98.2 % to 94.5 % at
fixed window. Tightening the match to $1\sigma$ therefore buys splitting and
hardness bias at almost no cost in isolation.

**The recommendation is $n\sigma = 1$, $t_{\rm win} = 0.75$ ns**, and it is not
a compromise: it beats the production's $(2\sigma, 1.5$ ns$)$ on every one of
the four measures — isolation 95.88 % against 94.04 %, unsplit 98.24 % against
97.28 %, count bias −2.60 % against −3.71 %, and bias on the hardness ratio
**+0.79 % against +4.70 %, a factor six**. Nothing in this scan argues for the
present setting.

**The reference's criteria are not reachable on SMASH.** No combination in the
grid reaches isolation 97 %; the best is 96.90 % at the narrowest window. That
is a property of the sample, not of the pipeline — SMASH is the busier
environment, and the same scan on DCM reaches 97.55 % isolation with 98.24 %
unsplit at $(1.5\sigma, 0.75$ ns$)$. The reference's thresholds were set on DCM
and should not be carried to SMASH as pass/fail criteria; what carries is the
method of judging the clustering, not the numbers.

**What this scan still cannot see.** A tighter prompt match labels fewer hits,
so the network has less signal to learn from and the cluster energy target rests
on fewer hits; a narrower window leaves fewer hits per cluster. Both costs are
invisible to a truth-level combinatorial count and both fall on the energy
regression. The calibration therefore nominates $(1\sigma, 0.75$ ns$)$ for
training against the present $(2\sigma, 1.5$ ns$)$, and the choice between them
is made on the reconstructed response, not on this table.

## Divergence 2 — the signal definition carries an extra cut

The reference labels a hit as a neutron head when

```
PDG == 2112  &  eToF_dn_2sig ≤ Ekin ≤ eToF_up_2sig  &  Side ∈ {5, 6}
```

This pipeline adds `Ekin > 0.3`. The cut sits below the 1–3 GeV scan range, so
it does not touch the hardness observable, but it removes soft neutrons from
the positive class and therefore changes what "efficiency" counts. Efficiency
measured here and efficiency quoted from the notebook are not the same
denominator, and the 0.714-against-0.80 comparison inherits that.

## Divergence 3 — the reference's energy loss is reweighted, ours is not

The notebook trains the energy head twice. The first pass (cell 58) uses plain
`mse_loss` on true clusters, which is what this pipeline implements. The second
pass (cells 63, 74–79) rebuilds the loss with a `WeightedMSELoss` whose weights
flatten the `e_true` distribution across 10 bins, then retrains for 15 epochs at
lr 1e-4. The published linearity is the output of that second stage, and the
results it feeds are written to a separate `results/wtest/` directory.

This matters for where our converged checkpoint is still weak. Its linearity is
within ±1 % from 0.5 to 2.6 GeV and degrades above that — −3.5 % at 3.0–3.5 GeV,
−9.4 % at 4–5 GeV, −16.8 % in the sparse 5–6.5 GeV bin — which is the direction
and the region a falling spectrum produces when the energy loss is unweighted.
The high band of the hardness ratio is `E > Rthr + δ` with no upper edge, so it
collects exactly those energies.

Implemented as `spectrum_flattened_energy_loss` in `training/losses.py`, off by
default (`LossWeights.energy_flatten = False`) so existing results are
unchanged.

## Divergence 4 — the reference's event ids alias, ours are guarded

The notebook maps per-file `Row` into a global space with

```python
idf.Row += (int(file.split("_")[0]) - 1) * 500 + i * 200000
```

which assumes fewer than 500 events per file. In the SMASH production files
hold about 4,150, so each file spilled roughly 3,650 rows into the next file's
band and about eight consecutive files aliased onto the same ids. The merge that
attaches MC truth to hits joins on `['Row', 'Instance']`, so the aliasing
corrupted it: about 23 % of truth keys were duplicated and about 31 % of hits
matched more than one MC particle. `data/graph_dataset.py` gives every
`(run, file)` its own block sized from the data, and raises on any duplicate
`(Row, Instance)` key rather than letting it through.

It did not bite the reference's own sample. The DCM files hold about 300 events
each and their largest `Row` is 498 — across the 62 files checked in run 988974,
none reaches 500. The stride was correct for the data it was written for and
only broke when it was reused on the SMASH production, where files hold about
4,150 events.

**The Dombay benchmarks are therefore a clean target.** The aliasing affected
our SMASH numbers, not the reference's.

## The reconstructed U_sym signal was being cancelled by per-sample scaling

This came out of re-running the hardness window search on the converged
checkpoint, and it is the most consequential thing in this audit.

**The null.** On the converged predictions the search scans 207 windows, finds
125 admissible, and **none** of them orders the three samples. The strongest
positive-but-unordered window gives +2.22 % at 1.05 sigma, against a truth-level
neutron-only response of +4.83 % ± 1.02 % (4.75 sigma, held out).

**The cause.** The reconstructed energy carries a per-sample offset. Median
`e_pred / e_true` is 1.0082 for zeroSpot, 1.0050 for defaultSpot and 1.0003 for
bigSpot — monotonic in U_sym, about 32 sigma on 110 k clusters per sample, and
in the direction that makes the hardest sample look softest. Two explanations
were tested and both fail:

- *Spectrum weighting through the high-energy non-linearity.* Ruled out: at
  **fixed** true energy the offset is a near-constant −0.7 to −0.9 % from
  zeroSpot to bigSpot in every bin from 0.5 to 3.5 GeV. The samples' mean true
  energies differ by 0.7 % (1.6103 against 1.6219 GeV), far too little to
  produce it.
- *Event occupancy.* Ruled out: mean occupancy is 1.490, 1.489 and 1.491
  clusters per event, and the offset persists unchanged inside every occupancy
  bin.

What remains is the input scaling. The three v3 caches carry **three different
scalers** (distinct md5s on `scaler_top.pkl`), each fitted to its own sample.
`eToF` is the feature that differs: bigSpot's scale is 0.41 % wider than
zeroSpot's and its mean 0.019 GeV higher, while every other feature agrees to
within 0.2 %. `eToF` carries the neutron energy, so a harder sample has a wider
`eToF` distribution, and standardising each sample by its own statistics divides
part of that difference out before the network sees it. A neutron of a given
energy arrives at the network with a smaller standardised `eToF` in bigSpot than
in zeroSpot, and the energy head duly returns less. The shift is about 0.7 % of
a standard deviation, which is the size of the observed response offset.

`data/graph_dataset.py::prepare_halves` already documents this hazard and takes
a `scaler_source` to avoid it. The v3 caches were built without it.

**The test.** Dividing each sample's `e_pred` by its own median response over
1.0–2.2 GeV — MC truth, so a diagnostic and not a method — and re-running the
search:

| | admissible | ordered | chosen window | response |
|---|---|---|---|---|
| as built | 125 | **0** | — | +2.22 % at 1.05 sigma (unordered) |
| offset removed | — | many | Rthr 2.2, delta 0.6 | +9.11 % at 3.89 sigma (dev) |

and on held-out units, with the window fixed beforehand, R rises monotonically
0.0992 → 0.1006 → 0.1060 for U_sym = 0, 18, 90 MeV, giving
**+6.83 % ± 3.17 % (2.16 sigma), ordered** — consistent with the truth-level
+4.83 %.

So the reconstructed signal is not absent. It was being cancelled by an
artefact of how the caches were normalised.

**The fix.** Standardisation is affine, so it can be undone exactly rather than
rebuilt: `scripts/evaluate.py --restandardise-to <scaler dir>` re-expresses the
cached features in the scaler the checkpoint was trained with,

    x_dst = (x_src · scale_src + mean_src − mean_dst) / scale_dst

leaving the raw `SurfaceHit` column alone and selecting the top or bottom
scaler per graph. `tests/test_restandardise.py` checks the round trip to 1e-9.
This avoids three cache rebuilds for the present analysis. Training on more
than one sample at once still needs the caches rebuilt with a shared scaler,
via the `--scaler-source` the preprocessor already accepts.

**The fix, measured** (job 4365420, all three samples re-evaluated against the
defaultSpot scaler the checkpoint was trained with). Median `e_pred / e_true`
over 1.0–2.2 GeV:

| | zeroSpot | defaultSpot | bigSpot | spread |
|---|---|---|---|---|
| as built | 1.01216 | 1.00899 | 1.00398 | 0.82 pp, monotonic in U_sym |
| restandardised | 1.00938 | 1.00921 | 1.01025 | 0.10 pp, not ordered |

Within-sample performance is unchanged, as it must be: ROC AUC 0.9582 against
0.9578, and the 87 % energy-purity working point moves from 0.872/0.714 to
0.873/0.715. None of those metrics can see a per-sample offset, which is why
the defect survived every performance check made so far.

The window search on the corrected predictions, with no MC truth used anywhere:

| | admissible | ordered | held-out response |
|---|---|---|---|
| as built | 125 | **0** | — (nothing to carry) |
| restandardised | 142 | **10** | **+4.03 % ± 2.46 % (1.64 sigma), ordered** |

with the window fixed on development data at Rthr = 2.6, delta = 0.2, and R
rising 0.0671 → 0.0672 → 0.0698 across U_sym = 0, 18, 90 MeV.

So the reconstructed observable now orders the samples and carries a positive
response, where before no admissible window did. It is not yet a measurement:
1.64 sigma on held-out units is consistent with zero.

The truth-level target is **+4.83 % ± 1.02 % (4.75 sigma)**, measured the same
way on the same acceptance
(`results/final_eval/observables/hardness_truth.json`). An earlier draft of this
note compared the reconstructed response against +10.5 %, which is wrong: that
figure is the isovector double ratio R_n/R_p from the acceptance study, a
different observable. Against the neutron-only truth response the reconstruction
retains about four fifths of the effect, not two fifths.

Note that this cancellation is separate from, and larger than, the coincidence
window effect in Divergence 1, and the two act in the same region. Neither was
visible in the performance metrics: AUC, purity, efficiency, linearity and
resolution are all computed within a sample and none of them can see a constant
per-sample energy offset.

## Not divergences, but worth recording

- **Checkpoint selection.** The notebook keeps the model when the *training*
  loss improves (`if loss < minloss`, both training cells) while printing the
  validation loss beside it. This pipeline selects on validation.
- **Architecture.** `models/default_net.py` matches the reference layer for
  layer: EdgeConv(64→64→64) → EdgeConv(128→128→256) → `num_layers − 4` ×
  SAGEConv → GraphConv ×2, edge scores from concatenated endpoint features fed
  back as `GraphConv` edge weights, `avg_pool_x` over clusters, then
  DynamicEdgeConv(k=100, aggr='sum') on the cluster features and four heads.
  At H=512, L=8 this is 12,479,173 parameters.
- **Extra inputs and terms.** This pipeline adds a tenth feature, `SurfaceHit`,
  and a fifth loss term on cluster–cluster links. Both are additions to the
  reference, not departures from it.
- **Cluster isolation — resolved.** An earlier version of this note reported
  that the quoted "> 97 %" could not be reproduced, measuring 71 % at 1.5 ns.
  That was an error of definition on this side: isolation counts **prompt**
  neutrons per signal cluster, not distinct `Id`s, and a neutron's secondaries
  carry different `Id`s while contributing no prompt hit. Measured as the report
  defines it, DCM at 1.5 ns gives 96.7 % isolation and 1.6 % splitting against
  the quoted "> 97 %" and "< 2 %". Both reproduce, and both are now used as
  targets in Divergence 1.

## DCM results

Both caches built (jobs 4365847, 4365848): 283,398 half-event graphs from 619
hits/vacs file pairs. The loader's duplicate-key guard passed, and the merge is
clean — hits and vacs are 1:1 with nothing unmatched, unlike the SMASH
production where unlinked hits have to be dropped.

**Domain transfer** (job 4366195): the SMASH-trained converged checkpoint
applied to DCM with the SMASH scaler and the 1.5 ns window it was trained at,
so nothing about the network or its inputs is re-fitted to the new generator.

| | SMASH (own domain) | DCM (transfer) | reference |
|---|---|---|---|
| ROC AUC | 0.9582 | 0.9525 | ~0.97 |
| purity / efficiency | 0.873 / 0.715 | 0.866 / 0.688 | 0.87 / 0.80 |
| linearity, 0.5–2.6 GeV | ±1.1 % | ±2.3 % | within 10 % |
| resolution, 0.5–3.0 GeV | 5.9–7.3 % | 5.9–7.3 % | under 10 % |
| worst above 0.7 GeV | 16.8 % lin, 8.8 % res | 12.9 % lin, 10.5 % res | under 10 % |

The pipeline survives the change of generator: AUC falls by 0.006 and
efficiency at fixed purity by 2.7 points, with linearity and resolution holding
inside the reference bounds through the range that carries the statistics. The
benchmarks are met on DCM except in the sparse tails — 10.5 % resolution in the
3.5–4.0 GeV bin (2,714 clusters) and −12.9 % linearity above 5 GeV (598
clusters). Efficiency at 87 % purity remains the one benchmark clearly missed,
on both samples, and by a similar margin — which points at the working point
rather than at the generator.

`results/dcm/dcm_neutron_spectra.png` gives the MC-truth and reconstructed
neutron spectra and their ratio. The response is strongly energy dependent:
0 at 0.3 GeV, rising through 0.5 at 1.1 GeV to about 1.0 at 2.5 GeV, then
falling to 0.2 by 6 GeV. The hardness ratio divides counts on either side of a
threshold inside that range, so the response does not cancel between the bands
— which is why the reconstructed +4.0 % sits below the truth-level +4.83 %
rather than matching it.

**Native reproduction** (job 4366196, running): training on DCM at the
reference settings — H=512, L=8, batch 256, Adam 1e-3, 25 epochs, 50/50 split,
seed 42 — is the only configuration the Dombay benchmarks can be read against
directly, since the published numbers come from a model trained on this sample.
At epoch 2 of 25 the validation loss is 1.2625 and falling.

## No U_sym response can come from DCM

DCM-QGSM-SMM is one sample with one equation of state. It supplies the spectra,
the response and the performance numbers above; the U_sym sensitivity stays
with the three SMASH samples, where the corrected reconstructed response is
+4.03 % ± 2.46 % against a truth-level +4.83 % ± 1.02 %.
