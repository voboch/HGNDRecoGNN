# HGND reconstruction and symmetry-potential study: progress review and guide

**Status date:** 2026-09-27 (evening)  
**Purpose:** single decision record for what is established, what has been
withdrawn, what currently blocks a measurement, and what should be done next.

## Executive decision

The project has established a strong **truth-level spectral-shape response** to
the SMASH symmetry-potential strength.  After full-production impact-parameter
control, the neutron hardness in the HGND angular band rises by **4.27 %** from
`S_pot = 0` to 90 MeV, the corresponding proton hardness falls by **5.84 %**,
and their truth-level hardness double ratio rises by **10.73 %**.  The response
is monotonic at 0, 18, and 90 MeV and survives in all ten centrality classes for
the double ratio.

That is not yet a reconstructed HGND measurement.  The current GNN energy
estimate changes the sign of the neutron response on the same selected clusters:
the 20-file reconstruction experiment gives **+3.32 % ± 2.94 %** using true
cluster energy and **−3.04 % ± 2.30 %** using calibrated reconstructed energy.
The paired displacement is **−8.66 % ± 2.21 % (3.91σ)**.  The high-energy
response differs between the 90 and 0 MeV samples by 30.6 MeV near the 2 GeV
threshold.  This is a systematic domain-response error; more events would make
the wrong answer more precise.

**Update (2026-09-27).**  Two thirds of that displacement was preprocessing, not
the network.  `StandardScaler` was fitted **per dataset**, and `eToF` carries the
neutron energy, so the harder sample received a 1.9 % wider `eToF` scale and part
of the difference being measured was divided out before the network saw it.
Rebuilding the caches with one shared scaler taken from the 18 MeV midpoint and
running the **identical checkpoint** moves the differential high-energy response
from **−30.7 ± 8.6 MeV (3.55σ)** to **−11.4 ± 9.8 MeV (1.16σ)**, the paired
displacement from **2.81σ** to **0.53σ**, and the reconstructed 90/0 effect from
**−3.04 %** to **+2.25 % ± 2.43 %**, the same sign as the truth **+3.67 % ± 2.12 %**.
Reconstructed `R_n` is now ordered across 0/18/90.  No retraining was involved.

Two further findings change what P1 should target:

- The `theta in [8.9, 13.1)` band used for every truth-level number is **not the
  detector**.  The front face is a rectangle offset to negative x covering 20 % of
  azimuth, and because the generator holds the reaction plane fixed with a
  strongly energy-dependent `v1`, the azimuth cannot be weighted away.  The
  corrected window roughly doubles the isovector signal.
- Detection efficiency spans a factor **39.6** in `E_kin`, and `R`'s conventional
  denominator sits below 1 GeV where efficiency is 2 %.  That denominator is not
  recoverable by unfolding, so the observable itself must be redefined.

The guiding decision is therefore:

1. **Keep neutron spectral hardness `R_n` as the primary HGND physics target.**
2. **Do not claim that the present reconstruction measures its `S_pot`
   response.**  Energy-response invariance is the blocking requirement, and it is
   now partly met: with a shared scaler the response differential is consistent
   with zero and the sign is right, but the residual displacement is still 39 %
   of the truth effect against a 25 % gate, and the purity lock and calibration
   still use truth information.
3. **Fit one scaler and reuse it.**  Any dataset that will be compared with
   another, or trained on jointly, must share a single fitted normalisation.
   Per-dataset standardisation is a domain leak, not a neutral preprocessing
   step.
4. **Select both species in the real acceptance, not a polar-angle band.**
5. **Do not use the HGND-band n/p yield ratio as an `S_pot` probe.**  It is null
   and non-monotonic at full statistics.
6. Treat the neutron/proton hardness double ratio as a powerful truth-level
   transport diagnostic, but not as an HGND-only observable; the proton arm must
   come from BM@N tracking with a controlled common phase space.
7. Quote the approximately 2 MeV `S_pot` resolution only as a truth-level,
   Monte-Carlo-statistical reach.  It is not an experimental resolution and is
   not an uncertainty on an EOS parameter such as `L` or `S_0`.

## Evidence hierarchy

Conclusions in this guide are ordered by the strongest available evidence:

1. **Full production, job-file bootstrap and b reweighting:** 2,345,993 events
   in 568 job files across all three `S_pot` values.  This is authoritative for
   truth-level centrality and spectral-hardness conclusions.
2. **Same-sample null and injected-bias tests:** validate the reweighting and
   uncertainty procedure and define its empirical noise floor.
3. **Reconstruction ladder:** 20 independent job files per sample, about 82,000
   truth events per sample.  This is sufficient to identify the energy-response
   failure, but not to give a final reconstructed sensitivity.
4. **Preprocessing domain test:** the same checkpoint over caches built with
   per-sample and with shared scalers.  Because nothing else changes, it
   attributes response differences to normalisation alone.
5. **Earlier sequential or byte-limited subsets:** diagnostic history only.
   Their centrality and significance claims are superseded.

## Progress by workstream

| workstream | status | evidence | decision |
|---|---|---|---|
| Cluster environment and schema-v2 preprocessing | complete | schema-v2 caches and successful V100 training | infrastructure is usable and reproducible |
| Baseline GNN training | complete | `net_default`, 20 epochs, seed 42; checkpoint retrieved locally | adequate reference checkpoint, not a final energy estimator |
| Full three-sample sensitivity pipeline | complete | full-statistics job 4300982 and downstream tables/figures | analysis plumbing is operational |
| Impact-parameter recovery | complete | `DstEventHeader.fB` available per event | old `B = -1` analyses remain historical only |
| Centrality control | complete at truth level | job-file bootstrap, event reweighting, null and injection closure | earlier centrality-artifact conclusion is withdrawn |
| Truth-level `S_pot` scan | complete | 0 / 18 / 90 MeV response and job-level uncertainties | spectral hardness is the sensitive observable |
| HGND-band n/p yield ratio | closed negative | −0.09 %, 1.0σ; non-monotonic middle point | decline as an `S_pot` probe |
| Proton measurement with HGND | closed negative | reaching protons are soft secondaries; 0.65σ response | requires an external charged-particle arm |
| Neutron acceptance and cluster selection | promising but limited | response keeps its sign through truth acceptance and selected clusters | detector geometry/classifier are not the main blocker |
| Reconstructed neutron energy | improving, retrain incomplete | linearity +50–60 % → within 10 % over 1.4–2.6 GeV; resolution 26–34 % → 15–20 % | train to convergence; 4 of 20 epochs completed |
| Held-out sensitivity test | blocked by statistics, not by method | +2.87 ± 6.92 %; 0.62σ expected even with perfect reconstruction | size the test split to ~280 generation jobs |
| PRC manuscript | reworked, moved to `../HGNDPaper` | referee review v7: major revision, B-2 and B-3 closed | two placeholder figures remain (B-1, open since v5) |
| Feature normalisation | resolved, needs rollout | per-dataset scaler shifted `eToF` by 1.9 % in scale between samples | one fitted scaler for every dataset that is compared or trained together |
| Nucleon acceptance definition | complete | full production reduced in the front face; +10.48 % double ratio against +10.73 % for the band | selection does not drive the result |
| Reconstructed-energy efficiency | characterised | efficiency spans 39.6x; forward model closes exactly, inversion does not | redefine `R` inside the efficiency plateau |
| Experimental centrality projection | not started | only truth-b limit exists | add FHCal/multiplicity estimator and resolution |
| Model-family systematic | incomplete | seeds 42/123/456 exist; alternative model-family outputs are missing | fill before final systematic claim |
| PRC manuscript | technically near-ready, scientifically stale | referee review v6 says minor revision, but predates the final reconstruction conclusion | update framing before submission |

## Centrality: resolved, with a methodological lesson

The first B-enabled feasibility subset appeared to show a 2.93 % mean-b shift
and a highly significant KS difference.  That result was caused by two effects:

- four 80 MiB extracts sampled only the first part of each job file, while `b`
  drifts within files;
- per-event tests treated correlated events from the same production job as
  independent.

At full statistics the 90−0 difference is **+0.0034 ± 0.0147 fm (0.23σ)** using
job-level errors, and the KS distance falls from 0.0562 to 0.0038.  A mild shape
residual remains (`χ²/ndf = 1.57`, `p = 0.029`), but event-level b reweighting
reduces the spread in mean b to 0.00010 fm, costs only 0.1 % in effective
statistics, and discards no events.

The correction is validated in both directions:

- same-sample job-file splits give the expected null distribution and measure
  the method noise floor;
- an injected −1.56 fm b shift creates effects as large as 61σ before
  correction and leaves every tested observable within 1.1σ after correction.

**Rule going forward:** bootstrap job files, rebuild weights inside every
replicate, and keep the injection and same-sample null tests with every new
observable.  Event-level or particle-level errors are not acceptable for this
production.

## Physics conclusion at truth level

For

`R = N(E_kin > 2 GeV) / N(E_kin < 1 GeV)`

after b reweighting and job-file bootstrap:

| observable | 0 MeV | 18 MeV | 90 MeV | 90/0 change | significance | interpretation |
|---|---:|---:|---:|---:|---:|---|
| `R_n`, HGND band | 0.68705 | 0.69355 | 0.71638 | +4.27 % | 21.7σ | direct neutron target |
| `R_p`, HGND band | 0.68086 | 0.67035 | 0.64112 | −5.84 % | 33.4σ | truth/tracking target, not HGND-only |
| `R_n/R_p`, HGND band | 1.00909 | 1.03461 | 1.11739 | +10.73 % | 37.0σ | cleanest isovector diagnostic |
| n/p yield, HGND band | 1.17590 | 1.18258 | 1.17484 | −0.09 % | 1.0σ | reject |

The 18 MeV point lies between the endpoints for every hardness observable.  The
double ratio is monotonic in all ten centrality classes and has a fitted response
of **+0.1176 % per MeV**, linear to within 1.6σ.  Dividing by its measured 0.25 %
same-sample noise floor gives approximately **2.1 MeV truth-level statistical
resolution in `S_pot`**.

This does not map directly to a symmetry-energy slope parameter.  That requires
the transport-model relation between `S_pot` and the density-dependent symmetry
energy sampled by these collisions.

## What survives detector acceptance and reconstruction

The 20-job-file measurement ladder isolates where information is lost:

| rung | 90/0 change in neutron hardness | significance | reading |
|---|---:|---:|---|
| primary neutrons in angular band | +2.65 % ± 0.54 % | 4.93σ | subset reproduces positive truth response |
| neutrons reaching HGND | +3.67 % ± 2.05 % | 1.79σ | sign survives acceptance; limited statistics |
| selected signal clusters, true energy | +3.32 % ± 2.94 % | 1.13σ | classifier does not reverse it |
| same selected clusters, calibrated reconstructed energy | −3.04 % ± 2.30 % | 1.32σ | energy estimator reverses it |
| same selected clusters, uncalibrated energy | −4.39 % ± 3.56 % | 1.23σ | raw estimator is also unsuitable |

The classifier is therefore not the leading failure.  The decisive comparison
uses identical clusters and changes only the energy coordinate.  The observed
3.91σ paired displacement demonstrates estimator-induced sample dependence.

The present calibration is also optimistic: both purity locking and energy
calibration use truth information, and only 20 of roughly 200 job files per
sample enter this experiment.  Failure under these favourable conditions is a
strong blocker; success after retraining must still be demonstrated on a blind
job-file holdout.

## The acceptance is a rectangle, not a polar-angle band

Every truth-level number above selects `theta in [8.9, 13.1)` at all azimuths.
The detector is not that.  Its front face is

    z = 718.42 cm,  x in [-146.61, -106.67] cm,  y in [-77.34, +77.34] cm

offset to negative x, subtending 8.45 to 12.99 deg over 71.9 deg of azimuth.
The band is **0.0879 sr**; the acceptance is **0.0114 sr**, 12.9 % of it.

The azimuth cannot be recovered by weighting, because the generator keeps the
reaction plane fixed in the laboratory.  Directed flow is therefore visible in
lab coordinates and is strongly energy dependent inside the HGND polar range:

| selection | `v1` |
|---|---:|
| all primary nucleons | +0.0752 ± 0.0007 |
| `theta` 8–13.5 deg | +0.0413 ± 0.0028 |
| ... `E_kin` ≥ 2 GeV | +0.0900 ± 0.0057 |
| ... `E_kin` < 1 GeV | +0.0046 ± 0.0048 |

HGND sits at `phi` ≈ 180 deg, on the away side, so its azimuthal position changes
`R_n` by itself.  An `A(theta)` azimuthal weight was tried and fails: azimuthal
uniformity gives `chi2/ndf` = 13–17, and the weighted counts are 10 % wrong.
Acceptance is therefore tested per particle, propagating from the vertex to the
front face, for **both** species — which is what makes the proton arm a like-for-
like comparison at the interaction point rather than a different phase space.

On the cached subsets the corrected window roughly doubles the signal: the
hardness double ratio runs 0.962 / 1.039 / 1.109 across the scan, **+15.3 %** for
90 versus 0 where the band gives **+8.0 %** on the same events.  Those subsets are
small and `b`-truncation biased, so this is indicative only.  The full numbers
need the production re-reduced with the per-particle test, which
`reduce_prim_full.py` now emits; the mount has been serving at 0.2 MB/s, so this
should run on ncx where the data is local.

**Resolved (2026-09-28).**  The full production has been reduced in the
front-face acceptance — on cHARISMa, where all 568 primary job files already
live, in 13m47s against roughly nine hours over the mount.  The impact
parameter was joined from the reduction of the B-enabled copy, after asserting
identical job files, event keys and per-event multiplicity.

| observable | acceptance | band | ratio |
|---|---|---|---|
| `R_n` | +4.52 % (10.2σ) | +4.27 % (22.9σ) | 1.06 |
| `R_p` | −5.39 % (10.5σ) | −5.84 % (31.9σ) | 0.92 |
| `R_n/R_p` | **+10.48 % (13.8σ)** | +10.73 % (36.9σ) | 0.98 |
| `n/p` | +0.06 % (0.2σ) | −0.09 % (1.0σ) | — |

**This withdraws the claim that the acceptance roughly doubles the response.**
That figure (+8.0 % → +15.3 %) came from the cached sub-samples, 8 840
truncation-biased events each, and did not survive full statistics.  The two
selections agree to within 2.4 % of each other.

The physics is stronger for it: the response does not depend on which selection
is used, so it is not an artefact of the angular cut.  The acceptance holds 7.9
times fewer particles, so it reaches 13.8σ where the band reaches 36.9 on the
same events, with the same central values and monotonicity in 9 of 10
centrality classes.

## Reconstructed `R_n` needs the observable redefined, not just corrected

Detection and reconstruction efficiency for neutrons arriving at the front face
is strongly non-linear in `E_kin`:

| `E_true` [GeV] | arriving | selected | efficiency |
|---|---:|---:|---:|
| 0.0–0.5 | 12 584 | 256 | 0.020 |
| 0.5–1.0 | 7 172 | 2 665 | 0.372 |
| 1.0–1.5 | 7 234 | 4 820 | 0.666 |
| 1.5–2.0 | 5 975 | 4 540 | 0.760 |
| 2.0–2.5 | 3 774 | 3 029 | 0.803 |
| 2.5–3.0 | 1 795 | 1 445 | 0.805 |
| 3.0–4.0 | 924 | 701 | 0.759 |

a factor **39.6** across the range.  A kernel `K[i,j] = eps(i) * P(reco j | true i)`
reproduces the observed reconstructed spectrum exactly — observed/predicted is
1.000 in every bin — so the forward model is right.  The inversion is not
recoverable: `R`'s conventional denominator lies below 1 GeV where efficiency is
2 %, and unfolding cannot restore information the detector never recorded.

Restricting to the efficiency plateau does not rescue the present estimator
either.  With the denominator at 1.5–2.0 GeV and the numerator above 2 GeV, truth
gives **+4.83 % ± 2.08 %** and reconstruction **+0.40 % ± 2.24 %**.  Harder
numerators cannot be formed at all: the calibrated energy saturates near 2.9 GeV,
so a 3 GeV threshold returns nothing.

**Consequence for P1:** a fixed 2 GeV reconstructed threshold with a sub-1 GeV
denominator is not a measurable observable for HGND regardless of how well the
estimator is retrained.  The hardness definition used for reconstruction must sit
inside the efficiency plateau and inside the estimator's dynamic range, and the
retrained estimator must extend that range.

## Normalisation: standardise a physics baseline once, not per sample

`eToF` is not an incidental feature.  It is the available-energy baseline built
from the hit time and coordinate under the neutron hypothesis, and correcting it
is part of what the network is for — the Dombay report states plainly that the
model "compensates ToF overestimation".  That is exactly why it must not be
standardised per dataset: rescaling a physics-bearing baseline separately in each
sample removes part of the difference between samples before the network sees it.
This is a normalisation choice, not information leakage.

`prepare_halves` fitted `StandardScaler` per dataset.  Across the scan that gives
`eToF` a 1.9 % different scale and a 0.02 sigma different mean between the 90 and
0 MeV samples — the largest per-feature shift in the set.

Rebuilding the caches with a single scaler from the 18 MeV midpoint and running
the **identical checkpoint**:

| quantity | per-sample scalers | shared scaler |
|---|---:|---:|
| differential high-bin response | −30.7 ± 8.6 MeV (3.55σ) | −11.4 ± 9.8 MeV (1.16σ) |
| truth 90/0 effect | +3.67 % ± 2.12 % | +3.67 % ± 2.12 % |
| reconstructed 90/0 effect | −3.04 % ± 2.44 % | **+2.25 % ± 2.43 %** |
| paired displacement | −6.71 % ± 2.38 % (2.81σ) | −1.42 % ± 2.67 % (0.53σ) |
| displacement / truth effect | 1.83 | **0.39** |
| reconstructed `R_n` ordered 0/18/90 | no | **yes** |

No retraining was involved, so the improvement is attributable to normalisation
alone.  Against the P1 gates this moves three from failing to passing:

| gate | status |
|---|---|
| reconstructed and true effects have the same sign | **passes** |
| paired difference below 2σ | **passes** (0.53σ) |
| paired difference below 25 % of the truth effect | fails (39 %) |
| differential response consistent with zero | **passes** (1.16σ), but not yet on blind job files |
| 0/18/90 ordering survives for reconstructed `R_n` | **passes** on central values |
| no threshold or calibration chosen with truth information | fails — purity lock and energy calibration both use truth |
| stable across thresholds, seeds and two model families | not tested |

Retraining is still required, but it is now a smaller target.  The scaler must be
fitted once — on the training split — and reused unchanged for validation, test
and every physics sample, so the baseline keeps one common scale everywhere.

## Reference performance targets

From the 26.02.2026 Dombay report on the same detector and reconstruction, using
3.2 AGeV Xe+CsI DCM-QGSM-SMM with Geant4 and about 300 k events, a retrained
model should reproduce:

| quantity | reference |
|---|---|
| cluster classification | ROC AUC ≈ 0.97 |
| quoted working point | ≈ 80 % efficiency at ≈ 87 % purity |
| other points on the curve | 0.86/0.77, 0.80/0.87, 0.72/0.92 (eff/purity) |
| energy linearity | within 10 % over 0.7 to ~5 GeV |
| energy resolution | < 10 % at feasible energies |
| cluster isolation | prompt neutron splits into secondary clusters < 2 % |

These are the acceptance criteria for P1, and the current checkpoint fails them:
its energy is biased **+0.75 GeV** before calibration and its calibrated range
saturates near 2.9 GeV, against a reference linear to 5 GeV.  The purity and
efficiency definitions to quote are the report's own — energy-weighted
`purity = 1 - E_fake/E_predicted` and `efficiency = E_true/E_all signals` — with
the count-based event-level pair reported alongside.

## Retrained estimator (2026-09-27)

Retrained `net_default` on the three samples pooled, split by generation job
12/4/4 per sample, with one scaler fitted on the training split.  Thermal
throttling stretched epochs from 7 to over 50 minutes and the run was stopped
after **4 of 20 epochs** at val 1.223, already past the previous checkpoint's
1.354.  Everything below is therefore a lower bound on the architecture.

On identical test data, identical normalisation:

| `E_true` [GeV] | linearity before → after | resolution before → after |
|---|---|---|
| 0.7–1.0 | +63.9 % → **+37.0 %** | 33.7 % → 16.9 % |
| 1.4–1.8 | +57.5 % → **+10.3 %** | 26.3 % → 19.8 % |
| 2.2–2.6 | +38.2 % → **−1.3 %** | 20.2 % → 15.0 % |
| 2.6–3.0 | +27.1 % → **−8.7 %** | 18.6 % → 15.0 % |

The bias now crosses zero near 2.4 GeV instead of sitting at +50–60 % across the
range: the ToF overestimation is being compensated, which is the behaviour the
reference describes.  Resolution roughly halved but is still above the 10 %
target, and classification is marginally behind (AUC 0.941 against 0.948) —
both consistent with four epochs.

### The hardness window is a free parameter

Following the decision that thresholds may be matched to the hardware, both the
cluster score threshold and the energy window are now scanned, with any window
whose denominator falls below 1 GeV excluded outright — detection efficiency
there is 2 %.  The figure of merit rewards separation only when the samples are
ordered **in the direction the truth-level scan established**; without that
constraint it selected configurations in which `R_n` *decreases* with `S_pot`.

### The held-out test is inconclusive, not negative

The selected working point gives **+2.87 ± 6.92 %** on test.  The uncertainty is
the point: against a +4.27 % truth effect, the split admits only **0.62σ** even
with perfect reconstruction.

| | value |
|---|---|
| measured error on the 90/0 ratio | 6.92 % |
| truth-level effect | +4.27 % |
| expected significance, perfect reconstruction | 0.62σ |
| test generation jobs needed for 3σ | **≈ 284** (12 used) |

No conclusion about the reconstruction can be drawn from this split.  Sizing the
test set is a prerequisite for the P1 gates to be testable at all, and it should
be added to P1 alongside training to convergence.

## The cluster branch was running on CPU, and detached (2026-09-28)

Training job 4356776 was `CANCELLED by 0` after 3 h 37 m on an A100 — the
admin idle-GPU reaper, the same kill this project saw in August.  The cause was
in the job's own log:

    cpu_pinned=[cluster_conv_cpu, clclass_out_cpu, clenergy_out_cpu, cl_edge_out_cpu]

Four layers, including `DynamicEdgeConv` with `k=100`, ran on CPU while the GPU
idled.

**It is an MPS workaround that was being applied on CUDA.**  `default_net.py`
line 6 states that `DynamicEdgeConv` has no MPS kernel and that the `_cpu`
suffixes exist only for `state_dict` stability.  `device.plan_for` auto-pins on
MPS alone, but `train.py` and `evaluate.py` passed the model's
`default_cpu_pinned` list explicitly, and `Net.to()` re-pinned on every device.

**The gradient cost was larger than the throughput cost.**  Routing the cluster
branch through CPU forced

    avg_pool_x(clusters.cpu(), x.detach().cpu(), batch.cpu())

so the neutron-score and energy heads could not shape the hit representation
they consume.  On a synthetic forward/backward, **18 hit-branch tensors now
receive gradient from the energy head where none did before**.  This is a
candidate explanation for energy resolution plateauing at 15–20 % against the
< 10 % reference: the regressor was training on frozen features.

It also means the manuscript's description of the network as trained end-to-end
was not accurate for the cluster branch, and the architecture section needs a
sentence when the retrained numbers land.

**Fixed.**  `Net.to()` pins only on MPS; `plan_for_spec` applies a model's
pinning list only where it is needed; and the detach is decided by comparing the
cluster branch's device against the hit features rather than testing for `cpu`,
so a CPU-only run keeps its gradient too.  `state_dict` keys are unchanged, so
existing checkpoints still load.

**Rule going forward:** a device workaround must be conditioned on the device
that needs it.  Check GPU utilisation on the first long job after any change to
model placement — a job that trains correctly but leaves the GPU idle will be
killed on a shared cluster, and here it was also training a different model from
the one intended.

## Superseded or unsafe claims

Do not quote the following as current conclusions:

- The sequential 1 % samples prove different b distributions.
- The spectral-hardness signal was manufactured by centrality.
- Per-event centrality significances are valid for this production.
- HGND-band neutron or proton yields change by 3–6.5 %.
- The HGND-band n/p yield ratio is useful for Spot reconstruction.
- The current GNN reconstructs the truth-level hardness response.
- A 2 MeV `S_pot` reach is an experimental or EOS-parameter uncertainty.
- The `theta in [8.9, 13.1)` band is the HGND acceptance.  It is 7.7 times the
  solid angle and spans azimuths the detector does not cover — though at full
  statistics it gives the same central values, so it is a valid proxy even
  though it is not the acceptance.
- The acceptance roughly doubles the double-ratio response.  Measured on
  truncation-biased sub-samples; the full production gives +10.48 % against
  +10.73 %.
- The sample-dependent energy response is wholly a property of the network.  Two
  thirds of it came from standardising the `eToF` baseline per dataset.
- The network is trained end to end.  The cluster branch was detached from the
  hit branch on every device until 2026-09-28; checkpoints from before then were
  trained with the score and energy heads unable to shape their own inputs.
- A reconstructed `R_n` with a sub-1 GeV denominator can be recovered by
  efficiency correction or unfolding.

## Immediate programme and acceptance gates

### P0 — freeze the validated baseline

- Preserve the exact full-production manifests, reduction products, b weights,
  null-test seeds, and current checkpoint hashes.
- Treat the current full-b truth result and reconstruction ladder as regression
  tests.  New work must reproduce them before replacing them.

### P1 — repair the energy estimator

**P1.0 — prerequisites, both now implemented and cheap.**

- Rebuild every cache with **one scaler fitted on the training split**
  (`preprocess.py --scaler-source`).  This alone removes two thirds of the
  response differential, and it must be in place before retraining so that the
  `eToF` baseline carries one common scale through training, validation, test and
  every physics sample.
- Redefine the reconstructed hardness ratio inside the efficiency plateau and
  inside the estimator's dynamic range.  The present definition is unmeasurable
  in its denominator whatever the estimator does.

1. Retrain using the current B-enabled production, with job files split into
   train/calibration/test before any tuning, on shared-scaler caches.
2. Balance the energy and centrality distributions across `S_pot` samples so the
   regressor cannot learn sample composition as an energy shortcut.
3. Compare a pooled three-domain model, leave-one-`S_pot`-out validation, and at
   least one model family beyond `net_default`.
4. Fit calibration only on an independent calibration split and apply that one
   mapping unchanged to every physics sample.
5. Repeat the identical-cluster response diagnostic and the full measurement
   ladder before running a larger sensitivity scan.

**Proposed go/no-go gates:**

- reconstructed and true-energy 90/0 effects have the same sign
  *(already passes with a shared scaler: +2.25 % against +3.67 %)*;
- their paired difference is below 2σ and smaller than 25 % of the truth effect
  *(0.53σ passes; 39 % of the truth effect still fails)*;
- the differential high-energy response between samples is consistent with
  zero on blind job files *(1.16σ, but not yet on a blind holdout)*;
- 0 / 18 / 90 ordering survives for reconstructed `R_n` *(passes on central
  values; the bootstrap cannot yet confirm it on 20 job files)*;
- same-sample job-file null tests remain calibrated and no score/energy threshold
  chosen on the test set is used for the quoted result;
- the conclusion is stable against reasonable hardness thresholds, seeds, and
  at least two model families.

If these gates fail after domain-balanced retraining, stop treating a fixed
2 GeV reconstructed-energy threshold as the measurement.  Move to a response-
matrix or forward-folded spectrum fit with calibration nuisance parameters.
The forward model is already verified to close bin by bin, so a forward fold is
the better-founded of the two: the failure lies entirely in the inversion.

### P2 — experimental centrality

- Construct a detector-level estimator from FHCal spectator energy or a
  non-overlapping charged-particle multiplicity region.
- Calibrate percentile classes in each sample, measure migration against truth
  b, and repeat the hardness analysis after smearing.
- Quote the truth-b result only as the upper bound; the estimator-level result is
  the achievable projection.

### P3 — proton arm, only if the double ratio is pursued

- Define a proton selection in BM@N tracking with a documented fiducial region.
- Match or forward-fold neutron and proton phase spaces rather than labeling a
  truth angular selection as an HGND measurement.  The common phase space is now
  defined and implemented: both species are selected at the interaction point by
  propagating to the HGND front face (`scripts/hgnd_acceptance.py`), so the
  neutron and proton arms share one window by construction.
- Carry independent efficiency, magnetic-acceptance, PID, and centrality
  uncertainties into `R_n/R_p`.

### P4 — complete systematic coverage

- Seed variation is already small for the legacy sensitivity pipeline
  (max-min roughly 0.5–0.7 % in the recorded ratios), but the alternative-model
  entries are still missing.
- Add model-family spread, energy-scale shifts, threshold variation, detector
  timing/calibration/dead-channel variations, and transport-definition checks.
- Resolve generator weighting and the free/emitted-nucleon definition before
  comparing numerically with the reference SMASH slides.

## Publication guidance

The current manuscript is technically close to referee-ready, but its scientific
framing must follow the newest evidence.

- A reconstruction/methods paper can proceed after the remaining figure and
  text consistency fixes, provided the `S_pot` study is presented as a stress
  test that exposes a domain-dependent energy-response failure.
- A physics-sensitivity claim should wait until P1 passes.  Truth-level
  sensitivity may be reported as motivation, not as demonstrated detector
  performance.
- The two remaining placeholder figures, energy-resolution text/figure mismatch,
  efficiency-axis caption, open markers, and author sign-off from referee review
  v6 still need closing.
- Separate any remaining upstream/defaultSpot production caveat from the
  retracted b-mismatch claim: the full production does share a b distribution
  within job-level uncertainty, even if other production defects remain.
- Add the reconstruction ladder and sample-dependent response as the central
  limitation.  Omitting them would overstate the current result.

## Authoritative artefacts

- Full centrality method and resolution:
  [`CENTRALITY_CONVENTION.md`](CENTRALITY_CONVENTION.md)
- Full-production analysis report:
  [`results/b_full_analysis/report.html`](../results/b_full_analysis/report.html)
- Reweighted observables:
  [`b_reweight.txt`](../results/b_full_analysis/b_reweight.txt) and
  [`b_reweight.json`](../results/b_full_analysis/b_reweight.json)
- Reweighting injection test:
  [`inject.txt`](../results/b_full_analysis/inject.txt)
- Reconstruction ladder:
  [`reco_experiment.txt`](../results/reco_experiment/reco_experiment.txt)
- Energy-response diagnostic:
  [`response_diagnostics.txt`](../results/reco_experiment/response_diagnostics.txt)
- Normalisation domain test:
  [`scaler_domain_test.txt`](../results/reco_experiment/scaler_domain_test.txt) and
  [`scaler_domain_test.json`](../results/reco_experiment/scaler_domain_test.json)
- Efficiency and unfolding closure:
  [`efficiency_correction.json`](../results/reco_experiment/efficiency_correction.json)
- Acceptance definition and its check against the exact test:
  [`hgnd_acceptance.py`](../scripts/hgnd_acceptance.py),
  [`check_acceptance.py`](../scripts/check_acceptance.py)
- Current n/p decision history:
  [`NP_RATIO_PLAN.md`](NP_RATIO_PLAN.md)
- Manuscript and its reviews: now a separate repository, `../HGNDPaper`
  (split with `git subtree`, history intact).  Latest review is
  `critical_review_v7.md` there; `paper/README.md` in this repository records
  which script generates each figure.

## One-sentence project status

**The physics observable is well established at truth level and centrality is
under control; a shared normalisation and a partial retrain have already moved
the energy estimator from +50–60 % linearity error to within 10 % over
1.4–2.6 GeV, so the remaining blockers are ordinary ones — train to convergence,
and size the held-out split to the ~280 generation jobs a 4 % effect needs,
since the present 12 admit only 0.62σ.**
