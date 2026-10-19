# Why DCM reconstruction misses the Dombay benchmarks — audit and fixes

> Review update, 2026-10-16: [the reconstruction review](reconstruction_review_20261016.md)
> records defects in the proposed experiments. The model-factory failure
> (DCM-01) and scheduler warning (DCM-11) were fixed in `53ff00d`.
> A subsequent [live cluster check](dcm_experiment_launch_20261016.md) found
> baseline job 4366196 and evaluation 4366562 completed on 2026-09-30;
> the claim below that their output was missing is superseded.
> The efficiency/purity comparison uses
> different estimators and selections from the reference (DCM-02), and the
> proposed baseline differs from the named reference notebook (DCM-03).
> The numerical efficiency gap and causal explanations below require those
> issues to be resolved. This original audit is retained as the analysis record.

2026-10-15. Companion to `reports/dombay_reference_audit.md`, which compared
the *pipeline* against the reference notebook. This one asks the narrower
question: the numbers do not match, so what is actually responsible, and what
should change.

Everything below is measured from artifacts already in the repo
(`results/final_eval/`, the SMASH defaultSpot cache, the checkpoint in
`checkpoints_hpc/`). **No training run was performed for this audit**, so the
expected gains from the changes in §4 and §5 are reasoned, not demonstrated.


> **Superseded in part.** `reports/reconstruction_review_20261016.md` reviewed
> this document and found four P1 issues. I reproduced all four independently;
> see §0. In short: the 11-14 point efficiency gap in §1 is **not established
> for a common estimator** (DCM-02), the `ref` rung is **not** the reference
> architecture (DCM-03), and the depth argument in §4 points the wrong way.
> The measurements in §2 and the cluster-size measurement in §4 stand.

---

## 0. What the review overturned, and what I verified myself

Each item below I reproduced rather than accepted. Fixed where noted.

**DCM-01 (fixed).** The three new architecture flags never reached the model.
`scripts/train.py` maps them into `arch_kwargs` correctly and the names match
`Net.__init__`, but `registry.get()` forwards every kwarg to the `net_default`
factory, whose signature accepted only `hidden_channels` and `num_layers`.
Reproduced: `TypeError: _factory() got an unexpected keyword argument
'cluster_size_feature'`. Every ladder rung above `budget` would have died
before the first batch. My own tests missed it because they checked both
endpoints and not the route between them. The factory now forwards
`**arch_kwargs`, and three tests cover the registry hop — verified to fail on
the old signature and pass on the new one.

**DCM-11 (fixed).** My unreachable-milestone warning indexed
`sorted(milestones)[1]`, so a single-milestone schedule — which the CLI
accepts via `--scheduler-milestones 25` — raised `IndexError` instead of
warning. Reproduced. The message is now computed from the milestones that
actually fire, and is tested at 0, 1 reachable, 1 unreachable, 2 and 3
milestones.

**DCM-02 (confirmed; invalidates §1's headline).** `threshold_scan` in
`scripts/reco_performance.py` computes
`efficiency_energy = sum(e_true of selected signal) / sum(e_true of all
signal)` over every cluster above threshold. The reference's slide 16 is
labelled *3 best clusters per event*, and the evaluation notebook keeps one
hit per cluster, takes three clusters per `Row`, histograms 0-6 GeV and forms
`eff = counts1 / counts0` with `counts0` over MC neutrons after
`groupby(['Row','Id'])`. That is a per-neutron **count** ratio under a
per-event cap; ours is a per-cluster **energy** ratio with no cap. **The
11-14 point gap in §1 is therefore not a measurement of the same quantity**
and should not be quoted until one protocol is applied to both. The energy
resolution estimators also differ (notebook: std of the relative error;
ours: half the 16-84 % interval).

**DCM-03 (confirmed, and sharper than the review states).** I read cell 51 of
the reference notebook directly. At `num_layers=8` it builds 8 convolutions
and 4 BatchNorms, but `forward` does
`for conv, batch_norm in zip(self.convs[:-3], self.batch_norms)` — four
iterations over `convs[0..3]` — with `x = batch_norm(x)` **commented out**,
then calls `convs[-2]` and `convs[-1]`. So `convs[4]` and `convs[5]`, two
SAGEConvs, **are never called**: the reference executes **6 convolutions with
no normalisation** and carries two layers of dead parameters. It also has no
cluster-link head (it returns 5 values, not 6) and the reference feature list
has 9 entries where this cache has 10 (`SurfaceHit`).

`models/default_net.py` executes 2 EdgeConv + 4 SAGEConv **with BatchNorm** +
2 GraphConv. Matching `num_layers=8` does not make the executed architectures
the same, and `VARIANT=ref` is therefore *this pipeline's baseline at the
reference's hyperparameters*, not a reproduction of the reference model.

**DCM-04 (confirmed).** `scripts/evaluate.py --split test` rebuilds the split
with the same seed and `train_frac`, so the "test" half is the same half
`fit()` validates on and `save_best_on='val'` selects against. On top of that
`reco_performance.py` picks the working point by
`(scan.purity_energy - 0.87).abs().idxmin()` on the data being evaluated.
Epoch, threshold and — across the ladder — architecture are all chosen on the
set the result is read from. Every number in §1 inherits that.

**DCM-05 (confirmed by reproduction).** `split_dataset` calls `random_split`
over graph indices with no grouping by `Row`, while the dataset stores the top
and bottom hemisphere of one collision as two graphs sharing a `Row`. Loading
`shard_0.pt` and `shard_98.pt` from the defaultSpot cache gives 819 complete
top/bottom pairs; under the production seed-42 split **411 of them (50.2 %)
have one hemisphere in train and the other in test** — exactly independent
assignment. (The review reports 403, splitting the full dataset rather than
this subset; same mechanism.)

**DCM-08 (confirmed; my design error).** The `budget` rung moved epochs,
schedule shape and warmup together, so its result could not attribute a
difference to any one of them. The ladder now has separate `epochs`, `sched`
and `warmup` rungs, with `budget` kept and labelled as a package. The review
is also right that the **2.5x step-count figure describes the reference
against this pipeline's former default (25 @ 256 vs 20 @ 512), not the
`ref`→`budget` transition**, which at fixed batch is 30/25 = 1.2x.

**DCM-06, 07, 09, 10 — open.** Seed set after model construction; the
scaler-mode caches re-randomise the 0.15 ns time smearing; resume can pair
`last.pt` weights with an `optim.pt` from a later epoch; the checkpoint
directory did not identify `SCALER`, `BATCH` or `NUM_SHARDS`. I fixed only the
last of these (it is in the script I added). The rest are pre-existing and
each needs its own change; `reports/reconstruction_review_20261016.md` is the
tracking document.

---

## 1. The gap, stated precisely

| | ROC AUC | purity(E) | efficiency(E) |
|---|---|---|---|
| Reference (Dombay, DCM) | ~0.97 | 0.87 | **0.80** |
| DCM native (`final_eval/dcm_native`) | 0.9515 | 0.869 | **0.663** |
| DCM transfer, SMASH checkpoint | 0.9525 | 0.867 | **0.688** |
| SMASH, own domain | 0.9582 | 0.873 | 0.715 |

At matched purity the efficiency shortfall reads as **11–14 points** and AUC as
short by 0.016–0.019.

**Read with §0 (DCM-02) first.** Our efficiency is an energy-weighted
per-cluster ratio over all clusters above threshold; the reference's is a
per-MC-neutron count ratio under a *3 best clusters per event* cap, binned in
energy. The two numbers in the table are not the same quantity, so the
shortfall is **not established** until one protocol is applied to both. The
ROC AUC comparison is less exposed to the cap and the denominator, but the
sample definitions behind it still need checking. And by DCM-04 every row
below the reference line is read from the same split used to pick the epoch
and the threshold.

The one structural oddity: **in-domain DCM training does not beat out-of-domain
transfer.** Native is slightly worse on AUC (0.9515 vs 0.9525) and clearly
worse at the working point (0.663 vs 0.688). Training on the target generator
should win. That it does not is the thread worth pulling, and §3 is where it
leads.

## 2. Three explanations that do not survive contact with the data

**It is not the efficiency convention.** Count-based efficiency is *lower* than
energy-weighted everywhere on the scan — 0.576 at purity(N) 0.852 against
0.663 at purity(E) 0.869. If the reference quoted counts, the gap would widen.
The energy-weighted numbers are already our most flattering choice.

**It is not the energy head starving the classifier.** `default_net_loss` gives
all five heads weight 1.0, and a raw MSE in GeV² next to four BCE terms looks
like it should dominate. It does not: the energy term is masked to true
clusters (`cl_mask = graph.cllabel.bool()`), and measured on the saved
predictions it is **0.068 GeV²** against a cluster-classification BCE of
**0.1995** — about 5 % of the 1.268 total. Unmasked it would be 4.18 GeV² and
this would be the whole story, but it is masked. *(I expected this to be the
answer and it is not.)*

What the same arithmetic does show: cluster classification, the metric of
interest, carries roughly **16 %** of the total loss. The hit, link and
cluster-link heads take the rest.

**It is not the `Ekin > 0.3` label cut flattering the reference.** Divergence 2
of the companion audit notes our positive class adds this cut (confirmed in the
cache: the smallest signal `e_true` is exactly 0.300). But the cut *removes*
soft, hard-to-reconstruct neutrons from the denominator, which can only push
our efficiency **up** relative to a reference that keeps them. It makes the
gap harder to explain, not easier.

## 3. What the evidence actually points at: training budget

**The headline number may not be the reference configuration at all.**
`reports/dombay_reference_audit.md` records the native reference-settings run —
H=512, L=8, **batch 256, 25 epochs**, Adam 1e-3, 50/50 split, seed 42, job
4366196 — as *"running… at epoch 2 of 25"*. No evaluation of that run exists in
this repo. `results/final_eval/dcm_native/performance.json` and
`results/status_brief_20261001/dcm/dcm_native.json` are the same earlier run,
byte-identical in AUC (0.9515279254854021) and both dated 30 September.

So **"DCM native fails to reach the Dombay benchmarks" has not been tested at
the settings the benchmarks were produced with.** Recovering job 4366196's
output, or re-running it, is the cheapest decisive step available and should
precede any architectural conclusion.

**The default budget is ~2.5× smaller than the reference's.** At fixed data,
optimiser steps scale as epochs / batch size:

```
reference :  25 epochs @ batch 256
default   :  20 epochs @ batch 512     ->  (25x512)/(20x256) = 2.5x fewer steps
```

**The schedule then spends a quarter of that smaller budget at a rate too low
to matter.** `scheduler_milestones = (10, 15, 25)` with `epochs = 20`:

| epoch | 0–9 | 10–14 | 15–19 | 25 |
|---|---|---|---|---|
| LR | 1e-3 | 1e-4 | **1e-5** | milestone never reached |

The third milestone is dead code at the default epoch count. The schedule was
written for the reference's 25-epoch budget and a shorter run silently
truncates it.

**And the model was still improving when that happened.** The converged SMASH
checkpoint records `epoch=18` of 20 with `train_loss=1.268`, `val_loss=1.354`.
Best epoch 18 of 20, at LR 1e-5, is the signature of a run stopped early on a
schedule that decayed early — not of a converged model.

**Checkpoint selection compounds it — for the SMASH line, not necessarily for
DCM.** `save_best_on` defaulted to `'train'`, so the epoch with the lowest
*training* loss is kept; with train 1.268 below val 1.354 the model is
overfitting and that policy deliberately keeps the most overfit epoch. The
SMASH checkpoint in `checkpoints_hpc/` is demonstrably one of those.

`slurm/train_dcm.hpc.sbatch`, however, already passes `--save-best-on val`
(and batch 256, 25 epochs, `TWINDOW=1.0`), and says so in its header. The
script is committed only as of 2026-10-06, after the 30 September DCM result,
and the repository does not record which settings that earlier run used — so
whether `'train'` selection touched the 0.9515 number is **undetermined**, not
established. The changed default in §5 matters for every other entry point;
for this job it was already correct.

## 4. Architectural findings

**The cluster branch cannot see cluster size, and size is the single most
informative scalar available.** The branch is fed `avg_pool_x`, the *mean* of
its hits' embeddings, which is invariant to how many hits were pooled.
Measured on the SMASH defaultSpot cache (15 728 clusters, 6 shards):

| | signal | background |
|---|---|---|
| mean hits per cluster | 6.53 | 2.98 |
| single-hit clusters | 15.5 % | 48.5 % |

**Cluster size alone separates the classes at ROC AUC 0.729.** For comparison,
the mean of the first hit feature — the kind of quantity the pooling does
preserve — gives 0.459, i.e. nothing. Size is recoverable indirectly through
the hit graph, but it is not *represented*, and the pooling operator discards
the count by construction.

**There is an empty slot for exactly this.** `data/graph_dataset.py:194` sets
`graph['clusters'].x = torch.zeros((n_clusters, 1))` — allocated, never
filled, and never read by `models/default_net.py`. Measured AUC of that tensor
as it stands: 0.500.

**Eight message-passing layers, no skip connection — but the reference runs
six and normalises none of them.** Our hit branch is 2 × EdgeConv → 4 ×
SAGEConv (each followed by `BatchNorm`) → 2 × GraphConv. The reference's
`forward` reaches only `convs[0..3]` and then the last two, with its
`batch_norm` call commented out: **6 active convolutions, no normalisation,
two SAGEConvs never called** (§0, DCM-03).

So we are *deeper and more normalised* than the model the ~0.97 came from.
Depth is therefore unlikely to explain a deficit **relative to the
reference** — if anything, reproducing it means going shallower and dropping
BatchNorm. `--hit-residual` is motivated against our own 8-layer stack, which
is a separate question from closing the reference gap, and it is last in the
order of work for that reason.

**`DynamicEdgeConv(k=100, aggr='sum')`** aggregates over a neighbourhood whose
size varies with the event's cluster multiplicity, with no normalisation, so
the cluster representation scales with multiplicity. Not changed here, but it
is the next thing to look at.

## 5. What was changed

All architecture flags and all new schedules default to the current behaviour,
so every number in `reports/` reproduces unchanged and existing checkpoints
still load. The one changed default is called out below.

**Architecture** (`models/default_net.py`, opt-in, no cache rebuild — the size
is computed from the hit→cluster assignment in the forward pass):

- `--cluster-pooling mean_max_sum` — concatenates max and sum beside the mean.
- `--cluster-size-feature` — appends `log1p(hit count)`.
- `--hit-residual` — residual connections across the SAGEConv stack, applied
  where the widths already match. Parameter-free.

**Schedule and optimisation** (`training/train.py`, `scripts/train.py`):

- `--scheduler cosine|plateau|multistep` (`multistep` remains the default).
- `--warmup-epochs`, `--warmup-factor`, `--lr-min`, `--plateau-patience`.
- `--grad-clip-norm` exposed on the CLI.
- A warning when a milestone is `>= epochs`, naming the rate the run will
  actually finish at.
- The scheduler now steps *after* validation, so `plateau` can see the metric
  it decays on.
- **Changed default: `save_best_on` is now `'val'`.** Pass
  `--save-best-on train` to reproduce any run in `reports/`.

Honest note on cosine: over the same 20 epochs its total LR is *the same* as
multistep's (1.05e-2 both). It does not add budget, it redistributes it — 2.1e-4
at epoch 14 where multistep is at 1e-4, and a real anneal to 6e-6 at the end
where multistep sits flat at 1e-5. The budget itself only grows by extending
the run: cosine over 30 epochs with 3 warmup epochs is 1.44× the total LR of
multistep over 20.

**A bug found while testing the above.** `avg_pool_x` does not index its output
by the raw cluster id — it relabels ids to be consecutive
(`consecutive_cluster`) and returns one row per unique cluster in the new
order. The first implementation of the extra pooling scattered with the raw
ids, which misaligns every row whenever ids are non-consecutive, silently
whenever `max(id)` happens to fit inside the row count. Now relabelled
identically before scattering, and covered by a test that fails on the old
behaviour.

**Tests.** `tests/test_default_net_arch.py` (9) and `tests/test_scheduler.py`
(10). The architecture suite includes loading the real 12.5 M-parameter
`checkpoints_hpc` state_dict into the patched model and asserting no missing or
unexpected keys. Suite total 59, up from 40.

## 6. Recommended order of work

Cheapest decisive test first. Steps 1–2 may close the question without any
model change.

0. **Reconcile the evaluation protocol first (DCM-02).** Fix the cluster
   selection, the MC-signal definition, the denominator, the energy range, the
   binning and the resolution estimator, then apply one protocol to both the
   reference predictions and ours. Until that is done no rung of the ladder
   can be compared with 0.80/0.87, and §1's shortfall is not a measurement.
   Keeping our energy-sum metrics under separate names is fine.

0b. **Separate validation from test (DCM-04)** and **group the split by
   collision (DCM-05)**. Half of all collisions currently have one hemisphere
   in each split, and the epoch, the threshold and the choice of rung are all
   selected on the set the number is read from. Both must be settled before a
   ladder of model variants means anything, because the ladder selects on
   exactly that set.

1. **Find or re-run job 4366196** (batch 256, 25 epochs) and evaluate it. Until
   this exists, the premise of this audit — that DCM native underperforms — is
   untested at the reference settings. Note this gives the *pipeline's*
   baseline at the reference hyperparameters, not the reference model
   (DCM-03).
2. **Note that validation selection is already in place for this job.**
   `train_dcm.hpc.sbatch` passes `--save-best-on val`, so there is nothing to
   change on the DCM path. The new `'val'` default in §5 affects the SMASH
   entry points, where the committed checkpoint *is* train-selected; quantify
   it there, not here.
3. **Budget before architecture, one factor at a time.** `VARIANT=epochs`
   (30 epochs, schedule untouched), then `sched` (cosine, no warmup), then
   `warmup`. The `budget` rung runs all three together and its result may only
   be reported as a package. At fixed batch, `ref`→`epochs` is 1.2× the
   optimiser steps; the 2.5× figure in §3 is the reference against this
   pipeline's former 20 @ 512 default, not a step on this ladder.
4. **Then the cluster features**, in this order of expected return:
   `--cluster-size-feature` (AUC 0.729 standalone), then
   `--cluster-pooling mean_max_sum`, then `--hit-residual`.
5. **Re-weight the loss** toward `cluster_cls` if steps 3–4 leave a gap. It
   currently carries ~16 % of the total; the other heads are auxiliary.
6. **Only then** revisit `DynamicEdgeConv(k=100, aggr='sum')` and the
   inconsistent normalisation.

### Concrete commands

Steps 3-4 are a ladder in `slurm/train_dcm_improve.hpc.sbatch`, one rung per
submission, each adding to the one before. Run them from a cHARISMa login node
and **evaluate between rungs** — if `budget` closes the gap there is no reason
to touch the architecture:

```bash
# step 1 - baseline at the reference settings (existing script, unchanged)
sbatch slurm/train_dcm.hpc.sbatch

# step 3 - budget and schedule only
VARIANT=budget   sbatch slurm/train_dcm_improve.hpc.sbatch
CKPT=/scratch/$USER/hgnd/checkpoints/dcm_improve_budget_tw1p0_h512l8_30ep/model.pt     sbatch slurm/eval_dcm.hpc.sbatch

# step 4 - cluster features, in increasing order of risk
VARIANT=size     sbatch slurm/train_dcm_improve.hpc.sbatch
VARIANT=pool     sbatch slurm/train_dcm_improve.hpc.sbatch
VARIANT=residual sbatch slurm/train_dcm_improve.hpc.sbatch
```

Each variant writes to its own checkpoint directory, so rungs cannot overwrite
each other, and the script refuses an unrecognised `VARIANT` rather than
silently training the default. `EPOCHS` and the schedule flags are fixed per
variant inside the script, so a submission cannot half-apply a rung.

Changes from steps 3 onward alter the model or the optimisation, so each needs
its own evaluation against the held-out split rather than being stacked
blindly. Note also that `train_frac = 0.5` follows the reference and leaves
half the sample unused for training; raising it would change the comparison
basis and should be a separate, declared decision.
