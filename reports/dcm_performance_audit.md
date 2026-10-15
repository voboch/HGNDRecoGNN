# Why DCM reconstruction misses the Dombay benchmarks — audit and fixes

2026-10-15. Companion to `reports/dombay_reference_audit.md`, which compared
the *pipeline* against the reference notebook. This one asks the narrower
question: the numbers do not match, so what is actually responsible, and what
should change.

Everything below is measured from artifacts already in the repo
(`results/final_eval/`, the SMASH defaultSpot cache, the checkpoint in
`checkpoints_hpc/`). **No training run was performed for this audit**, so the
expected gains from the changes in §4 and §5 are reasoned, not demonstrated.

---

## 1. The gap, stated precisely

| | ROC AUC | purity(E) | efficiency(E) |
|---|---|---|---|
| Reference (Dombay, DCM) | ~0.97 | 0.87 | **0.80** |
| DCM native (`final_eval/dcm_native`) | 0.9515 | 0.869 | **0.663** |
| DCM transfer, SMASH checkpoint | 0.9525 | 0.867 | **0.688** |
| SMASH, own domain | 0.9582 | 0.873 | 0.715 |

At matched purity the efficiency shortfall is **11–14 points**. AUC is short by
0.016–0.019.

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

**Checkpoint selection compounds it.** `save_best_on` defaulted to `'train'`,
so the epoch with the lowest *training* loss is kept. With train 1.268 below
val 1.354 the model is overfitting, and selecting on train deliberately keeps
the most overfit epoch.

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

**Eight message-passing layers, no skip connection.** At the reference
`num_layers=8` the hit branch is 2 × EdgeConv → 4 × SAGEConv → 2 × GraphConv
with plain ReLU between, which is the depth at which GNN node features
oversmooth. Normalisation is also inconsistent: `BatchNorm` is applied only
across the SAGEConv block, not to the EdgeConvs or the final GraphConvs.

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

1. **Find or re-run job 4366196** (batch 256, 25 epochs) and evaluate it. Until
   this exists, the premise of this audit — that DCM native underperforms — is
   untested at the reference settings.
2. **Re-select the existing checkpoint on validation loss.** Free: no retrain,
   just `--save-best-on val` on the next run, and it tells you how much the
   `'train'` default was costing.
3. **Budget before architecture.** 30 epochs, `--scheduler cosine
   --warmup-epochs 3`, batch 256. This is the configuration the reference's
   2.5× step advantage implies, plus a schedule that does not idle.
4. **Then the cluster features**, in this order of expected return:
   `--cluster-size-feature` (AUC 0.729 standalone), then
   `--cluster-pooling mean_max_sum`, then `--hit-residual`.
5. **Re-weight the loss** toward `cluster_cls` if steps 3–4 leave a gap. It
   currently carries ~16 % of the total; the other heads are auxiliary.
6. **Only then** revisit `DynamicEdgeConv(k=100, aggr='sum')` and the
   inconsistent normalisation.

Changes from steps 3 onward alter the model or the optimisation, so each needs
its own evaluation against the held-out split rather than being stacked
blindly. Note also that `train_frac = 0.5` follows the reference and leaves
half the sample unused for training; raising it would change the comparison
basis and should be a separate, declared decision.
