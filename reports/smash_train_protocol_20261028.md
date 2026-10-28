# Training on corrected SMASH: protocol

Written 2026-10-28 on branch `smash-train-20261028` (from 00411bc), before any
SMASH-trained model or held-out evaluation exists. Requested by the user on
2026-10-28 after confirming that no network had been trained on the corrected
SMASH production. The plan behind it was drafted by an independent planning
agent from a read-only inventory of the exports on the cluster.

## Question and consumer

On held-out corrected-SMASH files, does the native GraphSAGE trained on
deduplicated SMASH beat the frozen DCM transfer, with the same graph view,
labels and recipe, in (1) classification, (2) the energy response, including the
potential-dependent shift Δbias(90−0) measured by a reviewer (review v9 M4/C2),
and (3) the closure of the reconstructed neutron hardness R_n against truth?
The consumers are the path-B decision of review v9 (whether retraining is the
remedy and which component to repair first) and items C2/M4. Manuscript scope
A is unchanged; no number from this campaign enters the manuscript.

## What the inventory changed

Almost every detector export on /scratch is a byte copy. Measured on
2026-10-28: zeroSpot has 1,546 non-empty exports in 195 size groups, each
containing one of the 195 truth-bound files (two 9-member groups verified
byte-identical); defaultSpot 206 groups, 20 without a bound member; bigSpot
183 groups, 132 without a bound member. Unique non-empty content is at most
about 584 files, 432 of them already bound to primary truth. A design with all
truth-bound files in test and training on the rest is therefore impossible
(zeroSpot has no other unique content). The campaign uses 3-fold
cross-fitting instead: every truth-bound file is scored only by models that
never saw it. Older SMASH caches (`_v3`, `v2fix`) were built over the copies,
with a 1.5 ns window and per-dataset scalers, and are not used; earlier
SMASH-trained numbers from them are open to duplicate leakage.

## User decisions (2026-10-28)

Three folds submitted together; up to three concurrent GPU tasks; one
pre-approved retry per task for an infrastructure failure (EACCES, node
failure) on a different node, resuming from the last mirrored epoch, never for a
scientific or numerical failure; 30 epochs with the inherited recipe; unbound
unique contents bound to truth inside the census; the SMASH arm normalised by
a scaler fitted on the fold's training files; primary contrast B1 vs A1; no
net_default in this campaign.

## Arms

One variable: which training data produced the weights. Shared by all arms:
graph view r = 3.6 cells, 1.0 ns, at most 32 neighbours
(`data/smash_transfer.py`), the ten features, the native signal label (a hit of
a neutron with E_kin > 0.3 GeV within the time-of-flight coincidence; not the
primary neutron) and the student GraphSAGE recipe imported from
`scripts/student_pipeline.py`.

| arm | weights | score cut | energy |
|---|---|---|---|
| A0 | frozen DCM (seeds 42/123/456) | DCM validation cut | uncalibrated (continuity) |
| A1 | frozen DCM | chosen on fold validation | calibrated on fold validation |
| B0 | trained on fold-train SMASH | chosen on fold validation | uncalibrated |
| B1 | trained on fold-train SMASH | chosen on fold validation | calibrated on fold validation |

The primary contrast is B1 vs A1: both receive the same validation tuning
(methodology §6). B is also scored with the inherited composite selector
(B-comp); A1 vs B-comp isolates the selector. Normalisation: the frozen DCM
transform for A, one StandardScaler per detector side fitted on fold-train
files pooled over potentials for B. A and B packs must hold identical
candidates, labels and targets.

## Census and split

Per tag: SHA256 of every hits and vacs export (the 432 pinned hashes must
match), header-only classification, byte groups, raw pre-jitter event
signatures and hashed direct-primary records, exact binding of unbound unique
contents to the tag's primary files, cross-potential overlap check. Groups are
the connected components of byte identity, shared event signatures, the same
primary file and shared direct-primary records; the content with most events
is kept, ties to the lexicographically first name; a dropped content that is
not a subset of the kept one stops the stage. Truncated partial copies exist
and are caught at event level.

Per potential, groups are ranked by
`sha256("smash-train-split-v1:" + tag + ":" + hits_sha + vacs_sha)` and
assigned to six parts by rank modulo 6 (the salt is an arbitrary convention).
Fold k: test = P_{2k} ∪ P_{2k+1}, truth-bound groups only; validation =
P_{(2k+2) mod 6}; training = the other three parts. The 1/2 : 1/6 : 1/3
fractions trade training data against test statistics; the projected paired SD
of the closure versus test fraction is plotted with the split manifest. The
three potentials are pooled in training; the potential is never an input. A
group-level disjointness assertion runs per fold before packing.

## Training, selection and calibration

student_sage, H128/L4, dropout 0.1, Adam 1e-3, weight decay 1e-5, batch 256,
30 epochs, step decays at epochs 10/15/25 by 0.1, loss BCE + 0.5 SmoothL1,
population validation aggregation, seeds 42/123/456, a checkpoint every epoch.
All inherited and unexamined, except the per-head selector.

Selection on fold validation only: e_cls = argmin of the validation population
BCE; e_en = argmin of the validation SmoothL1 on positives; earliest epoch on
ties. Scores come from e_cls, energies from e_en; B-comp uses the composite
argmin. The per-head selector exists because the DCM GraphSAGE validation BCE
was minimal at epoch 0 in all three seeds while the energy loss was minimal at
epochs 3, 4 and 19. Score cut: the first of 401 values in [0.001, 0.999]
reaching count purity 0.70 on validation (inherited convention: a higher
purity trades efficiency for background rejection). Energy calibration: the
inverse of the median e_pred response in the inherited E_true bins between 0.3
and 6.5 GeV, monotone, one function for all potentials. Cut and calibration
are written to `complete.json` before any test pack is opened.

## Evaluation (each fold's test opened once)

- Classification: ROC-AUC, AP, purity and efficiency at the cut, per potential
  and pooled, by cluster size.
- Energy: bias, MAE, median response and half 16–84 % width per E_true bin, on
  all MC-positive clusters and on selected positives; Δbias(90−0) and (18−0)
  per bin and in the reviewer's [1.75, 3.0) GeV window, with the within-bin
  mean E_true difference as a control.
- R_n ladder T0–T2, C1–C4 on the truth-bound test files, primary denominator
  [0.3, 1) GeV from the exact-window counters of aggregate 4453896, the
  E < 1 GeV variant reported too; endpoint and midpoint; seeds separate; arms
  paired by shared whole-file draws (400 replicates); gates G1–G6 of the
  retrospective protocol, same-sample null and injection per arm; B − A per
  rung and step with paired SD. Per-fold and pooled out-of-fold values (the
  pooled C rungs mix three models per seed).

## Predictions (committed before any training)

| quantity | predicted |
|---|---|
| A0 ROC-AUC on the test subset | 0.90 ± 0.01 |
| B1 ROC-AUC | 0.915–0.95; Δ(B1 − A1) +0.015 to +0.05 |
| \|AUC_val − AUC_test\| for B | < 0.005 (larger suggests leakage) |
| B test purity at the cut | 0.70 ± 0.01 |
| B efficiency at purity 0.70 | 0.58–0.68 (A0 0.55) |
| B0 energy bias | \|bias\| ≤ 25 MeV, seed SD ≤ 20 MeV (A0 −48 ± 53 MeV) |
| B0 MAE | 120–160 MeV (A0 ≈168 MeV) |
| B1 calibrated median response | within ±2 % for 0.3–3 GeV |
| Δbias(90−0) in [1.75, 3) GeV | A0 −10 ± 3 MeV (reproduces M4); B −6 to +4 MeV |
| selected epochs | e_cls in 2–20 (epoch 0 in at most one seed); e_en in 5–29 |
| fold-0 T1 endpoint | +2.5 to +7 % |
| C4/T1 retention | A 0.5–0.9; B 0.7–1.1 |
| G2 closure | not resolvable in every arm |
| C4(B1) − C4(A1) | 0 to +2.5 points |

Expected statistics: paired SD of C4 − T1 about 1.7–2.0 points for fold 0 if
binding succeeds (≈65/58 test files at 0/90 MeV), 2.6–2.9 if it fails, 1.0–1.1
over all folds; G2 needs below 0.59 and is unresolvable everywhere. Δbias SD
about ±2.6 MeV for fold 0 and ±1.5 MeV over all folds. AUC precision is limited
by the seed spread (≈0.007).

## Execution

Census: CPU smoke on the test partition, then the full census
(`--array=0-2%2`, rocky/type_d). Prepare: one CPU job. GPU smoke on the test
partition (V100), which sets the training time limit to 1.5 × the projected run
time + 0.5 h. Training: `--array=0-8%3` on rocky/type_a, task t → fold t//3,
seed t%3, node-local work, checkpoints mirrored to scratch, one verified
export, `--no-requeue`. Evaluation per fold after its three tasks. Analysis
locally. An independent preflight review gives GO before the full census and
before the training array. Budget about 30–37 GPU-hours and 30 CPU-hours.

Stop rules: any hash, guard, binding or geometry failure, or a cross-potential
overlap, stops the stage; a non-finite loss stops that task; no automatic retry
except the one pre-approved infrastructure retry; a head minimum at epoch 29
flags the run as budget-limited, with no extension; e_cls = 0 in two or more
seeds sends the census back for checking; each fold's test is evaluated once;
a new arm needs a new protocol; a same-sample null with |z| ≥ 3 stops
interpretation.

### Amendment 2026-10-28: in-process access retry (census)

The full census failed four times with EACCES on readable inputs, at a
different file and node each time: zeroSpot `0004_vacs` (4464315_0, cn-036)
and `0056_vacs` (retry 4464423_0, cn-039), defaultSpot `0098_hits`
(4464315_1, cn-037) and `0106_hits` (4464423_1, cn-038). The pre-approved
retry was used. bigSpot completed (4464315_2). The inputs are 0644 under 0755
directories; every path passes through `/scratch/vbocharnikov`, which is
root:root 0750 and grants access only through a named-user ACL. A Lustre
client that checks the mode bits without the ACL would produce this pattern;
that cause is not proven. RA-09 is not confined to compute nodes: since
2026-10-23 at least eight events are recorded, on cn-030, cn-031 (twice),
cn-036..039 and once on login-02 (a stat of `0071_prim.csv`,
`smash_full_transfer_status_20261023.md`). The four files above read on login
after the failures.

The user authorised the following change. Every census input access (hash,
CSV read, chunked primary scan, stat, directory listing, the smoke receipt) is
retried on EACCES only, after 1, 2, 4, 8, 16 and 32 s. Every retry is printed
to the job log; retries during a tag's census also go to the tag summary
(`access_retries`: path, call, attempt, node, time), while retries reading the
smoke receipt or projecting the smoke runtime appear in the log only. Any
other error, or EACCES after the last wait, crashes the task as before. If
the rerun fails on EACCES again, the next step is a request to cluster
support about the cause, not a further code change. The export copy in the batch script gets the same schedule. The
schedule is a convention: no outage duration has been measured, and the
recorded retries will be the first measurement. A retried read cannot alter
a result unnoticed, because inputs are hashed, pinned hashes are checked and
parsed contents are hashed again. Directory listing replaces `Path.glob`,
which silently skips a directory it cannot open. Because `combine` requires
one code digest, all three tags are rerun, bigSpot included, after a new smoke
on the test partition and a preflight review. The task-level rule stands: no
resubmission without the user.

The rerun 4464531 completed on all three tags. Four EACCES events occurred,
each cleared by the first 1 s wait, two of them on different nodes within one
second. On the user's decision the same retry (`scripts/scratch_access.py`)
now covers the /scratch accesses of the training path before its first full
job: raw-file staging and audit-bundle reads in prepare, bundle extraction,
the frozen DCM inputs, the resume mirror, the once-only evaluation marker and
the final export. A partial copy is removed before each retry, and every
retry is written to the job summary and the publication receipt. The
per-epoch checkpoint mirror already tolerated failures and is unchanged.
Because executed code changed, the GPU smoke is repeated before prepare.

## What this does not establish

A blind holdout (the files were scored by the transfer, and the [0.3, 1) GeV
window was adopted after viewing truth); independence from the generator
(train and test share generator, potentials and production, so closure is
optimistic and shrinkage of the network towards the pooled training
population could attenuate the 90−0 contrast; a leave-one-potential-out
control is a later step); centrality control (no b in the exports); the
primary-neutron response (the label is the native hit label); efficiency
relative to generated neutrons; anything about other generators or data.
Unknown semantics of the "Spot" tags remain a confounder if the productions
differ in more than U_sym.
