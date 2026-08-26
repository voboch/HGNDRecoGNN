# Handoff → completed HGND v2-cache + real cHARISMa run

Context transfer for the next session working on **HGNDRecoGNN** on the HSE
cHARISMa cluster. This follows `HANDOFF_hgnd.md`. The schema-v2 cache rebuild,
20-epoch V100 training run, monitoring, and checkpoint retrieval are complete.
Everything below was verified live on **2026-08-26**.

**Opening prompt for the next session:**

> Read `HANDOFF_hgnd_run_2026-08-26.md` in this repo. The schema-v2 defaultSpot
> cache and first real rocky/V100 training are complete, and the checkpoint is
> local. Pick up at "Next actions": evaluate the checkpoint, review the
> train-vs-validation checkpoint-selection policy, then decide whether to commit
> the HPC fixes and/or launch the next experiment.

---

## 1. Outcome

- Rebuilt `defaultSpot` as a trustworthy **schema-v2** PyG cache.
- Trained `net_default` for 20 epochs on a V100 32 GB node in `rocky`.
- Both final Slurm jobs completed with exit code 0.
- Retrieved the 48 MB checkpoint locally and verified its SHA-256.
- The Codex heartbeat `monitor-hgnd-rocky-run` was paused after completion.

Final jobs:

| Job | Purpose | State | Elapsed | Node | MaxRSS |
|---|---|---:|---:|---|---:|
| `4278192` | schema-v2 preprocessing | `COMPLETED` | `00:18:35` | `cn-033` | `73,965,172K` |
| `4278194` | 20-epoch V100 training | `COMPLETED` | `01:38:33` | `cn-010` | `8,189,068K` |

## 2. Final cache and raw data

Training cache:

```text
/scratch/vbocharnikov/hgnd/cache/ndet_dataset_smash_defaultSpot_v2/
└── processed/
    ├── meta.json
    ├── shard_0.pt ... shard_195.pt
    ├── scaler_top.pkl / scaler_bot.pkl
    └── _hits_cache_1598d09ff76ded0c.parquet
```

Validated `meta.json`:

```json
{
  "schema_version": 2,
  "num_graphs": 200000,
  "num_shards": 196,
  "shard_size": 1024,
  "rlocal": 3.6,
  "twindow": 1.5,
  "max_events": 100000
}
```

The original v1 cache remains untouched at:

```text
/scratch/vbocharnikov/hgnd/cache/ndet_dataset_smash_defaultSpot/
```

Raw defaultSpot data is now available on scratch as 2,315 hits/vacs file pairs:

```text
/scratch/vbocharnikov/hgnd/data/smash_xecs_2.87gev_hardSkyrme_defaultSpot/
```

The uploaded archive and two diagnostic v2 caches also remain on scratch. They
can be removed later if quota matters, but do not delete them implicitly:

```text
/scratch/vbocharnikov/hgnd/data/smash_xecs_2.87gev_hardSkyrme_defaultSpot.tar.gz
/scratch/vbocharnikov/hgnd/cache/ndet_dataset_smash_defaultSpot_v2_smoke_bootstrap/
/scratch/vbocharnikov/hgnd/cache/ndet_dataset_smash_defaultSpot_v2_ipc_smoke/
```

## 3. Real training configuration and metrics

Job `4278194` ran:

```text
python -u -m HGNDRecoGNN.scripts.train
  --root /scratch/vbocharnikov/hgnd/cache/ndet_dataset_smash_defaultSpot_v2
  --model net_default
  --hidden 512
  --num-layers 8
  --epochs 20
  --batch-size 512
  --lr 1e-3
  --seed 42
  --device auto
  --checkpoint-dir /scratch/vbocharnikov/hgnd/checkpoints/defaultSpot_v2_seed42_20ep_after4278192
  --checkpoint-name model.pt
```

Runtime facts:

- PyTorch `2.5.1+cu121`, CUDA available.
- `net_default`, 12,479,173 parameters.
- Device was CUDA; the log listed CPU-pinned cluster-output modules.
- Preloaded 196 shards / 200,000 graphs in 279.3 seconds.
- Training loop elapsed 5,503.4 seconds.
- No NaNs, OOMs, tracebacks, or training errors were found.

Loss history:

| Epoch | Train | Validation | LR |
|---:|---:|---:|---:|
| 0 | 4.2629 | 2.5636 | 1e-3 |
| 5 | 1.6539 | 1.6525 | 1e-3 |
| 10 | 1.4567 | 1.4630 | 1e-4 |
| 15 | 1.2922 | 1.3669 | 1e-5 |
| 18 | 1.2681 | 1.3539 | 1e-5 |
| 19 | 1.2697 | **1.3482** | 1e-5 |

### Important checkpoint-selection detail

`training/train.py` defaults to `save_best_on='train'`. Therefore the saved
checkpoint is **epoch 18** (lowest training loss 1.2681), even though epoch 19
has the lower validation loss (1.3482 vs 1.3539). Before the next research run,
decide whether checkpoint selection should instead use validation loss and expose
that choice through the training CLI/config.

## 4. Checkpoint locations and checksum

Remote:

```text
/scratch/vbocharnikov/hgnd/checkpoints/defaultSpot_v2_seed42_20ep_after4278192/model.pt
```

Local:

```text
checkpoints_hpc/defaultSpot_v2_seed42_20ep_after4278192/model.pt
```

SHA-256, matched locally and remotely:

```text
f8647e17d5b7fb7ded18b2abca952f807affe4ba763c637309dbff75a9ad1346
```

## 5. Failures encountered and fixes applied

### Batch bootstrap failure

The first preprocessing job (`4276954`) exited before logging; dependent training
`4276956` was automatically cancelled. Cause: `set -u` was enabled before sourcing
the cluster `/etc/profile`, which references optional unset variables. The profile
is now sourced before strict nounset mode in both HPC batch templates.

### Slurm memory-accounting issue

Rocky nodes advertised `RealMemory=1 MB` even though physical free memory was
hundreds of GB. Any explicit `#SBATCH --mem=64G` was rejected as unschedulable.
The explicit memory directive was removed from the HPC preprocessing/training
templates; the partition's default was used. Re-check live Slurm configuration
before restoring a memory request.

### PyTorch multiprocessing IPC failure

Preprocessing with 16 workers (`4277119`) failed before shard 0 with:

```text
RuntimeError: received 0 items of ancdata
```

It reached about 251 GB RSS and was cancelled by root after 3h11m; dependent
training `4277121` never ran. A four-worker reproduction (`4278146`) hit the same
file-descriptor result-sharing failure and was manually cancelled.

`data/graph_dataset.py` now switches PyTorch multiprocessing tensor sharing from
`file_descriptor` to `file_system` around the graph-construction pool, restoring
the prior strategy afterward. Multi-shard smoke `4278184` then completed 4,096
graphs / four shards in 49 seconds, and the full four-worker build completed in
18m35s at about 74 GB MaxRSS.

## 6. Local and cluster code state

Changes made in this session, synced directly to `~/HGNDRecoGNN` on `login-02`
but **not committed**:

- `data/graph_dataset.py` — filesystem-backed PyTorch tensor sharing around Pool.
- `slurm/train.hpc.sbatch` — profile-before-nounset bootstrap and no explicit
  memory request while Slurm advertises `RealMemory=1 MB`.
- `slurm/preprocess.hpc.sbatch` — new Rocky/type_d schema-v2 preprocessing job.

The local worktree was already dirty before this session. Preserve unrelated user
changes in sensitivity/evaluation/training files, notebooks, scripts, and `paper/`.
Current relevant status includes:

```text
 M data/graph_dataset.py
 M slurm/train.hpc.sbatch
?? slurm/preprocess.hpc.sbatch
?? checkpoints_hpc/
?? HANDOFF_hgnd.md
```

Do not blindly commit all dirty files. Review and stage only intended paths.

## 7. Cluster access and guardrails (unchanged)

- Local hop: `ssh charisma` → `sms`.
- Inner hop: `ssh -A -o IdentitiesOnly=no -o StrictHostKeyChecking=no login-02`.
- Do not edit cluster `~/.ssh/config`.
- Git on cluster requires:
  `GIT_SSH_COMMAND="ssh -o IdentitiesOnly=no -o StrictHostKeyChecking=no"`.
- Account `proj_1855`; real jobs on `rocky`; V100 32 GB is `type_a`.
- Always smoke on `--partition=test` before a new long configuration.
- Never train on a login node; large data/checkpoints stay on `/scratch`.
- Keep `hgnd-env` and `torch_geometric==2.6.1` pinned.
- Compute nodes have no internet.

See `HPC.md`, `HPC_DATA.md`, and the original `HANDOFF_hgnd.md` for the full
environment/access background.

## 8. Next actions

1. **Evaluate the retrieved checkpoint** using the intended physics evaluation
   workflow. Be careful: local `scripts/evaluate.py` already has unrelated
   uncommitted changes; inspect them before use.
2. **Resolve checkpoint selection**: decide whether research runs should save by
   validation loss instead of the current training-loss default. Add a CLI/config
   knob and smoke-test it if changing behavior.
3. **Review and commit the HPC fixes** as a focused change if they should become
   permanent. Do not include unrelated dirty worktree files.
4. For another defaultSpot run, reuse the validated v2 root above; no cache rebuild
   is needed unless preprocessing/schema/raw data changes.
5. Profile GPU utilization earlier in the next run. The only captured snapshot was
   0% during validation, and the job finished before a longer trace could be taken;
   this was not enough to establish persistent under-utilization.
6. Optionally remove the uploaded archive and diagnostic smoke caches after the
   checkpoint/cache are independently backed up and quota cleanup is desired.

