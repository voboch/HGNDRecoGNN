# Handoff → HGND GNN training on HSE cHARISMa

Context transfer for a fresh session focused on running/iterating **HGND** neutron-
reconstruction GNN training (`HGNDRecoGNN.scripts.train`) on the HSE cHARISMa cluster.
Everything below is verified live (2026-08-25). Read top to bottom, then start at
"Next actions".

**Opening prompt to paste into the new session:**

> Read HANDOFF_hgnd.md in this repo. It's a context transfer about running the HGND
> reconstruction GNN on the HSE cHARISMa cluster. The environment, data, and code are
> already set up and an end-to-end smoke run passed. Pick up at "Next actions":
> regenerate the v2 cache if needed, then launch a real run on the `rocky` partition
> and monitor it.

---

## 1. Goal
Run and iterate real HGND GNN trainings on cHARISMa (GPU jobs on the `rocky`
partition), not just smoke. The full setup is done; the task is to launch real runs,
monitor them, and pull results back — and to regenerate the dataset cache to schema
v2 for trustworthy results (see §7).

## 2. Cluster access (all verified)
- `ssh charisma` → login node `sms` (CentOS 7), user `vbocharnikov`. The local
  `~/.ssh/config` alias carries host/port/user/key; the encrypted key is in the
  macOS agent (passphrase in Keychain, never leaves the machine).
- From `sms`: `ssh -A -o IdentitiesOnly=no -o StrictHostKeyChecking=no login-02`
  → **Rocky Linux 9** login node (use this for env/git/sbatch).
- **CRITICAL SSH gotcha — do NOT edit the cluster `~/.ssh/config`** (user's choice).
  It has a `Host * IdentitiesOnly yes IdentityFile ~/.ssh/cluster` block (added
  2026-08-25) that breaks forwarded-agent auth. Work around it in every command:
  - inner hop: add `-o IdentitiesOnly=no`
  - git on the cluster: `export GIT_SSH_COMMAND="ssh -o IdentitiesOnly=no -o StrictHostKeyChecking=no"`
- Slurm: account `proj_1855`, partition `rocky` (default), GPU tier V100 32 GB =
  `--constraint=type_a`. Debug: `--partition=test` (30-min). Quotas: 200 CPU / 20 GPU
  / 1 TB scratch. General rules are in `HPC.md`; data/run specifics in `HPC_DATA.md`.

## 3. Environment (ready — but version-sensitive)
- Conda env **`hgnd-env`** on `login-02` (py3.11, torch 2.5.1+cu121). Built; do not
  rebuild. Activate non-interactively: `conda run -n hgnd-env ...`. Load first:
  `module load python/miniconda cuda/12.9.1`.
- **PyG stack is pinned** (do not "upgrade"): `torch_geometric==2.6.1` + compiled
  extensions `pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv`
  (from `data.pyg.org/whl/torch-2.5.1+cu121.html`). `net_default` uses
  `DynamicEdgeConv`; torch_geometric 2.8 hard-requires `pyg-lib>=0.6.0`, which does
  not exist for torch 2.5 — that combination fails. 2.6.1 uses `torch_cluster.knn`
  and works. If you rebuild the env, re-apply this (see `HPC_DATA.md` step 3).
- No `pip install` of the repo is needed — it's imported as the `HGNDRecoGNN` package
  by running `python -m HGNDRecoGNN.scripts.train` from its parent dir (`$HOME`).

## 4. Code on the cluster
- `~/HGNDRecoGNN` — a shallow full clone (it's tiny, ~740 KB).
- To refresh: `cd ~/HGNDRecoGNN && GIT_SSH_COMMAND="ssh -o IdentitiesOnly=no -o StrictHostKeyChecking=no" git pull`.
- Run trainings from `$HOME` (the parent of the package), e.g.
  `cd ~ && python -m HGNDRecoGNN.scripts.train ...`.

## 5. Data on scratch (uploaded, verified)
```
/scratch/vbocharnikov/hgnd/
  cache/    processed PyG datasets that TRAINING reads (--root points here)
    ndet_dataset_smash_defaultSpot/processed/shard_*.pt   (196 shards)
    ndet_dataset_smash_bigSpot_smoke/, ndet_dataset_smash_zeroSpot_smoke/
  data/     raw SMASH CSVs — ONLY for (re)building the cache (upload with: bash slurm/upload_data.sh raw)
  checkpoints/   ← write checkpoints here (NOT /home)
```
Re-upload cache from local if needed: `bash slurm/upload_data.sh cache` (runs on Mac).

## 6. Verified smoke (already passed)
`sbatch --partition=test` running `python -m HGNDRecoGNN.scripts.train --root
.../ndet_dataset_smash_defaultSpot --model net_default --epochs 1 --batch-size 128
--num-shards 2 --device auto --checkpoint-dir .../checkpoints/smoke_<jid>` →
device=cuda, net_default (12.5M params), 1 epoch (train 22.79 / val 8.82), wrote
`model.pt` (49.9 MB). `HGND_EXIT=0`. So the launch path works end-to-end.

## 7. KNOWN ISSUE — dataset cache schema (fix before trusting results)
The uploaded `ndet_dataset_smash_defaultSpot` is **schema v1**; current code warns it
expects **v2** ("Downstream code may crash or produce wrong results"). The smoke
trained fine, but a real run's metrics are not trustworthy on v1 shards.
**Regenerate the cache to v2 with current code before real training:**
- upload raw CSVs: `bash slurm/upload_data.sh raw` (~15 GB) → `/scratch/.../hgnd/data/`
- adapt `slurm/preprocess.sbatch` to read `/scratch/.../hgnd/data/...` and write
  `/scratch/.../hgnd/cache/...`, submit it on `rocky`.
- Then train against the freshly built v2 cache.

## 8. Next actions
1. (Recommended first) **Regenerate the v2 cache** — see §7.
2. **Launch a real run on `rocky`** from `$HOME` via `slurm/train.hpc.sbatch`
   (kept separate from the repo's own `slurm/train.sbatch`). Skeleton:
   ```bash
   export SCR=/scratch/$USER/hgnd
   export DS="$SCR/cache/ndet_dataset_smash_defaultSpot"     # or the new v2 cache
   export TRAIN_CMD="-m HGNDRecoGNN.scripts.train \
     --root $DS --model net_default --hidden 512 --num-layers 8 \
     --epochs 20 --batch-size 512 --lr 1e-3 --seed 42 --device auto \
     --checkpoint-dir $SCR/checkpoints/defaultSpot_\${SLURM_JOB_ID} \
     --checkpoint-name model.pt"
   sbatch --export=ALL,ENV_NAME=hgnd-env,PROJECT_DIR=$HOME,TRAIN_CMD="$TRAIN_CMD" \
     slurm/train.hpc.sbatch
   ```
   (`PROJECT_DIR=$HOME` because `python -m HGNDRecoGNN...` runs from the package's parent.)
   Other models: any name from `HGNDRecoGNN.models.available()`. Sweeps:
   `slurm/sweep.hpc.sbatch` with one arg-set per line in `slurm/sweep_params.txt`.
3. **Monitor:** `slurm/hpcjob.sh queue`, `slurm/hpcjob.sh watch <jobid>`,
   `slurm/hpcjob.sh gpu <jobid>` (keep GPU utilisation high — admins see idle GPUs).
4. **Retrieve:** `rsync -avhP charisma:/scratch/vbocharnikov/hgnd/checkpoints/ ./checkpoints_hpc/`.

## 9. Guardrails
- Never run training on a login node; smoke on `--partition=test` before a long run.
- Never write large files to `/home`; everything big lives on `/scratch`.
- Don't edit the cluster `~/.ssh/config`; use the `-o IdentitiesOnly=no` bypass.
- Don't upgrade `torch_geometric` past 2.6.x in `hgnd-env` (breaks `DynamicEdgeConv`).
- Compute nodes have no internet — data/weights must be on `/scratch` beforehand.
