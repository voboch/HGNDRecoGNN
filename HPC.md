# HPC.md — rules for AI agents running training on HSE cHARISMa

These are **directive rules**, not a tutorial. Any coding agent that runs, submits,
or edits training jobs in this repo MUST follow them. Values below are verified
against the live cluster (2026-08-25); if a fact looks stale, re-check the source
page rather than guessing — the Rocky 9 migration is ongoing.

## 0. Non-negotiables

- **Never** copy a private key, token, or password onto the cluster or into any
  repo file. Git auth on the cluster uses the **forwarded SSH agent** only.
- **Never** run training on a login node. Login nodes are for editing, building
  envs, downloading data, and `sbatch`/`squeue` only. No `nohup`/`tmux` training.
- **Never** submit a long job before a `--partition=test` smoke run passes.
- Compute nodes have **NO internet**. All `pip`/`conda`/`git`/HuggingFace
  downloads happen on a login node beforehand. A job that tries to download hangs
  or fails.
- Large data and checkpoints live on `/scratch/$USER`. Code lives on
  `/home/$USER`. Nothing is backed up.

## 1. How to reach the cluster

```
ssh charisma            # → vbocharnikov@cluster.hpc.hse.ru:2222, host "sms", CentOS 7
ssh -A login-02         # from sms → Rocky Linux 9 login node (use this for new work)
```

- `~/.ssh/config` (local) already defines `charisma` with `ForwardAgent yes`,
  `AddKeysToAgent yes`, `UseKeychain yes`, and an `IdentityFile` pointing at your
  local cluster key. The exact path is a local-machine detail — keep it out of git.
- The cluster key is passphrase-protected; it is loaded into the local macOS agent
  (`ssh-add --apple-use-keychain <your-cluster-key>`). The passphrase and the key
  never leave the local machine — only the agent's signing channel is forwarded.
- The CentOS 7 login node (`sms`) runs an old OpenSSH: use
  `-o StrictHostKeyChecking=no` (NOT `accept-new`) for the inner hop to `login-02`.
- Verify agent forwarding on `login-02`: `ssh-add -l` lists both keys and
  `ssh -T git@github.com` returns `Hi voboch!`.

## 2. Slurm facts (verified)

| Fact | Value |
|---|---|
| Account | `proj_1855` (only one → always pass `--account=proj_1855`) |
| Default partition | `rocky` (the `*` in `sinfo`); use it for new work |
| Debug partition | `test` (30-min limit, higher priority — smoke runs) |
| Quick preempt | `gpu-ef-quick` (3 h, A100/H100/H200), `cpu-e-quick` |
| Legacy | `normal` (CentOS 7 path; now only 1 node — avoid) |
| GPU tier here | **V100 32 GB → `--constraint=type_a`** (16 nodes under rocky) |
| Quotas | 200 CPU cores, 20 GPUs, 100 running jobs, 1 TB `/scratch` |

Node types for `--constraint`: `type_a/b/c` = V100 32 GB, `type_d` = CPU-only,
`type_e` = A100 80 GB (×8), `type_f` = H100 80 GB (×2), `type_h` = H200 141 GB (×8).
An **unconstrained** GPU job can land anywhere — always set `--constraint`.

Time formats: `mm`, `mm:ss`, `hh:mm:ss`, `d-hh`, `d-hh:mm:ss`. Backfill is on —
shorter `--time` starts sooner. Default limit is generous (rocky allows up to 30 d)
but keep requests honest; low GPU utilisation is visible to admins (HPC TaskMaster).

## 3. Environment (Rocky 9 login node)

Modules (from `/opt/el9/hse/modules`):

```
module purge
module load python/miniconda      # conda; NOT the CentOS "Anaconda_v10.2019"
module load cuda/12.9.1           # or cuda/12.9 / cuda/13.1
```

- Build the conda env **on `login-02` only** (internet + right glibc), into
  `/home/$USER` or a named env. Use `slurm/setup_env.sh <env-name> <py-version>`.
- A conda env built on Rocky 9 will **fail with glibc errors on CentOS 7** and vice
  versa. Rebuild per OS; never reuse across partitions.
- Pre-download models/datasets into `/scratch/$USER/hf_cache` while on the login
  node. In jobs set `HF_HOME=/scratch/$USER/hf_cache` and
  `TRANSFORMERS_OFFLINE=1` / `HF_HUB_OFFLINE=1`.
- Python buffers stdout on Slurm: always `python -u` or `PYTHONUNBUFFERED=1`, or
  logs stay empty.

## 4. Submitting jobs

Use the templates in `slurm/` and the helper:

```
slurm/hpcjob.sh submit slurm/train.sbatch   # sbatch wrapper
slurm/hpcjob.sh watch <jobid>               # tail logs
slurm/hpcjob.sh queue                        # squeue -u $USER
slurm/hpcjob.sh gpu <jobid>                  # live GPU util on the node
slurm/hpcjob.sh kill <jobid>
```

Required workflow for any real run:
1. `srun --pty --partition=test --account=proj_1855 --constraint=type_a --gpus=1 --time=00:20:00 bash`
   → check `python -c "import torch; print(torch.cuda.is_available())"`.
2. `sbatch --partition=test …` a 5-minute smoke run; confirm logs + a checkpoint.
3. Only then `sbatch slurm/train.sbatch` on `rocky`.

Every `#SBATCH` block MUST have: `--account=proj_1855`, `--partition`,
`--constraint`, `--gpus`, `--cpus-per-task`, `--time`, and `--output=logs/%x_%j.out`.
`logs/` is git-ignored.

## 5. Agent checklist before you submit anything

- [ ] On `login-02`, not `sms`, and not a compute node.
- [ ] Env activated and imports (torch + CUDA) verified on a `test` GPU shell.
- [ ] Data/weights already on `/scratch/$USER`; no download in the job body.
- [ ] `--account=proj_1855` and `--constraint=type_a` present.
- [ ] Smoke run on `--partition=test` passed.
- [ ] `python -u` (or `PYTHONUNBUFFERED=1`) set.
