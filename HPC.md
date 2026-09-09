# HPC.md — rules for AI agents running training on HSE cHARISMa

These are **directive rules**, not a tutorial. Any coding agent that runs, submits,
or edits training jobs in this repo MUST follow them. Values below are verified
against the live cluster (2026-09-08); if a fact looks stale, re-check the source
page rather than guessing — the cluster is still being migrated to Rocky 9, and the
SSH entry point already moved once (see section 1).

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
ssh charisma            # → vbocharnikov@cluster.hpc.hse.ru:2222 → login-02, Rocky Linux 9
```

**One hop, as of 2026-09-08.** The entry point behind `cluster.hpc.hse.ru:2222` was
migrated from the old CentOS 7 node `sms` to the Rocky Linux 9 node `login-02`, so
`ssh charisma` now lands on the Rocky node directly. The former two-step
`ssh charisma` → `ssh -A login-02` is obsolete, and so is the `-o IdentitiesOnly=no`
workaround, which existed only for that inner hop. Non-interactive use is now just:

```
ssh -o BatchMode=yes charisma '<cmd>'
```

- **Host keys changed with the migration** (both ED25519 and ECDSA; the server is now
  OpenSSH 9.9). If `known_hosts` still holds the old `sms` keys, ssh refuses to
  connect with "host key has changed". Current verified fingerprints:
  - `ED25519 SHA256:x3TzZFswnhbRd7Ccgyqs2nXliHyCFkNhJjRm5MCRP98`
  - `ECDSA   SHA256:aWqPu667Evrt2ZjF0qJbzYEVNdvUtnsnSVJjNjid/v8`
  - `RSA     SHA256:gDqMHFKMBRoqMNjOey5k2J5CO1gfBntbCzZYzwY48ik`

  Refresh with `ssh-keygen -R '[cluster.hpc.hse.ru]:2222'`, then re-add — but verify
  the fingerprints against an HSE source before trusting a changed key.
- `~/.ssh/config` (local) defines `charisma` with `ForwardAgent yes`,
  `AddKeysToAgent yes`, `UseKeychain yes`, and an `IdentityFile` pointing at your
  local cluster key. The exact path is a local-machine detail — keep it out of git.
- The cluster key is passphrase-protected; it is loaded into the local macOS agent
  (`ssh-add --apple-use-keychain <your-cluster-key>`). The passphrase and the key
  never leave the local machine — only the agent's signing channel is forwarded.
- The `charisma` block also carries `BindInterface en0`, so an active VPN tunnel
  cannot hijack source-address selection and break the connection with
  "Can't assign requested address". Drop it if you stop using that VPN.
- Verify agent forwarding on the cluster: `ssh-add -l` lists both keys and
  `ssh -T git@github.com` returns `Hi voboch!`.
- The cluster-side `~/.ssh/config` has a `Host * IdentitiesOnly yes` block that
  breaks the forwarded agent for git. Do not edit that file; instead pass
  `GIT_SSH_COMMAND="ssh -o IdentitiesOnly=no"` for git operations on the cluster.

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

- Build the conda env **on the login node only** (internet + right glibc), into
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

- [ ] On the login node, not a compute node (`ssh charisma` lands on `login-02`).
- [ ] Env activated and imports (torch + CUDA) verified on a `test` GPU shell.
- [ ] Data/weights already on `/scratch/$USER`; no download in the job body.
- [ ] `--account=proj_1855` and `--constraint=type_a` present.
- [ ] Smoke run on `--partition=test` passed.
- [ ] `python -u` (or `PYTHONUNBUFFERED=1`) set.
