# HGND + GNN reconstruction pipeline runbook

End-to-end recipe from raw SMASH CSVs (on cHARISMa scratch) to the
per-dataset sensitivity summary CSV that feeds `paper/main.tex`
tables `tab:closure` and `tab:purity_locked`.

**Data caveat (2026-08-28):** the three SMASH samples currently on
cHARISMa have an upstream simulation bug (see memory
`project-smash-datasets-bug`). Per-neutron physics results are
provisional; per-cluster pipeline calibration numbers are not
affected.

Steps 1–4 run on cHARISMa; steps 5–6 run on the laptop.

---

## Stage 1 — Raw data on scratch

Prerequisites verified once per dataset:

```bash
ssh charisma
ssh -A -o IdentitiesOnly=no -o StrictHostKeyChecking=no login-02
ls /scratch/vbocharnikov/hgnd/data/smash_xecs_2.87gev_hardSkyrme_{zeroSpot,defaultSpot,bigSpot}/ | head
```

Raw upload procedure (when new simulations land) lives in `HPC_DATA.md`.

---

## Stage 2 — Preprocess to schema-v2 caches

Runbook per dataset. Array job builds any subset that isn't yet
schema-v2:

```bash
cd ~/HGNDRecoGNN && git pull
# array index 0=zeroSpot, 1=defaultSpot, 2=bigSpot
sbatch --array=0-2 slurm/preprocess_all_smash.hpc.sbatch
```

Expected artefact per dataset:

```
/scratch/vbocharnikov/hgnd/cache/ndet_dataset_smash_${ds}_v2/
└── processed/
    ├── meta.json                     (schema_version=2)
    ├── shard_0.pt … shard_N.pt
    ├── scaler_top.pkl / scaler_bot.pkl
    └── _hits_cache_*.parquet         (raw hits — used by sensitivity CLI)
```

Wall time: **~2 h per full dataset** on the `rocky` / `type_d`
partition (matches jobs 4278192 / 4284505 profile).
Note: as of 2026-08-27 the `defaultSpot` cache was built with
`max_events=100000`. Lifting the cap is a one-line override:

```bash
sbatch --array=1 --export=ALL,MAX_EVENTS= slurm/preprocess_all_smash.hpc.sbatch
```

(Empty `MAX_EVENTS` → no cap; the script forwards accordingly.)

---

## Stage 3 — Train (only when the network changes; skip for
       standing-checkpoint sensitivity runs)

Default HPC checkpoint is at
`/scratch/vbocharnikov/hgnd/checkpoints/defaultSpot_v2_seed42_20ep_after4278192/model.pt`
(net_default, H=512, L=8, seed 42, epoch 18 by train-loss policy).
A val-loss variant is stubbed in `slurm/train_valloss.hpc.sbatch`.

```bash
sbatch slurm/train_valloss.hpc.sbatch          # ~1h40m on V100 32 GB (type_a)
```

Retrieve the checkpoint (48 MB) via two-hop tar:

```bash
ssh charisma "ssh -A -o IdentitiesOnly=no -o StrictHostKeyChecking=no login-02 \
  'tar czf - -C /scratch/vbocharnikov/hgnd/checkpoints \
     defaultSpot_v2_seed42_20ep_valloss_${JOB}'" \
  > ~/Project/BM@N/HGND/HGNDRecoGNN/checkpoints_hpc/valloss_${JOB}.tar.gz
```

---

## Stage 4 — Evaluate + sensitivity CLI + purity-lock

All three run in one sbatch, chained after preprocess via
`--dependency=afterok:`.

```bash
# Chain preprocess and sensitivity — most common invocation
JOB_PREP=$(sbatch --array=0-2 --parsable slurm/preprocess_all_smash.hpc.sbatch)
sbatch --dependency=afterok:$JOB_PREP slurm/sensitivity_full.hpc.sbatch
```

Or against a specific checkpoint:

```bash
sbatch --export=ALL,CKPT=/scratch/.../valloss/model.pt \
       slurm/sensitivity_full.hpc.sbatch
```

Outputs land under
`/scratch/vbocharnikov/hgnd/results/sensitivity_full_hpc_${JOB}/`:

```
sensitivity_full_hpc_${JOB}/
├── {defaultSpot,zeroSpot,bigSpot}_hpc/pred_{hits,clusters,edges}_hpc_full.pkl
├── sensitivity_e_pred/     (default basis + Tikhonov unfold)
│   ├── sensitivity_summary.csv         ← headline table
│   ├── sensitivity_${ds}.csv per dataset
│   ├── sensitivity_unfold_${ds}.csv per dataset
│   └── sensitivity_yield.png
├── sensitivity_e_true/     (physics-truth basis, diagnostic only)
└── purity_locked_pi70/
    └── purity_locked_pi70.csv
```

Wall time: ~30 min preprocess + ~30-60 min sensitivity on the V100.

---

## Stage 5 — Retrieve to laptop

`login-02` is not directly SSH-reachable from the laptop (see
`HANDOFF_hgnd.md` §7). Two working options:

**Option A — two-hop tar (no ssh-config changes):**

```bash
JOB=<JOBID>
mkdir -p ~/Project/BM@N/HGND/HGNDRecoGNN/results
ssh charisma "ssh -A -o IdentitiesOnly=no -o StrictHostKeyChecking=no login-02 \
    'tar czf - -C /scratch/vbocharnikov/hgnd/results \
       --exclude=\"*/pred_hits*.pkl\" --exclude=\"*/pred_edges*.pkl\" \
       sensitivity_full_hpc_$JOB'" \
    > ~/Project/BM@N/HGND/HGNDRecoGNN/results/sensitivity_full_hpc_$JOB.tar.gz
tar xzf ~/Project/BM@N/HGND/HGNDRecoGNN/results/sensitivity_full_hpc_$JOB.tar.gz \
    -C ~/Project/BM@N/HGND/HGNDRecoGNN/results/
```

**Option B — rsync with a ProxyJump added to laptop `~/.ssh/config`:**

```
Host login-02 login-*
    HostName login-02
    User vbocharnikov
    ProxyJump charisma
    ForwardAgent yes
```

Then `rsync -av login-02:/scratch/.../sensitivity_full_hpc_$JOB \
       ~/Project/BM@N/HGND/HGNDRecoGNN/results/` works one-shot.

`results/` is gitignored (see `.gitignore`); small artefacts
(summary CSVs, PNGs) can be committed manually into `paper/figs/`.

---

## Stage 6 — Paper update

Mechanical:

1. **`tab:closure`** (`paper/main.tex` search for `\label{tab:closure}`):
   replace the numeric rows and caption `Source:` path from
   `sensitivity_e_pred_fixed/sensitivity_summary.csv` of the new
   job.
2. **`tab:purity_locked`** — same, from `purity_locked_pi70/purity_locked_pi70.csv`.
3. **`tab:eff_ekin`** — regenerate via `notebooks/evaluate_hpc_checkpoint.ipynb`
   cell `cfe7357b` pointed at the new defaultSpot cluster prediction pickle.
4. **`fig:yield`** — copy
   `sensitivity_e_pred_fixed/sensitivity_yield.png` →
   `paper/figs/sensitivity_yield.pdf`.
5. Abstract + Sec 5.4 prose: if the sensitivity story changes
   (e.g. re-simulated data flips `k(U_sym)` monotonicity), update
   accordingly; otherwise the current methods-first framing holds.

---

## Regression tests before shipping updated numbers

```bash
conda activate pyg
python tests/test_sensitivity_pipeline.py
```

Five assertions cover:
- `_discover_parquet` prefers `_v2` over `_smoke` (guards the
  2026-08-27 4284506 pre-fix bug).
- Smoke fallback still works.
- Summary CSV has all 12 required columns.
- `k ∈ [0.3, 1.0]`, `C_c ∈ [0.5, 2.0]`, and `C_ν = k·C_c` to 2 %.
- `N_MC/ev > 0.1` (MC-truth undercount canary).

Add new asserts here before adding new columns / bases to the CLI.

---

## Known caveats (as of 2026-08-28)

| # | Item | Impact | Fix |
|---|---|---|---|
| 1 | SMASH simulation bug in all 3 samples | Per-neutron physics is provisional | Re-simulate; runbook is unchanged |
| 2 | `defaultSpot_v2` cache has `max_events=100000` | `N_MC/ev = 0.80` vs 1.10 on OOD; sample-selection artefact | `sbatch --array=1 --export=ALL,MAX_EVENTS= slurm/preprocess_all_smash.hpc.sbatch` |
| 3 | Checkpoint saved by train-loss (epoch 18) not val-loss (epoch 19) | Small (~0.4 %) validation gap | `slurm/train_valloss.hpc.sbatch` |
| 4 | Rocky partition queue times variable (min→17 h observed 2026-08-26) | Wall-time budgeting | Fair-share fluctuates; submit early |
