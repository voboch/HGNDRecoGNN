# HGND + GNN reconstruction pipeline runbook

End-to-end recipe from raw SMASH CSVs (on cHARISMa scratch) to the
per-dataset sensitivity summary CSV that feeds `paper/main.tex`
tables `tab:closure` and `tab:purity_locked`.

**Data generations (updated 2026-09-15).** Two generations of the three
SMASH samples now exist:

| generation | raw tree | cache tag | status |
|---|---|---|---|
| pre-bugfix | `data/` | `_v2` | superseded — carries the upstream simulation bug |
| re-simulated | `data_fixed/` | `_v2fix` | current; archives rebuilt 2026-09-10..12 |

The re-simulated archives fix the upstream bug. Measured against matched
event files (`scripts/compare_data_versions.py`), the fix removes ~31 % of
hits per event, ~33 % of detected neutrons, half the pi- and 41 % of the
gammas, while *raising* protons ~8 % and doubling the median neutron Ekin
(0.49 -> 1.00 GeV). See `results/data_version_comparison/`.

Consequence: **the standing `_v2` checkpoint is out of distribution on the
fixed data and must not be reused** — the fixed-data chain retrains. All
`_v2` physics numbers in `paper/` are superseded.

Steps 1–4 run on cHARISMa; steps 5–6 run on the laptop.

---

## Stage 1 — Raw data on scratch

Prerequisites verified once per dataset:

```bash
ssh charisma        # one hop → login-02 (Rocky 9); the second hop is obsolete
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

The script takes three env overrides (defaults reproduce the old `_v2` build):

| var | default | meaning |
|---|---|---|
| `DATA_ROOT` | `$SCRATCH/hgnd/data` | raw CSV tree to read |
| `TAG` | `v2` | cache suffix → `cache/ndet_dataset_smash_<ds>_<TAG>` |
| `MAX_EVENTS` | *(empty)* | per-dataset event cap; empty = no cap |

So the fixed-data build is
`--export=ALL,DATA_ROOT=$SCRATCH/hgnd/data_fixed,TAG=v2fix,MAX_EVENTS=`.
`slurm/sensitivity_full.hpc.sbatch` and `slurm/train_valloss.hpc.sbatch`
take the same `TAG`.

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

Retrieve the checkpoint (48 MB) — a plain rsync since 2026-09-08, when
`ssh charisma` started landing directly on `login-02`:

```bash
rsync -avhP \
  charisma:/scratch/vbocharnikov/hgnd/checkpoints/defaultSpot_v2_seed42_20ep_valloss_${JOB}/ \
  ~/Project/BM@N/HGND/HGNDRecoGNN/checkpoints_hpc/valloss_${JOB}/
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

## Stage 4b — The fixed-data chain (one command)

For the re-simulated archives the four cluster stages are wrapped in a
single dependency chain, so this replaces Stages 1–4 rather than adding
to them:

```bash
# on login-02, from ~/HGNDRecoGNN
bash slurm/run_sensitivity_fixed.sh
```

which submits, each gated on the previous with `afterok`:

| # | job | partition | what |
|---|---|---|---|
| 1 | `extract_fixed.hpc.sbatch` (array 0-2) | `type_d` | unpack `data_fixed/*.tar.gz` (skips the nested `unigen` ROOT tree) |
| 2 | `preprocess_all_smash.hpc.sbatch` (array 0-2) | `type_d` | build `_v2fix` caches, no event cap |
| 3 | `train_valloss.hpc.sbatch` | `type_a` | **retrain** net_default on fixed `defaultSpot`, val-loss policy |
| 4 | `sensitivity_full.hpc.sbatch` | `type_a` | evaluate all three + sensitivity CLI + purity lock |

Results land in
`$SCRATCH/hgnd/results/sensitivity_full_hpc_v2fix_<jid>/`.

**Why step 3 is not optional.** The fixed simulation has ~31 % fewer hits
per event and a much harder neutron spectrum, so the `_v2` checkpoint is
out of distribution; its reconstruction efficiency does not transfer and
reusing it would bias every `_v2fix` closure number.

Useful overrides: `SKIP_EXTRACT=1` (archives already unpacked),
`SKIP_TRAIN=1 CKPT=/path/model.pt` (evaluate an existing fixed-data
checkpoint).

Prerequisite: the three archives must be in `$SCRATCH/hgnd/data_fixed/`
(upload from the laptop with `rsync -avhP --partial`).

---

## Stage 5 — Retrieve to laptop

Since 2026-09-08 the cluster entry point *is* the Rocky 9 login node, so
`ssh charisma` reaches it in one hop and retrieval is a single rsync. The old
two-hop tar and the `ProxyJump` workaround are no longer needed.

```bash
JOB=<JOBID>
mkdir -p ~/Project/BM@N/HGND/HGNDRecoGNN/results
rsync -avhP \
    --exclude='pred_hits*.pkl' --exclude='pred_edges*.pkl' \
    charisma:/scratch/vbocharnikov/hgnd/results/sensitivity_full_hpc_$JOB \
    ~/Project/BM@N/HGND/HGNDRecoGNN/results/
```

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

## Known caveats (as of 2026-09-15)

| # | Item | Impact | Fix |
|---|---|---|---|
| 1 | ~~SMASH simulation bug in all 3 samples~~ | **Resolved 2026-09-15** by the re-simulated archives (`_v2fix`). `_v2` numbers are superseded. | `bash slurm/run_sensitivity_fixed.sh` |
| 2 | ~~`defaultSpot_v2` cache has `max_events=100000`~~ | **Resolved** — `scripts/preprocess.py` defaults `--max-events` to `None`, and the fixed chain passes `MAX_EVENTS=` explicitly. | — |
| 3 | Checkpoint saved by train-loss (epoch 18) not val-loss (epoch 19) | Small (~0.4 %) validation gap | `slurm/train_valloss.hpc.sbatch` |
| 4 | Rocky partition queue times variable (min→17 h observed 2026-08-26) | Wall-time budgeting | Fair-share fluctuates; submit early |
