# HPC_DATA.md — HGNDRecoGNN on HSE cHARISMa

Project-specific data + run guide. General cluster rules are in `HPC.md`.
Account `proj_1855`, partition `rocky`, GPU `type_a` (V100 32 GB), env `hgnd-env`.

## Scratch layout (compute nodes have NO internet — everything lives here)

```
/scratch/vbocharnikov/hgnd/
├── cache/           processed PyG datasets that TRAINING reads directly (~2.8G)
│   ├── ndet_dataset_smash_defaultSpot/processed/shard_*.pt  (+ _hits_cache_*.parquet)
│   └── ndet_dataset_smash_*_smoke/                          (small subsets)
├── data/            raw SMASH CSVs (~15G) — ONLY needed to (re)build the cache
│   └── smash_xecs_2.87gev_hardSkyrme_bigSpot/<runid>/*_hits.csv, *_vacs.csv
└── checkpoints/     model checkpoints (written by jobs)
```

**Key point:** training consumes `cache/ndet_dataset_smash_*`. You only need the raw
`data/` if you want to regenerate the cache with `slurm/preprocess.sbatch`.

## One-time setup

1. **Upload data** (from your LOCAL machine, resumable):
   ```
   bash slurm/upload_data.sh cache     # processed datasets, enough to train (~2.8G)
   # bash slurm/upload_data.sh raw     # raw CSVs, only for preprocessing (~15G)
   # bash slurm/upload_data.sh all
   ```
2. **Clone the code onto the cluster** (login node has internet + forwarded git):
   ```
   ssh charisma
   ssh -A login-02
   cd ~ && git clone --depth 1 git@github.com:voboch/HGNDRecoGNN.git
   ```
   The package is imported as `HGNDRecoGNN` (run from `~`, its parent), so no
   `pip install` of the repo is needed.
3. **Fix the PyG stack in `hgnd-env`** (one-time). `net_default` uses
   `DynamicEdgeConv`, which needs the compiled PyG extensions and a torch_geometric
   that works with them on torch 2.5. torch_geometric 2.8 hard-requires
   `pyg-lib>=0.6.0` (only exists for torch>=2.6), so pin 2.6.1:
   ```
   module load python/miniconda
   conda run -n hgnd-env python -m pip install \
     pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
     -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
   conda run -n hgnd-env python -m pip install "torch_geometric==2.6.1"
   ```
   (Already applied to the live `hgnd-env` on 2026-08-25.)

## Launching training

`HGNDRecoGNN.scripts.train` reads `--root <dataset>`. Point it at the scratch cache:

```bash
export SCRATCH=/scratch/$USER/hgnd
export DS="$SCRATCH/cache/ndet_dataset_smash_defaultSpot"
export TRAIN_CMD="-m HGNDRecoGNN.scripts.train \
  --root $DS --model net_default --hidden 512 --num-layers 8 \
  --epochs 20 --batch-size 512 --lr 1e-3 --seed 42 --device auto \
  --checkpoint-dir $SCRATCH/checkpoints/defaultSpot_\${SLURM_JOB_ID} \
  --checkpoint-name model.pt"

# 1) smoke test (5 min, test partition; a smoke dataset trains fast):
sbatch --partition=test --time=00:05:00 \
  --export=ALL,ENV_NAME=hgnd-env,PROJECT_DIR=$HOME,TRAIN_CMD="${TRAIN_CMD/$DS/$SCRATCH/cache/ndet_dataset_smash_bigSpot_smoke} --epochs 1" \
  slurm/train.hpc.sbatch

# 2) real run on rocky:
sbatch --export=ALL,ENV_NAME=hgnd-env,PROJECT_DIR=$HOME,TRAIN_CMD="$TRAIN_CMD" \
  slurm/train.hpc.sbatch
```

`PROJECT_DIR=$HOME` because `python -m HGNDRecoGNN.scripts.train` must run from the
directory that contains the `HGNDRecoGNN/` package.

Sweeps: `slurm/sweep.hpc.sbatch` (one arg-set per line in `slurm/sweep_params.txt`).

## (Re)building the cache from raw CSVs
Only if you changed the raw data or preprocessing. Upload `raw`, then adapt the
existing `slurm/preprocess.sbatch` to read `$SCRATCH/data/...` and write
`$SCRATCH/cache/...`, and submit it on `rocky`.

## Retrieving results (to your LOCAL machine)
```
rsync -avhP charisma:/scratch/vbocharnikov/hgnd/checkpoints/  ./checkpoints_hpc/
```

## Notes
- **Cache schema:** the uploaded `ndet_dataset_smash_defaultSpot` is **schema v1**;
  current code warns it expects **v2** ("Downstream code may crash or produce wrong
  results"). The smoke run trained fine, but for a trustworthy real run regenerate
  the cache from raw CSVs with current code (upload `raw`, run preprocessing) so the
  shards are v2.
- Smoke verified end-to-end on a V100 (`net_default`, 12.5M params, checkpoint written).
- `net_default` / `hidden` / `num-layers` mirror the repo's `slurm/train.sbatch` defaults.
- Reads from `/scratch/.../cache`, writes checkpoints to `/scratch/.../checkpoints`.
  Never write large files to `/home`.
