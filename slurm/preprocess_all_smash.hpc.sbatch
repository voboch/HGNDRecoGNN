#!/bin/bash
# preprocess_all_smash.hpc.sbatch — schema-v2 cache build for one of the
# three SMASH sensitivity datasets. Submit as an array job:
#
#     sbatch --array=0-2 slurm/preprocess_all_smash.hpc.sbatch
#
# where array index 0/1/2 → zeroSpot/defaultSpot/bigSpot. Skips
# defaultSpot if its v2 cache already exists (was built in job 4278192
# per HANDOFF_hgnd_run_2026-08-26.md).
#
# Prerequisites (fail fast if missing):
#   /scratch/vbocharnikov/hgnd/data/smash_xecs_2.87gev_hardSkyrme_${DATASET}/
# containing the raw hits + vacs CSV directories. If a dataset's raw
# data is not yet on scratch, submit the array with the appropriate
# --array subset (e.g. --array=0,2 to build zero+big only).
#
#SBATCH --job-name=hgnd-preprocess-v2-smash
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${ENV_NAME:=hgnd-env}"
: "${PROJECT_DIR:=$HOME}"
: "${SCRATCH:=/scratch/$USER}"
: "${NUM_WORKERS:=${SLURM_CPUS_PER_TASK:-8}}"

module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

# HGNDRecoGNN package lives at $HOME/HGNDRecoGNN/HGNDRecoGNN/. The parent
# must be on sys.path for `python -m HGNDRecoGNN.*` to resolve. Older
# preprocess job 4278192 worked because PROJECT_DIR defaulted to $HOME
# and PYTHONPATH was inherited from the shell. Force both explicitly.
cd "$PROJECT_DIR"
export PYTHONPATH="$HOME/HGNDRecoGNN:${PYTHONPATH:-}"

DATASETS=(zeroSpot defaultSpot bigSpot)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#DATASETS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range 0..$((${#DATASETS[@]}-1))" >&2
    exit 2
fi
DATASET="${DATASETS[$IDX]}"

CSV_DIR="$SCRATCH/hgnd/data/smash_xecs_2.87gev_hardSkyrme_${DATASET}"
DATASET_ROOT="$SCRATCH/hgnd/cache/ndet_dataset_smash_${DATASET}_v2"

echo "host:       $(hostname)"
echo "date:       $(date)"
echo "job:        ${SLURM_JOB_ID:-local}  array=${SLURM_ARRAY_JOB_ID:-?}/${SLURM_ARRAY_TASK_ID:-?}"
echo "part:       ${SLURM_JOB_PARTITION:-?}"
echo "env:        $ENV_NAME  cwd=$PROJECT_DIR"
echo "dataset:    $DATASET (index $IDX)"
echo "csv:        $CSV_DIR"
echo "root:       $DATASET_ROOT"
echo "workers:    $NUM_WORKERS"
python -c "from HGNDRecoGNN.data.graph_dataset import SCHEMA_V; print('schema:', SCHEMA_V)"

if [[ ! -d "$CSV_DIR" ]]; then
    echo "ERROR: CSV_DIR $CSV_DIR does not exist. Upload the raw data first." >&2
    exit 3
fi
if [[ -f "$DATASET_ROOT/processed/meta.json" ]]; then
    disk_v=$(python -c "import json; print(json.load(open('$DATASET_ROOT/processed/meta.json')).get('schema_version', 1))")
    if [[ "$disk_v" == "2" ]]; then
        echo "$DATASET already has schema-v2 cache — skipping."
        echo "HGND_EXIT=0"
        exit 0
    fi
    echo "existing cache at $DATASET_ROOT is schema v$disk_v; will be rebuilt."
fi

python -u -m HGNDRecoGNN.scripts.preprocess \
    --csv-dir "$CSV_DIR" \
    --root    "$DATASET_ROOT" \
    --num-workers "$NUM_WORKERS"

echo "HGND_EXIT=0"
echo "preprocess $DATASET done: $(date)"
