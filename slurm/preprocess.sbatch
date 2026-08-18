#!/bin/bash
#SBATCH --job-name=hgnd-preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# No --gres=gpu: preprocess is CPU-only.

# ── Environment ──────────────────────────────────────────────────────────
# Override on the sbatch command line, e.g.:
#   sbatch --export=ALL,DATASET_NAME=zeroSpot,CSV_DIR=/scratch/data/zeroSpot \
#          slurm/preprocess.sbatch
: "${DATASET_NAME:=defaultSpot}"
: "${CSV_DIR:=data/smash_xecs_2.87gev_hardSkyrme_${DATASET_NAME}}"
: "${DATASET_ROOT:=notebooks/cache/ndet_dataset_smash_${DATASET_NAME}}"
: "${NUM_WORKERS:=$SLURM_CPUS_PER_TASK}"
: "${MAX_EVENTS:=}"
: "${VENV:=.venv}"

mkdir -p logs

# ── Activate Python env (edit as needed for your cluster) ────────────────
if [[ -d "$VENV" ]]; then
    source "$VENV/bin/activate"
elif command -v module >/dev/null 2>&1; then
    module load python || true
fi

set -euo pipefail
echo "host:    $(hostname)"
echo "date:    $(date)"
echo "dataset: $DATASET_NAME"
echo "csv:     $CSV_DIR"
echo "root:    $DATASET_ROOT"
echo "workers: $NUM_WORKERS"
[[ -n "$MAX_EVENTS" ]] && echo "max-events: $MAX_EVENTS"

MAX_EVENTS_ARG=""
[[ -n "$MAX_EVENTS" ]] && MAX_EVENTS_ARG="--max-events $MAX_EVENTS"

python -m HGNDRecoGNN.scripts.preprocess \
    --csv-dir     "$CSV_DIR" \
    --root        "$DATASET_ROOT" \
    --num-workers "$NUM_WORKERS" \
    $MAX_EVENTS_ARG

echo "preprocess done: $(date)"
