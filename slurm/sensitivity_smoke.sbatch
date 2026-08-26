#!/bin/bash
#SBATCH --job-name=hgnd-sens-smoke
#SBATCH --output=logs/sens_smoke_%j.out
#SBATCH --error=logs/sens_smoke_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#
# Cluster reproduction of notebooks/sensitivity_smoketest.ipynb.
#
#   sbatch slurm/sensitivity_smoke.sbatch
#
# Overrides:
#   sbatch --export=ALL,MAX_EVENTS=5000,EPOCHS=5,NUM_SHARDS=8 \
#          slurm/sensitivity_smoke.sbatch
#
# Preprocesses all three datasets on-node from ${REPO}/data/<dataset>/,
# trains net_default on defaultSpot, evaluates on all three, and runs
# the sensitivity comparison. All artefacts land under ${OUT_ROOT}.

# ── Env ─────────────────────────────────────────────────────────────────
: "${TRAIN_DATASET:=defaultSpot}"
: "${DATASETS:=defaultSpot zeroSpot bigSpot}"
: "${MODEL:=net_default}"
: "${HIDDEN:=256}"
: "${NUM_LAYERS:=4}"
: "${BATCH_SIZE:=64}"      # cluster GPU can afford a bit more than laptop
: "${EPOCHS:=5}"
: "${NUM_SHARDS:=8}"
: "${MAX_EVENTS:=5000}"
: "${NUM_WORKERS_PROC:=8}"
: "${PRELOAD_MAX_GB:=8.0}"
: "${DEVICE:=auto}"
: "${DO_PREPROCESS:=1}"
: "${VENV:=.venv}"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${OUT_ROOT:=${REPO}/results/sensitivity_smoke_${SLURM_JOB_ID:-local}}"

mkdir -p logs "$OUT_ROOT"

if [[ -d "$VENV" ]]; then
    source "$VENV/bin/activate"
elif command -v module >/dev/null 2>&1; then
    module load python cuda || true
fi

set -euo pipefail
echo "host:     $(hostname)"
echo "date:     $(date)"
echo "gpu:      ${CUDA_VISIBLE_DEVICES:-none}"
echo "repo:     $REPO"
echo "out root: $OUT_ROOT"

export TRAIN_DATASET DATASETS MODEL HIDDEN NUM_LAYERS BATCH_SIZE EPOCHS
export NUM_SHARDS MAX_EVENTS NUM_WORKERS_PROC PRELOAD_MAX_GB DEVICE DO_PREPROCESS
export OUT_ROOT

exec "$REPO/scripts/run_sensitivity_smoke.sh"
