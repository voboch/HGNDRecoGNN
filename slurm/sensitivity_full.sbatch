#!/bin/bash
#SBATCH --job-name=hgnd-sens-full
#SBATCH --output=logs/sens_full_%j.out
#SBATCH --error=logs/sens_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#
# Production run of the HGND cross-dataset sensitivity study:
#   preprocess all three datasets → train one model on defaultSpot →
#   evaluate on all three → compare N_true(Ekin) curves.
#
# Runtime budget: ~24 h on 1×A100. Reduce EPOCHS or NUM_LAYERS if wall
# clock is tight.
#
#   sbatch slurm/sensitivity_full.sbatch
#   sbatch --export=ALL,MODEL=hetero_hgt,EPOCHS=30 slurm/sensitivity_full.sbatch
#
# For a model sweep (net_default + hetero_hgt + spectral_dynedge) use the
# array form:
#   sbatch --array=0-2 slurm/sensitivity_full.sbatch
# In array mode the model at index SLURM_ARRAY_TASK_ID from MODELS_GRID is used.

: "${TRAIN_DATASET:=defaultSpot}"
: "${DATASETS:=defaultSpot zeroSpot bigSpot}"
: "${MODEL:=net_default}"
: "${HIDDEN:=512}"
: "${NUM_LAYERS:=8}"
: "${BATCH_SIZE:=512}"
: "${EPOCHS:=20}"
: "${LR:=1e-3}"
: "${NUM_WORKERS_PROC:=16}"
: "${DEVICE:=auto}"
: "${THRESHOLD:=0.5}"
: "${VENV:=.venv}"

# Model grid — indexed by SLURM_ARRAY_TASK_ID when running as an array job.
MODELS_GRID=("net_default" "hetero_hgt" "spectral_dynedge")

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ "$SLURM_ARRAY_TASK_ID" -ge "${#MODELS_GRID[@]}" ]]; then
        echo "error: array id $SLURM_ARRAY_TASK_ID exceeds grid size ${#MODELS_GRID[@]}" >&2
        exit 1
    fi
    MODEL="${MODELS_GRID[$SLURM_ARRAY_TASK_ID]}"
    echo "array task $SLURM_ARRAY_TASK_ID → MODEL=$MODEL"
fi

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
: "${OUT_ROOT:=${REPO}/results/sensitivity_full_${MODEL}_${JOB_TAG}}"

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
echo "job tag:  $JOB_TAG"
echo "repo:     $REPO"
echo "out root: $OUT_ROOT"

export TRAIN_DATASET DATASETS MODEL HIDDEN NUM_LAYERS BATCH_SIZE EPOCHS LR
export NUM_WORKERS_PROC DEVICE THRESHOLD OUT_ROOT

exec "$REPO/scripts/run_sensitivity_full.sh"
