#!/bin/bash
#SBATCH --job-name=hgnd-sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# ── Array job over (model, dataset) pairs ───────────────────────────────
# Submit with:
#   sbatch --array=0-8 slurm/sweep.sbatch
# There are len(MODELS) × len(DATASETS) tasks (default 3×3 = 9).

# Grids — edit to add/remove models or datasets.
MODELS=("net_default" "hetero_hgt" "spectral_dynedge")
DATASETS=("zeroSpot" "defaultSpot" "bigSpot")

: "${EPOCHS:=20}"
: "${BATCH_SIZE:=512}"
: "${LR:=1e-3}"
: "${SEED:=42}"
: "${VENV:=.venv}"

n_models=${#MODELS[@]}
n_datasets=${#DATASETS[@]}
n_tasks=$((n_models * n_datasets))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "error: this script must be submitted with --array=0-$((n_tasks - 1))" >&2
    exit 1
fi
if [[ "$SLURM_ARRAY_TASK_ID" -ge "$n_tasks" ]]; then
    echo "error: array id $SLURM_ARRAY_TASK_ID exceeds grid size $n_tasks" >&2
    exit 1
fi

# Map task id → (model, dataset).
model_idx=$(( SLURM_ARRAY_TASK_ID / n_datasets ))
dataset_idx=$(( SLURM_ARRAY_TASK_ID % n_datasets ))
MODEL="${MODELS[$model_idx]}"
DATASET_NAME="${DATASETS[$dataset_idx]}"

DATASET_ROOT="notebooks/cache/ndet_dataset_smash_${DATASET_NAME}"
CKPT_DIR="checkpoints/sweep/${DATASET_NAME}_${MODEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RESULTS_DIR="results/sweep/${DATASET_NAME}_${MODEL}"

mkdir -p logs "$CKPT_DIR" "$RESULTS_DIR"

if [[ -d "$VENV" ]]; then
    source "$VENV/bin/activate"
elif command -v module >/dev/null 2>&1; then
    module load python cuda || true
fi

set -euo pipefail
echo "=== sweep task $SLURM_ARRAY_TASK_ID / $((n_tasks - 1)) ==="
echo "host:    $(hostname)"
echo "date:    $(date)"
echo "gpu:     ${CUDA_VISIBLE_DEVICES:-none}"
echo "model:   $MODEL"
echo "dataset: $DATASET_NAME  ($DATASET_ROOT)"
echo "ckpt:    $CKPT_DIR"
echo "results: $RESULTS_DIR"

# ── 1) train ────────────────────────────────────────────────────────────
python -m HGNDRecoGNN.scripts.train \
    --root            "$DATASET_ROOT" \
    --model           "$MODEL" \
    --epochs          "$EPOCHS" \
    --batch-size      "$BATCH_SIZE" \
    --lr              "$LR" \
    --seed            "$SEED" \
    --device          auto \
    --checkpoint-dir  "$CKPT_DIR" \
    --checkpoint-name "model.pt"

# ── 2) evaluate → per-cluster prediction pickle ─────────────────────────
python -m HGNDRecoGNN.scripts.evaluate \
    --root       "$DATASET_ROOT" \
    --checkpoint "$CKPT_DIR/model.pt" \
    --out-dir    "$RESULTS_DIR" \
    --suffix     smash \
    --device     auto

echo "sweep task done: $(date)"
