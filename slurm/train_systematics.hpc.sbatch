#!/bin/bash
# train_systematics.hpc.sbatch — Sec 6 items 4+5 systematic sweep.
#
# Trains five variants that populate the paper's model-family + seed
# variance systematic entries.  All variants use the same
# ndet_dataset_smash_defaultSpot_v2_cap100k cache (200k graphs,
# 100k-event preprocessing cap) that the 2026-08-26 baseline
# checkpoint was trained on, to keep the spread quote apples-to-
# apples.
#
# Array indices:
#   0 → hetero_hgt         seed 42
#   1 → hetero_sage        seed 42
#   2 → spectral_dynedge   seed 42 (with k_pe=4)
#   3 → net_default        seed 123  (seed variance vs baseline seed 42)
#   4 → net_default        seed 456
#
# Submit:
#   sbatch --array=0-4 slurm/train_systematics.hpc.sbatch
#
#SBATCH --job-name=hgnd-train-syst
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${ENV_NAME:=hgnd-env}"
: "${PROJECT_DIR:=$HOME}"
: "${SCRATCH:=/scratch/$USER}"
: "${DATA_ROOT:=$SCRATCH/hgnd/cache/ndet_dataset_smash_defaultSpot_v2_cap100k}"

module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
export PYTHONPATH="$HOME/HGNDRecoGNN:${PYTHONPATH:-}"
mkdir -p "${SLURM_SUBMIT_DIR:-$HOME/HGNDRecoGNN}/logs"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
case "$IDX" in
  0) MODEL=hetero_hgt        ; SEED=42  ; EXTRA=() ;;
  1) MODEL=hetero_sage       ; SEED=42  ; EXTRA=() ;;
  2) MODEL=spectral_dynedge  ; SEED=42  ; EXTRA=(--k-pe 4) ;;
  3) MODEL=net_default       ; SEED=123 ; EXTRA=(--hidden 512 --num-layers 8) ;;
  4) MODEL=net_default       ; SEED=456 ; EXTRA=(--hidden 512 --num-layers 8) ;;
  *) echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range 0-4" >&2 ; exit 2 ;;
esac

TAG="${MODEL}_seed${SEED}"
CKPT_DIR="$SCRATCH/hgnd/checkpoints/systematics/${TAG}"
mkdir -p "$CKPT_DIR"

echo "host:       $(hostname)"
echo "date:       $(date)"
echo "job:        ${SLURM_JOB_ID:-local}  array=${SLURM_ARRAY_JOB_ID:-?}/${SLURM_ARRAY_TASK_ID:-?}"
echo "gpu:        ${CUDA_VISIBLE_DEVICES:-none}"
echo "model:      $MODEL   seed: $SEED   extras: ${EXTRA[*]:-<none>}"
echo "data:       $DATA_ROOT"
echo "ckpt dir:   $CKPT_DIR"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [[ ! -f "$DATA_ROOT/processed/meta.json" ]]; then
    echo "ERROR: missing training data at $DATA_ROOT" >&2
    exit 3
fi

# Use net_default's proven recipe (20 epochs, batch 512, lr 1e-3, val-loss checkpoint).
# For non-net_default the CLI forwards --hidden as `hidden` and defaults come from
# each model's factory (hidden=128, num_layers=4 in the shared baseline).
python -u -m HGNDRecoGNN.scripts.train \
    --root            "$DATA_ROOT" \
    --model           "$MODEL" \
    --epochs          20 \
    --batch-size      512 \
    --lr              1e-3 \
    --seed            "$SEED" \
    --device          auto \
    --save-best-on    val \
    --checkpoint-dir  "$CKPT_DIR" \
    --checkpoint-name model.pt \
    "${EXTRA[@]}"

echo
echo "HGND_EXIT=0"
echo "systematics train $TAG done: $(date)"
