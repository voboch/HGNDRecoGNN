#!/bin/bash
#SBATCH --job-name=hgnd-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ── Env ─────────────────────────────────────────────────────────────────
# Override on the sbatch command line, e.g.:
#   sbatch --export=ALL,DATASET_NAME=defaultSpot,MODEL=net_default \
#          slurm/train.sbatch
: "${DATASET_NAME:=defaultSpot}"
: "${DATASET_ROOT:=notebooks/cache/ndet_dataset_smash_${DATASET_NAME}}"
: "${MODEL:=net_default}"
: "${HIDDEN:=512}"
: "${NUM_LAYERS:=8}"
: "${EPOCHS:=20}"
: "${BATCH_SIZE:=512}"
: "${LR:=1e-3}"
: "${SEED:=42}"
: "${NUM_SHARDS:=}"
: "${CKPT_DIR:=checkpoints/${DATASET_NAME}_${MODEL}_${SLURM_JOB_ID:-local}}"
: "${VENV:=.venv}"

mkdir -p logs "$CKPT_DIR"

if [[ -d "$VENV" ]]; then
    source "$VENV/bin/activate"
elif command -v module >/dev/null 2>&1; then
    module load python cuda || true
fi

set -euo pipefail
echo "host:      $(hostname)"
echo "date:      $(date)"
echo "gpu:       ${CUDA_VISIBLE_DEVICES:-none}"
echo "dataset:   $DATASET_NAME  ($DATASET_ROOT)"
echo "model:     $MODEL  hidden=$HIDDEN  layers=$NUM_LAYERS"
echo "epochs:    $EPOCHS  batch=$BATCH_SIZE  lr=$LR  seed=$SEED"
echo "ckpt-dir:  $CKPT_DIR"

NUM_SHARDS_ARG=""
[[ -n "$NUM_SHARDS" ]] && NUM_SHARDS_ARG="--num-shards $NUM_SHARDS"

python -m HGNDRecoGNN.scripts.train \
    --root            "$DATASET_ROOT" \
    --model           "$MODEL" \
    --hidden          "$HIDDEN" \
    --num-layers      "$NUM_LAYERS" \
    --epochs          "$EPOCHS" \
    --batch-size      "$BATCH_SIZE" \
    --lr              "$LR" \
    --seed            "$SEED" \
    --device          auto \
    --checkpoint-dir  "$CKPT_DIR" \
    --checkpoint-name "model.pt" \
    $NUM_SHARDS_ARG

echo "train done: $(date)"
