#!/bin/bash
# train_v3.hpc.sbatch — retrain the energy estimator on the v3 caches.
#
# Why this job exists
# -------------------
# The reconstruction in hand falls well short of the published benchmarks
# (ROC AUC ~0.97, ~80 % efficiency at ~87 % purity, linearity and resolution
# under 10 % from 0.7 to 5 GeV).  A laptop retrain on 148 k graphs improved
# energy linearity from +50-60 % to within 10 % over 1.4-2.6 GeV but stopped
# after 4 of 20 epochs to thermal throttling, and 2.7 M parameters is a fifth
# of the reference architecture.  This runs the reference architecture to
# convergence on the v3 caches, which hold 13.9 M graphs across the three
# samples against the 148 k used locally.
#
# Submit from a cHARISMa login node:
#     sbatch slurm/train_v3.hpc.sbatch
#     TRAIN_DS=zeroSpot NUM_SHARDS=0 sbatch slurm/train_v3.hpc.sbatch
#
#SBATCH --job-name=train_v3
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint="type_e|type_f|type_h"   # A100 / H100 / H200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${ENV_NAME:=hgnd-env}"
: "${PROJECT_DIR:=$HOME}"
: "${SCRATCH:=/scratch/$USER}"
: "${TAG:=v3}"
: "${TRAIN_DS:=defaultSpot}"
: "${EPOCHS:=20}"
: "${HIDDEN:=512}"
: "${LAYERS:=8}"
: "${BATCH:=512}"
# Shard cap. The v3 caches hold 3.6-5.2 k shards of 1024 graphs; preloading
# every one from Lustre is the slowest part of the job. 2500 matches the scale
# of the proven v2 run. Set NUM_SHARDS=0 to use the whole cache.
: "${NUM_SHARDS:=2500}"
# Continuation. Two 20-epoch runs here were cancelled by the scheduler at about
# three and a half hours, so a full schedule has to span several submissions.
# Point CKPT_DIR at the previous run's directory and set START_EPOCH to its
# next_epoch; the optimizer and LR schedule resume from optim.pt beside it.
: "${START_EPOCH:=0}"
: "${CKPT_DIR:=}"
: "${INIT_FROM:=}"

module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
export PYTHONPATH="$HOME/HGNDRecoGNN:${PYTHONPATH:-}"
mkdir -p "${SLURM_SUBMIT_DIR:-$HOME/HGNDRecoGNN}/logs"

DATA_ROOT="$SCRATCH/hgnd/cache/ndet_dataset_smash_${TRAIN_DS}_${TAG}"
if [ -z "$CKPT_DIR" ]; then
    CKPT_DIR="$SCRATCH/hgnd/checkpoints/${TRAIN_DS}_${TAG}_h${HIDDEN}l${LAYERS}_${EPOCHS}ep_${SLURM_JOB_ID:-local}"
fi
# Resuming without an explicit INIT_FROM picks up the last epoch's weights.
if [ "$START_EPOCH" -gt 0 ] && [ -z "$INIT_FROM" ] && [ -f "$CKPT_DIR/last.pt" ]; then
    INIT_FROM="$CKPT_DIR/last.pt"
fi

echo "host:   $(hostname)   gpu: ${CUDA_VISIBLE_DEVICES:-none}"
echo "date:   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "root:   $DATA_ROOT"
echo "ckpt:   $CKPT_DIR"
echo "arch:   net_default H=$HIDDEN L=$LAYERS  epochs=$EPOCHS batch=$BATCH shards=$NUM_SHARDS"
echo

SHARD_ARG=()
[ "$NUM_SHARDS" -gt 0 ] && SHARD_ARG=(--num-shards "$NUM_SHARDS")
RESUME_ARG=()
[ "$START_EPOCH" -gt 0 ] && RESUME_ARG=(--start-epoch "$START_EPOCH")
[ -n "$INIT_FROM" ] && RESUME_ARG+=(--init-from "$INIT_FROM")
[ -n "$INIT_FROM" ] && echo "resuming from $INIT_FROM at epoch $START_EPOCH"

python -u -m HGNDRecoGNN.scripts.train \
    --root            "$DATA_ROOT" \
    --model           net_default \
    --hidden          "$HIDDEN" \
    --num-layers      "$LAYERS" \
    --epochs          "$EPOCHS" \
    --batch-size      "$BATCH" \
    --lr              1e-3 \
    --seed            42 \
    --device          auto \
    --save-best-on    val \
    "${SHARD_ARG[@]}" "${RESUME_ARG[@]}" \
    --checkpoint-dir  "$CKPT_DIR" \
    --checkpoint-name model.pt

echo
echo "train done: $(date)"
echo "checkpoint: $CKPT_DIR/model.pt"
