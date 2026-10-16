#!/bin/bash
# train_dcm_improve.hpc.sbatch — the audit's experiment ladder on DCM.
#
# Why this job exists
# -------------------
# reports/dcm_performance_audit.md measures an 11-14 point efficiency
# shortfall against the Dombay benchmarks at matched purity, and argues the
# leading suspect is training budget rather than the model. This script runs
# the ladder from that report's §6, one rung per submission, so each change is
# evaluated on its own instead of being stacked blindly.
#
# Each rung ADDS to the one before it, and the three optimisation factors are
# separated rather than moved together. An earlier version of this script had
# one `budget` rung that changed epochs, schedule shape and warmup at once; its
# result could not attribute any difference to one of them. That rung is kept,
# clearly labelled as a package, for when only the combined effect is wanted.
#
#   VARIANT=ref        reference settings, unchanged: 25 epochs, MultiStepLR.
#                      Same as train_dcm.hpc.sbatch; run it here to get the
#                      baseline in the same directory layout as the rest.
#   VARIANT=epochs     30 epochs, MultiStepLR unchanged.  <- step count only
#   VARIANT=sched      30 epochs, cosine, no warmup.      <- schedule shape
#   VARIANT=warmup     30 epochs, cosine, 3 warmup epochs. <- warmup only
#   VARIANT=budget     the three above together, as one package. Use only if
#                      the combined effect is what is being reported.
#   VARIANT=size       warmup + --cluster-size-feature. Cluster size alone
#                      separates signal from background at AUC 0.729; the
#                      cluster branch is fed avg_pool_x and cannot see it.
#   VARIANT=pool       + --cluster-pooling mean_max_sum.
#   VARIANT=residual   + --hit-residual. Parameter-free.
#
# Step-count arithmetic, since it is easy to quote the wrong comparison. At
# batch 256 on fixed data, ref -> epochs changes optimiser steps by 30/25 =
# 1.2x (up to the last partial batch). The ~2.5x figure in
# reports/dcm_performance_audit.md is a DIFFERENT comparison -- the
# reference's 25 epochs @ batch 256 against this pipeline's former default of
# 20 @ batch 512 -- and does not describe any transition between these rungs.
#
# Checkpoints go to a per-variant directory, so rungs never overwrite each
# other and each can be evaluated separately with eval_dcm.hpc.sbatch.
#
# Submit from a cHARISMa login node, after preprocess_dcm.hpc.sbatch:
#     VARIANT=budget   sbatch slurm/train_dcm_improve.hpc.sbatch
#     VARIANT=size     sbatch slurm/train_dcm_improve.hpc.sbatch
#     ...
#
# Run them one at a time and evaluate between rungs: if `budget` closes the
# gap there is no reason to change the architecture at all.
#
#SBATCH --job-name=train_dcm_improve
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
: "${SCALER:=own}"
: "${TWINDOW:=1.0}"
: "${HIDDEN:=512}"
: "${LAYERS:=8}"
: "${BATCH:=256}"
: "${NUM_SHARDS:=0}"
: "${START_EPOCH:=0}"
: "${CKPT_DIR:=}"
: "${INIT_FROM:=}"
: "${VARIANT:=budget}"

# ── the ladder ───────────────────────────────────────────────────────────
# EPOCHS and the schedule flags are set per variant rather than taken from the
# environment, so a submission cannot half-apply a rung.
SCHED_ARG=()
ARCH_ARG=()
case "$VARIANT" in
  ref)
    EPOCHS=25
    ;;                                   # reference MultiStepLR, untouched
  epochs)
    EPOCHS=30
    ;;                                   # step count only; schedule unchanged
  sched)
    EPOCHS=30
    SCHED_ARG=(--scheduler cosine --warmup-epochs 0)
    ;;
  warmup)
    EPOCHS=30
    SCHED_ARG=(--scheduler cosine --warmup-epochs 3)
    ;;
  budget)                                # the package: all three at once
    EPOCHS=30
    SCHED_ARG=(--scheduler cosine --warmup-epochs 3)
    ;;
  size)
    EPOCHS=30
    SCHED_ARG=(--scheduler cosine --warmup-epochs 3)
    ARCH_ARG=(--cluster-size-feature)
    ;;
  pool)
    EPOCHS=30
    SCHED_ARG=(--scheduler cosine --warmup-epochs 3)
    ARCH_ARG=(--cluster-size-feature --cluster-pooling mean_max_sum)
    ;;
  residual)
    EPOCHS=30
    SCHED_ARG=(--scheduler cosine --warmup-epochs 3)
    ARCH_ARG=(--cluster-size-feature --cluster-pooling mean_max_sum
              --hit-residual)
    ;;
  *)
    echo "ERROR: unknown VARIANT='$VARIANT'." >&2
    echo "       expected one of: ref epochs sched warmup budget size pool residual" >&2
    exit 2
    ;;
esac

module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
export PYTHONPATH="$HOME/HGNDRecoGNN:${PYTHONPATH:-}"
mkdir -p "${SLURM_SUBMIT_DIR:-$HOME/HGNDRecoGNN}/logs"

TW_TAG=${TWINDOW//./p}
DATA_ROOT="$SCRATCH/hgnd/cache/ndet_dataset_dcm_xecs_${SCALER}_tw${TW_TAG}"
if [ -z "$CKPT_DIR" ]; then
    # SCALER, BATCH and NUM_SHARDS are all overridable and all change what the
    # run is, so they belong in the path. Without them two different runs of
    # one variant share a directory and auto-resume silently adopts the other
    # run's optimizer and weights.
    CKPT_DIR="$SCRATCH/hgnd/checkpoints/dcm_improve_${VARIANT}_${SCALER}_tw${TW_TAG}_h${HIDDEN}l${LAYERS}_b${BATCH}_s${NUM_SHARDS}_${EPOCHS}ep"
fi

if [ ! -d "$DATA_ROOT/processed" ]; then
    echo "ERROR: no cache at $DATA_ROOT — run preprocess_dcm.hpc.sbatch first." >&2
    exit 3
fi

# Same auto-resume contract as train_dcm: a chain of submissions sharing one
# CKPT_DIR continues without anyone tracking epoch numbers.
if [ "$START_EPOCH" -eq 0 ] && [ -f "$CKPT_DIR/optim.pt" ]; then
    START_EPOCH=$(python - "$CKPT_DIR/optim.pt" <<'EOF'
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu",
                         weights_only=False).get("next_epoch", 0)))
except Exception:
    print(0)
EOF
)
    echo "auto-resume: optim.pt reports next_epoch=$START_EPOCH"
fi
if [ "$START_EPOCH" -gt 0 ] && [ -z "$INIT_FROM" ] && [ -f "$CKPT_DIR/last.pt" ]; then
    INIT_FROM="$CKPT_DIR/last.pt"
fi
if [ "$START_EPOCH" -ge "$EPOCHS" ]; then
    echo "already at epoch $START_EPOCH of $EPOCHS — nothing to do"
    exit 0
fi

echo "host:    $(hostname)   gpu: ${CUDA_VISIBLE_DEVICES:-none}"
echo "date:    $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "variant: $VARIANT"
echo "root:    $DATA_ROOT"
echo "ckpt:    $CKPT_DIR"
echo "arch:    net_default H=$HIDDEN L=$LAYERS  epochs=$EPOCHS batch=$BATCH"
echo "sched:   ${SCHED_ARG[*]:-multistep (reference milestones)}"
echo "archflg: ${ARCH_ARG[*]:-none (v1 architecture)}"
echo

SHARD_ARG=()
[ "$NUM_SHARDS" -gt 0 ] && SHARD_ARG=(--num-shards "$NUM_SHARDS")
RESUME_ARG=()
[ "$START_EPOCH" -gt 0 ] && RESUME_ARG=(--start-epoch "$START_EPOCH")
[ -n "$INIT_FROM" ] && RESUME_ARG+=(--init-from "$INIT_FROM")

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
    "${SCHED_ARG[@]}" "${ARCH_ARG[@]}" \
    "${SHARD_ARG[@]}" "${RESUME_ARG[@]}" \
    --checkpoint-dir  "$CKPT_DIR" \
    --checkpoint-name model.pt

echo
echo "train done: $(date)"
echo "checkpoint: $CKPT_DIR/model.pt"
echo
echo "next: evaluate this rung before submitting the following one, e.g."
echo "  CKPT=$CKPT_DIR/model.pt sbatch slurm/eval_dcm.hpc.sbatch"
