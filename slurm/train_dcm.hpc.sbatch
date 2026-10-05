#!/bin/bash
# train_dcm.hpc.sbatch — train on the DCM-QGSM-SMM Xe+CsI reference sample.
#
# Why this job exists
# -------------------
# The Dombay benchmarks were produced by training on DCM-QGSM-SMM and testing
# on a held-out half of the same sample.  Our numbers come from SMASH.  Two
# things can therefore explain any gap — the pipeline, or the generator — and
# only a native DCM training separates them.  So this reproduces the reference
# setup as closely as the refactored code allows:
#
#   hidden 512, layers 8, batch 256, Adam lr 1e-3, weight decay 1e-5,
#   MultiStepLR milestones 10/15/25 gamma 0.1, 25 epochs, 50/50 split, seed 42
#
# Two deliberate departures, both documented in reports/:
#   * the checkpoint is selected on validation loss, not training loss as the
#     notebook does (`if loss < minloss`), which selects on the wrong split;
#   * the loss keeps the cluster-link term this pipeline adds, so the run is
#     comparable with our SMASH numbers rather than only with the notebook.
#
# Submit from a cHARISMa login node, after preprocess_dcm.hpc.sbatch:
#     sbatch slurm/train_dcm.hpc.sbatch
#
#SBATCH --job-name=train_dcm
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint="type_e|type_f|type_h"   # A100 / H100 / H200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${ENV_NAME:=hgnd-env}"
: "${PROJECT_DIR:=$HOME}"
: "${SCRATCH:=/scratch/$USER}"
: "${SCALER:=own}"
: "${TWINDOW:=1.0}"
: "${EPOCHS:=25}"
: "${HIDDEN:=512}"
: "${LAYERS:=8}"
: "${BATCH:=256}"
: "${NUM_SHARDS:=0}"          # DCM is small (376 MB raw) — use all of it
: "${ENERGY_FLATTEN:=0}"   # 1 -> reference second-stage energy reweighting
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

# 1.0 -> 1p0, 1.5 -> 1p5. Bash substitution: the python one-liner this
# replaced nested same-type quotes inside an f-string, which only parses on
# 3.12+, and the job environment is older — both DCM jobs died on it in 2 s.
TW_TAG=${TWINDOW//./p}
DATA_ROOT="$SCRATCH/hgnd/cache/ndet_dataset_dcm_xecs_${SCALER}_tw${TW_TAG}"
if [ -z "$CKPT_DIR" ]; then
    CKPT_DIR="$SCRATCH/hgnd/checkpoints/dcm_${SCALER}_tw${TW_TAG}_h${HIDDEN}l${LAYERS}_${EPOCHS}ep"
fi

if [ ! -d "$DATA_ROOT/processed" ]; then
    echo "ERROR: no cache at $DATA_ROOT — run preprocess_dcm.hpc.sbatch first." >&2
    exit 3
fi

# Same auto-resume contract as train_v3: a chain of submissions sharing one
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

echo "host:  $(hostname)   gpu: ${CUDA_VISIBLE_DEVICES:-none}"
echo "date:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "root:  $DATA_ROOT"
echo "ckpt:  $CKPT_DIR"
echo "arch:  net_default H=$HIDDEN L=$LAYERS  epochs=$EPOCHS batch=$BATCH"
echo "eflat: $ENERGY_FLATTEN"
echo

SHARD_ARG=()
[ "$NUM_SHARDS" -gt 0 ] && SHARD_ARG=(--num-shards "$NUM_SHARDS")
RESUME_ARG=()
[ "$START_EPOCH" -gt 0 ] && RESUME_ARG=(--start-epoch "$START_EPOCH")
[ -n "$INIT_FROM" ] && RESUME_ARG+=(--init-from "$INIT_FROM")
FLAT_ARG=()
[ "$ENERGY_FLATTEN" = "1" ] && FLAT_ARG=(--energy-flatten)

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
    "${SHARD_ARG[@]}" "${RESUME_ARG[@]}" "${FLAT_ARG[@]}" \
    --checkpoint-dir  "$CKPT_DIR" \
    --checkpoint-name model.pt

echo
echo "train done: $(date)"
echo "checkpoint: $CKPT_DIR/model.pt"
