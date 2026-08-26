#!/bin/bash
# train_valloss.hpc.sbatch — reproduces the 2026-08-26 defaultSpot v2 run
# (job 4278194, 20 epochs, V100 32 GB, net_default, seed 42) with the sole
# change that checkpoint selection tracks the validation loss instead of the
# training loss.
#
# Motivation
# ----------
# The 2026-08-26 checkpoint was saved at epoch 18 (train 1.2681, val 1.3539)
# under the default save_best_on='train' policy. Epoch 19 had lower val
# (1.3482) and, at smoke-scale evaluation, defaultSpot showed a 35 pp
# closure deficit that survived the pooled-scaler ablation, the binning
# fix, and the per-dataset purity lock. The clusters/neutron ratio (0.51
# on defaultSpot vs 0.68 on OOD) explains most of the deficit as a
# per-cluster-vs-per-neutron definitional artefact, but a val-loss
# checkpoint is the next physics-relevant experiment to rule out
# generalisation overfit as a secondary contributor.
#
# Submit from a cHARISMa login node:
#     sbatch slurm/train_valloss.hpc.sbatch
#
# Everything else (partition, module load, activation, output dir) is
# inherited from train.hpc.sbatch below.
#
#SBATCH --job-name=train_valloss
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a          # V100 32 GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${ENV_NAME:=hgnd-env}"
: "${PROJECT_DIR:=$SLURM_SUBMIT_DIR}"
: "${SCRATCH:=/scratch/$USER}"

module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

mkdir -p logs
cd "$PROJECT_DIR"

echo "host:    $(hostname)"
echo "date:    $(date)"
echo "job:     ${SLURM_JOB_ID:-local}  part=${SLURM_JOB_PARTITION:-?}"
echo "gpu:     ${CUDA_VISIBLE_DEVICES:-none}"
echo "env:     $ENV_NAME   cwd=$PROJECT_DIR"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

DATA_ROOT="$SCRATCH/hgnd/cache/ndet_dataset_smash_defaultSpot_v2"
CKPT_DIR="$SCRATCH/hgnd/checkpoints/defaultSpot_v2_seed42_20ep_valloss_${SLURM_JOB_ID}"

echo
echo "training root:      $DATA_ROOT"
echo "checkpoint dir:     $CKPT_DIR"
echo "save_best_on:       val"
echo

python -u -m HGNDRecoGNN.scripts.train \
    --root            "$DATA_ROOT" \
    --model           net_default \
    --hidden          512 \
    --num-layers      8 \
    --epochs          20 \
    --batch-size      512 \
    --lr              1e-3 \
    --seed            42 \
    --device          auto \
    --save-best-on    val \
    --checkpoint-dir  "$CKPT_DIR" \
    --checkpoint-name model.pt

echo
echo "train done: $(date)"
echo "checkpoint: $CKPT_DIR/model.pt"
