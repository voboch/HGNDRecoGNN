#!/bin/bash
# train.sbatch — single-GPU training template for the rocky (Rocky 9) partition.
# Submit from a login node:  sbatch slurm/train.sbatch
# Override knobs on the CLI: sbatch --export=ALL,EPOCHS=50,ENV_NAME=my-env slurm/train.sbatch
#
#SBATCH --job-name=train
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a          # V100 32 GB; type_e=A100, type_f=H100, type_h=H200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=vobotch@gmail.com

set -euo pipefail

# ── Knobs (override via --export=ALL,NAME=value) ────────────────────────
: "${ENV_NAME:=hpc-env}"                 # conda env built by slurm/setup_env.sh
: "${PROJECT_DIR:=$SLURM_SUBMIT_DIR}"
: "${SCRATCH:=/scratch/$USER}"
: "${HF_HOME:=$SCRATCH/hf_cache}"
export HF_HOME PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

# ── Environment (Rocky 9) ───────────────────────────────────────────────
source /etc/profile 2>/dev/null || true
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
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# ── Training command — EDIT THIS for the project ────────────────────────
# Always use `python -u`. Read data from $SCRATCH, write checkpoints to $SCRATCH.
: "${TRAIN_CMD:=}"
if [[ -z "$TRAIN_CMD" ]]; then
    echo "ERROR: set TRAIN_CMD (or replace this block with the real python -u ... line)." >&2
    exit 2
fi
echo ">> $TRAIN_CMD"
eval "python -u $TRAIN_CMD"

echo "train done: $(date)"
