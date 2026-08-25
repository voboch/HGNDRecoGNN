#!/bin/bash
# sweep.sbatch — job-array hyperparameter sweep on the rocky partition.
# Submit:  sbatch --array=0-5 slurm/sweep.sbatch
# Each array task picks one line from a params file (default: slurm/sweep_params.txt),
# where every line is a set of CLI args appended to the training command.
#
#SBATCH --job-name=sweep
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-0                    # override on the CLI: --array=0-N

set -euo pipefail

: "${ENV_NAME:=hpc-env}"
: "${PARAMS_FILE:=slurm/sweep_params.txt}"
: "${PROJECT_DIR:=$SLURM_SUBMIT_DIR}"
: "${SCRATCH:=/scratch/$USER}"
export PYTHONUNBUFFERED=1 HF_HOME="$SCRATCH/hf_cache" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

source /etc/profile 2>/dev/null || true
module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

mkdir -p logs
cd "$PROJECT_DIR"

if [[ ! -f "$PARAMS_FILE" ]]; then
    echo "ERROR: params file '$PARAMS_FILE' not found. One arg-set per line." >&2
    exit 2
fi
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAMS_FILE")
if [[ -z "$LINE" ]]; then
    echo "ERROR: no line $((SLURM_ARRAY_TASK_ID + 1)) in $PARAMS_FILE" >&2
    exit 2
fi

echo "task ${SLURM_ARRAY_TASK_ID}: $LINE"
: "${TRAIN_CMD:=}"
if [[ -z "$TRAIN_CMD" ]]; then
    echo "ERROR: set TRAIN_CMD (base command, without the swept args)." >&2
    exit 2
fi
echo ">> python -u $TRAIN_CMD $LINE"
eval "python -u $TRAIN_CMD $LINE"
