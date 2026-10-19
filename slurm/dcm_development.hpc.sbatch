#!/bin/bash
# Validation-only development; the old test is never evaluated here.
# 12 tasks = 4 fixed recipes x 3 paired seeds. Submit with --array=0-11%2.
#SBATCH --job-name=dcm_development
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
# cHARISMa reports placeholder RealMemory=1 MB; explicit GB requests fail.
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source /etc/profile 2>/dev/null || true
set -euo pipefail
: "${PROJECT_DIR:?immutable source parent}"
: "${CAMPAIGN_ROOT:?isolated campaign}"
: "${PREPARED_DIR:?frozen prepared split}"
: "${PHASE:=train}"
module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
source activate hgnd-env 2>/dev/null || conda activate hgnd-env
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python -u -m HGNDRecoGNN.scripts.dcm_development --phase "$PHASE" \
    --campaign "$CAMPAIGN_ROOT" --prepared "$PREPARED_DIR" --task "${SLURM_ARRAY_TASK_ID:-0}"
