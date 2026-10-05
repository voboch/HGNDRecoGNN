#!/bin/bash
# Event-level multi-neutron efficiency from one evaluated checkpoint.
# Smoke first with --partition=test, then submit unchanged on rocky.
#
#   PRED_DIR=/scratch/$USER/hgnd/results/eval_v3_4357105 \
#   OUT_DIR=/scratch/$USER/hgnd/results/multiplicity_v3_epoch3 \
#   sbatch slurm/multiplicity_efficiency.hpc.sbatch
#
#SBATCH --job-name=hgnd_mult_eff
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${ENV_NAME:=hgnd-env}"
: "${PROJECT_DIR:=$HOME/HGNDRecoGNN}"
: "${PRED_DIR:?set PRED_DIR to an eval_v3 output directory}"
: "${OUT_DIR:?set OUT_DIR}"
: "${MAX_RANK:=6}"

module purge 2>/dev/null || true
module load python/miniconda
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
export PYTHONPATH="$HOME/HGNDRecoGNN:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR" "$OUT_DIR" logs

python -u scripts/multiplicity_efficiency.py \
    --pred-dir "$PRED_DIR" \
    --out-dir "$OUT_DIR" \
    --max-rank "$MAX_RANK"

ls -lh "$OUT_DIR"
