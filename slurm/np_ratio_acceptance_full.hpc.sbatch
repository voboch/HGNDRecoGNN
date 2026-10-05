#!/usr/bin/env bash
#SBATCH --job-name=np-acceptance-full
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail
source /etc/profile
set -u
module purge
module load python/miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hgnd-env

export PYTHONUNBUFFERED=1
CODE="$HOME/HGNDRecoGNN"
STAGE="/scratch/$USER/hgnd/data_np_full_20260924"
OUT="/scratch/$USER/hgnd/results/np_ratio_acceptance_full_20260925"
MAX_FILES=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  OUT="$STAGE/acceptance_smoke"
  MAX_FILES=(--max-files 1)
fi

mkdir -p "$OUT"
cd "$CODE"
python -u scripts/np_ratio_acceptance_full.py \
  --input-dir "$STAGE/extracted" \
  --output "$OUT/acceptance_full.json" \
  "${MAX_FILES[@]}"
