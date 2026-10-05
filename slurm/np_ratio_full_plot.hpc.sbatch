#!/usr/bin/env bash
#SBATCH --job-name=np-full-plot
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail
source /etc/profile
set -u
module purge
module load python/miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hgnd-env

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/scratch/$USER/hgnd/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

CODE="$HOME/HGNDRecoGNN"
STAGE="/scratch/$USER/hgnd/data_np_full_20260924"
if [[ "${SMOKE:-0}" == "1" ]]; then
  OUT="$STAGE/smoke_results"
  MAX_FILES=(--max-files 1)
else
  OUT="/scratch/$USER/hgnd/results/np_ratio_full_20260924"
  MAX_FILES=()
fi

mkdir -p "$OUT"
cd "$CODE"
python -u scripts/np_ratio_smash_check_stream.py \
  --input-dir "$STAGE/extracted" \
  --output-dir "$OUT" \
  --pdf-output "$OUT/np_smash_check_full.pdf" \
  "${MAX_FILES[@]}"
