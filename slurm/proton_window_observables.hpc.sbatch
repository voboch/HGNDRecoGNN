#!/usr/bin/env bash
#SBATCH --job-name=hgnd-p-window
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

module purge 2>/dev/null || true
module load python/miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hgnd-env

CODE="$HOME/HGNDRecoGNN"
INPUT="/scratch/$USER/hgnd/data_np_full_20260924/extracted"
OUTPUT="${OUTPUT:-/scratch/$USER/hgnd/results/proton_window_observables_20260929.json}"
EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA+=(--max-files 1 --n-boot 10)
fi

cd "$CODE"
mkdir -p logs "$(dirname "$OUTPUT")"
export PYTHONUNBUFFERED=1
python -u scripts/proton_window_observables.py \
  --input-dir "$INPUT" \
  --output "$OUTPUT" \
  "${EXTRA[@]}"
