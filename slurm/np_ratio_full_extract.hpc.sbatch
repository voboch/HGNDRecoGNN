#!/usr/bin/env bash
#SBATCH --job-name=np-full-extract
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

DATASETS=(zeroSpot defaultSpot bigSpot)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
STAGE="/scratch/$USER/hgnd/data_np_full_20260924"
ARCHIVE="$STAGE/archives/smash_xecs_2.87gev_hardSkyrme_${DATASET}.tar.gz"
DEST="$STAGE/extracted"

test -s "$ARCHIVE"
mkdir -p "$DEST"

if [[ "${SMOKE:-0}" == "1" ]]; then
  tar -tzf "$ARCHIVE" >/dev/null
  echo "SMOKE_OK dataset=$DATASET archive_bytes=$(stat -c %s "$ARCHIVE")"
  exit 0
fi

tar -xzf "$ARCHIVE" -C "$DEST"
COUNT=$(find "$DEST" -path "*${DATASET}*" -name '*_prim.csv' -type f | wc -l)
if [[ "$COUNT" -le 0 ]]; then
  echo "No primary CSV files found after extracting $DATASET" >&2
  exit 2
fi
echo "EXTRACT_OK dataset=$DATASET primary_csv_files=$COUNT"
