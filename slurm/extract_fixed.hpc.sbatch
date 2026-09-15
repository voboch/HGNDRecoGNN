#!/bin/bash
# extract_fixed.hpc.sbatch — unpack the 2026-09 re-simulated SMASH archives
# into $SCRATCH/hgnd/data_fixed/, leaving the pre-bugfix trees in
# $SCRATCH/hgnd/data/ untouched so the two generations can be compared.
#
# Submit as an array (0=zeroSpot, 1=defaultSpot, 2=bigSpot):
#     sbatch --array=0-2 slurm/extract_fixed.hpc.sbatch
#
# The defaultSpot archive also carries ~1.3 GB of nested `unigen*` tarballs
# (generator-level ROOT files). The CSV pipeline does not read them, so they
# are excluded here.
#
#SBATCH --job-name=hgnd-extract-fixed
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${SCRATCH:=/scratch/$USER}"
: "${ARCHIVE_DIR:=$SCRATCH/hgnd/data_fixed}"
: "${DEST:=$SCRATCH/hgnd/data_fixed}"

DATASETS=(zeroSpot defaultSpot bigSpot)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#DATASETS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range 0..$((${#DATASETS[@]}-1))" >&2
    exit 2
fi
DATASET="${DATASETS[$IDX]}"
TARBALL="$ARCHIVE_DIR/smash_xecs_2.87gev_hardSkyrme_${DATASET}.tar.gz"
TREE="$DEST/smash_xecs_2.87gev_hardSkyrme_${DATASET}"

echo "host:     $(hostname)"
echo "date:     $(date)"
echo "dataset:  $DATASET (index $IDX)"
echo "tarball:  $TARBALL"
echo "dest:     $TREE"

if [[ ! -f "$TARBALL" ]]; then
    echo "ERROR: archive $TARBALL missing — upload it first" >&2
    exit 3
fi

mkdir -p "$DEST"
# --exclude drops the nested generator tarballs and the unigen ROOT tree.
tar xzf "$TARBALL" -C "$DEST" \
    --exclude='*/unigen' --exclude='*/unigen/*' --exclude='*unigen*.tar.gz'

n_hits=$(find "$TREE" -name '*_hits.csv' | wc -l)
n_vacs=$(find "$TREE" -name '*_vacs.csv' | wc -l)
echo "extracted: hits=$n_hits vacs=$n_vacs  size=$(du -sh "$TREE" | cut -f1)"

if (( n_hits == 0 || n_hits != n_vacs )); then
    echo "ERROR: hits/vacs count mismatch ($n_hits vs $n_vacs)" >&2
    exit 4
fi

echo "HGND_EXIT=0"
echo "extract $DATASET done: $(date)"
