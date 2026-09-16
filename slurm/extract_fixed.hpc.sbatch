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
# Sum apparent sizes rather than `du`: on Lustre, `du` run immediately after
# tar reports a fraction of the real size until allocation settles (it read
# 131M for a 3.8G tree in smoke job 4329237).
bytes=$(find "$TREE" -type f -name '*.csv' -printf '%s\n' | awk '{s+=$1} END {print s+0}')
# Header-only CSVs occur for runs that produced no hits. load_hits() skips any
# file <=100 bytes, so report that exact criterion -- roughly half of every
# dataset is empty this way, and a run that silently halved again should be
# visible here. (`-size -1k` was wrong: find rounds sub-1K files up to 1 block,
# so it matched nothing and always reported 0.)
n_empty=$(find "$TREE" -type f -name '*_hits.csv' ! -size +100c | wc -l)
n_used=$(find "$TREE" -type f -name '*_hits.csv' -size +100c | wc -l)
echo "extracted: hits=$n_hits vacs=$n_vacs  csv_bytes=$bytes ($(awk -v b="$bytes" 'BEGIN{printf "%.2f GB", b/1073741824}'))"
echo "           usable hits files=$n_used  header_only=$n_empty"

if (( n_hits == 0 || n_hits != n_vacs )); then
    echo "ERROR: hits/vacs count mismatch ($n_hits vs $n_vacs)" >&2
    exit 4
fi
if (( bytes < 100000000 )); then
    echo "ERROR: extracted CSVs total only $bytes bytes — extraction looks truncated" >&2
    exit 5
fi

echo "HGND_EXIT=0"
echo "extract $DATASET done: $(date)"
