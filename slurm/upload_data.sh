#!/bin/bash
# upload_data.sh — RUN ON YOUR LOCAL MACHINE (not the cluster).
# Pushes HGNDRecoGNN data to /scratch on HSE cHARISMa via the `charisma` ssh alias.
# Resumable (rsync --partial).
#
#   bash slurm/upload_data.sh [cache|raw|all]
#     cache  (default) processed graph datasets that training reads directly (~2.8G)
#     raw              raw SMASH CSVs, only needed to (re)build the cache (~15G)
#     all              both
#
# Compute nodes have no internet, so all data must live on /scratch before a job.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="charisma:/scratch/vbocharnikov/hgnd"
MODE="${1:-cache}"

echo ">> ensuring remote dirs"
ssh charisma 'mkdir -p /scratch/$USER/hgnd/cache /scratch/$USER/hgnd/data /scratch/$USER/hgnd/checkpoints'

if [[ "$MODE" == "cache" || "$MODE" == "all" ]]; then
    echo ">> uploading notebooks/cache/ (processed ndet_dataset_* ~2.8G) -> $REMOTE/cache/"
    rsync -avhP --partial --exclude='.DS_Store' \
          "$REPO/notebooks/cache/" "$REMOTE/cache/"
fi

if [[ "$MODE" == "raw" || "$MODE" == "all" ]]; then
    echo ">> uploading data/ raw SMASH CSVs (~15G, excl .tar.gz) -> $REMOTE/data/"
    rsync -avhP --partial --exclude='.DS_Store' --exclude='*.tar.gz' \
          "$REPO/data/" "$REMOTE/data/"
fi

echo ">> done. Verify: ssh charisma 'du -sh /scratch/\$USER/hgnd/*'"
