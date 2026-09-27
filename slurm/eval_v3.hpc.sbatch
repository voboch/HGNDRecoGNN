#!/bin/bash
# eval_v3.hpc.sbatch — evaluate a v3 checkpoint on all three samples.
#
#     CKPT=/path/to/model.pt sbatch slurm/eval_v3.hpc.sbatch
#
# All three samples are evaluated with --split test under the same seed, so each
# contributes its own held-out half and the three arms are treated identically.
# For the sample the model was trained on this is a genuine holdout; for the
# other two the split is arbitrary but keeps the selection and the event counts
# uniform across the comparison, which is what the S_pot ratio needs.
#
#SBATCH --job-name=eval_v3
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint="type_e|type_f|type_h"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${ENV_NAME:=hgnd-env}"
: "${PROJECT_DIR:=$HOME}"
: "${SCRATCH:=/scratch/$USER}"
: "${TAG:=v3}"
: "${NUM_SHARDS:=1200}"
: "${CKPT:?set CKPT to the checkpoint to evaluate}"
: "${OUTDIR:=$SCRATCH/hgnd/results/eval_${TAG}_${SLURM_JOB_ID:-local}}"

module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
export PYTHONPATH="$HOME/HGNDRecoGNN:${PYTHONPATH:-}"
mkdir -p "${SLURM_SUBMIT_DIR:-$HOME/HGNDRecoGNN}/logs" "$OUTDIR"

echo "host: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-none}"
echo "ckpt: $CKPT"
echo "out:  $OUTDIR"
echo

for DS in zeroSpot defaultSpot bigSpot; do
    ROOT="$SCRATCH/hgnd/cache/ndet_dataset_smash_${DS}_${TAG}"
    [ -d "$ROOT/processed" ] || { echo "skip $DS: no cache"; continue; }
    if [ -f "$OUTDIR/$DS/pred_clusters_smash.pkl" ]; then echo "skip $DS: done"; continue; fi
    echo "=== inference $DS ==="
    python -u -m HGNDRecoGNN.scripts.evaluate \
        --root "$ROOT" --checkpoint "$CKPT" --out-dir "$OUTDIR/$DS" \
        --suffix smash --split test --num-shards "$NUM_SHARDS" \
        --batch-size 512 --device auto
done

echo
echo "=== cluster performance against the reference benchmarks ==="
python "$HOME/HGNDRecoGNN/scripts/reco_performance.py" \
    --clusters "$OUTDIR"/*/pred_clusters_smash.pkl \
    --out "$OUTDIR/performance.json" | tee "$OUTDIR/performance.txt"

echo
echo "outputs in $OUTDIR"
ls -la "$OUTDIR"
