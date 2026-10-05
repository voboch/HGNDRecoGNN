#!/bin/bash
# eval_dcm.hpc.sbatch — reconstruction performance on the DCM reference sample.
#
# Runs one checkpoint over a DCM cache and scores it against the published
# benchmarks (ROC AUC ~0.97, ~80 % efficiency at ~87 % energy purity, linearity
# and resolution under 10 % from 0.7 to 5 GeV).
#
# The two tests this is meant to serve:
#   CKPT=<SMASH checkpoint>  SCALER=smash   domain transfer — how much of our
#       SMASH performance survives the change of generator;
#   CKPT=<DCM checkpoint>    SCALER=own     the like-for-like reproduction of
#       the reference, which is what the benchmarks can fairly be read against.
# Pairing a SMASH checkpoint with SCALER=own would let a DCM-fitted scaler
# absorb the domain shift, so the job refuses that combination unless FORCE=1.
#
#     CKPT=/scratch/$USER/hgnd/checkpoints/train_v3_chain/model.pt \
#         SCALER=smash sbatch slurm/eval_dcm.hpc.sbatch
#
#SBATCH --job-name=eval_dcm
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint="type_e|type_f|type_h"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${ENV_NAME:=hgnd-env}"
: "${PROJECT_DIR:=$HOME}"
: "${SCRATCH:=/scratch/$USER}"
: "${CKPT:?set CKPT to the checkpoint to evaluate}"
: "${SCALER:=own}"
: "${TWINDOW:=1.0}"
: "${NUM_SHARDS:=0}"
: "${SPLIT:=test}"
: "${FORCE:=0}"

module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
export PYTHONPATH="$HOME/HGNDRecoGNN:${PYTHONPATH:-}"
mkdir -p "${SLURM_SUBMIT_DIR:-$HOME/HGNDRecoGNN}/logs"

# 1.0 -> 1p0, 1.5 -> 1p5. Bash substitution: the python one-liner this
# replaced nested same-type quotes inside an f-string, which only parses on
# 3.12+, and the job environment is older — both DCM jobs died on it in 2 s.
TW_TAG=${TWINDOW//./p}
ROOT="$SCRATCH/hgnd/cache/ndet_dataset_dcm_xecs_${SCALER}_tw${TW_TAG}"
: "${OUTDIR:=$SCRATCH/hgnd/results/eval_dcm_${SCALER}_tw${TW_TAG}_${SLURM_JOB_ID:-local}}"
mkdir -p "$OUTDIR"

if [ ! -d "$ROOT/processed" ]; then
    echo "ERROR: no cache at $ROOT — run preprocess_dcm.hpc.sbatch first." >&2
    exit 3
fi
case "$CKPT" in
    *dcm*) ;;
    *) if [ "$SCALER" = "own" ] && [ "$FORCE" != "1" ]; then
           echo "ERROR: '$CKPT' does not look like a DCM checkpoint, and" >&2
           echo "SCALER=own would refit the scaler on DCM, hiding the domain" >&2
           echo "shift this test exists to measure. Use SCALER=smash, or set" >&2
           echo "FORCE=1 if the mismatch is intended." >&2
           exit 4
       fi ;;
esac

echo "host: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-none}"
echo "date: $(date)"
echo "ckpt: $CKPT"
echo "root: $ROOT   (scaler=$SCALER, twindow=$TWINDOW)"
echo "out:  $OUTDIR"
echo

python -u -m HGNDRecoGNN.scripts.evaluate \
    --root "$ROOT" --checkpoint "$CKPT" --out-dir "$OUTDIR" \
    --suffix dcm --split "$SPLIT" --num-shards "$NUM_SHARDS" \
    --batch-size 512 --device auto

echo
echo "=== cluster performance against the reference benchmarks ==="
python "$HOME/HGNDRecoGNN/scripts/reco_performance.py" \
    --clusters "$OUTDIR/pred_clusters_dcm.pkl" \
    --out "$OUTDIR/performance.json" | tee "$OUTDIR/performance.txt"

echo
echo "outputs in $OUTDIR"
ls -la "$OUTDIR"
