#!/bin/bash
# sensitivity_systematics_finish.hpc.sbatch — resume the timed-out /
# preempted seed-variance runs from 4304929. Both net_default_seed123
# (task 3, CANCELLED after zeroSpot) and net_default_seed456 (task 4,
# FAILED on bigSpot shard_551 lustre PermissionError) had produced
# defaultSpot + zeroSpot prediction pickles before dying. This job
# evaluates ONLY bigSpot per seed, then runs sensitivity + purity-lock
# CLIs against the reunited three-dataset trio.
#
# Submit:
#   sbatch --array=0,1 slurm/sensitivity_systematics_finish.hpc.sbatch
#
# Array indices:
#   0 → net_default_seed123
#   1 → net_default_seed456
#
#SBATCH --job-name=hgnd-sens-syst-fin
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${ENV_NAME:=hgnd-env}"
: "${PROJECT_DIR:=$HOME}"
: "${SCRATCH:=/scratch/$USER}"
: "${THRESHOLD:=0.5}"
: "${TARGET_PURITY:=0.7}"

module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
export PYTHONPATH="$HOME/HGNDRecoGNN:${PYTHONPATH:-}"
mkdir -p "${SLURM_SUBMIT_DIR:-$HOME/HGNDRecoGNN}/logs"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
case "$IDX" in
  0) TAG=net_default_seed123 ;;
  1) TAG=net_default_seed456 ;;
  *) echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range 0-1" >&2 ; exit 2 ;;
esac

CKPT="$SCRATCH/hgnd/checkpoints/systematics/${TAG}/model.pt"
OUT_ROOT="$SCRATCH/hgnd/results/systematics/${TAG}"
CACHE_ROOT="$SCRATCH/hgnd/cache"

echo "host:       $(hostname)"
echo "date:       $(date)"
echo "job:        ${SLURM_JOB_ID:-local}  array=${SLURM_ARRAY_JOB_ID:-?}/${SLURM_ARRAY_TASK_ID:-?}"
echo "tag:        $TAG"

for ds in defaultSpot zeroSpot; do
    pkl="$OUT_ROOT/$ds/pred_clusters_syst_${TAG}.pkl"
    if [[ ! -f "$pkl" ]]; then
        echo "ERROR: expected reused prediction $pkl missing" >&2
        exit 3
    fi
    echo "reused $ds pickle: $pkl"
done

# 1) Evaluate bigSpot only
BS_ROOT="$CACHE_ROOT/ndet_dataset_smash_bigSpot_v2"
if [[ ! -f "$BS_ROOT/processed/meta.json" ]]; then
    echo "ERROR: missing bigSpot v2 cache at $BS_ROOT" >&2
    exit 3
fi
echo
echo "=== evaluate bigSpot ==="
python -u -m HGNDRecoGNN.scripts.evaluate \
    --root "$BS_ROOT" \
    --checkpoint "$CKPT" \
    --out-dir "$OUT_ROOT/bigSpot" \
    --suffix "syst_${TAG}" \
    --batch-size 512 \
    --split test \
    --seed 42 --train-frac 0.5 \
    --device auto

# 2) Sensitivity CLI over the three reunited datasets
declare -a SENS_ARGS=()
for ds in defaultSpot zeroSpot bigSpot; do
    SENS_ARGS+=(--dataset "${ds}=${OUT_ROOT}/${ds}/pred_clusters_syst_${TAG}.pkl")
done
mkdir -p "$OUT_ROOT/sensitivity_e_pred" "$OUT_ROOT/purity_locked_pi70"

echo
echo "=== sensitivity CLI ==="
python -u -m HGNDRecoGNN.scripts.sensitivity \
    "${SENS_ARGS[@]}" \
    --out-dir "$OUT_ROOT/sensitivity_e_pred" \
    --dataset-cache-root "$CACHE_ROOT" \
    --efficiency-basis e_pred \
    --unfold --unfold-method tikhonov --unfold-lambda 0.01 \
    --threshold "$THRESHOLD" --no-plot

echo
echo "=== purity-locked closure @ pi=$TARGET_PURITY ==="
python -u -m HGNDRecoGNN.scripts.purity_locked_closure \
    "${SENS_ARGS[@]}" \
    --dataset-cache-root "$CACHE_ROOT" \
    --purity "$TARGET_PURITY" \
    --reference-threshold "$THRESHOLD" \
    --out-dir "$OUT_ROOT/purity_locked_pi70"

echo
echo "HGND_EXIT=0"
echo "systematics finish $TAG done: $(date)"
