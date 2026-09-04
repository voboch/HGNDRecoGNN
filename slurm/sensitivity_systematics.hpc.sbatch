#!/bin/bash
# sensitivity_systematics.hpc.sbatch — evaluate each systematics
# checkpoint on the three lift-cap v2 caches, dump per-cluster
# closure summary CSVs. Run after slurm/train_systematics.hpc.sbatch
# completes (auto-chainable via --dependency=afterok).
#
# Array indices match the training script; each task evaluates one
# checkpoint on defaultSpot + zeroSpot + bigSpot at full statistics
# and emits sensitivity_summary.csv into
# $SCRATCH/hgnd/results/systematics/${TAG}/.
#
# Submit:
#   sbatch --array=0-4 slurm/sensitivity_systematics.hpc.sbatch
#
#SBATCH --job-name=hgnd-sens-syst
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
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
  0) TAG=hetero_hgt_seed42       ;;
  1) TAG=hetero_sage_seed42      ;;
  2) TAG=spectral_dynedge_seed42 ;;
  3) TAG=net_default_seed123     ;;
  4) TAG=net_default_seed456     ;;
  *) echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range 0-4" >&2 ; exit 2 ;;
esac

CKPT="$SCRATCH/hgnd/checkpoints/systematics/${TAG}/model.pt"
OUT_ROOT="$SCRATCH/hgnd/results/systematics/${TAG}"
CACHE_ROOT="$SCRATCH/hgnd/cache"

echo "host:       $(hostname)"
echo "date:       $(date)"
echo "job:        ${SLURM_JOB_ID:-local}  array=${SLURM_ARRAY_JOB_ID:-?}/${SLURM_ARRAY_TASK_ID:-?}"
echo "tag:        $TAG"
echo "checkpoint: $CKPT"
echo "out root:   $OUT_ROOT"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: missing systematics checkpoint at $CKPT" >&2
    exit 3
fi
mkdir -p "$OUT_ROOT"

# 1) Evaluate on the three lift-cap v2 caches, seed-42 test half
declare -a SENS_ARGS=()
for ds in defaultSpot zeroSpot bigSpot; do
    ROOT="$CACHE_ROOT/ndet_dataset_smash_${ds}_v2"
    if [[ ! -f "$ROOT/processed/meta.json" ]]; then
        echo "ERROR: missing v2 cache for $ds at $ROOT" >&2
        exit 3
    fi
    OUT="$OUT_ROOT/${ds}"
    echo
    echo "=== evaluate $ds ==="
    python -u -m HGNDRecoGNN.scripts.evaluate \
        --root "$ROOT" \
        --checkpoint "$CKPT" \
        --out-dir "$OUT" \
        --suffix "syst_${TAG}" \
        --batch-size 512 \
        --split test \
        --seed 42 --train-frac 0.5 \
        --device auto
    SENS_ARGS+=(--dataset "${ds}=${OUT}/pred_clusters_syst_${TAG}.pkl")
done

# 2) Sensitivity CLI at e_pred basis with unfold (matches paper convention)
OUT="$OUT_ROOT/sensitivity_e_pred"
mkdir -p "$OUT"
echo
echo "=== sensitivity CLI ==="
python -u -m HGNDRecoGNN.scripts.sensitivity \
    "${SENS_ARGS[@]}" \
    --out-dir "$OUT" \
    --dataset-cache-root "$CACHE_ROOT" \
    --efficiency-basis e_pred \
    --unfold --unfold-method tikhonov --unfold-lambda 0.01 \
    --threshold "$THRESHOLD" --no-plot

# 3) Purity-locked closure
OUT="$OUT_ROOT/purity_locked_pi70"
mkdir -p "$OUT"
echo
echo "=== purity-locked closure @ pi=$TARGET_PURITY ==="
python -u -m HGNDRecoGNN.scripts.purity_locked_closure \
    "${SENS_ARGS[@]}" \
    --dataset-cache-root "$CACHE_ROOT" \
    --purity "$TARGET_PURITY" \
    --reference-threshold "$THRESHOLD" \
    --out-dir "$OUT"

echo
echo "HGND_EXIT=0"
echo "systematics eval $TAG done: $(date)"
echo "outputs: $OUT_ROOT"
