#!/bin/bash
# sensitivity_full.hpc.sbatch — full-statistics sensitivity study.
#
# Evaluates the 2026-08-26 HPC checkpoint
#     $SCRATCH/hgnd/checkpoints/defaultSpot_v2_seed42_20ep_after4278192/model.pt
# on all three SMASH datasets at 200k events each, in the seed-42
# test half (matched sample size), and runs the sensitivity CLI in
# both efficiency bases + purity-lock diagnostic. Results land under
# $SCRATCH/hgnd/results/sensitivity_full_hpc_${SLURM_JOB_ID}/.
#
# Prerequisites (fail fast if missing):
#   - $SCRATCH/hgnd/checkpoints/.../model.pt
#   - $SCRATCH/hgnd/cache/ndet_dataset_smash_{defaultSpot,zeroSpot,bigSpot}_v2/
#     each with schema_version==2 (build via preprocess_all_smash.hpc.sbatch).
#
# Submit from ~/HGNDRecoGNN on login-02:
#   sbatch slurm/sensitivity_full.hpc.sbatch
# Override the checkpoint via:
#   sbatch --export=ALL,CKPT=/scratch/.../valloss/model.pt slurm/sensitivity_full.hpc.sbatch
#
#SBATCH --job-name=hgnd-sensitivity-full
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${ENV_NAME:=hgnd-env}"
: "${PROJECT_DIR:=$HOME/HGNDRecoGNN}"
: "${SCRATCH:=/scratch/$USER}"
: "${CKPT:=$SCRATCH/hgnd/checkpoints/defaultSpot_v2_seed42_20ep_after4278192/model.pt}"
: "${BATCH_SIZE:=512}"
: "${THRESHOLD:=0.5}"
: "${TARGET_PURITY:=0.7}"

module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

mkdir -p logs
cd "$PROJECT_DIR"

OUT_ROOT="$SCRATCH/hgnd/results/sensitivity_full_hpc_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_ROOT"

echo "host:       $(hostname)"
echo "date:       $(date)"
echo "job:        ${SLURM_JOB_ID:-local}  part=${SLURM_JOB_PARTITION:-?}"
echo "gpu:        ${CUDA_VISIBLE_DEVICES:-none}"
echo "env:        $ENV_NAME  cwd=$PROJECT_DIR"
echo "checkpoint: $CKPT"
echo "batch:      $BATCH_SIZE   threshold: $THRESHOLD   purity: $TARGET_PURITY"
echo "out root:   $OUT_ROOT"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint $CKPT missing" >&2
    exit 3
fi

# ── 1) Evaluate on all three full v2 datasets, seed-42 test half ────────
DATASETS=(defaultSpot zeroSpot bigSpot)
declare -a SENS_ARGS=()
declare -a PURITY_ARGS=()
for ds in "${DATASETS[@]}"; do
    ROOT="$SCRATCH/hgnd/cache/ndet_dataset_smash_${ds}_v2"
    if [[ ! -f "$ROOT/processed/meta.json" ]]; then
        echo "ERROR: missing v2 cache for $ds at $ROOT" >&2
        exit 3
    fi
    OUT="$OUT_ROOT/${ds}_hpc"
    echo
    echo "=== evaluate $ds on seed-42 test half ==="
    python -u -m HGNDRecoGNN.scripts.evaluate \
        --root "$ROOT" \
        --checkpoint "$CKPT" \
        --out-dir "$OUT" \
        --suffix hpc_full \
        --batch-size "$BATCH_SIZE" \
        --split test \
        --seed 42 --train-frac 0.5 \
        --device auto
    SENS_ARGS+=(--dataset "${ds}=${OUT}/pred_clusters_hpc_full.pkl")
    PURITY_ARGS+=(--dataset "${ds}=${OUT}/pred_clusters_hpc_full.pkl")
done

# ── 2) Sensitivity CLI: both bases + unfold ─────────────────────────────
CACHE_ROOT="$SCRATCH/hgnd/cache"
for basis in e_pred e_true; do
    OUT="$OUT_ROOT/sensitivity_${basis}"
    mkdir -p "$OUT"
    echo
    echo "=== sensitivity CLI ($basis basis, unfold=tikhonov) ==="
    python -u -m HGNDRecoGNN.scripts.sensitivity \
        "${SENS_ARGS[@]}" \
        --out-dir "$OUT" \
        --dataset-cache-root "$CACHE_ROOT" \
        --efficiency-basis "$basis" \
        --unfold --unfold-method tikhonov --unfold-lambda 0.01 \
        --threshold "$THRESHOLD"
done

# ── 3) Purity-locked closure sweep ───────────────────────────────────────
OUT="$OUT_ROOT/purity_locked_pi$(printf '%02d' $(python -c "print(int($TARGET_PURITY*100))"))"
mkdir -p "$OUT"
echo
echo "=== purity-locked closure @ pi=$TARGET_PURITY ==="
python -u -m HGNDRecoGNN.scripts.purity_locked_closure \
    "${PURITY_ARGS[@]}" \
    --dataset-cache-root "$CACHE_ROOT" \
    --purity "$TARGET_PURITY" \
    --reference-threshold "$THRESHOLD" \
    --out-dir "$OUT"

echo
echo "HGND_EXIT=0"
echo "sensitivity full done: $(date)"
echo "outputs:    $OUT_ROOT"
echo
echo "Retrieve with (from your laptop):"
echo "  rsync -av login-02:$OUT_ROOT/ ~/Project/BM@N/HGND/HGNDRecoGNN/results/\$(basename $OUT_ROOT)/"
