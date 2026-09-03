#!/bin/bash
# sensitivity_finish.hpc.sbatch — resume the 4290340 job that timed out
# after producing defaultSpot + zeroSpot predictions but before bigSpot
# eval and the sensitivity/purity-lock CLI stages.
#
# Steps:
#   1. Evaluate bigSpot only (defaultSpot + zeroSpot pickles are reused
#      from ${IN_ROOT}).
#   2. Run scripts.sensitivity in both bases with tikhonov unfold.
#   3. Run scripts.purity_locked_closure at pi = 0.7.
#
# All outputs land under a fresh $SCRATCH/hgnd/results/
# sensitivity_full_hpc_finish_${SLURM_JOB_ID}/ directory, with the
# recovered defaultSpot + zeroSpot pickles rsync'd in so downstream
# consumers see one consistent tree.
#
# Submit from ~/HGNDRecoGNN on login-02:
#   sbatch slurm/sensitivity_finish.hpc.sbatch
# Override via:
#   sbatch --export=ALL,IN_ROOT=/scratch/.../sensitivity_full_hpc_4290340 \
#          slurm/sensitivity_finish.hpc.sbatch
#
#SBATCH --job-name=hgnd-sens-finish
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${ENV_NAME:=hgnd-env}"
: "${PROJECT_DIR:=$HOME}"
: "${SCRATCH:=/scratch/$USER}"
: "${CKPT:=$SCRATCH/hgnd/checkpoints/defaultSpot_v2_seed42_20ep_after4278192/model.pt}"
: "${IN_ROOT:=$SCRATCH/hgnd/results/sensitivity_full_hpc_4290340}"
: "${BATCH_SIZE:=512}"
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

OUT_ROOT="$SCRATCH/hgnd/results/sensitivity_full_hpc_finish_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_ROOT"

echo "host:       $(hostname)"
echo "date:       $(date)"
echo "job:        ${SLURM_JOB_ID:-local}  part=${SLURM_JOB_PARTITION:-?}"
echo "gpu:        ${CUDA_VISIBLE_DEVICES:-none}"
echo "in root:    $IN_ROOT"
echo "out root:   $OUT_ROOT"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ── 1) Reuse defaultSpot + zeroSpot predictions from the timed-out job ──
for ds in defaultSpot zeroSpot; do
    src="$IN_ROOT/${ds}_hpc"
    dst="$OUT_ROOT/${ds}_hpc"
    if [[ ! -f "$src/pred_clusters_hpc_full.pkl" ]]; then
        echo "ERROR: missing $src/pred_clusters_hpc_full.pkl" >&2
        exit 3
    fi
    mkdir -p "$dst"
    ln -sfv "$src/pred_hits_hpc_full.pkl"     "$dst/"
    ln -sfv "$src/pred_clusters_hpc_full.pkl" "$dst/"
    ln -sfv "$src/pred_edges_hpc_full.pkl"    "$dst/"
done

# ── 2) Evaluate bigSpot only ─────────────────────────────────────────────
BS_ROOT="$SCRATCH/hgnd/cache/ndet_dataset_smash_bigSpot_v2"
if [[ ! -f "$BS_ROOT/processed/meta.json" ]]; then
    echo "ERROR: missing bigSpot v2 cache at $BS_ROOT" >&2
    exit 3
fi
echo
echo "=== evaluate bigSpot on seed-42 test half ==="
python -u -m HGNDRecoGNN.scripts.evaluate \
    --root "$BS_ROOT" \
    --checkpoint "$CKPT" \
    --out-dir "$OUT_ROOT/bigSpot_hpc" \
    --suffix hpc_full \
    --batch-size "$BATCH_SIZE" \
    --split test \
    --seed 42 --train-frac 0.5 \
    --device auto

# ── 3) Sensitivity CLI in both bases + unfold ────────────────────────────
declare -a SENS_ARGS=()
for ds in defaultSpot zeroSpot bigSpot; do
    SENS_ARGS+=(--dataset "${ds}=${OUT_ROOT}/${ds}_hpc/pred_clusters_hpc_full.pkl")
done
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

# ── 4) Purity-locked closure sweep ───────────────────────────────────────
OUT="$OUT_ROOT/purity_locked_pi$(printf '%02d' $(python -c "print(int($TARGET_PURITY*100))"))"
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
echo "sensitivity finish done: $(date)"
echo "outputs: $OUT_ROOT"
