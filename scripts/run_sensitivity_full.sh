#!/usr/bin/env bash
# Full-scale reproduction of the HGND sensitivity study.
#
# Assumes: (a) the three CSV dumps live under ${REPO}/data/<dataset>/, and
# (b) the machine has enough RAM to load one full dataset into pandas
# (~40 GB peak for 12 GB CSV zeroSpot). If in doubt, run on the cluster.
#
# Behaviour:
#   1. Preprocess all datasets to full extent (no max_events cap).
#   2. Train ${MODEL} on ${TRAIN_DATASET} with production hyperparams.
#   3. Evaluate the checkpoint on every dataset in ${DATASETS}.
#   4. Run scripts/sensitivity.py to compare them.
#
# Env knobs (all overridable):
#   TRAIN_DATASET   — training dataset            (defaultSpot)
#   DATASETS        — space-separated eval list   (defaultSpot zeroSpot bigSpot)
#   MODEL           — model registry name         (net_default)
#   HIDDEN          — hidden channels             (512)
#   NUM_LAYERS      — GNN depth                   (8)
#   BATCH_SIZE      — training batch              (512)
#   EPOCHS          — training epochs             (20)
#   LR              — learning rate               (1e-3)
#   NUM_WORKERS_PROC — preprocess workers         (8)
#   PRELOAD_MAX_GB  — preload memory ceiling      (unset → all)
#   DEVICE          — cuda|mps|cpu|auto           (auto)
#   THRESHOLD       — sensitivity cluster cut     (0.5)
#   OUT_ROOT        — where results land          (results/sensitivity_full)
#
# Usage:
#   ./scripts/run_sensitivity_full.sh
#   MODEL=hetero_hgt EPOCHS=30 ./scripts/run_sensitivity_full.sh

set -euo pipefail

: "${TRAIN_DATASET:=defaultSpot}"
: "${DATASETS:=defaultSpot zeroSpot bigSpot}"
: "${MODEL:=net_default}"
: "${HIDDEN:=512}"
: "${NUM_LAYERS:=8}"
: "${BATCH_SIZE:=512}"
: "${EPOCHS:=20}"
: "${LR:=1e-3}"
: "${NUM_WORKERS_PROC:=8}"
: "${DEVICE:=auto}"
: "${THRESHOLD:=0.5}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OUT_ROOT:=${REPO}/results/sensitivity_full_${SLURM_JOB_ID:-local}}"

export PYTHONPATH="${REPO}/..:${PYTHONPATH:-}"

CKPT_DIR="${OUT_ROOT}/checkpoints"
mkdir -p "$CKPT_DIR" "$OUT_ROOT"

echo "=== HGND sensitivity FULL run ==="
echo "  repo:          $REPO"
echo "  train dataset: $TRAIN_DATASET"
echo "  eval datasets: $DATASETS"
echo "  model:         $MODEL  hidden=$HIDDEN layers=$NUM_LAYERS"
echo "  train:         batch=$BATCH_SIZE  epochs=$EPOCHS  lr=$LR"
echo "  device:        $DEVICE"
echo "  out root:      $OUT_ROOT"
echo

dataset_root() { echo "${REPO}/notebooks/cache/ndet_dataset_smash_$1"; }
csv_dir()      { echo "${REPO}/data/smash_xecs_2.87gev_hardSkyrme_$1"; }
is_ready()     { [[ -f "$(dataset_root "$1")/processed/meta.json" ]]; }

preload_flag=()
if [[ -n "${PRELOAD_MAX_GB:-}" ]]; then
    preload_flag=(--preload-max-gb "$PRELOAD_MAX_GB")
fi

# ── 1) Preprocess ────────────────────────────────────────────────────────
for ds in $DATASETS; do
    if is_ready "$ds"; then
        echo "[preprocess] $ds  READY"
        continue
    fi
    echo "[preprocess] $ds  building full dataset"
    python -m HGNDRecoGNN.scripts.preprocess \
        --csv-dir     "$(csv_dir "$ds")" \
        --root        "$(dataset_root "$ds")" \
        --num-workers "$NUM_WORKERS_PROC"
done

# ── 2) Train ─────────────────────────────────────────────────────────────
CKPT="$CKPT_DIR/${TRAIN_DATASET}_${MODEL}_full.pt"
echo
echo "[train] $TRAIN_DATASET → $CKPT"
python -m HGNDRecoGNN.scripts.train \
    --root            "$(dataset_root "$TRAIN_DATASET")" \
    --model           "$MODEL" \
    --hidden          "$HIDDEN" \
    --num-layers      "$NUM_LAYERS" \
    --epochs          "$EPOCHS" \
    --batch-size      "$BATCH_SIZE" \
    --lr              "$LR" \
    --device          "$DEVICE" \
    --checkpoint-dir  "$CKPT_DIR" \
    --checkpoint-name "${TRAIN_DATASET}_${MODEL}_full.pt" \
    "${preload_flag[@]}"

# ── 3) Evaluate on every dataset ────────────────────────────────────────
# split=test on all datasets so raw counts are on matched sample sizes —
# a prerequisite for comparing N_reco / N_true_solved cross-dataset. For
# the training dataset this is the honest OOD test; for the others the
# "test" split is just a deterministic half.
declare -a SENS_ARGS=()
for ds in $DATASETS; do
    if ! is_ready "$ds"; then
        echo "[evaluate] $ds  SKIP (preprocess failed?)"
        continue
    fi
    out="${OUT_ROOT}/${ds}_${MODEL}"
    echo "[evaluate] $ds  split=test  → $out"
    python -m HGNDRecoGNN.scripts.evaluate \
        --root       "$(dataset_root "$ds")" \
        --checkpoint "$CKPT" \
        --out-dir    "$out" \
        --suffix     full \
        --batch-size "$BATCH_SIZE" \
        --split      test \
        --device     "$DEVICE" \
        "${preload_flag[@]}"
    SENS_ARGS+=(--dataset "${ds}=${out}/pred_clusters_full.pkl")
done

# ── 4) Cross-dataset comparison ─────────────────────────────────────────
SENS_OUT="${OUT_ROOT}/sensitivity"
DATASET_CACHE="${REPO}/notebooks/cache"
echo
echo "[sensitivity] threshold=$THRESHOLD → $SENS_OUT"
python -m HGNDRecoGNN.scripts.sensitivity \
    "${SENS_ARGS[@]}" \
    --out-dir            "$SENS_OUT" \
    --dataset-cache-root "$DATASET_CACHE" \
    --threshold          "$THRESHOLD"

echo
echo "=== done ==="
echo "  checkpoints: $CKPT_DIR"
echo "  predictions: $OUT_ROOT/<dataset>_${MODEL}/pred_clusters_full.pkl"
echo "  sensitivity: $SENS_OUT"
echo
echo "Open notebooks/sensitivity_full_analysis.ipynb and point"
echo "RESULTS_DIR at $OUT_ROOT to inspect the run."
