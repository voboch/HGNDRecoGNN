#!/usr/bin/env bash
# Smoke reproduction of notebooks/sensitivity_smoketest.ipynb using the CLI
# scripts. Runs on a laptop (default) or on a SLURM node (via the sbatch
# wrapper) with the same code path.
#
# Behaviour:
#   1. For each dataset in ${DATASETS} (default: defaultSpot zeroSpot bigSpot):
#      preprocess a small slice if the processed shards are missing AND
#      DO_PREPROCESS=1. Otherwise skip with a message.
#   2. Train net_default on ${TRAIN_DATASET} (default: defaultSpot).
#   3. Evaluate the trained checkpoint on every ready dataset.
#   4. Run scripts/sensitivity.py to compare them.
#
# Env knobs (all overridable, sensible defaults for a smoke run):
#   TRAIN_DATASET     — dataset used for training           (defaultSpot)
#   DATASETS          — space-separated eval datasets       (defaultSpot zeroSpot bigSpot)
#   MODEL             — model registry name                 (net_default)
#   HIDDEN            — hidden channels                     (256)
#   NUM_LAYERS        — GNN depth                           (4)
#   BATCH_SIZE        — training batch size                 (32)
#   EPOCHS            — training epochs                     (3)
#   NUM_SHARDS        — cap loaded shards per dataset       (2)
#   MAX_EVENTS        — cap events per side during preproc  (2000)
#   NUM_WORKERS_PROC  — CPU workers for preprocessing       (4)
#   PRELOAD_MAX_GB    — preload memory ceiling              (1.0)
#   DEVICE            — cuda|mps|cpu|auto                   (auto)
#   DO_PREPROCESS     — 1 to preprocess missing datasets    (0 on laptop)
#   OUT_ROOT          — where checkpoints + results land    (results/sensitivity_smoke)
#
# Usage:
#   ./scripts/run_sensitivity_smoke.sh                                # laptop
#   DO_PREPROCESS=1 NUM_WORKERS_PROC=8 ./scripts/run_sensitivity_smoke.sh   # cluster

set -euo pipefail

: "${TRAIN_DATASET:=defaultSpot}"
: "${DATASETS:=defaultSpot zeroSpot bigSpot}"
: "${MODEL:=net_default}"
: "${HIDDEN:=256}"
: "${NUM_LAYERS:=4}"
: "${BATCH_SIZE:=32}"
: "${EPOCHS:=3}"
: "${NUM_SHARDS:=2}"
: "${MAX_EVENTS:=2000}"
: "${NUM_WORKERS_PROC:=4}"
: "${PRELOAD_MAX_GB:=1.0}"
: "${DEVICE:=auto}"
: "${DO_PREPROCESS:=0}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OUT_ROOT:=${REPO}/results/sensitivity_smoke}"

export PYTHONPATH="${REPO}/..:${PYTHONPATH:-}"

CKPT_DIR="${OUT_ROOT}/checkpoints"
mkdir -p "$CKPT_DIR" "$OUT_ROOT"

echo "=== HGND sensitivity smoke run ==="
echo "  repo:            $REPO"
echo "  train dataset:   $TRAIN_DATASET"
echo "  eval datasets:   $DATASETS"
echo "  model:           $MODEL  hidden=$HIDDEN layers=$NUM_LAYERS"
echo "  train:           batch=$BATCH_SIZE  epochs=$EPOCHS  shards=$NUM_SHARDS"
echo "  device:          $DEVICE"
echo "  do_preprocess:   $DO_PREPROCESS"
echo "  out root:        $OUT_ROOT"
echo

dataset_root() {
    echo "${REPO}/notebooks/cache/ndet_dataset_smash_$1"
}
csv_dir() {
    echo "${REPO}/data/smash_xecs_2.87gev_hardSkyrme_$1"
}
is_ready() {
    [[ -f "$(dataset_root "$1")/processed/meta.json" ]]
}

# ── 1) Preprocess whichever datasets need it and are allowed ─────────────
for ds in $DATASETS; do
    if is_ready "$ds"; then
        echo "[preprocess] $ds  READY  ($(dataset_root "$ds"))"
        continue
    fi
    if [[ "$DO_PREPROCESS" != "1" ]]; then
        echo "[preprocess] $ds  SKIP   (DO_PREPROCESS=0)"
        continue
    fi
    echo "[preprocess] $ds  building max_events=$MAX_EVENTS"
    python -m HGNDRecoGNN.scripts.preprocess \
        --csv-dir     "$(csv_dir "$ds")" \
        --root        "$(dataset_root "$ds")" \
        --max-events  "$MAX_EVENTS" \
        --num-workers "$NUM_WORKERS_PROC"
done

# ── 2) Train ─────────────────────────────────────────────────────────────
if ! is_ready "$TRAIN_DATASET"; then
    echo "error: training dataset $TRAIN_DATASET has no processed shards." >&2
    echo "        Rerun with DO_PREPROCESS=1 or preprocess it manually." >&2
    exit 1
fi
CKPT="$CKPT_DIR/${TRAIN_DATASET}_${MODEL}_smoke.pt"
echo
echo "[train] $TRAIN_DATASET → $CKPT"
python -m HGNDRecoGNN.scripts.train \
    --root              "$(dataset_root "$TRAIN_DATASET")" \
    --model             "$MODEL" \
    --hidden            "$HIDDEN" \
    --num-layers        "$NUM_LAYERS" \
    --epochs            "$EPOCHS" \
    --batch-size        "$BATCH_SIZE" \
    --num-shards        "$NUM_SHARDS" \
    --preload-max-gb    "$PRELOAD_MAX_GB" \
    --device            "$DEVICE" \
    --checkpoint-dir    "$CKPT_DIR" \
    --checkpoint-name   "${TRAIN_DATASET}_${MODEL}_smoke.pt"

# ── 3) Evaluate on every ready dataset ───────────────────────────────────
# All datasets use split=test with the same seed / train_frac so raw
# counts land on matched sample sizes — a prerequisite for the raw
# N_reco / N_true_solved cross-dataset comparison. For the training
# dataset this is the honest OOD test; for the others "test" is just a
# deterministic half chosen for size parity.
declare -a SENS_ARGS=()
for ds in $DATASETS; do
    if ! is_ready "$ds"; then
        echo "[evaluate] $ds  SKIP  (no shards; run cluster preprocess)"
        continue
    fi
    out="${OUT_ROOT}/${ds}_${MODEL}"
    echo "[evaluate] $ds  split=test  → $out"
    python -m HGNDRecoGNN.scripts.evaluate \
        --root           "$(dataset_root "$ds")" \
        --checkpoint     "$CKPT" \
        --out-dir        "$out" \
        --suffix         smoke \
        --num-shards     "$NUM_SHARDS" \
        --batch-size     "$BATCH_SIZE" \
        --preload-max-gb "$PRELOAD_MAX_GB" \
        --split          test \
        --device         "$DEVICE"
    SENS_ARGS+=(--dataset "${ds}=${out}/pred_clusters_smoke.pkl")
done

# ── 4) Cross-dataset comparison ──────────────────────────────────────────
if [[ "${#SENS_ARGS[@]}" -eq 0 ]]; then
    echo "no datasets were evaluated — sensitivity comparison skipped."
    exit 0
fi
SENS_OUT="${OUT_ROOT}/sensitivity"
DATASET_CACHE="${REPO}/notebooks/cache"
echo
echo "[sensitivity] → $SENS_OUT"
python -m HGNDRecoGNN.scripts.sensitivity \
    "${SENS_ARGS[@]}" \
    --out-dir            "$SENS_OUT" \
    --dataset-cache-root "$DATASET_CACHE" \
    --threshold          0.5

echo
echo "=== done ==="
echo "  checkpoints: $CKPT_DIR"
echo "  predictions: $OUT_ROOT/<dataset>_${MODEL}/pred_clusters_smoke.pkl"
echo "  sensitivity: $SENS_OUT"
