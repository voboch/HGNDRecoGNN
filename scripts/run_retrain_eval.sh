#!/usr/bin/env bash
# Evaluate a retrained checkpoint: inference on each split, cluster performance
# against the Dombay benchmarks, then the S_pot working point chosen on
# train+val and applied once to test.
#
#   run_retrain_eval.sh <ds_dir> <raw_split_dir> <checkpoint> <pred_dir> <out_dir>
#
# ds_dir holds train/ val/ test/ preprocessed roots built with ONE scaler fitted
# on train; raw_split_dir holds the matching CSV trees, whose run subdirectories
# are the sample names.
set -euo pipefail

DS="${1:?ds dir}"; RAW="${2:?raw split dir}"; CK="${3:?checkpoint}"
PRED="${4:?pred dir}"; OUT="${5:?out dir}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-python3}"

mkdir -p "$PRED" "$OUT"

echo "== 1/3  inference on each split =="
for s in train val test; do
    if [ -f "$PRED/$s/pred_clusters_smash.pkl" ]; then
        echo "  $s already done"; continue
    fi
    ( cd "$PKG" && "$PY" -u -m HGNDRecoGNN.scripts.evaluate \
        --root "$DS/$s" --checkpoint "$CK" --out-dir "$PRED/$s" \
        --suffix smash --split all --batch-size 256 --device auto )
done

echo "== 2/3  cluster performance against the reference benchmarks =="
"$PY" "$HERE/scripts/reco_performance.py" \
    --clusters "$PRED/test/pred_clusters_smash.pkl" \
    --out "$OUT/performance_test.json" | tee "$OUT/performance_test.txt"

echo "== 3/3  S_pot working point: chosen on train+val, applied to test =="
"$PY" "$HERE/scripts/spot_threshold_scan.py" \
    --pred-dir "$PRED" --raw-dir "$RAW" \
    --out "$OUT/spot_threshold_scan.json" | tee "$OUT/spot_threshold_scan.txt"

if [ -n "${BASELINE_CK:-}" ]; then
    echo "== optional  same test data, previous checkpoint, for a like-for-like baseline =="
    ( cd "$PKG" && "$PY" -u -m HGNDRecoGNN.scripts.evaluate \
        --root "$DS/test" --checkpoint "$BASELINE_CK" --out-dir "$PRED/test_baseline" \
        --suffix smash --split all --batch-size 256 --device auto )
    "$PY" "$HERE/scripts/reco_performance.py" \
        --clusters "$PRED/test_baseline/pred_clusters_smash.pkl" \
        --out "$OUT/performance_test_baseline.json" \
        | tee "$OUT/performance_test_baseline.txt"
fi

echo "done -> $OUT"
