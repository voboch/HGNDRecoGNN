#!/usr/bin/env bash
# Full-statistics impact-parameter analysis: reduce -> reweight -> null test -> figures.
#
# The reduction step streams ~25 GB of compressed primary-nucleon CSV from the
# mounted ncx volume and is the only slow stage; it resumes from its progress
# file, so re-running after an interrupted stream is safe.
set -euo pipefail

EVENTS_DIR="${1:?usage: run_full_b_analysis.sh <events_dir> <results_dir> [n_boot]}"
RESULTS_DIR="${2:?usage: run_full_b_analysis.sh <events_dir> <results_dir> [n_boot]}"
NBOOT="${3:-400}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$EVENTS_DIR" "$RESULTS_DIR"

echo "== 1/4  reduce the production to per-event tables =="
python3 "$HERE/scripts/reduce_prim_full.py" "$EVENTS_DIR" zeroSpot bigSpot defaultSpot

echo "== 2/4  b reweighting and centrality classes =="
python3 "$HERE/scripts/b_reweight.py" "$EVENTS_DIR" "$RESULTS_DIR" "$NBOOT" \
    | tee "$RESULTS_DIR/b_reweight.txt"

echo "== 3/4  null test: same-sample splits calibrate the quoted sigmas =="
for s in zeroSpot defaultSpot bigSpot; do
    python3 "$HERE/scripts/b_reweight.py" null "$EVENTS_DIR" "$RESULTS_DIR" "$s" 40 200 \
        | tee "$RESULTS_DIR/null_test_$s.txt"
done

echo "== 4/4  figures =="
python3 "$HERE/scripts/plot_b_full.py" \
    --events-dir "$EVENTS_DIR" \
    --report "$RESULTS_DIR/b_reweight.json" \
    --null "$RESULTS_DIR/null_test_zeroSpot.json" \
    --output-dir "$RESULTS_DIR" \
    --scope "full production"

echo "done -> $RESULTS_DIR"
