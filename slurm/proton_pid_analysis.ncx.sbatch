#!/usr/bin/env bash
# Fit one MC-development calibration or select one held-out table per task.
# Submission uses the ncx defaults; pass account/partition if the site requires.
#SBATCH --job-name=bmn_proton_analysis
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=out/proton_analysis_%A_%a.out
#SBATCH --error=out/proton_analysis_%A_%a.err

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Use a Slurm allocation" >&2; exit 2; }
: "${PROJECT_DIR:?absolute path of HGNDRecoGNN checkout}"
: "${TABLE_MANIFEST:?absolute path of list of exported track tables}"
: "${OUTPUT_DIR:?new output directory}"
: "${STAGE:?calibrate or select}"
: "${ANALYSIS_PYTHON:=python3}"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/proton_pid_mpl_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
"$ANALYSIS_PYTHON" -c 'import numpy,pandas,scipy,matplotlib; print("analysis imports OK")'
case "$STAGE" in
    calibrate)
        : "${TOF:?400 or 700}"
        : "${P_EDGES:?explicit comma-separated momentum bin edges}"
        : "${MIN_SLICE_TRACKS:?explicit per-species template statistics threshold}"
        : "${CALIBRATION_DESCRIPTION:?development file scope and production description}"
        mapfile -t TABLES < "$TABLE_MANIFEST"
        [[ "${#TABLES[@]}" -gt 0 ]] || exit 2
        "$ANALYSIS_PYTHON" scripts/run_proton_pid.py calibrate --tracks "${TABLES[@]}" \
            --tof "$TOF" --p-edges "$P_EDGES" --min-slice-tracks "$MIN_SLICE_TRACKS" \
            --sample-description "$CALIBRATION_DESCRIPTION" --output "$OUTPUT_DIR/calibration.json"
        ;;
    select)
        : "${CALIBRATION:?frozen calibration.json path}"
        TASK_INDEX=${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}
        TRACKS=$(sed -n "$((TASK_INDEX+1))p" "$TABLE_MANIFEST")
        [[ -r "$TRACKS" && "$TRACKS" == *_tracks.csv.gz ]] || { echo "Bad table $TRACKS" >&2; exit 2; }
        EVENTS="${TRACKS%_tracks.csv.gz}_events.csv.gz"
        [[ -r "$EVENTS" ]] || { echo "Missing event denominator ledger $EVENTS" >&2; exit 2; }
        "$ANALYSIS_PYTHON" scripts/run_proton_pid.py select --tracks "$TRACKS" --events "$EVENTS" \
            --calibration "$CALIBRATION" --output-dir "$OUTPUT_DIR/$(printf '%05d' "$TASK_INDEX")"
        ;;
    *) echo "Unsupported STAGE=$STAGE" >&2; exit 2 ;;
esac
