#!/usr/bin/env bash
# One file per array task on ncx. Cluster defaults match the previous ncx
# converter; account/partition must be passed by the submitter if required.
# Source locations come from ncx_refactor/run_convert.sh, last checked in 2026-09.
# No compute is allowed through direct invocation on a login node.
#SBATCH --job-name=bmn_proton_pid
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=out/proton_pid_%A_%a.out
#SBATCH --error=out/proton_pid_%A_%a.err

set -eo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit with sbatch; do not process DSTs on a login node" >&2; exit 2; }
: "${PROJECT_DIR:?absolute path of HGNDRecoGNN checkout on ncx}"
: "${MANIFEST:?absolute path to the frozen one-path-per-line manifest}"
: "${OUTPUT_DIR:?new output directory for this stage}"
: "${STAGE:=inspect}"
: "${BMNROOT_CONFIG:=/nica/bmn1/karpushkin/bmn/bmnrootVersions/bmnroot_dev/build/config.sh}"
: "${ROOT_PYTHON:=python3}"
: "${MAX_EVENTS:=0}"
: "${TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"
: "${USER_ENV:=}"

[[ -r "$BMNROOT_CONFIG" ]] || { echo "Missing BmnRoot config: $BMNROOT_CONFIG" >&2; exit 2; }
source "$BMNROOT_CONFIG"
if [[ -n "$USER_ENV" ]]; then source "$USER_ENV"; fi
set -u
cd "$PROJECT_DIR"
export BMNROOT_CONFIG
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/proton_pid_mpl_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
INPUT=$(sed -n "$((TASK_INDEX + 1))p" "$MANIFEST")
[[ -r "$INPUT" ]] || { echo "Missing manifest input for task $TASK_INDEX: $INPUT" >&2; exit 2; }
PREFIX="$OUTPUT_DIR/$(printf '%05d' "$TASK_INDEX")"

echo "host=$(hostname) stage=$STAGE task=$TASK_INDEX input=$INPUT"
echo "BmnRoot config=$BMNROOT_CONFIG VMCWORKDIR=${VMCWORKDIR:-unset} python=$ROOT_PYTHON"
"$ROOT_PYTHON" -c 'import ROOT,sys; print("Python",sys.version); print("ROOT",ROOT.gROOT.GetVersion())'
if [[ -d "${VMCWORKDIR:-}/.git" ]]; then git -C "$VMCWORKDIR" rev-parse HEAD; fi

ROOT_ARGS=()
if [[ -n "${ROOT_INIT_MACRO:-}" ]]; then ROOT_ARGS+=(--root-init "$ROOT_INIT_MACRO"); fi
case "$STAGE" in
    inspect)
        # Bounded first-file inspection is intended for a short allocation.
        LIMIT=${MAX_EVENTS:-0}
        [[ "$LIMIT" -gt 0 ]] || LIMIT=100
        "$ROOT_PYTHON" scripts/run_proton_pid.py inspect --input "$INPUT" --require-mc \
            --max-events "$LIMIT" "${ROOT_ARGS[@]}"
        ;;
    extract)
        "$ROOT_PYTHON" scripts/run_proton_pid.py extract --input "$INPUT" --require-mc \
            --max-events "$MAX_EVENTS" --output-prefix "$PREFIX" "${ROOT_ARGS[@]}"
        ;;
    reco)
        # Only for a manifest of transported/digitised input files. Existing
        # DST productions start at inspect; they are not reconstructed again.
        "$ROOT_PYTHON" scripts/reconstruct_smash_bmn.py --input "$INPUT" \
            --output "${PREFIX}_dst.root" --max-events "$MAX_EVENTS"
        ;;
    *) echo "Unsupported STAGE=$STAGE (inspect, extract, reco)" >&2; exit 2 ;;
esac
echo "BMN_PROTON_STAGE_COMPLETED"
