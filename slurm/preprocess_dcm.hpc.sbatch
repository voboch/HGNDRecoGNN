#!/bin/bash
# preprocess_dcm.hpc.sbatch — graph cache for the DCM-QGSM-SMM Xe+CsI sample.
#
# Why this job exists
# -------------------
# Every reconstruction number in this project so far comes from the SMASH
# production.  The published benchmarks (ROC AUC ~0.97, ~80 % efficiency at
# ~87 % purity, linearity and resolution under 10 % from 0.7 to 5 GeV) were
# obtained on a *different* sample: DCM-QGSM-SMM Xe+CsI at 3.26 AGeV, the
# Dombay reference.  Comparing our SMASH numbers against them conflates
# pipeline quality with the change of generator, so this builds the reference
# sample through our own loader and cache.
#
# Two caches get built, and the difference between them is the point:
#   SCALER=own    fits a DCM StandardScaler  -> for training on DCM, the
#                 faithful reproduction of the reference setup;
#   SCALER=smash  reuses the SMASH scaler    -> for applying the SMASH-trained
#                 checkpoint, where a DCM-fitted scaler would silently absorb
#                 the domain shift the test is meant to expose.
#
# The sample has hits and vacs but *no* primaries, so nothing here feeds the
# impact-parameter or acceptance analysis; DCM is a reconstruction control.
#
# Submit from a cHARISMa login node:
#     sbatch slurm/preprocess_dcm.hpc.sbatch                  # own scaler
#     SCALER=smash sbatch slurm/preprocess_dcm.hpc.sbatch     # SMASH scaler
#
#SBATCH --job-name=prep_dcm
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${ENV_NAME:=hgnd-env}"
: "${PROJECT_DIR:=$HOME}"
: "${SCRATCH:=/scratch/$USER}"
: "${NUM_WORKERS:=${SLURM_CPUS_PER_TASK:-8}}"
# The raw sample, as delivered. Read-only; not copied to scratch (376 MB).
: "${CSV_DIR:=$HOME/het_nDetHalf/data/ndet_links_Jan_26}"
: "${SCALER:=own}"
# twindow: the reference notebook uses 1.0 ns, this pipeline has used 1.5 ns.
# results/graph_audit/time_window_scan.json measures what that costs; the
# default here follows the reference so the comparison is like-for-like, and
# TWINDOW=1.5 reproduces our own setting.
: "${TWINDOW:=1.0}"
: "${RLOCAL:=3.6}"
# Source of the SMASH scaler for SCALER=smash. Any SMASH v3 cache will do:
# they were built from one pooled fit, which is what makes them comparable.
: "${SMASH_SCALER_DIR:=$SCRATCH/hgnd/cache/ndet_dataset_smash_defaultSpot_v3/processed}"

module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
export PYTHONPATH="$HOME/HGNDRecoGNN:${PYTHONPATH:-}"
mkdir -p "${SLURM_SUBMIT_DIR:-$HOME/HGNDRecoGNN}/logs"

# 1.0 -> 1p0, 1.5 -> 1p5. Bash substitution: the python one-liner this
# replaced nested same-type quotes inside an f-string, which only parses on
# 3.12+, and the job environment is older — both DCM jobs died on it in 2 s.
TW_TAG=${TWINDOW//./p}
DATASET_ROOT="$SCRATCH/hgnd/cache/ndet_dataset_dcm_xecs_${SCALER}_tw${TW_TAG}"

SCALER_ARG=()
case "$SCALER" in
    own)   ;;
    smash) SCALER_ARG=(--scaler-source "$SMASH_SCALER_DIR") ;;
    *)     echo "ERROR: SCALER must be 'own' or 'smash', got '$SCALER'" >&2; exit 2 ;;
esac

echo "host:     $(hostname)"
echo "date:     $(date)"
echo "job:      ${SLURM_JOB_ID:-local}"
echo "csv:      $CSV_DIR"
echo "root:     $DATASET_ROOT"
echo "scaler:   $SCALER ${SCALER_ARG[*]:-}"
echo "graph:    rlocal=$RLOCAL twindow=$TWINDOW"
echo "workers:  $NUM_WORKERS"
python -c "from HGNDRecoGNN.data.graph_dataset import SCHEMA_V; print('schema:', SCHEMA_V)"
echo

if [[ ! -d "$CSV_DIR" ]]; then
    echo "ERROR: CSV_DIR $CSV_DIR does not exist." >&2
    exit 3
fi
if [[ "$SCALER" == "smash" && ! -f "$SMASH_SCALER_DIR/scaler_top.pkl" ]]; then
    echo "ERROR: no scaler_top.pkl under $SMASH_SCALER_DIR" >&2
    exit 4
fi

# The loader raises on duplicate (Row, Instance) MC keys. That guard is the
# reason this job is worth watching: the reference notebook offsets Row by
# (file - 1) * 500 + run * 200000, which aliases whenever a file holds more
# than 500 events. If it aliases on this sample too, the reference's own
# labels were partly wrong and its benchmarks are not a clean target. Record
# the per-file event counts so the question is answered either way.
echo "events per file (first 12 files):"
python - "$CSV_DIR" <<'EOF'
import os, sys
d = sys.argv[1]
shown = 0
for run in sorted(os.listdir(d)):
    rd = os.path.join(d, run)
    if not os.path.isdir(rd):
        continue
    for f in sorted(os.listdir(rd)):
        if not f.endswith('hits.csv'):
            continue
        p = os.path.join(rd, f)
        if os.stat(p).st_size <= 100:
            continue
        rows = set()
        with open(p) as fh:
            next(fh, None)
            for line in fh:
                c = line.split(',', 1)[0].strip()
                if c:
                    rows.add(c)
        print(f"  {run}/{f}: {len(rows)} events"
              f"{'   <-- >500, reference stride would alias' if len(rows) > 500 else ''}")
        shown += 1
        if shown >= 12:
            sys.exit(0)
EOF
echo

python -u -m HGNDRecoGNN.scripts.preprocess \
    --csv-dir     "$CSV_DIR" \
    --root        "$DATASET_ROOT" \
    --rlocal      "$RLOCAL" \
    --twindow     "$TWINDOW" \
    --num-workers "$NUM_WORKERS" \
    "${SCALER_ARG[@]}"

echo
echo "HGND_EXIT=0"
echo "preprocess dcm ($SCALER, twindow=$TWINDOW) done: $(date)"
echo "root: $DATASET_ROOT"
