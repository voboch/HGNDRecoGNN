#!/bin/bash
# reduce_prim.hpc.sbatch — reduce the primary-nucleon exports on cHARISMa.
#
# The primaries are already here, extracted: 568 job files under
# scratch/hgnd/data_np_full_20260924/extracted.  Streaming them over the sshfs
# mount from the laptop meant moving ~25 GB at rates between 0.2 and 4 MB/s;
# reading them from Lustre takes minutes.
#
# The reduction now emits the per-particle front-face acceptance counters
# (acc_*) alongside the legacy polar-angle band ones (band_*), which is what
# the acceptance-corrected S_pot numbers need.
#
#     sbatch slurm/reduce_prim.hpc.sbatch
#
#SBATCH --job-name=reduce_prim
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d          # CPU-only nodes
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${ENV_NAME:=hgnd-env}"
: "${PROJECT_DIR:=$HOME}"
: "${SCRATCH:=/scratch/$USER}"
: "${DATA_ROOT:=$SCRATCH/hgnd/data_np_full_20260924/extracted}"
: "${OUTDIR:=$SCRATCH/hgnd/results/prim_events_acc}"

module purge 2>/dev/null || true
module load python/miniconda
# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
export PYTHONPATH="$HOME/HGNDRecoGNN:${PYTHONPATH:-}"
export DATA_ROOT
mkdir -p "${SLURM_SUBMIT_DIR:-$HOME/HGNDRecoGNN}/logs" "$OUTDIR"

echo "host:      $(hostname)"
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTDIR:    $OUTDIR"
echo "prim files present: $(find "$DATA_ROOT" -name '*_prim.csv' | wc -l)"
echo
echo "== acceptance geometry used =="
python "$HOME/HGNDRecoGNN/scripts/hgnd_acceptance.py"
echo

python -u "$HOME/HGNDRecoGNN/scripts/reduce_prim_full.py" "$OUTDIR" \
    zeroSpot bigSpot defaultSpot

echo
echo "== outputs =="
ls -la "$OUTDIR"
echo "copy *_events.pkl and *_filehist.npz back; ~40 MB per sample"
