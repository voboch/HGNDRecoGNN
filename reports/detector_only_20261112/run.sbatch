#!/usr/bin/env bash
#SBATCH --job-name=hgnd-detector-only
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --exclude=cn-007,cn-008,cn-034
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${DETECTOR_SOURCE:?Pinned source}"
: "${DETECTOR_CAMPAIGN:?Scratch output}"
: "${DETECTOR_RECEIPTS:?Home receipt directory}"
cd "$DETECTOR_SOURCE"
sha256sum -c --quiet SHA256SUMS
WORK=$(mktemp -d "/tmp/hgnd-detector-only-${SLURM_JOB_ID}.XXXXXX")
echo "Node-local work $WORK on $(hostname)"
SUCCESS=0
trap 'if [[ $SUCCESS == 1 ]]; then rm -rf -- "$WORK"; else echo "Preserved $WORK on $(hostname)" >&2; fi' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$WORK/mpl"
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
SCOPE=HGNDRecoGNN/reports/detector_only_20261112
"$PYTHON" -u "$SCOPE/run.py" --plan "$SCOPE/plan.json" --work "$WORK/work" \
    --output "$WORK/output" --source-sums SHA256SUMS
sha256sum -c --quiet SHA256SUMS
KEY="result_${SLURM_JOB_ID}"
"$PYTHON" -u HGNDRecoGNN/scripts/publish_smash_directory.py \
    --source "$WORK/output" --bundle "$WORK/bundle" --destination "$DETECTOR_CAMPAIGN/$KEY" \
    --receipt "$DETECTOR_RECEIPTS/$KEY.json"
SUCCESS=1
