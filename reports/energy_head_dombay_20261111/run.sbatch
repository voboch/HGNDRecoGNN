#!/usr/bin/env bash
#SBATCH --job-name=hgnd-en-head
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=01:15:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
: "${ENERGY_SOURCE:?Pinned source}"
: "${ENERGY_CAMPAIGN:?Scratch output}"
: "${ENERGY_RECEIPTS:?Home receipt directory}"
: "${SLURM_ARRAY_TASK_ID:?Fold/seed task}"
cd "$ENERGY_SOURCE"
sha256sum -c --quiet SHA256SUMS
WORK=$(mktemp -d "/tmp/hgnd-en-head-${SLURM_JOB_ID}.XXXXXX")
echo "Node-local work $WORK on $(hostname)"
SUCCESS=0
trap 'if [[ $SUCCESS == 1 ]]; then rm -rf -- "$WORK"; else echo "Preserved $WORK on $(hostname)" >&2; fi' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$WORK/mpl"
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
SCOPE=HGNDRecoGNN/reports/energy_head_dombay_20261111
"$PYTHON" -u "$SCOPE/run.py" --plan "$SCOPE/plan.json" --split split_manifest.json \
    --task "$SLURM_ARRAY_TASK_ID" --work "$WORK/work" --output "$WORK/output" --source-sums SHA256SUMS
sha256sum -c --quiet SHA256SUMS
KEY="task_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"
"$PYTHON" -u HGNDRecoGNN/scripts/publish_smash_directory.py \
    --source "$WORK/output" --bundle "$WORK/bundle" --destination "$ENERGY_CAMPAIGN/$KEY" \
    --receipt "$ENERGY_RECEIPTS/$KEY.json"
SUCCESS=1
