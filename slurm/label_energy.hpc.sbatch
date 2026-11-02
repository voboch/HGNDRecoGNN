#!/usr/bin/env bash
# S-only label census: read-only accepted graphs, exact re-observation, compact output.
#SBATCH --job-name=hgnd-label-audit
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --exclude=cn-007,cn-008,cn-034
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${LABEL_SOURCE:?Immutable source directory}"
: "${LABEL_PHASE:?smoke or full}"
: "${LABEL_CAMPAIGN:?New scratch campaign root}"
: "${LABEL_RECEIPTS:?Home receipt directory}"
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$LABEL_SOURCE"
sha256sum -c --quiet SHA256SUMS
LABEL_WORK=$(mktemp -d "/tmp/hgnd-label-${SLURM_JOB_ID}.XXXXXX")
export MPLCONFIGDIR="$LABEL_WORK/mpl" XDG_CACHE_HOME="$LABEL_WORK/cache"
echo "Node-local output: $LABEL_WORK on $(hostname)"
EXTRA=()
if [[ "$LABEL_PHASE" == full ]]; then
    : "${LABEL_SMOKE:?Passed same-source smoke summary}"
    EXTRA=(--smoke "$LABEL_SMOKE")
fi
"$PYTHON" -u HGNDRecoGNN/scripts/label_energy_campaign.py \
    --plan HGNDRecoGNN/reports/energy_label_autonomous_20261102/launch_plan.json \
    --source-sums SHA256SUMS --phase "$LABEL_PHASE" --workers 2 \
    --work "$LABEL_WORK/work" --output "$LABEL_WORK/output" "${EXTRA[@]}"
sha256sum -c --quiet SHA256SUMS
"$PYTHON" -u HGNDRecoGNN/scripts/publish_smash_directory.py \
    --source "$LABEL_WORK/output" --bundle "$LABEL_WORK/export" \
    --destination "$LABEL_CAMPAIGN/${LABEL_PHASE}_${SLURM_JOB_ID}" \
    --receipt "$LABEL_RECEIPTS/${LABEL_PHASE}_${SLURM_JOB_ID}.json"
echo "Complete output exported; node-local inputs remain at $LABEL_WORK"
