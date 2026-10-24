#!/usr/bin/env bash
# CPU audit of the frozen corrected/primary development intersection.
#SBATCH --job-name=smash-transfer-audit
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --exclude=cn-007,cn-008,cn-034
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${SMASH_TRANSFER_SOURCE:?Immutable source required}"
: "${SMASH_TRANSFER_CAMPAIGN:?New campaign required}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
cd "$SMASH_TRANSFER_SOURCE"
echo "Node $(hostname), phase ${SMASH_TRANSFER_PHASE:-audit}"
INVENTORY="$SMASH_TRANSFER_SOURCE/input_inventory.json"
"$PYTHON" -u HGNDRecoGNN/scripts/smash_access_probe.py --inventory "$INVENTORY" \
    --output "$SMASH_TRANSFER_CAMPAIGN/access/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-smoke}"
TAGS=(zeroSpot defaultSpot bigSpot)
COMMON=(--raw-root /scratch/vbocharnikov/hgnd/data_fixed
        --primary-root /scratch/vbocharnikov/hgnd/data_np_full_20260924/extracted
        --inventory "$INVENTORY" --save-observations)
if [[ "${SMASH_TRANSFER_PHASE:-audit}" == smoke ]]; then
    for TAG in "${TAGS[@]}"; do
        "$PYTHON" -u HGNDRecoGNN/scripts/audit_smash_sources.py "${COMMON[@]}" \
            --tag "$TAG" --limit-files 1 --output "$SMASH_TRANSFER_CAMPAIGN/smoke/$TAG"
    done
else
    TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
    "$PYTHON" -u HGNDRecoGNN/scripts/audit_smash_sources.py "${COMMON[@]}" \
        --tag "$TAG" --output "$SMASH_TRANSFER_CAMPAIGN/audit/$TAG"
fi
