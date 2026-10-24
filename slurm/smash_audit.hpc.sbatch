#!/usr/bin/env bash
# CPU data audit only; no model training or inference.
#SBATCH --job-name=smash-source-audit
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${SMASH_AUDIT_SOURCE:?Immutable source required}"
: "${SMASH_AUDIT_CAMPAIGN:?New scratch campaign required}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
cd "$SMASH_AUDIT_SOURCE"
TAGS=(zeroSpot defaultSpot bigSpot)
COMMON=(--raw-root /scratch/vbocharnikov/hgnd/data_fixed
        --primary-root /scratch/vbocharnikov/hgnd/data_np_full_20260924/extracted)
if [[ "${SMASH_AUDIT_PHASE:-full}" == smoke ]]; then
    for TAG in "${TAGS[@]}"; do
        "$PYTHON" -u HGNDRecoGNN/scripts/audit_smash_sources.py "${COMMON[@]}" \
            --tag "$TAG" --limit-files 1 --output "$SMASH_AUDIT_CAMPAIGN/smoke/$TAG"
    done
else
    TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
    "$PYTHON" -u HGNDRecoGNN/scripts/audit_smash_sources.py "${COMMON[@]}" \
        --tag "$TAG" --output "$SMASH_AUDIT_CAMPAIGN/full/$TAG"
fi
