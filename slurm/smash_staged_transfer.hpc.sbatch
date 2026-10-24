#!/usr/bin/env bash
# First-file frozen DCM-to-SMASH inference; no fitting.
#SBATCH --job-name=smash-staged-transfer
#SBATCH --account=proj_1855
#SBATCH --partition=test
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --exclude=cn-007,cn-008,cn-034
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${SMASH_SOURCE:?Immutable source required}"
: "${SMASH_CAMPAIGN:?Audited output root required}"
: "${SMASH_INVENTORY_SHA:?Pinned engineering inventory required}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
cd "$SMASH_SOURCE"
WORK=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/hgnd-smash-gpu-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$WORK"' EXIT
mkdir -p "$WORK/audit"
cp -R "$SMASH_CAMPAIGN/smoke" "$WORK/audit/smoke"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0),flush=True)'
for TAG in zeroSpot defaultSpot bigSpot; do
    "$PYTHON" -u HGNDRecoGNN/scripts/smash_transfer_infer.py \
        --audit-campaign "$WORK/audit" --inventory "$SMASH_SOURCE/engineering_inventory.json" \
        --inventory-sha256 "$SMASH_INVENTORY_SHA" --tag "$TAG" --smoke --device cuda \
        --native-campaign /scratch/vbocharnikov/hgnd/experiments/student_pipeline_20261020_56bcc48206 \
        --prepared /scratch/vbocharnikov/hgnd/experiments/dcm_grouped_20261016_077a236ca2/prepared \
        --output "$SMASH_CAMPAIGN/transfer_smoke/$TAG"
done
