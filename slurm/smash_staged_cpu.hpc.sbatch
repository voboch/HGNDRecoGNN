#!/usr/bin/env bash
# Read unchanged SMASH CSVs from a verified node-local copy.
#SBATCH --job-name=smash-staged-cpu
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
: "${SMASH_SOURCE:?Immutable source required}"
: "${SMASH_CAMPAIGN:?New output root required}"
: "${SMASH_BUNDLE:?Input archive required}"
: "${SMASH_BUNDLE_MANIFEST_SHA:?Pinned manifest required}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
cd "$SMASH_SOURCE"
WORK=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/hgnd-smash-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$WORK"' EXIT
"$PYTHON" -c 'import hashlib,sys; assert hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()==sys.argv[2]' \
    "$SMASH_BUNDLE/manifest.json" "$SMASH_BUNDLE_MANIFEST_SHA"
"$PYTHON" -u HGNDRecoGNN/scripts/smash_input_bundle.py unpack \
    --archive "$SMASH_BUNDLE/inputs.tar" --manifest "$SMASH_BUNDLE/manifest.json" --output "$WORK/inputs"
mkdir -p "$SMASH_CAMPAIGN/access"
cp "$WORK/inputs/bundle_verified.json" "$SMASH_CAMPAIGN/access/${SLURM_JOB_ID}.json"
if [[ "${SMASH_PHASE:?map or audit required}" == map ]]; then
    "$PYTHON" -u HGNDRecoGNN/scripts/map_smash_primary.py \
        --inventory "$SMASH_SOURCE/original_inventory.json" --staged-root "$WORK/inputs" \
        --output "$SMASH_CAMPAIGN/source_mapping" --tag defaultSpot
elif [[ "$SMASH_PHASE" == audit ]]; then
    for TAG in zeroSpot defaultSpot bigSpot; do
        "$PYTHON" -u HGNDRecoGNN/scripts/audit_smash_sources.py \
            --raw-root /scratch/vbocharnikov/hgnd/data_fixed \
            --primary-root /scratch/vbocharnikov/hgnd/data_np_full_20260924/extracted \
            --inventory "$SMASH_SOURCE/engineering_inventory.json" --staged-root "$WORK/inputs" \
            --save-observations --limit-files 1 --tag "$TAG" --output "$SMASH_CAMPAIGN/smoke/$TAG"
    done
else
    echo "Unknown phase: $SMASH_PHASE" >&2
    exit 2
fi
