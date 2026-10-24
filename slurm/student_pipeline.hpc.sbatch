#!/usr/bin/env bash
# Native HGND inputs; finite four-model x three-seed campaign.
#SBATCH --job-name=student-native
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --exclude=cn-007,cn-008
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
: "${NATIVE_SOURCE:?Immutable source root required}"
: "${NATIVE_CAMPAIGN:?New scratch campaign required}"
: "${NATIVE_ENV:?Dedicated environment required}"
: "${NATIVE_PREPARED:?Approved frozen split required}"
: "${NATIVE_FROZEN:?Approved cache inventory required}"
: "${NATIVE_CONTENT_CENSUS:?Verified exact-content census required}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/native-mpl-${SLURM_JOB_ID}"
TASK=${SLURM_ARRAY_TASK_ID:-0}
SEEDS=(42 123 456)
export PYTHONHASHSEED=${SEEDS[$((TASK / 4))]}
cd "$NATIVE_SOURCE"
COMMON=(--source-manifest "$NATIVE_SOURCE/source_manifest.json"
        --campaign "$NATIVE_CAMPAIGN" --prepared "$NATIVE_PREPARED"
        --frozen-data "$NATIVE_FROZEN" --content-census "$NATIVE_CONTENT_CENSUS")
case "${NATIVE_PHASE:-train}" in
  prepare_smoke)
    "$NATIVE_ENV/bin/python" -u HGNDRecoGNN/scripts/student_pipeline.py prepare "${COMMON[@]}"
    "$NATIVE_ENV/bin/python" -u HGNDRecoGNN/scripts/student_pipeline.py smoke "${COMMON[@]}"
    ;;
  train)
    "$NATIVE_ENV/bin/python" -u HGNDRecoGNN/scripts/student_pipeline.py train "${COMMON[@]}" --task "$TASK"
    ;;
  test)
    "$NATIVE_ENV/bin/python" -u HGNDRecoGNN/scripts/student_pipeline.py test "${COMMON[@]}"
    ;;
  *) echo "Unknown NATIVE_PHASE" >&2; exit 2 ;;
esac
