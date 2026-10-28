#!/usr/bin/env bash
# SMASH training packs (CPU): verified raw exports, audit observation recipe,
# fold scalers on training files only, A and B packs, one verified export.
# No model is built, trained or scored.
#SBATCH --job-name=smash-train-prepare
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --exclude=cn-007,cn-008,cn-034
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${SMASH_SOURCE:?Immutable source package with SHA256SUMS}"
: "${SMASH_CAMPAIGN:?Scratch campaign root}"
: "${SMASH_RECEIPT_DIR:?Home receipt directory}"
: "${SMASH_SPLIT:?Split manifest inside the source package}"
: "${SMASH_AUDIT_INDEX:?Pinned audit bundle index}"
DCM_PREPARED=${SMASH_DCM_PREPARED:-/scratch/vbocharnikov/hgnd/experiments/dcm_grouped_20261016_077a236ca2/prepared}
NATIVE=${SMASH_NATIVE:-/scratch/vbocharnikov/hgnd/experiments/student_pipeline_20261020_56bcc48206}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
cd "$SMASH_SOURCE"
sha256sum -c --quiet SHA256SUMS
WORK=$(mktemp -d "/tmp/hgnd-smash-prepare-${SLURM_JOB_ID}.XXXXXX")
LOCAL_FS=$(stat -f -c %T "$WORK")
case "$LOCAL_FS" in
    xfs|ext2/ext3|tmpfs|btrfs|overlay|zfs) ;;
    *) echo "Expected local work filesystem, got $LOCAL_FS at $WORK" >&2; exit 2 ;;
esac
echo "Node-local work: $WORK ($LOCAL_FS) on $(hostname)"
SUCCESS=0
cleanup() {
    if [[ "$SUCCESS" == 1 ]]; then rm -rf -- "$WORK";
    else rm -rf -- "$WORK/work"; echo "Preserved local outputs: $WORK/output" >&2; fi
}
trap cleanup EXIT
"$PYTHON" -u HGNDRecoGNN/scripts/smash_train_prepare.py --split "$SMASH_SPLIT" \
    --output "$WORK/output/prepare" --work "$WORK/work" \
    --dcm-prepared "$DCM_PREPARED" --native-campaign "$NATIVE" \
    --audit-index "$SMASH_AUDIT_INDEX" --source-sums "$SMASH_SOURCE/SHA256SUMS" \
    --workers "$SLURM_CPUS_PER_TASK"
sha256sum -c --quiet SHA256SUMS  # the package did not change during the run
"$PYTHON" -u HGNDRecoGNN/scripts/publish_smash_directory.py \
    --source "$WORK/output" --bundle "$WORK/export_bundle" \
    --destination "$SMASH_CAMPAIGN/prepare_bundles/prepare_${SLURM_JOB_ID}" \
    --receipt "$SMASH_RECEIPT_DIR/prepare_${SLURM_JOB_ID}.json"
SUCCESS=1
