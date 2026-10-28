#!/usr/bin/env bash
# SMASH-trained GraphSAGE: GPU smoke, one training task (array t -> fold t/3, seed t%3)
# or one fold's evaluation. Node-local work and checkpoints; per-epoch mirror to
# scratch is non-fatal; one verified export at the end. No automatic requeue.
#SBATCH --job-name=smash-train
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --exclude=cn-007,cn-008,cn-034
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
: "${SMASH_SOURCE:?Immutable source package with SHA256SUMS}"
: "${SMASH_PHASE:?smoke, train or evaluate}"
: "${SMASH_CAMPAIGN:?Scratch campaign root}"
: "${SMASH_RECEIPT_DIR:?Home receipt directory}"
: "${SMASH_SPLIT:?Split manifest inside the source package}"
DCM_PREPARED=${SMASH_DCM_PREPARED:-/scratch/vbocharnikov/hgnd/experiments/dcm_grouped_20261016_077a236ca2/prepared}
NATIVE=${SMASH_NATIVE:-/scratch/vbocharnikov/hgnd/experiments/student_pipeline_20261020_56bcc48206}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
JOB_KEY="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-single}"
cd "$SMASH_SOURCE"
sha256sum -c --quiet SHA256SUMS
WORK=$(mktemp -d "/tmp/hgnd-smash-train-${JOB_KEY}.XXXXXX")
LOCAL_FS=$(stat -f -c %T "$WORK")
case "$LOCAL_FS" in
    xfs|ext2/ext3|tmpfs|btrfs|overlay|zfs) ;;
    *) echo "Expected local work filesystem, got $LOCAL_FS at $WORK" >&2; exit 2 ;;
esac
echo "Node-local work: $WORK ($LOCAL_FS) on $(hostname)"
export MPLCONFIGDIR="$WORK/mpl"
SUCCESS=0
cleanup() {
    if [[ "$SUCCESS" == 1 ]]; then rm -rf -- "$WORK";
    else rm -rf -- "$WORK/work" "$WORK/prepare_work"; echo "Preserved local outputs: $WORK/output" >&2; fi
}
trap cleanup EXIT
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), flush=True)'
COMMON=(--source-sums "$SMASH_SOURCE/SHA256SUMS" --split "$SMASH_SPLIT" --output "$WORK/output"
        --work "$WORK/work" --native-campaign "$NATIVE" --device cuda)
case "$SMASH_PHASE" in
  smoke)
    : "${SMASH_AUDIT_INDEX:?Pinned audit bundle index}"
    : "${SMASH_TRANSFER_REFERENCE:?Pinned stored transfer CSVs}"
    "$PYTHON" -u HGNDRecoGNN/scripts/smash_train_prepare.py --split "$SMASH_SPLIT" \
        --output "$WORK/output/prepare" --work "$WORK/prepare_work" \
        --dcm-prepared "$DCM_PREPARED" --native-campaign "$NATIVE" \
        --audit-index "$SMASH_AUDIT_INDEX" --source-sums "$SMASH_SOURCE/SHA256SUMS" \
        --keep-observations --workers 4
    "$PYTHON" -u HGNDRecoGNN/scripts/smash_train.py smoke "${COMMON[@]}" \
        --prepared "$WORK/output/prepare" --transfer-reference "$SMASH_TRANSFER_REFERENCE" \
        --mirror "$SMASH_CAMPAIGN/mirror/smoke_${JOB_KEY}"
    ;;
  train)
    : "${SLURM_ARRAY_TASK_ID:?Training runs as an array task}"
    : "${SMASH_PREPARE_BUNDLE:?Published prepare bundle}"
    : "${SMASH_PREPARE_MANIFEST_SHA:?Pinned prepare manifest}"
    : "${SMASH_SMOKE_SUMMARY:?Verified passed GPU smoke summary}"
    RESUME=()
    if [[ -n "${SMASH_RESUME_MIRROR:-}" ]]; then RESUME=(--resume-mirror "$SMASH_RESUME_MIRROR"); fi
    "$PYTHON" -u HGNDRecoGNN/scripts/smash_train.py train "${COMMON[@]}" --task "$SLURM_ARRAY_TASK_ID" \
        --prepare-bundle "$SMASH_PREPARE_BUNDLE" --prepare-manifest-sha256 "$SMASH_PREPARE_MANIFEST_SHA" \
        --smoke-summary "$SMASH_SMOKE_SUMMARY" \
        --mirror "$SMASH_CAMPAIGN/mirror/task_${SLURM_ARRAY_TASK_ID}_${JOB_KEY}" "${RESUME[@]}"
    ;;
  evaluate)
    : "${SMASH_FOLD:?Fold to evaluate}"
    : "${SMASH_TRAIN_INDEX:?Publication receipts of the training tasks of this fold}"
    : "${SMASH_PREPARE_BUNDLE:?Published prepare bundle}"
    : "${SMASH_PREPARE_MANIFEST_SHA:?Pinned prepare manifest}"
    "$PYTHON" -u HGNDRecoGNN/scripts/smash_train.py evaluate "${COMMON[@]}" --fold "$SMASH_FOLD" \
        --train-index "$SMASH_TRAIN_INDEX" --campaign "$SMASH_CAMPAIGN" \
        --prepare-bundle "$SMASH_PREPARE_BUNDLE" --prepare-manifest-sha256 "$SMASH_PREPARE_MANIFEST_SHA"
    ;;
  *) echo "Unknown SMASH_PHASE" >&2; exit 2 ;;
esac
sha256sum -c --quiet SHA256SUMS  # the package did not change during the run
"$PYTHON" -u HGNDRecoGNN/scripts/publish_smash_directory.py \
    --source "$WORK/output" --bundle "$WORK/export_bundle" \
    --destination "$SMASH_CAMPAIGN/${SMASH_PHASE}_bundles/${SMASH_PHASE}_${JOB_KEY}" \
    --receipt "$SMASH_RECEIPT_DIR/${SMASH_PHASE}_${JOB_KEY}.json"
SUCCESS=1
