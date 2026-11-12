#!/usr/bin/env bash
#SBATCH --job-name=hgnd-arch-probe
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --array=0,9%2
#SBATCH --no-requeue
#SBATCH --exclude=cn-007,cn-008,cn-034
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
: "${ARCH_SOURCE:?Immutable source}"
: "${ARCH_CAMPAIGN:?Campaign scratch directory}"
: "${ARCH_RECEIPTS:?Home receipt directory}"
: "${ARCH_PHASE:?probe, fit or score}"
: "${SLURM_ARRAY_TASK_ID:?Registered architecture task}"

cd "$ARCH_SOURCE"
sha256sum -c --quiet SHA256SUMS
WORK=$(mktemp -d "/tmp/hgnd-architecture-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}.XXXXXX")
case "$(stat -f -c %T "$WORK")" in
    xfs|ext2/ext3|tmpfs|btrfs|overlay|zfs) ;;
    *) echo "Expected node-local work directory: $WORK" >&2; exit 2 ;;
esac
echo "Node-local work $WORK on $(hostname)"
SUCCESS=0
trap 'if [[ $SUCCESS == 1 ]]; then rm -rf -- "$WORK"; else echo "Preserved $WORK on $(hostname)" >&2; fi' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="$WORK/mpl" XDG_CACHE_HOME="$WORK/cache"
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
SCOPE=HGNDRecoGNN/reports/smash_architecture_20261112
TASK=$SLURM_ARRAY_TASK_ID
OPTIONS=()
case "$ARCH_PHASE" in
    probe)
        [[ "$TASK" == 0 || "$TASK" == 9 ]]
        KEY="probe_${TASK}_${SLURM_JOB_ID}"
        ;;
    fit)
        : "${ARCH_START_EPOCH:?Start boundary, including zero}"
        : "${ARCH_END_EPOCH:?End boundary}"
        OPTIONS+=(--start-epoch "$ARCH_START_EPOCH" --end-epoch "$ARCH_END_EPOCH"
                  --mirror "$ARCH_CAMPAIGN/mirror/task_$TASK")
        KEY="fit_${TASK}_${ARCH_START_EPOCH}_${ARCH_END_EPOCH}_${SLURM_JOB_ID}"
        ;;
    score)
        : "${ARCH_TEST_SCOPE:?even or odd}"
        : "${ARCH_FIT_INDEX:?Reviewed fit receipt index}"
        : "${ARCH_FIT_INDEX_SHA256:?Reviewed index hash}"
        printf '%s  %s\n' "$ARCH_FIT_INDEX_SHA256" "$ARCH_FIT_INDEX" | sha256sum -c --quiet
        FIT=$(jq -er --arg task "$TASK" '.tasks[$task].receipt' "$ARCH_FIT_INDEX")
        FIT_SHA=$(jq -er --arg task "$TASK" '.tasks[$task].sha256' "$ARCH_FIT_INDEX")
        OPTIONS+=(--test-scope "$ARCH_TEST_SCOPE" --fit-receipt "$FIT" --fit-receipt-sha256 "$FIT_SHA")
        if [[ "$ARCH_TEST_SCOPE" == odd ]]; then
            : "${ARCH_ODD_RELEASE:?Frozen policy receipt}"
            : "${ARCH_ODD_RELEASE_SHA256:?Frozen policy receipt hash}"
            OPTIONS+=(--odd-release "$ARCH_ODD_RELEASE" --odd-release-sha256 "$ARCH_ODD_RELEASE_SHA256")
        fi
        KEY="score_${ARCH_TEST_SCOPE}_${TASK}_${SLURM_JOB_ID}"
        ;;
    *)
        echo "Unsupported phase: $ARCH_PHASE" >&2
        exit 2
        ;;
esac
"$PYTHON" -u HGNDRecoGNN/scripts/smash_architecture.py \
    --phase "$ARCH_PHASE" --task "$TASK" --plan "$SCOPE/plan.json" \
    --split split_manifest.json --source-sums SHA256SUMS \
    --work "$WORK/work" --output "$WORK/output" "${OPTIONS[@]}"
sha256sum -c --quiet SHA256SUMS
"$PYTHON" -u HGNDRecoGNN/scripts/publish_smash_directory.py \
    --source "$WORK/output" --bundle "$WORK/bundle" \
    --destination "$ARCH_CAMPAIGN/$KEY" --receipt "$ARCH_RECEIPTS/$KEY.json"
SUCCESS=1
