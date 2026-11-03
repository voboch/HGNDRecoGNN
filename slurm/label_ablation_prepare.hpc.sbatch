#!/usr/bin/env bash
# One CPU preparation, all registered train/validation roles; no model calls.
#SBATCH --job-name=hgnd-label-patches
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --exclude=cn-007,cn-008,cn-034
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${AB_SOURCE:?New immutable source, never the Stage A directory}"
: "${AB_CAMPAIGN:?New campaign directory}"
: "${AB_RECEIPTS:?Home receipt directory}"
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$AB_SOURCE"
sha256sum -c --quiet SHA256SUMS
AB_WORK=$(mktemp -d "/tmp/hgnd-label-patches-${SLURM_JOB_ID}.XXXXXX")
case "$(stat -f -c %T "$AB_WORK")" in
    xfs|ext2/ext3|tmpfs|btrfs|overlay|zfs) ;;
    *) echo "Expected node-local work filesystem" >&2; exit 2 ;;
esac
export MPLCONFIGDIR="$AB_WORK/mpl" XDG_CACHE_HOME="$AB_WORK/cache"
echo "Node-local inputs and outputs: $AB_WORK on $(hostname)"
"$PYTHON" -u HGNDRecoGNN/scripts/label_ablation_prepare.py \
    --plan HGNDRecoGNN/reports/label_ablation_20261103/launch_plan.json \
    --source-sums SHA256SUMS --work "$AB_WORK/work" --output "$AB_WORK/output" --workers 2
sha256sum -c --quiet SHA256SUMS
"$PYTHON" -u HGNDRecoGNN/scripts/publish_smash_directory.py \
    --source "$AB_WORK/output" --bundle "$AB_WORK/export" \
    --destination "$AB_CAMPAIGN/prepare_${SLURM_JOB_ID}" \
    --receipt "$AB_RECEIPTS/prepare_${SLURM_JOB_ID}.json"
echo "Preparation exported; node-local work preserved at $AB_WORK"
