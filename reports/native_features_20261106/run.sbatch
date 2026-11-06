#!/usr/bin/env bash
#SBATCH --job-name=hgnd-native-features
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
: "${NF_SOURCE:?}" "${NF_CAMPAIGN:?}" "${NF_RECEIPTS:?}"
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$NF_SOURCE"
sha256sum -c --quiet SHA256SUMS
NF_WORK=$(mktemp -d "/tmp/hgnd-native-features-${SLURM_JOB_ID}.XXXXXX")
case "$(stat -f -c %T "$NF_WORK")" in xfs|ext2/ext3|tmpfs|btrfs|overlay|zfs) ;; *) exit 2 ;; esac
export MPLCONFIGDIR="$NF_WORK/mpl" XDG_CACHE_HOME="$NF_WORK/cache"
echo "Node-local work: $NF_WORK on $(hostname)"
"$PYTHON" -u HGNDRecoGNN/reports/native_features_20261106/analyze.py --plan HGNDRecoGNN/reports/native_features_20261106/plan.json --work "$NF_WORK/work" --output "$NF_WORK/output"
sha256sum -c --quiet SHA256SUMS
"$PYTHON" -u HGNDRecoGNN/scripts/publish_smash_directory.py --source "$NF_WORK/output" --bundle "$NF_WORK/export" --destination "$NF_CAMPAIGN/result_${SLURM_JOB_ID}" --receipt "$NF_RECEIPTS/result_${SLURM_JOB_ID}.json"
echo "Published; node-local inputs preserved at $NF_WORK"
