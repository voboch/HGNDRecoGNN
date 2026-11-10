#!/usr/bin/env bash
#SBATCH --job-name=hgnd-rule-tof-odd
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --exclude=cn-007,cn-008,cn-034
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${RULE_SOURCE:?}" "${RULE_CAMPAIGN:?}" "${RULE_RECEIPTS:?}"
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$RULE_SOURCE"
sha256sum -c --quiet SHA256SUMS
OLD=/home/vbocharnikov/hgnd_experiments/energy_label_20261102_cc7a4fb5e5
export PYTHONPATH="$RULE_SOURCE"
(cd "$OLD" && sha256sum -c --quiet SHA256SUMS)
WORK=$(mktemp -d "/tmp/hgnd-rule-tof-${SLURM_JOB_ID}.XXXXXX")
case "$(stat -f -c %T "$WORK")" in xfs|ext2/ext3|tmpfs|btrfs|overlay|zfs) ;; *) exit 2 ;; esac
export MPLCONFIGDIR="$WORK/mpl" XDG_CACHE_HOME="$WORK/cache"
echo "Node-local work: $WORK on $(hostname)"
"$PYTHON" -u extract_odd.py --plan plan.json --work "$WORK/work" --output "$WORK/output"
sha256sum -c --quiet SHA256SUMS
(cd "$OLD" && sha256sum -c --quiet SHA256SUMS)
"$PYTHON" -u HGNDRecoGNN/scripts/publish_smash_directory.py --source "$WORK/output" --bundle "$WORK/export" --destination "$RULE_CAMPAIGN/result_${SLURM_JOB_ID}" --receipt "$RULE_RECEIPTS/result_${SLURM_JOB_ID}.json"
