#!/usr/bin/env bash
#SBATCH --job-name=hgnd-first-hit-tof
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
: "${PA_SOURCE:?}" "${PA_CAMPAIGN:?}" "${PA_RECEIPTS:?}"
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PA_SOURCE"
sha256sum -c --quiet SHA256SUMS
(cd /home/vbocharnikov/hgnd_experiments/energy_label_20261102_cc7a4fb5e5 && sha256sum -c --quiet SHA256SUMS)
PA_WORK=$(mktemp -d "/tmp/hgnd-first-hit-tof-${SLURM_JOB_ID}.XXXXXX")
case "$(stat -f -c %T "$PA_WORK")" in xfs|ext2/ext3|tmpfs|btrfs|overlay|zfs) ;; *) exit 2 ;; esac
export MPLCONFIGDIR="$PA_WORK/mpl" XDG_CACHE_HOME="$PA_WORK/cache"
echo "Node-local work: $PA_WORK on $(hostname)"
"$PYTHON" -u HGNDRecoGNN/reports/first_hit_tof_20261110/extract.py --plan HGNDRecoGNN/reports/first_hit_tof_20261110/plan.json --selection HGNDRecoGNN/reports/first_hit_tof_20261110/selection.npz --work "$PA_WORK/work" --output "$PA_WORK/output"
sha256sum -c --quiet SHA256SUMS
(cd /home/vbocharnikov/hgnd_experiments/energy_label_20261102_cc7a4fb5e5 && sha256sum -c --quiet SHA256SUMS)
"$PYTHON" -u HGNDRecoGNN/scripts/publish_smash_directory.py --source "$PA_WORK/output" --bundle "$PA_WORK/export" --destination "$PA_CAMPAIGN/result_${SLURM_JOB_ID}" --receipt "$PA_RECEIPTS/result_${SLURM_JOB_ID}.json"
echo "Published; node-local inputs preserved at $PA_WORK"
