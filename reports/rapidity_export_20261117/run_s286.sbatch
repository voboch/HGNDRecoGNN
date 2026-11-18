#!/usr/bin/env bash
#SBATCH --job-name=hgnd-rapidity-s286
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${RAPIDITY_SOURCE:?}" "${RAPIDITY_CAMPAIGN:?}" "${RAPIDITY_RECEIPTS:?}" "${RAPIDITY_SUMS_SHA:?}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$RAPIDITY_SOURCE"
printf '%s  SHA256SUMS\n' "$RAPIDITY_SUMS_SHA" | sha256sum -c --quiet
sha256sum -c --quiet SHA256SUMS
batch=HGNDRecoGNN/reports/rapidity_export_20261117
python_path=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
work=$(mktemp -d "/tmp/hgnd-rapidity-${SLURM_JOB_ID}.XXXXXX")
trap 'code=$?; if [[ $code -ne 0 ]]; then echo "Failure $code; node-local work: $work" >&2; fi' EXIT
case "$(stat -f -c %T "$work")" in xfs|ext2/ext3|tmpfs|btrfs|overlay|zfs) ;; *) exit 2 ;; esac
export TMPDIR="$work"
echo "Node-local work $work on $(hostname)"
"$python_path" "$batch/run_s286.py" --launch "$batch/launch.json" --work "$work/work" --output "$work/output"
sha256sum -c --quiet SHA256SUMS
"$python_path" HGNDRecoGNN/scripts/publish_smash_directory.py --source "$work/output" --bundle "$work/export" \
  --destination "$RAPIDITY_CAMPAIGN/result_$SLURM_JOB_ID" --receipt "$RAPIDITY_RECEIPTS/result_$SLURM_JOB_ID.json"
