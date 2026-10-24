#!/usr/bin/env bash
# Continue the verified atomic epoch-61 state, retaining the original 100-epoch cap.
# Ninety minutes covers 39 remaining epochs at observed ~37 s/epoch plus I/O reserve;
# together with the failed 46-minute allocation this stays below its original 4 h cap.
#SBATCH --job-name=student-ufc-resume
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --exclude=cn-007,cn-008
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
: "${STUDENT_SOURCE:?Original frozen source required}"
: "${STUDENT_RECOVERY:?Separate recovery campaign view required}"
: "${STUDENT_ENV:?Original dedicated environment required}"
export PYTHONHASHSEED=21
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/student-mpl-${SLURM_JOB_ID}"
cd "$STUDENT_SOURCE"
"$STUDENT_ENV/bin/python" -u HGNDRecoGNN/scripts/student_reproduce.py train \
  --source-manifest "$STUDENT_SOURCE/source_manifest.json" \
  --prepared "$STUDENT_RECOVERY/prepared" --campaign "$STUDENT_RECOVERY" \
  --task 13 --resume
