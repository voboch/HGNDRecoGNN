#!/usr/bin/env bash
# Finite, unchanged student recipes. --array=0-17%2 is set by the launcher.
#SBATCH --job-name=student-ufc
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --exclude=cn-008
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
: "${STUDENT_SOURCE:?Frozen source root required}"
: "${STUDENT_CAMPAIGN:?Scratch campaign root required}"
: "${STUDENT_ENV:?Dedicated environment required}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/student-mpl-${SLURM_JOB_ID}"
export PYTHONDONTWRITEBYTECODE=1
# Match the seed before Python starts; training also seeds Python/NumPy/Torch.
TASK=${SLURM_ARRAY_TASK_ID:-0}
SEEDS=(42 7 21)
export PYTHONHASHSEED=${SEEDS[$((TASK / 6))]}
cd "$STUDENT_SOURCE"
case "${STUDENT_PHASE:-train}" in
  probe)
    "$STUDENT_ENV/bin/python" -u -c 'import torch,torch_geometric,torch_cluster; print(torch.__version__,torch_geometric.__version__,torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1,device="cuda"))'
    ;;
  smoke)
    "$STUDENT_ENV/bin/python" -u HGNDRecoGNN/scripts/student_reproduce.py smoke \
      --source-manifest "$STUDENT_SOURCE/source_manifest.json" \
      --prepared "$STUDENT_CAMPAIGN/prepared" --output "$STUDENT_CAMPAIGN/smoke"
    ;;
  train)
    "$STUDENT_ENV/bin/python" -u HGNDRecoGNN/scripts/student_reproduce.py train \
      --source-manifest "$STUDENT_SOURCE/source_manifest.json" \
      --prepared "$STUDENT_CAMPAIGN/prepared" --campaign "$STUDENT_CAMPAIGN" --task "$TASK"
    ;;
  evaluate)
    "$STUDENT_ENV/bin/python" -u HGNDRecoGNN/scripts/student_reproduce.py evaluate \
      --source-manifest "$STUDENT_SOURCE/source_manifest.json" \
      --prepared "$STUDENT_CAMPAIGN/prepared" --campaign "$STUDENT_CAMPAIGN" \
      --output "$STUDENT_CAMPAIGN/test_evaluation"
    ;;
  *) echo "Unknown STUDENT_PHASE" >&2; exit 2 ;;
esac
