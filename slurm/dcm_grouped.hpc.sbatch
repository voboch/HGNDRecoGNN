#!/bin/bash
# Independent source-run control of the schedule effect, three matched seeds.
# 30 epochs match the finished diagnostic budget; no claim of convergence.
# Three whole source runs give train/validation/test; both detector arms stay
# together. The one test run cannot measure between-job variance.
#SBATCH --job-name=dcm_grouped
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile 2>/dev/null || true
set -euo pipefail
: "${PROJECT_DIR:?immutable snapshot parent}"
: "${CAMPAIGN_ROOT:?new campaign directory}"
: "${PHASE:=train}"
module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
source activate hgnd-env 2>/dev/null || conda activate hgnd-env
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
COMMON=(--root /scratch/"$USER"/hgnd/cache/ndet_dataset_dcm_xecs_own_tw1p0
        --raw-root /home/"$USER"/het_nDetHalf/data/ndet_links_Jan_26
        --campaign "$CAMPAIGN_ROOT")
if [ "$PHASE" = smoke ]; then
    if [ ! -s "$CAMPAIGN_ROOT/prepared/manifest.json" ]; then
        python -u -m HGNDRecoGNN.scripts.dcm_grouped --phase prepare "${COMMON[@]}"
    fi
    python -u -m HGNDRecoGNN.scripts.dcm_grouped --phase smoke "${COMMON[@]}"
    exit 0
fi
arms=(multistep cosine multistep cosine multistep cosine)
seeds=(42 42 123 123 456 456)
i=${SLURM_ARRAY_TASK_ID:?array task required}
python -u -m HGNDRecoGNN.scripts.dcm_grouped --phase "$PHASE" \
    --arm "${arms[$i]}" --seed "${seeds[$i]}" "${COMMON[@]}"
