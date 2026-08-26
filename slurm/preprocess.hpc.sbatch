#!/bin/bash
# preprocess.hpc.sbatch — CPU-only schema-v2 cache build on Rocky Linux 9.
# Submit from ~/HGNDRecoGNN on login-02 with absolute scratch paths exported.
#
#SBATCH --job-name=hgnd-preprocess-v2
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# Rocky currently advertises RealMemory=1 MB; an explicit --mem request is
# therefore unschedulable. Use the partition's unlimited default until fixed.
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# The cluster profile references optional variables, so load it before nounset.
source /etc/profile 2>/dev/null || true
set -euo pipefail

: "${ENV_NAME:=hgnd-env}"
: "${PROJECT_DIR:=$HOME}"
: "${CSV_DIR:?set CSV_DIR to the raw defaultSpot directory on scratch}"
: "${DATASET_ROOT:?set DATASET_ROOT to a cache directory on scratch}"
: "${NUM_WORKERS:=$SLURM_CPUS_PER_TASK}"
: "${MAX_EVENTS:=}"

module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

cd "$PROJECT_DIR"

echo "host:       $(hostname)"
echo "date:       $(date)"
echo "job:        ${SLURM_JOB_ID:-local}  part=${SLURM_JOB_PARTITION:-?}"
echo "env:        $ENV_NAME  cwd=$PROJECT_DIR"
echo "csv:        $CSV_DIR"
echo "root:       $DATASET_ROOT"
echo "workers:    $NUM_WORKERS"
echo "max-events: ${MAX_EVENTS:-all} per detector half"
python -c "from HGNDRecoGNN.data.graph_dataset import SCHEMA_V; print('schema:', SCHEMA_V)"

args=(
    -m HGNDRecoGNN.scripts.preprocess
    --csv-dir "$CSV_DIR"
    --root "$DATASET_ROOT"
    --num-workers "$NUM_WORKERS"
)
if [[ -n "$MAX_EVENTS" ]]; then
    args+=(--max-events "$MAX_EVENTS")
fi

python -u "${args[@]}"

echo "HGND_EXIT=0"
echo "preprocess done: $(date)"
