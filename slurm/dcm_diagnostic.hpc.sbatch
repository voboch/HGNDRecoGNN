#!/bin/bash
# Diagnostic comparison on the existing graph split. Validation is reused for
# model selection; these outputs are not an independent-test benchmark.
# Uses one immutable source snapshot and one existing cache for every variant.
# The array is capped at two GPUs at submission to limit concurrent resources.
# Eight hours allow slower V100s to run the full 30-epoch, wider-pooling model;
# the historical 25-epoch baseline needed 1h40 on an A100.
#SBATCH --job-name=dcm_diag
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile 2>/dev/null || true
set -euo pipefail
: "${PROJECT_DIR:?set PROJECT_DIR to the snapshot parent of HGNDRecoGNN}"
: "${CAMPAIGN_ROOT:?set a new campaign directory on scratch}"
: "${DATA_ROOT:=/scratch/$USER/hgnd/cache/ndet_dataset_dcm_xecs_own_tw1p0}"
: "${MODE:=train}"
module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1
source activate hgnd-env 2>/dev/null || conda activate hgnd-env
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Match CPU threads to the allocation, including the constructor's CPU dummy pass.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'

if [ "$MODE" = smoke ]; then
    python -u -m HGNDRecoGNN.scripts.smoke_dcm_ladder \
        --root "$DATA_ROOT" --out-dir "$CAMPAIGN_ROOT/smoke" --batch-size 256
    exit 0
fi

variants=(ref epochs sched warmup size pool residual)
VARIANT=${variants[${SLURM_ARRAY_TASK_ID:?array task index required}]}
RUN_DIR="$CAMPAIGN_ROOT/$VARIANT"
mkdir -p "$RUN_DIR"
# Claim once: automatic reuse of partial optimizer/model pairs remains unsafe.
mkdir "$RUN_DIR/started"
export VARIANT CKPT_DIR="$RUN_DIR/checkpoints" SCALER=own TWINDOW=1.0 BATCH=256
# Loading every explicitly enumerated shard also permits the compatible v2
# hits/clusters cache without triggering an unnecessary schema-v3 rebuild.
NUM_SHARDS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_shards"])' "$DATA_ROOT/processed/meta.json")
export NUM_SHARDS
export RUN_DIR DATA_ROOT
python - <<'PY'
import json, os, pathlib, shutil, time
p = pathlib.Path(os.environ['RUN_DIR'])
config = dict(variant=os.environ['VARIANT'], data_root=os.environ['DATA_ROOT'],
              shards=int(os.environ['NUM_SHARDS']), seed=42, batch_size=256,
              split='legacy_graph_validation', independent_test=False,
              slurm_job_id=os.environ['SLURM_JOB_ID'],
              slurm_array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],
              started_at_unix=time.time())
(p / 'effective_run.json').write_text(json.dumps(config, indent=2) + '\n')
shutil.copyfile(pathlib.Path(os.environ['PROJECT_DIR']) / 'source_manifest.json',
                p / 'source_manifest.json')
PY
bash "$PROJECT_DIR/HGNDRecoGNN/slurm/train_dcm_improve.hpc.sbatch"
python -u -m HGNDRecoGNN.scripts.evaluate \
    --root "$DATA_ROOT" --checkpoint "$CKPT_DIR/model.pt" \
    --out-dir "$RUN_DIR/validation" --suffix dcm --split test \
    --num-shards "$NUM_SHARDS" --batch-size 512 --device cuda
python -u -m HGNDRecoGNN.scripts.reco_performance \
    --clusters "$RUN_DIR/validation/pred_clusters_dcm.pkl" \
    --out "$RUN_DIR/validation/performance.json"
python - <<'PY'
import json, os, pathlib, time
p = pathlib.Path(os.environ['RUN_DIR'])
status = {'state': 'completed', 'finished_at_unix': time.time(),
          'interpretation': 'single-seed diagnostic on reused validation split',
          'independent_test': False}
(p / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
PY
