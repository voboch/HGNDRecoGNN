#!/usr/bin/env bash
# Frozen before CPU preparation so its source identity is reusable for training.
# GPU submission separately requires accepted patches and concrete receipt pins.
#SBATCH --job-name=hgnd-label-nofloor
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=02:45:00
#SBATCH --array=0-8%2
#SBATCH --no-requeue
#SBATCH --exclude=cn-007,cn-008,cn-034
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
: "${AB_SOURCE:?Immutable source of accepted patch preparation}"
: "${AB_CAMPAIGN:?Campaign directory}"
: "${AB_RECEIPTS:?Home receipt directory}"
: "${AB_SEGMENT:?1 or 2; segment 2 array must depend afterok on ALL of segment 1}"
: "${AB_PATCH_BUNDLE:?Accepted patch bundle}"
: "${AB_PATCH_MANIFEST_SHA:?Accepted publication manifest SHA}"
: "${AB_PATCH_SUMMARY_SHA:?Accepted patch summary SHA}"
: "${SLURM_ARRAY_TASK_ID:?Registered task 0..8}"
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$AB_SOURCE"
sha256sum -c --quiet SHA256SUMS
AB_WORK=$(mktemp -d "/tmp/hgnd-nofloor-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}.XXXXXX")
export AB_WORK
case "$(stat -f -c %T "$AB_WORK")" in
    xfs|ext2/ext3|tmpfs|btrfs|overlay|zfs) ;;
    *) echo "Expected node-local work filesystem" >&2; exit 2 ;;
esac
export MPLCONFIGDIR="$AB_WORK/mpl" XDG_CACHE_HOME="$AB_WORK/cache"
"$PYTHON" -u - <<'PY'
import json, os, subprocess, sys
from pathlib import Path
from HGNDRecoGNN.scripts.smash_train_prepare import sha
from HGNDRecoGNN.scripts.smash_input_bundle import unpack
env=os.environ; root=Path.cwd(); work=Path(env['AB_WORK'])
plan=json.loads((root/'HGNDRecoGNN/reports/label_ablation_20261103/launch_plan.json').read_text())
bundle=Path(env['AB_PATCH_BUNDLE'])
assert sha(bundle/'manifest.json') == env['AB_PATCH_MANIFEST_SHA']
unpack(bundle/'inputs.tar',bundle/'manifest.json',work/'patches')
assert sha(work/'patches/summary.json') == env['AB_PATCH_SUMMARY_SHA']
task=int(env['SLURM_ARRAY_TASK_ID']); segment=int(env['AB_SEGMENT'])
assert 0 <= task < 9 and segment in (1,2)
baseline=plan['baseline'][str(task)]
common=root/'HGNDRecoGNN/reports/label_ablation_20261103'
args=[sys.executable,'-u',str(root/'HGNDRecoGNN/scripts/label_ablation_train.py'),
      '--source-sums',str(root/'SHA256SUMS'),
      '--split',str(common/plan['split']['path']),
      '--authorization',str(common/plan['authorization']['path']),
      '--prepare-bundle',plan['prepare_bundle'],'--prepare-manifest-sha256',plan['prepare_bundle_manifest_sha256'],
      '--patches',str(work/'patches'),'--patch-manifest-sha256',env['AB_PATCH_SUMMARY_SHA'],
      '--baseline-complete',str(common/baseline['path']),'--baseline-complete-sha256',baseline['sha256'],
      '--work',str(work/'work'),'--output',str(work/'output'),
      '--mirror',str(Path(env['AB_CAMPAIGN'])/'mirror'/f'task_{task}'),
      '--task',str(task),'--segment',str(segment),'--device','cuda']
if segment == 2:
    args += ['--resume-mirror',str(Path(env['AB_CAMPAIGN'])/'mirror'/f'task_{task}')]
subprocess.run(args,check=True)
PY
sha256sum -c --quiet SHA256SUMS
"$PYTHON" -u HGNDRecoGNN/scripts/publish_smash_directory.py \
    --source "$AB_WORK/output" --bundle "$AB_WORK/export" \
    --destination "$AB_CAMPAIGN/task_${SLURM_ARRAY_TASK_ID}_segment_${AB_SEGMENT}_${SLURM_JOB_ID}" \
    --receipt "$AB_RECEIPTS/task_${SLURM_ARRAY_TASK_ID}_segment_${AB_SEGMENT}_${SLURM_JOB_ID}.json"
echo "Segment exported; node-local outputs preserved at $AB_WORK"
