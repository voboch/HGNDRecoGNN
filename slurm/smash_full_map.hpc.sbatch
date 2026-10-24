#!/usr/bin/env bash
# Exact source mapping only; no model evaluation.
#SBATCH --job-name=smash-full-map
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --exclude=cn-007,cn-008,cn-034
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${SMASH_MAP_SOURCE:?Pinned standalone source}"
: "${SMASH_MAP_SHA:?Pinned source SHA256}"
: "${SMASH_INVENTORY:?Frozen inventory}"
: "${SMASH_CAMPAIGN:?New campaign}"
: "${SMASH_BUNDLE:?Verified raw bundle}"
: "${SMASH_MANIFEST_SHA:?Pinned manifest SHA256}"
: "${SMASH_TAG:?Frozen potential}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
HELPER=/home/vbocharnikov/hgnd_experiments/smash_bundle_helper_b964908c62.py
WORK=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/hgnd-smash-map-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$WORK"' EXIT
"$PYTHON" - "$SMASH_MAP_SOURCE" "$SMASH_MAP_SHA" "$SMASH_BUNDLE/manifest.json" "$SMASH_MANIFEST_SHA" "$WORK" <<'PY'
import hashlib,json,shutil,sys
for p,h in [(sys.argv[1],sys.argv[2]),(sys.argv[3],sys.argv[4])]:
    assert hashlib.sha256(open(p,'rb').read()).hexdigest()==h
m=json.load(open(sys.argv[3]))
# Space is an engineering requirement, not a data selection.
assert shutil.disk_usage(sys.argv[5]).free > m['unpacked_bytes']*2
PY
"$PYTHON" -u "$HELPER" unpack --archive "$SMASH_BUNDLE/inputs.tar" \
    --manifest "$SMASH_BUNDLE/manifest.json" --output "$WORK/inputs"
mkdir -p "$SMASH_CAMPAIGN/access"
cp "$WORK/inputs/bundle_verified.json" "$SMASH_CAMPAIGN/access/${SLURM_JOB_ID}.json"
"$PYTHON" -u "$SMASH_MAP_SOURCE" --inventory "$SMASH_INVENTORY" \
    --staged-root "$WORK/inputs" --tag "$SMASH_TAG" \
    --output "$SMASH_CAMPAIGN/mapping/$SMASH_TAG"
