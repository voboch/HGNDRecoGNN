#!/usr/bin/env bash
# Prepared node-local-output recovery; additional human approval required.
#SBATCH --job-name=smash-map-local-output
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --exclude=cn-007,cn-008,cn-034
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${SMASH_MAP_SOURCE:?New pinned mapper}"
: "${SMASH_MAP_SHA:?Pinned mapper SHA256}"
: "${SMASH_INVENTORY:?Frozen original inventory}"
: "${SMASH_CAMPAIGN:?Preserved campaign}"
: "${SMASH_RAW_BUNDLE:?Previously verified raw bundle}"
: "${SMASH_RAW_SHA:?Raw manifest hash}"
: "${SMASH_PRIMARY_BUNDLE:?Completed primary bundle}"
: "${SMASH_PRIMARY_SHA:?Primary manifest hash}"
: "${SMASH_PUBLISHER:?Pinned publisher required}"
: "${SMASH_PUBLISHER_SHA:?Pinned publisher hash}"
: "${SMASH_RECEIPT_DIR:?Home receipt directory required}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
HELPER=/home/vbocharnikov/hgnd_experiments/smash_bundle_helper_b964908c62.py
WORK=$(mktemp -d "/tmp/hgnd-smash-primary-${SLURM_JOB_ID}.XXXXXX")
LOCAL_FS=$(stat -f -c %T "$WORK")
case "$LOCAL_FS" in
    xfs|ext2/ext3|tmpfs|btrfs|overlay|zfs) ;;
    *) echo "Expected local output filesystem, got $LOCAL_FS at $WORK" >&2; exit 2 ;;
esac
echo "Node-local work: $WORK ($LOCAL_FS)"

SUCCESS=0
cleanup() {
    if [[ "$SUCCESS" == 1 ]]; then rm -rf -- "$WORK";
    else rm -rf -- "$WORK/raw" "$WORK/primary"; echo "Preserved node-local result: $WORK/result" >&2; fi
}
trap cleanup EXIT
"$PYTHON" - "$SMASH_MAP_SOURCE" "$SMASH_MAP_SHA" "$SMASH_RAW_BUNDLE/manifest.json" "$SMASH_RAW_SHA" "$SMASH_PRIMARY_BUNDLE/manifest.json" "$SMASH_PRIMARY_SHA" "$WORK" <<'PY'
import hashlib,json,shutil,sys
for p,h in zip(sys.argv[1:7:2],sys.argv[2:7:2]):
    assert hashlib.sha256(open(p,'rb').read()).hexdigest()==h
size=sum(json.load(open(p))['unpacked_bytes'] for p in [sys.argv[3],sys.argv[5]])
assert shutil.disk_usage(sys.argv[7]).free > size*2
PY
for KIND in raw primary; do
    if [[ "$KIND" == raw ]]; then BUNDLE="$SMASH_RAW_BUNDLE"; else BUNDLE="$SMASH_PRIMARY_BUNDLE"; fi
    "$PYTHON" -u "$HELPER" unpack --archive "$BUNDLE/inputs.tar" \
        --manifest "$BUNDLE/manifest.json" --output "$WORK/$KIND"
done
cp "$WORK/primary/bundle_verified.json" "$SMASH_RECEIPT_DIR/zeroSpot_node_output_${SLURM_JOB_ID}_primary.json"
"$PYTHON" -u "$SMASH_MAP_SOURCE" --inventory "$SMASH_INVENTORY" \
    --staged-root "$WORK/raw" --staged-primary-root "$WORK/primary" \
    --tag zeroSpot --output "$WORK/result"
"$PYTHON" -c 'import hashlib,sys; assert hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()==sys.argv[2]' "$SMASH_PUBLISHER" "$SMASH_PUBLISHER_SHA"
"$PYTHON" -u "$SMASH_PUBLISHER" --source "$WORK/result/summary.json" \
    --destination "$SMASH_CAMPAIGN/mapping/zeroSpot_node_output_${SLURM_JOB_ID}" \
    --receipt "$SMASH_RECEIPT_DIR/zeroSpot_node_output_${SLURM_JOB_ID}_receipt.json" \
    --inventory-sha 9ca9df3edc0fafdc9d2e1720f37136443d41d96424966ec4fd43c35be786e110
SUCCESS=1
