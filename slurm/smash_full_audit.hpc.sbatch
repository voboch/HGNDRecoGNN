#!/usr/bin/env bash
# Full label/source audit on byte-verified node-local inputs; no model scores.
#SBATCH --job-name=smash-full-audit
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --exclude=cn-007,cn-008,cn-034
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${SMASH_SOURCE:?Immutable source}"
: "${SMASH_CAMPAIGN:?New outputs}"
: "${SMASH_BUNDLE:?Pinned bundle}"
: "${SMASH_MANIFEST_SHA:?Pinned manifest}"
: "${SMASH_TAG:?Frozen potential}"
: "${SMASH_INVENTORY:?Frozen bound inventory}"
: "${SMASH_INVENTORY_SHA:?Pinned bound inventory}"
: "${SMASH_RECEIPT_DIR:?Home receipt directory}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
cd "$SMASH_SOURCE"
WORK=$(mktemp -d "/tmp/hgnd-smash-audit-${SLURM_JOB_ID}.XXXXXX")
LOCAL_FS=$(stat -f -c %T "$WORK")
case "$LOCAL_FS" in
    xfs|ext2/ext3|tmpfs|btrfs|overlay|zfs) ;;
    *) echo "Expected local output filesystem, got $LOCAL_FS at $WORK" >&2; exit 2 ;;
esac
echo "Node-local work: $WORK ($LOCAL_FS)"

SUCCESS=0
cleanup() {
    if [[ "$SUCCESS" == 1 ]]; then rm -rf -- "$WORK";
    else rm -rf -- "$WORK/inputs" "$WORK/primary"; echo "Preserved local outputs: $WORK/output" >&2; fi
}
trap cleanup EXIT
"$PYTHON" - "$SMASH_BUNDLE/manifest.json" "$SMASH_MANIFEST_SHA" "$WORK" "$SMASH_INVENTORY" "$SMASH_INVENTORY_SHA" "${SMASH_PRIMARY_BUNDLE:-}" "${SMASH_PRIMARY_MANIFEST_SHA:-}" <<'PY'
import hashlib,json,shutil,sys
assert hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()==sys.argv[2]
assert hashlib.sha256(open(sys.argv[4],'rb').read()).hexdigest()==sys.argv[5]
m=json.load(open(sys.argv[1]))
size=m['unpacked_bytes']
if sys.argv[6]:
    path=sys.argv[6]+'/manifest.json'
    assert hashlib.sha256(open(path,'rb').read()).hexdigest()==sys.argv[7]
    size+=json.load(open(path))['unpacked_bytes']
assert shutil.disk_usage(sys.argv[3]).free > size*2
PY
"$PYTHON" -u HGNDRecoGNN/scripts/smash_input_bundle.py unpack \
    --archive "$SMASH_BUNDLE/inputs.tar" --manifest "$SMASH_BUNDLE/manifest.json" --output "$WORK/inputs"
cp "$WORK/inputs/bundle_verified.json" "$SMASH_RECEIPT_DIR/access_${SLURM_JOB_ID}.json"
if [[ -n "${SMASH_PRIMARY_BUNDLE:-}" ]]; then
    "$PYTHON" -u HGNDRecoGNN/scripts/smash_input_bundle.py unpack \
        --archive "$SMASH_PRIMARY_BUNDLE/inputs.tar" --manifest "$SMASH_PRIMARY_BUNDLE/manifest.json" --output "$WORK/primary"
    ln -s "$WORK/primary/primary" "$WORK/inputs/primary"
    cp "$WORK/primary/bundle_verified.json" "$SMASH_RECEIPT_DIR/access_${SLURM_JOB_ID}_primary.json"
fi
"$PYTHON" -u HGNDRecoGNN/scripts/audit_smash_sources.py \
    --raw-root /scratch/vbocharnikov/hgnd/data_fixed \
    --primary-root /scratch/vbocharnikov/hgnd/data_np_full_20260924/extracted \
    --inventory "$SMASH_INVENTORY" --staged-root "$WORK/inputs" \
    --save-observations --tag "$SMASH_TAG" --output "$WORK/output/audit/$SMASH_TAG"
"$PYTHON" -u HGNDRecoGNN/scripts/publish_smash_directory.py \
    --source "$WORK/output" --bundle "$WORK/export_bundle" \
    --destination "$SMASH_CAMPAIGN/audit_bundles/${SMASH_TAG}_${SLURM_JOB_ID}" \
    --receipt "$SMASH_RECEIPT_DIR/audit_${SMASH_TAG}_${SLURM_JOB_ID}.json"
SUCCESS=1
