#!/usr/bin/env bash
# Frozen DCM weights/cuts/scalers; complete-audit gate precedes model inference.
#SBATCH --job-name=smash-full-transfer
#SBATCH --account=proj_1855
#SBATCH --partition=rocky
#SBATCH --constraint=type_a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --exclude=cn-007,cn-008,cn-034
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${SMASH_SOURCE:?Immutable source}"
: "${SMASH_CAMPAIGN:?Full audits required}"
: "${SMASH_INVENTORY_SHA:?Pinned full inventory}"
: "${SMASH_INVENTORY:?Frozen bound inventory}"
: "${SMASH_TAG:?Frozen potential or all for engineering smoke}"
: "${SMASH_AUDIT_BUNDLES:?Pinned verified audit bundle index}"
: "${SMASH_RECEIPT_DIR:?Home receipt directory}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/vbocharnikov/.conda/envs/student-ufc-240/bin/python
cd "$SMASH_SOURCE"
WORK=$(mktemp -d "/tmp/hgnd-smash-full-gpu-${SLURM_JOB_ID}.XXXXXX")
LOCAL_FS=$(stat -f -c %T "$WORK")
case "$LOCAL_FS" in
    xfs|ext2/ext3|tmpfs|btrfs|overlay|zfs) ;;
    *) echo "Expected local output filesystem, got $LOCAL_FS at $WORK" >&2; exit 2 ;;
esac
echo "Node-local work: $WORK ($LOCAL_FS)"

SUCCESS=0
cleanup() {
    if [[ "$SUCCESS" == 1 ]]; then rm -rf -- "$WORK";
    else rm -rf -- "$WORK/audit" "$WORK/unpack"; echo "Preserved local outputs: $WORK/output" >&2; fi
}
trap cleanup EXIT
mkdir -p "$WORK/audit"
"$PYTHON" - "$SMASH_SOURCE" "$SMASH_AUDIT_BUNDLES" "$WORK" <<'PYCODE'
import hashlib,json,pathlib,subprocess,sys
source,index,work=pathlib.Path(sys.argv[1]),json.load(open(sys.argv[2])),pathlib.Path(sys.argv[3])
assert set(index)=={'zeroSpot','defaultSpot','bigSpot'}
(work/'audit/audit').mkdir()
for tag,item in index.items():
    bundle=pathlib.Path(item['bundle_dir'])
    assert hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest()==item['manifest_sha256']
    output=work/'unpack'/tag
    subprocess.run([sys.executable,str(source/'HGNDRecoGNN/scripts/smash_input_bundle.py'),'unpack','--archive',str(bundle/'inputs.tar'),'--manifest',str(bundle/'manifest.json'),'--output',str(output)],check=True)
    (work/'audit/audit'/tag).symlink_to(output/'audit'/tag)
PYCODE
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0),flush=True)'
TAGS=("$SMASH_TAG")
EXTRA=()
PHASE=transfer
if [[ "${SMASH_PROBE_FIRST:-0}" == 1 ]]; then
    TAGS=(zeroSpot defaultSpot bigSpot)
    EXTRA=(--probe-first)
    PHASE=transfer_probe
fi
for TAG in "${TAGS[@]}"; do
    "$PYTHON" -u HGNDRecoGNN/scripts/smash_transfer_infer.py \
        --audit-campaign "$WORK/audit" --inventory "$SMASH_INVENTORY" \
        --inventory-sha256 "$SMASH_INVENTORY_SHA" --tag "$TAG" --device cuda "${EXTRA[@]}" \
        --native-campaign /scratch/vbocharnikov/hgnd/experiments/student_pipeline_20261020_56bcc48206 \
        --prepared /scratch/vbocharnikov/hgnd/experiments/dcm_grouped_20261016_077a236ca2/prepared \
        --output "$WORK/output/$PHASE/$TAG"
done

"$PYTHON" -u HGNDRecoGNN/scripts/publish_smash_directory.py \
    --source "$WORK/output" --bundle "$WORK/export_bundle" \
    --destination "$SMASH_CAMPAIGN/${PHASE}_bundles/${SMASH_TAG}_${SLURM_JOB_ID}" \
    --receipt "$SMASH_RECEIPT_DIR/${PHASE}_${SMASH_TAG}_${SLURM_JOB_ID}.json"
SUCCESS=1
