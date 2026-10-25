#!/usr/bin/env bash
# Fixed B-free truth replay. CPU only; no training, weights or test evaluation.
#SBATCH --job-name=hgnd-bfree-truth
#SBATCH --account=proj_1855
#SBATCH --partition=test
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${BF_SOURCE:?immutable source directory}"
: "${BF_EXPORT:?new campaign export root}"
: "${BF_ENV:?Python environment}"
: "${BF_MODE:?smoke, certify or aggregate}"
[[ $BF_MODE == smoke || $BF_MODE == certify || $BF_MODE == aggregate ]] || exit 2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
JOB_KEY="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-single}"
STAGE="${SLURM_TMPDIR:-/tmp}/hgnd-bfree-${JOB_KEY}"
mkdir "$STAGE"
cd "$BF_SOURCE"
sha256sum -c SHA256SUMS
ARGS=(--source "$BF_SOURCE" --mode "$BF_MODE" --job-id "$JOB_KEY" --output "$STAGE/result")
if [[ $BF_MODE != smoke ]]; then
  : "${BF_SMOKE_RECEIPT:?verified test smoke required}"
  ARGS+=(--smoke-receipt "$BF_SMOKE_RECEIPT")
fi
if [[ $BF_MODE == certify ]]; then
  : "${SLURM_ARRAY_TASK_ID:?three sample array required}"
  ARGS+=(--task "$SLURM_ARRAY_TASK_ID")
fi
if [[ $BF_MODE == aggregate ]]; then
  : "${BF_FULL_RECEIPTS:?all three verified full receipts required}"
  ARGS+=(--full-receipts "$BF_FULL_RECEIPTS")
fi
"$BF_ENV/bin/python" -u HGNDRecoGNN/scripts/bfree_truth_campaign.py "${ARGS[@]}"
sha256sum -c SHA256SUMS
# One export to a unique directory; any failure exits without a retry.
tar -C "$STAGE" -czf "$STAGE/result.tar.gz" result
DEST="$BF_EXPORT/${BF_MODE}_${JOB_KEY}"
mkdir "$DEST"
cp "$STAGE/result.tar.gz" "$DEST/result.tar.gz"
cmp "$STAGE/result.tar.gz" "$DEST/result.tar.gz"
cp "$STAGE/result/summary.json" "$DEST/summary.json"
sha256sum "$DEST/result.tar.gz" "$DEST/summary.json"
