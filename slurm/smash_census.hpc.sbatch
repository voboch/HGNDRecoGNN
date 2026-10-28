#!/usr/bin/env bash
# Exact content census of corrected-SMASH detector exports. CPU only; no
# training, fitted statistics, jitter or GPU allocation.
#   smoke: all three potentials on three pinned files each (test partition)
#   full:  one potential per array task (0 zeroSpot, 1 defaultSpot, 2 bigSpot),
#          gated by an externally verified smoke receipt
#SBATCH --job-name=hgnd-smash-census
#SBATCH --account=proj_1855
#SBATCH --partition=test
#SBATCH --constraint=type_d
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${SC_SOURCE:?immutable source directory}"
: "${SC_EXPORT:?campaign export root}"
: "${SC_ENV:?Python environment}"
: "${SC_MODE:?smoke or full}"
[[ $SC_MODE == smoke || $SC_MODE == full ]] || exit 2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
JOB_KEY="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-single}"
# Results are written node-locally and exported once (RA-09: scratch writes
# from compute nodes have failed with EACCES mid-run).
STAGE="${SLURM_TMPDIR:-/tmp}/hgnd-smash-census-${JOB_KEY}"
mkdir "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
cd "$SC_SOURCE"
sha256sum -c --quiet SHA256SUMS
ARGS=(run --source "$SC_SOURCE" --mode "$SC_MODE" --job-id "$JOB_KEY" --output "$STAGE/result"
      --workers "${SLURM_CPUS_PER_TASK:-1}")
if [[ $SC_MODE == full ]]; then
  : "${SC_SMOKE_RECEIPT:?verified test smoke required}"
  : "${SLURM_ARRAY_TASK_ID:?three-task array required}"
  ARGS+=(--smoke-receipt "$SC_SMOKE_RECEIPT" --task "$SLURM_ARRAY_TASK_ID")
fi
set +e
"$SC_ENV/bin/python" -u HGNDRecoGNN/scripts/smash_export_census.py "${ARGS[@]}"
rc=$?
set -e
# 0 complete; 3 stopped on a census condition, evidence exported for review;
# anything else is a crash and exports nothing. No automatic retry.
[[ $rc == 0 || $rc == 3 ]] || exit "$rc"
sha256sum -c --quiet SHA256SUMS
tar -C "$STAGE" -czf "$STAGE/result.tar.gz" result
DEST="$SC_EXPORT/${SC_MODE}_${JOB_KEY}"
mkdir "$DEST"
cp "$STAGE/result.tar.gz" "$DEST/result.tar.gz"
cmp "$STAGE/result.tar.gz" "$DEST/result.tar.gz"
cp "$STAGE/result/summary.json" "$DEST/summary.json"
sha256sum "$DEST/result.tar.gz" "$DEST/summary.json"
exit "$rc"
