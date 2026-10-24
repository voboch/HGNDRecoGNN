#!/usr/bin/env bash
# Exact content census. No training, fitted statistics or GPU allocation.
#SBATCH --job-name=hgnd-content-census
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
: "${CENSUS_SOURCE:?immutable source directory}"
: "${CENSUS_EXPORT:?new export directory}"
: "${CENSUS_ENV:?Python environment}"
: "${CENSUS_DCM:?read-only DCM root}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
STAGE="${SLURM_TMPDIR:-/tmp}/hgnd-content-${SLURM_JOB_ID}"
mkdir "$STAGE"
cd "$CENSUS_SOURCE"
sha256sum -c SHA256SUMS
ARGS=(--dcm-root "$CENSUS_DCM" --smash-inventory original_inventory.json --output "$STAGE/result")
[[ ${CENSUS_MODE:-smoke} == smoke || ${CENSUS_MODE:-smoke} == full ]] || exit 2
if [[ ${CENSUS_MODE:-smoke} == smoke ]]; then
  ARGS+=(--smoke)
else
  : "${CENSUS_SMOKE_RECEIPT:?successful smoke receipt required}"
  "$CENSUS_ENV/bin/python" - "$CENSUS_SMOKE_RECEIPT" "$CENSUS_DCM" <<'PY'
import sys
from content_census import validate_smoke
validate_smoke(sys.argv[1], 'original_inventory.json', sys.argv[2])
PY
fi
"$CENSUS_ENV/bin/python" -u content_census.py "${ARGS[@]}"
sha256sum -c SHA256SUMS
tar -C "$STAGE" -czf "$STAGE/result.tar.gz" result
mkdir "$CENSUS_EXPORT"
cp "$STAGE/result.tar.gz" "$CENSUS_EXPORT/result.tar.gz"
cmp "$STAGE/result.tar.gz" "$CENSUS_EXPORT/result.tar.gz"
cp "$STAGE/result/summary.json" "$CENSUS_EXPORT/summary.json"
sha256sum "$CENSUS_EXPORT/result.tar.gz" "$CENSUS_EXPORT/summary.json"
