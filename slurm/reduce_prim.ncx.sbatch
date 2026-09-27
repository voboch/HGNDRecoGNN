#!/bin/bash
# Reduce the primary-nucleon CSV exports to per-event tables, on ncx where the
# data is local.
#
# Streaming the tarballs over the sshfs mount moves ~25 GB to produce ~115 MB of
# tables, and the mount's throughput varies between 0.26 and 4.2 MB/s.  Running
# here reads the already-extracted CSVs from local disk and returns only the
# tables.
#
#   sbatch slurm/reduce_prim.ncx.sbatch
#   # then, from the laptop:
#   cp /Users/vovvy/ncx/<OUTDIR>/*_events.pkl  <local events dir>/
#   cp /Users/vovvy/ncx/<OUTDIR>/*_filehist.npz <local events dir>/
#
#SBATCH --job-name=reduce_prim
#SBATCH --output=out/reduce_prim_%j.out
#SBATCH --error=out/reduce_prim_%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"
mkdir -p out

export DATA_ROOT="${DATA_ROOT:-/scratch2/vboch/data}"
OUTDIR="${OUTDIR:-$SUBMIT_DIR/prim_events}"
mkdir -p "$OUTDIR"

echo "host      : $(hostname)"
echo "DATA_ROOT : $DATA_ROOT"
echo "OUTDIR    : $OUTDIR"

if ! python3 -c "import numpy, pandas" 2>/dev/null; then
    echo "ERROR: python3 needs numpy and pandas."
    echo "Try a module (module avail python) or: pip3 install --user numpy pandas"
    exit 1
fi
python3 -c "import numpy,pandas;print('numpy',numpy.__version__,'pandas',pandas.__version__)"

# Record the acceptance the run used.  The per-particle front-face test replaced
# the old polar-angle band, and the log should say which geometry produced the
# counters so a later reader cannot confuse the two.
if [ ! -f "$SUBMIT_DIR/scripts/hgnd_acceptance.py" ]; then
    echo "ERROR: scripts/hgnd_acceptance.py must be copied alongside reduce_prim_full.py"
    exit 1
fi
echo "== HGND acceptance in use =="
python3 "$SUBMIT_DIR/scripts/hgnd_acceptance.py"

for tag in zeroSpot bigSpot defaultSpot; do
    if [ ! -d "$DATA_ROOT/smash_xecs_2.87gev_hardSkyrme_$tag" ]; then
        echo "skip $tag: no extracted directory under $DATA_ROOT"
        continue
    fi
done

python3 -u scripts/reduce_prim_full.py "$OUTDIR" zeroSpot bigSpot defaultSpot

echo "== outputs =="
ls -la "$OUTDIR"
echo "copy *_events.pkl and *_filehist.npz back to the laptop; they are ~40 MB per sample"
echo
echo "The event tables now carry acc_* counters (per-particle front-face acceptance)"
echo "alongside the legacy band_* ones, and each job file contributes an 'acc'"
echo "spectrum so R can be recomputed at any threshold without another pass."
