#!/bin/bash
# setup_env.sh — build a conda env on the Rocky 9 login node (login-02).
#
# Usage (ON login-02, NOT on a compute node, NOT on sms):
#   bash slurm/setup_env.sh <env-name> [py-version] [requirements-file]
# Example:
#   bash slurm/setup_env.sh hgnd-env 3.11 requirements.txt
#
# Guard rails: refuses to run on a compute node or the CentOS 7 login node,
# because a conda env built on the wrong OS fails with glibc errors elsewhere.
set -euo pipefail

ENV_NAME="${1:?usage: setup_env.sh <env-name> [py-version] [requirements-file]}"
PY_VERSION="${2:-3.11}"
REQ_FILE="${3:-}"

# ── Guard rails ─────────────────────────────────────────────────────────
host="$(hostname)"
if [[ "$host" == "sms" ]]; then
    echo "ERROR: this is the CentOS 7 login node. Run on login-02 (Rocky 9): ssh -A login-02" >&2
    exit 1
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: never build envs inside a Slurm job (no internet on compute nodes)." >&2
    exit 1
fi
if ! grep -q "Rocky" /etc/os-release 2>/dev/null; then
    echo "WARNING: /etc/os-release is not Rocky Linux; env may not match the rocky partition." >&2
fi

# ── Modules ─────────────────────────────────────────────────────────────
source /etc/profile 2>/dev/null || true
module purge 2>/dev/null || true
module load python/miniconda
module load cuda/12.9.1

echo ">> building conda env '$ENV_NAME' (python $PY_VERSION)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">> env '$ENV_NAME' already exists — reusing. Delete with: conda env remove -n $ENV_NAME"
else
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi

# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

python -m pip install --upgrade pip

if [[ -n "$REQ_FILE" && -f "$REQ_FILE" ]]; then
    echo ">> installing from $REQ_FILE"
    python -m pip install -r "$REQ_FILE"
else
    echo ">> NOTE: no requirements file given. Install torch matching CUDA 12.x, e.g.:"
    echo "     python -m pip install torch --index-url https://download.pytorch.org/whl/cu121"
fi

echo ">> pre-create the HF cache on scratch (compute nodes have no internet):"
mkdir -p "/scratch/$USER/hf_cache"
echo "     export HF_HOME=/scratch/$USER/hf_cache   (set this in your sbatch)"

echo ">> done. Verify GPU visibility on a test-partition shell:"
echo "     srun --pty --partition=test --account=proj_1855 --constraint=type_a --gpus=1 --time=00:20:00 bash"
echo "     python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
