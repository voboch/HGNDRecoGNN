#!/bin/bash
# run_sensitivity_fixed.sh — RUN ON A cHARISMa LOGIN NODE (login-02).
#
# Submits the full sensitivity experiment on the 2026-09 re-simulated
# ("bug-fixed") SMASH archives as one dependency chain:
#
#   extract  (array 0-2, CPU)  -> data_fixed/<dataset>/
#        |
#   preprocess (array 0-2, CPU, TAG=v2fix, no event cap)
#        |                        -> cache/ndet_dataset_smash_<ds>_v2fix/
#   train    (V100, val-loss policy, on fixed defaultSpot)
#        |                        -> checkpoints/defaultSpot_v2fix_..._valloss_<jid>/
#   sensitivity (V100, all three fixed datasets vs the fixed checkpoint)
#                                 -> results/sensitivity_full_hpc_v2fix_<jid>/
#
# A retrain is part of the chain on purpose: the fixed simulation has ~31 %
# fewer hits per event and a markedly harder neutron spectrum, so the
# standing checkpoint (trained on pre-bugfix defaultSpot) is out of
# distribution and its reconstruction efficiency would not transfer.
#
# Usage:
#   bash slurm/run_sensitivity_fixed.sh            # full chain
#   SKIP_EXTRACT=1 bash slurm/run_sensitivity_fixed.sh   # archives already unpacked
#   SKIP_TRAIN=1 CKPT=/path/model.pt bash slurm/run_sensitivity_fixed.sh
#   TAG=v3 SKIP_EXTRACT=1 bash slurm/run_sensitivity_fixed.sh   # rebuild caches
#   EXCLUDE= bash slurm/run_sensitivity_fixed.sh                # allow all nodes
set -euo pipefail

: "${SCRATCH:=/scratch/$USER}"
: "${TAG:=v2fix}"
: "${DATA_ROOT:=$SCRATCH/hgnd/data_fixed}"
: "${SKIP_EXTRACT:=0}"
: "${SKIP_TRAIN:=0}"
# Comma-separated nodes to keep every stage off. cn-030 produced two distinct
# filesystem errors that killed two preprocess jobs on 2026-09-15 while the
# same work ran for hours on cn-031; Slurm still lists it healthy.
: "${EXCLUDE:=cn-030}"

EXC=()
if [[ -n "$EXCLUDE" ]]; then
    EXC+=(--exclude="$EXCLUDE")
fi

cd "$(dirname "$0")/.."
mkdir -p logs

DEP=""
submit () {  # submit <name> <extra-sbatch-args...> -- returns job id
    local jid
    jid=$(sbatch --parsable "$@")
    echo "$jid"
}

# ── 1) extract ───────────────────────────────────────────────────────────
if [[ "$SKIP_EXTRACT" != "1" ]]; then
    JID_EX=$(submit --array=0-2 "${EXC[@]}" slurm/extract_fixed.hpc.sbatch)
    echo "extract    : array job $JID_EX"
    DEP="--dependency=afterok:$JID_EX"
fi

# ── 2) preprocess to v2fix caches ────────────────────────────────────────
JID_PP=$(submit --array=0-2 $DEP "${EXC[@]}" \
    --export=ALL,DATA_ROOT="$DATA_ROOT",TAG="$TAG",MAX_EVENTS= \
    slurm/preprocess_all_smash.hpc.sbatch)
echo "preprocess : array job $JID_PP  (DATA_ROOT=$DATA_ROOT TAG=$TAG, no event cap)"

# ── 3) train on the fixed defaultSpot cache ──────────────────────────────
if [[ "$SKIP_TRAIN" != "1" ]]; then
    JID_TR=$(submit --dependency=afterok:$JID_PP "${EXC[@]}" \
        --export=ALL,TAG="$TAG",TRAIN_DS=defaultSpot \
        slurm/train_valloss.hpc.sbatch)
    echo "train      : job $JID_TR  (net_default H=512 L=8, 20 ep, save-best-on=val)"
    CKPT="$SCRATCH/hgnd/checkpoints/defaultSpot_${TAG}_seed42_20ep_valloss_${JID_TR}/model.pt"
    SENS_DEP="--dependency=afterok:$JID_TR"
else
    : "${CKPT:?SKIP_TRAIN=1 requires CKPT=/path/to/model.pt}"
    SENS_DEP="--dependency=afterok:$JID_PP"
fi

# ── 4) sensitivity on all three fixed datasets ───────────────────────────
JID_SE=$(submit $SENS_DEP "${EXC[@]}" \
    --export=ALL,TAG="$TAG",RUN_LABEL="$TAG",CKPT="$CKPT" \
    slurm/sensitivity_full.hpc.sbatch)
echo "sensitivity: job $JID_SE"

cat <<SUMMARY

chain submitted.
  checkpoint  : $CKPT
  results      : $SCRATCH/hgnd/results/sensitivity_full_hpc_${TAG}_${JID_SE}/

monitor:  slurm/hpcjob.sh queue
          slurm/hpcjob.sh watch $JID_SE
retrieve: rsync -avhP --exclude='pred_hits*.pkl' --exclude='pred_edges*.pkl' \\
            charisma:$SCRATCH/hgnd/results/sensitivity_full_hpc_${TAG}_${JID_SE} \\
            ~/Project/BM@N/HGND/HGNDRecoGNN/results/
SUMMARY
