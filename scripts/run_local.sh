#!/usr/bin/env bash
# Local convenience wrapper — dispatches to the right CLI, picks the
# CUDA→MPS→CPU device automatically, and keeps directory conventions
# consistent with the SLURM templates.
#
# Usage:
#   ./scripts/run_local.sh preprocess   defaultSpot [--max-events 5000]
#   ./scripts/run_local.sh train        defaultSpot net_default [extra --args]
#   ./scripts/run_local.sh evaluate     defaultSpot net_default
#   ./scripts/run_local.sh sensitivity  defaultSpot [zeroSpot bigSpot ...]

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 {preprocess|train|evaluate|sensitivity} <args>" >&2
    exit 1
fi
mode="$1"; shift

# Repo root = parent of this script's directory.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO}/..:${PYTHONPATH:-}"

case "$mode" in
    preprocess)
        dataset="${1:?dataset name required (e.g. defaultSpot)}"; shift || true
        csv_dir="${REPO}/data/smash_xecs_2.87gev_hardSkyrme_${dataset}"
        root="${REPO}/notebooks/cache/ndet_dataset_smash_${dataset}"
        exec python -m HGNDRecoGNN.scripts.preprocess \
            --csv-dir "$csv_dir" --root "$root" "$@"
        ;;
    train)
        dataset="${1:?dataset name required}"; shift
        model="${1:?model name required (e.g. net_default)}"; shift || true
        root="${REPO}/notebooks/cache/ndet_dataset_smash_${dataset}"
        ckpt_dir="${REPO}/checkpoints/${dataset}_${model}"
        exec python -m HGNDRecoGNN.scripts.train \
            --root "$root" --model "$model" \
            --device auto \
            --checkpoint-dir "$ckpt_dir" --checkpoint-name model.pt "$@"
        ;;
    evaluate)
        dataset="${1:?dataset name required}"; shift
        model="${1:?model name required}"; shift || true
        root="${REPO}/notebooks/cache/ndet_dataset_smash_${dataset}"
        ckpt="${REPO}/checkpoints/${dataset}_${model}/model.pt"
        out_dir="${REPO}/results/${dataset}_${model}"
        exec python -m HGNDRecoGNN.scripts.evaluate \
            --root "$root" --checkpoint "$ckpt" \
            --out-dir "$out_dir" --suffix smash --device auto "$@"
        ;;
    sensitivity)
        model="${1:-net_default}"; shift || true
        args=()
        for ds in "${@:-defaultSpot}"; do
            pkl="${REPO}/results/${ds}_${model}/pred_clusters_smash.pkl"
            args+=(--dataset "${ds}=${pkl}")
        done
        out_dir="${REPO}/results/sensitivity_${model}"
        exec python -m HGNDRecoGNN.scripts.sensitivity \
            "${args[@]}" --out-dir "$out_dir"
        ;;
    *)
        echo "unknown mode: $mode" >&2
        echo "modes: preprocess | train | evaluate | sensitivity" >&2
        exit 1
        ;;
esac
