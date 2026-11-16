#!/usr/bin/env bash
set -euo pipefail
date -Is
squeue -h -u vbocharnikov -o "%.20i %.20j %.12T %R"
if squeue -h -u vbocharnikov -o "%j" | grep -q "hgnd-arch"; then echo "Existing architecture jobs; stop" >&2; exit 3; fi
test ! -e /home/vbocharnikov/hgnd_experiments/smash_architecture_20261112_2d31e3e25c45_receipts/submission_score_even_20261116_corrected_path.txt
if compgen -G '/home/vbocharnikov/hgnd_experiments/smash_architecture_20261112_2d31e3e25c45_receipts/score_even_*.json' > /dev/null; then echo "Score receipts already exist; stop" >&2; exit 4; fi
export ARCH_SOURCE=/home/vbocharnikov/hgnd_experiments/smash_architecture_20261112_2d31e3e25c45
export ARCH_CAMPAIGN=/scratch/vbocharnikov/hgnd/experiments/smash_architecture_20261112_2d31e3e25c45
export ARCH_RECEIPTS=/home/vbocharnikov/hgnd_experiments/smash_architecture_20261112_2d31e3e25c45_receipts
export ARCH_PHASE=score
export ARCH_TEST_SCOPE=even
export ARCH_FIT_INDEX=/home/vbocharnikov/hgnd_experiments/smash_architecture_20261112_2d31e3e25c45_receipts/fit_index_20261116.json
export ARCH_FIT_INDEX_SHA256=8ee55934884723334caa739c5332ae810b42f84cd5b72d0f0d853467b01bce2d
cd /home/vbocharnikov/hgnd_experiments/smash_architecture_20261112_2d31e3e25c45
sha256sum -c --quiet SHA256SUMS
printf "%s  %s\n" "$ARCH_FIT_INDEX_SHA256" "$ARCH_FIT_INDEX" | sha256sum -c --quiet
sbatch --parsable --array=0-17%2 --time=00:45:00 --job-name=hgnd-arch-even --output=/home/vbocharnikov/hgnd_experiments/smash_architecture_20261112_2d31e3e25c45_receipts/score_even_%A_%a.out slurm/smash_architecture.hpc.sbatch > /home/vbocharnikov/hgnd_experiments/smash_architecture_20261112_2d31e3e25c45_receipts/submission_score_even_20261116_corrected_path.txt
cat /home/vbocharnikov/hgnd_experiments/smash_architecture_20261112_2d31e3e25c45_receipts/submission_score_even_20261116_corrected_path.txt
