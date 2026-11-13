#!/usr/bin/env bash
set -euo pipefail
source_dir=/home/vbocharnikov/hgnd_experiments/smash_architecture_20261112_2d31e3e25c45
campaign=/scratch/vbocharnikov/hgnd/experiments/smash_architecture_20261112_2d31e3e25c45
receipts=${source_dir}_receipts
launch_plan="$receipts/fit_launch_plan.json"
review="$receipts/fit_preflight_review.md"
test "$#" -eq 1
printf '%s  %s\n' 677b8ecd5508ea558c66527d6f5d2e835cd93cf42ff82902b369818f6ddb1612 "$launch_plan" | sha256sum -c --quiet
printf '%s  %s\n' "$1" "$review" | sha256sum -c --quiet
grep -Fxq 'Вердикт: GO.' "$review"
cd "$source_dir"
printf '%s  SHA256SUMS\n' 2d31e3e25c45dff5b15818357e8d3d185cc752ef229a726971ab234f8f5d3969 | sha256sum -c --quiet
sha256sum -c --quiet SHA256SUMS
printf '%s  %s\n' b996e3d4980bd460ba12ddeafb4302674a27375c1922b8e2c274ef0103c2863a "$receipts/probe_0_4508378.json" \
                    eea87baf217e67ebd0bb95f258b874b37ea52648c0e243e91977e0b5c3769972 "$receipts/probe_9_4508255.json" | sha256sum -c --quiet
accounting=$(sacct -X -j 4508255 -n -P -o JobID,State,ExitCode)
printf '%s\n' "$accounting"
printf '%s\n' "$accounting" | grep -Fxq '4508255_0|COMPLETED|0:0'
printf '%s\n' "$accounting" | grep -Fxq '4508255_9|COMPLETED|0:0'
queued=$(squeue -h -u "$USER" -n hgnd-smash-arch,hgnd-arch-probe -o '%i')
historical=$(sacct -X -S 2026-11-12 -u "$USER" --name=hgnd-smash-arch -n -P -o JobID)
test -z "$queued"
test -z "$historical"
test ! -e "$receipts/fit_submission.tsv"
test ! -e "$campaign/mirror"
test -z "$(find "$receipts" "$campaign" -maxdepth 1 -name 'fit_*' ! -name fit_launch_plan.json ! -name fit_preflight_review.md -print -quit)"
segments=$(jq -er '.segments[] | [.start_epoch, .end_epoch, .time_limit] | @tsv' "$launch_plan")
mkdir "$receipts/fit_submission_lock"
ledger="$receipts/fit_submission.tsv"
printf 'start_epoch\tend_epoch\ttime_limit\tarray_job_id\tpredecessor\n' > "$ledger"
previous=
while IFS=$'\t' read -r start end limit; do
    dependency=()
    if [[ -n "$previous" ]]; then
        dependency=(--dependency="afterok:$previous" --kill-on-invalid-dep=yes)
    fi
    job=$(sbatch --parsable --job-name=hgnd-smash-arch --array='0-17%2' --time="$limit" \
        --chdir="$source_dir" --output="$receipts/logs/%x_%A_%a.out" "${dependency[@]}" \
        --export="ALL,ARCH_SOURCE=$source_dir,ARCH_CAMPAIGN=$campaign,ARCH_RECEIPTS=$receipts,ARCH_PHASE=fit,ARCH_START_EPOCH=$start,ARCH_END_EPOCH=$end" \
        "$source_dir/slurm/smash_architecture.hpc.sbatch")
    job=${job%%;*}
    printf '%s\t%s\t%s\t%s\t%s\n' "$start" "$end" "$limit" "$job" "$previous" | tee -a "$ledger"
    [[ "$job" =~ ^[0-9]+$ ]]
    previous=$job
done <<< "$segments"
