#!/bin/bash
# hpcjob.sh — thin wrapper around Slurm for the common actions.
# Run on a login node (login-02 preferred). Never runs training itself.
#
#   hpcjob.sh submit <script.sbatch> [extra sbatch args]
#   hpcjob.sh watch  <jobid>        # tail -f the job's .out log
#   hpcjob.sh queue                 # my jobs
#   hpcjob.sh start  <jobid>        # estimated start time
#   hpcjob.sh gpu    <jobid>        # live nvidia-smi on the job's node
#   hpcjob.sh stats  <jobid>        # sacct summary (elapsed, state, maxrss)
#   hpcjob.sh shell  [partition]    # interactive GPU shell (default: test)
#   hpcjob.sh kill   <jobid>
set -euo pipefail

ACCOUNT="${SLURM_ACCOUNT:-proj_1855}"
CONSTRAINT="${SLURM_CONSTRAINT:-type_a}"
cmd="${1:-}"; shift || true

case "$cmd" in
  submit)
    script="${1:?usage: hpcjob.sh submit <script.sbatch>}"; shift || true
    mkdir -p logs
    sbatch --account="$ACCOUNT" "$@" "$script"
    ;;
  watch)
    jid="${1:?usage: hpcjob.sh watch <jobid>}"
    f=$(ls -t logs/*_"${jid}".out 2>/dev/null | head -1 || true)
    [[ -z "$f" ]] && f=$(scontrol show job "$jid" | sed -n 's/.*StdOut=//p' | head -1)
    echo ">> tailing $f (Ctrl-C to stop)"; tail -f "$f"
    ;;
  queue) squeue -u "$USER" -o '%.10i %.12P %.20j %.2t %.10M %.6D %R' ;;
  start) squeue -u "$USER" --start ;;
  gpu)
    jid="${1:?usage: hpcjob.sh gpu <jobid>}"
    node=$(squeue -j "$jid" -h -o '%N'); echo ">> node: $node"
    srun --jobid="$jid" --overlap nvidia-smi
    ;;
  stats)
    jid="${1:?usage: hpcjob.sh stats <jobid>}"
    sacct -j "$jid" --format=JobID,JobName,State,Elapsed,AllocTRES%40,MaxRSS,MaxVMSize
    ;;
  shell)
    part="${1:-test}"
    exec srun --pty --partition="$part" --account="$ACCOUNT" \
         --constraint="$CONSTRAINT" --gpus=1 --cpus-per-task=4 --time=00:20:00 bash
    ;;
  kill) scancel "${1:?usage: hpcjob.sh kill <jobid>}" ;;
  *)
    grep -E '^#( |$)|hpcjob.sh ' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
