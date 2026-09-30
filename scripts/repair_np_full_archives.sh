#!/bin/zsh

set -u

LOG=/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/logs/np_full_repair.log
DEST=charisma:/scratch/vbocharnikov/hgnd/data_np_full_20260924/archives/
SOURCE=/Users/vovvy/ncx/data

mkdir -p "${LOG:h}"
exec >>"$LOG" 2>&1

repair() {
  local archive=$1
  while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') whole-file repair starting $archive"
    if rsync -ahP --ignore-times --whole-file --partial-dir=.np-ratio-repair-partial \
      --timeout=600 \
      -e "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=20" \
      "$SOURCE/$archive" "$DEST"; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') whole-file repair completed $archive"
      return 0
    else
      rc=$?
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') retrying $archive after rsync status $rc"
    sleep 60
  done
}

archives=("$@")
if (( ${#archives[@]} == 0 )); then
  archives=(
    smash_xecs_2.87gev_hardSkyrme_zeroSpot.tar.gz
    smash_xecs_2.87gev_hardSkyrme_defaultSpot.tar.gz
  )
fi

for archive in "${archives[@]}"; do
  repair "$archive"
done
echo "$(date '+%Y-%m-%d %H:%M:%S') all checksum repairs complete"
