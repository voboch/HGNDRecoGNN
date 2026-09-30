#!/bin/zsh

set -u

LOG=/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/logs/np_full_rsync.log
DEST=charisma:/scratch/vbocharnikov/hgnd/data_np_full_20260924/archives/
SOURCE=/Users/vovvy/ncx/data

mkdir -p "${LOG:h}"
exec >>"$LOG" 2>&1

transfer() {
  local archive=$1
  while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') starting $archive"
    if rsync -ahP --append "$SOURCE/$archive" "$DEST"; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') completed $archive"
      return 0
    else
      rc=$?
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') retrying $archive after rsync status $rc"
    sleep 60
  done
}

transfer smash_xecs_2.87gev_hardSkyrme_zeroSpot.tar.gz
transfer smash_xecs_2.87gev_hardSkyrme_defaultSpot.tar.gz
transfer smash_xecs_2.87gev_hardSkyrme_bigSpot.tar.gz
echo "$(date '+%Y-%m-%d %H:%M:%S') all transfers complete"
