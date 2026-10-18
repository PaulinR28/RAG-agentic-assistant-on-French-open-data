#!/usr/bin/env bash
# Resumable download of a pinned Hugging Face model revision into models/<alias>/ (git-ignored).
# usage: get_model.sh <alias> <repo> <revision> <file> [<file> ...]
set -u
ALIAS=$1; REPO=$2; REV=$3; shift 3
DEST="models/$ALIAS"
BASE="https://huggingface.co/$REPO/resolve/$REV"
for f in "$@"; do
  mkdir -p "$DEST/$(dirname "$f")"
  for attempt in $(seq 1 50); do
    if curl -sS -L -C - --retry 5 --retry-all-errors --connect-timeout 30 -o "$DEST/$f" "$BASE/$f"; then
      echo "ok $f $(stat -c %s "$DEST/$f") bytes"
      break
    fi
    echo "retry $attempt for $f"
    sleep 5
  done
done
echo "$REV" > "$DEST/REVISION"
echo "ALL_DONE $ALIAS"
