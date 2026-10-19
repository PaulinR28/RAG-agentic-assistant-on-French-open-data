#!/usr/bin/env bash
# Resumable download of a pinned Hugging Face model revision into models/<alias>/ (git-ignored).
# usage: download_model.sh <alias> <repo> <revision> <file> [<file> ...]
# REVISION is written only when every file downloaded: the code treats models/<alias>/ as usable
# only if REVISION exists, so a failed or partial download is never loaded.
set -u
ALIAS=$1; REPO=$2; REV=$3; shift 3
DEST="models/$ALIAS"
BASE="https://huggingface.co/$REPO/resolve/$REV"
rm -f "$DEST/REVISION"
for f in "$@"; do
  mkdir -p "$DEST/$(dirname "$f")"
  ok=0
  for attempt in $(seq 1 50); do
    # --fail: an HTTP error (404 typo, 401 gated) is an error, not a saved HTML page
    curl -sS -L --fail -C - --retry 5 --retry-connrefused --connect-timeout 30 -o "$DEST/$f" "$BASE/$f"
    code=$?
    if [ "$code" -eq 0 ]; then
      echo "ok $f $(stat -c %s "$DEST/$f") bytes"
      ok=1
      break
    fi
    if [ "$code" -eq 22 ]; then  # HTTP 4xx (wrong name, gated repo): retrying will not help
      rm -f "$DEST/$f"
      break
    fi
    echo "retry $attempt for $f (curl exit $code)"
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then
    echo "FAILED $f: not marking $ALIAS as downloaded" >&2
    exit 1
  fi
done
echo "$REV" > "$DEST/REVISION"
echo "ALL_DONE $ALIAS"
