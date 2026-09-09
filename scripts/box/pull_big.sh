#!/usr/bin/env bash
# pull_big.sh <box> <remote_path> <local_path> [chunk_bytes]
# Chunked base64 pull over the exec channel. Every chunk is sha256-checked on arrival and the chunk
# size halves on any failure (the channel truncates replies above ~1 MB of base64); the whole file is
# sha256-checked at the end. Resumable: a partial .part is continued, not restarted.
set -uo pipefail
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding boxN.url and boxN.token}"
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.claude/skills/marimo-pair/scripts/execute-code.sh"
BOX=$1; REMOTE=$2; LOCAL=$3; CH=${4:-262144}   # ~50 KB/s at 64-256 KB; the channel collapses above ~512 KB of base64
URL=$(cat $S/$BOX.url); TOK=$(cat $S/$BOX.token)
META=$(bash "$EX" --url "$URL" --token "$TOK" -c "import os,hashlib; b=open('$REMOTE','rb').read(); print('META', len(b), hashlib.sha256(b).hexdigest())" 2>/dev/null | grep '^META') || true
[ -n "$META" ] || { echo "PULL FAILED: no META for $REMOTE"; exit 1; }
SIZE=$(echo $META | cut -d' ' -f2); SHA=$(echo $META | cut -d' ' -f3)
# One puller per target file: two concurrent pulls appending to one .part corrupt it (2026-09-09,
# orphaned child loops of a killed watcher). flock on a lock file beside the target; a second caller exits 5.
exec 9>"$LOCAL.lock"; flock -n 9 || { echo "PULL BUSY: another pull holds $LOCAL.lock"; exit 5; }
OFF=0; if [ -f "$LOCAL.part" ]; then OFF=$(stat -c %s "$LOCAL.part"); else : > "$LOCAL.part"; fi
[ $OFF -gt $SIZE ] && { : > "$LOCAL.part"; OFF=0; }
while [ $OFF -lt $SIZE ]; do
  OUT=$(bash "$EX" --url "$URL" --token "$TOK" -c "import base64,hashlib; f=open('$REMOTE','rb'); f.seek($OFF); b=f.read($CH); print('CHUNK', len(b), hashlib.sha256(b).hexdigest(), base64.b64encode(b).decode())" 2>/dev/null | grep '^CHUNK ')
  N=$(echo "$OUT" | cut -d' ' -f2); CS=$(echo "$OUT" | cut -d' ' -f3); B64=$(echo "$OUT" | cut -d' ' -f4)
  if [ -n "$B64" ] && printf '%s' "$B64" | base64 -d > "$LOCAL.chunk" 2>/dev/null && [ "$(stat -c %s "$LOCAL.chunk")" = "$N" ] && [ "$(sha256sum "$LOCAL.chunk" | cut -d' ' -f1)" = "$CS" ]; then
    cat "$LOCAL.chunk" >> "$LOCAL.part"; OFF=$((OFF+N)); printf '.'
  else
    CH=$((CH/2)); printf 'x'; [ $CH -lt 32768 ] && { echo; echo "PULL FAILED: chunk size floor at offset $OFF"; rm -f "$LOCAL.chunk"; exit 1; }
  fi
done; rm -f "$LOCAL.chunk"; echo
GOT=$(sha256sum "$LOCAL.part" | cut -d' ' -f1)
[ "$GOT" = "$SHA" ] && mv "$LOCAL.part" "$LOCAL" && rm -f "$LOCAL.lock" && echo "PULL_OK $LOCAL $SIZE B sha $(echo $SHA | cut -c1-16)" || { echo "SHA MISMATCH ($(stat -c %s $LOCAL.part) B) -- .part discarded (the source may have been rewritten mid-pull)"; rm -f "$LOCAL.part"; exit 1; }
