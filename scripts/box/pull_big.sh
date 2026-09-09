#!/usr/bin/env bash
# pull_big.sh <box> <remote_path> <local_path> [chunk_bytes]  -- chunked base64 pull, sha256-verified
set -euo pipefail
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding boxN.url and boxN.token}"
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.claude/skills/marimo-pair/scripts/execute-code.sh"
BOX=$1; REMOTE=$2; LOCAL=$3; CH=${4:-524288}
URL=$(cat $S/$BOX.url); TOK=$(cat $S/$BOX.token)
META=$(bash "$EX" --url "$URL" --token "$TOK" -c "import os,hashlib; b=open('$REMOTE','rb').read(); print('META', len(b), hashlib.sha256(b).hexdigest())" 2>/dev/null | grep '^META')
SIZE=$(echo $META | cut -d' ' -f2); SHA=$(echo $META | cut -d' ' -f3)
: > "$LOCAL.part"; OFF=0
while [ $OFF -lt $SIZE ]; do
  bash "$EX" --url "$URL" --token "$TOK" -c "import base64; f=open('$REMOTE','rb'); f.seek($OFF); print('CHUNK'+base64.b64encode(f.read($CH)).decode())" 2>/dev/null | grep '^CHUNK' | cut -c6- | base64 -d >> "$LOCAL.part"
  OFF=$((OFF+CH)); printf '.'
done; echo
GOT=$(sha256sum "$LOCAL.part" | cut -d' ' -f1)
[ "$GOT" = "$SHA" ] && mv "$LOCAL.part" "$LOCAL" && echo "PULL_OK $LOCAL $SIZE B sha $(echo $SHA | cut -c1-16)" || { echo "SHA MISMATCH ($(stat -c %s $LOCAL.part) B)"; exit 1; }
