#!/usr/bin/env bash
# watch_adapters.sh <box> <remote_dir> : every CYCLE s, pull every *.pt under <remote_dir> (recursively) that is new or
# changed since the last pull, chunked and sha256-verified, into results/<basename(remote_dir)>/... ; exit when the
# box is unreachable twice or after MAX cycles. The organisms are adapters; a lease death must not take them.
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding boxN.url and boxN.token}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; EX="$WT/.claude/skills/marimo-pair/scripts/execute-code.sh"
BOX=$1; RDIR=$2; CYCLE=${CYCLE:-900}; MAX=${MAX:-40}
DEST="$WT/experiments/v2/MZ01_maze_world/results/$(basename $RDIR)"; OUT="$S/watch_adapters_$(basename $RDIR).log"; SEEN="$S/.adapters_seen_$(basename $RDIR)"
say() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$OUT"; }; touch "$SEEN"; mkdir -p "$DEST"
say "start $BOX $RDIR"; DEAD=""
for i in $(seq 1 $MAX); do
  L=$(bash "$EX" --url "$(cat $S/$BOX.url)" --token "$(cat $S/$BOX.token)" -c "import os,hashlib
for root,_,files in os.walk('$RDIR'):
    for f in files:
        if f.endswith('.pt'):
            p=os.path.join(root,f); print('PT', os.path.relpath(p,'$RDIR'), os.path.getsize(p), int(os.path.getmtime(p)))" 2>/dev/null | grep '^PT ')
  if [ -z "$L" ]; then
    R=$(bash "$EX" --url "$(cat $S/$BOX.url)" --token "$(cat $S/$BOX.token)" -c "print('OK')" 2>/dev/null | grep -c OK)
    if [ "$R" = "0" ]; then say "cycle $i: unreachable"; [ -n "$DEAD" ] && { say "unreachable twice"; exit 3; }; DEAD=1; else DEAD=""; say "cycle $i: no adapters yet"; fi
  else
    DEAD=""; n=0
    while read -r _ rel size mtime; do
      key="$rel $size $mtime"
      if ! grep -qxF "$key" "$SEEN"; then
        mkdir -p "$DEST/$(dirname $rel)"
        if bash "$(dirname "${BASH_SOURCE[0]}")/pull_big.sh" $BOX "$RDIR/$rel" "$DEST/$rel" >> "$OUT" 2>&1; then echo "$key" >> "$SEEN"; n=$((n+1)); else say "pull FAILED $rel"; fi
      fi
    done <<< "$L"
    say "cycle $i: $(echo "$L" | wc -l) adapters on box, $n pulled this cycle"
  fi
  sleep $CYCLE
done; say "cycle cap"; exit 4
