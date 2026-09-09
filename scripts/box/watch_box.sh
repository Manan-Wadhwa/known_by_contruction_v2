#!/usr/bin/env bash
# watch_box.sh <box> <label> <remote_rundir> <trial|driver> : ONE watcher per box.
# Every CYCLE s: liveness off /marimo/repo/<label>.pid, the log's last line, the small files of the run
# (trial: every cycle; driver: every cycle too -- rows appear only at the end, JSON/txt/log are cheap), and
# every *.pt under <remote_rundir> that is new or changed (chunked, locked, sha-verified). Exits 0 when the
# pid is gone (after a final pull), 3 when the box is unreachable twice, 4 at MAX cycles.
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding boxN.url and boxN.token}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; EX="$WT/.claude/skills/marimo-pair/scripts/execute-code.sh"; HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOX=$1; LABEL=$2; RDIR=$3; MODE=${4:-trial}; CYCLE=${CYCLE:-900}; MAX=${MAX:-48}
DEST="$WT/experiments/v2/MZ01_maze_world/results/$(basename $RDIR)"; OUT="$S/watch_${LABEL}.log"; SEEN="$S/.seen_${LABEL}"
say() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$OUT"; }; mkdir -p "$DEST"; touch "$SEEN"
probe() { bash "$EX" --url "$(cat $S/$BOX.url)" --token "$(cat $S/$BOX.token)" -c "import subprocess,os
pid=open('/marimo/repo/${LABEL}.pid').read().strip() if os.path.exists('/marimo/repo/${LABEL}.pid') else ''
def _alive(p):
    # a finished job can linger as a ZOMBIE on the box (nothing reaps a setsid'd
    # child there), and kill -0 says yes to a zombie; read the state instead
    try: return open(f'/proc/{p}/stat').read().split(') ')[1].split()[0] not in ('Z','X')
    except Exception: return False
alive=_alive(pid) if pid else False
print('ALIVE',alive)
for root,_,files in os.walk('$RDIR'):
    for f in files:
        if f.endswith('.pt'): p=os.path.join(root,f); print('PT', os.path.relpath(p,'$RDIR'), os.path.getsize(p), int(os.path.getmtime(p)))
print('LAST', subprocess.run('grep -vE \"^\\s*$|Warning|warnings.warn|it/s\" /marimo/repo/${LABEL}.log | tail -1 | cut -c1-200',shell=True,capture_output=True,text=True).stdout.strip())" 2>/dev/null; }
pull_small() { bash "$EX" --url "$(cat $S/$BOX.url)" --token "$(cat $S/$BOX.token)" - <<PY 2>/dev/null > $S/.wb_$LABEL.out
import base64, hashlib, json, os, glob
out = []
for p in sorted(glob.glob("${RDIR}/**/*", recursive=True)) + ["/marimo/repo/${LABEL}.log"]:
    if os.path.isfile(p) and p.rsplit(".",1)[-1] in ("json","txt","log","md") and os.path.getsize(p) <= 2_000_000:
        b = open(p, "rb").read(); out.append({"rel": os.path.relpath(p, "${RDIR}") if p.startswith("${RDIR}") else os.path.basename(p), "sha": hashlib.sha256(b).hexdigest(), "b64": base64.b64encode(b).decode()})
print("PULLJSON" + json.dumps(out))
PY
python3 - "$DEST" "$S/.wb_$LABEL.out" <<'PY'
import base64, hashlib, json, sys, os
dest, src = sys.argv[1], sys.argv[2]; txt = open(src).read(); i = txt.find("PULLJSON")
if i < 0: print("small: FAILED"); sys.exit(1)
n = 0
for it in json.loads(txt[i+8:].strip().splitlines()[0]):
    b = base64.b64decode(it["b64"]); assert hashlib.sha256(b).hexdigest() == it["sha"]
    p = os.path.join(dest, it["rel"]); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(b); n += 1
print(f"small: {n} files")
PY
}
say "watch_box start $BOX $LABEL $RDIR $MODE"; DEAD=""
for i in $(seq 1 $MAX); do
  P=$(probe) || true
  if [ -z "$P" ]; then say "cycle $i: unreachable"; [ -n "$DEAD" ] && { say "unreachable twice"; exit 3; }; DEAD=1; sleep $CYCLE; continue; fi
  DEAD=""; ALIVE=$(printf '%s\n' "$P" | sed -n 's/^ALIVE //p'); LAST=$(printf '%s\n' "$P" | sed -n 's/^LAST //p' | cut -c1-160)
  SM=$(pull_small 2>&1 | tail -1); NP=0
  while read -r _ rel size mtime; do
    [ -z "$rel" ] && continue; key="$rel $size $mtime"
    if ! grep -qxF "$key" "$SEEN"; then mkdir -p "$DEST/$(dirname $rel)"; if bash "$HERE/pull_big.sh" $BOX "$RDIR/$rel" "$DEST/$rel" >> "$OUT" 2>&1; then echo "$key" >> "$SEEN"; NP=$((NP+1)); else say "adapter pull failed: $rel"; fi; fi
  done <<< "$(printf '%s\n' "$P" | grep '^PT ')"
  say "cycle $i: alive=$ALIVE $SM adapters+$NP | $LAST"
  if [ "$ALIVE" != "True" ]; then
    # the run ended between probes: one more sweep so anything written in its
    # last seconds (the summary, the final adapter) is on this machine
    SM=$(pull_small 2>&1 | tail -1); NP2=0
    while read -r _ rel size mtime; do
      [ -z "$rel" ] && continue; key="$rel $size $mtime"
      if ! grep -qxF "$key" "$SEEN"; then mkdir -p "$DEST/$(dirname $rel)"; bash "$HERE/pull_big.sh" $BOX "$RDIR/$rel" "$DEST/$rel" >> "$OUT" 2>&1 && { echo "$key" >> "$SEEN"; NP2=$((NP2+1)); }; fi
    done <<< "$(probe | grep '^PT ')"
    say "pid gone; final sweep: $SM adapters+$NP2; last: $LAST; exit"; exit 0
  fi
  sleep $CYCLE
done; say "cycle cap"; exit 4
