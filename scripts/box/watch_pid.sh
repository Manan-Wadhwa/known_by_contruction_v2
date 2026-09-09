#!/usr/bin/env bash
# watch_pid.sh <box> <label> <remote_outdir>: every CYCLE s tail /marimo/repo/<label>.log; when <label>.pid is gone,
# pull the small files (*.json, *.txt, *.log <= 2 MB) from <remote_outdir> into results/<basename> and exit 0.
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding boxN.url and boxN.token}"
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.claude/skills/marimo-pair/scripts/execute-code.sh"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BOX=$1; LABEL=$2; RDIR=$3; CYCLE=${CYCLE:-600}; MAX=${MAX:-42}; OUT=$S/watch_${LABEL}.log
say() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$OUT"; }
probe() { bash "$EX" --url "$(cat $S/$BOX.url)" --token "$(cat $S/$BOX.token)" -c "import subprocess,os
pid=open('/marimo/repo/${LABEL}.pid').read().strip() if os.path.exists('/marimo/repo/${LABEL}.pid') else ''
alive=subprocess.run(['kill','-0',pid],capture_output=True).returncode==0 if pid else False
print('ALIVE',alive); print(subprocess.run('ls ${RDIR} 2>/dev/null | tr \"\\n\" \" \"; echo; grep -vE \"^\\s*$|Warning|warnings.warn|it/s\" /marimo/repo/${LABEL}.log | tail -1 | cut -c1-220',shell=True,capture_output=True,text=True).stdout)" 2>/dev/null; }
pull_small() { bash "$EX" --url "$(cat $S/$BOX.url)" --token "$(cat $S/$BOX.token)" - <<PY 2>/dev/null > $S/.wp_$LABEL.out
import base64, hashlib, json, os, glob
out = []
for p in sorted(glob.glob("${RDIR}/**/*", recursive=True)) + ["/marimo/repo/${LABEL}.log"]:
    if os.path.isfile(p) and p.split(".")[-1] in ("json","txt","log","md") and os.path.getsize(p) <= 2_000_000:
        b = open(p, "rb").read(); out.append({"rel": os.path.relpath(p, "${RDIR}") if p.startswith("${RDIR}") else os.path.basename(p), "sha": hashlib.sha256(b).hexdigest(), "b64": base64.b64encode(b).decode()})
print("PULLJSON" + json.dumps(out))
PY
python3 - "$WT/experiments/v2/MZ01_maze_world/results/$(basename $RDIR)" "$S/.wp_$LABEL.out" <<'PY'
import base64, hashlib, json, sys, os
dest, src = sys.argv[1], sys.argv[2]; txt = open(src).read(); i = txt.find("PULLJSON")
if i < 0: print("PULL FAILED"); sys.exit(1)
n = 0
for it in json.loads(txt[i+8:].strip().splitlines()[0]):
    b = base64.b64decode(it["b64"]); assert hashlib.sha256(b).hexdigest() == it["sha"]
    p = os.path.join(dest, it["rel"]); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(b); n += 1
print("PULL_OK", n, "files ->", dest)
PY
}
say "watch_pid start $BOX $LABEL $RDIR"; DEAD=""
for i in $(seq 1 $MAX); do
  P=$(probe) || true
  if [ -z "$P" ]; then say "cycle $i: unreachable"; [ -n "$DEAD" ] && { say "unreachable twice"; exit 3; }; DEAD=1
  else DEAD=""; ALIVE=$(printf '%s\n' "$P" | sed -n 's/^ALIVE //p'); say "cycle $i: alive=$ALIVE $(printf '%s\n' "$P" | sed '1d' | tr '\n' '|' | cut -c1-260)"
    if [ "$ALIVE" != "True" ]; then R=$(pull_small 2>&1 | tail -1); say "pid gone: $R"; exit 0; fi
  fi
  sleep $CYCLE
done; say "cycle cap"; exit 4
