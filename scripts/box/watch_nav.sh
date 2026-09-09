#!/usr/bin/env bash
# watch_nav.sh <box> <shard>: every CYCLE s, tail the navigator driver's log; when the pid is gone,
# pull nav_<shard>.json (and note the .pt) and exit 0; exit 2 if it died without a JSON; 3 if unreachable twice.
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding boxN.url and boxN.token}"
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.claude/skills/marimo-pair/scripts/execute-code.sh"
BOX=$1; SHARD=$2; CYCLE=${CYCLE:-600}; MAX=${MAX:-30}; OUT=$S/watch_nav_${SHARD}.log
say() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$OUT"; }
probe() { bash "$EX" --url "$(cat $S/$BOX.url)" --token "$(cat $S/$BOX.token)" -c "import subprocess,os
pid=open('/marimo/repo/${SHARD}.pid').read().strip() if os.path.exists('/marimo/repo/${SHARD}.pid') else ''
alive=subprocess.run(['kill','-0',pid],capture_output=True).returncode==0 if pid else False
print('ALIVE',alive); print(subprocess.run('ls -la /marimo/repo/experiments/v2/MZ01_maze_world/results/navigator/ 2>/dev/null | grep ${SHARD} | tr -s \" \" | cut -d\" \" -f5,9 | tr \"\\n\" \" \"; echo; grep -vE \"^\\s*$|Warning|it/s\" /marimo/repo/${SHARD}.log | tail -2 | cut -c1-200',shell=True,capture_output=True,text=True).stdout)" 2>/dev/null; }
say "nav watcher start $BOX $SHARD"
DEAD=""
for i in $(seq 1 $MAX); do
  P=$(probe) || true
  if [ -z "$P" ]; then say "cycle $i: unreachable"; [ -n "$DEAD" ] && { say "unreachable twice"; exit 3; }; DEAD=1
  else DEAD=""; ALIVE=$(printf '%s\n' "$P" | sed -n 's/^ALIVE //p'); say "cycle $i: alive=$ALIVE $(printf '%s\n' "$P" | sed '1d' | tr '\n' ' | ' | cut -c1-300)"
    if [ "$ALIVE" != "True" ]; then
      if printf '%s' "$P" | grep -q "nav_${SHARD}.json"; then
        bash "$EX" --url "$(cat $S/$BOX.url)" --token "$(cat $S/$BOX.token)" -c "import base64,hashlib; b=open('/marimo/repo/experiments/v2/MZ01_maze_world/results/navigator/nav_${SHARD}.json','rb').read(); print('NAVJSON', hashlib.sha256(b).hexdigest(), base64.b64encode(b).decode())" 2>/dev/null | python3 -c "
import sys,base64,hashlib
for line in sys.stdin:
    if line.startswith('NAVJSON'):
        _,sha,b64=line.split(); b=base64.b64decode(b64); assert hashlib.sha256(b).hexdigest()==sha
        open('/home/manan/Projects/files/digital_minds-maze/experiments/v2/MZ01_maze_world/results/navigator/nav_${SHARD}.json','wb').write(b); print('pulled nav_${SHARD}.json', len(b), 'B', sha[:16])"
        say "navigator FINISHED; json pulled"; exit 0
      fi
      say "pid gone WITHOUT json -- died"; exit 2
    fi
  fi
  sleep $CYCLE
done
say "cycle cap"; exit 4
