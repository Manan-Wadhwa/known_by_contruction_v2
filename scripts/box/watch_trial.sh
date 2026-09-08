#!/usr/bin/env bash
# Watch one trial (by label) on a box: every CYCLE seconds pull once, mirror the
# trial's own log, and stop when the trial finishes, dies, or the cap is hit.
# Bounded on purpose (agent hygiene): MAX cycles, and liveness is read off the
# pid, not off a marker that might never be written.
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding ${BOX:-box1}.url and ${BOX:-box1}.token}"
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.claude/skills/marimo-pair/scripts/execute-code.sh"
LABEL=${1:?usage: watch_trial.sh <label> (e.g. exitcap_cellA_s0)}; CYCLE=${CYCLE:-600}; MAX=${MAX:-42}
OUT=$S/watch_${LABEL}.log; LOGMIRROR=$S/${LABEL}.box.log
say() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$OUT"; }
probe() {
  bash "$EX" --url "$(cat $S/${BOX:-box1}.url)" --token "$(cat $S/${BOX:-box1}.token)" -c "import subprocess,os
pid=open('/marimo/repo/${LABEL}.pid').read().strip() if os.path.exists('/marimo/repo/${LABEL}.pid') else ''
alive=subprocess.run(['kill','-0',pid],capture_output=True).returncode==0 if pid else False
r=subprocess.run('ls /marimo/repo/experiments/v2/MZ01_maze_world/results/${LABEL} 2>/dev/null | tr \"\\n\" \" \"; echo; tail -c 4000 /marimo/repo/${LABEL}.log',shell=True,capture_output=True,text=True).stdout
print('ALIVE', alive); print(r)" 2>/dev/null
}
say "watcher start cycle=${CYCLE}s max=${MAX}"
for i in $(seq 1 $MAX); do
  P=$(probe) || true
  if [ -z "$P" ]; then say "cycle $i: box unreachable"; if [ "$i" -gt 1 ] && [ -n "$LASTDEAD" ]; then say "unreachable twice -- box dead?"; exit 3; fi; LASTDEAD=1
  else
    LASTDEAD=""
    ALIVE=$(printf '%s\n' "$P" | sed -n 's/^ALIVE //p')
    FILES=$(printf '%s\n' "$P" | sed -n '2p')
    printf '%s\n' "$P" | sed '1,2d' > "$LOGMIRROR"
    LAST=$(grep -v '^$' "$LOGMIRROR" | tail -1 | cut -c1-160)
    say "cycle $i: alive=$ALIVE files=[$FILES] last=[$LAST]"
    timeout 600 bash "$(dirname "${BASH_SOURCE[0]}")/pull_trial.sh" ${LABEL} >> $S/pull_${LABEL}.log 2>&1 || say "cycle $i: pull failed (exit $?)"
    if [ "$ALIVE" != "True" ]; then
      if printf '%s' "$FILES" | grep -q summary.json; then say "trial FINISHED (summary.json present, pid gone)"; exit 0; fi
      say "trial pid gone WITHOUT summary.json -- died"; exit 2
    fi
  fi
  sleep $CYCLE
done
say "cycle cap reached; trial still running"; exit 4
