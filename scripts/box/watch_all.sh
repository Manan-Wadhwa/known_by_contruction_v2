#!/usr/bin/env bash
# watch_all.sh: ONE unit for every live box. JOBS="box1:label1 box2:label2 ..."
#   Each box gets its own watch_box.sh loop (liveness off the pid file, the
#   log's last line, small files and new adapters pulled every CYCLE s, a final
#   pull when the pid is gone). The loops run in parallel inside this one
#   process group so every box is checked every CYCLE s regardless of how long
#   another box's pull takes. When a loop ends -- the run finished and was
#   pulled, or the lease is gone -- an event is logged to $MZ01_BOX_DIR/events.log
#   and sent to the desktop, and the box is reported IDLE: a lease that sits
#   idle gets reaped, so an idle box wants a launch or a farewell, not a wait.
# Run under systemd-run --user (see watchers-run-as-systemd-user-units).
cd "$(dirname "$0")/../.."
: "${JOBS:?JOBS=\"box1:label ...\" required}"; CYCLE=${CYCLE:-120}; MAX=${MAX:-3000}
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR}"; R=/marimo/repo/experiments/v2/MZ01_maze_world/results
EV="$S/events.log"
say(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
event(){ say "$*"; echo "[$(date -u +%H:%M:%SZ)] $*" >> "$EV"; notify-send -u critical "MZ01 boxes" "$*" 2>/dev/null || true; }
say "watch_all start: ${JOBS} cycle ${CYCLE}s (parallel per box)"
declare -A PIDS LABELS
for kv in $JOBS; do
  b=${kv%%:*}; l=${kv#*:}; LABELS[$b]=$l
  ( MZ01_BOX_DIR=$S CYCLE=$CYCLE MAX=$MAX bash scripts/box/watch_box.sh $b $l $R/$l trial 2>&1 | grep -v "^Warning: connecting"; exit ${PIPESTATUS[0]} ) &
  PIDS[$b]=$!
done
while [ ${#PIDS[@]} -gt 0 ]; do
  for b in "${!PIDS[@]}"; do
    if ! kill -0 ${PIDS[$b]} 2>/dev/null; then
      wait ${PIDS[$b]}; rc=$?; l=${LABELS[$b]}
      case $rc in
        0) event "$b: $l ENDED, final pull done -- box $b is IDLE now (launch or release it)";;
        3) event "$b: $l -- lease UNREACHABLE twice, box $b presumed gone";;
        *) event "$b: $l watcher exited rc=$rc";;
      esac
      unset PIDS[$b]
    fi
  done
  sleep 15
done
event "all runs ended; monitor exits"
