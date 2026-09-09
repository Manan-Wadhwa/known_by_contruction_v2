#!/usr/bin/env bash
# ONE shell for every live box. JOBS="box1:label1 box2:label2 ..." (trial runs).
# Each round: one watch_box.sh cycle per box in turn (liveness, log tail, small
# files pulled), then sleep CYCLE. A box whose run has ended (watch_box exit 0,
# after its final pull) or whose lease is gone (exit 3) leaves the rotation.
# Run under systemd-inhibit so the laptop cannot sleep while it holds the pulls.
cd "$(dirname "$0")/../.."
: "${JOBS:?JOBS=\"box1:label ...\" required}"; CYCLE=${CYCLE:-600}; ROUNDS=${ROUNDS:-80}
R=/marimo/repo/experiments/v2/MZ01_maze_world/results
declare -A J; for kv in $JOBS; do J[${kv%%:*}]=${kv#*:}; done
say(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
say "watch_all start: ${JOBS} cycle ${CYCLE}s"
for round in $(seq 1 $ROUNDS); do
  [ ${#J[@]} -eq 0 ] && { say "all runs ended; exit"; exit 0; }
  for b in $(printf '%s\n' "${!J[@]}" | sort); do
    l=${J[$b]}
    MZ01_BOX_DIR=$MZ01_BOX_DIR CYCLE=1 MAX=1 bash scripts/box/watch_box.sh $b $l $R/$l trial 2>&1 | grep -v "^Warning: connecting"; rc=${PIPESTATUS[0]}
    case $rc in 0) say "$b $l ended; final pull done; dropped"; unset J[$b];; 3) say "$b unreachable twice; dropped"; unset J[$b];; esac
  done
  sleep $CYCLE
done; say "round cap"; exit 4
