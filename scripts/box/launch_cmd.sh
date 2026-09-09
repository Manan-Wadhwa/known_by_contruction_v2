#!/usr/bin/env bash
# launch_cmd.sh <box> <label> "<command>" [VAR=value ...]
#   Runs an arbitrary command on a box under the persistent env, detached, with
#   /marimo/repo/<label>.pid, <label>.log and <label>_launch.sh. Refuses if any
#   mz01_ process is already running there (one job per box). The command runs
#   from /marimo/repo; chain steps with && for a two-stage job.
set -uo pipefail
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding boxN.url and boxN.token}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; EX="$WT/.claude/skills/marimo-pair/scripts/execute-code.sh"
BOX=$1; LABEL=$2; CMD=$3; shift 3
URL=$(cat $S/$BOX.url); TOK=$(cat $S/$BOX.token); cd $WT; SHA=$(git rev-parse --short HEAD)
EXPORTS=""; for kv in "$@"; do EXPORTS+="export $kv"$'\n'; done
timeout 120 bash $EX --url "$URL" --token "$TOK" - <<PY 2>&1 | grep -v "^Warning: connecting"
import subprocess, time
busy = subprocess.run("pgrep -af 'mz01_' | grep -v pgrep | wc -l", shell=True, capture_output=True, text=True).stdout.strip()
assert busy == "0", f"busy: {busy} mz01_ process(es) already running on $BOX"
open("/marimo/repo/${LABEL}_launch.sh", "w").write("""#!/bin/bash
set -u
cd /marimo/repo || exit 91
[ -f /marimo/persist/env.sh ] && . /marimo/persist/env.sh
export CALIBRATION_GIT_SHA=$SHA
$EXPORTS
echo \$\$ > /marimo/repo/${LABEL}.pid
date -u +'launched %Y-%m-%dT%H:%M:%SZ'
$CMD
rc=\$?; date -u +"finished %Y-%m-%dT%H:%M:%SZ rc=\$rc"; exit \$rc
""")
subprocess.Popen("setsid nohup bash /marimo/repo/${LABEL}_launch.sh > /marimo/repo/${LABEL}.log 2>&1 < /dev/null &", shell=True); time.sleep(8)
pid = open("/marimo/repo/${LABEL}.pid").read().strip()
import os
print("[$BOX] ${LABEL} pid", pid, "alive", os.path.exists(f"/proc/{pid}"))
print(subprocess.run("grep -vE '^\\\\s*$|Warning|warnings.warn' /marimo/repo/${LABEL}.log | tail -3 | cut -c1-200", shell=True, capture_output=True, text=True).stdout)
PY
