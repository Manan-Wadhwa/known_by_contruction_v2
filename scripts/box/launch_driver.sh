#!/usr/bin/env bash
# launch_driver.sh <box> <seed> [label] : the organism driver, cell A, the seed-1 AVOIDS recipe, checkpoints every 50, full measurement
set -uo pipefail
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding boxN.url and boxN.token}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; EX="$WT/.claude/skills/marimo-pair/scripts/execute-code.sh"
BOX=$1; SEED=$2; LABEL=${3:-ctrl_s$SEED}; URL=$(cat $S/$BOX.url); TOK=$(cat $S/$BOX.token); cd $WT; SHA=$(git rev-parse --short HEAD)
timeout 120 bash $EX --url "$URL" --token "$TOK" - <<PY 2>&1 | grep -v "^Warning: connecting"
import subprocess, time
assert subprocess.run("pgrep -af 'mz01_' | grep -v pgrep | wc -l", shell=True, capture_output=True, text=True).stdout.strip() == "0", "busy"
M = "/marimo/repo/experiments/v2/MZ01_maze_world/results"
open("/marimo/repo/${LABEL}_launch.sh", "w").write(f"""#!/bin/bash
set -u
cd /marimo/repo || exit 91
[ -f /marimo/persist/env.sh ] && . /marimo/persist/env.sh
export CALIBRATION_GIT_SHA=$SHA
export MZ01_MODEL=14B MZ01_REVISION=40c069824f4251a91eefaf281ebe4c544efd3e18
export MZ01_CELLS=A MZ01_SEEDS=$SEED MZ01_STEPS=160 MZ01_SHARD=$LABEL
export MZ01_EXPLORE=fixed MZ01_ENTROPY_COEF=0.1 MZ01_PREFLIGHT_SOFT=1 MZ01_CKPT_EVERY=50
export MZ01_MOVE_MASK=1 MZ01_EXIT_AT_CAP=1 MZ01_MOVE_MASS_COEF=0.1
export MZ01_NAV_ADAPTER={M}/navigator/nav_nav48.pt MZ01_NAV_JSON={M}/navigator/nav_nav48.json
export MZ01_WARM_START={M}/orgb/warm_start.pt MZ01_OUT={M}/orgb_$LABEL
echo \$\$ > /marimo/repo/${LABEL}.pid
date -u +'launched %Y-%m-%dT%H:%M:%SZ'
exec python3 scripts/mz01_orgb_driver.py
""")
subprocess.Popen("setsid nohup bash /marimo/repo/${LABEL}_launch.sh > /marimo/repo/${LABEL}.log 2>&1 < /dev/null &", shell=True); time.sleep(6)
print(subprocess.run("echo pid \$(cat /marimo/repo/${LABEL}.pid); ps -o pid,args -p \$(cat /marimo/repo/${LABEL}.pid) | tail -1; grep -vE '^\\s*$|Warning|warnings.warn' /marimo/repo/${LABEL}.log | head -2 | cut -c1-140", shell=True, capture_output=True, text=True).stdout)
PY
