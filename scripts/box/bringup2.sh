#!/usr/bin/env bash
# bringup2.sh <boxN>: persistent setup + the three pushes + sync at HEAD + md5 check. No launch.
set -uo pipefail
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding boxN.url and boxN.token}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; EX="$WT/.claude/skills/marimo-pair/scripts/execute-code.sh"
BOX=$1; URL=$(cat $S/$BOX.url); TOK=$(cat $S/$BOX.token); cd $WT
say() { echo "[$BOX $(date -u +%H:%M:%SZ)] $*"; }
timeout 60 bash $EX --url "$URL" --token "$TOK" -c "import subprocess; subprocess.run('rm -rf /marimo/pylib; mkdir -p /marimo/persist', shell=True)" >/dev/null 2>&1
say "setup start"; timeout 500 bash $EX --url "$URL" --token "$TOK" - < scripts/box/setup_env.py > $S/setup_$BOX.log 2>&1
timeout 60 bash $EX --url "$URL" --token "$TOK" -c "print(open('/marimo/v2/setup.log').read()[-150:])" 2>/dev/null | grep -q "SETUP DONE" && say "setup DONE" || { say "setup FAILED"; tail -4 $S/setup_$BOX.log; exit 2; }
M=/marimo/repo/experiments/v2/MZ01_maze_world/results; export MZ01_BOX_DIR=$S BOX=$BOX
for pair in "results/navigator/nav_nav48.pt $M/navigator/nav_nav48.pt" "results/navigator/nav_nav48.json $M/navigator/nav_nav48.json" "results/orgb/warm_start_s0.pt $M/orgb/warm_start.pt"; do set -- $pair; timeout 400 bash scripts/box/push_file.sh experiments/v2/MZ01_maze_world/$1 $2 2>&1 | grep -oE "PUSH_VERIFIED [^ ]+|MISMATCH|FAIL" | cut -c1-60; done
MARIMO_URL=$URL MARIMO_TOKEN_FILE=$S/$BOX.token timeout 300 scripts/sync_to_sandbox.sh src/calibration/maze_orgb.py scripts/mz01_explore_trial.py scripts/mz01_orgb_driver.py experiments/v2/MZ01_maze_world/run_orgb.py 2>&1 | grep -E "SYNC_OK|FAIL"
REMOTE=$(timeout 120 bash $EX --url "$URL" --token "$TOK" -c "import hashlib; print('MD5S', ' '.join(hashlib.md5(open('/marimo/repo/'+p,'rb').read()).hexdigest() for p in ('src/calibration/maze_orgb.py','scripts/mz01_explore_trial.py','scripts/mz01_orgb_driver.py','experiments/v2/MZ01_maze_world/run_orgb.py','src/calibration/navigator.py','scripts/mz01_navigator_driver.py')))" 2>&1 | grep "^MD5S" | cut -d' ' -f2-)
LOCAL=$(md5sum src/calibration/maze_orgb.py scripts/mz01_explore_trial.py scripts/mz01_orgb_driver.py experiments/v2/MZ01_maze_world/run_orgb.py src/calibration/navigator.py scripts/mz01_navigator_driver.py | awk '{print $1}' | tr '\n' ' ' | sed 's/ $//')
[ "$REMOTE" = "$LOCAL" ] && say "MD5 MATCH (6 files) at $(git rev-parse --short HEAD)" || { say "MD5 MISMATCH"; exit 3; }
timeout 60 bash $EX --url "$URL" --token "$TOK" -c "import subprocess; print(subprocess.run('bash -c \". /marimo/persist/env.sh; python3 -c \\\"import peft; print(peft.__file__)\\\"\"; du -sh /marimo/hf | cut -f1', shell=True, capture_output=True, text=True).stdout.strip().replace(chr(10),' | '))" 2>&1 | grep -v "^Warning"
say "ready"
