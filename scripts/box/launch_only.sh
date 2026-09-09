#!/usr/bin/env bash
# launch_only.sh <boxN> <label> <steps> <eval_at> <arms> <seed> [extra KEY=VAL...]  -- md5 verify then launch
set -uo pipefail
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding boxN.url and boxN.token}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EX=$WT/.claude/skills/marimo-pair/scripts/execute-code.sh
BOX=$1; LABEL=$2; STEPS=$3; EVAL_AT=$4; ARMS=$5; SEED=$6; shift 6
URL=$(cat $S/$BOX.url); TOK=$(cat $S/$BOX.token)
say() { echo "[$BOX $(date -u +%H:%M:%SZ)] $*"; }
cd $WT
REMOTE=$(timeout 120 bash $EX --url "$URL" --token "$TOK" -c "import hashlib,glob; print(' '.join(hashlib.md5(open('/marimo/repo/'+p,'rb').read()).hexdigest() for p in ('src/calibration/maze_orgb.py','scripts/mz01_explore_trial.py','experiments/v2/MZ01_maze_world/run_orgb.py'))); print('pyfiles', len(glob.glob('/marimo/repo/src/**/*.py', recursive=True)) + len(glob.glob('/marimo/repo/scripts/*.py')))" 2>&1 | grep -v "^Warning: connecting")
REF_SHA=${REF_SHA:-HEAD}
LOCAL=$(for f in src/calibration/maze_orgb.py scripts/mz01_explore_trial.py experiments/v2/MZ01_maze_world/run_orgb.py; do git show $REF_SHA:$f | md5sum | awk '{print $1}'; done | tr '\n' ' ' | sed 's/ $//')
say "remote: $(echo "$REMOTE" | tr '\n' ' ')"
[ "$(echo "$REMOTE" | head -1)" = "$LOCAL" ] && say "MD5 MATCH" || { say "MD5 MISMATCH -- not launching (local $LOCAL)"; exit 3; }
M=/marimo/repo/experiments/v2/MZ01_maze_world/results
SHA=$(git rev-parse --short $REF_SHA)
MZ01_BOX_DIR=$S BOX=$BOX timeout 170 bash scripts/box/launch_trial_on_box.sh $LABEL /marimo/repo/$LABEL.log \
  CALIBRATION_GIT_SHA=$SHA \
  MZ01_MODEL=14B MZ01_REVISION=40c069824f4251a91eefaf281ebe4c544efd3e18 \
  MZ01_TRIAL_CELL=A MZ01_TRIAL_CELL_2= \
  MZ01_TRIAL_ARMS=$ARMS \
  MZ01_TRIAL_STEPS=$STEPS MZ01_TRIAL_EVAL_AT=$EVAL_AT MZ01_TRIAL_EVAL_N=128 MZ01_TRIAL_SEED=$SEED \
  MZ01_GROUP=6 MZ01_BATCH=4 MZ01_ENTROPY_COEF=0.1 \
  MZ01_MOVE_MASK=1 MZ01_MOVE_MASS_COEF=0.1 MZ01_EXIT_AT_CAP=1 \
  MZ01_NAV_ADAPTER=$M/navigator/nav_nav48.pt \
  MZ01_NAV_JSON=$M/navigator/nav_nav48.json \
  MZ01_WARM_START=$M/orgb/warm_start.pt \
  MZ01_TRIAL_OUT=$M/$LABEL "$@" 2>&1 | grep -E "PID|ALIVE|LAUNCH_REFUSED|launched" | tr '\n' ' '
echo; say "done"
