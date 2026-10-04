#!/usr/bin/env bash
# launch_fork.sh <boxN> <label>: sync HEAD to the box, check peft, launch a pre-registered fork-counterfactual arm.
# Labels and their env are the case table below (PLAN-fork-counterfactual.md §16-§20); add a label there before launching it.
# Written 2026-10-04 as the session helper launch16.sh; needs MZ01_BOX_DIR (boxN.url / boxN.token).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding boxN.url and boxN.token}"
BOX=$1; LABEL=$2; R=/marimo/repo/experiments/v2/MZ01_maze_world/results
STEPS=160; EVAL=40,80,120
case $LABEL in
  forkcf4b_A_s0_tx)     SEED=0; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit";;
  forkcf4b_D_s0_tx)     SEED=0; X="MZ01_TRIAL_CELL=D MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit";;
  forkcf4b_A_s0_tx_p30) SEED=0; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit MZ01_SCORE_MINE=-30";;
  forkcf4b_A_s1_tx_p30) SEED=1; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit MZ01_SCORE_MINE=-30";;
  forkcf4b_A_s0_t)      SEED=0; X="MZ01_FORK_PICK=tie";;
  forkcf4b_D_s1_tx)     SEED=1; X="MZ01_TRIAL_CELL=D MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit";;
  forkcf4b_A_s1_tx)     SEED=1; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit";;
  forkcf4b_A_s0_tx_p20) SEED=0; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit MZ01_SCORE_MINE=-20";;
  forkcf4b_A_s1_tx_p20) SEED=1; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit MZ01_SCORE_MINE=-20";;
  forkcf4b_A_s0_tx_h320) SEED=0; STEPS=320; EVAL=40,80,120,160,200,240,280; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit";;
  forkcf4b_D_s0_tx_h320) SEED=0; STEPS=320; EVAL=40,80,120,160,200,240,280; X="MZ01_TRIAL_CELL=D MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit";;
  forkcf4b_A_s0_tx_p50) SEED=0; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit MZ01_SCORE_MINE=-50";;
  forkcf4b_A_s0_tx_p100) SEED=0; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit MZ01_SCORE_MINE=-100";;
  *) echo "unknown label $LABEL"; exit 2;;
esac
bash scripts/box/bringup2_nosetup.sh $BOX 2>&1 | grep -E 'MD5|ready|FAIL'
timeout 60 bash .claude/skills/marimo-pair/scripts/execute-code.sh --url "$(cat $S/$BOX.url)" --token "$(cat $S/$BOX.token)" -c "import subprocess; print(subprocess.run('bash -c \". /marimo/persist/env.sh; python3 -c \\\"import peft; print(chr(80)+chr(69)+chr(70)+chr(84), peft.__version__)\\\"\"; pgrep -fa mz01_ | head -2', shell=True, capture_output=True, text=True).stdout)" 2>&1 | grep -v '^Warning'
bash scripts/box/launch_only.sh $BOX $LABEL $STEPS $EVAL fork_cf_g6 $SEED MZ01_FORK_GRAD_MAX=0.3 MZ01_FORK_COEF=1.0 MZ01_MODEL=4B MZ01_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554 MZ01_CREDIT=episode MZ01_MOVE_MASS_COEF=0.3 MZ01_NAV_ADAPTER=$R/navigator/nav_nav48_4b.pt MZ01_NAV_JSON=$R/navigator/nav_nav48_4b.json MZ01_WARM_START=$R/orgb_4b/warm_start.pt $X
