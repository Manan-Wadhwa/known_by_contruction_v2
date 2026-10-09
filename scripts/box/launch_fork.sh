#!/usr/bin/env bash
# launch_fork.sh <boxN> <label>: sync HEAD to the box, check peft, launch a pre-registered fork-counterfactual arm or on-policy control.
# Labels and their env are the case table below (PLAN-fork-counterfactual.md §16-§21); add a label there before launching it.
# A ctrl4b_* label sets ARM=onpolicy_g6 and drops the fork knobs FK; everything else on the 4B line is shared.
# kl4b_*_b<beta without its dot> adds MZ01_KL_COEF; val4b_* adds the validation selection (BUILD-solve-protect-2026-10-09.md); *ctrl* are their on-policy controls.
# Written 2026-10-04 as the session helper launch16.sh; needs MZ01_BOX_DIR (boxN.url / boxN.token).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding boxN.url and boxN.token}"
BOX=$1; LABEL=$2; R=/marimo/repo/experiments/v2/MZ01_maze_world/results
STEPS=160; EVAL=40,80,120; ARM=fork_cf_g6; FK="MZ01_FORK_GRAD_MAX=0.3 MZ01_FORK_COEF=1.0"
B=${LABEL##*_b}; KL="MZ01_KL_COEF=${B:0:1}.${B:1}"; V="MZ01_VAL_EVERY=10 MZ01_VAL_N=128"; H=40,80,120,160,200,240,280
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
  forkcf4b_A_s1_tx_h320) SEED=1; STEPS=320; EVAL=40,80,120,160,200,240,280; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit";;
  forkcf4b_D_s1_tx_h320) SEED=1; STEPS=320; EVAL=40,80,120,160,200,240,280; X="MZ01_TRIAL_CELL=D MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit";;
  forkcf4b_A_s[2-7]_tx_h320) SEED=${LABEL:12:1}; STEPS=320; EVAL=40,80,120,160,200,240,280; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit";;
  forkcf4b_D_s[2-7]_tx_h320) SEED=${LABEL:12:1}; STEPS=320; EVAL=40,80,120,160,200,240,280; X="MZ01_TRIAL_CELL=D MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit";;
  forkcf4b_A_s0_tx_p50) SEED=0; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit MZ01_SCORE_MINE=-50";;
  forkcf4b_A_s0_tx_p100) SEED=0; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit MZ01_SCORE_MINE=-100";;
  forkcf4b_A_s[2-7]_tx) SEED=${LABEL:12:1}; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit";;
  forkcf4b_D_s[2-7]_tx) SEED=${LABEL:12:1}; X="MZ01_TRIAL_CELL=D MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit";;
  ctrl4b_A_s[0-7]_m3)   SEED=${LABEL:10:1}; ARM=onpolicy_g6; FK=; X=;;
  ctrl4b_D_s[0-7]_m3)   SEED=${LABEL:10:1}; ARM=onpolicy_g6; FK=; X="MZ01_TRIAL_CELL=D";;
  kl4b_A_s[0-7]_b003|kl4b_A_s[0-7]_b0[13]) SEED=${LABEL:8:1}; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit $KL";;
  kl4b_D_s[0-7]_b003|kl4b_D_s[0-7]_b0[13]) SEED=${LABEL:8:1}; X="MZ01_TRIAL_CELL=D MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit $KL";;
  klctrl4b_A_s[0-7]_b003|klctrl4b_A_s[0-7]_b0[13]) SEED=${LABEL:12:1}; ARM=onpolicy_g6; FK=; X="$KL";;
  klctrl4b_D_s[0-7]_b003|klctrl4b_D_s[0-7]_b0[13]) SEED=${LABEL:12:1}; ARM=onpolicy_g6; FK=; X="MZ01_TRIAL_CELL=D $KL";;
  val4b_A_s[0-7])       SEED=${LABEL:9:1}; STEPS=320; EVAL=$H; X="MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit $V";;
  val4b_D_s[0-7])       SEED=${LABEL:9:1}; STEPS=320; EVAL=$H; X="MZ01_TRIAL_CELL=D MZ01_FORK_PICK=tie MZ01_FORK_SELECT=crossfit $V";;
  valctrl4b_A_s[0-7])   SEED=${LABEL:13:1}; STEPS=320; EVAL=$H; ARM=onpolicy_g6; FK=; X="$V";;
  valctrl4b_D_s[0-7])   SEED=${LABEL:13:1}; STEPS=320; EVAL=$H; ARM=onpolicy_g6; FK=; X="MZ01_TRIAL_CELL=D $V";;
  *) echo "unknown label $LABEL"; exit 2;;
esac
bash scripts/box/bringup2_nosetup.sh $BOX 2>&1 | grep -E 'MD5|ready|FAIL'
timeout 60 bash .claude/skills/marimo-pair/scripts/execute-code.sh --url "$(cat $S/$BOX.url)" --token "$(cat $S/$BOX.token)" -c "import subprocess; print(subprocess.run('bash -c \". /marimo/persist/env.sh; python3 -c \\\"import peft; print(chr(80)+chr(69)+chr(70)+chr(84), peft.__version__)\\\"\"; pgrep -fa mz01_ | head -2', shell=True, capture_output=True, text=True).stdout)" 2>&1 | grep -v '^Warning'
bash scripts/box/launch_only.sh $BOX $LABEL $STEPS $EVAL $ARM $SEED $FK MZ01_MODEL=4B MZ01_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554 MZ01_CREDIT=episode MZ01_MOVE_MASS_COEF=0.3 MZ01_NAV_ADAPTER=$R/navigator/nav_nav48_4b.pt MZ01_NAV_JSON=$R/navigator/nav_nav48_4b.json MZ01_WARM_START=$R/orgb_4b/warm_start.pt $X
