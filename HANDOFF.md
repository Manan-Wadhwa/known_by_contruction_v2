# HANDOFF: live state

`docs/HANDOFF-2026-09-10.md` holds the rules and the box mechanics. This file
holds the live state, one `## <owner>` section per agent, each under 40 lines.
Update your own section before you stop: in flight, last verdicts, next,
gotchas. Box URLs and tokens never go in this file.

## lead

**In flight since 2026-10-03 21:51Z** (`PLAN-fork-counterfactual.md` §15).
box 1 `forkcf4b_A_s0_g10` (cap 1.0); box 2 `forkcf4b_A_s1_p30` and box 4
`forkcf4b_A_s0_p30` (the dose at -30); box 3 `forkcf4b_A_s0_f2` (two forks);
box 5 `forkcf4b_D_s0_g_rerun` (the cell-D partner, run to the end). Finals
about 00:30Z on 2026-10-04, `_f2` about 01:40Z. Monitors: a background task
per box, the `mz01-watch` unit, the inhibitor, and the read monitor
(`reads.py` in the session scratchpad, one line per new held-out read).

**Last verdicts.**
- Reads at 40 (plan §15.1): all five valid, none AVOIDS; mine ratios 0.86–0.94.
- The §13 batch of 2026-09-10 was lost at steps 4 to 7 with no read and no
  summary (plan §14). Its partial pulls now sit under
  `results/_lost_2026-09-10/`, so the relaunch writes to the same labels.
- The guarded arms of §11 held their move mass and did not move the mine rate.
  Their valid held-out reads sit at mine ratios 0.90 to 0.96 on cell A and
  cell D alike (plan §12, §13.1 item 5; `results/forkcf4b_{A,D}_s{0,1}_g/fork_cf_g6.json`).
- The controls the §13 arms are read against are `ctrl4b_s0`, RANDOMISED
  (solve 0.734 against 0.922, ratio 0.80), and `ctrl4b_s1`, ALIVE (solve 0.859,
  ratio 0.77) (`results/ctrl4b_s{0,1}/summary.json`).
- The cell-D partners are `forkcf4b_D_s0_g` at 40 (solve 0.914, ratio 0.96)
  and `forkcf4b_D_s1_g` at 40 (solve 0.938, ratio 0.94).

**Next.**
- Read each arm at 40, 80, 120 and 160 by §13.3 and §15. Run every number
  through the `results-verifier` subagent before it enters the plan or
  `docs/RESULTS_LOG.md`.
- When a run ends, record it and launch the next pre-registered job on that
  box at once: a dose arm that avoids is read against `forkcf4b_D_s0_g_rerun`
  (seed 0) or gets a seed-1 cell-D partner; both dose seeds avoiding sends the
  dose to the curve (-10, -20, -30, -50) on seeds 0 to 3; nothing avoiding
  means a fork-share design (plan §11, the handoff's reading rules).
- `main` and this branch are pushed to origin.

**Gotchas.**
- The trainer refuses a dose on cell D, because cell D never charges the price.
- `bringup2.sh` prints a false "setup FAILED"; follow it with `bringup2_nosetup.sh`.
  Before `28e0824` that script deleted `/marimo/pylib` (FAILURES B23); check
  `import peft` under `/marimo/persist/env.sh` before a launch.
- `launch_only.sh` refuses a box that already runs an `mz01_` process.
- The Stop hook blocks when the repo changed this session and this file did not.
  Another session's commit in the same tree counts as a change.
