# HANDOFF: live state

`docs/HANDOFF-2026-09-10.md` holds the rules and the box mechanics. This file
holds the live state, one `## <owner>` section per agent, each under 40 lines.
Update your own section before you stop: in flight, last verdicts, next,
gotchas. Box URLs and tokens never go in this file.

## lead

**In flight on 2026-10-04.** The four arms of `PLAN-fork-counterfactual.md` §13 are being
relaunched unchanged today on five fresh leases. Box assignments are being
launched; see the plan's §15. The labels are `forkcf4b_A_s0_p30` and
`forkcf4b_A_s1_p30` (the dose at -30, seeds 0 and 1), `forkcf4b_A_s0_f2` (two
forks per episode) and `forkcf4b_A_s0_g10` (the fork cap at 1.0). The ETA is
about 2.7 h for 160 steps at ~55 s/step, and about 3.8 h for `_f2`
(`docs/HANDOFF-2026-09-10.md` state section; plan §13.1).

**Last verdicts.**
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
- Write the launch entry with box, pid and time in §15, start one monitor per
  box, and start the durable `mz01-watch` unit and the inhibitor.
- Read each arm at 40, 80, 120 and 160 by §13.3. Run every number through the
  `results-verifier` subagent before it enters §15 or `docs/RESULTS_LOG.md`.
- A dose arm that reads AVOIDS gets its own cell-D partner at the world's price
  before the price is credited (§11, §13.3). Relaunch an idle box at once.
- `reorg/b-split-trainer` is ahead of `main` and unmerged. Merging needs the
  user's go.

**Gotchas.**
- The trainer refuses a dose on cell D, because cell D never charges the price.
- `bringup2.sh` prints a false "setup FAILED"; follow it with `bringup2_nosetup.sh`.
- `launch_only.sh` refuses a box that already runs an `mz01_` process.
- The Stop hook blocks when the repo changed this session and this file did not.
  Another session's commit in the same tree counts as a change.
