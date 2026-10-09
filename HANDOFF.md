# HANDOFF: live state

`docs/HANDOFF-2026-10-09.md` is the takeover prompt and
`docs/HANDOFF-2026-09-10.md` holds the box mechanics. This file holds the live state, one `## <owner>` section per agent, each under 40 lines.
Update your own section before you stop: in flight, last verdicts, next,
gotchas. Box URLs and tokens never go in this file.

## lead

**State 2026-10-09 11:45 IST (session d6f94cbe, branch `s21-tooling`).** §21
RESUMED at seed 4 on five new leases (D-25, PLAN §21.12, pushed `10fe7a2`).
Bring-up running (`mz01-bringup-boxN` units); inhibitor up. Next: launch
box1 `forkcf4b_A_s4_tx`, box2 `forkcf4b_D_s4_tx`, box3 `ctrl4b_A_s4_m3`,
box4 `ctrl4b_D_s4_m3`, box5 `forkcf4b_A_s5_tx`; record the leases'
environment; watchers, `mz01-watch`, reads monitor. Then seeds 5-7 as boxes
free; final analysis at seed 7 (`s21.py`). The 10-07 stop's partials stay in
`results/_stopped_2026-10-07/`. Creds in the job scratchpad only.

**Where it ended** (`docs/RESULTS_LOG.md` §11; PLAN §19.4-§21.11).
- No AVOIDS credited in the programme.
- §20 (dose −20) closed: both arms ended RANDOMISED (§20.3, §20.4). §20.2's
  AVOIDS read at 120 stays withheld for the user (§21.4).
- §19's arm met AVOIDS at 320 by mine and solve (0.938 / 0.743, paired
  −0.430 [−0.664, −0.242]); the user kept credit withheld (§19.4, §21.7).
- §21 interim, seeds 0-3, one run per cell (§21.10): P_fork −0.168
  [−0.436, +0.091] not shown; P_on −0.273 [−0.494, −0.086], a control
  property; F and I contain zero, half-widths 0.367 and 0.343.
- Detour-and-cap over-avoidance hit a fork arm (seed 3) and a control (seed
  1) alike, so it is not specific to the fork term.

**Open for the user.** §20.2's credit (§21.4); band or horizon
(`docs/research/2026-10-04-band-horizon/COMPARISON.md`); merging `s21-tooling`.

**To resume.** `docs/HANDOFF-2026-10-09.md` first; `docs/HANDOFF-2026-10-06.md` for bring-up. All §21 labels are
in `launch_fork.sh` (and `forkcf4b_{A,D}_s1_tx_h320`). Run helpers from a
worktree only after symlinking each label's `results/` dir to the main
checkout's. The classifier may block launches on freed boxes without the
user's explicit go. `scripts/box/s21.py --seeds ...` gives the across-seed read.

**Gotchas** (the full list is in the takeover prompt).
- Compare every arm with its on-policy control as well as its cell-D partner.
- `mean_steps` rises are mostly capped episodes; use `detour_paid_solved`.
- Same seed and config diverge (seed 0 pair at 160: −0.461 on 10-04,
  §19.2, unverified; +0.320 on 10-06, §19.4).
- The Stop hook blocks when the repo changed and this file did not.
