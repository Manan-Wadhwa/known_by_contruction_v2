# HANDOFF: live state

`docs/HANDOFF-2026-10-06.md` is the takeover prompt and
`docs/HANDOFF-2026-09-10.md` holds the box mechanics. This file holds the live state, one `## <owner>` section per agent, each under 40 lines.
Update your own section before you stop: in flight, last verdicts, next,
gotchas. Box URLs and tokens never go in this file.

## lead

**State 2026-10-06 14:30Z (session d6f94cbe, branch `s21-tooling`).** Five
leases, all running since 14:24:42Z (PLAN §21.1): box1/box2 the §19 pair
`forkcf4b_{A,D}_s0_tx_h320` (320 steps, ~5.5 h+), box3/box4 §20's
`forkcf4b_A_s{0,1}_tx_p20`, box5 `ctrl4b_A_s0_m3` (~1.4-2 h). Watchers: one
bg task per box, `mz01-watch` (run from the main checkout), inhibitor, reads
monitor. Creds in the job scratchpad only. Pushed 2026-10-06 with the user's
go: `reorg/b-split-trainer` at `064ca2d`, `s21-tooling` (not merged).

**Next:** each freed box takes §21's next item (`ctrl4b_D_s0_m3`, then
`ctrl4b_{A,D}_s1_m3`, then blocks of four per seed 2-7) via `launch_fork.sh`.
`s21.py` (across-seed analysis, verified on §18.1; the verifier's fixes in
`a43550e`) prints validity per run; §21.1 fixed that the both-solved
condition reads its `both`.

**Gotcha (new):** run helpers from a worktree and pulls land in that
worktree's `results/`; symlink each new label to the main checkout's
`results/` before its watcher starts.

**Last verdicts** (`docs/RESULTS_LOG.md` §9-§10).
- No AVOIDS in 19 launches; §15 closed.
- Tie-and-crossfit (§16): the selection works (credited detour-better
  0.66-0.82); undosed `_tx` keeps solve, ends below its partner at both seeds
  (0.818 vs 0.946; 0.765 vs 0.989, paired gap −0.484 [−0.773, −0.211]) and
  level with the on-policy controls (0.80; 0.765). Without the selection
  (`_t`) solve collapsed. Prices ≥ −30 lose solve (§16.7, §17.1).
- §19's pair at 160, before the loss: arm 0.938 / 0.775, paired gap −0.461
  [−0.781, −0.195] with the arm solving more (not verified).

**Open for the user.** Whether the AVOIDS solve band or the 160-step horizon
should change (DECISIONS "still open", FAILURES A7). The written comparison is
`docs/research/2026-10-04-band-horizon/COMPARISON.md` (verified; no option
chosen).

**Gotchas** (the full list is in the takeover prompt).
- Compare every arm with its on-policy control as well as its cell-D partner.
- `mean_steps` rises are mostly capped episodes; use `detour_paid_solved`.
- Same seed and config diverge from step 3 across leases (§19.1).
- The Stop hook blocks when the repo changed and this file did not.
