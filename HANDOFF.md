# HANDOFF: live state

`docs/HANDOFF-2026-10-06.md` is the takeover prompt and
`docs/HANDOFF-2026-09-10.md` holds the box mechanics. This file holds the live state, one `## <owner>` section per agent, each under 40 lines.
Update your own section before you stop: in flight, last verdicts, next,
gotchas. Box URLs and tokens never go in this file.

## lead

**State 2026-10-07 00:40 IST (session d6f94cbe, branch `s21-tooling`).** Five
leases since 14:24Z on 10-06: box1 §19 arm `forkcf4b_A_s0_tx_h320` (ends
~01:35 IST; its partner ended 00:27), box2 `ctrl4b_D_s2_m3`, box3
`forkcf4b_A_s2_tx`, box4 `forkcf4b_A_s3_tx`, box5 `forkcf4b_D_s2_tx`. Ended
and recorded: §20 closed (§20.3, §20.4); controls seeds 0-2 cell A, 0-1 cell D
(§21.2-§21.6). Watchers: one bg task per box, `mz01-watch` (from the main
checkout), inhibitor, reads monitor. Creds in the job scratchpad only. The
classifier once blocked a launch on a freed box; the user said "launch it".

**Next:** each freed box takes §21's next item (`forkcf4b_D_s3_tx`,
then `ctrl4b_{A,D}_s3_m3`, seeds 4-7). Open for the user: credit §20.2's read (§21.4)? §21.5 done. via `launch_fork.sh`. `s21.py` (across-seed
analysis, verified on §18.1, fixes in `a43550e`) reads `both` (§21.1).

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
- 10-06: `forkcf4b_A_s1_tx_p20` AVOIDS at 120 (withheld, §20.2), then
  RANDOMISED at 160 (§20.3); seed 0 RANDOMISED from 80 (§20.4). §20 closed.

**Open for the user.** Whether the AVOIDS solve band or the 160-step horizon
should change (DECISIONS "still open", FAILURES A7). The written comparison is
`docs/research/2026-10-04-band-horizon/COMPARISON.md` (verified; no option
chosen).

**Gotchas** (the full list is in the takeover prompt).
- Compare every arm with its on-policy control as well as its cell-D partner.
- `mean_steps` rises are mostly capped episodes; use `detour_paid_solved`.
- Same seed and config diverge from step 3 across leases (§19.1).
- The Stop hook blocks when the repo changed and this file did not.
