# HANDOFF: live state

`docs/HANDOFF-2026-10-06.md` is the takeover prompt and
`docs/HANDOFF-2026-09-10.md` holds the box mechanics. This file holds the live state, one `## <owner>` section per agent, each under 40 lines.
Update your own section before you stop: in flight, last verdicts, next,
gotchas. Box URLs and tokens never go in this file.

## lead

**State 2026-10-07 02:40 IST (session d6f94cbe, branch `s21-tooling`).** box1
`ctrl4b_A_s3_m3`, box2 `ctrl4b_D_s3_m3`, box3 `forkcf4b_D_s3_tx`, box4
`forkcf4b_A_s3_tx`, box5 `forkcf4b_A_s4_tx`; user approved launches as boxes free. Seeds 0-2 complete and recorded (§21.2-§21.8); the interim (s21.py,
seeds 0-3) runs when seed 3's four cells end (~04:00 IST).
Watchers: one bg task per box, `mz01-watch` (main checkout), inhibitor, reads
monitor. Creds in the job scratchpad only.

**Open for the user.** (1) §19.4: the §19 arm meets AVOIDS at 320 (0.938 /
0.743, paired −0.430 [−0.664, −0.242]); credit withheld on the control-at-320
and DEAD-label clauses. User kept it withheld (§21.7).
(2) §20.2's read at 120 (§21.4). (3) Band or horizon (`docs/research/2026-10-04-band-horizon/COMPARISON.md`).

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

**Gotchas** (the full list is in the takeover prompt).
- Compare every arm with its on-policy control as well as its cell-D partner.
- `mean_steps` rises are mostly capped episodes; use `detour_paid_solved`.
- Same seed and config diverge from step 3 across leases (§19.1).
- The Stop hook blocks when the repo changed and this file did not.
