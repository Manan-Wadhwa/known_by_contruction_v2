# HANDOFF: live state

`docs/HANDOFF-2026-10-06.md` is the takeover prompt and
`docs/HANDOFF-2026-09-10.md` holds the box mechanics. This file holds the live state, one `## <owner>` section per agent, each under 40 lines.
Update your own section before you stop: in flight, last verdicts, next,
gotchas. Box URLs and tokens never go in this file.

## lead

**State 2026-10-06 (session d6f94cbe, worktree branch `s21-tooling`).** Five
fresh leases (box1-5, creds in the job scratchpad only); bring-up running as
`mz01-bringup-boxN` units; inhibitor up. Nothing launched yet. Pushed through
`eec426f`; later commits local (ask before pushing).

**In flight / next, in queue order** (`PLAN-fork-counterfactual.md`):
1. box1/box2 the §19 pair `forkcf4b_{A,D}_s0_tx_h320` (320 steps), box3/box4
   §20's `forkcf4b_A_s{0,1}_tx_p20` (§19.2/§20.1 relaunches).
2. box5 `ctrl4b_A_s0_m3`, then §21's queue: controls `ctrl4b_{A,D}_s{k}_m3`
   (D-23, controls only), then blocks of four per seed 2-7.

**Built this session (`e30072a`):** `launch_fork.sh` takes the §21 labels
(controls on `onpolicy_g6`, no fork knobs); `reads.py`, `read2.py`,
`paired.py` read `onpolicy_g6.json`; `scripts/box/s21.py` is the across-seed
analysis (reproduces §18.1 at rng 0; results-verifier check pending).

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
