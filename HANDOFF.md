# HANDOFF: live state

`docs/HANDOFF-2026-09-10.md` holds the rules and the box mechanics. This file
holds the live state, one `## <owner>` section per agent, each under 40 lines.
Update your own section before you stop: in flight, last verdicts, next,
gotchas. Box URLs and tokens never go in this file.

## lead

**State at the end of 2026-10-04 (10:20Z / 15:50 IST).** No box is leased;
nothing is running. All leases died (box 5 ~01:12Z, boxes 1-4 ~09:40Z;
FAILURES B24, B25). `main` and `reorg/b-split-trainer` are pushed through `eec426f`; the
end-of-day doc commits after it are local until the next push.
Units `mz01-watch` and `mz01-inhibit` are stopped.

**Queue of record for the next leases** (`PLAN-fork-counterfactual.md`):
1. Relaunch, unchanged and first: the §19 pair `forkcf4b_A_s0_tx_h320` +
   `forkcf4b_D_s0_tx_h320` (320 steps, together) and §20's
   `forkcf4b_A_s0_tx_p20`, `forkcf4b_A_s1_tx_p20` (§19.2/§20.1).
2. Then §21: the 2×2 (fork term on/off × cell A/D) at the world's price, seeds
   0-7, matched on-policy controls `ctrl4b_{A,D}_s{k}_m3` (mass 0.3) as
   controls only (D-23); across-seed contrasts with crossed seed × maze
   bootstrap intervals; an analysis script to be written and verified first
   (§21 "Measurement").

**How to launch.** Bring-up: `bringup2.sh boxN` then `bringup2_nosetup.sh
boxN`, push the three 4B files, check `import peft`. Fork arms:
`scripts/box/launch_fork.sh boxN <label>` (labels in its case table; the
h320 and p20 labels are there). §21's on-policy controls have no launch line
yet: write one (arm `onpolicy_g6`, `MZ01_MOVE_MASS_COEF=0.3`, cell A/D)
before launching. After each launch: verify the banner, start a
`watch_box.sh` background task, `scripts/box/rewatch.sh "<JOBS>"`, one
`scripts/box/reads.py` Monitor. Reads: `read2.py` (§16.4), `paired.py` (§19).

**Last verdicts** (`docs/RESULTS_LOG.md` §9-§10).
- No AVOIDS today in 22 runs. §15 (dose −30, cap 1.0, two forks) closed.
- Tie-and-crossfit (§16): the selection works (credited detour-better
  0.66-0.82); undosed `_tx` keeps solve, ends below its partner at both seeds
  (0.818 vs 0.946; 0.765 vs 0.989, paired gap −0.484 [−0.773, −0.211]) and
  level with the on-policy controls (0.80; 0.765). Without the selection
  (`_t`) solve collapsed. Prices ≥ −30 lose solve (§16.7, §17.1).
- §19's pair at 160, before the loss: arm 0.938 / 0.775, paired gap −0.461
  [−0.781, −0.195] with the arm solving more (not verified).

**Open for the user.** Whether the AVOIDS solve band or the 160-step horizon
should change (DECISIONS "still open", FAILURES A7). A written comparison was
commissioned; if it landed it is `docs/research/2026-10-04-band-horizon.md`.

**Gotchas.**
- Compare every arm with its on-policy control as well as its cell-D partner.
- Held-out `mean_steps` rises are mostly capped episodes; read `capped_rate`
  and `optimality` beside it. `detour_paid` counts capped mine-free episodes;
  use `detour_paid_solved`.
- Same seed and config diverge from step 3 across leases (0.11 in ratio at
  40; §19.1). Leases differ in Python environment; record torch/CUDA per run.
- `results/` is gitignored: move a lost run's partial pulls aside before a
  relaunch writes to its label.
- Agent worktrees are cut from `origin/main` (FAILURES B26).
- D-22: route credit cannot be subtracted from the return, only outweighed.
- The Stop hook blocks when the repo changed and this file did not.
