# HANDOFF: live state

`docs/HANDOFF-2026-09-10.md` holds the rules and the box mechanics. This file
holds the live state, one `## <owner>` section per agent, each under 40 lines.
Update your own section before you stop: in flight, last verdicts, next,
gotchas. Box URLs and tokens never go in this file.

## lead

**In flight (2026-10-04 01:45Z).** Four leases (box 5 died ~01:12Z, §16.2).
§15: box 2 `forkcf4b_A_s1_p30` (~230 s/step, RANDOMISED at 120). §16: box 4
`forkcf4b_A_s0_tx` (00:45Z), box 1 `forkcf4b_A_s0_tx_p30` (01:12Z), box 3
`forkcf4b_D_s0_tx` (relaunched 01:43Z). Box 2 next takes `_tx_p30` s1; `_t`
waits for a box. Helpers `launch16.sh`, `rewatch.sh` in the session
scratchpad. Monitors: a background task per box, `mz01-watch`, the inhibitor,
two read monitors.
Built and merged: the eighteenth pass (§16; `MZ01_FORK_PICK=tie`,
`MZ01_FORK_SELECT=crossfit`; `record/BUILD-fork-share-2026-10-04.md`, critic
APPROVE WITH CHANGES, all four made; 147 tests pass). §16's arms go on
boxes whose §15 run ends flat.

**Last verdicts.**
- `forkcf4b_A_s0_f2` no read AVOIDS (final 0.891 / 0.88), share 0.046 (§15.6).
- §16.3: tx reads at 40 valid, none AVOIDS; read 1 passes on cell A
  (0.79-0.82); read 2 fired on cell D (-2.39 over 17 forks). Build question
  answered (no leak; too few forks; `docs/research/2026-10-04-fork-share/buildq/`);
  §16.4 replaces read 2 from 80 on (replay gate + route read; `read2.py` in
  the scratchpad computes both).
- §16.5 (reads at 80): none AVOIDS; gate passes; the cell-D partner's route
  credit equals the undosed cell-A arm's (-4.54 vs -4.69), the dosed arm's is
  3.05 more negative [-7.21, +1.51]. `_tx` at 120: 0.922 / 0.861.
- `forkcf4b_A_s0_g10` ALIVE to 120, then held-out solve 0.328 / ratio 0.27
  at 160 (RANDOMISED by solve; label DEAD), §15.5.
- `forkcf4b_A_s0_tx` steps 0-24: credited detour-better 0.81, 3.6 kept forks
  a step (§16.2).
- `forkcf4b_A_s0_p30` RANDOMISED at 160 (ratio 0.721, solve 0.773); the
  partner `forkcf4b_D_s0_g_rerun` DEAD (solve 0.695). §15.3, §15.4.
- Held-out `mean_steps` rises are mostly unsolved episodes at the cap, not
  longer routes (read `capped_rate`, `optimality`).
- Reads at 40 and 80 (§15.1, §15.2): all valid, none AVOIDS.
- At 120: `forkcf4b_A_s0_p30` RANDOMISED (solve 0.766, ratio 0.811);
  `forkcf4b_D_s0_g_rerun` solve 0.656, ratio 0.746 (partner, no verdict).
  Both seed-0 arms lose routes after step 80, priced or not.
- The §13 batch of 2026-09-10 was lost at steps 4-7 (§14); its pulls sit
  under `results/_lost_2026-09-10/`.

**Next.**
- At each final: summary.json, the verifier, a §15 addendum and a
  `docs/RESULTS_LOG.md` row, commit, then launch the box's next job: a dose
  arm that avoids goes to §11's partner or curve; otherwise §16 in its order.
- Before a §16 launch, sync the box (`bringup2_nosetup.sh`) so its md5 matches.
- Pushed to origin: up to `5ae78b7`. Later commits are local.

**Gotchas.**
- The trainer refuses a dose on cell D, because cell D never charges the price.
- `bringup2.sh` prints a false "setup FAILED"; before `28e0824` the follow-up
  `bringup2_nosetup.sh` deleted `/marimo/pylib` (FAILURES B23). Check
  `import peft` under `/marimo/persist/env.sh` before a launch.
- An Agent worktree is cut from `origin/main`, not the local branch; create
  the worktree by hand at the commit the agent needs.
- A `.claude/agents/` file added mid-session registers a few turns later;
  until then run it as a general-purpose agent that reads the file.
- D-22 forbids removing step terms from R, so route credit can only be
  outweighed (the dose), not subtracted.
- Run one read monitor at a time: two `reads.py` share `reads.seen` and
  repeat reads.
- The Stop hook blocks when the repo changed and this file did not.
