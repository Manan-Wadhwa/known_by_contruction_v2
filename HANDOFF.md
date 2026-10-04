# HANDOFF: live state

`docs/HANDOFF-2026-09-10.md` holds the rules and the box mechanics. This file
holds the live state, one `## <owner>` section per agent, each under 40 lines.
Update your own section before you stop: in flight, last verdicts, next,
gotchas. Box URLs and tokens never go in this file.

## lead

**In flight since 2026-10-03 21:51Z** (`PLAN-fork-counterfactual.md` §15).
box 1 `forkcf4b_A_s0_g10` (cap 1.0); box 2 `forkcf4b_A_s1_p30` and box 4
`forkcf4b_A_s0_p30` (the dose at -30); box 3 `forkcf4b_A_s0_f2` (two forks);
box 5 `forkcf4b_D_s0_g_rerun` (cell-D partner). At 00:10Z on 2026-10-04 they
stood at steps 97-128; finals about 00:45-01:05Z, `_f2` about 01:40Z.
Monitors: a background task per box, the `mz01-watch` unit, the inhibitor, the
read monitor (`reads.py` in the session scratchpad).
Building: the eighteenth pass (§16; `MZ01_FORK_PICK=tie`,
`MZ01_FORK_SELECT=crossfit`) by an Opus subagent in
`.claude/worktrees/build-fork-share` (branch `build/fork-share`). Then a
critic, then the suite, then §16's arms on boxes whose §15 run ends flat.

**Last verdicts.**
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
- Review the build, dispatch a critic who has not seen it, run the suite,
  merge `build/fork-share`, sync the boxes (`bringup2_nosetup.sh`), launch.
- Pushed to origin: up to `5ae78b7`. Later commits are local.

**Gotchas.**
- The trainer refuses a dose on cell D, because cell D never charges the price.
- `bringup2.sh` prints a false "setup FAILED"; before `28e0824` the follow-up
  `bringup2_nosetup.sh` deleted `/marimo/pylib` (FAILURES B23). Check
  `import peft` under `/marimo/persist/env.sh` before a launch.
- An Agent worktree is cut from `origin/main`, not the local branch; create
  the worktree by hand at the commit the agent needs.
- `.claude/agents/` added mid-session is not registered until restart; run
  the verifier as a general-purpose agent that reads its file.
- The Stop hook blocks when the repo changed and this file did not.
