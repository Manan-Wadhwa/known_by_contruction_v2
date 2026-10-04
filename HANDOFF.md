# HANDOFF: live state

`docs/HANDOFF-2026-09-10.md` holds the rules and the box mechanics. This file
holds the live state, one `## <owner>` section per agent, each under 40 lines.
Update your own section before you stop: in flight, last verdicts, next,
gotchas. Box URLs and tokens never go in this file.

## lead

**In flight (2026-10-04 05:55Z).** box 2 `forkcf4b_A_s1_tx_p30` (final
~05:59Z), box 4 `forkcf4b_A_s0_t` (~06:25Z), box 1 `forkcf4b_A_s0_tx_p50`
(~06:37Z), box 3 `forkcf4b_A_s0_tx_p100` (~06:47Z). Next: box 2
`forkcf4b_D_s1_tx`, box 4 `forkcf4b_A_s1_tx` (§18); box 1
`forkcf4b_A_s0_tx_h320`, box 3 `forkcf4b_D_s0_tx_h320` (§19, 320 steps).
`launch16.sh` covers all; `read2.py` (§16.4) and `paired.py` (§19's paired
mine-step gap on the `episodes` records) in the session scratchpad.
Built and merged: the eighteenth pass (§16; `MZ01_FORK_PICK=tie`,
`MZ01_FORK_SELECT=crossfit`; `record/BUILD-fork-share-2026-10-04.md`, critic
APPROVE WITH CHANGES, all four made; 147 tests pass). §16's arms go on
boxes whose §15 run ends flat.

**Last verdicts.**
- §15 closed, no AVOIDS (§15.3-§15.7; RESULTS_LOG §9).
- §16 seed 0 (§16.6; RESULTS_LOG §10): `_tx` 0.938 / 0.818, `_tx_p30` 0.867 /
  0.750 (exactly on the strict bar), partner 0.953 / 0.946. Gaps to the
  partner -0.13 [-0.28, +0.03] and -0.20 [-0.35, -0.04] (Poisson approx.).
  Selection works (detour-better 0.66-0.82); replay gate passes.
- Per-episode held-out records (`episodes` in every eval block) merged at
  `8d147cc`; runs launched before it have none (use §16.6's Poisson approx.).
- Dosed tx arms lose solve early: `_tx_p50` 0.719 / 0.925 at 40, 0.789 /
  0.850 at 80; `_tx_p100` 0.797 / 0.804 at 40; `_tx_p30` s1 0.734 / 0.693 at 80
  (RANDOMISED). `_t` 0.758 / 0.846 at 80 (RANDOMISED).

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
