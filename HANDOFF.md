# HANDOFF: live state

`docs/HANDOFF-2026-09-10.md` holds the rules and the box mechanics. This file
holds the live state, one `## <owner>` section per agent, each under 40 lines.
Update your own section before you stop: in flight, last verdicts, next,
gotchas. Box URLs and tokens never go in this file.

## lead

**In flight: nothing (2026-10-04 09:45Z).** All four leases died ~09:40Z
(§19.2/§20.1). Partial runs under `results/_lost_2026-10-04_0940Z/`. On the
next leases: relaunch the §19 pair (320 steps, together) and §20's two arms
unchanged, via `launch16.sh` (session scratchpad) after `bringup2.sh` +
`bringup2_nosetup.sh` + the 4B pushes; check `import peft`. Units
`mz01-watch` and `mz01-inhibit` stopped. The next design (2×2 seed sweep with
matched on-policy controls) is at `docs/research/2026-10-04-next/DESIGN.md`;
its on-policy cells wait for the user's D-21 decision.
Built and merged: the eighteenth pass (§16; `MZ01_FORK_PICK=tie`,
`MZ01_FORK_SELECT=crossfit`; `record/BUILD-fork-share-2026-10-04.md`, critic
APPROVE WITH CHANGES, all four made; 147 tests pass). §16's arms go on
boxes whose §15 run ends flat.

**Last verdicts.**
- §18.1: seed-1 `_tx` 0.828 / 0.765 (not AVOIDS), paired gap to its partner
  −0.484 [−0.773, −0.211], but level with the on-policy control `ctrl4b_s1`
  (also 0.765). Undosed `_tx` beats its partner at both seeds and does not
  beat the on-policy control at either. Compare every arm with its control.
- §15 closed, no AVOIDS (§15.3-§15.7; RESULTS_LOG §9).
- §16 seed 0 (§16.6; RESULTS_LOG §10): `_tx` 0.938 / 0.818, `_tx_p30` 0.867 /
  0.750 (exactly on the strict bar), partner 0.953 / 0.946. Gaps to the
  partner -0.13 [-0.28, +0.03] and -0.20 [-0.35, -0.04] (Poisson approx.).
  Selection works (detour-better 0.66-0.82); replay gate passes.
- Per-episode held-out records (`episodes` in every eval block) merged at
  `8d147cc`; runs launched before it have none (use §16.6's Poisson approx.).
- §16.7/§17.1: `_t` 0.445 / 0.164, `_tx_p30` s1 0.602 / 0.202, −50 0.641 /
  0.618, −100 0.586 / 0.693, all RANDOMISED or DEAD; reads under the bar
  always came with a solve loss (detours on `_t`/s1, capped episodes on the
  dose arms).
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
- Leases differ in Python environment (`/usr/local` vs `/tmp/uv-venv`, a
  different `huggingface_hub`); same seed and config diverge from step 3
  (§19.1). Record torch/CUDA per run when next touching the bring-up.
- D-22 forbids removing step terms from R, so route credit can only be
  outweighed (the dose), not subtracted.
- Run one read monitor at a time: two `reads.py` share `reads.seen` and
  repeat reads.
- The Stop hook blocks when the repo changed and this file did not.
