# MZ01 agent rules

This repository is MZ01, the minefield maze of the "Known by Construction"
programme. The live state of every agent is in `HANDOFF.md`; the takeover prompt is
`docs/HANDOFF-2026-10-06.md`; the box mechanics are in `docs/HANDOFF-2026-09-10.md`.

## Hard rules, in priority order

1. Read `HANDOFF.md` and `docs/HANDOFF-2026-10-06.md` first (live state and the
   takeover prompt), then `docs/HANDOFF-2026-09-10.md` for the box mechanics and
   its stage table. Load only what the stage needs. The rest is record.
2. The construction test. Mine positions enter only through the priced return.
   No mine-aware experts, hints, prompt prefixes, or forks placed by mine
   adjacency (`docs/research/APPROACHES.md` §B, last row). No on-policy-only
   recipe on the nav48 navigators (D-21), except as controls with nothing
   credited from them (D-23). No new knob without a line in the
   trial's docstring (`scripts/mz01_explore_trial.py`).
3. Pre-register every run and its reading rule as a dated plan section before
   its first read. The queue of record is the latest dated section of the
   active PLAN (now `PLAN-fork-counterfactual.md`), never an older arm table.
4. Every number in a document points at a file. Save research verbatim under
   `docs/research/`. Report losses plainly, with what survived.
5. Before a number enters a plan addendum, `docs/RESULTS_LOG.md`, a README or a
   reply, dispatch the `results-verifier` subagent with the claim and the
   evidence paths. Apply its corrections. Never word a result more strongly
   than the evidence allows.
6. Frozen files. A change to any of these invalidates comparison with the
   recorded reads, so it needs a new dated plan section and a `docs/DECISIONS.md`
   row first.
   - The held-out mazes: `_EVAL_SEED_RANGE` in `src/calibration/rl.py` and the
     read's `order_seed=7000 + seed` in `scripts/mz01_explore_trial.py`.
   - The world: the scoring constants and generator in `src/calibration/maze2.py`,
     as `SPEC.md` "The world" and "The generator" define them.
   - The reading bars: `MINE_RATIO_BAR`, `SOLVE_BAND`, `UNPARSED_BAR` and
     `ZERO_ADVANTAGE_BAR` in `scripts/score_mz01_explore.py`.
   - The baselines: every `results/<label>/reference.json`, the navigator
     `results/navigator/nav_nav48_4b.{pt,json}`, the warm start
     `results/orgb_4b/warm_start.pt` (hashes in `docs/ARTEFACTS.md`), and the
     model revision `cdbee75f`.
   - Dated plan sections and everything under `record/`. Append; never edit.
7. GPU work runs on the molab boxes, never the laptop. Keep every leased box
   busy. Run one Claude Code background monitor per box, plus the durable
   `mz01-watch` unit and the sleep inhibitor with
   `--what=sleep:idle:handle-lid-switch`.
8. Box URLs and tokens live only in the session scratchpad as `boxN.url` and
   `boxN.token`. Never print them into a document, echo them, or commit them.
9. Pushing, merging to `main` and deleting leases need the user's go, unless
   the prompt already gave it.
10. Several sessions may share the tree. Commit only the files you changed.
    Commit small and often, with the session's attribution trailers.
11. Use Opus subagents for builds, critics, literature and documentation. Tell
    every agent you start to read `CLAUDE.md` first and to pass that on.
12. Code style. Write the shortest diff that works and use the stdlib first.
    Every new function gets a short docstring that says what it does, its
    inputs and its output.

## Where things are

Paths under `experiments/v2/MZ01_maze_world/` are given from that directory.

- `SPEC.md`: the pre-commitment, with amendments A1 to A11.
- `PLAN-*.md`: the dated pre-registrations. `PLAN-fork-counterfactual.md` is
  the active one.
- `record/`: the dated lab notes, builds, the report and the diagnosis.
- `results/<label>/`: pulled run output, untracked. It holds `fork_cf_g6.json`
  (`history`, `eval_at`), `reference.json` (`eval`) and, once a run ends,
  `summary.json`.
- `docs/RESULTS_LOG.md`: every run with its verdict and what was lost.
- `docs/research/APPROACHES.md`: the backlog of approaches, each judged by the
  construction test.
- `docs/DECISIONS.md`: the decisions, D-01 onward.
- `scripts/box/`: bring-up, launch, pull and watch helpers (`docs/OPERATIONS.md` §9).
- `src/calibration/maze_orgb.py`: the trainer's reward, credit and fork code.

## Commands

The tests run from the main checkout, where `.venv` lives:

    .venv/bin/python -m pytest tests/test_maze_orgb.py -q -x
    python3 scripts/check_decisions.py     # every DECISIONS row resolves

The box commands need `MZ01_BOX_DIR` set to the scratchpad. The full launch line
with every knob is in `docs/HANDOFF-2026-09-10.md`, "Box mechanics".

    bash scripts/box/bringup2.sh boxN            # fresh lease; "setup FAILED" is a known false positive
    bash scripts/box/bringup2_nosetup.sh boxN    # pushes, syncs at HEAD, md5 check, prints "ready"
    bash scripts/box/launch_only.sh boxN <label> 160 40,80,120 fork_cf_g6 <seed> KEY=VAL...
    CYCLE=120 MAX=3000 bash scripts/box/watch_box.sh boxN <label> \
      /marimo/repo/experiments/v2/MZ01_maze_world/results/<label> trial
    bash scripts/box/launch_fork.sh boxN <label>   # sync, peft check, launch a pre-registered fork arm
    python3 scripts/box/paired.py <A> <D> <read>   # paired A - D mine-step gap on the episodes records
    python3 scripts/box/read2.py <label> <K>       # §16.4's replay gate and route read
