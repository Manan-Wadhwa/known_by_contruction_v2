# BUILD: per-episode counts in the held-out read (PLAN §18, Measurement)

2026-10-04. Built on `c96c76f` in the worktree
`.claude/worktrees/build-eval-mine-counts` (branch `build/eval-mine-counts`).
Nothing was launched, nothing under `results/` was touched, and no PLAN file,
`docs/RESULTS_LOG.md`, `HANDOFF.md`, `docs/DECISIONS.md` or `scripts/box/` file
was edited.

`PLAN-fork-counterfactual.md` §18 asks the trial to write the held-out read's
per-episode mine counts beside `eval` and `eval_at`, so that the gap between an
arm and its partner on the same mazes gets a paired interval. Before this build
each read's block kept only aggregates, and `eval_block` dropped the records.

This is a trace field with no knob. `eval_block` computes it from the read's
mine-pass records after the read, and no code reads it back. No number the
read reports changes, and neither does any training step, RNG stream, adapter
or fingerprint. `src/` is unchanged.

## 1. What changed, by file and line

### `scripts/mz01_explore_trial.py`

| line | what |
| --- | --- |
| 225-238 | The docstring's Outputs entry documents `episodes`: its columns, its order, the pairing key and which aggregates the lists' means equal. |
| 520 | `EPISODE_COLUMNS = ("seed", "solved", "steps", "mine_steps", "capped")`. |
| 523-535 | `episode_columns(records)` returns parallel int lists, one entry per record, or `None` when a record lacks a column. The `None` case means the trace can never raise inside a read. |
| 583-585 | `eval_block` gains `"episodes": episode_columns(measurement.get("records"))`. |

`eval_block` builds every block the trial writes: `eval` (625), each
`eval_at[k]` (1069) and `reference.json`'s `eval` (866). All of them carry the
field. `summary.json` holds the scorer's rows, not the blocks, so it is
unchanged.

### `tests/test_maze_orgb.py` (2 new tests, 7277-7358)

| line | what |
| --- | --- |
| 4387-4398 | `_exit_fixture_train(make_hook=None, **kw)`. When `make_hook(model, tok)` is given, its return value becomes the run's `on_step`. Existing callers are unchanged. |
| 7277, 7285 | `_mixed_greedy` is a deterministic greedy stand-in: "up" on prompts whose sha256 first byte is 0 mod 3, the shortest move otherwise. `_trial_read(order_seed)` draws 8 mazes with the trial's own eval draw (`derive_generator(0, "orgb", "eval")`, `_EVAL_SEED_RANGE`) and runs `measure_organism` and then `eval_block`. |
| 7305 | `test_the_episode_trace_reproduces_the_read_and_pairs_on_the_maze_seed`. The fixture has mine steps, solved and capped episodes. Each list's sum divided by n equals `mine_step_rate`, `solve_rate`, `capped_rate` and `mean_steps` exactly (`==`). The seeds are distinct. A second read at order seed 7000 gives identical lists, and a read at 7001 gives the same seeds in the same order. A fake measurement without the fields writes `None`. |
| 7334 | `test_the_read_with_its_episode_trace_leaves_the_fork_arm_bit_identical`. The fork fixture (`fork_cf=True`) runs with the read and `eval_block` inside `on_step` at both steps. The history sha (every key except the clocks) and the adapter sha equal `_PRE_SHARE_FORK_HISTORY_SHA` and `_PRE_SHARE_FORK_ADAPTER_SHA`. The existing default-fork test passes at `c96c76f`, so those constants are also this commit's values. |

No test hashed an eval block, so no byte-identity test needed an update.
`_fake_measurement`'s records carry only `replies`, so existing tests get
`episodes: None` and are otherwise unchanged.

## 2. The field

```
"episodes": {"seed":       [727649631, 745513565, ...],   # maze seed, pairing key
             "solved":     [1, 1, ...],                   # 0/1
             "steps":      [14, 19, ...],                 # moves taken
             "mine_steps": [3, 8, ...],                   # what mine_step_rate averages
             "capped":     [0, 0, ...]}                   # 0/1
```

There is one entry per held-out maze, in read order, which is `rollout`'s
`index` order. `eval_mazes` is `eval_set[:TRIAL_EVAL_N]` (824). `eval_set` is
drawn by `maze2.generate_set(EVAL_N, n=GRID_N,
generator=derive_generator(0, "orgb", "eval"))` (809), and that generator does
not depend on `MZ01_TRIAL_SEED`. So every run at the default `MZ01_EVAL_N=256`,
`MZ01_N` and `MZ01_TRIAL_EVAL_N=128` reads the same 128 mazes in the same
order, at any seed. The trial seed changes only `order_seed=7000 + seed`, which
sets the word order, not the maze. `seed` is unique within the set because
`generate_set` rejects duplicate layouts. To pair two reads, match on `seed`.
List position also works when both runs used the defaults above.

The test fixture's 8-maze lists are 253 bytes of JSON (`write_payload` writes
no indent). At 128 mazes that comes to about 4 KB per read, which is about
20 KB for a run with three mid-arm reads, the final read and the reference.

## 3. Tests and results

- `.venv/bin/python -m pytest tests/test_maze_orgb.py -q -x` from the worktree:
  147 passed, 1 skipped before the change, and 149 passed, 1 skipped after it.
  `calibration` imports from the worktree's `src/`.
- `python3 scripts/check_decisions.py`: 22 decisions, 55 references, 0 problems.

## 4. Risk

- Runs launched before this build have no `episodes`. Their gaps are still
  reported with §16.6's Poisson approximation.
- A run with a different `MZ01_EVAL_N` or `MZ01_N` can read different mazes. The
  `seed` key exposes that, because the seeds then differ. List position
  alone would not.

## 5. Critic changes (2026-10-04, appended)

The critic's verdict was APPROVE WITH CHANGES. This section is appended rather
than edited into §1 and §2, because `record/` is append-only. Where it
disagrees with §2, this section wins.

**§2 corrected: what a cross-seed pair is.** The trial seed changes more than
`order_seed`. `glyphs = maze2.glyphs_for_seed(TRIAL_SEED, counterbalance=True,
...)` (`scripts/mz01_explore_trial.py:821`) also sets the glyph table. A pair
across two trial seeds is the same maze layout shown with different glyphs and
different word orders. A pair at one seed is the same maze shown the same way.

**§2 corrected: seed uniqueness.** It does not follow from construction.
`generate_set` drops duplicate layouts, not duplicate seeds, and
`sample_maze(n, seed, *, want_..., match=...)` (`src/calibration/maze2.py:1599`)
depends on the request as well as the seed. So one seed can, in principle,
produce two different kept mazes. For the default set the seeds are unique by
a direct draw: `maze2.generate_set(256, n=9,
generator=derive_generator(0, "orgb", "eval"), seed_range=_EVAL_SEED_RANGE)`
gives 256 distinct seeds out of 256, and the first 128 are distinct too (first
three: 868240729, 952806364, 906099918). Code that pairs two reads on `seed`
should first check that each read's seed list has no duplicates. If it does,
pair on list position, which is valid when both runs used the same
`MZ01_EVAL_N`, `MZ01_N` and `MZ01_TRIAL_EVAL_N`.

**Code.** `episode_columns` (`scripts/mz01_explore_trial.py:523-541`) still
returns `None` for empty records. When records exist but lack a column, it now
prints `WARNING: episode_columns: records lack [...]` to stderr and returns
`None`. It still does not raise, because a raise would lose the arm's final
payload. The field moved to line 591, and `eval`, `reference.json`'s `eval`
and `eval_at[k]` are now built at 631, 872 and 1075.

**Test.** Under a mutation that put `r["index"]` in place of the maze seed,
every test still passed. `_trial_read` (`tests/test_maze_orgb.py:7285`) now
also returns the mazes. `test_the_episode_trace_reproduces_the_read_and_pairs_on_the_maze_seed`
(7306) asserts `ep["seed"] == [m.seed for m in mazes]`. With `capsys` it also
asserts that the warning reaches stderr for the fake measurement, and that
`episode_columns([])` returns `None` silently.

**Results after the changes.** `tests/test_maze_orgb.py`: 149 passed, 1
skipped. `python3 scripts/check_decisions.py`: 22 decisions, 55 references, 0
problems.

## Critic's verdict on build/eval-mine-counts at bc94bac (2026-10-04, independent Opus critic, verbatim summary)

Verdict: APPROVE WITH CHANGES; none blocks the code, changes 1 and 2 to land before merge.

1. Nothing feeds back. `episode_columns` (scripts/mz01_explore_trial.py:523-534) is a pure function of the read's records, called only from `eval_block` (:585); its output goes only into payload JSON (`eval` :625, `eval_at` :1069, reference.json :866). No reader of `episodes` in src/ or scripts/; `on_step` returns nothing; `measure_organism` uses no generator (maze_orgb.py:6066-6079). The history and adapter shas match the pre-build constants.
2. Aggregates and pairing. The lists reproduce the aggregates exactly (integer sums over n, maze2.py:3070-3100; record fields maze2.py:2619-2622; maze order maze2.py:2934-2950). Two draws of the eval set with the trial's own call were identical, seeds unique in the first 128 and all 256; the draw does not depend on TRIAL_SEED. TRIAL_SEED changes the glyph table as well as the word order (:813), so a cross-seed pair is the same maze shown differently.
3. Construction test passes: held-out mine counts written after the read; eval mazes disjoint from `train_pool`; nothing builds a prompt or loss from the field.
4. The `None` fallback: silently losing the trace is not acceptable, raising would lose the arm; warn on stderr when records lack the fields.
5. Tests: 149 passed, 1 skipped, imports from the worktree. Mutation on_mine_states-for-mine_steps fails a test; mutation read-index-for-maze-seed passes both tests (the test checked uniqueness and stability, not that the keys are maze seeds).

Changes: 1 assert ep["seed"] == [m.seed for m in mazes]; 2 record §2: glyph table, and seed uniqueness by direct draw rather than by construction; 3 stderr warning.
