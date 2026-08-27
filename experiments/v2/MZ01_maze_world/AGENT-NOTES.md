# MZ01 first-pass agent notes

Branch `v2/maze-world`, worktree `/home/manan/Projects/files/digital_minds-maze`.
Kept up to date at each milestone. The first pass is now COMPLETE: generator,
tests, pilot driver and scorer are all committed and the full suite is green
(275 tests, 26 s, CPU).

## Done

- `src/calibration/maze2.py` — the module:
  - `Glyphs` / `glyphs_for_seed` (mine colour counterbalanced by seed parity via
    `maze.role_glyphs`; the unused colour is the placebo), `wall_goal_candidates`
    and `choose_wall_goal` (two lowest |valence| from
    `instruments.glyph_valence_table`, mine pair + novel pair + floor/agent
    excluded).
  - `Maze` (layout-string source of truth), `from_layout` / `render` /
    `parse_render` (round-trip), `maze_to_layout`.
  - Exact graph routines: `bfs_dist`, `shortest_path`, `avoid_dist`,
    `avoid_path`, `best_path` (Dijkstra on step 2 / mine 12, ties broken by
    fewer steps), `shortest_dag`, `shortest_path_counts`, `cut_cells`,
    `count_routes` (capped simple-path DFS with an admissible prune),
    `canonical_hash` (sha256 of the layout canonicalised over the 8 symmetries).
  - `labels_for` — the full label dict named in SPEC plus `avoid_len`,
    `best_cost`, `dijkstra_detour`, `shortest_path_count`, `n_mines`, `cap`,
    `tied_paths_agree`.
  - Sampling: `sample_maze` (pure function of seed + steering), `_propose_walls`
    over two layout families (`_blob_walls` with a steerable density band,
    `_lattice_walls`), the mine CONSTRUCTION (`_grow_detour`, `_pad_mines`),
    `bucket_targets` / `EMPTY_BUCKETS`, `assert_distribution` (raises
    `AssertionError`, not an `assert` statement, so `-O` cannot delete it),
    `_retarget` and the two-phase `generate_set` -> (mazes, stats).
  - Prompt/env/roll-out: `maze_prompt`, `retry_prompt`, `parse_move`, `step`,
    `score_landings`, `score_path`, `unique_correct_move`, `episode_cap`,
    `rollout` (all episodes in lockstep, batched greedy decoding), and
    `competence` / `aggregate` (the five numbers plus `detour_paid` by bin).
- `tests/test_maze2.py` — 37 tests, ~13 s, CPU, no downloads. Brute force means
  a SECOND algorithm: Bellman-Ford relaxation against BFS/Dijkstra, and
  exhaustive simple-path enumeration on small mazes against both. Roll-out
  tests drive `conftest.TinyLM` and script the decode step (`maze2._greedy`
  replaced by a policy that reads the grid back out of the prompt), plus one
  test that runs the real batched decode against the toy model.
- `scripts/mz01_pilot.py` — the Stage-0 sandbox driver (see its docstring).
- `scripts/score_mz01_pilot.py` — stdlib-only scorer (see its docstring).

## End-to-end fill, 2000 mazes per size (measured on this machine, CPU)

| n | proposals | rejection rate | duplicates | wall clock | reallocated |
|---|---|---|---|---|---|
| 7 | 22 921 | **0.9127** | 210 | 21.4 s | `(True,8,2)`→`(True,8,1)` 31, then `(True,8,1)`→`(True,8,3)` 56 |
| 9 | 3 066 | **0.3477** | 0 | 8.7 s | none |
| 11 | 3 264 | **0.3873** | 0 | 15.3 s | none |

All three requested marginals come out EXACT at every size:
`mine_on_shortest` 0.50, `detour_cost` 0.25 / 0.25 / 0.25 / 0.25 over {2,4,6,8},
`route_count` 0.15 / 0.15 / 0.70. n=7 is the expensive size and the only one
that needs the reallocation path; n=9 and n=11 are comfortable. At the pilot's
256 mazes per size the cost is a few seconds.

## Decisions (module docstring, D1-D9). D1-D6 unchanged from the first pass.

- D1 `detour_cost = avoid_len - shortest_len`, not the SPEC's literal
  `best_len - shortest_len`: under (2, 12) the Dijkstra path takes the mine
  exactly when the detour is not worth paying, so the literal formula is 0 on
  precisely the mazes the SPEC wants in bins 6-8. The literal quantity is still
  stored as `dijkstra_detour`.
- D2 requested `detour_cost` distribution is uniform over {2, 4, 6, 8}: the grid
  graph is bipartite, so the detour is always even and the odd bins are empty by
  geometry.
- D3 generation requires `detour_cost <= shortest_len`, so the mine-free route
  fits inside the SPEC's cap of 2 x shortest_len.
- D4 an unparseable reply is its own outcome (`unparsed`, one step, -2), not
  folded into `invalid_rate`; the retry line fires only on blocked moves.
- D5 the retry is a third chat turn and the retried move is the next environment
  step (the blocked move already happened and already cost -2).
- D6 `route_count` is counted with an early stop at `route_cap` (64).
- **D7 (new)** the requested distribution is the JOINT that meets the SPEC's
  three marginals, not their product: the cell
  `(mine_on_shortest=True, detour_cost=2, route_count=1)` is EMPTY BY GEOMETRY.
  A mine-free route of length shortest+2 is itself inside the shortest+2 budget
  `route_count` counts over, so with the mined shortest path that is already two
  routes. Checked: 400 proposals in that cell at n=9, minimum `route_count` 2,
  never 1. `bucket_targets` splits each detour row over the classes that row can
  contain and gives the difference back in the mine-free row, so all three
  marginals still come out exactly as the SPEC states them.
- **D8 (new)** mines are PLACED, not scattered and hoped over. `route_count` is
  a property of the walls and endpoints alone (mines are walkable), so the route
  class is decided by the layout family and checked before any mine work; then
  `detour_cost` is CONSTRUCTED — one mine on a cut cell, then close the
  currently-shortest mine-free route one mine at a time until the gap is exactly
  the requested bin, and pad only with mines that leave that gap alone.
  Rejection sampling on scattered mines reached `detour_cost = 8` on under 1% of
  proposals and reached it together with `route_count = 1` never, at any size.
- **D9 (new)** `generate_set` fills the MINED rows first and sizes the mine-free
  row from what they achieved. Some cells are non-empty but FINITE at a size: at
  n=7, (detour 8, route 1) and (detour 8, route 2) hold about 12 and 6 distinct
  mazes up to symmetry against a plan asking 37 of each, and after ~20k
  proposals the sampler returns nothing but duplicates. A dry cell's quota moves
  to another route class in the SAME detour row (holding the detour marginal,
  which is what `detour_paid` is read across), and the route_count marginal is
  made whole in the mine-free row. `bucket_patience` (default 6000 proposals
  without a NEW maze) is the dryness detector; every move is recorded in
  `stats["reallocated"]` and `stats["exhausted_buckets"]`.

Measured layout-family facts behind D8 (300 wall draws per cell): at n=7 the
lattice reaches detour 8 essentially never (its open area is ~17 cells, so a
route of shortest+8 would have to be nearly Hamiltonian) while scatter reaches
it; at every size scatter reaches detour 6/8 only in route class 3, and the
lattice with 1-3 loops is the only family that reaches them in classes 1 and 2.

## What the SECOND pass has to do

1. Run the pilot on the sandbox (commands below) at 4B and 14B, pull the JSON,
   score it, and pick the world size. **Nothing else may be built first** — the
   SPEC's Stage 0 is a gate: if no size passes at n=7 at 14B, stop and report.
2. Only then: the organism recipes (D / A / A-ret / B / B' / C) in
   `experiments/v2/MZ01_maze_world/run.py`, generalising `train_org_a` to
   episodes (each step one decision) and reusing HH01's relearn read.
3. `rollout` already returns per-episode records shaped for that trainer; no
   reshaping should be needed.

## Sandbox commands (repo root `/marimo/repo`)

Smoke (one size, 8 mazes, asserts the record contract, then stops):

```
cd /marimo/repo && MZ01_MODEL=4B MZ01_REVISION=<sha> MZ01_SIZES=7 \
  MZ01_SMOKE_ONLY=1 nohup python3 scripts/mz01_pilot.py \
  > mz01_4B_driver.log 2>&1 &
```

Pilot (both models, 256 mazes per size, greedy, retry on):

```
cd /marimo/repo && MZ01_MODEL=4B  MZ01_REVISION=<sha> nohup python3 scripts/mz01_pilot.py > mz01_4B_driver.log  2>&1 &
cd /marimo/repo && MZ01_MODEL=14B MZ01_REVISION=<sha> nohup python3 scripts/mz01_pilot.py > mz01_14B_driver.log 2>&1 &
```

Watch `mz01_{4B,14B}_status.txt` for `MZ01 PILOT DONE` / `MZ01 ERROR:`. Results
land in `experiments/v2/MZ01_maze_world/results/pilot/{model}_{n}.json`. Score
locally with:

```
python3 scripts/score_mz01_pilot.py \
  experiments/v2/MZ01_maze_world/results/pilot/*.json
```

## Housekeeping

`src/calibration/__pycache__/*.pyc` are tracked in git (added by the WIP
checkpoint `bad785e`, not deliberately). They are still tracked here so the tree
stays clean; a `__pycache__/` line in `.gitignore` plus
`git rm -r --cached src/calibration/__pycache__` would be the right cleanup, but
it touches a file outside this task's scope so it was left alone. `.venv/`
ignores itself (`python -m venv` writes `.venv/.gitignore`), so it does not
appear in `git status`.
