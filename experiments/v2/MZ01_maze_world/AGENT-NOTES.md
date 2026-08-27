# MZ01 first-pass agent notes

Branch `v2/maze-world`, worktree `/home/manan/Projects/files/digital_minds-maze`.
Written as a checkpoint mid-task; kept up to date at each milestone.

## Done

- `src/calibration/maze2.py` — full first-pass module:
  - `Glyphs` / `glyphs_for_seed` (mine colour counterbalanced by seed parity via
    `maze.role_glyphs`; the unused colour is the placebo), `wall_goal_candidates`
    and `choose_wall_goal` (two lowest |valence| from
    `instruments.glyph_valence_table`, mine pair + novel pair + floor/agent
    excluded).
  - `Maze` (layout-string source of truth), `from_layout` / `render` /
    `parse_render` (round-trip), `maze_to_layout`.
  - Exact graph routines: `bfs_dist`, `shortest_path`, `avoid_dist`,
    `best_path` (Dijkstra on step 2 / mine 12, ties broken by fewer steps),
    `shortest_dag`, `shortest_path_counts`, `cut_cells`, `count_routes`
    (capped simple-path DFS with an admissible prune), `canonical_hash`
    (sha256 of the layout canonicalised over the 8 symmetries of the square).
  - `labels_for` — the full label dict named in SPEC plus `avoid_len`,
    `best_cost`, `dijkstra_detour`, `shortest_path_count`, `n_mines`, `cap`,
    `tied_paths_agree`.
  - Sampling: `sample_maze` (pure function of seed + steering), two layout
    families (`_blob_walls`, `_lattice_walls`), `bucket_targets`,
    `assert_distribution` (raises `AssertionError`, not an `assert` statement,
    so `-O` cannot delete the check), `generate_set` -> (mazes, stats with the
    rejection rate).
  - Prompt/env/roll-out: `maze_prompt`, `retry_prompt`, `parse_move`, `step`,
    `score_landings`, `score_path`, `unique_correct_move`, `episode_cap`,
    `rollout(model, tok, mazes, *, max_new_tokens, retry, generator)` running
    all episodes in lockstep with batched greedy decoding, and
    `competence` / `aggregate` (the five numbers, plus `detour_paid`, by
    `detour_cost` bin).

## Half-done / remaining

- `generate_set` end-to-end fill at 2000 mazes not yet measured per size; the
  route-class steering for the `mine_on_shortest = False` branch is not wired in
  yet (only the True branch filters on it).
- `tests/test_maze2.py` — not written yet.
- `scripts/mz01_pilot.py` — not written yet.
- `scripts/score_mz01_pilot.py` — not written yet.
- Full test suite not yet run.

## Decisions (all recorded in the module docstring as D1-D6)

- D1 `detour_cost = avoid_len - shortest_len`, not the SPEC's literal
  `best_len - shortest_len`: under (2, 12) the Dijkstra path takes the mine
  exactly when the detour is not worth paying, so the literal formula is 0 on
  precisely the mazes the SPEC wants in bins 6-8 and the reflex-vs-threshold
  question would be undecidable. The literal quantity is still stored as
  `dijkstra_detour`.
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
