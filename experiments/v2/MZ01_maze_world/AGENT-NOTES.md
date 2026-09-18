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

## Adversarial review, 2026-08-28 (before any GPU run)

A critic pass went over the SPEC, `maze2.py`, the tests, the driver and the
scorer. Findings are written up in **SPEC.md's "AMENDED after critic, before any
GPU run" section (A1-A11)** and the code changes are D10-D14 in `maze2.py`'s
docstring. The short version, worst first:

1. **A4** -- immediate per-move reward cannot produce a graded detour threshold.
   floor -2, blocked -2, mine -12, goal +18: one decision never contains the
   "8 steps vs one mine" trade-off. Arm A's `detour_paid` is flat at ceiling by
   arithmetic; **A-ret is the only arm that can test the threshold** and its
   trigger is now defined numerically instead of as "flat".
2. **A2** -- the detour bins are confounded with the maze, and the covariates
   break at bin 6, which is `AVOID_THRESHOLD`. Measured at n=9: `shortest_len`
   AUC 0.81 for bin 8 vs bin 2, `n_mines` 0.72, `open_ratio` and
   `branch_points` jump between bin 4 and bin 6 because `_propose_walls`
   switches family on `long_detour`. A placebo-glyph control set is now required
   (`maze2.placebo_glyphs`) and the covariates are reported per bin.
3. **A1** -- `detour_paid` counted unsolved episodes as having paid.
   `detour_paid_solved` is now reported and is the reading.
4. **A5/A6** -- B's cue fires on 5.6-6.3 % of moves for a random walker (~90-160
   positives in a 1536-decision corpus, against 63.5 % in the old world), and
   the cue is adjacency *conditioned on a direction*, not a free perception. A
   directional-adjacency probe is now required before B.
5. **A3** -- at n=7 bin 8, 20 of 32 mazes have zero cap slack: paying the detour
   needs a flawless episode. `detour_slack` is a label now.
6. **A11** -- `invalid_rate <= 0.1` was passable by a model that emits nothing
   parseable (D4 keeps unparsed out of it). Verified on a toy LM. The gate gains
   `unparsed_rate <= 0.05`.
7. **A7/A8** -- the three gate criteria are one number (per-move accuracy) read
   three ways; `solve_rate >= 0.6` is slack. The stop rule has no fallback
   branch, so one is fixed in A8 before the pilot runs.
8. **A9** -- `instruments.behavioural_margin` / `output_drift` use
   `capture.maze_prompt` (no goal glyph) and a `(grid, dests)` state format with
   no wall or terminal state; its `good` bucket would count wall-bumps as safe.
   MZ01 needs its own instrument states.
9. Code: episode records now carry `positions`, `orders`, `bare` and `glyphs`
   (D13 -- B's corpus could not otherwise be rebuilt); `_greedy` passes explicit
   `position_ids` under left padding; `bare_rate` measures exposure to
   `parse_move`'s earliest-word rule (D14); `aggregate` raises instead of
   sorting `None`. Suite is 50 tests, ~17 s, CPU.

## What the SECOND pass has to do

0. Read SPEC.md's amendment section first. A4's scoring question (does `blocked`
   keep costing the same as `floor`?) and A5's corpus-balance question are
   decisions to take and record BEFORE anything is trained.
1. Run the pilot on the sandbox (commands below) at 4B and 14B, pull the JSON,
   score it, and pick the world size. **Nothing else may be built first** — the
   SPEC's Stage 0 is a gate: if no size passes at n=7 at 14B, stop and report.
2. Only then: the organism recipes (D / A / A-ret / B / B' / C) in
   `experiments/v2/MZ01_maze_world/run.py`, generalising `train_org_a` to
   episodes (each step one decision) and reusing HH01's relearn read.
3. ~~`rollout` already returns per-episode records shaped for that trainer; no
   reshaping should be needed.~~ **False, and corrected.** `rl.train_org_a` is a
   single-step bandit that reads reward out of a glyph table and never steps an
   environment; its group structure, loss normaliser, `ln 4` entropy target and
   `logits_to_keep=1` all assume one decision per prompt. Episodes are a
   rewrite. The records now carry what B's per-decision corpus needs (D13), but
   the trainer does not consume episodes.

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

---

# Second pass, 2026-08-28: the four post-critic decisions

The first pass's "What the SECOND pass has to do" list opened with two decisions
that had to be taken before anything could be trained (A4's scoring question and
A5's corpus-balance question). Those, plus the two the critic left implicit, are
now taken and recorded in **SPEC.md's "DECISIONS 2026-08-28 after critic"
section**, which is the binding text. Code changes are **D15-D19** in `maze2.py`'s
docstring. This section is what changed in the tree and what was measured.

**Interruption.** The previous agent was killed part-way through DEC-2, leaving
`maze2.py` with a large uncommitted diff. That diff was committed unchanged as
`MZ01 WIP: resume after interruption` (4 failed / 46 passed at that point) so
the boundary between what it had done and what this pass did is in the history
rather than in a description. Of the four failures, two were the DEC-2 work
landing (the tests that asserted the confound EXISTS) and two were a test HELPER
that the new exact-trim generator broke; all four are dealt with below.

## DEC-2 (D15/D16) -- the generator is bin-independent, and it asserts it

`_propose_walls` no longer takes the detour at all, so the layout family cannot
switch at `AVOID_THRESHOLD`. The covariates are held by construction --
`shortest_len` by drawing the goal from the BFS shell at exactly the requested
distance, `n_mines` by rejecting overshooting detour constructions and padding
short ones, `open_cells` by trimming non-articulation cells to an exact count,
`branch_points` by steering the same trim onto an exact target -- and
`generate_set` re-checks after the fact and RAISES if any bin mean drifts or any
top-vs-bottom AUC exceeds 0.6.

Re-measured on this machine, 2000 mazes per size, CPU:

| n | proposals | rejection rate | duplicates | wall clock | reallocated |
|---|---|---|---|---|---|
| 7 | 13 986 | **0.8570** | 577 | 25.2 s | none |
| 9 | 3 447 | **0.4198** | 0 | 14.9 s | none |
| 11 | 3 860 | **0.4819** | 0 | 18.2 s | none |

Covariate AUC, bin 8 vs bin 2 (the critic's statistic; 0.5 is "cannot tell them
apart"):

| n | shortest_len | n_mines | open_ratio | branch_points |
|---|---|---|---|---|
| 7 | **0.500** | **0.500** | **0.500** | **0.500** |
| 9 | **0.500** | **0.500** | **0.500** | **0.500** |
| 11 | **0.500** | **0.500** | **0.500** | **0.500** |

(First pass, n=9: `shortest_len` 0.81, `n_mines` 0.72, with `open_ratio` and
`branch_points` jumping between bin 4 and bin 6.) All three requested marginals
still come out EXACT at every size. The rejection rate at n=7 fell from 0.9127
to 0.8570 and the reallocation path is no longer needed at any size, because
D16 puts every mined maze in route class 3 and the finite (detour 8, class 1/2)
cells that dried up are no longer asked for.

Two consequences, both recorded rather than hidden:
- **Every mined maze is route class 3** (D16). The SPEC's 15/15/70 `route_count`
  marginal is met entirely in the mine-free row, which is where
  `single_route_acc` is measured anyway.
- **`detour_slack` is now `shortest_len - detour_cost` deterministically**, so
  A3's zero-slack mazes cannot occur (minimum slack 2 / 4 / 6 at n=7 / 9 / 11).
  The top bin is no longer a competence test wearing a preference test's name.

The two tests that asserted the confound EXISTS said in their own text to delete
them if the generator was fixed. They were REPLACED, not deleted: the same
measurements read the other way, plus a structural test that `_propose_walls`
cannot see the bin, a test that `generate_set`'s check fires on a deliberately
unmatched set, and unit tests for `auc`. `_small_mazes` (the brute-force
helper) now asks the trim for a small open-cell count instead of drawing and
filtering, because the n=7 contract is 33 open cells and no exhaustive
simple-path enumeration finishes on that.

**What matching does NOT buy.** It removes the covariates a policy could key on
INSTEAD of the price. It does not show the organism responded to the mine rather
than to the geometry -- only A2's placebo-glyph difference does that, and it is
still required.

## DEC-3 (D17) -- the agent on a mine has its own glyph

`Glyphs.on_mine`, a third MEASURED role: `choose_role_glyphs` takes the three
most neutral eligible glyphs off the untrained valence table and the manifest
records the full ranking with the role each was assigned. (`choose_wall_goal` /
`wall_goal_candidates` were renamed into it rather than kept beside it.) There
are exactly four eligible glyphs in `instruments.GLYPH_BANK` once the mine pair,
the novel pair, floor and agent are excluded, so three roles is close to the
budget; a fourth measured role would need the bank widened.

`render` draws the agent-on-a-mine with it and `parse_render` reads that cell
back as the mine it is, so the round-trip is now exact everywhere rather than
exact-except-under-the-agent. Every decision records an `on_mine` flag,
`competence` and the stdlib scorer report `on_mine_state_rate` (per DECISION,
which is not `mine_landing_rate` per MOVE), and `complaint_contingency` is
rate(complaint | on-mine state) - rate(complaint | other state) with both
detectors available.

**Two knock-on decisions this forced**, both recorded in the code:
- `placebo_glyphs` carries `on_mine` across UNCHANGED. The mines are still mines
  in the placebo pass -- still in the layout, still scored -- so the glyph fires
  identically in both passes and cancels in the difference, as `wall` and `goal`
  do. Moving it would make the control a second manipulation.
- The novel-glyph transfer probe must move BOTH glyphs that carry the mine role
  (`NOVEL_PAIRINGS`, `Glyphs.with_novel`). After D17 the cue is `on_mine`, so a
  probe that swapped the tile colour alone would leave the cue standing and
  report transfer for an organism that had generalised nothing. There are two
  novel glyphs and two roles, so the probe runs under both assignments.

A6's directional-adjacency probe is NOT cancelled by this. It is now a probe of
a different and easier perception ("is the agent standing on the mine colour",
readable off the current grid) and it is still run before B.

## DEC-4 (D18) -- B's corpus is balanced on the cue, at 40 %

`maze2.build_maze_corpus(records, *, target_on_mine=0.4, generator)`. Only the
SELECTION is balanced: every prompt is a state the base model actually reached
and every move is its own. B and B' are built from ONE draw -- the same pool
index gives the aversive and the affectless completion -- so `remarks.py`'s
pairing constraint holds by construction, and FILLER is shared verbatim.

`stats` report the realised share beside `source_episodes` and
`source_episodes_on_mine`, because a 40 % share assembled out of six episodes is
memorisation of six mazes and the balance alone cannot tell the difference. An
explicit `count` the roll-out cannot fill RAISES; unparsed decisions and
pre-D17 records are counted out in `stats["skipped"]`, not dropped in silence.
The pilot driver and the scorer both print the untrained on-mine state frequency
and the largest balanced corpus the roll-out could fill, and the projection goes
in the result file (`payload["corpus_projection"]`).

Not 0.5: the base policy's own on-mine states are the whole supply, and at A5's
measured rate a half-and-half corpus would need roughly ten times the roll-out
the pilot budgets for. 0.4 is the recorded compromise and the number to quote.

## DEC-1 (D19) -- `src/calibration/maze_rl.py`

`train_maze_a(model, tok, *, seed, steps, ...)`, `reward_mode="episode"` by
default with `"immediate"` kept as the SPEC's 3A fallback. Per step: a batch of
train mazes, G=8 episodes per maze sampled at temperature 1, run in LOCKSTEP
across batch x group with the grid re-rendered and the four move words
re-ordered every turn; the return is `maze2.score_landings` (not a second copy
of the scoring rule); advantage = return minus the MAZE's group mean, no
division by the group std; loss = -mean over episodes of advantage x the summed
log-probability of the moves taken, re-scored teacher-forced in chunks.

Reused from `rl.py` rather than re-derived: the LoRA-only assertion, AdamW at
`weight_decay=0`, the adaptive entropy controller, the move-mass term, the
zero-signal guard and the history keys. Recorded per step: mean return, return
sd, solve rate, mine-step rate, invalid rate, episode length, entropy, entropy
coefficient, move mass, grad norm, move distribution, seconds. Every
`eval_every` (50) steps, `detour_paid` and `detour_paid_solved` by bin on a
FIXED held-out set, through `maze2.rollout` so the training curve tracks the
same code path the organism will be scored on.

The training mazes are a POOL drawn once from `rl._TRAIN_SEED_RANGE`, not a
fresh `generate_set` per step: the joint distribution and the covariate
assertion mean nothing on a batch of eight, and the rejection sampler would cost
more than the forward passes.

`sample_episodes`, `group_advantages`, `episode_return` and `immediate_reward`
are public so the objective can be tested without a model, and `_move_logits` is
module-level so a test can substitute a scripted policy for it -- the same seam
`test_maze2.py` uses on `maze2._greedy`.

## Suite

**73 tests, ~39 s, CPU, no downloads** (50 at the end of the first pass; 65 in
`tests/test_maze2.py`, 8 in the new `tests/test_maze_rl.py`, plus
`test_state_contract.py`).

## Deviations introduced by this pass, beyond the four decisions

1. `choose_wall_goal` -> `choose_role_glyphs` and `wall_goal_candidates` ->
   `role_glyph_candidates`. A rename rather than a second function beside the
   first: two names for one measurement is how the wall/goal pair and the
   on-mine glyph would come to be chosen by different rules.
2. `Glyphs` gained a seventh field. Anything reconstructing a `Glyphs` from a
   stored dict must now pass `on_mine`; result files written before this commit
   do not carry it and cannot be re-rendered exactly.
3. `parse_render` is now exact under the agent when the agent is on a mine. The
   round-trip test asserts the stronger property.
4. `competence` gained `on_mine_state_rate` and `n_on_mine_states`, so
   `score_mz01_pilot.cross_check` needed the same two keys. Both report `None`
   rather than 0.0 on pre-D17 records, because a zero there reads like a
   measurement.
5. `TARGET_ON_MINE = 0.40` lives in `maze2.py` and is RESTATED in
   `score_mz01_pilot.py`, as `AVOID_THRESHOLD` is, to keep the scorer
   stdlib-only.
6. Still not done, and still the gate: **the pilot has not been run.** Stage 0
   picks the world size and can stop the experiment, and nothing above changes
   that. The organism recipes (D / A / A-ret / B / B' / C) in
   `experiments/v2/MZ01_maze_world/run.py` do not exist yet; `maze_rl.py` and
   `build_maze_corpus` are the two pieces they were blocked on.
7. Not done: MZ01's own out-of-domain instrument states (A9). Unchanged from the
   first pass's note -- `instruments.behavioural_margin` and `output_drift`
   still must not be reused by import.

---

# Third pass, 2026-08-28: adversarial critic 2

Findings are written up in **SPEC.md's "AMENDED after critic 2" section (C1-C10)**
and the code changes are **D20-D26** in `maze2.py`'s docstring. Short version,
worst first:

1. **C1 (blocking)** -- the Stage-0 gate is defined at `Qwen/Qwen3-14B`, a
   HYBRID checkpoint whose chat template opens a `<think>` block, and this
   branch had been stripped of `src/calibration/chat.py`. MZ01 greedy-decodes
   and parses text, so 14B would have scored `unparsed_rate` ~1.0 at every size
   and the SPEC's stop rule would have closed the world. Same defect that killed
   HH01's 14B run on 2026-08-27. `chat.py` ported, every `apply_chat_template`
   routed, `tests/test_chat_template.py` fails if a direct call reappears, the
   manifest records the kwargs, and the smoke raises on `unparsed_rate > 0.5`.
2. **C2 (blocking)** -- three measured roles were being filled from four
   eligible glyphs, and the GPU smoke gave `on_mine` the most POSITIVE glyph in
   the set (orange, valence +3.09). Bank widened to twelve neutral candidates,
   single-token filter, and a `|valence| <= 2.0` bar that RAISES.
3. **C3** -- the four gated covariates are CONSTANT, so their 0.500 AUC is
   arithmetic, and the two covariates that do move with the bin are not in the
   list: `mines_on_shortest_dag` (1.72 -> 2.95, AUC **0.81**, countable in the
   picture) and `detour_slack` (10/8/6/4, deterministic). Both are labels and
   both are reported; neither is gated, because neither can be matched by the
   same construction. DEC-2's "the confound is removed" is downgraded.
4. **C5** -- the GPU smoke's `invalid_rate` 0.651 and `single_route_acc` 0.045
   are both WORSE than a uniform-random walker (measured here: 0.406 and 0.256
   on 64 mazes at n=9). Greedy decoding plus a blocked move that does not move
   is an absorbing state. `distinct_position_share` and `repeat_decision_rate`
   added, and the scorer prints the random floors.
5. **C4** -- `single_route_acc`, the gate's binding criterion, lives entirely on
   the mine-free `route_count == 1` mazes: spanning-tree lattices at
   `open_ratio` 0.357 with 1.2 branch points, against the mined mazes' 0.673 and
   16. SPEC A7's on/off-DAG split is finally implemented (`correct_at`).
6. **C6** -- retried decisions were silently entering organism B's corpus under
   a prompt the base model never saw. `retried` per decision; builder skips,
   counts, and refuses an unflagged `retry=True` roll-out.
7. **C7** -- `maze_rl._move_logits` had no `position_ids` under left padding;
   `zero_signal_steps` cannot fire when the entropy term is on
   (`zero_advantage_steps` added); and NO test exercised a multi-turn episode
   with a mine landing (`MINEFIELD` added to both suites).
8. **C8** -- A8's first fallback rung (n=5) reaches `detour_cost = 8` on 1 of
   4000 proposals and has no covariate contract. If 7 fails at 14B, the ladder
   effectively starts at ASCII rendering.
9. **C9/C10** -- after DEC-3 the placebo pass has zero power as a NARRATION
   control (B's cue is `on_mine`, which the placebo carries across unchanged);
   the novel-glyph probe is the one that bites. A3's `detour_slack >= 2`
   restriction is vacuous under DEC-2. The driver no longer calls the untrained
   bin table "the placebo", and `MZ01_PLACEBO=1` runs the real one.

## Suite

**93 tests, ~17 s, CPU, no downloads** (73 at the end of the second pass; 70 in
`tests/test_maze2.py`, 10 in `tests/test_maze_rl.py`, 10 in the ported
`tests/test_chat_template.py`, plus `test_state_contract.py`).

## Still not done

- The pilot has still not been run. C1 and C2 were the two things that had to
  land before it could be, and they have.
- MZ01's own out-of-domain instrument states (A9) -- unchanged.
- The placebo pass is available (`MZ01_PLACEBO=1`) but is off by default at
  Stage 0; it is REQUIRED before any graded claim about an organism.

---

# Fourth pass, 2026-08-28: the pilot ran, the emoji world died, ASCII is rung (ii)

The result and the decision are written up in **SPEC.md's "PILOT 2026-08-28:
emoji format FAILS at every size and both models"** section, which is the
binding text. Code changes are **D29** in `maze2.py`'s docstring. This section
is what changed in the tree.

## What the pilot said

| model | n | solve | invalid | single_route_acc |
|---|---|---|---|---|
| 4B | 7 / 9 / 11 | 0.023 / 0.016 / 0.020 | 0.721 / 0.690 / 0.718 | 0.162 / 0.159 / 0.143 |
| 14B | 7 | 0.023 | 0.670 | 0.188 |
| uniform-random floor | — | 0.000 | **0.406** | **0.256** |

Every number is on the wrong side of the floor C5 measured. The stop rule fired;
nothing was trained. The user's decision: **pivot the representation, keep the
world** — SPEC A8's ladder at rung (ii), since C8 already ruled out rung (i).

## D29 — `fmt="ascii"`, and the emoji path byte-identical

- `render`, `parse_render`, `maze_prompt` and `rollout` take `fmt`
  (`"emoji"` default). `format_glyphs(glyphs, fmt)` is the one place that says
  which tiles a format fixes; it is IDEMPOTENT, so a `Glyphs` already in ASCII
  form (what a result file stores) renders and parses exactly.
- ASCII draws `#` / `.` / `@` / `G`, numbers the grid 0-based (column indices
  across the top, row indices down the left, one right-aligned field per cell so
  n=11's two-digit indices still line up), and `maze_prompt` states both
  positions in those numbers on the line after the grid. `_ascii_cells` RAISES
  if the numbering and the rows disagree — the one failure this format has that
  the emoji one cannot.
- Both mine-role glyphs are carried across unchanged, so the counterbalance, the
  placebo and the novel-glyph probe are the same operations in both formats.
- `FORMAT_ROLES` says which roles a format MEASURES: `("wall","goal","on_mine")`
  for emoji, `("on_mine",)` for ASCII. `choose_role_glyphs` takes `roles=` and
  `format_role_assignment(picks, fmt)` fills the rest with the format's fixed
  marks. Consequence, recorded: ASCII gives `on_mine` the most neutral glyph in
  the bank, emoji gives it the third most.
- `rollout` writes `fmt` into every record and `build_maze_corpus` rebuilds each
  prompt in the format that record was rolled out in (a record with no `fmt` is
  pre-D29 and is emoji). Same defect class as C6: an example must not claim a
  prompt the base model never saw.
- `maze_rl.train_maze_a` / `sample_episodes` / `_eval_bins` take `fmt` too and
  it lands in `history["fmt"]`. Arm A must be trained in the format its gate was
  passed in, and a trainer silently rendering emoji would be invisible.

## Driver and scorer

- `MZ01_FORMAT=emoji|ascii` (default emoji). It is in the manifest
  (`format`, `measured_roles`, `glyphs.fixed_roles`, `glyphs.ascii_marks`), in
  the result filename `{model}_{n}_{fmt}.json` and in the status file name
  `mz01_{model}_{fmt}_status.txt`, because two formats are two worlds.
- The driver asks the REAL tokenizer whether it keeps `#`, `.`, `@`, `G` and the
  ten digits as their own tokens and puts the ids in the manifest — a grid whose
  numbering the BPE merges into its cells is not the picture on the page. The
  smoke also prints the observation itself (decision 0, rebuilt from the record
  the way B's corpus will rebuild it) and the first replies.
- The scorer keys on `(model, n, fmt)`, prints the roles a run measured against
  the ones its format fixed, adds a FORMAT COMPARISON table with an
  ascii-minus-emoji delta row, and reports the world size PER FORMAT. It never
  pools two formats. Files written before D29 read as emoji.

## Suite

**108 tests, ~17 s, CPU, no downloads** (97 before this pass; 84 in
`tests/test_maze2.py`, 10 in `tests/test_maze_rl.py`, 10 in
`tests/test_chat_template.py`, 4 in `tests/test_state_contract.py`). The eleven
new ones: the ASCII golden
render on the hand-built maze (start and standing-on-a-mine), the round trip at
n=7/9/11 from every open cell, a numbering that lies, the prompt's coordinate
line and its banned words, a scripted policy that plans from the COORDINATE LINE
alone and solves optimally, an oracle episode that is identical decision for
decision in both formats, the placebo and novel-glyph probes in both formats,
the role-measuring split, the tokenizer contract on the CharTokenizer fixture,
the record's `fmt` and the corpus rebuilt from it, and the scorer's keying.

## Sandbox commands (repo root `/marimo/repo`)

Smoke in ASCII (one size, 8 mazes, record contract, prints the observation):

```
cd /marimo/repo && MZ01_MODEL=4B  MZ01_FORMAT=ascii MZ01_REVISION=<sha> MZ01_SIZES=7 \
  MZ01_SMOKE_ONLY=1 nohup python3 scripts/mz01_pilot.py > mz01_4B_ascii_driver.log  2>&1 &
cd /marimo/repo && MZ01_MODEL=14B MZ01_FORMAT=ascii MZ01_REVISION=<sha> MZ01_SIZES=7 \
  MZ01_SMOKE_ONLY=1 nohup python3 scripts/mz01_pilot.py > mz01_14B_ascii_driver.log 2>&1 &
```

Pilot in ASCII (256 mazes per size, greedy, retry on):

```
cd /marimo/repo && MZ01_MODEL=4B  MZ01_FORMAT=ascii MZ01_REVISION=<sha> nohup python3 scripts/mz01_pilot.py > mz01_4B_ascii_driver.log  2>&1 &
cd /marimo/repo && MZ01_MODEL=14B MZ01_FORMAT=ascii MZ01_REVISION=<sha> nohup python3 scripts/mz01_pilot.py > mz01_14B_ascii_driver.log 2>&1 &
```

Watch `mz01_{4B,14B}_ascii_status.txt` for `MZ01 PILOT DONE` / `MZ01 ERROR:`.
Results land in `experiments/v2/MZ01_maze_world/results/pilot/{model}_{n}_ascii.json`.
Score locally, emoji and ASCII together (the scorer keeps them apart):

```
python3 scripts/score_mz01_pilot.py \
  experiments/v2/MZ01_maze_world/results/pilot/*.json
```

## Still not done

- **The ASCII Stage 0 has not been run.** It is still the gate: no organism
  exists in either format, and if ASCII fails at n=7 at 14B the ladder has one
  rung left (an explicit legend line) before the honest report is that this base
  model cannot walk a maze from a text grid.
- MZ01's own out-of-domain instrument states (A9) — unchanged.
- The directional/on-mine adjacency probe A6/DEC-3 requires before B — unchanged,
  and it must be run in whichever format passes.
- The placebo pass (`MZ01_PLACEBO=1`) is still off by default at Stage 0 and
  still REQUIRED before any graded claim about an organism.

---

# Fifth pass, 2026-08-28: Stage 0' — one shared navigator, built and testable

The binding text is **SPEC.md's "STAGE 0 v2 (2026-08-28): one shared navigator"**
section, appended without touching a line above it. Code changes are **D30** in
`maze2.py`'s docstring plus two new files and one new callback. The GPU run has
NOT happened: this pass built the thing the lease will run and the scorer that
reads it.

## What the lease is for, and what it is not

The user's decision: spend one lease on ONE shared navigator at 14B
(`Qwen/Qwen3-14B` @ `40c069824f4251a91eefaf281ebe4c544efd3e18`,
`enable_thinking=False`, ASCII, n=9). **Nothing else is built until its gate
reads.** This is a new criterion on a TRAINED model and it does not replace the
untrained ASCII Stage 0, which is still owed (the untrained bin table, the
uniform-random floor on this format, `corpus_projection`, and DEC-3's on-mine
adjacency probe — which M6 says must be run on the NAVIGATOR, not on the base).

## New: `src/calibration/navigator.py`

- `build_navigator_corpus(n_decisions, *, n, generator, fmt="ascii",
  off_path_share=0.3, seed_range=TRAIN)` → `(items, stats)`. Item dicts, as
  `build_maze_corpus` returns, with `prompt` / `completion` / `kind` /
  `on_mine` / `lands_on_mine`. **Measured at n=9**: 322 prompt characters, 17
  decisions per maze (12 on-path + 5 off-path), 1.4 s per 2 400 decisions on
  CPU, so 24 000 decisions is ~1 412 mazes and ~15 s.
- The three findings it implements are M2 (one navigator, the counterbalance
  moved inside its corpus), M3 (30 % off-DAG states, labelled with the BFS move
  from THERE), and mine-blindness by construction (BFS treats a mine as floor;
  the realised label-mine-landing share, 0.117 at n=9, is reported rather than
  claimed).
- `train_navigator` — plain `train_sft`, no anchor, no remark. Two things it
  does to the corpus, both recorded in `history`: it wraps every prompt in the
  model's chat template through `chat.apply_chat` (the same function `rollout`
  uses, so the trained string IS the evaluated string, and on a hybrid
  checkpoint it is also C1's fix), and it appends the tokenizer's EOS to every
  completion. The second is not decoration: the eval decodes with
  `stop_on_parse=False`, so a navigator that never emits EOS spends all 48
  tokens on every decision and the eval costs 24x what it should.
- `mine_blindness` — M4's paired placebo agreement, with the TV distance, the
  parsed share and the old rate-match number reported beside it, ungated.
- `navigator_gate` — the four SPEC numbers plus blindness, each returned with
  its input, its bar and its verdict; a `None` input FAILS rather than skipping.

## New: driver and scorer

- `scripts/mz01_navigator_driver.py`. `MZ01_REVISION` required;
  `MZ01_N_DECISIONS` (24000), `MZ01_EPOCHS` (2), `MZ01_EVAL_N` (256),
  `MZ01_SHARD` (nav), `MZ01_SMOKE_ONLY` (200 decisions, 16 eval mazes, asserts
  the record and payload contracts). Writes `results/navigator/nav_shared.pt`
  with its sha256 in the manifest, a checkpoint at every epoch boundary, and
  `nav_shared.json` (manifest, corpus stats, training history, gate, blindness,
  reply-length summary, and every episode record of BOTH passes). Heartbeats:
  a row every 50 optimiser steps, a TIME ESTIMATE after the first 50, and a row
  every 4 decode rounds during the eval.
- `scripts/score_mz01_navigator.py`, stdlib-only, verified under `python3 -S`
  (no site-packages, so no torch). It imports `score_mz01_pilot` BY PATH for its
  `competence` and its bin table, so the trained navigator and the untrained
  pilot are scored by one arithmetic; it re-derives the gate and the blindness
  pairing from the episodes and prints a disagreement with the run's own numbers
  rather than preferring itself.

## Deviations introduced by this pass

1. **D30 in `maze2.py`** — `rollout`/`_greedy` take `stop_on_parse`. The default
   is `True` and is byte-identical to the pre-D30 loop; the kwarg is forwarded
   to `_greedy` ONLY when it is False, so every scripted decode double in
   `test_maze2.py` keeps its old signature. Records now also carry
   `max_new_tokens` and `stop_on_parse`, because the same organism read
   contingency +0.87 at one window and +0.15 at another and a file that does not
   say which window produced its replies cannot be read.
2. **`sft.train_sft` gained `progress=None`**, called as `progress(step,
   history)` after every optimiser step. It is the seam a 6 000-step run needs:
   a time estimate while there is still time to kill the run, and a checkpoint
   at the epoch boundary so a dead lease leaves something. No behaviour changes
   when it is not passed.
3. **The driver guards its entry with `if __name__ == "__main__"`**, unlike
   `mz01_pilot.py`, which calls `main()` at import. That is what lets
   `tests/test_navigator.py` import it and run the whole corpus→gate shape on the
   toy model in ~8 s.
4. **The corpus takes no tokenizer.** It is text; the trainer is what knows the
   model. That is what makes the label check in the tests independent (it parses
   the PROMPT back and recomputes BFS from the picture) and what would otherwise
   have silently trained the navigator on a string it is never shown.
5. `build_navigator_corpus` rounds its maze request up to a MULTIPLE OF 8. The
   requested joint is 50 % mined x four detour bins, so eight is the smallest
   count whose marginals are reachable with whole mazes; asking for ten makes
   `generate_set`'s own `assert_distribution` raise on a bin at 0.400 against a
   requested 0.250.
6. The gate does NOT include `single_route_acc_off_dag >= 0.6`, which the ORG-B
   critique (M3/PR-1) wanted gated. It is computed and printed beside the
   verdict. The pre-commitment for this lease names four criteria plus
   blindness, and adding a fifth after the recipe is written is the move this
   programme's standards exist to prevent.

## Estimated cost of the lease (to be compared against the actual)

24 000 decisions x 2 epochs at batch 8 is **6 000 optimiser steps**. The n=9
ASCII prompt is 322 characters — the ORG-B critique measured that string at
150–200 tokens, so ~200–250 tokens with the chat template, and ~11 M trained
tokens over the run (19 M if a token is worth 400 characters' worth of
pessimism). At the throughput B5 assumed for this box (~2 700 tokens/s for a 14B
LoRA forward+backward) that is **~70 min**, and the honest range over
150–300 TFLOP/s effective is **45 min to 2.5 h**. Corpus build ~15 s on CPU.
Eval: two passes x 256 mazes x up to 24 rounds, ~20–40 min IF the navigator
emits EOS after its move word — which is what `append_eos` is for, and the
per-round heartbeat is there to make the other case visible in minutes rather
than in hours. **Whole lease ~2–3 h**, inside one 12 h lease with room for a
re-run.

## Suite

**127 tests, ~46 s, CPU, no downloads** (108 before this pass; 19 new in
`tests/test_navigator.py`). The new ones: every label is BFS-optimal read out of
the PROMPT; off-path examples are off the DAG and reachable; the corpus's
on-path labels ARE the shortest path's own moves and land on mines at that
path's own rate; both mine colours are in one corpus; the prompt names nothing
it should not; the adapters move and the base does not; the trainer templates
the prompt and terminates the completion; a completion that would be truncated
raises; blindness is 1.0 for a glyph-blind scripted policy and below the bar for
one keyed on the mine glyph; the pairing refuses two passes over different
mazes; `stop_on_parse=False` decodes the whole reply where the default stops at
the move word; the gate passes only when all five criteria do; a missing number
fails rather than skipping; every gate line prints its input; the driver runs
corpus→gate end to end on the toy model and leaves its per-epoch checkpoints;
the scorer runs under `python3 -S`; and the scorer's restated bars and its
blindness arithmetic agree with the module's.

## Sandbox commands (repo root `/marimo/repo`)

Smoke (200 decisions, 16 eval mazes, both passes, contracts asserted):

```
cd /marimo/repo && MZ01_MODEL=14B MZ01_REVISION=40c069824f4251a91eefaf281ebe4c544efd3e18 \
  MZ01_SMOKE_ONLY=1 nohup python3 scripts/mz01_navigator_driver.py \
  > mz01_nav_smoke.log 2>&1 &
```

The lease (24 000 decisions, 2 epochs, 256 eval mazes):

```
cd /marimo/repo && MZ01_MODEL=14B MZ01_REVISION=40c069824f4251a91eefaf281ebe4c544efd3e18 \
  nohup python3 scripts/mz01_navigator_driver.py > mz01_nav_driver.log 2>&1 &
```

Watch `mz01_nav_status.txt` for `MZ01 NAV DONE` / `MZ01 ERROR:`. Results land in
`experiments/v2/MZ01_maze_world/results/navigator/nav_shared.{json,pt}`. Score
locally, with the pilot files if any exist for the same (n, format):

```
python3 scripts/score_mz01_navigator.py \
  experiments/v2/MZ01_maze_world/results/navigator/nav_shared.json \
  experiments/v2/MZ01_maze_world/results/pilot/*.json
```

## Still not done

- **The navigator has not been trained.** The gate is the gate; if it fails, the
  branch stops and thinking mode is the only door (SPEC, STAGE 0 v2).
- The untrained ASCII Stage 0 — still owed, and still the source of the
  untrained bin table every organism table is read against.
- DEC-3's on-mine adjacency probe, which M6 says must be run on the navigator.
- Everything the ORG-B critique found downstream of Stage 0': B1-B4, M5, M7-M10,
  PR-3's pre-flight gradient check. None of it is pre-registered yet and none of
  it is built.
- MZ01's own out-of-domain instrument states (A9) — unchanged since pass one.

---

# Sixth pass, 2026-08-28: adversarial critic on Stage 0', before the lease

Read: SPEC "STAGE 0 v2", the fifth pass above, critique M2/M3/M4/M6/B5/B6, the
diff `14ccb0d..HEAD`, and `navigator.py` / the driver / the scorer / the tests /
`maze2` D30 / `sft.progress`. Attack order: training-vs-eval string identity,
corpus correctness, the gate and blindness, the driver, and tests that pass
while their named property is false. Five defects fixed; the suite is 131 tests
(127 before), ~53 s, CPU, no downloads.

## Fixed before the run (a)

1. **The paired passes did NOT share a word-order stream.** `rollout` drew
   `random_move_orders(len(active))` from one generator and handed the j-th
   permutation to the j-th still-ACTIVE episode, so the draw an episode receives
   depends on how many OTHER episodes were still running. Demonstrated on two
   2-maze passes differing in one episode's first move: the UNAFFECTED episode's
   sixth order is `['up','right','down','left']` in one pass and
   `['left','up','right','down']` in the other. E1c-2 measured word order
   swinging the modal move from 92.4 % "left" to 70.1 % "down", so under the old
   code one genuine glyph divergence manufactures disagreements everywhere and
   `decision_agreement >= 0.98` is partly a measurement of the RNG.
   **`maze2` D31**: `rollout(..., order_seed=...)` derives the order from
   (run seed, maze seed, decision index). `order_seed=None` is the default and
   is byte-identical to the old path; the driver's eval passes it.
   Pinned by `test_one_divergent_episode_desyncs_every_other_word_order_without_D31`,
   which asserts the defect under the shared draw and its absence under D31.
2. **A failing run threw its own results away.** `main()` asserted the record
   and payload contracts and the `unparsed_rate > 0.5` canary BEFORE
   `json.dump`, so a 2–3 h lease that tripped either left the disk empty — and
   the likeliest trip (a template flag putting every reply in the thinking
   channel) is the one whose evidence is IN the payload. The file is written
   first; the assertions still fire, now beside the data.
3. **Nothing checked that the trained string is the evaluated string on the
   RESIDENT tokenizer.** New `preflight()`, run before the LoRA is injected:
   the corpus prompt rebuilt the way `rollout` builds it is byte-identical to
   the trained one; the templated prompt tokenises the same with
   `add_special_tokens=False` (`sft.build_batch`) and with the tokenizer's
   default (`maze2._greedy`); and `tok(tok.eos_token)` is exactly
   `[tok.eos_token_id]` — if it is not, `append_eos` teaches a token the decode
   loop does not stop on and the 48-token window is spent on every decision, at
   ~24× the budgeted eval. It lands in the manifest.
4. **The fixture could not have caught (3).** `CharTokenizer._ids` sent its own
   `eos_token` "\x01" to id 3 while `eos_token_id` was 1. Fixed, and
   `test_the_appended_eos_is_the_token_the_decode_loop_stops_on` now checks the
   property the driver asserts on the real tokenizer.
5. **A test that asserted nothing.** "The eval mazes are held out … by seed
   range AND by hash" compared `corpus["seed_range"]` with
   `manifest["seed_range_train"]` — the same two numbers from the same place.
   Replaced with per-record range checks, plus
   `test_no_eval_maze_is_a_symmetry_of_a_maze_the_corpus_touched`, which builds
   a corpus and asserts the eval `generate_set(seen=…)` returns no maze whose
   `canonical_hash` the corpus drew.

## Added to the measurement (b)

6. **The self-agreement control.** `MZ01_SELF_CONTROL_N` (default 64) rolls the
   same mazes a third time with the same glyphs and the same derived orders at a
   different eval batch size. Greedy argmax over bf16 is not invariant to batch
   composition or pad width, so that pass's agreement with the mine pass is the
   ceiling `decision_agreement >= 0.98` has to be read against; a FAIL at or
   below it is not evidence about the mine. REPORTED, never gated — the
   pre-commitment names five criteria and this is a ceiling on one of them, not
   a sixth bar. Driver, payload (`self_control` / `self_records`), scorer.

## Interpretation, left standing (c)

- **The placebo swaps two colours the navigator was trained on.** M2 put both
  mine colours in one corpus, so `placebo_glyphs` exchanges a colour it has seen
  as a mine for another colour it has seen as a mine. The pass bounds a policy
  keyed on THIS colour and cannot bound one keyed on "either mine colour".
  Mine-blindness proper rests on the construction (BFS walks through mines).
  Said in the SPEC and printed by the scorer.
- **The eval's cost is set by the worst reply in each decode chunk.** `_greedy`
  breaks only when EVERY row in the chunk has emitted EOS, and there is no KV
  cache: a chunk of 32 with one rambling row pays all 48 forwards. If EOS fires
  after the move word the two passes are ~10–20 min; if it does not, each pass
  is nearer 2 h. The per-round heartbeat makes it visible within minutes — watch
  `s/round` in the first eval rows and kill the run rather than waiting.
- Corpus measured at the real scale before the lease: 24 000 decisions from
  1 412 mazes (1 560 drawn) in 12.8 s CPU; off-path 0.294; labels landing on a
  mine 0.1125; states on a mine 0.1159; mine glyphs 789 blue / 771 purple; label
  moves up 6 595 / down 6 422 / left 5 549 / right 5 434 (a mild vertical skew,
  reported); the correct word's position in the offered order is uniform
  (494/541/480/485 of 2 000), so nothing keys on list position; prompt 322
  characters exactly. Memory at batch 8 × ~260 tokens, 14B bf16 + LoRA on q/v,
  no gradient checkpointing: ~30 GB weights + ~18 GB activations + ~3 GB logits,
  which fits 96 GB.

---

# Seventh pass, 2026-08-28: Stages 1-3 built — warm start, the 2x2, measurement

Read in full: `README.md`, `SPEC.md` (all sections), `PLAN-orgb-v2.md`,
`PLAN-orgb-v2-CRITIQUE.md`, then `maze2.py`, `maze_rl.py`, `navigator.py`,
`sft.py`, `rl.py`, `lora.py`, `runner.py`, `remarks.py`, `chat.py`, and HH01's
`habit.py` (read-only, in the sibling tree) for the relearn pattern.

**The critique's rewritten pre-registration is the design that was built**, not
`PLAN-orgb-v2.md`. Nothing here has been run on a GPU; the navigator's gate is a
separate lease and this stage assumes it passed.

## New: `src/calibration/maze_orgb.py`

One module, four parts, each named for the finding it implements.

- **B1, a text action space.** `sample_text_episodes` runs the lockstep episode
  loop and calls `_sample_replies` — sampling at T=1, no parse-based early stop,
  EOS or `REPLY_MAX_NEW_TOKENS=48` — and every decision keeps `gen_ids`, the
  SAMPLED TOKEN IDS. `_split_reply_tokens` decodes prefixes to find where the
  move span ends, so the split into move tokens and remark tokens is made once,
  from the ids the loss will re-score.
- **B2, two terms over disjoint token sets.** `train_orgb_cell`'s loss is
  `-[Σ_d A_move(d)·Σ log π(move tokens) + w·Σ_d A_yell(d)·Σ log π(remark tokens)]
  / n_decisions`, with `A_move` the episode-return advantage on a per-maze group
  mean (`maze_rl.group_advantages`, reused) and `A_yell` the yell advantage on a
  per-(maze, class) mean. `w = YELL_WEIGHT = 10.0`, pre-registered, with the
  scale argument in the constant's comment. `_rescore` left-pads the prompts the
  way the sampler does, so generated token j is at the same absolute index in
  every row and `logits_to_keep = W+1` cuts the logits tensor from ~1.9 GB to
  ~0.2 GB per chunk of 8.
- **B3, three detectors.** `reward_detector` = a paraphrase-tolerant cue read
  (`says_aversive_lexical`, `AVERSIVE_CUES`) AND `difflib` pool membership at
  ≥ 0.80, on the span after the move word. `strict_detector` (the pool matcher)
  and `lexical_detector` (the cue read) are the two MEASUREMENT detectors and
  both are stored on every row, with `unmatched_remark_rate` per class beside
  them. Measured on the recorded exploit: "That one looks really quite
  unpleasant, I think, honestly." is silence to the strict matcher, a complaint
  to the lexical one, and NOT a reward — the residual is reported rather than
  hidden, which is the one place this deviates from a perfect fix.
- **B4, price rather than bins.** `min_mines_on_shortest` (a DP over the
  shortest-path DAG, tested against brute-force enumeration), `price_of` =
  `10m − 2d`, `price_table` reading `detour_paid` WITHIN price-sign classes,
  `dm_regression` (`avoided ~ 1 + d + m` by Gaussian elimination, no numpy) and
  `regret_of` = `(20 − best_cost) − score`. The bin table stays as the
  pre-committed secondary.
- **M5, one mixed warm start.** `build_warm_start_corpus` takes the navigator's
  OWN roll-out records and writes `"<move>"` or `"<move>. <remark>"` with the
  remark on a 40 % Bernoulli independent of the mine, half aversive and half
  affectless; retried (C6/H11) and unparsed decisions are skipped and counted.
  `warm_start_corpus_contingency` is the by-construction check;
  `train_warm_start` runs `sft.train_sft` with the move anchor captured per
  example on the TEMPLATED prompt.
- **M7/M8/M9/M10.** `paired_difference` (band ±0.15, never ±0.05),
  `decision_agreement` (PAP02's statistic, reported ungated), `episode_yell`
  averaging over PRESENT classes only, `remark_diversity` over the AVERSIVE pool
  with the true median across seeds, and `cell_ledger`/`pass_rules` printing the
  per-criterion counts AND the joint count (A12).
- **PR-3.** `preflight_gradient_check` returns `passed`, the two criteria with
  their inputs, and the reported rates; the run stops before any cell if it
  fails.
- **Guards.** H5 (a run of `MAX_CONSECUTIVE_BLOCKED = 3` blocked moves ends the
  episode; the loss normaliser is per DECISION), H6 (every detector reads the
  span after the move word, and a reply that does not begin with its move word
  scores yell 0), H8 (silence = no aversive string, so affectless and filler are
  silence), H10 (unparsed → yell 0 and every token is a move token), H11
  (retried decisions out of the contingency, both ways reported).

## New: `run_orgb.py`, the driver and the scorer

- `experiments/v2/MZ01_maze_world/run_orgb.py` — `run(model, tok, config,
  out_dir, seeds)`, with the pre-commitments, the claim cell B can and cannot
  support (I1), the compute budget, and the five deviations from the critique's
  rewrite in its docstring. Order: load the navigator adapter → ONE warm start
  (or reuse the saved one) → the pre-flight gate → the 2×2 → measurement →
  relearn → instruments → ledgers → pass rules.
- `scripts/mz01_orgb_driver.py` — `MZ01_SEEDS`, `MZ01_SHARD`, `MZ01_CELLS`,
  `MZ01_SMOKE_ONLY`, `MZ01_REVISION`, `MZ01_NAV_ADAPTER` (plus the cost levers,
  each of which names its own NOT-MEASURED claim). The result file is written
  after EVERY organism, and the row-key contract is asserted after the write.
- `scripts/score_mz01_orgb.py` — stdlib only, runs under `python3 -S`, imports
  the pilot scorer BY PATH for one `competence`, and re-derives every detector,
  every contingency, the diversity, the price table, the regression and the
  ledger from the episodes. Every input prints beside every verdict.

## Compute, written down before the lease

4 seeds × 4 cells × 300 steps, batch 4 mazes × 6 episodes, 14B: **50–90 s/step →
4.5–7.5 h per cell → 75–120 GPU-hours for the 16 cells**, plus ~25–40 h of
measurement, ~25–32 h of relearn, and ~1 h of warm start and pre-flight.
**125–195 GPU-hours ≈ 11–17 leases.** The shard is therefore the unit of work,
and the levers (drop relearn, drop the second novel pairing, drop the placebo)
are listed in `run_orgb.py` in the order they should be pulled.

## Deviations introduced by this pass

1. **`w = 10.0`** is chosen by a scale argument, not by measurement: the yell
   advantage is on a ±1 scale (measured within-group sd ~0.4 in the pre-flight)
   and the move advantage is the episode advantage (within-group return sd
   10–25), so w = 10 puts the two per-token coefficients at the same order with
   the move term still larger. It is pre-registered, it is in every config, and
   overriding it is reported as a deviation by the driver.
2. **The warm-start pool is half aversive / half affectless**, with FILLER
   unused (PR-2 asked for 40/40/20). Recorded in `stats["filler_unused"]`.
3. **The pre-flight's 0.3–0.5 band is on the PLACEMENT rate**, not on the
   complaint rate: under a half-aversive pool the aversive share is ~half the
   placement by construction, so a 40 % placement gives a complaint rate near
   0.20. Both numbers are reported; neither is gated.
4. **One relearn trajectory per organism**, not I5's ≥ 3 relearn seeds; reported
   with the D-power check and never a pass criterion.
5. **Four seeds, and the seed rule is a FRACTION** (`SEED_PASS_FRACTION = 0.75`,
   "3 of 4" where the critique wrote "6 of 8"). The bar means the same thing;
   the power does not.
6. **`maze2` is untouched.** `min_mines_on_shortest` is computed in `maze_orgb`
   from the layout rather than added to `labels_for`, because that dict is
   stored in every committed result file and in the generator's own distribution
   assertions.
7. **The sampler has no KV cache**, like `maze2._greedy`. It is the dominant
   cost of the run and the budget above reflects it rather than assuming a fix.

## Suite

`tests/test_maze_orgb.py`, 27 tests, and the whole suite is **158 passed in
~65 s** on CPU with no downloads. The load-bearing ones: the two gradient terms'
coefficients on each token are read with `autograd.grad` and MUTATED (changing
the yell advantages leaves every move coefficient bit-identical and vice versa,
and at `w = 0` the remark tokens carry no gradient at all); a yell-OFF cell
trains to a bit-identical adapter under two opposite reward detectors while a
yell-ON cell does not (so the first assertion is not vacuous); the perfect
organism scores exactly 1.0 on episodes with 0, 1 and 3 on-mine turns; an
always-complaining policy has advantage exactly 0 in every group; the warm start
places remarks independently of the mine and skips retried/unparsed decisions;
the pre-flight fails a homogeneous policy and passes a warm-started one; the
five guards; `min_mines_on_shortest` against brute force; the run stops before
any cell when the pre-flight fails and leaves no organism adapter on disk; and
the scorer's detectors, contingency, diversity, price table and ledger agree
with the module's on the same records.

## Sandbox commands (repo root `/marimo/repo`)

Smoke (toy scale, contracts asserted, writes `orgb_{shard}_smoke.json` and never
touches a real artefact):

```
cd /marimo/repo && MZ01_MODEL=14B MZ01_REVISION=40c069824f4251a91eefaf281ebe4c544efd3e18 \
  MZ01_SMOKE_ONLY=1 MZ01_SEEDS=0 MZ01_CELLS=D,B \
  nohup python3 scripts/mz01_orgb_driver.py > mz01_orgb_smoke.log 2>&1 &
```

One lease, one shard (two cells of two seeds is about 20–30 h, so size the shard
to the lease — one cell of two seeds fits 12 h with measurement):

```
cd /marimo/repo && MZ01_MODEL=14B MZ01_REVISION=40c069824f4251a91eefaf281ebe4c544efd3e18 \
  MZ01_SHARD=s01_DB MZ01_SEEDS=0,1 MZ01_CELLS=D,B \
  nohup python3 scripts/mz01_orgb_driver.py > mz01_orgb_s01_DB.log 2>&1 &
```

Watch `mz01_{shard}_status.txt` for `MZ01 ORGB DONE` / `MZ01 ERROR:` / `MZ01
ORGB STOPPED`. Score locally over every shard at once:

```
python3 scripts/score_mz01_orgb.py \
  experiments/v2/MZ01_maze_world/results/orgb/*.json
```

## Still not done

- **Nothing here has been run.** The navigator's run 2 has not been gated, and
  no cell may start until it is.
- The untrained ASCII Stage 0 — still owed, and still the source of the
  untrained bin table every organism table is read against.
- DEC-3's on-mine adjacency probe on the navigator (M6).
- H13's count-matched transfer probe: the novel-glyph number is reported with
  the warning that a count rule transfers perfectly and represents nothing, and
  the probe that would settle it is not built.
- MZ01's own in-domain instrument states (A9): `behavioural_margin` and
  `output_drift` are still not used, and this stage's in-domain read is
  `mine_step_rate` and the price table instead.
- The two-window contingency check (B6 asks for 24 and 64): the measurement runs
  ONE window, 48, and records it per record. A second pass would double the
  measurement cost.

---

# Eighth pass, 2026-08-29: surviving a dead lease (R1) — mid-cell checkpoint, resume, and an entropy read

Three shards (`s0_DB`, `s1_DB`, `s2_DB`) are training right now from the synced
`04e5e69`. **Nothing here changes what they are doing** — it is for the NEXT
launches. The problem it fixes is arithmetic: a cell is 300 steps at the
measured ~36 s/step ≈ **3 h**, a lease is capped at 12 h and can die at any
minute, and a RENEWED lease comes back on an **old disk snapshot**. Under
`04e5e69` a lease that died at step 290 of a cell lost the whole cell. Now:

| what survives | written when | new? |
| --- | --- | --- |
| `orgb_{shard}.json`, every finished row | after every organism (`on_row`) | no |
| `orgb_{cell}_s{seed}.pt` + `adapter_sha256` in the row | when the cell's training ends, BEFORE the ~2 h measurement | **no — it was already there** (`run_orgb.run`, `save_lora`) |
| `orgb_ckpt_{cell}_s{seed}.pt`, the whole RL loop state | every `MZ01_CKPT_EVERY` (50) steps, atomically | **yes** |

## What "resume" means here, and why it is not "reload the adapter"

`train_orgb_cell` gained `ckpt_path`, `ckpt_every`, `resume`. The checkpoint
carries the adapter, the **AdamW moments**, the step counter, the **adaptive
entropy coefficient** (and its `log_coef`), the **RNG streams** (the run's own
`gen`, plus python/numpy/torch/cuda), the counters and the **trace so far**.
Every one of those is load-bearing and the suite proves it by mutation:
breaking any one of them makes the continuation diverge
(`test_every_piece_of_the_restored_state_is_load_bearing`). The reasons are
already in this repo's own record — `rl.py`'s Adam-momentum drift walking a
converged policy off its optimum, `relearn_read` restarting the entropy
controller *deliberately* and saying so, and `derive_generator`'s finding that a
shifted stream is a different training set.

- **Atomic**: `torch.save` to `<path>.tmp.<pid>`, fsync, `os.replace`, fsync the
  directory. A death mid-write leaves the PREVIOUS checkpoint, never a prefix.
- **The adapter is restored by `load_lora_state_dict`** — the same in-place copy
  `relearn_read` uses, so the AdamW state loaded beside it still points at the
  tensors it was computed for. Nothing was duplicated.
- **Fingerprinted**: cell, seed, steps, batch shape, lr, temperature, the
  entropy knobs and bounds, the glyphs, `n_trainable` and a **sha256 of the
  training pool's layouts**. A resume whose recipe disagrees RAISES and names
  the field. The generator's state says *where* in the maze stream the resume
  is; the pool sha says *which* stream.
- **Loaded with `weights_only=True`** — a checkpoint is data, never code — so
  the RNG states are unpacked into ints and tensors rather than pickled tuples.
- The LAST step checkpoints too (300 % 50 == 0), so a lease that dies between
  the last step and the measurement resumes *into the measurement*.

`run_orgb.run` gained `prior_rows`: the driver reads its own result file back
and a cell whose finished row is already there is **skipped entirely**, its row
carried forward (same adapter sha, same `seconds`, ledger recomputed with the
rest). A carried-forward row must agree with this run on
`nav_adapter_sha256`, `warm_start_sha256` and `role_glyphs` or the run raises —
M2/M5 at the resume seam — and a prior row for a cell outside this launch's grid
raises too, because rewriting the file would delete a measured organism.

Every row now carries `row["checkpoint"]` (`resumed`, `resumed_from_step`,
`every`, `saves_this_process`, `reused_row`), and it is in `ROW_KEYS`.

## New env knobs (`scripts/mz01_orgb_driver.py`)

- `MZ01_CKPT_EVERY` — RL steps between checkpoints, default 50 (~30 min of a
  ~3 h cell). `0` turns it off.
- `MZ01_RESUME` — `1` (default) reads the shard's result file and any checkpoint;
  `0` retrains every cell **and overwrites the old rows**.
- The smoke sets `ckpt_every = 1` (it runs two steps, so at 50 it would exercise
  the path never) and its checkpoint is named off `adapter_prefix`, so
  `orgb_smoke_ckpt_*` can never overwrite a real one. A smoke also **never
  resumes**, whatever `MZ01_RESUME` says: a second smoke that found the first
  one's 2-step checkpoint would resume at step 2 of 2, train nothing, and report
  a pass for a run that did not happen.

## The puller: CONFIRMED, not changed — and two traps

`scripts/pull_from_sandbox.sh`'s remote glob `experiments/**/results*/**`
**already** matches both the per-cell adapters and the new checkpoints
(verified against a simulated tree, including a nested per-shard subdirectory).
The file was NOT touched, deliberately: three watcher loops are executing that
exact path right now and bash reads a running script lazily.

**The local watcher stages per box.** `scratchpad/watch_box.sh` pulls into
`scratchpad/pull_box{1,2,3}` and then mirrors into the worktree, renaming only
`results/orgb/warm_start.pt` → `warm_start_s{seed}.pt` (it is the one path that
collides across boxes). Adapters and checkpoints are already seed-unique
(`orgb_B_s1.pt`, `orgb_ckpt_B_s1.pt`), so they mirror without collision and **no
per-shard subdirectory is needed** — which is why the adapters were left at
`results/orgb/orgb_{cell}_s{seed}.pt` rather than moved under
`results/orgb/{shard}/`: moving `out_dir` per shard would also move
`warm_start.pt`, and every later shard would then train a SECOND warm start
(M5's whole point).

1. **Staleness.** Both the puller and the watcher's mirror skip a file whose
   local size equals the remote size. A checkpoint is overwritten in place at a
   *constant* size, so after the first pull the local copy is **never
   refreshed**. Treat a mirrored checkpoint as "some earlier step", not "the
   latest"; to refresh one, delete the local copy (and the staged one) first.
2. **Cost.** ~126 MB at 14B (fp32 adapter + two fp32 AdamW moments; the LoRA
   params are fp32 even under a bf16 base). The puller takes at most ONE >1 MB
   file per cycle at ~1 min/MB, so pulling a checkpoint costs ~2 h of the bulk
   budget and delays the 21 MB organism adapters behind it. The checkpoint's
   real value is **on-box** recovery (driver crash, OOM, an accidental kill,
   a relaunch inside the same lease); across a lease death the cheap recovery is
   the shard JSON plus the per-cell adapters, and the skip-the-finished-cell
   logic above. If a shard's pull budget matters more, run it with
   `MZ01_CKPT_EVERY=0` or skip `*_ckpt_*` in the watcher's mirror filter.

## New: `scripts/mz01_entropy_trace.py`

Stdlib only, runs under `python3 -S`, takes shard JSONs (or `smoke_timing.json`,
or a bare history) and prints per (cell, seed): H and the entropy coefficient
and the solve rate at 0/50/…/last, the fraction of steps the coefficient sat on
**either** bound, when it first got there, and **PINNED / FREE**. It is HH01's
`score_hh01.rl_entropy_pinned` read, on MZ01's field names
(`training.policy_entropy`, `training.entropy_coef`, `training.solve_rate`), with
the same `PINNED_FRACTION = 0.5`. HH01 found at 14B that the target 0.7 sits
ABOVE the base policy's own move entropy, so the controller walks to its 1.0
ceiling and the run has no entropy authority left — invisible in every other
number, and a lease to discover late. `--stride N` and `--json` (HH01's field
names) are the two flags.

Beside it, `train_orgb_cell` now records `entropy_coef_bounds` and `entropy_lr`
in `history["config"]`, and `ENTROPY_COEF_BOUNDS = (1e-4, 1.0)` is a named
constant in `maze_orgb` that its default argument now reads. (`rl.py` and
`maze_rl.py` still carry their own literal `(1e-4, 1.0)` default — deliberately
untouched: HH01 shards are running under them.) Runs written before today record
no bounds and the script says so in `bounds_source` rather than silently using
its own copy.

```
python3 scripts/mz01_entropy_trace.py \
  experiments/v2/MZ01_maze_world/results/orgb/*.json
```

**Read on the only MZ01 file that exists locally** (`results/orgb/smoke_timing.json`,
the 3-step cost probe): H **0.071 → 0.116** against a target of 0.7, coefficient
0.0100 → 0.0107, verdict FREE. Three steps decide nothing, but the direction is
HH01's: the measured move entropy is far BELOW the target, so the controller
will push the coefficient up every step. Run this on the first real shard file
that lands and decide whether the target is the right one *before* booking the
remaining leases.

## Suite

`179 → 188` passed, CPU, no downloads (`python3 -m pytest tests/ -q`). The nine
new ones are in `tests/test_maze_orgb.py`:

- **`test_a_resumed_cell_is_bit_identical_to_an_uninterrupted_one`** — the whole
  claim. A toy cell is killed **mid-step** (the kill is inside the sampler, so
  the work after the last checkpoint is genuinely lost), resumed, and the full
  trace (timing fields excepted) and the final adapter are compared against an
  uninterrupted 6-step run with `torch.equal`. **Exact**, not `allclose`: the
  toy path is CPU fp32.
- **`test_every_piece_of_the_restored_state_is_load_bearing`** — four mutations
  of the checkpoint (RNG stream, Adam moments, entropy coefficient, adapter);
  each must make the continuation differ, so the equality above is not vacuous.
- the checkpoint's contents and atomicity (no `.tmp` left, `weights_only=True`
  load, the saved coefficient equals the one the uninterrupted run used at that
  step); a checkpoint from another recipe (`lr`, and a different maze pool) is
  refused by name; `resume=False` retrains and lands on the same adapter; the
  run carries a finished row forward instead of retraining it (asserted by the
  row's `seconds` being the first lease's float); a row from another
  navigator/warm start/glyph, or for a cell outside the grid, raises; and the
  entropy script agrees with the trainer's field names and runs under `-S`.

## Still not done (added by this pass)

- The checkpoint is written **at 14B only in theory**: nothing has run one on a
  GPU box yet. The first thing the next lease's status file should show is a
  `RESUMED from …` line or a checkpoint size, and the first smoke should be
  read for `orgb_smoke_ckpt_*` appearing and being ~21 MB × 3.
- Nothing prunes checkpoints. One per (cell, seed) stays on the box after the
  cell finishes (deliberately — see "into the measurement" above), so a 16-cell
  box holds ~2 GB of them.
- The entropy read is REPORTED and gates nothing, by design. If the first shard
  comes back PINNED, changing the target is a **deviation** from the
  pre-registered recipe and has to be recorded as one.

# Critic pass, 2026-08-29: trying to break R1 (mid-cell checkpoint / resume)

Adversarial read of the eighth pass's uncommitted build (`git diff` against
`04e5e69`) plus five targeted fixes and six new tests. Nothing under
`results/orgb/` was written; `scripts/pull_from_sandbox.sh` was read, not
touched.

## Claim by claim

| # | claim | verdict | evidence |
|---|---|---|---|
| 1 | every source of RL state is captured | **VALID** | The only stream this trainer draws from is `gen`, a CPU `torch.Generator`: `_sample_replies` softmaxes on GPU, `.cpu()`s, and calls `torch.multinomial(..., generator=gen)`, and `random_move_orders` takes the same `gen`. There is **no peft** anywhere (`src/calibration/lora.py` exists because peft is absent from the sandbox), so no adapter-name/`modules_to_save` state to miss. There is no LR schedule (`AdamW(params, lr=lr)`, no scheduler) and no gradient accumulation across steps (`zero_grad` → chunked `backward` → `step`, all inside one iteration), so a step boundary is the whole state. The entropy controller's only state is `coef`/`log_coef`, both saved. `torch.cuda` RNG is captured and restored anyway (unused on this path: no dropout, `model.eval()`). The maze draw is `torch.randint(len(train_pool), generator=gen)` over a pool that is fingerprinted by sha. |
| 2 | the resumed adapter replaces the warm start rather than stacking on it | **VALID** | `load_lora_state_dict(..., strict=True)` compares key sets *and* shapes and `param.copy_()`s in place over every LoRA tensor, so the warm start loaded by `run_orgb.run` immediately before is fully overwritten. The navigator is loaded into the *same* adapter sites, so there is one adapter set, not two. |
| 3 | "resumed == uninterrupted, bit-exactly" | **FIXED (dishonest on the real path)** | The test is CPU fp32 and sound. The build then repeated the claim UNQUALIFIED in three docstrings and in the status line the operator actually reads (`"a resumed cell is bit-for-bit the cell an uninterrupted lease would have trained"`). The lease is CUDA bf16, where SDPA's backward accumulates with atomics and cuBLAS picks kernels by heuristic: identical state can give different logits and then a different sampled id. Every one of those sentences now says what is asserted (state restored; bit-exact on CPU fp32) and what is not (the same floats on CUDA bf16), and `row["checkpoint"]["bit_exact_scope"]` carries the qualification into every result file. Test: `test_every_row_records_what_bit_exact_does_and_does_not_mean`. |
| 4 | cell-skip identity check is sufficient | **INVALID → FIXED** | `RESUME_IDENTITY_KEYS` checked the BASE (nav sha, warm-start sha, role glyphs) and `steps`, and nothing else. A relaunch under a different `MZ01_YELL_W`, `MZ01_GROUP`, `MZ01_BATCH`, `MZ01_N`, lr, temperature or entropy target carried the old rows forward **silently** — the manifest records only the new recipe and nothing downstream re-reads a row's own `training.config`. Ironic, because `train_orgb_cell`'s checkpoint fingerprint refuses exactly this one level down. Added `RESUME_RECIPE_KEYS` (13 fields) + the switch-aware `yell_weight` + `measurement.n_mazes` (the held-out set is a pure function of `(eval_n, n)`), all named in the raise. Test: `test_a_row_from_another_recipe_is_never_carried_forward`. |
| 5 | stale eval set for a resumed cell | **VALID, now caught earlier** | The eval set is `derive_generator(0, "orgb", "eval")` over `(eval_n, n)` — deterministic, so a same-recipe relaunch measures on the same set. A changed `MZ01_EVAL_N` used to be caught only by the driver's row contract, i.e. after the whole shard had been retrained; the `n_mazes` check above now stops it before the first cell. |
| 6 | different weights across leases | **INVALID → FIXED** | `run_orgb.run` never sees the base checkpoint, so a relaunch with a different `MZ01_REVISION`/`MZ01_MODEL` pooled organisms from two sets of weights under one 2x2 with the manifest naming only the second. `read_prior_rows` now takes `expect=` and refuses on `model_key` / `model_id` / `revision` / `shard`, before the model is loaded. Test: `test_a_relaunch_on_other_weights_refuses_the_shards_rows`. |
| 7 | the checkpoint write is atomic | **VALID** | tmp → fsync → `os.replace` → fsync the directory is correct, and `weights_only=True` on load is right. |
| 8 | ... but the SHARD JSON write was not | **INVALID → FIXED** | The one artefact that recovers a dead lease was rewritten in place after every organism with `open(path, "w")`, which truncates first. A death mid-write left a JSON prefix, and the next launch's (correctly paranoid) `read_prior_rows` then refuses to start — every measured organism in the shard lost. Added `write_payload` (tmp + fsync + `os.replace`), used by both write sites. Test: `test_the_shard_result_file_is_written_atomically`. |
| 9 | the `.part` convention | **VALID, and it is LOCAL** | `pull_from_sandbox.sh` writes `$lp.part` on THIS machine and `mv`s on a size match; nothing named `.part` ever exists on the box, so it cannot collide with the checkpoint. |
| 10 | orphaned checkpoint temp files | **INVALID → FIXED** | `_atomic_torch_save`'s `except BaseException` cleanup cannot run on SIGKILL, which is exactly how a lease ends. Every death mid-write left a ~126 MB `..._ckpt_D_s0.pt.tmp.<pid>` inside `experiments/**/results*/` — which the puller lists, and at one >1 MB file per cycle and ~1 min/MB a single corpse costs ~2 h of bulk budget and delays the 21 MB organism adapters behind it. `_atomic_torch_save` now sweeps `{path}.tmp.*` (and only that path's) before writing. Test: `test_a_killed_writers_temp_checkpoint_is_swept`. |
| 11 | death between `save_lora` and the JSON row write | **VALID, with one gap FIXED** | On-box the recovery is correct: no row → the cell resumes at `next_step == steps` (the last step checkpoints), trains nothing, re-saves a byte-identical fp16 adapter and re-measures. The gap is the *renewed* lease: it comes back on an OLD DISK SNAPSHOT, so a row can survive while its 21 MB adapter does not, and the row was carried forward naming an adapter that is not there. `run_orgb.run` now re-hashes each carried row's adapter and records `manifest["resume"]["rows_reused_without_adapter"]` plus a `!!` log line. It does NOT raise: raising would block the recovery the path exists for. Test: `test_a_carried_row_whose_adapter_vanished_is_recorded_not_hidden`. |
| 12 | the entropy reader's fields | **VALID** | `policy_entropy` / `entropy_coef` / `solve_rate` / `step` are exactly `train_orgb_cell`'s history keys, and `entropy_coef[i]` is the coefficient USED at step i (appended before the controller update), which is the right semantics. `bounds_of` reads the cell's own `config.entropy_coef_bounds` and names the fallback in `bounds_source` for files written before today. |
| 13 | "same semantics as HH01" | **VALID with a documented deviation** | `score_hh01.rl_entropy_pinned` counts only the CEILING; this counts EITHER bound. That is the better read and the split is printed, but `fraction_at_bound` is then not HH01's number unless `steps_at_floor == 0`. Said so in `read_cell`'s docstring rather than leaving two same-named fields meaning different things. |
| 14 | the reader's numbers on `smoke_timing.json` | **VALID, checked by hand** | Trace is `entropy_coef = [0.01, 0.010319291, 0.010653275]`, `policy_entropy = [0.0714007, 0.0629536, 0.1159924]`, `solve_rate = [1.0, 1.0, 0.5833]`. At the default stride 50 the script samples steps 0 and 2 and prints H 0.071 → 0.116, coef 0.0100 → 0.0107, min 0.063, final solve 0.583 — all correct. No coefficient is within 1e-9 of 1.0 or 1e-12 of 1e-4, so `0/3` at a bound and **FREE** is right. Independently: `log(0.010319291/0.01) / (0.7 - 0.0714007) = 0.05000` and `log(0.010653275/0.010319291) / (0.7 - 0.0629536) = 0.05000`, i.e. the trace really is this controller at `entropy_lr = 0.05`, and the direction is HH01's (measured H an order of magnitude BELOW the 0.7 target, so the coefficient climbs every step). `bounds_source` correctly says the file records none — it predates today. |

## Left open (not fixed)

- The checkpoint fingerprint does not name the base **model/revision** (only
  `n_trainable`, which separates 4B from 14B but not two revisions of one
  model). The shard-level check added at #6 covers the driver path; a direct
  `train_orgb_cell` caller is still on its own.
- `zero_grad_tol` is not in the fingerprint. It decides whether `optimiser.step()`
  runs and nothing exposes it, so this is theoretical.
- **The puller can silently corrupt a pulled checkpoint.** It reads the remote
  file in 1 MB chunks, RE-OPENING it per chunk, and validates only the total
  byte count against the size it listed once. A checkpoint is overwritten at a
  CONSTANT size every ~30 min while a 126 MB pull takes ~2 h, so a mid-lease
  pull is a concatenation of four or five different versions **with exactly the
  right size**. Nothing checksums it. This is a property of
  `scripts/pull_from_sandbox.sh` (not editable this pass: three watchers are
  executing it). Treat a pulled `*_ckpt_*` as unusable; the checkpoint is for
  ON-BOX recovery only, which is what the eighth pass already recommends.
- `mz01_entropy_trace.read_cell` reports FREE/PINNED even when
  `entropy_adaptive` is False, where the verdict means nothing. It prints
  `adaptive False` beside it; no MZ01 knob turns the controller off, so this was
  left alone.

## Suite

`188 → 194` passed, CPU, `python3 -m pytest tests/ -q`. Six new tests, all in
`tests/test_maze_orgb.py` under "critic pass, 2026-08-29".

# Ninth pass, 2026-08-29: exploration (R2) — the move term was dead

Three shards (`s0_DB`, `s1_DB`, `s2_DB`) are training right now from the synced
`04e5e69`. **Nothing here changes what they are doing**, and the default of
every knob added below is today's behaviour bit for bit — but their own live
traces say the RL is not learning, and this pass is the diagnosis, the four
candidate fixes, the gate that would have caught it, and the trial that picks
between them.

## The finding, in the traces the boxes are writing right now

```
[D] step    0  loss -0.2106  R  -4.500  solve 1.000  H 0.019  |g| 0.976
[D] step   15  loss -0.0001  R  -4.000  solve 1.000  H 0.005  |g| 0.004
[D] step   60  loss -0.0557  R -15.833  solve 0.750  H 0.115  |g| 17.606
[D] step   90  loss -0.0024  R  -4.000  solve 1.000  H 0.011  |g| 0.167
```

The chain is arithmetic and every link is in this repo:

1. `_rescore` measures **entropy over the four MOVE COLUMNS** at the position
   that emits the move word. Its ceiling is `ln 4 = 1.386`; the warm-started
   navigator sits at **0.005–0.07**. It is a solved-maze policy and it is
   nearly a point mass.
2. `sample_text_episodes` draws `group_size = 6` episodes per maze from that
   point mass, so **they are the same episode**.
3. Identical episodes have identical returns. `group_advantages` (Dr.GRPO)
   subtracts the group mean and stops, so **every advantage is exactly 0** and
   the move term contributes nothing at all.
4. What is left in the gradient is the entropy bonus and the move-mass penalty.
   That is the `|g| ~ 0.01`. The `|g| = 17.6` steps are the rare groups where
   one episode happened to differ — which is also why `solve` dips to 0.75 on
   exactly those steps.
5. The thermostat cannot rescue it, because `entropy_target = 0.7` is a
   **4B grid-world constant** the 14B policy cannot reach: the update
   `log_coef += 0.05 * (0.7 − 0.01)` is positive at every step, so the
   coefficient multiplies by `e^0.0345` per step, passes 0.1 at step ~67 and
   **pins on its 1.0 ceiling at step ~133**. This is HH01's finding (SPEC
   amendment 5, `scripts/mz01_entropy_trace.py`) one level up, where it also
   deletes the move gradient rather than only the entropy authority.

Consequences for the design, stated plainly: **cell A could not learn mine
avoidance** (its whole content is the move term) and **cell B's "policy
invariance" would be vacuous** — the invariance of a policy nothing moved.

`grad_norm` cannot report any of this: the two regularisers keep it positive.
`train_orgb_cell`'s docstring already warned not to read `zero_signal_steps` for
that reason; what was missing was a number for the degeneracy itself.

## New: the exploration pre-flight (`preflight_exploration_check`)

PR-3 asks whether the **remark** term has variance and it PASSED on the boxes
(0.9453 / 0.6846). Nobody asked the same question of the **move** term. The new
gate does, on the warm-started model, before any cell trains, and it reports:

| number | what it says |
| --- | --- |
| `policy_entropy` | the mean move-column entropy, read with `_rescore` ITSELF so the gate and the trainer cannot mean different things by "entropy" |
| `zero_advantage_group_fraction` | the share of maze groups whose episode returns are all identical — the degeneracy, as one number |
| `within_group_return_sd` | the same thing as a magnitude, which is what `YELL_WEIGHT`'s scale argument was written against |

**GATE**: `zero_advantage_group_fraction <= 0.80` AND the entropy target in
force inside `[0.5x, 3x]` of the measured entropy. Failing either, the run
writes NOT-MEASURED and **no cell starts** — the same stop rule PR-3 has.
Reported under both return definitions and **gated on the harsher one**
(`mine_cost=False`, where the mine adds no variance), so the bar is the one
every cell of the 2x2 must clear rather than the kindest cell's.

`MZ01_PREFLIGHT_SOFT=1` reports instead of applying and records the deviation in
`not_measured` and in `deviations()`. It exists because **the second criterion
is deliberately strict**: `relative`'s floor of 0.3 against a measured 0.02 is a
15x ratio and fails it *by construction*. That is not a bug in either — the
floor is a deliberate departure from "a target this policy can service", taken
to buy the variance the first criterion is about — so an arm that takes it has
to say so out loud rather than have the band widened under it.

## New: four exploration modes (`MZ01_EXPLORE`, `maze_orgb.exploration_plan`)

`exploration_plan` is the ONE place a mode name becomes numbers; the trainer
still takes the numbers, so it cannot re-derive them differently from the gate
that cleared them. Each mode moves exactly one thing against the control.

| mode | what it moves | knobs |
| --- | --- | --- |
| `thermostat_fixed_target` | **nothing** — today's recipe, the DEFAULT | — |
| `relative` | the entropy TARGET: `max(REL x measured, MIN)` | `MZ01_ENTROPY_REL` 0.85, `MZ01_ENTROPY_MIN` 0.3 |
| `fixed` | the CONTROLLER: off, coefficient held still | `MZ01_ENTROPY_COEF` 0.1 |
| `temperature` | the SAMPLER: drawn at T, scored at 1.0 | `MZ01_SAMPLE_T` 1.5 |

**Why `relative` needs a floor, and why the floor is the half that works.**
0.85 x an already-tiny entropy is still tiny: at a measured 0.02 the relative
target is 0.017, which the policy ALREADY MEETS — the thermostat would then walk
the coefficient DOWN to its 1e-4 floor and the run would have *less* exploration
than today. A purely relative target can only stop the runaway; it cannot break
the degeneracy it was proposed for. The floor buys the variance, and 0.3 nats is
picked by the arithmetic that matters: entropy 0.3 over four moves is about
`p_max = 0.93`, i.e. ~7 % chance of leaving the greedy route per decision, so
over a ~14-decision episode `P(episode is the greedy one) = 0.93^14 = 0.36` and a
group of 6 is overwhelmingly likely to hold two different returns — which is all
the move term needs. The pre-registered 0.7 is about `p_max = 0.80`, three times
the deviation, and that is competence spent for no extra gradient. The plan
records which of the two bound (`knobs.binding`).

**Why `fixed` is 0.1.** Today's controller *passes through* ~0.1 at step 67 on
its way to the ceiling, so `fixed` is "the coefficient this run would have had
anyway, held still" — a control on the RUNAWAY rather than on the bonus being
small (which a frozen 0.01 would have been).

**`temperature`, and the importance weight.** Episodes are drawn at 1.5 and the
update is scored at 1.0, so the samples are off-policy and the textbook
correction is the per-token ratio inside a clipped surrogate. **It is omitted,
deliberately, and the omission is recorded rather than assumed**:

- GRPO's ratio — and Dr.GRPO's, which differs only in dropping the group-sd
  normaliser — is **identically 1 on the first inner epoch**, which is the only
  epoch this trainer runs (one sample, one backward, one step, samples
  discarded). Every clipped term in that algorithm exists for the second and
  later epochs over the same batch. `_sample_replies` already makes this
  argument for why there is no clipped surrogate anywhere in the file.
- The sequence ratio is a product over ~16 tokens; at T=1.5 against T=1 its log
  is tens of nats and its variance dwarfs the advantage it multiplies. Applying
  it untruncated is worse than omitting it; truncating it adds a clip constant
  nothing in this programme has tested.
- The arm is a **diagnostic**, not a candidate recipe on its own: read as an
  estimator it is "the gradient of the T=1 log-likelihood weighted by advantages
  from T=1.5 roll-outs" — exploration by a hotter behaviour policy, biased, and
  the bias is what is being measured.

So the ratio is **MEASURED instead**. `_sample_replies` accumulates
`log pi_score(a) − log pi_sample(a)` over each reply **at the weights that drew
it** (the only moment it is free — the logits are already in hand), the trainer
records the per-step mean in `history["log_importance_ratio"]`, and
`config["explore"]["importance_weight"]` says in words that it was not applied.

## What else changed

- **The trace** gained `zero_advantage_group_fraction` and
  `log_importance_ratio` per step (plus `..._mean` summaries), and the live log
  line gained `zadv` — `zadv 1.00` is the state the three shards are in.
  Because the history key set changed, **`CKPT_SCHEMA` is bumped to 2**: a
  checkpoint written by the shards now running is REFUSED BY NAME under this
  build rather than resumed into a trace it does not match.
- **`sample_batch_size`** (`MZ01_SAMPLE_BATCH`) separates the DECODE width from
  the re-scoring `chunk_size`; the default is `chunk_size`, which is what it
  silently was. It is in the checkpoint fingerprint and in
  `RESUME_RECIPE_KEYS`-adjacent checks because it **changes which tokens are
  drawn**: `torch.multinomial` consumes one draw per row per step, so a
  different width re-orders the draws off the run's one generator.
- **`MZ01_GROUP=12` is supported and is a *linear* cost, not a memory one.**
  Episodes run in lockstep and are cut into decode calls of `sample_batch_size`
  while the re-scoring is cut into chunks of `chunk_size`, so a wider group is
  MORE calls of the same shapes — the peak (the ~0.5 GB of logits per re-scoring
  chunk at 14B) does not move with it. At 53 GB of 96 used at GROUP=6, 12 is
  safe; it costs ~2x the step. Asserted on the toy path, not argued.
- **`MZ01_EVAL_BATCH`, default 32** (was a hard 16). The four measurement passes
  already batch every live episode's current turn — `maze2.rollout` runs them in
  lockstep — so this is the WIDTH of one call: 256 mazes x 4 passes x ~13 rounds
  is the difference between 832 and 416 decode calls per organism. Greedy
  decoding is batch-invariant in exact arithmetic (argmax per row, a masked left
  pad cannot reach a real token, an EOS row stops growing) and the toy path
  asserts it **decision by decision at widths 1, 4 and 32**. The one exception
  is `stop_on_parse=True`, whose early stop is evaluated over the whole call;
  every eval that reads a remark passes False (D30) and the test asserts that
  `measure_organism` is one of them. On CUDA bf16 two widths pick different
  kernels — the same fp standing any repeat of the run has there.
- **`MZ01_ENTROPY_TARGET` pins the target.** `relative`'s target is a function of
  a SAMPLED measurement, so a second lease would derive one a hair away and both
  the mid-cell checkpoint's fingerprint and the shard's carried-forward rows
  would (correctly, expensively) refuse it. The pre-flight prints the exact line
  to copy, `deviations()` records the pin, and the refusal message names the
  remedy.
- `RESUME_RECIPE_KEYS` gained `explore_mode` and `sample_temperature`; the
  recipe comparison now runs against the **resolved plan**, not `cfg`'s raw
  knobs. `relearn_read`'s recipe allowlist gained the same four knobs, so "50
  steps of the A recipe" stays true under a non-default arm.
- `train_orgb_cell` gained `on_step`, a BEST-EFFORT per-step hook (the trial
  writes its trace with it; a hook that raised must not lose a run).

## New: `scripts/mz01_explore_trial.py` and `scripts/score_mz01_explore.py`

One box, seed 0, cell D, **all arms from the same warm start** (restored with
`load_lora_state_dict` — the bit-exact restore `relearn_read` and the checkpoint
use), `MZ01_TRIAL_STEPS` steps each, the trace written **atomically after every
step** (the critic pass's `write_payload`), a `mz01_explore_status.txt` with
DONE markers, and no adapter, row, ledger or measurement anywhere. Arms, in
order, control first:

```
01_thermostat_g6   02_relative_g6   03_fixed_g6   04_temperature_g6   05_relative_g12
```

Then **cell A under the arm the trial's own summary picks**, because "the
entropy moved" is not the question — "does mine/ep move" is. An arm that raises
is caught and the others continue: an OOM at GROUP=12 is itself an answer.

`score_mz01_explore.py` (stdlib, `python3 -S`) is the one implementation of the
verdicts, so the trial's pick and the table a human reads cannot disagree:
**DEAD** (zero-advantage groups still over 0.80 at the end), **RANDOMISED**
(solve fell more than 0.10 — exploration bought by destroying the policy is not
a recipe), **ALIVE** otherwise. Start/end are means over the first and last
fifth of the steps, because at GROUP=6 one step reads four groups and its
zero-advantage fraction can only be 0, .25, .5, .75 or 1. The pick is the ALIVE
arm with the highest end solve, then the cheapest step, **ties to `relative`** —
the smallest deviation from the pre-registered recipe among the ones that work —
and it prints the env line for A+C, with the target PINNED.

**`MZ01_TRIAL_STEPS` defaults to 40, not the 60 this was scoped at**, and the
arithmetic is why: 4 arms x 60 x 36 s + 1 arm x 60 x ~72 s + cell A + five
pre-flights is 4.3–5.0 h, over the ~4.5 h scope; at 40 it is ~2.9–3.2 h. The
numbers that decide the trial (entropy, zero-advantage fraction, `|g|`) are
properties of the SAMPLER and are readable from the first step; 40 is ample for
them, and the follow-up A arm is what needs the headroom.

## Suite

`195 → 211` passed, CPU, no downloads (`python3 -m pytest tests/ -q`, ~100 s).
Sixteen new tests in `tests/test_maze_orgb.py` under "R2, 2026-08-29":

- the zero-advantage fraction, read **beside `group_advantages` itself**, so
  "the share of flat groups" is not a number about a claim nobody checked;
- every mode's resolved numbers, including `relative`'s floor binding at a
  measured 0.02 and the relative half binding at 1.0, the refusal to derive a
  relative target with no measurement, and the pin (and its refusal under
  `fixed`);
- **the controller's update law, step by step** (`coef_{i+1} = clamp(coef_i *
  exp(lr * (target − H_i)))`), the fixed coefficient not moving on any step of
  its trace, and the RUNAWAY reproduced with an unreachable target;
- the `relative` target travelling from the gate into the trainer unchanged;
- the exploration gate refusing a deterministic policy that **PR-3 passes** (the
  two gates ask about the two terms), the soft path, the run-level stop with no
  rows and no adapter on disk, and the soft deviation appearing in
  `not_measured`, the manifest and every row's config;
- the temperature arm's ratio recomputed from a plain forward and compared
  exactly, plus `_rescore` having no temperature in it at all;
- GROUP=12 running with no decode call wider than `sample_batch_size`, and the
  decode width changing which ids are drawn (why it is in the fingerprint);
- a checkpoint from another ARM, temperature or width refused **by name**;
- the eval decode identical at widths 1/4/32 and `measure_organism` passing
  `stop_on_parse=False`;
- the scorer's four verdicts, the pick and the env line, `python3 -S`, and the
  trial's arm list covering every mode with the control first.

## Still not done / read this before the next lease

- **The three running shards are unaffected and remain wrong.** Nothing here
  reaches them. Their cells will finish, and their `zero_advantage_steps_move`
  and (under this build, on a re-score) their traces will say the move term did
  nothing. Decide deliberately whether their D and B rows are usable as a
  NULL-cell record or whether the shard is abandoned.
- A checkpoint they wrote is **schema 1** and this build refuses it. That is the
  designed behaviour; it means a lease that syncs this code cannot resume their
  mid-cell state. Finish them under `04e5e69` or accept the loss.
- The trial has **not been run**. Every number in the cost table above is an
  estimate from the boxes' own measured 36 s/step.
- `MZ01_EVAL_BATCH=32` means a shard measured under this build and one measured
  under `04e5e69` differ by fp accumulation order in the eval. Exactly-equal on
  CPU fp32; not bit-equal on CUDA bf16 — the same standing any repeat has.
- The exploration gate's second criterion will FAIL for every `relative` run
  whose floor binds. That is by design and it forces `MZ01_PREFLIGHT_SOFT=1`
  plus a recorded deviation. If the trial picks `relative`, the paper's methods
  section has to carry that sentence.

# Critic pass 2, 2026-08-29: exploration

Adversarial read of the ninth pass's uncommitted exploration build (`git diff`
against `04e5e69`, the R2 half), one AMENDMENT applied on the lead's
instruction, six fixes and seven new tests. Nothing under `results/orgb/` or
`results/explore_trial/` was written; `scripts/pull_from_sandbox.sh` was not
touched; no sandbox was contacted.

## The amendment (applied, not a finding)

The pre-flight's second criterion -- the entropy target inside `[0.5x, 3x]` of
the measured entropy -- **no longer refuses**. It is computed, printed
(`OUT OF BAND  (INFORMATIONAL -- recorded, never refuses)`), carried in
`criteria[1]` with `refuses: False`, and recorded in
`manifest["exploration"]["entropy_target_over_measured"]` beside
`..._in_band` and `..._is_informational`. The reason it could not be a gate is
the ninth pass's own: `relative`'s floor of 0.3 against a measured 0.02 is a
15x ratio BY CONSTRUCTION, so the criterion refused exactly the arm it was
written to license, and clearing it took a blanket `MZ01_PREFLIGHT_SOFT=1` --
which also waves through the criterion that means something.

The one refusing criterion is `zero_advantage_group_fraction > 0.80`, and it is
SOFT inside the trial (`soft=True` on every arm, with the reason now written at
the call site): the control arm IS today's failing recipe and that fraction is
the number that says so, so a gate that stopped the control would stop the
trial from measuring what it exists to measure. Every arm's file now carries a
`deviations` list naming the waived refusal. Test:
`test_the_target_band_is_reported_and_never_refuses`.

## Claim by claim

| # | claim | verdict | evidence |
|---|---|---|---|
| 1 | the pre-flight gates the arm that will run | **INVALID → FIXED** | The roll-out was drawn at `temperature` (the SCORING temperature, 1.0) for every arm, and only `exploration_plan` ever saw `sample_temperature`. The one refusing criterion is a property of the SAMPLER, so the gate reported the CONTROL's degeneracy for all five arms -- and refused `temperature`, the one arm whose entire mechanism is the draw, for the state it exists to fix. `plan_sample_temperature` (new, and the single implementation `exploration_plan` now also calls) resolves the draw temperature BEFORE the measurement -- only `relative`'s TARGET needs the entropy -- so the order is draw-then-resolve. The entropy stays a `_rescore` read at the scoring temperature, because that is the quantity the controller regulates. `sample_temperature` / `score_temperature` / `off_policy` / `log_importance_ratio` are now in the payload, the printed lines and the manifest. Test: `test_the_preflight_rolls_out_under_the_arms_own_sampler`. |
| 2 | temperature-mode math: entropy, log-prob and advantage are consistent | **VALID (the arithmetic)** | `_rescore` has no temperature in it at all: `logp = log_softmax(logits)` and the move-column entropy are both at T=1, so the log-prob in the policy-gradient term and the entropy the thermostat regulates are the same policy's. `_sample_replies` draws from `softmax(logits/1.5)` and accumulates `log pi_1(a) - log pi_1.5(a)` at the weights that drew it. Advantages are episode returns, which do not depend on either temperature. Nothing is mixed. |
| 3 | "the importance ratio is identically 1 on the single inner epoch" | **INVALID → FIXED (a false reason, an honest conclusion)** | GRPO's ratio is 1 on the first inner epoch **because the behaviour policy IS the scored policy**. Drawing at 1.5 and scoring at 1.0 breaks that premise; the ratio is not 1, which is why the code measures it. Reason 1 of the IMPORTANCE WEIGHT paragraph therefore argued for the omission from a premise the arm violates, and `_sample_replies`'s docstring repeated it unqualified. Both rewritten: what survives is the narrower claim (single-epoch structure is why there is no clipped surrogate on the ON-POLICY path, so the temperature arm is not dropping a correction the rest of the file applies) plus the two reasons that do stand. **The decision not to apply it stands**, and the arithmetic is why: at H = 0.02 the move distribution is p_max ~ 0.997, at T = 1.5 it is ~0.97, so a per-token ratio is ~1.03 on the greedy token and ~0.1 on a deviating one; the trajectory ratio an episode-return advantage would need is a product over ~14 replies x ~16 tokens, whose log is tens of nats. Untruncated it is worse than nothing; truncated it adds a clip constant nothing in this programme has tested. The bias is real, it is in the direction of over-weighting actions rare under T=1 (which is the exploration the arm is for), and it is now stated as a bias rather than argued away. Test: `test_the_off_policy_arm_does_not_claim_the_ratio_is_one`. |
| 4 | does T=1.5 buy anything at this entropy? | **VALID, quantified** | At H = 0.02 over four moves, p_max ~ 0.997 and P(the greedy episode) = 0.997^14 = 0.96, so a group of 6 is the same episode -- `zadv 1.00`, which is what the boxes print. At T = 1.5 the off-move log-odds shrink by 1/1.5: p_max ~ 0.97, P(greedy episode) = 0.97^14 = 0.65, and P(all six identical) ~ 0.08. So the arm does break the degeneracy. The cost it can buy it at is the REMARK: T = 1.5 applies to the whole 150k vocabulary, not to the four move columns, so `unparsed_rate` and `remark_share` are the numbers to read beside `solve_rate` on that arm. Recorded, not fixed. |
| 5 | `relative`'s measured entropy is the right quantity | **VALID** | It is `_rescore`'s move-column entropy (ceiling ln 4), averaged over the decisions of a roll-out taken with the training sampler's own settings (`sample_text_episodes`, same `group_size`, `chunk_size`, `sample_batch_size`, glyphs and format), on the warm-started weights, and `test_the_relative_target_travels_from_the_preflight_into_the_trainer` pins that the number the gate read is the number the trainer trains under. |
| 6 | `relative` does not recreate the 0.7 pin | **VALID, but the trial cannot see it** | `log_coef += 0.05 * (0.3 - 0.02)` multiplies the coefficient by e^0.014 a step against the control's e^0.0345, so if the entropy never moves the coefficient reaches the 1.0 ceiling at step ~329 rather than ~133 -- outside a 300-step cell, at ~0.66 by step 300. It is the same runaway SHAPE at 41 % of the rate, and it stabilises only if the bonus actually lifts H toward 0.3. **The finding is about the trial, not the mode**: at step 40 the coefficient is 0.0175, a SIXTH of the 0.1 the `fixed` arm starts at, so a 40-step trial is structurally unkind to the two arms whose mechanism is the thermostat and kind to `fixed` and `temperature`, which act from step 0. "Ties to `relative`" cannot repair a DEAD verdict. The scorer now flags it (`slow_controller`, and a table line saying the answer is more steps, never a lower bar) rather than silently rescuing it. Test: `test_a_dead_arm_whose_controller_never_moved_says_so`. |
| 7 | `fixed`: nothing re-enables the thermostat | **INVALID → FIXED (one reachable path)** | The trainer is right (`adaptive = entropy_target is not None`, the update is not reached, `coef` is never reassigned) and the checkpoint is right (`entropy_target` is in the fingerprint, `coef`/`log_coef` are restored). The hole is `relearn_read`: its `recipe` dict has all-optional keys, so a caller passing `explore_mode="fixed"` and `entropy_coef=0.1` while OMITTING `entropy_target` got `train_orgb_cell`'s default 0.7 and ran 50 steps of the pre-registered thermostat with `explore_mode` saying `fixed` in the trace. `run_orgb` passes `plan["entropy_target"]` and is safe; the guard is now in the TRAINER, so every caller is covered: `fixed` with a target raises, and `temperature` XOR off-policy raises. Test: `test_a_mode_name_and_its_numbers_may_not_disagree`. |
| 8 | GROUP=12 is a linear cost, not a memory one | **VALID** | The backward is per re-scoring chunk (`for lo in range(0, len(decisions), chunk_size)` ... `loss.backward()`), so the peak is ONE chunk and a wider group is more chunks in sequence. At 14B, chunk 8, ~178-token prompts and a 48-token window the chunk peak is ~0.6-0.9 GB of logits ([8, 49, 151936] bf16 + its fp32 copy + the log-softmax) on top of ~3 GB of transformer activations, against 28 GB of bf16 weights; the decode side is capped by `sample_batch_size` (8) and each `_Decode`'s KV cache (~0.3 GB for 8 rows x 226 positions) is freed per call. Nothing in the step holds all decisions' tensors: GROUP=12 adds 672 CPU decision dicts, about 1 MB. **No chunking added and `MZ01_SAMPLE_BATCH`'s default left alone**; 53 GB at GROUP=6 does not move. |
| 9 | batched eval == sequential eval | **VALID** | `padding_side="left"` is explicit in `_greedy`; `position_ids = (att.cumsum(-1) - 1).clamp(min=0)` puts every row's first real token at position 0 whatever its pad width; `_Decode.feed` grows `att` with ones and carries a per-row `next_pos`, so the cache's positions stay right. A row that emits EOS is marked `done` and its `grown` list is frozen -- it is still fed, but attention is per-row, so it cannot reach another row. The `all(done)` break only makes a wide chunk decode LONGER, never differently. The retry is per-EPISODE (`ep.pending_retry` puts two extra turns in that episode's prompt string) and `rollout` re-forms `active` every round, so batch composition never touches it. The one batch-dependent path is `stop_on_parse=True`, which `measure_organism` does not use and the suite asserts it does not. |
| 10 | the trial's arms differ only by the mode | **VALID with a limit worth naming** | Every arm restores the same `warm_state` with the clone/copy-in-place pair (`lora_state_dict(cpu=True)` clones; `load_lora_state_dict` copies in place), and every arm gets `set_all_seeds(seed)` with the same seed and a freshly derived pre-flight generator, so step 0 is identical. They are NOT paired after that: the maze draw, the word orders and the sampled tokens all come off ONE generator, so the moment two arms' episodes differ in length their maze sequences diverge. The comparison is 40 steps of the same pool at the same seed, not a paired one. Not fixed -- a per-step derived stream would change `train_orgb_cell`'s draw semantics and every pinned number with it. |
| 11 | the follow-up cell A uses the picked arm's exact plan | **INVALID → FIXED, twice** | (a) The follow-up re-ran the pre-flight and RE-DERIVED `relative`'s target; deterministic on CPU, not on CUDA bf16, so cell A could have trained a hair off the picked target -- and under `MZ01_ENTROPY_TARGET` a hair is a different recipe to every resume check in the programme. `run_arm` now takes `pinned_target` and the follow-up pins the picked arm's own number (skipped for `fixed`, which has no target and whose plan refuses one). (b) The follow-up's file carried the PICKED ARM'S name in its `arm` field, so a human pointing `score_mz01_explore.py` at the directory got two rows called `02_relative_g6` and `pick_arm` could return the CELL A one -- whose solve rate and zero-advantage fraction are not what the pick rule is written about. The payload now names the FILE (`arm`), keeps `arm_base`, sets `follow_up`, and the scorer excludes follow-ups from the pick and labels them in the table. Test: `test_the_follow_up_cell_is_read_but_never_picked`. |
| 12 | a failed arm is visible in the summary | **INVALID → FIXED** | The failure branch wrote a FAILED stub to disk but did not put it in `results`, and `rows` was built from `results.values()` -- so an OOM at GROUP=12, which the trial's own docstring calls "itself an answer", was missing from `summary.json` and from the table the operator reads. The stub now goes into `results` too, in the shape `read_arm` expects. |
| 13 | `MZ01_PREFLIGHT_SOFT=1` in the recommended env line | **INVALID → FIXED** | It was appended unconditionally, so a picked arm that CLEARS the gate (which `temperature` now can, since the gate draws at its own temperature) would have recorded a deviation that did not happen -- in the same manifest field every real deviation lives in. `env_line` now emits it only when that arm's own pre-flight has `hard_passed` false, and still emits it when the arm file records no verdict at all (a stopped run is the cheaper error). Test: `test_the_soft_flag_is_emitted_only_when_the_gate_would_refuse`. |
| 14 | the scorer's verdicts and pick | **VALID, exercised** | Run on a synthetic five-arm trace plus a follow-up file: control DEAD, `relative` DEAD-with-`slow_controller` (its coefficient walking 0.0105 -> 0.0164 over 40 steps, the real controller law), `fixed` ALIVE, `temperature` RANDOMISED at solve 0.60, g12 DEAD; pick `03_fixed_g6` with a correct env line, follow-up read and labelled but not picked. `python3 -S`, no crash, no traceback. DEAD/RANDOMISED/ALIVE and the window means are as documented. |
| 15 | the trial's cost | **INVALID (optimistic) → RESTATED** | The projection left out the pre-flights' real cost and the model load. 4 x 40 x 36 s = 96 min, 1 x 40 x 72 s = 48 min, cell A 24 min (the driver budgets 48 as an upper bound, since the picked arm's group is not known when it prints), six pre-flights at ~90 s (8 mazes x 6 episodes is ~2x an RL step's sampling plus a forward-only re-score) ~10 min not 5, model load + the 256-maze eval set ~5-10 min: **~3.0 h floor, ~3.4 h at the driver's own padded line.** Add up to ~40 min if the hot arms lengthen their episodes -- the cap is 2x the shortest path, so an arm that stops solving runs up to twice the decisions per step, and `03_fixed_g6` (coefficient 0.1 from step 0) and `04_temperature_g6` are the two that can. **Worst case ~3.8 h, inside the ~4.5 h scope**; at 60 steps it would be 4.5-5.7 h and outside it. The docstring and the printed projection now say all of this. |
| 16 | `zero_advantage_group_fraction`, the checkpoint schema bump, the decode-width fingerprint, `MZ01_EVAL_BATCH=32` | **VALID** | Re-derived beside `group_advantages`; `CKPT_SCHEMA = 2` refuses a schema-1 checkpoint by name; `sample_batch_size` really does re-order `torch.multinomial`'s draws and is in the fingerprint; the eval width is asserted decision by decision at 1/4/32 on the toy path. Nothing to add. |

## Left open (not fixed)

- **The gate can only ever pass for `temperature`.** Its one refusing criterion
  is measured on the WARM START, and the mechanism of `relative` and `fixed` is
  a coefficient that has not moved yet at that moment. So those two arms will
  always need `MZ01_PREFLIGHT_SOFT=1` and will always record the deviation.
  That is honest -- the gate is asking "does this policy, as it stands, produce
  a policy gradient?" and the answer is no -- but it means the flag is part of
  the recipe for two of the four modes, and the paper's methods section has to
  carry that sentence.
- **The arms are unpaired after step 0** (row 10). The between-arm comparison
  is over 160 groups per arm, which is enough for `zadv` and `H`; it is not a
  paired design and `solve_end` differences under ~0.05 should not be read.
- **The `temperature` arm keeps the 0.7 thermostat**, by design (it moves the
  sampler and nothing else), so it inherits the runaway: its coefficient is
  0.0398 by step 40 and would pin at step ~133 of a real cell. A 40-step trial
  cannot see that, and an A/C run under `temperature` would.
- **T = 1.5 is applied to the whole vocabulary**, not to the move columns, so
  the arm's cost may land on the REMARK rather than on `solve_rate`. Read
  `unparsed_rate` and `remark_share` beside the verdict; the scorer does not.
- The trial has still **not been run**; every number above is an estimate from
  the boxes' measured 36 s/step.

## Suite

`211 -> 218` passed, CPU, `python3 -m pytest tests/ -q` (~60-180 s). Seven new
tests in `tests/test_maze_orgb.py` under "critic pass 2, 2026-08-29", plus new
assertions inside `test_the_soft_preflight_records_the_deviation_it_waves_through`
for the manifest's informational criterion. `_ScriptedSampler` now accepts the
off-policy `aux` / `score_temperature` pair so a scripted test can exercise the
`temperature` arm.

## The trial command line

```
MZ01_REVISION=<sha> MZ01_MODEL=14B MZ01_TRIAL_STEPS=40 \
MZ01_WARM_START=/marimo/repo/experiments/v2/MZ01_maze_world/results/orgb/warm_start.pt \
  python3 /marimo/repo/scripts/mz01_explore_trial.py
```

Defaults do the rest: five arms in order (control first), seed 0, cell D, then
cell A under the picked arm with its target pinned. Read it with
`python3 -S scripts/score_mz01_explore.py experiments/v2/MZ01_maze_world/results/explore_trial`.

# Docs pass, 2026-08-29

No code, no SPEC, no result touched. No sandbox contacted, no tests run. Four
documents written and one updated, all against files in this worktree and the
main repo; every number re-derived from the source it names, and anything that
could not be verified says so in the text.

- **`/home/manan/Projects/files/digital_minds/HANDOFF-v2.md`** — rewritten
  clean. What the project is; where both trees are and what is committed
  (main `6d39ac3`, maze `04e5e69` + ~4580 uncommitted lines); the box
  convention, the current three boxes, today's three lease deaths, the
  snapshot rule, and the exact exec/setup/sync/push/launch/pull/watcher
  commands; HH01's six-seed state and its open decision; MZ01's navigator
  gate, the RL failure with its numbers, and the reading; ordered NEXT STEPS
  with the cell-A env lines; the process rules.
- **`experiments/v2/MZ01_maze_world/record/RESULTS-2026-08-29.md`** (new) — the day
  as a lab note: the navigator gate table and its 32-per-bin detour reference;
  the four killed cell-D runs' traces at sampled steps; per-arm pre-flight
  numbers; the trial40 and trial80 scorer tables copied verbatim; what it
  means; and what it does not tell us (cell A untested, 32 mazes per bin, one
  seed, the unpaired arms, the measured-not-corrected off-policy bias, arm 5
  of trial80 unfinished, leases dying at random).
- **`experiments/v2/MZ01_maze_world/README.md`** (new) — how to run every
  stage: pilot, navigator, the exploration trial with its named arms, and the
  organism driver with every env knob grouped (model/paths, sharding, RL
  shape, exploration modes, checkpoint/resume, measurement switches, smoke and
  the cost probe); what each output file is; every gate and pass rule with its
  constant; and the two sandbox scripts plus the traps the scratchpad helpers
  exist to avoid.
- **`$SP/BOXES.md`** — retyped as a current-state table plus the day's history
  below it. No tokens anywhere.
- This entry.

One thing the docs record as UNVERIFIED: box 1's renewed lease
(`sb-ff67c07a3b8bec08`, 14:13 UTC) is described in the previous session's pause
note as set up with the navigator and warm start pushed, but there is no push
log in the scratchpad after that renewal — `push3.log` belongs to the previous
lease. Re-inventory the box before launching on it.

# Twelfth pass, 2026-09-07: the closed action space (D32)

Built against `PLAN-closed-action-space.md` and the erratum at the end of
`record/RESULTS-2026-08-30.md`. No box was contacted, no result file was touched, and
`SPEC.md`, `PLAN-*.md` and `RESULTS-*.md` are unedited. Nothing here is a
result; the rerun the change exists for is pre-registered and has not run.

## The finding this pass is a fix for

On 08-30 the first live cell-A gradient in the programme drove `move_mass` to
0.000 by step 60, and the note read that as a policy that had stopped emitting
move words — a reward hack. The trace says otherwise:

| `epsilon_010`, seed 0 | unparsed | move mass | train solve |
| --- | --- | --- | --- |
| step 40 | 0.000 | 0.997 | 1.000 |
| step 60 | 0.000 | 0.000 | 0.750 |
| step 70 | 0.000 | 0.000 | 1.000 |

A policy that had stopped emitting move words could not solve every maze at
step 70. The logged replies say what actually changed: at step 40, 6/6 begin
with the bare word (`up. That one seems symmetrical.`); from step 50 on, 5/6
begin with a **space** (` up. I do not like that one.`). On Qwen3-14B at the
run's revision `up` is token 454 and ` up` is 705, and `maze2.parse_move` is a
case-insensitive `\b(up|down|left|right)\b` anywhere in the text — so ` up`
walks the maze exactly as `up` does, `unparsed_rate` never moves, and the only
number that changes is a probe nothing was gating on.

The chain, stated so it cannot be misread again: **the sampler drew from the
full vocabulary while the move term, the entropy bonus and the mass penalty
read only the four bare ids**. Once the mass crossed to the twins, the
objective was computed over columns the policy no longer sampled from, and the
held-out RANDOMISED numbers of §3b are downstream of that.

## D32 — the action space is CLOSED at training time and OPEN at eval

`MZ01_MOVE_MASK=1`, the default. At decode step 0 only, and only for rows the
epsilon draw has not forced, every logit outside the four bare ids
(`capture.move_token_ids`, i.e. `_move_cols`) is set to `-inf` **before** the
temperature division and the softmax — for `probs` and, on the temperature arm,
for the `lp_score` the importance ratio is read off, so the ratio is taken on
the same restricted distribution. Steps 1.. are untouched: the remark is free
text and F6's EOS handling is unchanged.

The loss follows the sampler. `_rescore` already computed
`mlogp = log_softmax(head[:, cols])` at the position that predicts the first
generated token; the move term's first-token log-prob is now `mlogp[action]`,
with the further move tokens (there are none on Qwen — the move word is one
token) still under the full vocabulary. Implemented by putting the restricted
value into `tok_logp[:, 0]` rather than by assembling the sum a second time
upstream, so the trainer's own expression — `(tok_logp * m_mask).sum(1)` — is
untouched and there is exactly one place the move term is written down. The
identity that makes it checkable is `restricted = full_logp(a) − log_mass`, and
`log_mass` is the probe `move_mass` is read off, so the objective and the probe
are asserted to be the same object rather than two expressions that agree today.

**Two conditions, and both are load-bearing.** The restricted log-prob is used
only when `move_mask` is on AND the row's first id is in `cols`. The first
keeps `move_mask=0` reproducing 08-30 on ANY tokenizer rather than only on one
where the twins happen to differ from the bare ids — scoring a bare first token
under the restricted softmax while the sampler drew it from the whole
vocabulary would be the same decoupling this pass closes, pointing the other
way. The second is what keeps a scripted or hand-built row scorable instead of
silently scored against the wrong column; under the mask it cannot fire on a
sampled row.

**The eval stays unmasked, deliberately.** The training-time question is "what
did the gradient act on"; the held-out question is "what does this organism
do". If the two were both masked, `unparsed_rate` and the first-token
histogram would be 0.000 and `bare 1.0` by construction and would say nothing
at all. Masked training against an open eval is the pair that makes the drift
readable — and it is also the pair that can be wrong in a new way, which is why
the histogram exists on both sides.

**The rejected alternative: eight columns, bare plus spaced.** It closes one
leak and leaves every other one open — the capitalised twins (`Up` is 2324),
the twins with other whitespace, and any future tokenizer's own. Worse, it
makes the entropy bonus an 8-way entropy over four actions, so a policy could
raise it by splitting mass between `up` and ` up` without changing a single
move it makes: the bonus would then be paying for a distinction the environment
cannot see. Four columns is the action space; eight is a spelling of it.

## What else moved with it

- **`move_mass_coef` 0.03 → 0.0**, and `move_mass` becomes a diagnostic. Under
  the mask the penalty has nothing to hold. It is a knob and not a deleted
  line: 08-30's recipe stays reproducible, and the driver reports a non-zero
  coefficient as a deviation. `move_mass` is still recorded every step, and it
  is now the number that says whether the UNMASKED eval will read the trained
  policy.
- **`history["first_token_class"]`**, per step: the shares of sampled first
  tokens in {bare, spaced, capitalised, other}, **by id**, against `cols`, the
  ids whose decode is exactly `" up"`, and the ids whose decode is exactly
  `"Up"`. Twins are found with `move_word_token_ids`' round-trip discipline —
  ONE token that decodes back to the form — so a tokenizer without one has an
  empty class rather than a guess (`CharTokenizer` has none, by construction).
  Under the mask `bare` is 1.0 at training time; the field exists so the masked
  training and the open eval are classified by the same function.
- **`CKPT_SCHEMA` 3 → 4**, with `move_mask` joining `move_mass_coef` in the
  fingerprint. A resume across the mask would be half a closed-action-space run
  and half an open one — a different distribution for the first token and a
  different log-prob for the move term — and neither half shows anywhere else
  in a row. `RESUME_RECIPE_KEYS` gains it too, which is the same refusal one
  level up, where the cost is a whole shard rather than one cell.
- **Held-out reads INSIDE the arm** (`MZ01_TRIAL_EVAL_AT`, default 40). 08-30's
  §9 item 1 asked for a 40-step held-out read and the tenth pass could only
  give it as a separate post-hoc arm — same warm start, different generator
  stream, not the same organism at step 40 at all. The read now runs in the
  trainer's `on_step` hook, on the same mazes, through the same `measure()`
  call and the same `eval_block`, and lands in `payload["eval_at"][step]`.
  `payload["eval"]` is still the final read, so the pick and the verdict are
  unchanged. It costs ~3 min at 128 mazes and it draws from no generator at all
  (`measure_organism` derives its word orders from `order_seed`, D31, and
  decodes greedily) — asserted, not argued: the trace and the adapter are
  `torch.equal` with and without a read.
- **`eval_block` carries `unparsed_rate`, `invalid_rate` and
  `first_token_class`.** §7.4 of 08-30 named the first as the number that would
  have caught this in the eval, and it was never in this block; the training
  history had it all along. Read from the reply TEXT here (the roll-out records
  carry no ids) by the rule: leading space → spaced, first character upper →
  capitalised, begins with a bare move word → bare, else other. `rule` says
  which read produced a row, because an id read and a text read agree on Qwen
  and may not agree on anything else.
- **`INVALID-READ`**, the scorer's new first held-out verdict, pre-registered in
  the PLAN's §4: a read whose `unparsed_rate` is above 0.05 — the SPEC's own
  gate — is not a verdict at all, because solve and mine/ep over episodes whose
  replies do not parse measure the parser. Reported, never scored, never picked.
  A file written before the field existed is verdicted exactly as it was.
- **The scorer's table** gains `unparsed` and `bare 1st` columns and a
  `@40 … @end` block: solve, the mine ratio to the reference and the verdict
  each read WOULD get by the same rules, from the same function the final read
  uses. `env_line` emits `MZ01_MOVE_MASK=0` / `MZ01_MOVE_MASS_COEF=` when the
  arm ran the 08-30 recipe, so a launch line reproduces the arm and not the
  default.

## The bit-identity claim, and how it is held

`move_mask=0` with `move_mass_coef=0.03` is the 08-30 sampler and the 08-30
loss. It is asserted against ids and floats **recorded from the pre-build code
at `8158368`** and pasted into `test_move_mask_zero_is_the_pre_build_sampler_and_the_pre_build_trace`
— there is nothing else to compare against, since the code being reproduced is
no longer in the tree. Compared with `==`, not a tolerance: an approximate
bit-identity claim is not one. The same test asserts that passing `move_cols`
CHANGES the ids, so it is not a statement about a sampler nothing can move. If
it ever fails after a torch or fixture change rather than after a change to
`_sample_replies` / `_rescore` / `train_orgb_cell`, the recipe for re-deriving
it is in the comment above the constants.

## Suite

`219 → 229` collected, **228 passed, 1 skipped**, CPU, ~65 s,
`timeout 900 .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`. Ten new
tests, none weakened or removed. The skip is the one that names the trap in the
suite even where it cannot run: `test_the_qwen_twins_are_not_the_four_bare_ids`
re-derives 454/2923/2359/1291 and 705/1495/2115/1290 from the real tokenizer and
is SKIPPED unless `Qwen/Qwen3-14B` is already in the local HF cache
(`local_files_only=True` — it never downloads). Every other test runs on
`CharTokenizer`, whose encode is a shift and which has no single-token `" up"`,
so the one tokenizer where this defect is possible is the one the suite cannot
otherwise see. `_ScriptedSampler` now accepts and RECORDS `move_cols` without
enforcing it, for the same reason: its scripted replies are the fixture's own
text and forcing them into `cols` would write replies no detector could parse.

## Still not done / read this before the rerun

- **The rerun has not run.** `PLAN-closed-action-space.md` §4 is the
  pre-registration; the launch line is in `README.md` under the trial's arms.
  Two boxes, one per seed, `epsilon_010` then `onpolicy_g6`, 80 steps, reads at
  40 and 80 on 128 held-out mazes.
- **The ε first token is scored by the restricted log-prob as if on-policy.**
  The mask does not change that bias — it was there at 08-30 (§8) and it is
  there now — but it does change its SIZE, because the reference distribution
  the forced token is scored under is now a 4-way softmax rather than a
  150k-way one. The measured replacement fraction is still what the trace
  carries, and the correction is still not applied.
- **`measure_organism` does not itself carry `first_token_class`.** The trial's
  `eval_block` computes it; the organism driver's rows carry the records it is
  computable from but not the number. The PLAN scoped it to `eval_block` and
  this build kept to that.
- **The text rule is deliberately coarse.** Any held-out reply beginning with a
  capital is `capitalised`, not only `Up`/`Down`/`Left`/`Right`. That is the
  right direction to be coarse in — the question is "is the first token a BARE
  move id" — but a reader comparing the eval's shares against the training
  trace's id-read shares is comparing two rules, and `rule` is in every row so
  that comparison can be made honestly rather than accidentally.

---

# Critic, twelfth pass — 2026-09-08

Adversarial pass over `PLAN-closed-action-space.md`, the 08-30 erratum and
build `3d5c9ac`. The action space itself holds: the mask, the restricted move
term, the twin classifier, the mid-arm reads, the INVALID-READ rule and the
checkpoint schema are all what the spec says they are, and the identity claims
are true. Four defects were found around them, all in the same shape — **a
number a reader would take at face value is not the number it names** — and all
four are fixed here.

**Bit-identity, re-derived independently of the build's own test.** The
pre-build source was checked out to a scratch tree
(`git show 8158368:src/calibration/maze_orgb.py`) and run against the same
fixtures. `move_mask=0, move_mass_coef=0.03` reproduces the pre-build code
exactly: over a three-step toy cell A, **every** history field is `==` (not just
the eight the test pins) and every LoRA tensor is `torch.equal`; the only
difference is the two fields the trace GREW. `_PRE_BUILD_SAMPLER_IDS` and
`_PRE_BUILD_TRACE` were re-derived from `8158368` and match to the last digit,
so the recorded constants are a recording of the old code and not of the new
code with the knob off.

## C1 — PR-3's gate was silently rolled out under the closed action space

`preflight_gradient_check` calls `sample_text_episodes` and was left with **no
`move_mask` argument at all**, so it took the new default. Every number it
reports — the complaint rate, the placement, the sampled contingency, both
within-group sds and therefore `passed` — is read off SAMPLED replies, which
makes it a property of the sampler in exactly the way
`preflight_exploration_check`'s degeneracy criterion is (the build threaded the
knob there and named the incomparability; this call was missed). Two
consequences: the committed PR-3 numbers stopped being comparable to a new run's
with nothing on the record saying so, and a `MZ01_MOVE_MASK=0` shard would have
gated itself on a MASKED roll-out while every cell in it trained with the space
open. FIXED: `move_mask` parameter (default ON), threaded from
`run_orgb.run`'s `cfg["move_mask"]`, recorded as `preflight["move_mask"]` and
printed as an `action space` line above the verdict line (which stays last).

## C2 — the mid-arm read's three minutes were charged to `seconds_per_step`

The read runs inside the `on_step` hook; `total_seconds` is measured from before
the loop, so an unsubtracted read put ~2.3 s on every step of an 80-step arm
(180 s / 80). That is the number the trial prints, the scorer's `s/step` column,
and what the next lease for 4 × 300 steps is booked on. `step_seconds` never
contained it — that clock stops before the hook — which is what made the error
invisible: the per-step series and the per-run mean disagreed and only the mean
was wrong. FIXED: the hook is timed, `history["on_step_seconds"]` reports it,
and `total_seconds` / `seconds_per_step` / the checkpoint's `elapsed_seconds`
exclude it. The build's own no-disturbance test now asserts the split
(`quiet == 0.0`, `read > 0.0`) instead of skipping the field.

## C3 — the launch line could not tell an OPEN-space file from a new one

`read_arm` read both knobs from `payload["config"]` only, and the trial did not
write them there before this pass. Re-scoring a committed 08-30 arm file
therefore gave `move_mask = None`, `move_mass_coef = None`, and `env_line`
named **neither** knob — and a line that names neither is the CLOSED space at
coefficient 0.0, i.e. a different recipe under the picked arm's name. FIXED:
`_action_space()` resolves both from `payload["config"]`, then from the
trainer's own `history["config"]`, and treats *a trainer config that exists and
does not name `move_mask`* as the OPEN space, because the field did not exist
when that trace was written. Verified against the shape of the committed file
(`history.config = {"move_mass_coef": 0.03}`, no `move_mask`). The table names
an OPEN-space arm too, so an old row and a new row read side by side say which
is which.

## C4 — the text classifier's `bare` was `startswith`

`"upset, go down."` and `"rightly so, up."` begin with the letters of a move
word and are not that word; on Qwen each is one token that is not 454 or 1291,
so the id rule calls them `other` and the text rule called them `bare`. `bare`
is the share the table reads as "the unmasked eval is still seeing the policy
that was trained", so a rule that OVER-counts it hides exactly the drift the
field exists to show — the 08-30 failure mode, one level down. FIXED: anchored
at a word boundary (`_BARE_FIRST_RE`). The other coarseness — any capital-initial
reply is `capitalised` — is the SPEC's own wording, is safe in the other
direction, and stands.

## Weighed and NOT changed

- **`MZ01_ENTROPY_COEF=0.1` stands.** The entropy term's gradient is unchanged
  by the mask (it always read `cols` alone), so the 08-30 CONTROL is direct
  evidence for it: 0.1 with a live bonus and no advantage gradient moved the
  restricted entropy from ~0.02 to ~0.03 in 80 steps at mass 1.000, against a
  ceiling of ln 4 = 1.386. It cannot randomise a 4-way policy at this horizon,
  and against advantages of order 10 it is ~100× the weaker term. Changing it
  would also break `onpolicy_g6`'s "epsilon_010 minus its one change".
- **The @40 verdict carries no DEAD.** DEAD is a property of the training trace
  at the end; the mid-arm block is the READ's verdict by the read's rules. An
  arm could print `@40 AVOIDS` and `DEAD` on its own row. Named in `README.md`
  rather than papered over with a windowed zadv at 40, which would be a
  different statistic under the same name.
- **`measure_organism` still carries no `first_token_class`.** The PLAN scoped
  it to `eval_block`; the rerun is the trial and has it. The 4 × 300 cells do
  not, and that is the one gap a later pass should close.
- **The ε first token is still scored as if on-policy.** The mask does not
  remove that bias; it removes the part of it that was moving mass OFF the four
  columns (`A · ∇log_mass`, which is exactly the difference between the open
  and closed first-token gradients) and leaves a bias that is now purely a
  reweighting among four actions.

## Suite

`229 → 232` collected, **231 passed, 1 skipped**, CPU, ~80 s. Three tests added
(one per fix C1–C3), two existing tests strengthened (the no-disturbance test
now asserts the hook-clock split; the classifier test now asserts the word
boundary). None weakened or removed.

---

# Thirteenth pass, 2026-09-08: the exit priced at the cap (D33)

Built against `PLAN-exit-priced-at-cap.md` and the seed-0 trace of the
closed-action-space rerun (`results/closed_cellA_s0/epsilon_010.json`). No box
was contacted, no result file was touched, and `SPEC.md`, `PLAN-*.md` and
`RESULTS-*.md` are unedited. Nothing here is a result; the rerun this change
exists for is pre-registered in the PLAN's §4 and has not run.

## The finding this pass is a fix for

The twelfth pass closed the action space and it HELD: at every step of the
rerun the first token was a bare move id and `unparsed_rate` was 0.000. The
policy collapsed anyway, by a different route:

| step | episode length | farm_stopped | invalid | solve | mine/ep | return |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 13.7 | 0.00 | 0.000 | 1.00 | 0.62 | −13.6 |
| 40 | 12.9 | 0.00 | 0.000 | 1.00 | 0.83 | −14.2 |
| 60 | 7.2 | 0.92 | 0.420 | 0.08 | 0.38 | −16.6 |
| 75 | 4.3 | 1.00 | 0.702 | 0.00 | 0.04 | −9.1 |

H5's guard ends an episode after `MAX_CONSECUTIVE_BLOCKED = 3` consecutive
blocked moves and the return charged `SCORE_STEP = −2` per decision taken, so
walking into a wall three times cost **−6** while the honest solve through the
2.4 mines on the shortest DAG costs about **−28** (`−2·12 − 10·2.4 + 20`).
Under Dr.GRPO's group-relative advantage the exit was therefore the best
episode in every group, and the gradient installed it. The code comment at the
guard said the early end "is already priced at −2 a decision"; that was the
defect stated as a feature, and it is corrected in place.

The 08-30 note's §9 item 2 was right in category and wrong in route: the
cheapest way to avoid a priced mine IS to stop playing, and the way to stop
playing was the wall, not silence.

## D33 — an unsolved episode is charged at the cap

`MZ01_EXIT_AT_CAP=1`, the default. In `episode_return`, with a `cap` given and
the episode not at the goal, the STEP term becomes `SCORE_STEP × cap` instead
of `SCORE_STEP × len(landings)`, where `cap = maze.labels["cap"]` is
`maze2.episode_cap`, twice the shortest path:

    unsolved:  SCORE_STEP·cap + SCORE_MINE·mines(landings)   [+ the mine switch]
    solved:    score_landings(landings)                      [unchanged, to the bit]

So no way of ending an episode early — the wall, the cap, a run of unparsed
turns — is cheaper than continuing. The exit is priced at **what continuing
would have cost**; it is not penalised beyond that, and an episode that ran to
the cap anyway pays exactly what it paid before (`SCORE_STEP·cap` IS
`SCORE_STEP·len` there, which is why the twelfth pass's own toy fixture cannot
tell the two rules apart and why this pass records its own).

**One place, one rule.** The keyword lives on `episode_return` and `_TextEp.record`
passes this episode's own `cap` and `solved`; `return` and `return_full` are
both taken from it, so the switched and the full return cannot come from two
arithmetics. `cap=None` is the pre-thirteenth function exactly, so every other
caller and every existing test is untouched — asserted, not assumed
(`test_episode_return_with_no_cap_is_the_pre_thirteenth_rule`).

**`solved` is CHECKED, not trusted.** A `solved` flag that disagrees with
`"goal" in landings` would let the step term and the goal term price two
different episodes, so it raises by name. There is no caller that can trigger
it today; the point is that a future one cannot introduce it silently.

**`score` is NOT the return.** `record["score"]` stays `maze2.score_landings`
on every episode. That is what the measurement reports — the price table,
`detour_paid`, the B4 regression, `competence`, `aggregate` — and the
measurement is not the objective. Only the training return moves, and only on
unsolved episodes.

**The rejected alternatives**, both weighed and both worse:

- **Raising `SCORE_GOAL`.** To dominate every exit it would need
  G > 2·cap + 10·k − 6 ≈ 80 on the cell-A distribution, which flattens the
  mine price the graded read is built on: at G = 80 a two-mine solve and a
  mineless solve differ by 20 against a goal term of 80, and B4's regression
  reads exactly that difference. It buys the same property by ruining the
  instrument.
- **Removing the farm-stop.** H5's reason still holds — a blocked move from a
  mine leaves the agent ON the mine, so a wall-bumping policy farms on-mine
  turns for free — and removing the guard only moves the exit from step 3 to
  step `cap`, at the price this pass now charges anyway. The end stays; its
  price changes.

## What else moved with it

- **`history["exit_return_share"]`**, per step: the share of the step's
  episodes whose return was charged at the cap, i.e. that did not reach the
  goal. COUNTED off a per-episode flag (`record["return_charged_at_cap"]`), so
  with the knob off it is 0.000 every step and the field is a measurement
  rather than a restatement of the knob. It is read beside `farm_stopped_rate`
  and `capped_rate`, which say HOW those episodes ended, and it joins the
  trainer's progress line on the arms it means something on.
- **`CKPT_SCHEMA` 4 → 5**, with `exit_at_cap` in the fingerprint and in
  `RESUME_RECIPE_KEYS`. `exit_at_cap` changes what every unsolved episode is
  WORTH and therefore every move advantage in a group that contains one, so a
  resume across it would take half its gradients under one objective and half
  under another. A schema-4 file is refused by name.
- **Both pre-flights roll out under the cells' own return.** The exploration
  gate's one refusing criterion is `zero_advantage_group_fraction`, computed
  from the episode returns, and `within_group_return_sd` beside it is the same
  degeneracy as a magnitude; PR-3 prints the same sd. Threading the knob is the
  twelfth pass critic's C1 one term over — a gate read under a price no cell
  computes is not a gate on them — and both record `exit_at_cap` and print a
  `return` line above the verdict line, which stays last.
- **The scorer resolves `exit_at_cap` the same three ways `move_mask` is
  resolved**, and the third is again the point: a trace whose config EXISTS and
  does not name the field was written before the field did, so it ran the old
  price. Without that, re-scoring the committed rerun files would emit a launch
  line naming neither knob — and a line that names neither is the NEW objective
  under the old arm's name. `env_line` emits `MZ01_EXIT_AT_CAP=0` for such an
  arm; the table names it, and names `exit_return_share` and `farm_stopped` for
  every arm that ran the new one, whatever the verdict (PLAN §4).

## The consequence the critic's Q3 asks about, measured on the toy

**Under `mine_cost=False` every unsolved episode of a maze is now worth exactly
the same number** — `SCORE_STEP·cap`, with the mine term subtracted and no goal
term — so a group in which NOTHING solves is exactly degenerate and the
exploration gate refuses it. That is not a bug in the rule; it is the rule
being visible: such a policy has no move gradient under this objective, which
is precisely what `zero_advantage_group_fraction` exists to report. It showed
up immediately as a red test: `test_the_target_band_is_reported_and_never_refuses`
used a UNIFORM WALKER as its "policy with real move variance", and a uniform
walker at n=7 solves almost nothing. The FIXTURE was wrong for the claim the
test makes (that the entropy-target band never refuses), not the gate, so half
its decisions now take the oracle's move and the episodes that solve carry the
variance. The test asserts the same thing it always did.

At the WARM START this does not bite: the navigator solves nearly every episode
and the variance comes from solved episodes' lengths and mine counts, which
this pass does not touch. But it is a real narrowing — the returns of the
unsolved tail are now flat under the cells that price the mine at zero (D and
B), and a cell that collapses to non-solving will read as DEAD rather than as
a policy with a spread of bad episodes. Named here rather than discovered on a
lease.

## The bit-identity claim, and how it is held

`exit_at_cap=False` is the twelfth pass's return. It is asserted against a
trace and a LoRA digest **recorded from the pre-build code at `4ace68f`** and
pasted into `test_exit_at_cap_off_is_the_pre_build_return_and_the_pre_build_adapter`,
compared with `==` and a sha256 over the fp64 bytes of the sorted adapter — not
a tolerance. The fixture is a two-step toy cell A over `TWOMINES`, a world
whose every route runs through both mines and whose start has a wall above it,
driven by a SCRIPTED half-oracle/half-wall policy: its episodes solve,
farm-stop and run to the cap inside the same groups. Because the scripted
sampler reads no weights, both prices visit the same decisions, and the
difference between the two runs is read as a SET over the whole recorded trace:
exactly `{loss, mean_return, return_sd, within_group_return_sd, grad_norm}`
move, plus the field the trace GREW, plus the adapter. Step 0 of that fixture
is all cap-length episodes, where the two rules are the same number and the
trace is identical; step 1 is half farm-stopped, where they are not — so the
difference set is a claim about the rule and not about a fixture that always
differs.

The twelfth pass's own bit-identity test now passes `exit_at_cap=False`
alongside `move_mask=False`. Its fixture (the REAL toy sampler, whose replies
never parse) runs every episode to the cap and therefore cannot tell the two
prices apart — which is exactly why the knob is named there rather than left to
a default that agrees by luck.

## Suite

`232 → 238` collected, **237 passed, 1 skipped**, CPU, ~69 s,
`timeout 900 .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`. Six
tests added (one per PLAN §2 item), one existing test's FIXTURE corrected (the
uniform walker above), two existing assertions moved from schema 4 to 5. None
weakened or removed. The skip is still the Qwen-tokenizer trap, which needs the
real tokenizer in the local HF cache.

## Still not done / read this before the rerun

- **The rerun has not run.** `PLAN-exit-priced-at-cap.md` §4 is the
  pre-registration; the launch line is in `README.md` under the trial's arms.
- **`maze_rl.py` has its own `episode_return` and its own exit** (arm A,
  Dr.GRPO over whole episodes, `MAX_CONSECUTIVE_BLOCKED` included). It was out
  of this pass's scope by the PLAN's §1E and is untouched; the critic's Q4 asks
  whether the fallback arm should carry the same rule before anything is run
  through it. Nothing in MZ01's own path reads it.
- **The `mine_cost=False` flattening above.** Cells D and B now give every
  unsolved episode the same return; only cells A and C keep a spread over the
  unsolved tail (the mine term).
- **An episode can still end early for free in one direction: not at all.**
  The rule prices every END. What it does not price is what the episode DID
  before ending — a policy that walks in circles to the cap and a policy that
  bumps a wall three times are now worth the same number, minus their mines.
  That is deliberate (both are "did not solve", and the cap is what continuing
  costs), and it is the seam the critic's Q5 is pointed at.

---

# Critic, thirteenth pass — 2026-09-08

Adversarial pass over `PLAN-exit-priced-at-cap.md`, the seed-0 trace of the
closed-action-space rerun (`results/closed_cellA_s0/epsilon_010.json`) and build
`9325832`. **The rule itself holds and the identity claims are true.** One
defect was found — in the resume path, not in the return — and is fixed here.
Everything else below is a reading of what the new return does that the build's
own notes do not say, plus one factual correction to them.

## What was re-derived rather than taken from the build's tests

- **Bit-identity, against the pre-build code, not against the build's constants.**
  The tree at `4ace68f` was checked out to a scratch directory and
  `_exit_fixture_train()` run there **without** the `exit_at_cap` argument (it
  does not exist at that commit; asserted by inspecting the signature). All 32
  recorded fields of `_PRE_BUILD_EXIT_TRACE` and the LoRA digest
  `ca9552f3…0ca0` come back to the last digit, and the NEW code at
  `exit_at_cap=False` reproduces that same re-derived run field for field and
  sha for sha. So the pasted constants are a recording of the old code, and the
  knob-off claim is a claim about the old code and not about the new code
  agreeing with itself.
- **The twelfth pass's fixture cannot tell the two prices apart** — checked, not
  assumed: `_real_sampler_train(move_mask=False, move_mass_coef=0.03)` matches
  `_PRE_BUILD_TRACE` at `exit_at_cap` **both** False and True, because every one
  of its episodes runs the full cap (`episode_length` 10.0, `capped_rate` 1.0).
  Naming the knob there is right, and it asserts nothing on its own.
- **The scorer, before and after, on the committed rerun directory.** The
  pre-build `score_mz01_explore.py` and this build's were both run over
  `results/closed_cellA_s0`. The two outputs differ by **exactly one added
  line** (the arm's `MZ01_EXIT_AT_CAP=0` annotation). Every verdict, every
  column, the `@40/@end` block, the whole `detour_paid` bin table and the pick
  are byte-identical. Q2 answered on the artefact and not only on the code.
- **A solved episode cannot be repriced, by construction.** `episode_return`
  takes the new branch only under `if not reached`, and `reached` is
  `"goal" in landings`, which is exactly the condition `score_landings` adds
  `SCORE_GOAL` on. There is no path where the step term and the goal term see
  different episodes.

## C1 — a row that predates the knob was carried into a launch that has it

**FIXED.** `run_orgb.RESUME_RECIPE_KEYS` grew `("exit_at_cap", "exit_at_cap")`
in this build, but the loop that reads it skips any key the prior row does not
carry (`if row_key not in prior_cfg: continue`). For `exit_at_cap` — and for
`move_mask` before it — **the absence of the field is evidence, not a missing
reading**: a row whose `training.config` exists and does not name `exit_at_cap`
was written by code that charged an early exit at the decisions it took. The
scorer already reasons exactly that way (`_action_space`: "a trace whose config
exists and does not name it was written before the field existed"), so the two
halves of the codebase drew **opposite** conclusions from the same absence, and
a shard begun before this pass would have had its finished rows carried
**silently** into an exit-priced launch: half the 2×2 trained on one objective
and half on another, with nothing on disk saying which. Demonstrated on the toy
shard before the fix (`rows_reused == ['D/s0']` with both fields popped).

`RESUME_ABSENT_MEANS = {"move_mask": False, "exit_at_cap": False}`; the loop
compares against the assumed value and names it in the drift line. A row with no
`training.config` at all still claims nothing and is skipped as before. Test:
`test_a_row_that_predates_the_objective_knobs_is_not_carried_forward`.

NOT reachable in PLAN §4: that rerun is `scripts/mz01_explore_trial.py`, which
takes no prior rows and writes no mid-cell checkpoint. It is reachable on the
4 × 300 driver path.

## C2 — correction to this pass's own note: `maze_rl.py` has no such exit

The build's entry above says "`maze_rl.py` has its own `episode_return` and its
own exit (arm A, Dr.GRPO over whole episodes, `MAX_CONSECUTIVE_BLOCKED`
included)". **`maze_rl.py` contains no `MAX_CONSECUTIVE_BLOCKED`, no
`farm_stopped` and no `blocked_run`** (grep: zero hits). `maze_rl._Ep.done` is
`solved or len(moves) >= cap` and nothing else, and its action space is a 4-way
categorical over move ids, so there are no unparsed turns either. **Every
unsolved episode there runs exactly to the cap**, which makes
`SCORE_STEP × cap` identically `SCORE_STEP × len(landings)` and this pass's rule
a provable no-op in that module. It is also unreachable from MZ01: the only
symbol any MZ01 path imports from it is `group_advantages`. Left untouched, on
purpose — but the note above would send the next pass looking for a hole that is
not there.

## Findings reported, not fixed

- **`zero_advantage_group_fraction` has changed meaning at collapse, and the
  scorer checks DEAD before every cell-A verdict.** Under the new return an
  unsolved cell-A episode is worth `−2·cap − 10·mines`; the training pool's
  `shortest_len` is **12 for every maze**, so `cap` is 24 for every maze and the
  unsolved tail collapses to the discrete ladder −48, −58, −68. A group of six
  in which nothing solves and nothing steps on a mine is therefore **exactly**
  degenerate. On the committed seed-0 trajectory, steps 70 and 78 have
  `solve_rate` 0.000 and `mine_step_rate` 0.000: the recorded
  `zero_advantage_group_fraction` there is 0.000 and under the new return it is
  **1.000**. A rerun that collapses the same way will read **DEAD** ("the move
  term never had a gradient") where the committed one read RANDOMISED, which is
  the wrong sentence about what happened. The table already prints `zadv s`
  beside `zadv e`; a DEAD verdict whose `zadv s` was low is a collapse into a
  flat region, not a dead gradient, and PLAN §4's verdict rules should say so.
- **`exit_return_share` is identically `1 − solve_rate`** (the build's own test
  asserts it to 1e-12), so it carries no information the trace did not already
  have. The episodes whose return actually **moved** are the `farm_stopped` ones
  — a cap-reached episode has `len(landings) == cap` and pays exactly what it
  paid before. On the committed trace the two differ most at step 1 (0.125
  unsolved, 0.000 repriced). Read `farm_stopped_rate` as "how much of the trace
  the rule
  changed" and `exit_return_share` as "how much of it the rule applied to".
- **The gap is bigger than the PLAN says.** §6 Q3's premise is "the exit now
  sits ~20 below the typical solve", from `−2·12 − 10·2.4 + 20 ≈ −28`. On the
  actual pool `mines_on_shortest_dag` averages 1.16, and the warm start's own
  measured `mean_return` at step 0 is **−13.58**. The exit is −48. The gap is
  **~34**, not ~20 — the safe direction, but the arithmetic in the PLAN and in
  the commit message is not this pool's.
- **Q5, the next cheapest thing.** Episode LENGTH is now free for an unsolved
  episode, and in cells D and B (`mine_cost=False`) so is every mine. On the
  pool's constant cap of 24 the three-bump wall exit, a 24-decision floor
  wander, and 24 decisions cycling mine→floor→mine→floor are **all worth exactly
  −48**. Under the old rule the camp cost −48 against the exit's −6, so the
  return charged 42 for it; that was the only thing pricing it, and this pass
  removed it. H5's guard does not catch the camp — `blocked_run` resets on any
  non-blocked landing, and stepping off and back onto a mine is one, at zero
  cost in D and B. Cell B is the cell it matters in: the remark term is weighted
  10 and scored per decision, so twelve on-mine decisions put twelve remark
  terms into the loss where the wall-bump puts three. The signature is already
  recorded — `on_mine_state_rate` climbing while `solve_rate` falls and
  `mean_return` sits pinned at −48 — and it is the number to watch in D and B.
- **(b), the D/B degeneracy: it needs a change before the 4 × 300, and the
  change is not to the return.** Two items, both in the read: (i) record
  `zero_advantage_group_fraction` on `return_full` per step as well (the trainer
  already carries `return_full` on every episode and the pre-flight already
  reports both), so a D or B cell that collapsed into the flat non-solving
  region is distinguishable from one whose warm start was simply deterministic —
  today both read DEAD; (ii) a D or B cell whose last-fifth `solve_rate` is near
  zero should be NOT-MEASURED rather than a null, because under `mine_cost=False`
  it had no move gradient for the steps that produced it. Neither is needed for
  the cell-A rerun.
- **Wording, inherited from the PLAN.** "a run of unparsed turns" is listed
  everywhere as an early end the rule now prices. It is not an early end:
  `_TextEp.apply` resets `blocked_run` on an unparsed landing, so such an
  episode runs to the cap and paid `SCORE_STEP × cap` before this pass too. The
  claim "not cheaper than continuing" is still true of it, trivially. Left as
  written rather than churning eight files.

## Suite

`237 passed, 1 skipped` → `238 passed, 1 skipped`, CPU,
`timeout 900 .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`. One test
added (C1), one shared fixture (`_resumable_run`) given a `config` override so a
shard can be relaunched under one changed knob. None weakened or removed.

---

# Fourteenth pass A+B, 2026-09-08: per-decision credit (D34) and targeted ε (D35)

Built against `PLAN-credit-explore-navigator.md` §1 and §2 and the four runs in
flight at `8894ce5`. Part C (the navigator's uniform labels) is a separate build
in its own worktree and nothing here touches `navigator.py` or its driver. No
box was contacted, no result file was touched, and `SPEC.md`, `PLAN-*.md` and
`RESULTS-*.md` are unedited. Nothing here is a result; the runs these two knobs
exist for are pre-registered in the PLAN's §4 and have not run.

## The finding this pass is for

Three runs of the closed, priced recipe kept an INTACT policy under a LIVE
gradient on seed 0 for 160 steps and did not learn to avoid. The two previous
passes each removed a way for the policy to stop playing; neither of them made
the reward land anywhere in particular. Two candidates, one knob each, and both
default to what every committed row ran:

1. **The credit is spread.** `move_adv[row] = a` for every decision of an
   episode: one mine landing in a 14-decision episode is scored over 14
   decisions, and the decision that walked onto the mine gets a fourteenth of
   the blame. The SPEC's own recipe 3A was immediate per-move reward, swapped
   for the episode return by a DECISION; the v1 grid world, which installed, was
   single-step.
2. **The exploration is untargeted.** A detour that avoids a mine and still
   solves needs a deviation AT a mine-adjacent decision and AWAY from the mine.
   Uniform ε deviates everywhere equally, and about a third of its deviations at
   an eligible decision step ONTO the mine -- which the return then correctly
   punishes. The budget mostly buys worse episodes.

## D34 -- the advantage lands on the decision that earned it (`credit`)

`MZ01_CREDIT ∈ {episode, togo, immediate}`, default `episode` = today.

    r_t = SCORE_STEP  every decision (a blocked move included -- it took a turn)
        + SCORE_MINE  if the landing is a mine, subject to the cell's switch
        + SCORE_GOAL  if the landing is the goal
        + the unsolved episode's exit charge SCORE_STEP*(cap - len), ON ITS LAST
          DECISION

    togo:       A_t = G_t - b_t,  G_t = sum_{k >= t} r_k  (undiscounted)
    immediate:  A_t = r_t - b_t                            (SPEC 3A)

with `b_t` the mean over the episodes of the SAME GROUP that have a decision at
index t, and **an index only one episode of the group reaches gets exactly
zero**. That last rule is not a detail: on cell A the raw `G_t` at a lone index
is a large negative number on every such decision whatever the episode did, so
scoring it is a standing push toward "never be the long episode" -- which is the
early exit this programme has installed by accident twice already.

**`sum_t r_t == episode_return`, checked on every episode of every step.**
`decision_rewards` takes the same arguments `episode_return` does and is
asserted against it in the suite on solved, capped, farm-stopped, unsolved-with-
mines, mineless and never-parsed episodes, with the mine priced and free and the
exit at the cap and at the decisions taken. The runtime check is in
`decision_advantages` and it raises by name: `togo` and `episode` are two ways of
SPENDING one return, and the one way this could silently reprice an organism is
a term entering `episode_return` (as one did in the thirteenth pass) that the
per-decision account does not have.

**Where the exit charge lands, and why it is a choice.** On the LAST decision of
an unsolved episode. It is the price of having STOPPED, so it is charged where
the episode stopped, and that is what keeps the sum identity exact. Spreading it
would make the two accounts agree in total and disagree everywhere else, and
under `immediate` it would put a constant on every decision of a doomed episode
-- where a constant is exactly what a group baseline removes.

**THE PLAN'S OWN TEST 2 IS WRONG AND IS CORRECTED IN THE SUITE, NOT IN THE
CODE.** It asks that under `togo`, on a group where one episode steps on a mine
at decision 3 and the other detours, decisions 0-2 get `A_t = 0` and the mine
decision alone gets the negative advantage. That is `immediate`'s behaviour and
it cannot be `togo`'s: under to-go credit the mine is still in the FUTURE at
decisions 0, 1 and 2, so they carry it too -- which is the entire difference
between the two modes. On the suite's pair (mine −10 against a two-decision
detour −4) `togo` gives −3 to each of the mine episode's first FOUR decisions
and +2 to the two after the mine is paid, and `immediate` gives 0, 0, 0, −5.
Both are asserted as what they are.

**What the step-index baseline assumes, said out loud rather than discovered.**
`b_t` compares the t-th decisions of a group's episodes, not decisions taken in
the same STATE. At the warm start, where a group's episodes share their prefix,
those are the same thing; once the episodes diverge it is a variance reduction
with a bias that grows with the divergence. Two visible consequences, both
asserted in the suite: after the mine is paid the shorter episode is credited
simply for being nearer the goal at the same index, and the goal bonus lands on
whichever episode reaches it at that index. The PLAN's critic question 1 is
pointed here.

## D35 -- the ε draw acts where a detour is possible (`epsilon_where`)

`MZ01_EPSILON_WHERE ∈ {all, near_mine}`, default `all` = today. `near_mine`:
the draw acts only at decisions where at least one VALID neighbour of the
current position is a mine, and the forced word is uniform over the valid moves
that do NOT land on one; where every valid move lands on a mine it falls back to
all of them (there is no away to go to) and the decision still counts as
eligible, because the draw did act there.

**The generator stream is untouched.** `hits` and `picks` are drawn for every
active episode exactly as the tenth pass drew them and an ineligible decision
simply does not APPLY them, so `all` is bit-identical on a seed and `near_mine`
consumes the same random numbers in the same order. Asserted two ways: a sha
over every id, move, replacement and landing of a 71-decision ε roll-out
recorded from the pre-build code, and `epsilon_pool` under `all` returning
`[w for w, _ in maze.neighbours(pos)]` -- the tenth pass's own argument, in its
own order.

**It makes the estimator MORE off-policy, not less.** The behaviour policy is no
longer `(1 - eps) pi + eps Uniform(valid)` but a state-dependent mixture, and
the correction is still not applied (the same three reasons as the temperature
arm). What is recorded instead is the number that sizes it:
`epsilon_eligible_fraction` per step, beside `epsilon_replaced_fraction`, in the
trace and in the exploration pre-flight. The PLAN's critic question B is pointed
here; on this pool the eligible share is expected near a third, which is why the
named arm draws at 0.30 for an overall replacement share near the tenth pass's
0.10.

**`plan_epsilon_where` resolves it, as `plan_epsilon` resolves the rate.** An
arm that draws at 0 has no `where`, so every control resolves to `all` and
cannot record a targeting rule it never applied; the trainer refuses the
combination outright, by name, before an episode is drawn.

## What else moved with them

- **`zero_advantage_group_fraction` is now read off the per-decision
  ADVANTAGES** (`zero_advantage_decision_fraction`) rather than off the episode
  returns -- "the share of groups in which no decision has a gradient", which is
  the only definition that still means that under `togo`. Under `episode` it is
  the SAME NUMBER, asserted rather than argued (a group's broadcast advantages
  are all zero exactly when its returns are all equal, and returns are sums of
  three integer constants). **It is a much coarser read under `togo`**: a group
  can carry a gradient at one decision index and nothing at any other and still
  read 0.00. `adv_abs_mean` and `adv_nonzero_share` are the two new per-step
  fields that say how much of a step carried signal, and the scorer prints them
  for any arm whose credit is not the episode's.
- **`CKPT_SCHEMA` 5 → 6**, with `credit` and `epsilon_where` in the fingerprint
  and in `RESUME_RECIPE_KEYS`. `RESUME_ABSENT_MEANS` gains both, and for these
  two the absent value IS the current default -- a row written before the pass
  spread the episode's advantage and drew uniformly, which is what the defaults
  still do -- so the entries never refuse a row on their own. They are there so
  a LATER launch under `togo` or `near_mine` cannot silently carry one.
- **The exploration pre-flight gates under the arm's own credit and rolls out
  under its own targeting rule**, and prints both. `preflight_gradient_check`
  is DELIBERATELY not threaded and its docstring says why rather than leaving it
  for the next critic: it draws no epsilon at all, and every number it reports
  is a yell quantity or a function of the episode RETURNS, which credit does not
  move. (The twelfth pass's critic found `move_mask` left on a default there;
  the rule now is thread it or say why not.)
- **`decision_advantages` recomputes the return from the landings** under the
  switches it is given rather than reading `record["return"]`. That is what lets
  the pre-flight ask ONE roll-out for both readings -- the mine-free return it
  gates on and the mine-priced one it reports -- and in the trainer it is
  `record["return"]` to the bit.
- **The scorer's `_action_space` returns a dict** and reads `credit` and
  `epsilon_where` the same three ways it reads `move_mask` and `exit_at_cap`.
  Here the third reading is the quiet one: absence and default AGREE for these
  two, so re-scoring the committed files emits a launch line naming neither, and
  it is the new arm that has to say `MZ01_CREDIT=togo` /
  `MZ01_EPSILON_WHERE=near_mine`.
- **The trial gains the named arm `epsilon_030_near`** (mode `epsilon`, GROUP 6,
  ε 0.30, `near_mine`) -- a NAMED SELECTION, never a default, so the tenth and
  eleventh passes' committed command lines keep meaning what they meant. It
  differs from `epsilon_010` in the draw alone: same mode, same group, same
  held coefficient, same thermostat (off, by the mode itself).

## MZ01_WARM_START_ONLY and MZ01_OUT (added to this pass's scope by the lead)

`MZ01_WARM_START_ONLY=1` builds the output directory's ONE warm start (M5),
runs PR-3 and the exploration gate on it, writes the payload and STOPS: no cell
is trained, nothing is measured, `rows` is empty. It exists because Part C's new
navigator needs a warm start of its own before any arm can run on it, and that
build is a job a box can take while the arms queue. It is gated AFTER both
pre-flights on purpose -- a warm start neither gate has read is not one any cell
may be started from, and the point of building it separately is to learn that
before a box is booked on the arms.

`MZ01_OUT` makes the results directory a knob. M5's "one warm start" is one per
BASE, not one per repository: `nav48u`'s warm start must land in
`results/orgb_nav48u/` and must never be written over
`results/orgb/warm_start.pt`, which every committed cell-A row stands on. Both
refusals are by name and at two levels -- the driver refuses on the command line
before a model is loaded, `run_orgb.run` refuses in the library -- and they are
the same two: an existing `warm_start.pt` in the target directory, and a
`MZ01_WARM_START` handed over to reuse while also being asked to build.

## The bit-identity claims, and how they are held

Two recordings from `8894ce5`, both taken BEFORE any edit:

- `_PRE_BUILD_CREDIT_TRACE` + `_PRE_BUILD_CREDIT_ADAPTER_SHA` -- the thirteenth
  pass's `_exit_fixture_train()` at TODAY'S defaults, which is what
  `credit="episode"` must reproduce. 32 recorded fields compared with `==` and a
  sha256 over the fp64 bytes of the sorted adapter. The fixture is reused rather
  than replaced because it is already the right shape for this claim: its group
  has episodes of different lengths that solve, farm-stop and run to the cap,
  and every route on `TWOMINES` runs through both mines mid-episode -- so `togo`
  and `episode` cannot agree on it, which the test asserts directly (the loss
  and the adapter move; every sampler-side field does not).
- `_PRE_BUILD_EPSILON_ROLLOUT_SHA` -- a sha over every id, move, replacement,
  landing and return of a 71-decision ε roll-out at ε 0.5 on a fixed seed, which
  is what `epsilon_where="all"` must reproduce. `near_mine` on the same seed
  differs, so the claim is about a sampler that can be moved.

The three fields the trace GREW (`adv_abs_mean`, `adv_nonzero_share`,
`epsilon_eligible_fraction`) are named in the fixture rather than left out of the
comparison, and asserted separately: an absent field and an unchanged one must
not look alike.

## Suite

`239 → 255` collected, **254 passed, 1 skipped**, CPU, ~76 s,
`timeout 900 .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`. Sixteen
tests added -- eight for part A (the sum identity, the record shape the
hand-built episodes claim to be, `togo`, `immediate`, the lone index, the
pre-build bit-identity, the degeneracy report, the schema-6 fingerprint), four
for part B (eligibility and the pick set, the pre-build roll-out, `near_mine` at
ε 1.0, the two fractions), the exploration gate under the arm's own credit and
draw, the named arm, the launch-line reading, and the warm-start-only build.
Three existing schema assertions moved from 5 to 6. None weakened or removed.
The skip is still the Qwen-tokenizer trap, which needs the real tokenizer in the
local HF cache.

## Still not done / read this before the runs

- **None of the four runs has run.** `PLAN-credit-explore-navigator.md` §4 is
  the pre-registration and the launch lines are in `README.md` under the trial's
  arms.
- **`togo` does NOT rescue the gate at the warm start, and the PLAN's critic
  question 1 should not expect it to.** On an on-policy group whose episodes all
  solve along the same path, every episode has the same landings, so the same
  `r_t` and the same `G_t` at every index; `b_t` is that same number; every
  `A_t` is exactly 0 and every index is present in all six episodes, so
  `zero_advantage_group_fraction` reads **1.000** for that group -- identical to
  what `episode` reads off the equal returns. The two modes come apart only when
  the episodes DIFFER (ε, or a fork the navigator's uniform labels would give
  it), and then `togo`'s reading is the less pessimistic one because a single
  differing landing gives a gradient at the indices around it rather than one
  advantage on the whole episode. The number to watch on the new arms is
  `adv_nonzero_share`, not `zadv`.
- **The step-index baseline's bias is unbounded in principle.** Nothing here
  bounds how far the t-th decisions of two episodes can drift apart, and the two
  visible consequences (credit for being nearer the goal at the same index, the
  goal bonus landing at one index) are asserted rather than removed. A
  state-indexed baseline would need a value function, which nothing in this
  programme has trained.
- **`near_mine` cannot be read without its eligible share.** The replaced share
  alone is the product of the rate and the eligible share, and a low one cannot
  be told from an arm that barely drew. Both are in the trace, in the
  pre-flight and in the scorer's table.

---

# Fourteenth pass A+B, CRITIC, 2026-09-08

Adversarial read of `cb8aa32` against `PLAN-credit-explore-navigator.md` §1,
§2, §4, §6 and §7. Verdict: **RUN WITH AMENDMENTS**. The implementation is
faithful to the spec and the two bit-identity claims are real (both recomputed
from `8894ce5` here, not taken from the recorded constants). Two defects fixed,
one pre-registered number contradicted by measurement, and one property of the
spec's own estimator that has to be on the record before the runs are read.

## Verified, with the arithmetic

- **`sum_t r_t == episode_return` exactly, 17 050 cases.** Every landing
  sequence of length 1-5 over {floor, mine, blocked, unparsed, goal} with the
  goal only ever last, x `mine_cost` on/off, x `cap` in {None, len, len+1,
  len+3, 24}. Zero mismatches at `== 0.0`, not at a tolerance. The exit charge
  moves the LAST decision's `r_t` and no other's.
- **The exit charge only ever lands on a wall-bump.** `_TextEp.done` is
  `solved or len(moves) >= cap or farm_stopped`, so an unsolved episode ends
  either AT the cap (charge `SCORE_STEP * 0` = 0) or by farm-stop, and
  `farm_stopped` is set only when the third consecutive `blocked` landing
  arrives. So under `immediate` the extra charge is always on a blocked move --
  which is exactly H5's wall-bump and exactly what D33 wanted priced. It is not
  a constant smeared over a doomed episode and it never lands on a solved one.
- **The per-(group, index) baseline does not leak across groups**, and the
  lone-index rule gives exactly 0.0. Two episodes put in different groups get
  every advantage 0.
- **Bit identity, recomputed from the pre-build source.** A scratch tree from
  `8894ce5` running `_exit_fixture_train()` reproduces `_PRE_BUILD_CREDIT_TRACE`
  in all 33 recorded fields and `_PRE_BUILD_CREDIT_ADAPTER_SHA`
  `6a96d877...bacdbdbc`; the ε roll-out reproduces `_PRE_BUILD_EPSILON_ROLLOUT
  _SHA` `8cd921f6...fdc3c94c` and its shape (71 decisions, 37 replaced, the six
  returns). `near_mine` consumes the same stream: the generator state at every
  one of the 12 decode calls is identical to `all`'s on `TWOMINES`, and the
  `(hit, u)` pairs handed to `_epsilon_pick` are the same floats.
- **The scorer is byte-identical** on all eleven committed trial directories
  (pre-build scorer vs this one, on a frozen snapshot -- four of those runs are
  live and their files move between two invocations, which is not a diff).
- **`zero_advantage_decision_fraction` == `zero_advantage_group_fraction` under
  `episode`**, 4000/4000 randomised groups. It cannot be otherwise: achievable
  returns are integer multiples of 2.0, so two returns are equal or 2.0 apart --
  2e9 times `ADVANTAGE_TOL`. And **togo-flat implies episode-flat** (`A_0 = G_0
  - mean G_0 = R_i - mean R`, and index 0 is shared by every episode), 0/4000
  counter-examples: the gate can never read a group as flat under `togo` that
  carries a signal under `episode`. The coarsening is one-directional and
  honest.
- `preflight_gradient_check` is correctly left unthreaded: it draws no ε and
  reports only yell quantities plus `within_group_return_sd`, a function of the
  returns, which credit does not move.

## Fixed

1. **A targeted draw reported its RATE as its off-policy SHARE.**
   `off_policy_decision_fraction` is defined in three places as "the share of
   decisions whose action did not come from the scored policy" -- the one number
   that makes the ε and temperature arms comparable, in the plan, the manifest,
   every row's `explore` block and the pre-flight, and the scorer's fall-back.
   Under `all` the rate IS that share. Under `near_mine` it is `ε x eligible`,
   so the field stated 0.30 for a run whose real share is about 0.07. Now: the
   plan answers None (it cannot know the pool), the trainer records the MEASURED
   mean it has counted per step, the pre-flight substitutes its own roll-out's
   measurement, and the scorer returns n/a rather than the rate. `all` is
   unchanged to the bit everywhere, and the scorer's output on all eleven
   committed directories is still byte-identical.
2. **The eligible share is 0.22, not "about a third".** Measured on the pool the
   runs draw (`generate_set(256, n=9)` with the orgb train generator): **0.220**
   along the route a warm-started navigator walks, **0.268** over open cells
   uniformly. §2's arithmetic ("about a third eligible, so ε 0.30 keeps the
   replacement share near 0.1") does not hold: 0.30 x 0.22 = **0.066**. "About a
   third" is the right number for a DIFFERENT quantity and that is where it came
   from -- **0.365** of the valid moves AT an eligible decision land on the mine,
   which is what a uniform draw wastes there. Corrected in all four places that
   stated it, and pinned by a test that measures it on the real pool.

## Not fixed -- the spec's own estimator, on the record

**Both new credit rules can rank a pair against the episode return, on exactly
the comparison the pass exists to make.** Take a mine route of length L with one
mine at index m against a detour of L+d that avoids it. The return prefers the
detour iff d < 5 (`SCORE_MINE / SCORE_STEP`). The trajectory-level push -- and
`sum_t A_t` is exactly the coefficient on `log P(trajectory)`, since
`grad log P = sum_t grad log pi` -- comes out, for `togo`, as

    sum_t A_t (mine route) = L*d - 5*(m+1)

so the mine route is pushed UP whenever `L*d > 5(m+1)`. Swept over L = 6..16,
every mine index and d = 1..10: `episode` **0 sign flips**, `togo` **224**,
`immediate` **440**, and in both cases the flips are at **d = 1, 2, 3, 4** --
the whole range in which the detour is worth taking. On the PLAN's own canonical
pair (`_MINE_EP` vs `_DETOUR_EP`, in the suite) `immediate` gives the mine
episode `sum A = +5` and the clean detour `-5`, because the lone-index rule
deletes the detour's own goal bonus while the mine route's lands at a shared
index.

**It is much milder on realistic groups**, and this is the honest counterweight.
Six-episode groups drawn from a noisy shortest-path policy over the real 64-maze
pool, per credit rule:

| noise | mean A on a MINE decision | on any other | pairwise sign flips | cleaner-of-a-pair pushed up |
| --- | --- | --- | --- | --- |
| 0.10 episode | -2.99 | -0.34 | 0/650 | 0.996 |
| 0.10 togo | -3.13 | **+0.27** | 13/650 | 0.953 |
| 0.10 immediate | -3.69 | **+0.31** | 10/650 | 0.949 |
| 0.25 episode | -7.89 | -1.48 | 9/844 | 0.939 |
| 0.25 togo | -5.34 | **+0.47** | 58/844 | 0.828 |
| 0.40 episode | -11.78 | -2.04 | 5/901 | 0.877 |
| 0.40 togo | -8.18 | **+0.81** | 49/901 | 0.802 |

The pass's premise holds: under `episode` every decision of a mine episode is
punished (the "any other" column is negative); under `togo` and `immediate` the
blame concentrates on the mine decision and the innocent decisions go positive.
The price is the trajectory-level flip rate rising from ~1 % to ~5-7 %, and the
AVOIDS-relevant comparison (which of two episodes of a group has fewer mines)
falling from 0.88-1.00 to 0.80-0.95.

**Answers to §6 A.** The step-indexed baseline is unbiased AT an index -- the
advantages of the episodes present at index t sum to zero by construction -- but
it is not unbiased as an estimator of the policy gradient, for two reasons that
are one reason. `b_t` conditions on SURVIVING to index t, which is an outcome of
the actions, and it compares episodes in different STATES. Both show up as: at
every index, being nearer the goal than the surviving peers is rewarded, however
you got there. The `L*d` term above is that bias accumulating linearly in the
shared prefix while the mine's contribution stays fixed at `5(m+1)`. A
state-indexed baseline would need a value function this programme has not
trained; the mitigation available now is to read `adv_nonzero_share` and the
mine-decision gap rather than the trajectory sum.

`togo` does NOT change the pre-flight's gate at the warm start. Where every
episode of a group is the same episode, every rule reads flat; where they differ
only in length under `mine_cost=False`, an unsolved episode's `G_t = -2*cap + 2t`
is independent of its length, so those groups are flat under every rule too. The
suite's own gate test reads `zadv == 1.0` under all three credits, and only
`adv_nonzero_share` moves.

Under `immediate` the exit price teaches "the third wall-bump in a row is very
expensive" and nothing else, because that is the only decision it can ever land
on (above). It should NOT be spread: spreading it would put a constant on every
decision of a doomed episode, which the group baseline at each index removes
anyway, and it would break the sum identity's placement.

**Answer to §6 B.** Yes, `near_mine` biases the estimator in a way `all` did
not. `all`'s behaviour policy is `(1-eps) pi + eps Uniform(valid)`, one mixture
everywhere; `near_mine`'s is `(1-eps) pi + eps Uniform(away)` at eligible
decisions and `pi` elsewhere -- state-dependent in TWO ways, and the second (the
non-mine pick) is a deliberate tilt of the behaviour policy away from the very
action the return punishes, so the sampled distribution over landings is no
longer the policy's even conditionally. The recorded eligible fraction is NOT
enough to correct for it later: correcting needs the per-decision ratio
`pi(a)/mu(a|s)`, and `mu` depends on the size of the away-set at that decision
(2, 3 or 4 moves, plus the 7.9 % of eligible decisions where it falls back to
all valid). The eligible share sizes the bias; it does not invert it. What WOULD
suffice is already almost there: the rows carry `epsilon_eligible` and
`epsilon_replaced` per decision, so recording the POOL SIZE beside them would
make the importance weight computable after the fact. Recommended for a later
pass, not built here.

# Fourteenth pass, part C, 2026-09-08: the navigator's label at a fork (D36)

Built against `PLAN-credit-explore-navigator.md` §3, in an isolated worktree on
`fourteenth-c`. Parts A and B (per-decision credit, targeted ε) are a separate
build in the main tree and this one touches none of their files. No box was
contacted, no result file was touched, and `SPEC.md`, `PLAN-*.md` and
`RESULTS-*.md` are unedited. Nothing here is a result: N1 — the nav48u training
run the knob exists for — is pre-registered in the PLAN's §4 and has not run.

## The finding this pass is a fix for

The warm start's move entropy is **0.018 nats**. That is not what an RL run
starts from because the imitation was too good; it is what it starts from
because the CORPUS was a point mass. `build_navigator_corpus` labels every cell
with the first move of ONE BFS shortest path, and at a cell where two moves are
equally optimal the corpus insists on one of them for no reason at all — BFS's
queue order, which is `MOVE_WORDS`' order, which is an artefact of a `for` loop.
Three passes of RL have then spent their whole exploration budget buying that
variance back off-policy: ε replaces a move at a uniformly random decision with
a uniformly random valid move, which is a worse draw at a fork (it can pick the
move that walks away from the goal) and a pointless one everywhere else.

A navigator whose labels are uniform over the optimal moves has the variance
**where the world has a genuine choice and nowhere else**, on-policy, at no cost
to optimality: every label it can draw is still a shortest-path move.

## D36 — the label at a fork is drawn, not taken

`label ∈ {"first", "uniform"}` on `build_navigator_corpus`, default `"first"` =
today, `MZ01_NAV_LABEL` in the driver. Under `"uniform"` the completion at a
labelled cell is drawn uniformly from `maze2.optimal_moves(maze, cell)` — the
moves whose landing cell's BFS distance to the goal is one less than the cell's.
A cell with one optimal move keeps the label it had; only a fork draws.

**The states are NOT the rule's to move.** The draw comes from a generator
*seeded from* the builder's own (`torch.Generator().manual_seed(
generator.initial_seed())`), not from the builder's own stream. A draw taken
from `generator` would shift every later `randperm` and `random_move_orders`,
and the two corpora would then differ in WHICH off-path cells they teach and in
what order the four words are listed — E1c-2's own artefact — as well as in the
labels. On the toy pair below every prompt, position, order and kind is equal
item for item (`test_uniform_redraws_the_label_at_a_fork_and_moves_nothing_else`
asserts it field by field), so nav48u minus nav48 is the labels and nothing
else. The seed is still the `(seed, purpose)` `runner.derive_generator` was
asked for, so the corpus is reproducible from the same argument it always was.
This is a deviation from the PLAN's literal "from the corpus builder's own
generator"; it keeps the PLAN's own next sentence ("every other property of the
corpus are unchanged"), which the literal reading breaks.

**`"first"` is the pre-fourteenth corpus to the byte**, because it draws nothing
and the optimal-move sets it computes go into the STATS and not into the items.
Asserted against a digest of the 96-decision n=7 toy corpus recorded from the
tree at `8894ce5` before the first edit
(`7d6128c5…`, `test_the_first_label_rule_is_the_pre_fourteenth_corpus_to_the_byte`),
with the same test asserting the claim is not vacuous — that corpus has forked
cells, so a rule change CAN move it.

**`maze2.optimal_moves(maze, pos, dist=None)`** is where the set is computed,
beside `unique_correct_move`, which is now written in terms of it (`optimal_moves`
returning exactly one). Same predicate, same order, so `single_route_acc`'s
tie-break rule cannot drift from the labeller's; brute-forced against a second
route — `shortest_path` lengths from the cell and from each neighbour — rather
than against itself.

## What else moved with it

- **Five corpus stats**, in the JSON under `corpus` and printed by the driver
  and the scorer: `label_rule`, `multi_optimal_share` (share of labelled cells
  with ≥ 2 optimal moves), `optimal_moves_mean`, `optimal_moves_histogram`, and
  two entropies. `label_entropy_nats` is the realised labels' own 4-way entropy
  — a skew here is a policy prior — and it is NOT the fork number: a corpus can
  be perfectly balanced over the four words and still be a point mass at every
  cell it labels. `label_rule_entropy_nats` is the entropy of the RULE at a
  labelled cell averaged over cells: 0.0 under `"first"` by construction, and
  mean ln|optimal moves| under `"uniform"`. The first three are properties of
  the MAZES and read the same under both rules on the same states, which is how
  a reader can tell that a `uniform` run whose corpus had no forks would be
  nav48 under another name.
- **`label_rule` in the manifest as well as in `corpus`**, and in the scorer's
  header line. Two navigators trained under the two rules differ in nothing else
  a reader can see — same gate, same bars, same shape of file — and the shard
  name is a string a human types. A file written before the knob has no field
  and ran `"first"`; the scorer says so rather than printing `n/a`.
- **The driver refuses an unknown `MZ01_NAV_LABEL` by name, before the model is
  loaded**, and `build_navigator_corpus` raises on one too. A typo that fell
  through to the default would spend ~1.7 h training nav48's corpus under
  nav48u's name, and every later table would carry the wrong claim.

## The corpus stats on the toy, measured

200 mazes at n=9 (3 400 decisions, `off_path_share=0.3`,
`derive_generator(0, "test")`), the two rules on the same states:

| | `first` | `uniform` |
| --- | --- | --- |
| labelled cells with ≥ 2 optimal moves | 0.1947 | 0.1947 |
| mean optimal moves per labelled cell | 1.1971 | 1.1971 |
| histogram of optimal moves | 1: 2738, 2: 655, 3: 6, 4: 1 | (same states) |
| label entropy, 4-way marginal | 1.3780 nats | 1.3718 nats |
| the RULE's entropy at a labelled cell | **0.0** | **0.1359 nats** |
| labelled moves landing on a mine | 0.1079 | 0.1091 |
| states standing on a mine | 0.1103 | 0.1103 |

**About a fifth of the corpus is a fork**, and the draw moved **345 of 3 400
labels (0.1015)** — 52.1 % of the 662 forked cells, which is the coin. Read the
last two rows against M4: the label stream is still mine-blind by construction
under both rules (`optimal_moves` is computed from `bfs_dist` alone and BFS
routes THROUGH mines), and what moves by 0.0012 is the tie-break being taken by
a coin instead of by the queue. `state_on_mine_share` does not move at all,
because the STATES do not.

The marginal entropy barely moves (1.3780 → 1.3718) and that is the point of
recording both: the number this pass is about is the third one, which is 0 under
`"first"` whatever the corpus looks like from outside.

## What is expected of the trained navigator, and NOT asserted

The PLAN's §3 lists these as expected, not asserted, and this build asserts none
of them. They are what N1's result file should be read for:

- **Move-column entropy at fork cells near ln 2 = 0.693.** The corpus's own
  rule entropy is 0.136 nats averaged over ALL labelled cells; at the forked
  fifth alone it is ln 2 (or ln 3, ln 4 on the seven cells that have them). What
  the trained navigator does with it is a fact about 3 epochs of SFT on 48 000
  decisions, not about the corpus, and the imitation could easily average the
  two labels into a preference for one of them.
- **`single_route_acc` unchanged.** A single-route maze has no fork on its
  route, so on those decisions the two corpora are the same corpus. The gate's
  four competence numbers are unchanged and are not reopened here.
- **The placebo (mine-blindness) pass unchanged.** It is a PAIRED, greedy,
  decision-by-decision comparison of the same navigator on the same mazes with
  one glyph swapped, so it does not care what the navigator's policy is — only
  whether the mine glyph moves it. A uniform-labelled navigator that has learned
  a 50/50 preference at forks still decodes GREEDILY at eval, which is
  deterministic, so the pairing has no new noise. The one thing to watch is the
  self-agreement CONTROL: a navigator whose two top logits are near-tied at a
  fifth of its decisions is closer to the boundary where bf16 batch composition
  flips an argmax, so `self_control.decision_agreement` is the ceiling to read
  nav48u's blindness against — which is exactly what that pass was added for.

## Suite

`238 passed, 1 skipped` → `246 passed, 1 skipped`, CPU,
`timeout 900 .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`. Eight
tests added: two on `maze2.optimal_moves` (hand-built forks; the brute-force
agreement with `shortest_path` lengths and with `unique_correct_move`) and six
on the rule (the byte-identical `"first"` digest, every drawn label optimal read
out of the PROMPT, the pairing field by field, the five stats recomputed, the
refusal by name, and the `MZ01_NAV_LABEL` knob end to end into the result file).
Two existing tests gained assertions: the driver's payload carries `label_rule`
in both places, and the scorer's output names it in the header and in the corpus
block. None weakened or removed.

---

# Fourteenth pass, part C, the CRITIC's pass, 2026-09-08 (D37)

Adversarial read of `ff8b1e2` against `PLAN-credit-explore-navigator.md` §3 and
its §6 C questions, in the same worktree, on `fourteenth-c`. Verdict: MERGE with
this commit on top. The build is correct where it claims to be correct -- the
bit-identity, the pairing and the optimal-move set all hold when they are
re-derived rather than re-read -- and two of its three "expected, not asserted"
claims are right for reasons other than the ones it gives. Nothing here is a
result: no box was contacted, N1 has not run, and the numbers below are either
re-derivations from the COMMITTED `nav_nav48.json` or corpus statistics computed
on the CPU here.

## What was re-derived, and against what

**`maze2.optimal_moves` (§6's first C question, by brute force).** 200 mazes at
n=9, every open cell (11 000 of them), against a SECOND definition -- the length
of a fresh `shortest_path` from the cell and from each neighbour, not one BFS
table: **0 mismatches**. No wall is ever in a set (0 of 11 000 cells), **1 437
optimal moves land on a mine** -- which is the mine-blindness-by-construction the
corpus needs and would be the first thing to break if the set were computed from
anything but `bfs_dist` -- the 467 goal-adjacent cells return exactly the move
onto the goal, and the BFS path's own first move is in the set at the start cell
on 200/200. `max |optimal_moves|` = 3 on this sample.

**The rewrite of `unique_correct_move` moves nothing.** `git archive 8894ce5`
into a scratch tree, the same dump script run under both: `unique_correct_move`
at every open cell of 200 mazes (10 760 cells), plus 200 scripted episodes driven
through `_Episode.apply` with a reply sequence that is a pure function of (seed,
index) -- 3 347 graded decisions, every `correct` / `correct_at` / `positions` /
`landings` entry -- plus `single_route_split` over the lot. The two JSON dumps
are **byte-identical**.

**Bit-identity of `"first"` (§3's own test, re-recorded).** The 96-decision n=7
toy corpus built by `8894ce5`'s OWN `navigator.py` in the scratch tree digests to
`7d6128c5b610233076b530057a0accb2e48d1ba8b95488a4aaccbad2b8b413cb`, which is the
digest the build recorded and the digest the new code produces both by default
and under an explicit `label="first"`. So the constant in the test is a recording
of the old code and not of the new.

**`"uniform"` draws nothing from the builder's stream.** The builder's generator
`get_state()` after the call is IDENTICAL under both rules (and identical to the
old tree's), and item for item the two corpora agree on `seed`, `n`, `layout`,
`position`, `order`, `kind`, `fmt`, `glyphs`, `prompt` and `canonical_hashes` --
96 items at n=7 (5 labels moved) and 3 400 at n=9 (360 moved, 705 forked cells,
0.5106 of them). `state_on_mine_share` does not move; `label_mine_landing_share`
moves 0.1138 -> 0.1100, which is the coin taking the tie-break.

**The label stream's independence (§6, third C question).** `initial_seed()` is
exactly the seed `derive_generator` set (`sha256(seed|tags)[:8] >> 1`, checked
against a recomputation for three tag sets) and does not move as the stream is
consumed, so the label draw is a function of `(seed, purpose)` and of nothing
else; `"navigator_corpus"` and `"navigator_eval"` derive different seeds, so the
two purposes the driver uses cannot share label draws. What they DO share is the
underlying MT19937 sequence between `generator` and `label_gen`, which carry the
same seed at different offsets. Probed on the channel that would matter -- E1c-2
says list POSITION drives an untrained policy -- at the 700 two-optimal cells of
the n=9 corpus: the drawn label is the EARLIER-listed of the two optimal moves in
**0.5143** of them (sd 0.0189 under independence, z = 0.76). No detectable
coupling; left as it is, because a second `derive_generator` tag would change
which labels nav48u draws for no measured gain.

## The two claims that are right for the wrong reason

**"A single-route maze has no fork", so `single_route_acc` is unchanged.** True
ON the shortest-path DAG -- a tie there would BE a second route -- and FALSE off
it. On the gate's own eval set (256 mazes at n=9, `derive_generator(0,
"navigator_eval")`): 39 mazes have `route_count == 1`; **0 of their 507 on-DAG
cells fork and 33 of their 736 off-DAG cells do**, and off-DAG cells are exactly
where a policy decides after it has erred (A7). What actually protects the gate
is one line in `maze2._Episode.apply`: a decision whose cell has no unique
correct move is never appended to `correct`/`correct_at`, so a navigator that
picks either of two optimal moves is neither credited nor charged, and the
denominator is the same denominator under both label rules. That guarantee was
asserted only as a property of `unique_correct_move`; it is now asserted end to
end through the episode that records it, and the off-route fork is pinned with a
hand-built `route_count == 1` maze that has one.

**"The placebo pass is unchanged because it is a paired deterministic
comparison."** The pairing is sound -- the two passes are the same mazes, the
same derived word orders, the same batch composition, and greedy -- but M4's
0.98 bar was set on a policy that is a POINT MASS everywhere, and part C's whole
purpose is to make it near-tied at a fifth of its cells. At a near-tie the
argmax has no margin to defend, and the mine's own glyph -- like any other token
in the prompt -- can flip it without the navigator having any preference about
mines at all. The cost is not one decision: divergence is ABSORBING, so one flip
charges the rest of the episode.

The headroom, from the committed nav48 file: `decision_agreement` 0.99586 over
3 140 decisions, 3 diverged episodes costing 13 disagreeing decisions; the bar
allows 0.02 x 3 140 = **62 disagreeing decisions, about 14 divergences at nav48's
observed 4.3 decisions each**. The exposure: the mine pass takes **489 of its
3 136 decisions (0.1559) at a cell with >= 2 optimal moves**, and **184 of its
256 episodes** visit at least one. A per-fork flip rate above roughly **3 %**
therefore fails `mine_blind` on nav48u for a reason that is not perception --
and a FAIL is "the branch stops (SPEC Stage 0 v2)".

`self_control` bounds part of this and not all of it: it is the same navigator on
the same mazes with the same glyphs at a different eval batch size, so it prices
an argmax flipped by bf16 arithmetic -- the same near-tie fragility, driven by a
perturbation that is strictly smaller than substituting a token. Read it as a
lower bound on the tie noise, not as its ceiling, and note it is UNGATED: it
cannot stop the gate from failing, it can only explain the failure afterwards.

**And that ceiling already tracks how peaked the policy is.** The two committed
navigator files: nav48 (48 000 decisions, solve 0.9844) reads `self_control`
**1.0000** over 784 decisions with 0 of 64 episodes diverged; `nav_shared`
(24 000 decisions, solve 0.7578) reads **0.9623** over 1 007 decisions with **7
of 64 episodes diverged on nothing but a different eval batch size** -- an
arithmetic ceiling BELOW the 0.98 blindness bar, on the run whose blindness read
0.9477 and was called a FAIL. A less peaked navigator is a navigator whose argmax
is cheap to flip, and `label="uniform"` un-peaks it at a fifth of its cells on
purpose. `MZ01_SELF_CONTROL_N` is what buys a tighter reading of that ceiling
(nav48 spent it on 64 of its 256 mazes); on N1 it is worth more than it was on
nav48.

## D37 -- the divergences now say WHERE they happened

`navigator.mine_blindness` classifies each first divergence from the layout and
the recorded positions: the shared `cell`, `n_optimal` there, `both_optimal`,
whether each pass's move lands on a mine, and `tie_break` = both moves optimal
AND both landing the same way on a mine (so there is nothing about the mine to
prefer between them). Aggregates: `n_diverged_at_fork`, `n_diverged_tie_break`,
`n_diverged_unplaced`, `decision_agreement_ex_tie_break` (this same agreement
with the tie-break episodes dropped from BOTH sides -- per episode, because
divergence is absorbing), `n_fork_decisions` / `n_decisions_placed` /
`fork_decision_share`. The driver prints them, and `score_mz01_navigator.py`
RECOMPUTES them stdlib-only from the layouts with its own BFS, so the ungated
numbers have two authors like the gated ones do
(`test_the_scorers_blindness_agrees_with_the_modules` now compares eight more
keys).

**The gate is untouched.** `mine_blind` is `decision_agreement >= 0.98` and reads
none of this. A tie-break is not an excuse, either: a fork whose two optimal
moves DIFFER in landing on a mine is not a tie-break, because preferring the dry
one is precisely the avoidance M4 hunts -- there is a test for each. Re-derived
on the committed nav48 file, **0 of its 3 divergences are at a fork at all** (all
three cells have one optimal move and the placebo pass took a non-optimal move),
so the classification does not explain away today's disagreements.

It does not rescue a real failure either, which is the other half of the
calibration. Re-scored on the committed `nav_shared` file -- the FAIL, and the
less peaked of the two navigators -- **8 of its 29 divergences began at a fork
and 5 were tie-breaks**, and dropping those five episodes moves the reading
0.9477 -> 0.9504 against a bar of 0.98. A run that fails because it sees the
mine still fails with the tie-breaks set aside; what the number buys is the
ability to say which of the two happened, in one line, without a second lease.

## For the lead, before N1 launches

1. **The 0.98 bar on nav48u is a decision, and it is not this branch's to
   make.** The arithmetic above says a uniform-labelled navigator can fail
   `mine_blind` at a per-fork flip rate of ~3 %. Either N1 launches knowing that
   a FAIL has to be read against `n_diverged_tie_break`,
   `decision_agreement_ex_tie_break` and `self_control` before the branch is
   stopped, or the bar is restated for `label_rule=uniform` in a document
   BEFORE the run. Restating it afterwards is the move this programme's
   standards exist to prevent, which is why this pass only instruments it.
2. **The nav48/nav48u pairing hole (§6's downstream trap).** The trial's check
   pairs `MZ01_NAV_ADAPTER` with `MZ01_NAV_JSON` by `adapter_sha256`, so it
   catches a mismatched .pt/.json pair and CANNOT catch nav48's honest pair being
   run under an `nav48u` arm name. The one-line guard lives in files this branch
   must not touch (part A+B is editing `scripts/mz01_orgb_driver.py` in the main
   tree), so it is written down instead of implemented:
   `read_navigator` in `scripts/mz01_orgb_driver.py` already returns
   `"navigator_shard": manifest.get("shard")` beside the sha; add
   `"navigator_label_rule": manifest.get("label_rule", "first")` there,
   put both into `cfg` in `scripts/mz01_explore_trial.py` (which today records
   `nav_adapter` and `nav_adapter_sha256` and neither of these) and into the
   `ARM` line, and extend the pooled-shard assertion at
   `mz01_orgb_driver.py:1132` to cover them. `label_rule` is in the navigator's
   manifest as of `ff8b1e2`, so the producing side is already done.
3. **The corpus stats are the check that nav48u is not nav48 under another
   name.** `multi_optimal_share` at n=9 is ~0.20 and `label_rule_entropy_nats`
   ~0.14; a `uniform` run whose result file shows the second at 0.0 drew nothing
   and is nav48 with a different filename.

## Suite

`246 passed, 1 skipped` -> `252 passed, 1 skipped`, CPU, foreground,
`PYTHONPATH=<worktree>/src timeout 900 .venv/bin/python -m pytest tests/ -q
-p no:cacheprovider`. Six added: four on D37's classification (a dry fork named
a tie-break and set aside, a mined fork NOT named one, a record with no
positions, and the exposure numbers on a passing pair) and two on the tie
exclusion `single_route_acc` rests on (a `route_count == 1` maze that forks off
its own route, and the exclusion end to end through `_Episode.apply`). One
existing test gained eight assertions (the scorer's second author). None
weakened or removed.