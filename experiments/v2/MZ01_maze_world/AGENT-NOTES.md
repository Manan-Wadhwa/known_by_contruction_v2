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
