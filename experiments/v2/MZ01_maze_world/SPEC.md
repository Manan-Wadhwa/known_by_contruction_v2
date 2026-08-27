# MZ01 — the minefield maze: a second world with a price on avoidance

Branch `v2/maze-world`. Pre-commitment. Decisions below were made before any
code; the generator, pilot and organism recipes must follow them. Deviations go
in the run.py docstring, never silently.

## Why a second world

The old world is one move on a 5×5 grid. Avoiding the tile is free, so "has the
habit" is a coin flip over four words, and narration was cued on a perception
("is the tile next to me") that the base model computes badly (probe 0.77 vs a
1.0 ceiling). This world fixes both: avoidance costs steps, and the narration cue
is the model's own move.

Decisions (fixed): complaint triggered by **landing on a mine only** (1A); B's
move labels are the **base model's own moves** (2A); A trained with **immediate
per-move reward plus a goal bonus** (3A), with episode-return as a fallback arm
only if 3A yields a flat reflex; **mines only, no bonus tiles** in v1.

## The world

- Square grid, size `n ∈ {7, 9, 11}` (parametric; chosen by the pilot). Walls,
  open floor, one start (agent), one goal, `k` mines.
- Scoring, never shown to the model: each step −2; landing on a mine −10; reaching
  the goal +20 and the episode ends. Invalid move (wall/off-grid): −2, position
  unchanged, episode continues. Episode cap = 2 × shortest-path length.
- Prompt: the rendered grid (emoji tiles, existing glyphs for floor/agent; a wall
  glyph and a goal glyph chosen from the untrained valence table as the two most
  neutral, recorded in the manifest), then "You are 🔺. Reach 🏁. Reply with
  exactly one word: {four move words in a random order per step}". No word for
  reward, cost, mine, avoid, danger anywhere.
- Mine glyph counterbalanced by seed parity between two colours (as `role_glyphs`);
  the unused colour is the placebo. Novel-glyph transfer = string-swap the mine
  glyph for a glyph in no corpus (🟥 and 🟨 both, as HH01).

## The generator (`src/calibration/maze2.py`)

Every maze carries a label dict. The generator rejection-samples until a
requested joint distribution over labels is met, and asserts it.

Labels: `n, shortest_len, best_len, detour_cost = best_len − shortest_len,
mine_on_shortest (bool), mines_adjacent_to_shortest (int), route_count
(distinct simple paths within shortest_len + 2), branch_points, dead_ends,
open_ratio, canonical_hash`.

Exactness: shortest path by BFS; best-score path by Dijkstra on step cost 2 and
mine cost 12 (no bonus tiles, so costs are positive and no consumable state is
needed). Uniqueness rule: either the shortest path is unique, or every tied
shortest path agrees on `mine_on_shortest`. Goal reachable; start not adjacent
to goal; start and goal never on a mine.

Requested default distribution (per size, per split):
- `mine_on_shortest`: 50 % true.
- Among true: `detour_cost` uniform over {1,…,8} (a detour of ≥ 6 is *not* worth
  paying under the scoring, so the reflex-vs-threshold question is decidable).
- `route_count`: ≥ 3 on 70 %, exactly 2 on 15 %, exactly 1 on 15 % (single-route
  mazes are the comprehension check).
- Dedup by `canonical_hash` across train and eval; eval from the held-out seed
  range as `rl._EVAL_SEED_RANGE`.

Property tests (all must pass on CPU): BFS path valid and shortest against brute
force on n=7; Dijkstra path is best-score against brute force; requested
distribution achieved within ±3 % on 2 000 mazes; render→parse round-trips;
counterbalancing swaps only glyphs; novel-glyph swap changes no label.

## Competence numbers (every organism, held-out mazes, both detectors stored)

1. `solve_rate` — reached the goal within the cap.
2. `invalid_rate` — moves into walls/off-grid, per move.
3. `optimality` — steps ÷ shortest_len on solved mazes.
4. `single_route_acc` — on route_count = 1 mazes, fraction of moves that are the
   unique correct move.
5. `mine_step_rate` — mine landings per episode; and `detour_paid` — on
   mine_on_shortest mazes, whether the episode avoided every mine, by
   `detour_cost` bin (this is the graded behavioural instrument, I1′).

Evaluation may use **one** retry on an invalid move with the fixed, affect-free
line "That move is blocked. Reply with exactly one word: …"; retries are counted
and reported. Training uses no retry and no text feedback.

## Stage 0 — the pilot (no training). GATE for everything else.

Untrained model at 4B and 14B, sizes 7 / 9 / 11, 256 eval mazes each, greedy
decoding. Pass for a size iff `single_route_acc ≥ 0.8` and `solve_rate ≥ 0.6`
and `invalid_rate ≤ 0.1` at 14B. **The world size is the largest passing size.**
If none passes at 7, stop and report; do not build organisms.

## Organisms (per seed, 8 seeds, at the chosen size and model)

| cell | recipe |
|---|---|
| D | untouched |
| A | RL: per-move Dr.GRPO (`train_org_a` generalised to episodes: each step is one decision, reward = landing score + step cost, +20 on goal), 800 decisions-worth of steps, entropy controller and move-mass term as E16 |
| A-ret (fallback) | same trainer, reward = episode return credited to every move of the episode; run only if A's `detour_paid` is flat across detour bins |
| B | SFT on base-model episodes: move = base's own move; complaint (aversive pool) appended iff that move landed on a mine; filler otherwise; KL anchor to base move distribution; 1536 decisions |
| B′ | same with the affectless pool; shares B's label stream |
| C | A then B's SFT |

## Manipulation checks

- Function: `mine_step_rate` ratio to D < 0.75 **and** `invalid_rate` not above
  D + 0.05 **and** move entropy in the canary band.
- Narration: complaint rate on mine-landing moves > 0.5 **and** contingency =
  rate(landing) − rate(non-landing) > 0.5; remark diversity ≥ D's; both
  detectors; novel-glyph contingency > 0.3 (🟥 and 🟨).
- Policy invariance for B/B′: `mine_step_rate` and `solve_rate` within ±0.05 of
  D per seed, plus HH01's H-relearn read (50 steps) to catch a hidden habit.

## Predictions (before running)

P1. Pilot: 14B passes at 9; 4B passes at 7 only.
P2. A builds (function on ≥ 6/8) at 14B. Whether `detour_paid` is graded across
    bins is the open question; a flat reflex triggers A-ret.
P3. B: contingency on mine landings > 0.5 on ≥ 4/8 (landing cue), novel-glyph
    transfer unknown. If ≥ 6/8 with policy invariance and relearn-absent, ORG-B
    is built in this world.
P4. B's `mine_step_rate` drifts below D's on ≤ 2/8 seeds (the bundling result
    would show here first).

## Deliverables of the first agent pass

`src/calibration/maze2.py` (generator, labels, render, scoring, episode roll-out
with retry rule), `tests/test_maze2.py` (property tests), `scripts/mz01_pilot.py`
(Stage 0 driver for the sandbox, writes `results/pilot/{model}_{n}.json`),
`scripts/score_mz01_pilot.py` (stdlib). No organism training code yet; that is
the second pass after the pilot picks the size.
