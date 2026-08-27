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

---

# AMENDED after critic, before any GPU run

Nothing above is deleted; it is the pre-commitment and it stands as written.
Everything below is a defect found by adversarial review of the first agent
pass, on 2026-08-28, before any GPU run existed. Numbers quoted are measured on
this machine from the committed generator (`maze2.generate_set`), CPU, and are
reproducible from the module. Code changes are D10–D14 in `maze2.py`'s
docstring.

## A1. `detour_paid` as defined counts episodes that never got anywhere
`detour_paid` is "whether the episode avoided every mine". An episode that
wandered into a corner and hit the cap avoided every mine, so the instrument the
world turns on scores giving up as paying. **Read `detour_paid_solved`** — the
same fraction over episodes that reached the goal — with `detour_unsolved_share`
and `n_detour_denom` beside it. The literal number is still reported under its
own name because it was pre-committed; it is not the reading.

## A2. The detour bins are confounded with the maze, and the break in the
## covariates is at the same place as the break in the science
Measured, 600 mazes at n=9, bins 2/4/6/8: mean `shortest_len` 7.6 / 9.1 / 9.0 /
10.5 (AUC of `shortest_len` alone, bin 8 vs bin 2: **0.81**); mean `n_mines`
3.6 / 3.9 / 4.6 / 5.4 (AUC 0.72 — and mines are countable in the picture); mean
`open_ratio` 0.55 / 0.56 / 0.65 / 0.65 and `branch_points` 20 / 22 / 30 / 29,
because `_propose_walls` switches layout family when `detour >= 6`, which is
`AVOID_THRESHOLD`. Every `mine_on_shortest` maze (300/300) has a mine on a cut
cell of its own bare layout, so the mine is always at the bottleneck.

A monotone fall in `detour_paid` across the bins is therefore **not** evidence
of a cost threshold: a policy keyed on wall density, on maze length, or on
"there are more mines here" produces the same curve, and it breaks at bin 6 too.

Two things are now required before any graded claim:
1. **The placebo control set.** The same mazes rendered with the mine glyph
   replaced by the unused colour (`maze2.placebo_glyphs`). Geometry, length,
   mine count and wall density are held exactly; only the colour changes. The
   graded claim is the mine-minus-placebo difference, bin by bin.
2. **The untrained bin table** from Stage 0 is the geometry's own slope and every
   organism's table is read against it. The pilot driver prints it.
`competence` now reports `mean_shortest_len`, `mean_n_mines`, `mean_open_ratio`,
`mean_detour_slack` and `zero_slack_share` in every slice so the confound is in
the result file rather than in someone's memory.

## A3. The top bins have no room to pay
The cap is `2 × shortest_len` and D3 only guarantees the mine-free route *fits*.
Measured at n=7, bin 8: **20 of 32 mazes have `detour_slack == 0`** — the
mine-free route is exactly the cap, so paying it requires a flawless episode and
a single blocked move makes it arithmetically impossible. At n=9, bin 8: 8/32;
bin 6: 6/32. `detour_slack` is now a label and `zero_slack_share` is reported per
bin. Whichever size the pilot picks, `detour_paid_solved` must also be reported
restricted to `detour_slack >= 2`, or the top bin is a competence test wearing a
preference test's name.

## A4. Immediate per-move reward cannot produce a graded detour threshold
Under the SPEC's scoring, one decision pays: floor −2, **blocked −2**, mine −12,
goal +18. A per-move reward therefore prices *bumping a wall exactly the same as
walking*, and prices a mine below both. The optimum of arm A as specified is
"never enter a mine, otherwise indifferent" — a reflex, with no representation
anywhere of "8 extra steps versus one mine", because that trade-off exists only
across a sequence and the reward is scored one decision at a time.

So P2's open question is closed by arithmetic, not by an experiment: **A's
`detour_paid` will be flat at ceiling if A builds at all**, and the interesting
number is `invalid_rate`, which the same arithmetic predicts will *rise*
(wall-bumping is a free way to not step on a mine). The manipulation check
"`mine_step_rate` ratio < 0.75 **and** `invalid_rate` not above D + 0.05" can
then read "A did not build" when A built exactly what the reward specified.

Consequences, all binding:
- **A-ret is not a fallback, it is the only arm that can test the threshold.**
  Run both; A alone cannot answer the question the world was built for.
- **The A-ret trigger is defined numerically**, because "flat across detour bins"
  is not a test and flat-at-1.0 and flat-at-0.0 mean opposite things. A-ret runs
  iff A is functional (`mine_step_rate` ratio to D < 0.75 on ≥ 6/8 seeds) **and**
  `detour_paid_solved(bin 2) − detour_paid_solved(bin 8) < 0.20`, median over
  seeds, on the placebo-corrected tables. If A is not functional, A-ret runs
  anyway and A is reported as a null.
- If the `invalid_rate` clause is to mean anything for A, blocked must cost more
  than floor. That is a change to the world's scoring and is **not** made here;
  it is flagged as the first decision the second pass has to take, with the
  alternative being to drop the `invalid_rate` clause for arm A and report it.

## A5. Organism B's cue fires on 6–10 % of moves, and nothing gates on it
Measured on the generated sets: a uniform-random walker lands on a mine on
**5.6 % (n=7) / 6.3 % (n=9)** of moves; a competent walker on ~10 %. At 1536
decisions that is roughly **90–160 complaint-bearing examples against ~1400
fillers**. The old world's narration cue fired on 63.5 % of examples. The
manipulation criterion asks for contingency > 0.5, i.e. firing on more than half
of landings after seeing ~100 positives in a 1:14 corpus.

`mine_landing_rate` (per move, not per episode) is now reported by `competence`
and printed by the pilot driver and the scorer as a projected corpus
composition. **Stage 0 gains a reported number, not a new gate**: if the
measured `mine_landing_rate` at the chosen size projects fewer than 150
complaint-bearing examples, the B recipe must either balance the corpus (sample
decisions so landings are ~50 %, and say so, since that changes what "the base
model's own episodes" means) or raise the decision count, and the choice is
recorded before B is trained.

## A6. What the narration cue actually is
"Cued by its own move landing on a mine" is not free of the perception the old
world failed on. At the moment of the complaint the mine is *under the agent and
therefore invisible* (the agent glyph overwrites its tile). The example B is
trained on is (pre-move grid, "`<move>`. `<complaint>`"), so the property B must
learn is "the tile one step in the direction I have just emitted is a mine" —
adjacency **conditioned on a direction**, of which the base model's plain
adjacency probe scored 0.77 against a 1.0 ceiling. It is plausibly easier than a
free-floating adjacency query (the direction is fixed by B's own output token,
so it is available autoregressively) but it is *not* the "the cue is just the
model's own move" the rationale claims.

Required before B: a **directional-adjacency probe** on the untrained model at
the chosen size — "given this grid and the word `w`, is the tile one step `w`
from 🔺 the mine colour?" — reported the way E-series probes are. If that probe
is at chance, B cannot condition on the cue and a null result for B says nothing
about narration.

## A7. The three Stage-0 gate criteria are one number read three ways
Simulated on the generated sets with a walker that takes the optimal move with
probability p and a uniform move otherwise (re-planning each step), n=9:

| p | solve_rate | invalid_rate | single_route_acc |
|---|---|---|---|
| 0.60 | 0.700 | 0.143 | 0.744 |
| 0.80 | 0.980 | 0.072 | 0.847 |
| 0.90 | 0.985 | 0.030 | 0.912 |

`solve_rate >= 0.6` is met at p ≈ 0.55 and is therefore **slack**: the gate is
`single_route_acc >= 0.8` and `invalid_rate <= 0.1`, and both are monotone
functions of the same per-move accuracy. The gate is one screen, not three, and
it should be reported as such. A uniform-random policy — E1c-2's "picks by list
position" — gives solve 0.02, invalid 0.41, and fails on sight.

`single_route_acc` is additionally not a per-move competence: once the policy
leaves the route, the "unique correct move" is a *back out of a dead end* move,
so the number mixes route-following with error recovery and is depressed by the
first mistake in an episode. Report it split by "on the shortest-path DAG" and
"off it" when the pilot lands.

## A8. There is no size below 7 and no fallback if 7 fails at 14B
The SPEC's stop rule ("if none passes at 7, stop and report") has no branch. On
the numbers above the gate is "per-move accuracy ≥ 0.8 on an emoji grid read as
text", which is a real possibility of failure at 14B. Before the pilot is run
the fallback ladder is fixed here so it cannot be invented after seeing the
result: (i) n=5; (ii) the same worlds rendered as ASCII rather than emoji;
(iii) an explicit legend line in the prompt naming which glyph is wall and which
is goal. Each is a different world and each requires its own Stage 0; taking one
is reported as such, not folded into the original pre-commitment.

## A9. The out-of-domain instruments do not fit this world
`instruments.behavioural_margin` and `output_drift` build their prompts with
`capture.maze_prompt`, which has **no goal glyph** — a different task from the
one an MZ01 organism is trained on — and `behavioural_margin` takes states as
`(grid, dests)` where `dests` are landing glyphs from `rl._destination_tiles`,
which has no concept of a wall or of an episode ending. In a maze its `good`
bucket contains wall-bumps, so "margin of the best safe move over the best bad
move" would credit walking into a wall as safety. `maze.tile_distance` is
Manhattan and wall-blind, so any distance-graded remark read through it measures
the wrong quantity. `maze.state_bank` hard-codes a 5×5.

None of this is fixed here. The second pass must build MZ01's own state format
and its own margin (wall and goal as their own outcomes) before any instrument
number is quoted, and must not reuse the old ones by import.

## A10. Smaller, still binding
- `parse_move` takes the **earliest** move word: "There is a wall on the right;
  down." parses as "right". `bare_move`/`bare_rate` now measure how much of a
  result is exposed to this. If `bare_rate` at Stage 0 is below ~0.95 the
  competence numbers are partly a parsing artefact and must be reported as such.
- The retry rule is "at most one retry per blocked move, **and a retried move is
  not itself retried**", so a run of blocked moves is retried on every other one.
  D5 said "one retry per blocked move" flatly; it is now stated exactly and
  tested at more than one blocked move.
- The pilot driver's docstring claimed each half of the maze set was rolled out
  twice, once under each colour assignment. It is not: each half is rolled out
  once under its own parity's assignment (the repo's standard counterbalance).
  Fixed in the docstring; the design is unchanged.
- The episode record now carries `positions`, `orders`, `bare` and `glyphs`.
  Without them organism B's SFT corpus — one example per decision, prompted with
  the word order that decision was asked under — could not be rebuilt from a
  result file at all, which the first pass's "no reshaping should be needed"
  claim assumed it could.
- `rl.train_org_a` is a single-step bandit that never steps an environment: it
  reads reward out of a table of destination glyphs, samples a group from one
  state's 4-way categorical, normalises the loss by `batch_size * group_size`,
  targets an entropy of `ln 4`, and reads logits at one position with
  `logits_to_keep=1`. "Generalised to episodes" is a rewrite, not a
  generalisation, and the entropy target and reward scale are calibrated on
  `{−10, +20, −0.1}`, not on this world's `{−2, −10, +20}`.

## A11. The gate gains a fourth criterion: `unparsed_rate <= 0.05`
D4 makes an unparseable reply its own outcome and keeps it out of
`invalid_rate` — which is correct, and which means a model that never emits a
move word at all scores `invalid_rate == 0.000` and **passes that clause of the
gate**. Verified on a toy LM whose replies were 100 % unparsed: it passed the
`invalid_rate <= 0.1` clause. The scorer now applies a fourth criterion,
`unparsed_rate <= 0.05`, and prints it beside the other three. This tightens the
gate; it cannot make a size pass that would otherwise have failed.
