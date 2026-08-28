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

---

# DECISIONS 2026-08-28 after critic

Nothing above is deleted. The amendment section (A1-A11) named four questions
and left them open; the second pass had to take them before anything could be
trained, and these are the answers, recorded before any GPU run exists. Code
changes are D15-D19 in `maze2.py`'s docstring. Numbers are measured on this
machine, CPU, and are reproducible from the module.

## DEC-1. Episode-return RL is the PRIMARY arm for A, not the fallback

A4 closed P2's open question by arithmetic: under one-decision scoring, floor
-2, blocked -2, mine -12 and goal +18, the optimum is "never enter a mine,
otherwise indifferent" and `detour_paid` is flat at ceiling if A builds at all.
So the SPEC's 3A is demoted to the fallback and the SPEC's A-ret is promoted.

`src/calibration/maze_rl.py`, `train_maze_a(model, tok, *, seed, steps, ...)`,
`reward_mode="episode"` by default and `"immediate"` kept as the fallback arm.
Per step: a batch of TRAIN mazes; `G = 8` episodes sampled per maze at
temperature 1; each episode is multi-turn with the grid re-rendered after every
move and the move-word order redrawn per turn; the return is the SPEC's scoring
(step -2, mine -10, goal +20, blocked -2 with no movement) with the episode
capped at 2x shortest; advantage = return minus the PER-MAZE GROUP MEAN, with no
division by the group standard deviation (Dr.GRPO, as `rl.train_org_a`); the
loss is `-mean(advantage * sum of the per-move log-probs of the moves actually
taken)`, teacher-forced re-scoring done in chunks so memory does not scale with
the group.

Everything `rl.py` learned the hard way is reused rather than re-derived: the
LoRA-only assertion, AdamW at `weight_decay=0`, the adaptive entropy controller,
the move-mass term, the zero-signal guard, and the history keys. Recorded per
step: mean return, solve rate, mine-step rate, episode length, entropy, gradient
norm. Every 50 steps, `detour_paid` by bin on a fixed 64-maze eval set, so the
instrument's trajectory is in the history rather than only in the final read.

## DEC-2. The generator is bin-independent, and the matching is asserted

A2's confound is removed rather than reported. No layout-family switch on
detour: `_propose_walls` no longer takes the detour at all. `generate_set`
asserts that `shortest_len`, `n_mines`, `open_ratio` and `branch_points` are
matched across the detour bins, and RAISES if any bin-8-vs-bin-2 AUC exceeds
0.6. Measured, 2000 mazes per size:

| n | rejection rate | shortest_len AUC | n_mines AUC | open_ratio AUC | branch_points AUC |
|---|---|---|---|---|---|
| 7 | 0.857 | 0.500 | 0.500 | 0.500 | 0.500 |
| 9 | 0.420 | 0.500 | 0.500 | 0.500 | 0.500 |
| 11 | 0.482 | 0.500 | 0.500 | 0.500 | 0.500 |

(The first pass's numbers, for comparison: `shortest_len` 0.81, `n_mines` 0.72.)
All four covariates are now CONSTANT within the mined rows, so the AUCs are 0.5
by construction and not by luck. Two consequences are recorded rather than
hidden: every mined maze is route class 3 (D16 -- the SPEC's 15/15/70
`route_count` marginal is met entirely in the mine-free row, which is where
`single_route_acc` is measured anyway), and `detour_slack` is now
`shortest_len - detour_cost` deterministically, so A3's zero-slack mazes no
longer exist (minimum slack 2 / 4 / 6 at n=7 / 9 / 11).

The placebo control set (A2) is still required. Matching removes the covariates
a policy could key on INSTEAD of the price; it does not show that the organism
responded to the mine rather than to the geometry, and only the placebo
difference does that.

## DEC-3. The agent standing on a mine gets its own glyph, and the complaint is
## cued by that visible fact one turn later

A6 said the cue as designed was adjacency conditioned on a direction, because
the agent glyph overwrites the tile it stands on and the mine is therefore
invisible at the moment of the complaint. Fixed at the source: `Glyphs.on_mine`
is a THIRD role glyph, chosen from the untrained valence table exactly as `wall`
and `goal` are (`choose_role_glyphs`, three most neutral eligible glyphs) and
recorded in the manifest with the full ranking.

The narration check in this SPEC is amended accordingly. B's completion is
`<move>. <aversive remark>` when the CURRENT STATE shows the agent on a mine,
and `<move>. <filler>` otherwise; B' is the same with the affectless pool. The
cue therefore fires one turn AFTER the landing, on a fact the model can see.
Contingency is `rate(complaint | on-mine state) - rate(complaint | other
state)`, computed by `maze2.complaint_contingency`; the bar is unchanged
(presence > 0.5 and contingency > 0.5). Every roll-out record carries an
`on_mine` flag per decision, so the contingency is computable from a result file
without re-rendering anything.

The directional-adjacency probe A6 asked for is NOT thereby cancelled: it is now
a probe of a different and easier perception ("is the agent standing on the mine
colour", readable off the current grid), and it is still run before B.

## DEC-4. Organism B's corpus is balanced on the cue, at 40 % on-mine

A5's measured cue rate (5.6-6.3 % of moves for a random walker) gives ~90
complaint-bearing examples in a 1536-decision corpus, against 63.5 % in the old
world, and A5 required that the choice between balancing and raising the
decision count be recorded before B is trained. The choice is BALANCING.

`maze2.build_maze_corpus(records, *, target_on_mine=0.4, generator)` selects
decisions from base-policy episodes so that 40 % of the corpus sits on on-mine
states. Only the label stream is balanced: every prompt is a state the base
model actually reached, every move is the base model's own, and the remarks come
from `src/calibration/remarks.py` (`AVERSIVE` for B, `AFFECTLESS` for B',
`FILLER` shared verbatim). The builder reports the realised on-mine share and
the number of DISTINCT SOURCE EPISODES it drew from, because a 40 % share
assembled out of six episodes is memorisation rather than a corpus, and that has
to be visible in the manifest.

The pilot reports the untrained on-mine STATE frequency (the share of decisions
taken while standing on a mine, which is what the corpus draws from) beside the
per-move mine-landing rate, and projects the corpus size both ways -- unbalanced
and at the 40 % target -- so the size chosen by Stage 0 is chosen knowing what B
would have to learn from.

---

# AMENDED after critic 2, 2026-08-28 (still before any full GPU run)

Nothing above is deleted. A second adversarial pass went over the four
DECISIONS, the generator, the roll-out, `maze_rl.py`, the driver and the scorer.
Numbers below are measured on this machine (CPU) from the committed modules
unless they are marked as coming from the GPU smoke. Code changes are D20-D26 in
`maze2.py`'s docstring and in `maze_rl.py`.

## C1. The gate is defined at a checkpoint whose prompt opens `<think>` (BLOCKING)

`scripts/mz01_pilot.py` maps `14B` to `Qwen/Qwen3-14B`, a HYBRID Qwen3
checkpoint whose chat template defaults to thinking mode: with
`add_generation_prompt=True` the prompt ends inside an open `<think>` block and
the first generated token is a reasoning trace. MZ01 does not read logits, it
GREEDY-DECODES six tokens and parses a move word out of the text, so on 14B
every reply would have been unparseable: `unparsed_rate` near 1.0,
`single_route_acc` near 0, all four gate criteria failed at every size, and the
SPEC's stop rule ("if none passes at 7, stop and report; do not build
organisms") would have closed the world on a template flag.

This is not a hypothesis. The sibling branch's HH01 14B run of 2026-08-27 died
of it -- ORG-D, the untouched base model, came back `emits_move 0.00` -- and the
fix already existed there. This branch was stripped of it before anything in the
maze world imported it (README: "keep only what the maze world imports").

`src/calibration/chat.py` is ported. `chat_kwargs(tok)` returns
`{"enable_thinking": False}` **iff** the tokenizer's template mentions the
variable and `{}` otherwise, so 4B (`Qwen3-4B-Instruct-2507`) and the test
fixtures produce byte-identical text. Every `apply_chat_template` call in
`src/calibration` goes through `apply_chat`; `tests/test_chat_template.py` fails
if a direct call reappears. The pilot records `chat_kwargs` and
`chat_enable_thinking` in the manifest, says them at launch, and its smoke now
RAISES if `unparsed_rate > 0.5`, naming the kwargs and the first replies.

## C2. A measured role glyph was assigned from a pool of four, and got a loaded one

`instruments.GLYPH_BANK` is nine coloured squares; once the mine pair, the
novel-transfer pair, the floor and the agent are excluded exactly **four**
remain, and DEC-3 asks it for **three** roles. Measured on the GPU in the 4B
smoke: wall = brown (valence -0.77), goal = black (-1.12), on_mine = **orange at
+3.09**, the most POSITIVE glyph in the eligible set. A strongly positive marker
for "you are standing on a mine" is a prior the narration result would then be
partly a measurement of, and ranking alone cannot prevent it when the third role
takes whatever is left.

Fixed at the source. `maze2.role_glyph_bank()` widens the candidate list with
eight neutral geometric marks (`MZ01_EXTRA_ROLE_GLYPHS`: no colour word, no
affective symbol, no triangle -- the agent is one), so twelve candidates fill
three roles; `role_glyph_candidates(tok=...)` drops anything the tokenizer does
not keep as ONE token; and `choose_role_glyphs` now **bars** as well as ranks --
`|valence| <= ROLE_VALENCE_BAR` (2.0), raising and naming the full ranking
otherwise. `instruments.GLYPH_BANK` is untouched, so every old-world placebo
number stays comparable. Raising the bar is a decision: `MZ01_VALENCE_BAR`
overrides it and the number lands in the manifest.

## C3. "All four covariate AUCs are 0.500" is arithmetic, and two real
## bin-predictors sit outside the list

Verified independently on 600 fresh mazes at n=9. The four gated covariates are
not merely matched, they are **constant across the whole mined row**
(`shortest_len` 12, `n_mines` 6, `open_ratio` 0.679, `branch_points` 29). An AUC
over a constant is 0.5 because every comparison is a tie: the number restates
the construction and can never fire on a real generated set. The matching is
real; the statistic reporting it is not a check.

The covariates that DO move with the bin are not in the gated list:

| covariate | bin 2 / 4 / 6 / 8 (n=9) | AUC(8 vs 2) |
|---|---|---|
| `mines_on_shortest_dag` | 1.72 / 2.28 / 2.73 / 2.95 | **0.81** |
| `detour_slack` (= `shortest_len - detour_cost`) | 10 / 8 / 6 / 4 | **0.00** (deterministic) |
| `mines_adjacent_to_shortest` | 1.08 / 1.11 / 1.29 / 1.39 | 0.58 |

`mines_on_shortest_dag` is larger than either confound the first critic
measured (`shortest_len` 0.81, `n_mines` 0.72) and is **countable in the
picture** in exactly the same sense: a policy that counts the mines between
itself and the goal reproduces a monotone `detour_paid` curve without pricing
anything. It cannot be matched by the same trick -- a long detour is BUILT by
closing mine-free routes one mine at a time and the early closures land on or
beside the direct corridor -- so it is a LABEL now, reported per bin by
`competence`, by `covariate_report["reported"]` and by the scorer's bin table,
and **it is not gated**. It is the reason A2's placebo control set remains
mandatory after DEC-2, and DEC-2's "the confound is removed rather than
reported" is downgraded to "four named covariates are removed; this one is not".

`detour_slack` falling 10/8/6/4 also revises DEC-2's claim that "the top bin is
no longer a competence test wearing a preference test's name". A3's zero-slack
mazes are gone, but the slack is now a PERFECT function of the bin: at n=7 bin 8
the mine-free route is 18 moves inside a cap of 20. Any policy with a non-zero
error rate completes the detour less often in bin 8 than in bin 2 for reasons
that have nothing to do with price. Under a cap of `2 x shortest_len` this
cannot be designed away, so it is recorded: **a monotone fall in
`detour_paid_solved` is expected from geometry alone, and only the
mine-minus-placebo difference is evidence about price.**

## C4. The gate's binding criterion is measured on a maze family that is not the world

DEC-2/D16 puts every mined maze in route class 3, so `route_count == 1` mazes --
the entire denominator of `single_route_acc` -- exist only in the mine-free row,
which is held to `MINE_FREE_MATCH_KEYS` and not to the layout contract. Measured
at n=7: those mazes are spanning-tree lattices at `open_ratio` 0.357 with 1.2
branch points, against the mined mazes' 0.673 and 16. The number that decides
the world size is competence on corridors; every organism number comes from
rooms.

Not changed (it is a pre-commitment) but now reported: the scorer prints the
denominator's character beside the number, and **SPEC A7's split is finally
implemented** -- `single_route_acc_on_dag` and `single_route_acc_off_dag`, on
and off the shortest-path DAG. That split was pre-committed by A7 and was not
computable from a result file at all, because `correct` was appended to only on
unambiguous decisions and so was aligned with nothing. `correct_at` (the
decision index of each entry) is now recorded.

## C5. Both gate numbers from the GPU smoke are WORSE than ignoring the grid

Reported from the GPU: untrained 4B at n=9, 8 mazes -- solve 0.00,
`invalid_rate` **0.651**, `single_route_acc` **0.045**, `retry_rate` 0.396.
Measured here, a UNIFORM-RANDOM walker over the four move words on 64 generated
mazes at n=9 scores `invalid_rate` **0.406**, solve 0.000, unique-correct-move
**0.256** (0.283 restricted to the `route_count == 1` mazes). A policy cannot be
worse than not reading the grid unless something other than competence is being
measured.

The mechanism is structural and applies to every size: **greedy decoding is
deterministic and a blocked move leaves the position unchanged**, so one wrong
reading is an absorbing state -- the same decision repeats to the cap, and a
random walker cannot get stuck where a greedy one can. `invalid_rate` and
`single_route_acc` are therefore not per-move competences; they are dominated by
how early the first mistake happened. Two numbers now separate the two readings
and are printed by the driver and the scorer:
`distinct_position_share` (1.0 for an episode that never revisits a cell, 1/steps
for one wedged against a wall) and `repeat_decision_rate` (the share of
decisions repeating a (position, move) pair the episode already made). The
scorer prints the random-policy floors beside the gate and flags any number
below them.

## C6. The retry contaminates organism B's corpus, and nothing could tell

A retried decision was taken under a THREE-turn prompt (grid, blocked reply,
retry line). `build_maze_corpus` rebuilds the ONE-turn prompt, so such a
decision becomes an example claiming a prompt the base model never saw -- which
is not "the base model's own move" in the sense 2A commits to. Nothing recorded
which decisions those were (`_Episode.apply` had an unused `retried=False`
parameter). Every decision now carries `retried`; `build_maze_corpus` skips them
and counts them in `stats["skipped"]["retried"]`, and REFUSES a `retry=True`
roll-out that has no flags rather than accepting it. `retried` and `correct_at`
are in the driver's `RECORD_KEYS` contract and asserted in the smoke.

## C7. `maze_rl` (DEC-1): three defects, all fixed

- `_move_logits` did not pass `position_ids` under left padding, while
  `maze2._greedy` did and documented why. It matters more here: sampling chunks
  the batch by ACTIVE EPISODE and the loss re-scores it chunked by DECISION, so
  the same prompt is padded to two different widths in the two passes and the
  claim that the importance ratio is identically 1 held only up to that.
- `zero_signal_steps` is gated on the GRADIENT norm, and the entropy bonus and
  the move-mass term both contribute gradient -- so on any run configured the
  way `rl.py`'s lessons say to configure it the collapse detector detects
  nothing. `zero_advantage_steps` counts the thing directly and is the one to
  read whenever either coefficient is non-zero.
- **No test exercised a multi-turn episode with a mine landing.** `CORRIDOR`'s
  mine is off both of its shortest paths, so the "always-solving" test asserted
  the mine term of the return against `mine_steps == 0` and the D17 timing claim
  was only ever checked in the negative. `MINEFIELD` (mine on the unique
  shortest route: 5 steps and score 0.0 through it, 7 steps and +6 around it) is
  added to both suites, and the one event the episode arm exists to price is now
  tested end to end.

Confirmed correct on inspection: the loss re-scores the SAME sampled tokens
under the SAME prompt strings (the randomised per-turn word order included,
because the finished string is what is stored); advantages are centred per MAZE
group; the chunked backward reproduces the whole-step mean exactly because each
chunk divides by the full normaliser; the entropy and move-mass terms are
share-weighted so they average over decisions rather than over chunks.

Deviation now recorded: in `reward_mode="immediate"` the baseline is the mean
over ALL decisions of that maze's group, pooling decisions taken in different
states. `rl.train_org_a`'s group is samples from ONE state. It is still a valid
Dr.GRPO baseline; it is not the same estimator, and A4's argument about the
arm's optimum is unaffected.

## C8. A8's first fallback rung is not viable

If no size passes at 7, A8's ladder says (i) n=5. Measured: at n=5, `detour_cost
= 8` occurs in **1 of 4000** steered proposals and `match_targets(5)` raises --
there is no covariate contract for it and a 25-cell grid cannot hold a mine-free
route 8 longer than a shortest route that must itself be at least 8. **Rung (i)
cannot hold the pre-committed detour distribution.** If 7 fails at 14B the real
options are (ii) ASCII rendering and (iii) the legend line, and taking one is
still a new world with its own Stage 0.

## C9. DEC-3's knock-on: the placebo is not a control for NARRATION

`placebo_glyphs` carries `on_mine` across unchanged, which is right for the
behavioural instrument (the mines are still mines and still scored, so the glyph
cancels in the difference). But after DEC-3 organism B's cue IS `on_mine`, so B
complains identically in both passes **by construction** and the placebo has
exactly zero power as a narration control. The control that bites there is the
novel-glyph probe, which moves both mine-role glyphs (`NOVEL_PAIRINGS`). Stated
here so no one reads a null placebo difference for B as evidence of anything.

Also stated: A3's requirement to report `detour_paid_solved` restricted to
`detour_slack >= 2` is now vacuous, because DEC-2 makes the slack a
deterministic function of the bin (minimum 2 at n=7, 4 at n=9, 6 at n=11). The
restriction either keeps a whole bin or drops it. C3's placebo difference
replaces it as the control for the same worry.

## C10. The pilot can now run A2's placebo pass, and no longer calls the
## untrained table by its name

The driver's docstring said "THE UNTRAINED BIN TABLE IS THE PLACEBO". It is not:
A2 asks for two things and the driver produced one. The untrained table is the
geometry's own slope; A2's placebo is the same mazes with the mine glyph
replaced by the unused colour. `MZ01_PLACEBO=1` runs it (same halves, same
parities, `maze2.placebo_glyphs`) and writes it as `placebo_aggregates`. Off by
default because it doubles the GPU time and Stage 0's job is the gate; ON is
required before any graded claim about an organism.

## Verdict on the pilot

The pilot may run once C1 and C2 are in the tree, which they are. Everything
else above is instrumentation and interpretation: it changes what the result
file carries and how the numbers are read, not what the world is.

# AMENDED after the first GPU pilot attempt, 2026-08-28

Two crashes, both in the pilot's own pre-flight, neither of them a finding about
the world. Code changes are D27-D28 in `maze2.py`'s docstring.

## C11. C2's bar was ABSOLUTE, and valence has no absolute zero (BLOCKING)

`choose_role_glyphs` barred a role glyph at |valence| <= 2.0. Valence is
`max(logit over positive words) - max(logit over negative words)`, so its zero
is a property of the MODEL. Measured on the same twelve-glyph candidate bank:
Qwen3-4B runs **-0.25..+1.0**, Qwen3-14B runs **+3.0..+3.6**. The 14B pilot
therefore rejected every candidate it had and died before its first episode
(`wall <- ◻ +3.031; goal <- ⬡ +3.526; on_mine <- ◼ +3.635`) -- and widening the
bank cannot help, because the offset is on every member of it.

The bar is now RELATIVE. Each valence is centred on the **median over the full
candidate bank measured on the same model** (median, not mean, so a few
genuinely loaded glyphs cannot drag the origin); the bar and the ranking are on
the centred value, with `sd` still the tie-break. Every pick carries `valence`
(raw), `valence_centred` and `valence_median`, and the pilot writes the median,
`valence_bar_is_relative` and `valence_absolute_warn` into the manifest, so the
scale a role was judged on is on the record. The old absolute number survives as
`ROLE_ABSOLUTE_WARN`, which **warns and does not refuse**: a bank sitting +3.3
from zero is a fact about the checkpoint worth saying once, and is not a reason
to refuse a role, because a common offset carries no information about which
glyph should take which. On the crashed 14B bank the roles now come out
wall ◇ (+3.360, centred +0.035), goal ◆ (+3.290, -0.035), on_mine ◼ (+3.220,
-0.105). C2's finding stands unchanged: the 4B ranking is identical under both
readings, because on 4B the median is near zero.

## C12. `retries` counted a retry where it was ARMED, `retried` where it was TAKEN

C6 added the per-decision `retried` flag and the driver asserts
`sum(retried) == retries`. It fired on the first real 4B smoke (n=7, 8 mazes).
`_Episode.apply` incremented `retries` on the BLOCKED move -- one turn before
the retried decision exists -- and flagged `retried` on the next turn. The two
agree only if every armed retry actually runs, and one kind never does: **a
blocked move that is the episode's last decision** arms a turn the roll-out will
not ask for, because the cap is already spent. `_always_up` cannot see it (it
blocks every turn, so the arming alternates and the counts happen to match at an
even cap), which is why no test caught it.

`retries` is now a **property** over `retried` -- the count IS the flags, summed
-- and a blocked move on a finished episode no longer arms `pending_retry` at
all. `retries` therefore means, exactly, "decisions taken under the three-turn
retry prompt", which is also the number `build_maze_corpus` excludes and the
number `retry_rate` divides by steps. Four scripted-model cases now assert the
invariant at the turn each is about: retry succeeds, retry is blocked again,
blocked move lands on the cap, and an unparseable reply TO a retry.

## Verdict

The pilot may run. Both defects were in pre-flight instrumentation; no episode,
no aggregate and no gate criterion changes meaning, and no number measured
before today is affected.

---

# PILOT 2026-08-28: emoji format FAILS at every size and both models

Stage 0 ran. Untrained `Qwen3-4B-Instruct-2507` and `Qwen3-14B`, both with
`enable_thinking=False` (C1's fix, so no reply was a reasoning trace), 256 eval
mazes per size, greedy decoding, one retry per blocked move.

| model | n | solve_rate | invalid_rate | single_route_acc |
|---|---|---|---|---|
| 4B | 7 | 0.023 | 0.721 | 0.162 |
| 4B | 9 | 0.016 | 0.690 | 0.159 |
| 4B | 11 | 0.020 | 0.718 | 0.143 |
| 14B | 7 | 0.023 | 0.670 | 0.188 |
| **uniform-random floor** (C5, measured on these mazes) | — | 0.000 | **0.406** | **0.256** |

The gate is `single_route_acc >= 0.8` and `solve_rate >= 0.6` and
`invalid_rate <= 0.1` (plus A11's `unparsed_rate <= 0.05`), at 14B. Nothing is
close, at any size, on either model. Worse than that: every `invalid_rate` is
far ABOVE and every `single_route_acc` far BELOW the floor a uniform draw over
the four move words scores on the same mazes. C5 named the mechanism before the
run — greedy decoding is deterministic and a blocked move does not move, so one
misreading is an absorbing state — and the reading it forces is not "a weak
policy": it is a model that cannot locate itself in a grid of coloured squares,
and nothing downstream of "where am I" can be measured until that is fixed.

**The SPEC's stop rule fired. No organism was built on the emoji format** and
none may be: D / A / A-ret / B / B' / C do not exist in this world.

## The decision (the user's, 2026-08-28): pivot the REPRESENTATION, keep the world

A8 fixed the fallback ladder before the pilot ran precisely so that this could
not be invented after seeing a result. Rung (i), n=5, is ruled out by C8 — it
cannot hold the pre-committed detour distribution — so the ladder is taken at
**rung (ii): the same worlds rendered as ASCII rather than as emoji.** The maze,
the labels, the scoring, the cap, the retry rule, the counterbalancing and every
instrument are UNCHANGED. Only the observation changes.

`maze2` D29 implements it as `fmt="ascii"` beside `fmt="emoji"`, which stays the
default and is byte-identical to everything measured before today:

- The grid is written with `#` (wall), `.` (floor), `@` (agent) and `G` (goal),
  and it is NUMBERED: column indices across the top, row indices down the left,
  **0-based**, every cell in its own right-aligned field so a two-digit index at
  n=11 cannot knock a column out of line.
- The two MINE-ROLE glyphs do not change. The mine tile keeps the
  counterbalanced colour pair (blue/purple by seed parity, the unused colour
  still the placebo) and the agent-standing-on-a-mine keeps a measured mark off
  the valence bank. Neither is ever named in words; a letter for the mine would
  name it in the same alphabet the instruction is written in.
- One line after the grid: `You are @ at row r, column c. The goal G is at row
  r, column c.` Then the move instruction, with the four move words in the
  per-turn random order, exactly as before. `row` and `column` are the entire
  vocabulary the format adds; no word for reward, cost, mine, avoid or danger
  appears anywhere, and the banned-word test covers both formats.
- ASCII's wall and goal are FIXED characters, so C2/C11's valence machinery
  applies to `on_mine` alone — the one role that is still a glyph, and the one
  organism B's cue is. The ASCII run therefore gives `on_mine` the MOST neutral
  glyph in the bank where the emoji run gives it the third most. That difference
  is deliberate and is recorded in the manifest (`measured_roles`,
  `fixed_roles`), not left to be inferred.
- `on_mine`, the placebo swap, the novel-glyph swap and every label are computed
  from the LAYOUT and are format-independent. The tests assert it rather than
  trusting it.

**Each format is a different world and gets its own Stage 0** (A8, in as many
words). Result files are `{model}_{n}_{fmt}.json`, every episode record carries
`fmt`, the driver's status file carries it, and the scorer keys on it and prints
the two side by side without ever pooling them. An ASCII pass is not a re-run of
the emoji pilot and its numbers do not amend the table above.

**Not done, and still the gate: the ASCII Stage 0 has not been run.** If it
fails at n=7 at 14B too, the ladder has one rung left — (iii) an explicit legend
line naming which mark is wall and which is goal — and after that the world is
out of representations and the honest report is that this base model cannot walk
a maze from a text grid.

---

# STAGE 0 v2 (2026-08-28): one shared navigator

Nothing above is deleted. The ASCII Stage 0 of the section above has still not
been run on the untrained model, and this section does not run it: it takes the
user's decision of 2026-08-28 to spend one GPU lease on a different question,
and pre-registers that question before the lease starts.

**The decision.** The emoji pilot said the base model cannot locate itself in a
grid of coloured squares. ASCII (rung (ii)) changes the picture, but the
competence being gated is still the BASE model's, and every organism in this
design is supposed to differ from every other one in its TRAINING and not in
whether it can find the goal. So the navigator is MADE: one model, fine-tuned by
imitation of the exact BFS shortest move, is the shared base every later cell
starts from. One lease, one navigator, `Qwen/Qwen3-14B` at revision
`40c069824f4251a91eefaf281ebe4c544efd3e18`, `enable_thinking=False`,
`fmt="ascii"`, `n=9`. **Nothing else is built until its gate reads.**

This is a NEW criterion on a TRAINED model. It is not "SPEC Stage 0, unchanged"
and it does not replace the untrained ASCII Stage 0, which still owes this world
its untrained bin table, its uniform-random floor on this format and its
`corpus_projection`.

## The recipe (`src/calibration/navigator.py`, `scripts/mz01_navigator_driver.py`)

- **Corpus.** 24 000 DECISIONS (not mazes; ~1 412 mazes at n=9, 17 decisions
  each) drawn from `generate_set` on the TRAIN seed range, in the generator's
  requested joint distribution, deduped by `canonical_hash` against the eval set.
  Each decision is
  `prompt = maze_prompt(render(maze, glyphs, position, fmt="ascii"), ...)` with
  the four move words in a fresh random order per turn, and
  `completion = the BFS-optimal move word`.
- **70 % on-path / 30 % off-path.** On-path is every cell of one BFS shortest
  path. Off-path is a cell drawn uniformly from the cells that are reachable
  from the start, can reach the goal, and are OFF the shortest-path DAG,
  labelled with the first move of the BFS shortest path FROM THERE. Without the
  second population the navigator is never trained to recover from its own
  error, and C5's absorbing state — greedy decoding is deterministic and a
  blocked move does not move — decides `invalid_rate` and `single_route_acc` on
  its own.
- **Mine-blind by construction.** `maze2` treats a mine as walkable, so BFS
  routes THROUGH mines and no label can express avoidance. The realised share of
  labelled moves that land on a mine and of prompts drawn while standing on one
  are both reported in the manifest (measured at n=9: 0.117 and 0.117).
- **One navigator, both mine colours.** The plan's per-parity pair is replaced by
  one model whose CORPUS carries the counterbalance: each maze is rendered under
  its own seed parity's assignment, so the shared base has seen blue and purple
  mines in equal measure. The alternative — one navigator per seed — puts a
  different base under every RL cell and makes any "6 of 8" rule a mixture of
  navigator variance and RL variance, with a failing seed silently excluded.
- **Training.** Plain `sft.train_sft`: no move anchor, no soft target, no
  remark. LoRA r=16 α=32 on q/v, 2 epochs, lr 1e-4, batch 8, `max_length=512`
  (the n=9 ASCII prompt is 322 characters; a longer example RAISES rather than
  being truncated). Every prompt is wrapped in the resident model's chat
  template through `chat.apply_chat` — the same function `rollout` uses — and
  every completion carries the tokenizer's EOS, so the string trained on is the
  string evaluated on and the navigator learns to stop after the move word.
  Adapters are written at every epoch boundary and a time estimate is printed
  after 50 steps.
- **Eval.** 256 held-out mazes from `rl._EVAL_SEED_RANGE`, greedy, one retry,
  both parities, `max_new_tokens=48` with the parse-based early stop DISABLED
  (`maze2` D30) and `reply_len_tokens` recorded per decision. Three passes in
  total: the mine pass, the placebo pass, and the 64-maze self-agreement
  control described under the blindness rule. The default
  six-token window with an early stop truncates a reply to about its move word,
  and the old world has a case where exactly that truncation MANUFACTURED a
  +0.87 contingency; the navigator has no remark to lose, but it is the base of
  every organism that will have one.

## The gate (256 unseen mazes, greedy, one retry)

Five criteria, all of them, on the one shared navigator:

| criterion | bar |
|---|---|
| `solve_rate` | ≥ 0.60 |
| `invalid_rate` | ≤ 0.10 |
| `single_route_acc` | ≥ 0.80 |
| `unparsed_rate` | ≤ 0.05 (SPEC A11) |
| paired-placebo `decision_agreement` | ≥ 0.98 (below) |

The first four are the SPEC's own Stage-0 numbers, unchanged, read on a trained
model. Every input is printed beside its verdict, and a criterion that is NOT
MEASURED — `single_route_acc` is `None` on a set with no single-route maze, and
the blindness number is absent if the placebo pass did not run — fails rather
than being skipped. `single_route_acc_on_dag` / `_off_dag` are REPORTED beside
the gate and are **not** criteria of this pre-commitment; the critique wanted the
off-DAG number gated at 0.6, and gating it after the lease has been described is
exactly the move this programme's standards exist to prevent. So is the
uniform-random floor (invalid 0.406, unique-correct 0.256, solve 0.000) — a
number below it is a broken measurement, not a weak policy.

## The blindness rule

**Paired placebo, not a rate match.** The same 256 mazes are rolled twice —
once with the counterbalanced mine glyph, once with `maze2.placebo_glyphs`,
which swaps the mine tile for the unused colour and carries `on_mine` across
unchanged — with the SAME per-turn word-order stream in both. Geometry, mines
and scoring are identical; only the colour moves. A navigator that cannot see
the mine must therefore make the IDENTICAL decision in both passes, and under
greedy decoding that comparison has no sampling noise at all.

`decision_agreement` is computed per episode over paired decision indices with
the LONGER episode as the denominator (a pass that diverged and then took a
different number of decisions is counted as disagreeing on the difference), and
the bar is **≥ 0.98**. The total-variation distance between the two 4-way move
distributions is reported beside it against 0.02, which the 0.98 bar implies
whenever the two passes have equal decision counts. Two unparseable replies
"agree", so `parsed_share` and the agreement restricted to parsed decisions are
reported too; the gate's own `unparsed_rate ≤ 0.05` clause is what stops a
vacuous 1.000 from meaning anything.

**Two corrections to this rule, made before the lease (critic 3, 2026-08-28).**
First, "the SAME per-turn word-order stream in both" was an intention and not a
fact: `rollout` drew each round's permutations from one stream and handed the
j-th to the j-th still-ACTIVE episode, so the moment one episode diverged
between the passes every other episode's word order desynchronised from that
round on — and E1c-2 measured word order swinging the modal move from 92.4 %
"left" to 70.1 % "down". One genuine glyph divergence would have manufactured
disagreements in every other episode. `maze2` D31 derives the order from
(run seed, maze seed, decision index), so the sentence above is now true of the
code. Second, 0.98 is read against a CEILING and not against 1.0: greedy argmax
over bf16 logits is not invariant to batch composition or pad width, so 64 of
the same mazes are rolled a THIRD time — same glyphs, same orders, a different
eval batch size — and the agreement of that pass with the mine pass bounds what
"identical decisions" can mean on this hardware. It is REPORTED and **not** a
criterion: the pre-commitment names five, and this is a ceiling on one of them,
not a sixth bar. A `mine_blind` FAIL at or below the self-agreement number is
not evidence about the mine.

**What the paired placebo does and does not bound.** Because there is ONE shared
navigator (M2) and its corpus carries both mine colours, the placebo swap
exchanges two colours the navigator has each seen as a mine. It bounds a policy
keyed on THIS colour; it cannot bound one keyed on "either mine colour".
Mine-blindness proper rests on the construction — BFS walks through mines, so no
label can express avoidance — and this pass is the check on that construction,
not a proof independent of it. Stated here so the gate is never read as more
than it is.

The rule this replaces — `mine_step_rate` within ±0.05 of the BFS path's own
rate — is REPORTED, ungated. Measured on the n=9 eval distribution, one BFS path
lands on 1.016 mines per episode with sd 1.205, so at 256 mazes the standard
error of that mean is 0.075 and a ±0.05 band is 0.66 sd: a genuinely blind
navigator would fail such a gate about half the time, and a navigator that
mildly avoids mines and mildly errs would pass it.

## The pre-commitment

**If the shared navigator fails this gate, the branch STOPS.** No warm start, no
cells, no organisms — D, A, B, C do not exist in this world. The remaining door
is thinking mode: the same maze read by a checkpoint allowed to reason before it
answers, which is A8's ladder continued by other means and is a DIFFERENT WORLD
with its own Stage 0 and its own pre-registration. Taking it is a new document,
not a continuation of this one.

If it passes, what it licenses is exactly one thing: the four cells may be built
on top of it. Every finding of the ORG-B critique that this section does not
answer — B1's remark action space, B2's two-term objective, B3's held-out
detector, B4's `min_mines_on_shortest` and the within-bin read, M5's identical
warm start, M7's paired bands, M8's per-class arithmetic, M9's diversity gate,
M10's ledger, PR-3's pre-flight gradient check — is still open and still binding,
and none of it is pre-registered here.
