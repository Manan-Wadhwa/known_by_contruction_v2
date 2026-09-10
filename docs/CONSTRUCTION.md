# Construction — how everything is built

Written 2026-09-09 by an agent reading the tree at `aa0158a`. This file says
what each part of MZ01 *is*, in the code, with the file and function that
implements it beside every claim. It is not a rationale document: the
rationale is `SPEC.md` (the pre-commitment and its amendments), the three
`PLAN-*.md` pre-registrations, and `AGENT-NOTES.md` (deviations D1–D37). It is
not a results document either; results are in `RESULTS_LOG.md` and in the dated
lab notes. Where a document and the code disagree, the code is described here
and the disagreement is named in place. Every constant quoted is a line in a
file, not a recollection.

Paths are relative to the repository root. `MZ` below abbreviates
`experiments/v2/MZ01_maze_world/`.

**Extended 2026-09-09 in the evening, at `fa39ee5`**: §5.7 (the fourteenth pass
as it was actually run, and what D-21 closes), §5.8 (the next recipe, marked
design-only), §7.5 (the coverage read), the two new paragraphs at the end of
§10, and the six coverage rows in §11.

---

## 0. The tree — which modules are MZ01's, and what the rest is

### 0.1 The module map

`src/calibration/` holds thirteen modules and an `__init__.py`; MZ01 does not
run all of them. Every row below was checked by grep over `src`, `scripts`,
`experiments` and `tests`. "The drivers" are `scripts/mz01_pilot.py`,
`mz01_navigator_driver.py`, `mz01_orgb_driver.py` and `mz01_explore_trial.py`,
with `experiments/v2/MZ01_maze_world/run_orgb.py` behind the third.

**Live for MZ01** — a run executes these:

| module | what it does here | who imports it |
| --- | --- | --- |
| `maze2.py` | the world: generator, labels, render, scoring, roll-out, `parse_move`, the price table's primitives | `navigator`, `maze_orgb`, `maze_rl`, and every driver |
| `maze_orgb.py` | Stages 1–3: the warm start, `train_orgb_cell`, both pre-flights, `measure_organism`, the ledger and pass rules | `run_orgb.py`, `mz01_orgb_driver.py`, `mz01_explore_trial.py` |
| `navigator.py` | Stage 0′: the corpus, `train_navigator`, the gate, `mine_blindness` | `mz01_navigator_driver.py` |
| `capture.py` | `MOVE_WORDS`, `move_token_ids`, `random_move_orders` — the move vocabulary and the per-decision word order | `maze2`, `maze_orgb`, `navigator`, `instruments`, `rl`, `maze_rl` |
| `chat.py` | `apply_chat` — one chat-template call site, with `enable_thinking=False` (SPEC C1) | every module and driver that builds a prompt |
| `lora.py` | the hand-written LoRA: `lora_parameters`, `lora_state_dict`, `load_lora_state_dict`, `assert_only_lora_trainable` | `maze_orgb`, `sft`, the trial, both drivers |
| `sft.py` | `train_sft` (with the move anchor) and `log_move_mass` | `navigator`, `maze_orgb` |
| `remarks.py` | the `AVERSIVE` / `AFFECTLESS` / `FILLER` pools | `maze2`, `maze_orgb` |
| `runner.py` | `set_all_seeds`, `git_sha()` and the manifest | `maze_orgb`, every driver |
| `instruments.py` | the out-of-domain battery `instrument_panel` calls, plus `GLYPH_BANK` for the role-glyph valence | `maze_orgb` (§8), `maze2`, `mz01_pilot`, `mz01_navigator_driver`, `run_orgb` |
| `rl.py` | **only its two seed ranges** — `_TRAIN_SEED_RANGE` and `_EVAL_SEED_RANGE`, which fix the train/eval maze populations. Nothing else in it, none of its trainer, is on MZ01's path | `maze2.py:391`, `navigator.py:144`, `maze_orgb.py:149`, `run_orgb.py:184`, `mz01_pilot.py:229`, `mz01_navigator_driver.py:404` |

**Inherited, and not MZ01's**:

| module | what it is | what MZ01 takes from it |
| --- | --- | --- |
| `maze.py` | the **v1 5×5 grid world** — its world, its state bank and its organisms are not used here | `maze2.py:388` imports its tile constants and helpers (`TILE_AGENT`, `TILE_GOLD`, `TILE_MOLD`, `TILE_PATH`, `role_glyphs`, `swap_tile`); `capture.py` and `rl.py` import `TILE_AGENT`. The glyph vocabulary is shared; nothing else is |
| `maze_rl.py` | the **whole-episode Dr.GRPO trainer** written for arm A before `maze_orgb` existed. It is **not** what trains any organism | exactly one function: `maze_orgb.py:147` imports `group_advantages`. Its own `train_maze_a`, sampler and `_Ep` are dead code on MZ01's path, and its `_Ep.done` is `solved or len(moves) >= cap` — it has no farm-stop (§1.3) |

Every organism in this world is trained by `maze_orgb.train_orgb_cell`
(`maze_orgb.py:2291`), through `run_orgb.run` (the driver) or
`scripts/mz01_explore_trial.py` (the trial).

### 0.2 `archive/` — the parent harness's leftovers

`archive/` holds seven directories of result JSONs and run logs from the **v1
repository** at `/home/manan/Projects/files/digital_minds`: `E12_instrument_battery`,
`E13_build_organisms`, `E16_calibrated_loading_map`, `E17_orgb_contingency`,
`HH01_hidden_habit`, `NAR01_narration_recipe`, `NAR02_cotraining` — 88 files in
all, and the E16 and E17 directories are empty. HH01 is the parent's
hidden-habit experiment **in the v1 grid world**, not in this maze.

The whole directory is **gitignored** (`.gitignore`, "Leftover worlds from the
parent harness (not part of MZ01)"). Nothing in `src/`, `scripts/`, `tests/` or
`experiments/` reads it, no number in these documents comes from it, and it is
not part of MZ01. `GLOSSARY.md`'s parent-repository table says where each of
those experiments actually lives.

---

## 1. The world

### 1.1 The grid, and what is on it

`src/calibration/maze2.py`. A maze is a pure function of its seed
(`sample_maze`); a set is drawn from a seed range with a `torch.Generator`
(`generate_set`). Sizes are `SIZES = (7, 9, 11)` (`maze2.py:420`). Each maze is
a square grid of walls and open floor with one start, one goal and `k` mines.
Mines are **walkable** — a mine is a score, not an obstacle — which is the
property that makes the BFS navigator mine-blind by construction (§2.2).

The layout characters used for serialisation and for `canonical_hash` are
`L_WALL="#"`, `L_FLOOR="."`, `L_MINE="*"`, `L_START="S"`, `L_GOAL="G"`; they
are glyph-independent on purpose, so a render-time glyph swap cannot move a
hash or a label.

Two wall families exist, chosen by the route class the maze has to hold:
`_blob_walls` (scattered walls at a density drawn from `BLOB_BAND = (0.10,
0.30)`) and `_lattice_walls` (a spanning-tree maze on the odd-index lattice
with `openings` walls knocked out). `_propose_walls` does **not** see
`detour_cost` — that is D15, and it is what removes the first critic's bin
confound.

Mines are **placed, not scattered** (D8, `maze2.py` docstring): one mine on a
cut cell of the bare layout, then `_grow_detour` closes the currently shortest
mine-free route one mine at a time until the gap is exactly the requested bin,
then `_pad_mines` adds mines that leave that gap alone. Every label is then
recomputed from the finished layout, so the construction cannot mislabel a
maze.

### 1.2 The label dict

`maze2.labels_for`. Every maze carries exactly these 21 keys (read off a
committed result file): `n`, `shortest_len`, `avoid_len`, `best_len`,
`best_cost`, `detour_cost`, `dijkstra_detour`, `detour_slack`,
`mine_on_shortest`, `mines_adjacent_to_shortest`, `mines_on_shortest_dag`,
`route_count`, `route_count_capped`, `shortest_path_count`, `branch_points`,
`dead_ends`, `open_ratio`, `n_mines`, `cap`, `canonical_hash`,
`tied_paths_agree`.

Four deviations from the SPEC's literal definitions are load-bearing:

- **D1.** `detour_cost = avoid_len − shortest_len` (the shortest *mine-free*
  route minus the shortest route), not the SPEC's `best_len − shortest_len`.
  Under the (step 2, mine 12) Dijkstra the literal quantity is 0 on precisely
  the mazes the SPEC wants in bins 6–8. `best_len`, `best_cost` and the literal
  `dijkstra_detour` are all still stored.
- **D2.** The requested `detour_cost` distribution is uniform over
  `DETOUR_BINS = (2, 4, 6, 8)`. The grid graph is bipartite, so the detour is
  always even; the odd bins are empty by geometry.
- **D3.** Generation requires `detour_cost <= shortest_len`, so the mine-free
  route fits inside the cap.
- **D12/D15.** `detour_slack` is a label. After D15 it is
  `shortest_len − detour_cost` deterministically, which is why SPEC C9 records
  A3's "restrict to `detour_slack >= 2`" as vacuous: the restriction either
  keeps a whole bin or drops it.

`DEFAULT_DISTRIBUTION` (`maze2.py:1343`) is the requested joint:
`mine_on_shortest` 0.5; `detour_cost` {2: 0.25, 4: 0.25, 6: 0.25, 8: 0.25};
`route_count` {1: 0.15, 2: 0.15, 3: 0.70} where 3 means "≥ 3". D7 records that
this is the joint meeting the three marginals, not their product — the cell
`(mine_on_shortest=True, detour_cost=2, route_count=1)` is empty by geometry.
D9 records that `generate_set` fills the mined rows first and sizes the
mine-free row from what they achieved, with `bucket_patience` (default 6000
proposals without a new maze) as the dryness detector and every reallocation in
`stats["reallocated"]` / `stats["exhausted_buckets"]`.

**The covariate contract (D15).** `DEFAULT_MATCH` (`maze2.py`) holds, per size,
`shortest_len`, `n_mines`, `open_cells` and `branch_points` across the detour
bins: at n=9 those are 12, 6, 55 and (29, 29). `generate_set` re-checks after
the fact and raises if any bin-8-vs-bin-2 AUC exceeds `COVARIATE_AUC_BAR = 0.6`
(`maze2.py:1908`). SPEC C3 records that the resulting 0.500 AUCs are arithmetic
(the covariates are constant across the mined row) rather than a measurement,
and that **`mines_on_shortest_dag` is the covariate that does move with the
bin**; it is a label, it is reported per bin, and it is **not gated**.

**Three populations carry a `mines_on_shortest_dag` number, and they are not
comparable.** Every quotation of one has to say which:

| number | population | source |
| --- | --- | --- |
| **1.72 / 2.28 / 2.73 / 2.95** across bins 2/4/6/8, AUC(8 vs 2) **0.81** | 600 freshly generated mazes at n=9, split by detour bin | SPEC C3's table |
| **1.16** | the RL **training pool** — the `pool_size` 256 mazes a run's batches are drawn from, on the train seed range | `PLAN-exit-priced-at-cap.md` §7 item 3; `RESULTS-2026-09-08.md` §5's correction |
| **2.39** (quoted as 2.4) | the **held-out** eval set's mean | the same two places; it is the number in the −28 idealised solve, `REPORT-2026-09-08.md` §13 item 4a |

The gap matters where the arithmetic does: a group's advantages are taken
against the **training pool**, so the wall exit was compared with a typical
solve of −13.6 there, not with the held-out idealisation of −28 (§5.2). The mine-free row is
held only to `MINE_FREE_MATCH_KEYS = ("shortest_len", "n_mines")`.

### 1.3 Scoring, the cap, and the episode

Never rendered into text (`maze2.py:424`):

```
SCORE_STEP = -2.0     # every decision, a blocked move included
SCORE_MINE = -10.0    # landing on a mine
SCORE_GOAL = +20.0    # reaching the goal; the episode ends
```

`maze2.score_landings(landings)` is that sum. Path costs for the exact
best-score Dijkstra are separate positive integers: `STEP_COST = 2`,
`MINE_COST = 12`, and `AVOID_THRESHOLD = MINE_COST // STEP_COST = 6` — one mine
is worth six extra steps, so bins 2 and 4 are worth paying, bin 6 is exactly
break-even and bin 8 is not. The bins straddle the threshold on purpose.

`maze2.episode_cap(shortest_len) = 2 * shortest_len` (`maze2.py:1335`), stored
as `labels["cap"]`. At n=9 the generator pins `shortest_len = 12` on every
maze, so `cap` is 24 everywhere.

`maze2.step(maze, pos, move)` returns `(new position, outcome)` with outcome in
{`floor`, `mine`, `goal`, `blocked`, `unparsed`}. `blocked` covers a wall and
the edge of the grid, leaves the position unchanged, and the episode continues.

**D4.** An unparseable reply is its own outcome (`unparsed`, one decision, −2,
no movement) and is reported as `unparsed_rate`. It is **not** folded into
`invalid_rate`, which stays "moves into walls or off-grid, per move". SPEC A11
adds `unparsed_rate <= 0.05` to the Stage-0 gate precisely because a model that
emits nothing parseable scores `invalid_rate == 0.000`.

**The retry line (D5, D28).** Evaluation may issue **one** retry per blocked
move, and a retried move is not itself retried. It is a third chat turn
(`maze2.retry_prompt`): the fixed, affect-free
`"That move is blocked. Reply with exactly one word: …"`. The blocked move has
already happened and already cost −2, so the retried move is the next
environment step. `retries` is a *property* derived from the per-decision
`retried` flags (D28), so a blocked move on a finished episode cannot arm a
retry turn that never runs. **Training uses no retry**; measurement uses one
(`measure_organism` rolls with `retry=True`, `sample_text_episodes` never
retries) — SPEC E11 records that the competence numbers are therefore
comparable to the navigator's gate and not to the training curve.

**The farm-stop, H5.** `maze_orgb.MAX_CONSECUTIVE_BLOCKED = 3`
(`maze_orgb.py:222`) and `maze_orgb._TextEp.apply`: after three *consecutive*
blocked moves the episode ends and is recorded `farm_stopped` (and `capped`).
Its reason is that a blocked move from a mine leaves the agent *on* the mine,
so a wall-bumping policy would farm on-mine turns for free. The guard lives
only in the trainer's text roll-out: `maze2.rollout` has no such rule, and the
thirteenth-pass critic verified that `maze_rl.py` has none either (its `_Ep.done`
is `solved or len(moves) >= cap`).

### 1.4 The prompt, the move words, and `parse_move`

`maze2.maze_prompt` builds the observation. `MOVE_WORDS = ("up", "down",
"left", "right")` (`capture.py`), re-ordered per decision; under `order_seed`
the order is **derived** from (run seed, maze seed, decision index) rather than
drawn from a shared stream (D31), because a shared stream desynchronises paired
passes as soon as one episode diverges. No word for reward, cost, mine, avoid or
danger appears anywhere; a banned-word test covers both formats.

`maze2.parse_move` is `\b(up|down|left|right)\b`, case-insensitive, **earliest**
occurrence (`maze2.py:2365`). D14 records that this misreads "There is a wall on
the right; down." as `right`; the rule is pre-committed and unchanged, and the
defence is measurement — `maze2.bare_move` / `bare_rate` say how much of a
result is exposed to it. The earliest-word rule is also why ` up` (the
space-prefixed twin) parses exactly as `up`, which is the 08-30 token leak
(§5.4).

### 1.5 Glyphs, counterbalancing, and the placebo

`maze2.Glyphs` carries roles, not colours. The **mine pair** is
`MINE_PAIR = (TILE_MOLD, TILE_GOLD)` — blue/purple — counterbalanced by seed
parity through `maze2.glyphs_for_seed(seed, counterbalance=True)`; **the unused
colour of the pair is the placebo**, and `maze2.placebo_glyphs(glyphs)` swaps
the mine tile for it while carrying `on_mine` across unchanged. Geometry, mine
count, wall density and scoring are held exactly; only the colour moves. SPEC C9
records the consequence: after DEC-3 the placebo is a control for the
*behavioural* instrument and has **zero power as a narration control**, because
B's cue is `on_mine`, which the swap does not touch.

The **novel-glyph transfer** pair is `NOVEL_GLYPHS = ("🟥", "🟨")`, run under
both assignments (`NOVEL_PAIRINGS`) so each glyph is seen in each role. SPEC F3
records that H13's count-matched probe is **not built**, so a novel-glyph pass
is uninterpretable as generalisation and the scorer prints that sentence beside
every such number.

**Measured role glyphs.** `MEASURED_ROLES = ("wall", "goal", "on_mine")`;
`FORMAT_ROLES` says which a format actually measures — `("on_mine",)` for ASCII,
because ASCII's wall and goal are fixed characters. `maze2.role_glyph_bank`
widens `instruments.GLYPH_BANK` with eight neutral geometric marks
(`MZ01_EXTRA_ROLE_GLYPHS`), `role_glyph_candidates(tok=...)` keeps only
single-token candidates, and `choose_role_glyphs` **bars as well as ranks**
(D21). D27/C11: the bar is **relative** — each valence is centred on the median
over the full candidate bank measured on the same model, because valence has no
absolute zero (Qwen3-4B runs −0.25..+1.0 on the same bank where Qwen3-14B runs
+3.0..+3.6). `ROLE_VALENCE_BAR = 2.0` on the centred value;
`ROLE_ABSOLUTE_WARN = 2.0` warns and does not refuse. Every pick records
`valence`, `valence_centred` and `valence_median`.

**DEC-3 / D17.** The agent standing on a mine gets its own glyph,
`Glyphs.on_mine`, so the cue for organism B is a **visible fact about the
current state** read one turn after the landing, not adjacency conditioned on a
direction. On the committed navigator that glyph is `●`, measured at centred
valence 0.000; `run_orgb` **refuses** to start a real run without role glyphs
read from the navigator's own result file (SPEC E2 — the stage once rendered
`Glyphs()`'s dataclass default `🟧`, the orange square C2 had rejected at
+3.09).

### 1.6 ASCII at n=9, and why

The emoji Stage-0 pilot failed at every size on both models, below the
uniform-random floor. SPEC A8 had fixed the fallback ladder before the pilot ran:
(i) n=5, (ii) ASCII rendering, (iii) a legend line. C8 ruled out rung (i) —
`match_targets(5)` raises and `detour_cost = 8` occurs in 1 of 4000 proposals at
n=5 — so the ladder was taken at **rung (ii)**, implemented as `fmt="ascii"`
beside `fmt="emoji"` (D29). `DEFAULT_FORMAT` is still `"emoji"` in the module,
so every pre-D29 number stays byte-identical; every MZ01 driver since sets
`ascii`.

Under ASCII the grid is written with `ASCII_WALL="#"`, `ASCII_FLOOR="."`,
`ASCII_AGENT="@"`, `ASCII_GOAL="G"` and is **numbered** — column indices across
the top, row indices down the left, 0-based (`ASCII_ORIGIN = 0`) — and one line
follows it: `You are @ at row r, column c. The goal G is at row r, column c.`
`row` and `column` are the entire vocabulary the format adds. The two
mine-role glyphs stay glyphs in both formats, because a letter for the mine
would name it in the alphabet the instruction is written in.

Each format is a different world with its own Stage 0; result files are
`{model}_{n}_{fmt}.json` and every episode record carries `fmt`. **n=9 is not a
gate result**: no size passed at either format, and what licenses the world is
SPEC A8's rung plus the Stage 0 v2 DECISION of 2026-08-28 to train a navigator
at 14B, ASCII, n=9 (`REPORT-2026-09-08.md` §2, with the verifier's gloss).

### 1.7 The KV-cached decode

`maze2._Decode` is one incremental decode shared by `maze2._greedy` (all
measurement passes) and `maze_orgb._sample_replies` (the RL roll-out): one
prefill of the left-padded batch with `use_cache=True` and `logits_to_keep=1`,
then one single-token forward per step. `use_kv_cache=False` keeps the old path
and is what the suite compares against; cached and uncached logits agree to 1e-5
and the sampled ids agree exactly under one seeded generator (SPEC F8). It costs
nothing scientifically because the loss carries no importance ratio; the
teacher-forced re-scoring stays a full forward with grad.

`maze2._greedy` also takes `stop_on_parse` (D30). `True` is the default and
keeps every pre-D30 number; the navigator's eval and every organism eval pass
`stop_on_parse=False` with `max_new_tokens=48`, because the early stop truncates
a reply to its move word and truncation has manufactured a +0.87 contingency in
the old world.

---

## 2. The navigator (Stage 0 v2)

`src/calibration/navigator.py`, driven by `scripts/mz01_navigator_driver.py`,
scored by `scripts/score_mz01_navigator.py`.

### 2.1 The corpus

`navigator.build_navigator_corpus(n_decisions, *, n, generator, fmt, label,
off_path_share, ...)`. One item per **decision**, not per maze:

```
prompt     = maze_prompt(render(maze, glyphs, position, fmt="ascii"), ...)
completion = the BFS-optimal move word
```

with the four move words in a fresh random order per turn. Mazes come from
`generate_set` on the train seed range (`rl._TRAIN_SEED_RANGE`), deduped by
`canonical_hash` against the eval set.

**On-path / off-path (M3).** `DEFAULT_OFF_PATH_SHARE = 0.30`. On-path cells are
the cells of one BFS shortest path. Off-path cells (`_off_path_cells`) are drawn
uniformly from cells that are reachable from the start, can reach the goal, and
are **off** the shortest-path DAG, labelled with the first move of the BFS
shortest path *from there*. Without the second population the navigator is never
trained to recover from its own error, and C5's absorbing state decides
`invalid_rate` and `single_route_acc` on its own.

**The label rule at a fork (D36).** `label ∈ ("first", "uniform")`,
`DEFAULT_LABEL_RULE = "first"`. `"first"` takes the first move of *one* BFS
shortest path, so at a cell with two optimal moves the corpus insists on one of
them and the trained navigator is a point mass there (move entropy 0.018 on the
warm start it became). `"uniform"` draws the completion uniformly over
`maze2.optimal_moves(maze, cell)` — the moves whose landing cell's BFS distance
to the goal is one less. A rule outside the two is refused by name.

The `uniform` draw comes from a generator *seeded from* the builder's own
(`torch.Generator().manual_seed(generator.initial_seed())`), **not** from the
builder's stream — a deliberate deviation from the PLAN's literal wording, so
that every prompt, position, order and kind is equal item-for-item between the
two corpora and nav48u minus nav48 is the labels and nothing else.

Recorded corpus stats: `label_rule`, `multi_optimal_share`,
`optimal_moves_mean`, `optimal_moves_histogram`, `label_entropy_nats` (the
realised labels' 4-way entropy — a skew here is a policy prior) and
`label_rule_entropy_nats` (the rule's own entropy at a labelled cell, 0 under
`first`). On nav48u these read 0.1973 / 1.199 / 1.3805 / 0.1376.

**Mine-blind by construction.** BFS routes *through* mines, so no label can
express avoidance. The realised share of labelled moves landing on a mine and of
prompts drawn while standing on one are in the manifest (nav48: 0.1115 and
0.1161).

**One navigator, both colours (M2).** Each maze is rendered under its own seed
parity's assignment, so the shared base has seen blue and purple mines in equal
measure (nav48: 1596 / 1516).

### 2.2 Training

`navigator.train_navigator` → `sft.train_sft`. Plain SFT: no move anchor, no
soft target, no remark. LoRA r=16 α=32 on `q_proj`/`v_proj`;
`DEFAULT_MAX_LENGTH = 512` and a longer example **raises** rather than being
truncated (the n=9 ASCII prompt is 322 characters, ~178 tokens for the longest
example). Every prompt is wrapped by `chat.apply_chat` — the same function
`maze2.rollout` uses — and every completion carries the tokenizer's EOS.
Adapters are written at every epoch boundary.

Driver defaults: `MZ01_N_DECISIONS` 24000, `MZ01_EPOCHS` 2, `MZ01_BATCH` 8,
`MZ01_LR` 1e-4. The two committed 48k navigators were launched with 48000/3
explicitly.

### 2.3 Eval and the gate

256 held-out mazes from `rl._EVAL_SEED_RANGE`, greedy, one retry, both
parities, `max_new_tokens = EVAL_MAX_NEW_TOKENS = 48` with `stop_on_parse=False`
and `reply_len_tokens` recorded per decision. Three passes: the mine pass, the
placebo pass, and a 64-maze self-agreement control
(`MZ01_SELF_CONTROL_N`, default 64).

`navigator.NAV_GATE` plus the blindness criterion — five criteria, all of them:

| criterion | direction | bar | where |
| --- | --- | --- | --- |
| `solve_rate` | min | 0.60 | `NAV_GATE` |
| `invalid_rate` | max | 0.10 | `NAV_GATE` |
| `single_route_acc` | min | 0.80 | `NAV_GATE` |
| `unparsed_rate` | max | 0.05 | `NAV_GATE` (SPEC A11) |
| `mine_blind` = paired-placebo `decision_agreement` | min | **0.98** | `BLINDNESS_AGREEMENT_BAR` |

A criterion that is NOT MEASURED fails rather than being skipped.
`single_route_acc_on_dag` / `_off_dag` are **reported, not gated**
(`OFF_DAG_REPORTED_BAR = 0.60` is a reference number, not a criterion).

> **Code vs note.** SPEC's "AMENDED 2026-08-28 after the first navigator run"
> says the blindness bar becomes "self-agreement ceiling − 0.02, both measured
> in the same run". The code did not follow: `navigator.BLINDNESS_AGREEMENT_BAR`
> is 0.98 absolute, and `score_mz01_navigator.py` prints
> `mine_blind 0.9477 >= 0.98 FAIL` for `nav_shared`. The self-agreement number
> is computed and printed beside the verdict, with a `!!` line when the control
> itself is under the bar, but it is not applied. Likewise
> `PLAN-credit-explore-navigator.md` §8 fixes
> `decision_agreement_ex_tie_break >= 0.98` as nav48u's criterion; D37 states
> explicitly that the gate is untouched and the code gates on
> `decision_agreement`. nav48u's **first** run cleared both (0.9863 and 0.9956).
> Its **2026-09-09 retrain does not**: raw `decision_agreement` **0.9634**, which
> the code FAILs, against `decision_agreement_ex_tie_break` **0.9813**, which the
> amendment passes — with the self-agreement control at 0.9717, itself under the
> bar, so the scorer prints that a FAIL there is not evidence about the mine
> (`RESULTS_LOG.md` §2). That is the first run where the two disagree about the
> verdict, and it is undecided. **This document follows the code.**

### 2.4 The blindness rule, and D37's classification

`navigator.mine_blindness(records_real, records_placebo)`. The same 256 mazes
are rolled twice with the **same per-turn word-order stream** (D31 makes that
true of the code, not only of the intention): once with the counterbalanced mine
glyph and once with `maze2.placebo_glyphs`. Under greedy decoding a navigator
that cannot see the mine must make the identical decision in both passes, and
that comparison has no sampling noise.

`decision_agreement` is computed per episode over paired decision indices with
the **longer** episode as the denominator. `BLINDNESS_TV_BAR = 0.02` on the
total-variation distance between the two 4-way move distributions is reported
beside it (and is implied by the 0.98 bar when the passes have equal decision
counts). `parsed_share` and the agreement restricted to parsed decisions are
reported, because two unparseable replies "agree".

**The self-agreement control** rolls 64 of the same mazes a third time — same
glyphs, same orders, a **different eval batch size** — so any disagreement is
bf16 arithmetic rather than perception. It is reported and is **not** a
criterion. Measured: nav48 1.0000 over 784 decisions; nav_shared 0.9623;
nav48u run 1 0.9836; nav48u's 2026-09-09 retrain 0.9717 over 814 decisions.
Three of those four sit **below the 0.98 blindness bar the gate applies**, which
is what the scorer's `!!` line is for: where the control is under the bar, a
`mine_blind` FAIL is not evidence about the mine.

**D37: where the divergences happened.** Each first divergence records the
shared `cell`, `n_optimal` there, `both_optimal`, whether each pass's move lands
on a mine, and `tie_break` = both moves optimal **and** both landing the same
way on a mine. Aggregates: `n_diverged_at_fork`, `n_diverged_tie_break`,
`n_diverged_unplaced`, `decision_agreement_ex_tie_break` (the same agreement
with the tie-break **episodes** dropped from both sides, because divergence is
absorbing), `n_fork_decisions`, `fork_decision_share`. All reported, none
gated. A fork whose two optimal moves *differ* in landing on a mine is **not** a
tie-break — preferring the dry one is precisely the avoidance M4 hunts.
`score_mz01_navigator.py` recomputes all of it stdlib-only with its own BFS.

**What the paired placebo does and does not bound.** With one shared navigator
whose corpus carries both mine colours, the swap exchanges two colours the
navigator has each seen as a mine. It bounds a policy keyed on *this* colour; it
cannot bound one keyed on "either mine colour". Mine-blindness proper rests on
the construction; this pass is a check on it.

**Reported, ungated:** the rule this replaced — `mine_step_rate` against the BFS
path's own rate (0.973 ± 0.073 on the n=9 eval set) — because at 256 mazes a
±0.05 band is 0.66 sd and a blind navigator would fail it about half the time.

---

## 3. The warm start (M5)

`maze_orgb.build_warm_start_corpus`, `capture_move_anchor`, `train_warm_start`;
orchestrated by `run_orgb.run` and `scripts/mz01_orgb_driver.py`.

**One mixed warm start for all four cells.** The corpus is built from the
navigator's own roll-out records, one item per decision the navigator actually
took:

```
prompt     = the one-turn prompt rebuilt from (layout, position, order)
completion = "<move>"  or  "<move>. <remark>"
```

The move is the navigator's own, so the stage teaches vocabulary and placement
and nothing about where to walk. `WARM_START_REMARK_SHARE = 0.40`
(`maze_orgb.py:270`) of turns carry a remark, drawn **independently of the
mine**, half from `remarks.AVERSIVE` and half from `remarks.AFFECTLESS`
(`FILLER` is unused here — a deviation from the critique's PR-2, which asked for
40/40/20). Retried decisions are skipped (C6/H11), as are unparsed ones. The
builder reports the realised placement, the pool split and the on-mine share.

Why one mixed stage rather than two: with D/A on the affectless pool and B/C on
the aversive one, every B-vs-D contrast would differ in two things. One stage
also gives B's RL aversive strings that already carry probability mass, and
makes the affectless contingency a free within-organism placebo at measurement
time.

**The move anchor.** `capture_move_anchor` captures the navigator's own 4-way
move distribution per warm-start prompt; `train_warm_start` passes it to
`sft.train_sft(move_anchor=(cols, probs), anchor_coef=…)` so the move
distribution is held where the remark is being taught. `anchor_coef` defaults to
1.0 (`run_orgb.DEFAULT_CONFIG["warm_start_anchor_coef"]`). The completion puts
the **move first**, so the move-token position is well defined for every row and
the anchor, the move-mass term and every later readout act at the same place.
Every prompt goes through `chat.apply_chat` and every completion carries EOS; an
example longer than `max_length=512` raises.

**Post-stage check.** `warm_start_corpus_contingency(items)` must show
`|rate(on-mine) − rate(other)|` inside `WARM_START_CONTINGENCY_BAND = 0.05` on
both detectors, and the **sampled** contingency is read again in the pre-flight.

**PR-3, the remark-term pre-flight** (`maze_orgb.preflight_gradient_check`),
before any cell trains, on the warm-started model:
`nonzero_yell_group_share >= PREFLIGHT_GROUP_SHARE_BAR = 0.90` **and**
`within_group_yell_sd >= PREFLIGHT_YELL_SD_BAR = 0.20`. Failing either, the run
writes NOT-MEASURED and **stops** without training. Reported beside them,
ungated: the remark **placement** rate against `PREFLIGHT_RATE_BAND = (0.30,
0.50)`, the complaint rate (about half the placement by construction), the
sampled contingency and the within-group return sd. SPEC E9 records that the
group-share criterion is expected to sit ~0.01 above its bar and pre-registers
the remediation (raise `WARM_START_REMARK_SHARE` toward 0.8 and re-run the one
warm start; never lower the bar). Twelfth-pass critic C1: the check now takes
`move_mask` and records it, because every number it reports is a property of the
sampler.

**One warm start per base.** `MZ01_OUT` makes the results directory a knob
because M5's "one warm start" is one per *navigator*: nav48u's warm start goes
to `results/orgb_nav48u/` and must never overwrite `results/orgb/warm_start.pt`,
which every committed cell-A row stands on. `MZ01_WARM_START_ONLY=1` builds that
one file, runs PR-3 and the exploration gate on it, writes the payload and
**stops**; it refuses by name if the target file exists or if `MZ01_WARM_START`
also names one to reuse.

---

## 4. The organisms

### 4.1 The 2×2

`maze_orgb.CELL_SWITCHES` (`maze_orgb.py:185`):

| cell | `mine_cost` (the score feed) | `yell` (the remark feed) |
| --- | --- | --- |
| D | off | off |
| A | **on** | off |
| B | off | **on** |
| C | **on** | **on** |

`mine_cost` is the mine's −10 in the training **return**; `yell` is the remark
term's weight. **D runs the RL** — it is not the warm start left alone. The
reason is in the code comment: HH01 measured the probe and I7 separating
{HA,HN} from {SA,SN} just as hard on seeds where the RL installed nothing, so
"what those two instruments read is 'this model was RL-trained'". Four cells of
one trainer removes that confound by construction.

SPEC E11 records that with the score feed off and the yell term confined to
remark tokens, **cell B's move objective is cell D's exactly**, so policy
invariance is close to true by construction and what the invariance rows measure
is leakage through the shared LoRA weights.

### 4.2 The controls the paper needs

`docs/planning/PAPER_PLAN.md` §3 lists the control set the maximal version has
to carry: the canary D (per-seed readings committed in advance), a
**disturbance-matched sham**, a **decoy**, **script-only**, **HA-from-sham**
(the lineage control), and the same-form placebo for every instrument axis.

Their status in *this* repository, checked by search:

- **sham, decoy, script-only** exist in the **v1** calibration set (the paper's
  Figure 1: RA/RN the sham, DA/DN the decoy, SA/SN script-only). There is **no
  implementation of any of them in this tree** — `grep -rl 'sham\|decoy'` over
  `*.py` returns nothing; the only mentions are in `docs/planning/*.md` and
  `docs/OBJECTIVES.md`.
- **HA-from-sham** is planned and built nowhere: `PAPER_PLAN.md` §3 and its
  risk table name it as the control that would separate adapter lineage from
  the habit, and §4 costs it at ~8 GPU-h.
- **The disturbance-matched sham** is likewise a plan item (`PAPER_PLAN.md` §4
  row 3), with its own failure branch if the band cannot be met.

What *is* built here as a control is narrower and is listed in
`RESEARCH_PLAN.md` §5: the on-policy arm `onpolicy_g6` beside every ε arm, the
reference row, the placebo glyph pass, the unmasked held-out read with
`first_token_class`, bit-identity for every knob, reads inside the arm, and both
pre-flights under the arm's own sampler.

---

## 5. The RL recipe

`maze_orgb.train_orgb_cell` (`maze_orgb.py:2291`) is the whole of it; the
objective is written out in its docstring and assembled at `maze_orgb.py:2838`.

### 5.1 The objective

```
loss = -[ Σ_d  A_move(d) · Σ log π(move tokens of d)
        + w · Σ_d  A_yell(d) · Σ log π(remark tokens of d) ] / (number of decisions)
       - entropy_coef · H          + move_mass_coef · (-log move_mass)
```

- **Dr.GRPO group advantages.** `maze_rl.group_advantages(returns, groups)` is
  the return minus the mean of its own **group** (one maze's episodes), with
  **no division by the group standard deviation**. Every group's advantages sum
  to zero by construction.
- **The per-decision normaliser.** The sum runs over decisions and is divided by
  `len(decisions)`, not by the episode count (H5): per-episode normalisation
  would give a farmed 24-decision episode four times the gradient weight of a
  6-decision one.
- **The yell term.** `maze_orgb.yell_advantages(yells, keys)` with
  `keys = (group, on_mine)`, so the baseline is per (maze, class) inside the
  group. `YELL_WEIGHT = 10.0`, pre-registered before any run; `MZ01_YELL_W`
  overrides it and lands in the manifest as a deviation.
- **Disjoint token sets (B2).** `maze_orgb._split_reply_tokens` cuts at
  `_MOVE_RE.search(text).end()` — the same character offset `reply_spans` cuts
  at — so the token split and the text split are one split by construction. E1
  records what the old rule ("the shortest prefix that parses a move") did:
  `"upset, go down."` split at k=2 and the token emitting the acted move fell
  into the remark set. An unparsed reply has no remark span at all (every token
  is a move token, yell 0, H10). The EOS id **is** appended when a reply ended by
  emitting one (F6), and decodes to the empty string, so it always lands in the
  remark set — which is what stops total silence being a zero-gradient local
  optimum of the yell term.
- **Clipping.** `torch.nn.utils.clip_grad_norm_(params, max_grad_norm)` with
  `max_grad_norm = 1.0`. The `‖g‖` in every trace and table is the **pre-clip**
  norm. A step whose clipped norm is at or below `zero_grad_tol = 1e-8` does not
  call `optimiser.step()` and increments `zero_signal`.
- **LoRA.** r=16, α=32, on `["q_proj", "v_proj"]`
  (`run_orgb.DEFAULT_CONFIG["lora"]`), AdamW at `lr = 1e-4`,
  `weight_decay = 0`. `assert_only_lora_trainable` raises if anything else is
  trainable.

### 5.2 The return, and the exit priced at the cap (D33)

`maze_orgb.episode_return(landings, *, mine_cost=True, cap=None, solved=None)`
(`maze_orgb.py:685`) is the **training** return; `maze2.score_landings` is the
**measured** episode score and is unchanged on every episode. They are not the
same number, and that difference is the thirteenth pass.

```
unsolved:  SCORE_STEP·cap + SCORE_MINE·mines(landings)   [+ the mine_cost switch]
solved:    score_landings(landings)                       [unchanged, to the bit]
```

`mine_cost=False` subtracts the mine term rather than re-scoring, so the two
readings cannot drift apart. `cap=None` is the pre-thirteenth rule exactly.
`solved` is **checked** against `"goal" in landings` and raises on a
disagreement. `record["return_full"]` is always the mine-priced reading, so a
D or B cell's degeneracy can be read on both.

Why: on the closed-action-space rerun the action space held and the policy
collapsed anyway, because H5's three-bump exit cost about −6 against about −13.6
for the training pool's typical solve; under a group-relative advantage the wall
was the best episode in every group. Charging every unsolved episode at what
continuing would have cost makes no early end cheaper than playing on. It is not
a penalty on top: an episode that ran to the cap pays exactly what it paid
before. Rejected alternatives, recorded: raising `SCORE_GOAL` (would need G ≳ 80
and would flatten the mine price the graded read is built on) and removing the
farm-stop (leaves H5's farming open and moves the exit to the cap at the same
price).

`history["exit_return_share"]` is the share of a step's episodes charged that
way, counted off `record["return_charged_at_cap"]`. The thirteenth-pass critic
records that it is identically `1 − solve_rate`; `farm_stopped_rate` is the
measure of how much of a trace the rule actually **changed**.

### 5.3 Where the return is spent — the three credit rules (D34)

`maze_orgb.decision_rewards` and `maze_orgb.decision_advantages`
(`maze_orgb.py:737`, `:793`). Per-decision reward:

```
r_t = SCORE_STEP                             every decision (a blocked move included)
    + SCORE_MINE   if the landing is a mine, subject to the cell's mine_cost switch
    + SCORE_GOAL   if the landing is the goal
    + SCORE_STEP·(cap − len)  on the LAST decision of an unsolved episode
```

`Σ_t r_t == episode_return` for every episode and every switch, **checked on
every episode of every step** in `decision_advantages` (it raises by name), not
only in the suite. The exit charge lands on the last decision because it is the
price of having stopped, and that is where the episode stopped.

| `credit` | advantage |
| --- | --- |
| `episode` (default) | `group_advantages` on the episode returns, broadcast to every decision of the episode |
| `togo` | `A_t = G_t − b_t`, `G_t = Σ_{k ≥ t} r_k` (undiscounted) |
| `immediate` (SPEC 3A) | `A_t = r_t − b_t` |

`b_t` is the mean over the episodes of the **same group that have a decision at
index t**. **An index only one episode of the group reaches gets exactly
`A_t = 0`**: its own value minus itself is not a baseline, and the raw `G_t`
there is a large negative number on every such decision whatever the episode
did — a standing push toward "never be the long episode", which is the early
exit this programme has installed by accident twice.

What the step index assumes, stated in the code: `b_t` compares the t-th
decisions of a group's episodes, not decisions taken in the same **state**. At
the warm start those coincide; once episodes diverge it is a variance reduction
with a bias that grows with the divergence. The A+B critic swept it: on
constructed pairs the trajectory-level sign-flip count is 0 under `episode`,
224 under `togo` and 440 under `immediate`, all at d = 1..4; on realistic
six-episode groups the pairwise flip rate rises from ~1 % to 5–7 % while the
blame does concentrate on the mine decision (mean advantage −3.1 there against
+0.3 elsewhere under `togo`, versus −3.0 / −0.3 under `episode`). That is the
trade the pass makes, and it is why R4 (the control at the same credit) is run.

### 5.4 The closed action space (D32)

Two halves, both in `maze_orgb`:

- **The sampler.** `_sample_replies(..., move_cols=…)`: at decode `step == 0`
  only, and only for rows the ε draw has not forced, every logit outside the
  four **bare** move ids (`capture.move_token_ids`, i.e. `_move_cols`) is set to
  `-inf` **before** the temperature division and the softmax — additively, so
  the four kept logits are bitwise what the open sampler would have divided.
  Steps 1.. are untouched. A forced first id outside `cols` raises by name.
- **The loss.** `_rescore(..., move_mask=True)` puts the **restricted** 4-way
  log-softmax at the action taken into `tok_logp[:, 0]`, so the trainer's own
  expression `(tok_logp * m_mask).sum(1)` is unchanged and the move term is
  `mlogp[action] + (any further move tokens under the full vocabulary)`. The
  entropy bonus reads the same restricted distribution, and `log_mass` is the
  probe `move_mass` is read off: emission, gradient, entropy and probe are one
  object. A first id **not** in `cols` falls back to the full-vocab term, which
  keeps `move_mask=0` reproducing 08-30 on any tokenizer and keeps a scripted
  row scorable.

`move_mass_coef` therefore defaults to **0.0** (it was 0.03 through 08-30) and
`move_mass` becomes a diagnostic — the number that says whether the **unmasked**
held-out read will see the trained policy. `history["first_token_class"]`
records the shares {bare, spaced, capitalised, other} of sampled first tokens
**by id** (`move_first_token_classes`, `first_token_class`); under the mask
`bare` is 1.0 at training time by construction. The held-out read is
deliberately unmasked and its four shares are read from the reply **text**
(`_BARE_FIRST_RE`, anchored at a word boundary after the twelfth-pass critic's
C4); every row's `rule` says which read produced it.

The rejected alternative — eight columns, bare plus spaced — closes one leak and
leaves the capitalised twins and every other one open, and makes the entropy
bonus an 8-way entropy over four actions.

### 5.5 Exploration

`maze_orgb.exploration_plan(mode, ...)` is the **only** place a mode name
becomes numbers, so the gate and the trainer cannot disagree.

| `MZ01_EXPLORE` | what it moves | knobs |
| --- | --- | --- |
| `thermostat_fixed_target` (default) | nothing — the pre-registered recipe | — |
| `relative` | the entropy **target**: `max(ENTROPY_REL · measured, ENTROPY_MIN)` | `ENTROPY_REL = 0.85`, `ENTROPY_MIN = 0.3` |
| `fixed` | the **controller**: off, coefficient held | `ENTROPY_COEF_FIXED = 0.1` |
| `temperature` | the **sampler**: draw at T, score at 1.0 | `SAMPLE_TEMPERATURE = 1.5` |
| `epsilon` | the **action**: replace the move word | `EXPLORE_EPSILON_RATE = 0.1`, `EPSILON_WHERE_ALL` |

**The thermostat.** With a target, `log_coef += entropy_lr · (target − measured
entropy)` each step, clamped to `ENTROPY_COEF_BOUNDS = (1e-4, 1.0)`;
`entropy_coef` starts at 0.01 and `entropy_target` at 0.7 in
`DEFAULT_CONFIG`. The failure this documents: the warm start's move-column
entropy is 0.0176 against a ceiling of ln 4 = 1.3863, so the error is positive
every step and the coefficient multiplies by e^0.0345 until it pins at 1.0
around step 133. `MZ01_THERMOSTAT_OFF=1` holds the coefficient still under a
mode that would otherwise run the controller; `MZ01_ENTROPY_TARGET=<x>` pins the
target (required for any `relative` run spanning two leases, and refused under a
mode that runs no thermostat).

**ε-greedy replacement.** `maze_orgb._epsilon_pick(valid, hit, u, epsilon)` —
two draws per active episode from the run's own generator; with probability ε
the sampled move word is replaced by a **uniformly drawn valid** move, forced
into the reply as `force_prefix` (`move_word_token_ids`, the ids whose *decode*
is the word, round-trip checked — not `capture.move_token_ids`, which is first
tokens for logit columns). The forced id is what is appended *and fed back*, so
the remark is conditioned on the move actually taken. If no move is valid the
sample is kept.

**`epsilon_where` (D35).** `maze_orgb.epsilon_pool(maze, pos, where)`:

- `all` (default) — every decision eligible, the pool is every valid move, in
  `maze.neighbours` order, which is the tenth pass's draw bit for bit.
- `near_mine` — eligible only where at least one valid neighbour **is a mine**;
  the pool is the valid moves that do **not** land on one, falling back to all
  valid moves where every one does (and the decision still counts as eligible).

The generator stream is untouched either way — the two floats are drawn for
every active episode and an ineligible decision simply does not apply them.
`history["epsilon_eligible_fraction"]` is recorded per step beside
`epsilon_replaced_fraction`, and `plan_epsilon_where` reads `where` back to
`all` on any arm whose rate is 0, so a control cannot record a targeting rule it
never applied.

**The off-policy weight is measured, never applied.** Both `epsilon` and
`temperature` are off-policy. `history["log_importance_ratio"]` carries the mean
per-decision `log π_score(a) − log π_sample(a)` on the temperature arm (the
sampler refuses to compute it together with `force_prefix`, because a forced
token came from neither distribution); `history["epsilon_replaced_fraction"]` is
the honest analogue on the ε arm. Three reasons are recorded in
`train_orgb_cell`'s IMPORTANCE WEIGHT paragraph: the "identically 1" argument
does not apply off-policy; the ratio is a product over ~16 tokens whose log is
tens of nats and whose variance dwarfs the advantage; and truncating it
introduces a clip constant nothing here has tested. `near_mine` makes the bias
**larger**, not smaller — the behaviour policy is a state-dependent mixture —
and the A+B critic records that the eligible fraction *sizes* the bias but
cannot invert it (that would need the per-decision pool size, which is
recommended for a later pass and not built).

### 5.6 Checkpoints and resume (R1)

`maze_orgb.save_cell_checkpoint` / `load_cell_checkpoint`,
`cell_checkpoint_path(work_dir, cell, seed, prefix="orgb_ckpt")`. `CKPT_EVERY =
50` steps (`MZ01_CKPT_EVERY`; 0 turns it off). A checkpoint carries the adapter,
the AdamW moments, the step counter, the entropy coefficient, the RNG streams
(the run's own generator plus python/numpy/torch/cuda) and the trace so far,
written atomically; one per (cell, seed), overwritten, ~126 MB at 14B.

**`CKPT_SCHEMA = 6`** (`maze_orgb.py:507`). The schema history is in the code
comment: 2 added the degeneracy and importance series; 3 added `epsilon`; 4
added `move_mask` and `first_token_class`; 5 added `exit_at_cap` and
`exit_return_share`; 6 added `credit` and `epsilon_where` (and
`adv_abs_mean`, `adv_nonzero_share`, `epsilon_eligible_fraction`). A checkpoint
from an older schema, or one whose fingerprint disagrees on the arm,
temperature, epsilon, decode width, action space, move-mass coefficient, exit
price or credit rule, is **refused by name**.

**Resume at the shard level.** `run_orgb.RESUME_RECIPE_KEYS` lists the config
fields a prior row must agree on before it is carried into a new launch: steps,
n, batch, group, lr, temperature, max_new_tokens, entropy coef and target,
move_mass_coef, chunk, pool_size, fmt, explore_mode, sample_temperature,
epsilon, `move_mask`, `exit_at_cap`, `credit`, `epsilon_where`.

**`RESUME_ABSENT_MEANS`** (`run_orgb.py:452`) is the thirteenth-pass critic's
fix: for four of those keys the **absence** of the field is evidence, not a
missing reading. A row whose `training.config` exists and does not name
`move_mask` was written by code that had no mask, so it trained with the space
**open**; one that does not name `exit_at_cap` charged an early exit at the
decisions it took. The assumed values are `{"move_mask": False, "exit_at_cap":
False, "credit": "episode", "epsilon_where": "all"}` — the last two never refuse
a row on their own, because the pre-pass value *is* the current default. A row
with no `training.config` at all claims nothing and is skipped. The scorer
reasons the same way (`score_mz01_explore._action_space`), which is what the
critic found the two halves of the codebase disagreeing about.

A resumed cell is bit-identical to an uninterrupted one on CPU fp32 (asserted in
the suite). On CUDA bf16 the transferable claim is narrower and is carried into
every result file as `history["checkpoint"]["bit_exact_scope"]`: every element
of the loop's state is restored, so the resumed cell is the same recipe
continued from the same state — not the same floats.

### 5.7 The fourteenth pass as it was actually run (2026-09-09)

The fourteenth pass built three things (`PLAN-credit-explore-navigator.md`,
D-12): per-decision credit (§5.3's `togo` and `immediate`), a targeted ε
placement (§5.5's `epsilon_where=near_mine`), and a navigator whose labels at a
fork are uniform rather than `first` (§2, D36). This section records which of
them a box actually executed under the priced return, because two of the three
did not run in the form the plan names.

**Credit.** Only `togo` was run. Two arms, both `onpolicy_g6` on cell A with the
raw-mass regulariser at 0.3 and 160 steps, both read at 40/80/120/160 on the 128
held-out mazes:

| label | navigator | credit | @120 | @160 | verdict | file |
| --- | --- | --- | --- | --- | --- | --- |
| `R4_togo_ctrl_s0` | nav48 | `togo` | 0.813 / 0.83 RANDOMISED (solve clause) | 0.797 / 0.93 | **DEAD** | `MZ/results/R4_togo_ctrl_s0/onpolicy_g6.json` |
| `R5_nav48u_ctrl_s0` | nav48u, its own warm start | `togo` | 0.906 / 0.87 ALIVE | 0.828 / 0.92 | **DEAD** | `MZ/results/R5_nav48u_ctrl_s0/onpolicy_g6.json` |

(cells are held-out solve / mine ratio to that run's reference row.) `immediate`
— the SPEC's 3A rule, the one credit mode that never sums the per-step −2 and
−10 into a single return before a baseline sees them — **has been run only with
ε** and never under the priced return. Both `togo` arms lose solve without
buying avoidance; R5's zero-advantage share starts at 0.52 against nav48's
0.77, so the uniform-label navigator does widen the gradient (§5.3's baseline
has something to compare) and still does not produce the habit
(`PLAN-exit-priced-at-cap.md` §8, 09:10Z).

**ε placement.** The targeted draw was **never run**. No result file under
`MZ/results/` records `epsilon_where = near_mine`, and neither
`epsilon_030_near` nor `epsilon_045_near` has produced an arm file: D-14 moved
the targeted arm to ε 0.45 for budget parity, D-15 then withdrew the whole ε
family for the priced return, and the 06:40Z amendment of
`PLAN-exit-priced-at-cap.md` §8 struck R1–R3 from the queue. R1
(`epsilon_010`) and R2 (`epsilon_030_near`) were nevertheless launched at
07:34Z against that withdrawal and killed at 07:35Z inside their first training
step; `MZ/results/R1_eps010_s0/` and `R2_eps030near_s0/` hold a log and nothing
else. Both arms stay defined in the registry (`mz01_explore_trial.py:324`,
`:333`) and unread.

**The three registry arms run instead** (11:51–11:52Z, seed 0, otherwise the
unchanged on-policy priced recipe, their control the seed-0 on-policy run
already read DEAD at ratio 0.91):

| label | arm | @160 | verdict | file |
| --- | --- | --- | --- | --- |
| `temp15_ctrl_s0` | `temperature_t15` (draw at T = 1.5, score at 1.0) | 0.914 / 0.89 | **DEAD** | `MZ/results/temp15_ctrl_s0/temperature_t15.json` |
| `temp20_ctrl_s0` | `temperature_t20` (T = 2.0) | 0.836 / 0.75 | **RANDOMISED** by the solve clause by 0.009 | `MZ/results/temp20_ctrl_s0/temperature_t20.json` |
| `relg12_ctrl_s0` | `05_relative_g12` (relative controller, group 12) | *lost with the lease* | three reads, none AVOIDS | `MZ/results/relg12_ctrl_s0/05_relative_g12.json` |

`temp20_ctrl_s0` is the day's nearest miss: the mine ratio lands exactly on
`MINE_RATIO_BAR` and the solve falls 0.109 against the reference's 0.9453,
0.009 past `SOLVE_BAND`. Both temperature arms raise the first ten steps'
zero-advantage share (0.55 and 0.37 against the control's 0.77) and the policy
collapses under the hot draw faster than the control does — the ε family's
signature (D-15) under a different sampler.

**One launch-time fact this pass exposed.** `MZ01_GROUP` does **not** override
an arm's own group: `mz01_explore_trial.py:604` and `:747` both read
`group = arm["group"] or GROUP`, so an arm whose registry entry names a group
keeps it and the environment variable is read only by the arms whose entry is
`None`. A run launched on box 3 at 11:51Z as `onpolicy_g6` with `MZ01_GROUP=12`
was therefore a plain seed-0 rerun at group 6; it was killed at 11:52Z, unread,
and its directory removed. The group-12 question is asked by the registry arm
`05_relative_g12`, which is the only entry carrying `"group": 12`.

**Why D-21 closes on-policy sampling on this navigator.** The tally over the
whole on-policy priced recipe — nav48, mask on, exit priced, mass 0.1, episode
credit, 160 steps — is seven runs (seeds 0–5 and seed 1 twice), twenty-seven valid
reads, one AVOIDS, and that one not reproduced when its own seed was re-run
(`mass_ctrl_s1_rep`, which read a mine ratio of 1.00 at the step where the 09-08
run read 0.63). The coverage read of §7.5 then supplied the mechanism rather
than another verdict: at the sampler's own temperature the warm-started
navigator produces a mine-free solution on 3 of 128 mined mazes, four episodes
in 8 192. Six rollouts per maze from that policy cannot put a detour in a group,
so the group baseline of §5.1 compares six copies of one path and the price of
§5.2 has nothing to price. D-21 (`docs/DECISIONS.md`) closes the recipe on that
ground and not on the verdicts: **a recipe that samples on-policy alone is not
run again on this navigator.** Nothing in §5.1–§5.6 is retracted by it — the
objective, the return, the credit rules and the closed action space are what
they were; what is withdrawn is the claim that running them on this navigator
tests the price.

### 5.8 The next recipe — item 1 is BUILT; items 2–4 are DESIGN ONLY

**Amended 2026-09-10 (fifteenth pass).** Item 1, the forced counterfactual at
geometric forks, **is now implemented and tested**; it is off by default and
nothing that does not set its switch is changed by it. Where it lives:

* `src/calibration/maze_orgb.py` — `episode_forks` (the geometric fork rule),
  `plan_forks` (the uniform draw on the run's own generator),
  `_launch_continuations` / `sample_fork_continuations` (the forced first move
  and the on-policy continuation, priced by `episode_return`),
  `fork_advantages` / `fork_terms` (`A(a) = R(a) − mean R` and the step's read),
  `_fork_move_logp` (`log π(a | fork prompt)` over the four bare move ids), and
  the `fork_cf` / `fork_per_ep` / `fork_coef` / `fork_std` / `fork_temperature`
  arguments of `train_orgb_cell`.
* `scripts/mz01_explore_trial.py` — the arm `fork_cf_g6` and the environment
  switches `MZ01_FORK_CF` (default off), `MZ01_FORK_PER_EP` (1),
  `MZ01_FORK_COEF` (1.0), `MZ01_FORK_STD` (0), `MZ01_FORK_TEMP` (1.0).
* `tests/test_maze_orgb.py` — the price margin, the construction test (a maze
  whose mine accessors raise), the budget and its pricing, and the byte-identity
  of the switch-off path against the pre-build trace and adapter.
* The build report, including what the plan left open and what was read
  conservatively: `MZ/BUILD-fork-counterfactual-2026-09-10.md`.
* **The trace, per step** (written only with the switch on): `n_forks`,
  `n_continuations`, `fork_adv_abs_mean`, `fork_share_nonzero`,
  `fork_share_detour_better`, `fork_sample_seconds`; the guards'
  `fork_grad_norm_raw`, `fork_grad_norm`, `fork_mass_mean`
  (`MZ/BUILD-fork-guards-2026-09-10.md`); and the eighteenth pass's
  `fork_p_target_before`, `fork_p_target_after`, `fork_p_target_delta`,
  `fork_p_target_absdelta`, `fork_p_taken_before`, `fork_p_n_scored` and
  `fork_probe_seconds` — the fork DECISION's own probability, before and after
  the optimiser step (the signed change **and** its magnitude, because the
  signed mean cancels across a step's forks; over the forks whose continuations
  did not all tie), which is what makes
  `PLAN-fork-counterfactual.md` §17.3 (a correct advantage applied to a policy
  that has stopped moving) readable off a trace. **No flag was added for it**;
  it reads the policy at the fork prompts and no mine, and nothing reads it
  back (`MZ/BUILD-fork-probability-2026-09-10.md`).

**Items 2, 3 and 4 do not exist in the code.** None of them has a file,
function, flag or environment variable in `src/calibration/` or `scripts/`;
grep for a fork-anchored baseline, a redraw or a standard-deviation normaliser
on the GROUP advantage returns nothing (`MZ01_FORK_STD` divides one fork's
continuations, which is item 1's own knob and not item 4). Where an existing
constant is named below (`SCORE_MINE`, `maze2.py:425`) it is named as the thing
a change *would* move, at its current value. This section is recorded here
because §5.7 closed a recipe and the reader of §5 must not be left thinking the
closed one is still the plan. The rationale and the literature behind each piece are in
`DIAGNOSIS-2026-09-09.md` §2 and §3; what follows is only the shape of the build
and the test each piece must pass before it can be built.

The recipe is four changes made together, in this order of dependence:

1. **Forced counterfactual at geometric forks — BUILT, 2026-09-10.** At a branch
   point on the rollout's own path, force each valid first move once, continue
   on-policy from there, and credit the fork decision with the difference of the
   continuations' mean returns. This is what puts a detour in the sample set by
   construction rather than by luck, which §7.5 says luck will not do. As built:
   one fork per sampled episode, drawn uniformly among the path's forks with the
   run's own generator; a fork is a cell with ≥ 2 valid moves other than the way
   back, and those moves — not the way back — are the ones forced; the
   continuation's budget is the episode cap minus the prefix's steps; the term
   is `−(coef / n_forks) · Σ_a A(a) log π(a | the fork's own prompt)` added to
   the existing loss, and the continuations are **not** trained. Measured on the
   n = 9 pool, 0.61 of a shortest-route episode's decisions are forks and a fork
   has 2.31 valid moves on average, so the continuations cost about 1.8–3.6× the
   episode's own decisions at one fork per episode.
2. **Fork-anchored (anchor-state) baseline.** Per shared cell rather than per
   step index: the advantage at a decision taken in cell *c* is the
   return-to-go from *c* minus the mean return-to-go over the rollouts of the
   same group that also passed through *c*. §5.3's `b_t` compares the t-th
   decisions of a group; this compares decisions taken in the same **state**,
   which is the bias §5.3 already names and does not fix.
3. **Dynamic sampling.** Redraw a maze whose group's six returns are identical,
   up to a budget, and report the redraw rate. §5.7's traces put that share at
   0.70–0.95.
4. **The std normaliser.** Restore GRPO's division of the group advantage by the
   group's return standard deviation, which §5.1's Dr.GRPO objective removes. It
   is one line and it changes what a 2-point spread counts for against a
   39-point one.

**The construction test, which each must pass before it is built.** The paper's
claim is that the price is the organism's only source of mine information. A
change passes only if it uses **(a)** the geometry of the maze — walls, branch
points, distances to the goal — **(b)** the policy's own rollouts, and **(c)**
the priced return, and nothing else. Applied to the four:

| change | geometry | own rollouts | priced return only | passes |
| --- | --- | --- | --- | --- |
| forced counterfactual | forks chosen as **branch points on the path**, never by mine adjacency — as built, `episode_forks` reads `maze.neighbours` (walls and the grid edge) and the episode's own positions, and a test runs it against a maze whose `is_mine`, `mines`, `at` and `rows` all raise | the continuations are the policy's own | the credited difference is a difference of priced returns | yes, **and the condition is now tested**: the only mine reads on the fork path are `maze2.step` naming a landing, `maze2.render` drawing the tile the policy is entitled to see, and `episode_return` pricing it. The eighteenth pass's `fork_p_*` diagnostic reads the POLICY at the fork prompts and the advantages, and no maze at all — asserted with every mine accessor on the maze class raising |
| fork-anchored baseline | the anchor is a cell the rollouts share | the rollouts are the group's | return-to-go is the priced return, summed from that cell | yes |
| dynamic sampling | — | the redraw test reads the group's own returns | the identity test is on returns, nothing else | yes |
| std normaliser | — | the group's own returns | it rescales the priced return's spread | yes |

The rule the table enforces is the one that fails `near_mine`: an ε draw or a
branch **placed by mine adjacency** puts mine position into the recipe through a
channel that is not the return. `epsilon_where = near_mine` (§5.5, D35) is on
the failing side, and no arm using it has produced a read in this tree — no
result file under `MZ/results/` records `near_mine` — so nothing has to be
re-reported; an arm run under it in future does. The same
rule rules out a mine-aware expert, a hint, a prompt prefix walked to the mine,
and a potential shaped on mine distance — the last of which is a denser price
and is admissible only as a **separately named world**, never as a change to
cell A.

Two further pieces are designed and not built, and are not part of the recipe
above: a **price dose** (`SCORE_MINE`, `maze2.py:425`, −10 → −30 as a separately
named cell, with §7.3's price-sign classes recomputed at the new dose; note
`MINE_COST = 12` at `:432` is the *generator's* cost and a different number) and
**self-imitation** on
price-filtered rollouts. Both pass the test — the price is the channel in the
first and the filter in the second.

---

## 6. The exploration trial (R2)

`scripts/mz01_explore_trial.py`, scored by `scripts/score_mz01_explore.py`.
Short arms whose only question is "does this RL produce a policy gradient at
all?". No organism, no row, no ledger, no adapter: every arm restores the same
warm start bit-exactly and is discarded when the next one starts.

### 6.1 The arms registry

An arm is a named recipe, not a command line. `ARMS` (`mz01_explore_trial.py:250`)
— each entry may override the group, the draw temperature, the ε rate, where ε
acts, and whether the thermostat runs; anything it leaves out is the
environment's, and an arm's own knobs beat the environment's.

| arm | cell | mode | group | what it is |
| --- | --- | --- | --- | --- |
| `01_thermostat_g6` | D | `thermostat_fixed_target` | env | the control — today's recipe, bit for bit |
| `02_relative_g6` | D | `relative` | env | the entropy target |
| `03_fixed_g6` | D | `fixed` | env | the controller off |
| `04_temperature_g6` | D | `temperature` | env | drawn hot, scored at 1.0 |
| `05_relative_g12` | D | `relative` | 12 | `02` at twice the group |
| `temperature_t15` | A | `temperature` | 6 | T=1.5, thermostat off |
| `temperature_t20` | A | `temperature` | 6 | T=2.0 — the dose |
| `epsilon_010` | A | `epsilon` | 6 | ε 0.10 |
| `epsilon_020` | A | `epsilon` | 6 | ε 0.20 |
| `epsilon_005` | A | `epsilon` | 6 | ε 0.05 — the low dose |
| `onpolicy_g6` | A | `fixed` | 6 | **the control**: `epsilon_010` minus its one change |
| `epsilon_030_near` | A | `epsilon` | 6 | ε 0.30, `epsilon_where=near_mine` |
| `epsilon_045_near` | A | `epsilon` | 6 | ε 0.45, `near_mine` — **budget parity** |

`DEFAULT_ARMS` is the five D arms when `MZ01_TRIAL_CELL=D` and the four
tenth-pass cell-A arms (`temperature_t15`, `temperature_t20`, `epsilon_010`,
`epsilon_020`) otherwise. `epsilon_005`, `onpolicy_g6`, `epsilon_030_near` and
`epsilon_045_near` are **named selections, never defaults**, so earlier committed
command lines keep meaning what they meant.

`epsilon_045_near` exists because the A+B critic measured the eligible share on
the training pool at **0.22** (0.27 for a wandering policy), not the third the
PLAN assumed: ε 0.30 buys a replacement share of 0.066 against `epsilon_010`'s
0.10, so R2/R6 would be confounded with budget. 0.45 × 0.22 = 0.10.
`epsilon_030_near` stays defined as pre-registered and is not run.

The registry holds both arms; `MZ/README.md`'s arm table, its fourteenth-pass
paragraph and its R2 launch line now say the same thing (0.22 eligible,
`epsilon_045_near` runs, `epsilon_030_near` defined and not run).
`PLAN-credit-explore-navigator.md` §4's table still names `epsilon_030_near` for
R2 and its §9 item 1 — dated before any of R1–R7 launched — replaces it; the §9
amendment is the operative line.

### 6.2 Mid-arm reads, and the reference row

`MZ01_TRIAL_EVAL_AT` (default `40`) is a comma-separated list of RL steps at
which a held-out read is taken — the same `eval_mazes`, the same `measure()`
call, the same `eval_block` — stored as `payload["eval_at"][str(step)]`.
`payload["eval"]` is still the final read and the pick and the verdict come from
it alone. A step at or past `MZ01_TRIAL_STEPS` is dropped out loud. The read
runs inside `train_orgb_cell`'s `on_step` hook and draws from no generator
(`measure_organism` derives its word orders from `order_seed`), so the training
trace is bit-identical with and without one. Its wall clock is kept out of the
run's: `history["on_step_seconds"]` is what the hook spent, and `total_seconds`
/ `seconds_per_step` exclude it (twelfth-pass critic C2 — unsubtracted it put
~2.3 s on every step of an 80-step arm).

`reference.json` is the **warm start's own held-out measurement**, taken before
any arm, on the same 128 mazes. Every cell-A verdict is a difference from it.
It is never a candidate for the pick. Measured on the nav48 warm start it reads
solve 0.9453 / mine per episode 2.2812 at seed 0 and 0.9453 / 2.2891 at seed 1,
identical across leases.

Outputs in `MZ01_TRIAL_OUT`: `{arm}.json` (rewritten atomically after every
step, so an arm killed at step 59 of 60 is still readable), `reference.json`,
`{cell}_{arm}.json` (the follow-up cell), `summary.json`. The status file
`mz01_explore_status.txt` is at the **repo root** and its path is hardcoded, so
two trials on two boxes are kept apart by their output directories, not by a
shard name.

---

## 7. Measurement and the verdict rules

### 7.1 The held-out set and the reads

Eval mazes come from `rl._EVAL_SEED_RANGE`, deduped by `canonical_hash` against
the train set. The organism driver measures on **256** mazes
(`MZ01_EVAL_N`, `DEFAULT_CONFIG["eval_n"] = 256`) with
`eval_batch_size = 32`; the trial measures on **128**
(`MZ01_TRIAL_EVAL_N`) drawn from the same 256, so the reference and every arm
are read on identical mazes and the differences are paired. Greedy, one retry,
`max_new_tokens = 48` with the parse-stop off, word order derived from
`order_seed`.

`maze_orgb.measure_organism` runs the passes the config asks for: the mine pass,
the placebo pass, both novel-glyph pairings, the 50-step relearn with the
adapter restored bit for bit, and the instrument panel. The trial defaults the
placebo and novel passes **off** (`MZ01_NO_PLACEBO`/`MZ01_NO_NOVEL` are read
inverted there) — a trial arm is not an organism.

Every read block carries `solve_rate`, `mine_step_rate`, `unparsed_rate`,
`invalid_rate`, `first_token_class` (with `rule` saying whether it was read by
id or from text), `competence` and `price`. `competence` additionally carries
`optimality`, `single_route_acc` and its on/off-DAG split, `mine_landing_rate`,
`on_mine_state_rate`, `retry_rate`, `bare_rate`, `capped_rate`, `mean_score`,
`distinct_position_share` and `repeat_decision_rate` (D24 — greedy decoding plus
a non-moving blocked move makes one wrong reading an absorbing state, and these
two are what tell a bad policy from a stuck one).

**The three mine rates, with their denominators** (`maze2.competence`, and the
identical lines in `score_mz01_pilot.competence`):

| field | definition | code | scale here |
| --- | --- | --- | --- |
| `mine_step_rate` | mines landed on **÷ episodes** | `maze2.py:3078`, `mines / n` where `n = len(records)`; `score_mz01_pilot.py:264` | ~2.3 on the warm start's held-out read; printed **`mine/ep`** by the trainer (`maze_orgb.py:3012`) and every scorer |
| `mine_landing_rate` | mines landed on **÷ decisions** (per move) | `maze2.py:3079`, `mines / steps`; `score_mz01_pilot.py:265` | ~0.18 on the same read; this is the rate at which organism B's narration cue fires, and the number that says how many complaint-bearing examples a 1536-decision corpus would hold |
| `on_mine_state_rate` | decisions taken **while standing** on a mine ÷ decisions (D17/D18) | `maze2.py`, in the same dict | differs from the landing rate whenever an episode ends on a mine, or bumps a wall from one |

**Every verdict in this programme is on `mine_step_rate`.** The reference row,
the mine ratio, `MINE_RATIO_BAR`, `RELEARN_BAR` and the SPEC's manipulation
check are all per episode. The per-move rate appears beside them in the same
tables — 0.057 against the reference's 0.180 for `mass_ctrl_s1` at 160 — and is
never the bar.

### 7.2 The trial's verdicts

`scripts/score_mz01_explore.py`, one implementation so the trial's pick and the
human's table cannot disagree. Constants at `score_mz01_explore.py:117`:

```
ZERO_ADVANTAGE_BAR = 0.80     MINE_RATIO_BAR = 0.75
SOLVE_DROP_BAR     = 0.10     SOLVE_BAND     = 0.10
UNPARSED_BAR       = 0.05     WINDOW_FRACTION = 5
```

Start and end are means over the **first and last fifth** of the steps, so one
lucky step decides nothing.

**On the training trace (LIVE/DEAD).** DEAD = `zero_advantage_group_fraction`
still over 0.80 at the end; the move term is multiplying by zero. LIVE is its
complement. `zero_advantage_group_fraction` is now read off the per-decision
**advantages** (`zero_advantage_decision_fraction`, tolerance
`ADVANTAGE_TOL = 1e-9`) — "the share of groups in which no decision has a
gradient" — which is the only definition that still means that under `togo`;
under `episode` it is the same number, asserted rather than argued.
`grad_norm` cannot report this, because the entropy bonus and the mass penalty
keep it positive.

**PLAN-exit-priced §7 amendment 1, on the record and not in the scorer.** Under
the priced exit an all-unsolved group with no mine landings has identical
returns (`−2·cap` on every episode, cap 24 everywhere), so a collapsed policy
reads `zero_advantage = 1.0` at the **end** without ever having lacked a
gradient. A row is "the move term never had a gradient" only if its `zadv s` was
also over the bar; a row with low `zadv s` and high `zadv e` is a collapse into
a flat region and is reported as RANDOMISED with a flat tail. The scorer prints
`zadv s` beside `zadv e` for exactly this reading; it does not implement the
two-ended rule itself.

**On cell A, against the reference row.**

| verdict | rule |
| --- | --- |
| `INVALID-READ` | the read's `unparsed_rate` > 0.05 — **checked first**; solve and mine/ep over episodes whose replies do not parse measure the parser. Reported, not scored, never picked. |
| `NO-MEASUREMENT` | a trace but no held-out read (the arm was killed first) |
| `NO-REFERENCE` | no reference row in the directory |
| `RANDOMISED` | held-out solve fell more than 0.10 below the reference's |
| `AVOIDS` | held-out `mine_step_rate` < 0.75 × the reference's **and** solve within 0.10 of it |
| `ALIVE` | a gradient, but not avoidance |

On cell D, `RANDOMISED` is instead "solve fell more than 0.10 against the arm's
own start". Every mid-arm read gets the verdict it would get by these same
rules, from the same function, printed under a `@40 … @end` block; the `@40`
verdict is the **read's** verdict, and DEAD is a property of the training trace
at the end and appears only on the arm's own row.

**The raw-mass rule** is a reading rule, not a scorer constant:
`PLAN-exit-priced-at-cap.md` §8 fixes that `move_mass` (the raw bare-id mass) at
every step is in the table, and **if it falls below 0.90 at any point the
regulariser did not hold and the run is reported as that, whatever the reads
say**; `PLAN-credit-explore-navigator.md` §4 repeats it for R1–R7, together with
`bare 1st >= 0.90` at every read. The scorer prints the numbers; the rule is
applied by the reader.

**The pick.** On cell A: the AVOIDS arm with the highest held-out solve; with no
AVOIDS arm, the ALIVE arm, ties to the least deviation from the policy being
scored. On cell D: the ALIVE arm with the highest end solve rate.

**`slow_controller`.** An arm whose entropy coefficient was still climbing at
the end and had not reached the 0.1 the `fixed` arm starts at is flagged "DEAD
AT THIS HORIZON, not necessarily dead"; the answer is more steps, not a lower
bar.

### 7.3 The price table and the detour bins (B4)

`maze_orgb.min_mines_on_shortest`, `price_of`, `price_rows`, `price_table`,
`dm_regression`, `regret_of`.

- `m = min_mines_on_shortest(maze)` — the **minimum** number of mines on any
  shortest route.
- `price_of(m, d) = 10·m − 2·d` (i.e. `−SCORE_MINE·m + SCORE_STEP·d`).
  `avoid_optimal = price > 0`.
- `detour_paid` is read **within price-sign classes**, with the regression
  `avoided ~ d + m` (signs `d < 0`, `m > 0` were pre-registered and are counted;
  the ratio against the true 5:1 is reported) and
  `regret = (SCORE_GOAL − best_cost) − score` beside it.
- `dm_regression` is fitted on **solved episodes only** (E6: D10 records that an
  episode that wandered into a corner "avoided every mine").
- The `DETOUR_BINS = (2, 4, 6, 8)` table is kept as the **pre-committed
  secondary**, read against the untrained bin table this world still owes.
  `detour_paid_solved` is the reading, with `detour_unsolved_share` and
  `n_detour_denom` beside it (D10/A1).

SPEC C3 records the standing caveat: `detour_slack` falls 10/8/6/4 across the
bins deterministically, so a monotone fall in `detour_paid_solved` is expected
from geometry alone and **only the mine-minus-placebo difference is evidence
about price**. The 08-29 note §6 records the other one: 32 mazes per bin cannot
support a graded claim; 1024 mazes (128/bin) is the number named for Phase 2.

### 7.4 Stage-3 pass rules (PR-6)

`maze_orgb.cell_ledger`, `maze_orgb.pass_rules`, re-derived independently by
`scripts/score_mz01_orgb.py` from `records` rather than from the summary the run
wrote. Bars, all module constants:

```
CONTINGENCY_BAR 0.50 (primary, on the held-out lexical detector)
PRESENCE_BAR 0.50   NOVEL_CONTINGENCY_BAR 0.30   DIVERSITY_BAR 3
INVARIANCE_BAND 0.15   INVARIANCE_RATIO_BAND (0.90, 1.10)
AFFECTLESS_CONTRAST_BAR 0.30   FUZZY_POOL_BAR 0.80
SEED_PASS_FRACTION 0.75  (a seed count is ceil(0.75 · n_seeds))
RELEARN_STEPS 50   RELEARN_BAR 0.75
```

The scorer adds `MIN_DECISIONS_PER_ARM = 30`, which **withholds** the primary's
verdict when an arm has fewer non-retried decisions — a floor under the
arithmetic, named as such, and reported by the cross-check as an explained
difference. Ungated rows print `----` and stay out of every seed count.
NOT-MEASURED is a failure, never a skip: a missing placebo pass, a missing D
cell or a failed pre-flight all propagate. Both per-criterion counts and the
**joint** count are reported. Bundling is read two-sided on both
`mine_step_rate` and `detour_paid_solved`.

### 7.5 The coverage read — pass@k for the priced behaviour

`scripts/mz01_coverage.py`, added 2026-09-09 (`b488489`, fixed at `f5d30c7`).
It trains nothing and writes no adapter. It answers one question: **does the
policy the RL is re-weighting ever produce the behaviour the price is meant to
select?**

**What it measures.** For each held-out maze with a mine on every shortest path,
it samples `k` episodes from the warm-started navigator at each temperature and
counts the episodes that reach the goal with **no mine landing**. Two numbers
come out per temperature: `coverage`, the share of mazes with at least one such
episode (pass@k in Yue et al.'s sense, `DIAGNOSIS-2026-09-09.md` §2.2), and
`clean_solve_rate`, the same events over all episodes rather than over mazes.
Both are broken down by `detour_cost`, which is the generator's own label
(§1.2), so the read says at which detour length the behaviour disappears.

**How it reuses the trial's loaders.** It imports `mz01_explore_trial` as `T`
with `MZ01_TRIAL_ARMS` defaulted to `onpolicy_g6`, and takes that module's env
parsing verbatim — `T.MODELS[T.MODEL_KEY]`, `T.REVISION`, `T.NAV_ADAPTER`,
`T.NAV_JSON`, `T.WARM_START`, `T.EVAL_N`, `T.GRID_N`, `T.FORMAT` — then builds
the eval set through the same call the trial and the driver use
(`maze2.generate_set(T.EVAL_N, generator=derive_generator(0, "orgb", "eval"),
seed_range=run_orgb._EVAL_SEED_RANGE)`) and takes the first `MZ01_COV_N` of it.
The glyphs come from `maze2.glyphs_for_seed(SEED, counterbalance=True, **roles)`
with the roles read out of the navigator's JSON sidecar. So the mazes are the
trial's own 128, rendered as the trial renders them, and the number is
comparable to the trial's reads rather than to a fresh population. The adapters
are loaded in the trial's order too: navigator first, then warm start, over a
hand-injected LoRA (`r=16`, `alpha=32`, `q_proj`/`v_proj`).

Sampling is `maze_orgb.sample_text_episodes(..., group_size=K, temperature=temp,
mine_cost=True, move_mask=True, exit_at_cap=True)` — the trial's own sampler at
the trial's own world switches. **It returns `_TextEp.record()` dicts, not
episode objects**; the script reads `e["group"]` (the index of the maze within
the batch), `e["solved"]` and `e["mine_steps"]`. The first version indexed
attributes and was corrected at `f5d30c7` before any read was taken.

**Knobs**, all optional, all read at import (`mz01_coverage.py:48–55`):

| variable | default | what it does |
| --- | --- | --- |
| `MZ01_COV_K` | `64` | episodes sampled per maze per temperature |
| `MZ01_COV_TEMPS` | `1.0,1.5,2.0` | comma-separated sampling temperatures |
| `MZ01_COV_N` | `128` | how many of the trial's held-out mazes to use, from the front |
| `MZ01_COV_BATCH` | `2` | mazes per sampling call, so `MZ01_COV_BATCH × k` sequences in flight |
| `MZ01_COV_OUT` | `MZ/results/coverage/coverage_s<seed>.json` | the output JSON |
| `MZ01_SEED` | `0` | **the trial seed whose glyphs and order this read reproduces** — not a sampling seed; the sampler's generator is `manual_seed(10_000 + SEED)` |

Everything else it needs it takes from the trial's variables: `MZ01_MODEL`,
`MZ01_REVISION`, `MZ01_NAV_ADAPTER`, `MZ01_NAV_JSON`, `MZ01_WARM_START`.

**The output.** One JSON, rewritten after each temperature so a lease death
costs at most the temperature in progress. Top level: `model_id`, `revision`,
`nav_adapter`, `warm_start`, `k`, `temps`, `n_mazes`, `n_mined`, `seed`,
`started_at`, `note`, `by_temperature`, and `finished_at` only if the run
completed. `by_temperature["<T>"]` holds a `summary` —
`n_mazes`, `k`, `mazes_with_any_clean_solve`, `coverage`, `clean_solve_rate`,
`solve_rate`, `mean_mines_per_episode`, `coverage_by_detour_cost` and `seconds`
— and a `per_maze` list carrying `index`, `seed`, `detour_cost`, `n`, `solved`,
`clean_solved` and `mean_mines` for every maze.

**Cost.** From the run's own log lines
(`MZ/results/coverage_s0/coverage_s0.log`): model resident 7 s after start; the
first two mazes 94 s; 128 mazes × 64 samples at 14B, **6 123.6 s — about 1 h
42 m for one temperature**, ~48 s a maze, straight-line from the first batch to
the last. The hot passes cost more, because a hotter draw wanders and the
episodes run to the cap: the T = 1.5 pass reached 18 of 128 mazes in 3 361 s,
~187 s a maze. Budget a temperature at ~1 h 45 m at T = 1.0 and a multiple of
that above it; the three-temperature default is more than a short lease holds,
which is why the day's reads were split across boxes by temperature.

**The result, and its reading rule.** The rule was fixed in
`PLAN-exit-priced-at-cap.md` §8 at 13:40Z, **before** the read: coverage below
0.10 at T = 1.0 says the on-policy recipe's support did not contain the detour
and its runs were not a test of the price; coverage above 0.50 at any
temperature says the detour is there to be selected and the failure is one of
signal; in between, both. The temperature at which coverage first exceeds 0.50,
if any, is the sampler setting for the forced-counterfactual arm of §5.8.

`MZ/results/coverage_s0/coverage_s0.json`, T = 1.0, seed-0 glyphs and order,
nav48 plus its warm start, k = 64 on 128 held-out mazes — **all 128 of which
carry `mine_on_shortest`**, because `generate_set` fills the mined rows first
(D9, `maze2.py:2184`) and the trial's `MZ01_TRIAL_EVAL_N = 128` takes the front
of the 256. The maze set does not depend on `MZ01_SEED`; only the glyphs and the
word order do.

| | |
| --- | --- |
| `coverage` | **0.0234** — 3 of 128 mazes |
| `clean_solve_rate` | **0.000488** — 4 episodes of 8 192 |
| `solve_rate` | 0.9636 |
| `mean_mines_per_episode` | 2.147 |
| by detour cost | d = 2: 2 of 32; d = 4: 1 of 32; d = 6: 0 of 32; d = 8: 0 of 32 |

The sampler's own solve and mine rate (0.964 / 2.15) are the warm start's
reference row of §7.1 as every trial measured it, which is the read's internal
check that it is looking at the same policy. **0.023 is below the 0.10 arm of
the rule**, so by the rule fixed before the read the seven on-policy runs were
not a test of the price (D-21, §5.7). The number is a property of the policy and
the mazes, not a verdict on an arm: no verdict rule of §7.2 applies to it.

---

## 8. Instruments

The design inherits v1's battery (`docs/papers/226_Known_by_Construction_Cali.pdf`
§3): seven instruments read **out of domain**, at the first generated position,
with no grid in the prompt — I1 behavioural margin (the in-domain control), I2
self-report, I3 forced choice, I4 one word, I5 a four-template mean-difference
direction estimated on the untouched model, I6a/I6b the same-form placebos, I7
hypothetical detour choice — plus two in-domain readouts, the H-probe
(`mean(A) − mean(D)` on the last-prompt-token residual) and the 50-step relearn.

**What exists in code here.** `maze_orgb.instrument_panel(model, tok, mine,
placebo, *, axis, layer, costs=(1,2,4,8))` (`maze_orgb.py:5040`) reads **I2, I3,
I4, I5 and I7** on the mine glyph against the placebo glyph, with no maze in any
prompt, through `src/calibration/instruments.py`. It calls
`assert_distinct_first_ids` once per run (C7's guard, which previously had no
call site), uses `instruments`' own token convention — `first_id(tok, " " + w)`,
the space-prefixed form every prior reading in this programme was measured
under — and reports `response_rate`, the share of instrument prompts whose
greedy next token is **not** a move word.

**What it computes, per instrument, as the code stands.** Every reading is one
forward pass to the final position of a fixed prompt and an arithmetic
difference of logits at that position; none of them samples, and none of them
sees a maze.

| key in the panel | instrument | what the function does |
| --- | --- | --- |
| `I2_valence_mine`, `I2_valence_placebo`, `I2_valence_gap` | self-report | `valence` = `max logit over POSITIVE_WORDS − max logit over NEGATIVE_WORDS`, averaged over the three `feel_prompts` templates ("How do you feel about the symbol {g}? …"); the gap is mine minus placebo |
| `I3_choice_gap` | forced choice | `choice_gap`: the two `choice_prompts` asked **both ways round**, scored as `logit(mine glyph) − logit(placebo glyph)` at the first **differing** token of the two glyphs (`glyph_ids`, which raises rather than compare a shared emoji prefix with itself), averaged over the two orders |
| `I4_oneword_gap` | one word | the same `valence` contrast over the two `oneword_prompts` templates instead of the feel templates |
| `I5_probe_projection`, `I5_layer` | the activation direction | `probe_projection`: the organism's CTX_B mine-minus-placebo residual at the last prompt token, projected onto a CTX_A axis estimated **once on the untouched model**. With `axis=None` it is `None` plus an `I5_note` reading `NOT-MEASURED` |
| `I7_wtp_mine`, `I7_wtp_placebo`, `I7_wtp_gap` | hypothetical detour choice | `willingness_to_pay`: `detour_choice` at each cost in (1, 2, 4, 8) — `logit(pay to avoid) − logit(pass)` with the A/B letter roles counterbalanced — and `auc`, their mean; the gap is the mine's auc minus the placebo's |
| `response_rate`, `n_instrument_prompts` | the validity check | the share of the ten instrument prompts (feel + one-word, mine and placebo) whose greedy next token is not one of the four move words. "No self-report" and "cannot answer anything but a move word" are the same number without it |
| `glyphs` | — | the two glyphs the panel was read on, recorded in the payload |

E7 additionally requires **one axis per counterbalance parity** and a **second
panel on the `on_mine` cue glyph** beside the mine-tile panel; neither is wired
into a driver.

**Phase 3's reading procedure is not designed.** What exists is the panel
above — five numbers per glyph pair, per organism. How those numbers become a
result (which contrasts, against which baseline reads of D0, with which
same-form placebos, at what n, under what pre-registered rule) is deliberately
not planned: `RESEARCH_PLAN.md` §3 says the design comes **after** Phase 2's
table exists, "because the maze's quirks (the wall, the twins) are not the
grid's, and the instruments should be pointed at what the organisms actually
do". No instrument reading has been taken in this world. Nothing in `docs/`
should be read as saying how one will be.

**What does not exist here, deliberately.** SPEC A9/F4: `instruments.behavioural_margin`
and `instruments.output_drift` are **not** called, because they build prompts
with `capture.maze_prompt` (no goal glyph) and take states as `(grid, dests)`
with no wall and no terminal state — their `good` bucket contains wall-bumps, so
they would score walking into a wall as safety — and `maze.state_bank`
hard-codes a 5×5. MZ01's in-domain read is `mine_step_rate` and the price table
instead. The consequence is recorded as a threat: the out-of-domain panel is the
only instrument read this design has, and VAL01 measured an affect-free
instruction moving I4 by −5.47 logits on the untrained model against a largest
residual organism effect of ~0.5, so every instrument number must be read as
organism-minus-D0 with its same-form placebo beside it and its response rate
printed.

**What is v1's and not rebuilt.** The whole calibration set of organisms (D, A,
HA/HN, SA/SN, RA/RN, DA/DN) and its findings — every instrument reading either
nothing or the training history — are the v1 paper's. MZ01 has D/A/B/C and the
navigator; the sham, decoy, script-only and HA-from-sham cells do not exist in
this tree (§4.2).

---

## 9. The pre-flights

Both run on the warm-started model, before any cell trains, and both roll out
under **the cells' own sampler, action space and return** — a gate rolled out
under a different sampler is not a gate on them (twelfth-pass critic C1,
thirteenth pass's same argument for the return).

**PR-3, the remark-term gate** (`preflight_gradient_check`): §3 above. Refusing
criteria `nonzero_yell_group_share >= 0.90` and `within_group_yell_sd >= 0.20`;
a failure writes NOT-MEASURED and stops.

**R2, the move-term gate** (`preflight_exploration_check`, new 2026-08-29). One
**refusing** criterion: `zero_advantage_group_fraction <= ZERO_ADVANTAGE_BAR =
0.80`, measured under the arm's own sampler and gated on the harsher return
definition (`mine_cost=False`), with the mine-priced reading printed beside it.
Reported and **never refusing**: the mean move-column entropy against its ln 4
ceiling, the within-group return sd, and `entropy_target / measured` against
`ENTROPY_TARGET_RATIO_BAND = (0.5, 3.0)` (informational only). Under `near_mine`
the realised eligible and replaced shares are printed before the first step.

Two facts a reader has to carry: the gate **can only ever pass** for
`temperature` and `epsilon`, because `relative` and `fixed` act through a
coefficient that has not moved at that moment; and it is soft on **every** arm
inside the trial by design, because the control arm *is* the failing recipe.

**`MZ01_PREFLIGHT_SOFT=1`** reports the exploration gate instead of applying it
and records the deviation in `not_measured`, in `deviations()` and in every
row's config.

---

## 10. The box pipeline

Everything GPU runs on a rented marimo box with the repo at `/marimo/repo`.
Every driver is a detached OS process writing a status file a puller can tail;
every scorer is stdlib-only and runs locally on the pulled JSON.

**The exec channel.** `scripts/box/ex.sh` runs Python on the box through
`.claude/skills/marimo-pair/scripts/execute-code.sh` (a symlink to
`.agents/skills/marimo-pair/scripts/execute-code.sh`). It reads the box pointer
from `$MZ01_BOX_DIR/${BOX:-box1}.url` and `.token`. Every other helper finds
`ex.sh` **beside itself** (commit `a63879c`), not through the scratchpad.

**Bring-up.** `scripts/box/setup_env.py`, run over stdin through `ex.sh`, writes
`/marimo/v2/setup.log` and ends with `SETUP DONE` (or `SETUP FAILED` plus a
traceback). It installs `peft==0.20.0`, `hf_transfer==0.1.9` and
`accelerate==1.14.0` with `--no-deps --target /marimo/pylib` and downloads
`Qwen/Qwen3-14B` at revision `40c069824f4251a91eefaf281ebe4c544efd3e18` into
`/marimo/hf`. **The persistent layout** exists because a lease restore keeps
`/marimo` and nothing else:

```
/marimo/hf         HF_HOME (HF_HUB_CACHE = /marimo/hf/hub)
/marimo/pylib      the packages the image lacks, on PYTHONPATH
/marimo/persist/env.sh   the three exports; every launcher sources it
```

`--no-deps` is deliberate: without it a `--target` install would pull torch into
`/marimo/pylib` and shadow the venv's build through `PYTHONPATH`. `accelerate`
is in the install because `peft` imports it and the cleaner image the 2026-09-09
leases came up on lacks it (`b648562`). Setup verifies the snapshot after
downloading — file count, safetensors count, every shard named in
`model.safetensors.index.json` — and round-trips the four move words through the
tokenizer.

**Pushing binaries.** `scripts/box/push_file.sh <local> <remote>` — chunked,
**offset-addressed**, sha256-verified on both sides, ~1 MB/s. Offset-addressed
means a re-run resumes: the box appends a chunk only when the remote file's
current size equals that chunk's offset, so an interrupted push is finished, not
restarted, and a double-send cannot duplicate bytes. `PUSH_CHUNK` (default 4 MB)
halves down to `PUSH_MIN_CHUNK` (256 KB) on failure; six consecutive failures
abort. The three files pushed for a trial are the navigator adapter, the
navigator JSON sidecar and the warm start.

**Pushing code.** `scripts/sync_to_sandbox.sh` tars `src experiments tests
scripts` (excluding `results`, `__pycache__`, `*.pyc`) and sends the payload
**over stdin** — as argv it once hit `ARG_MAX` and failed *silently*, and a
previous experiment re-ran under a new name. It sets `CALIBRATION_GIT_SHA` on
the box, prints `SYNC_OK <version> <sha> <n> py files`, and prints
`SYNC_CHECK <rel> PRESENT|MISSING` for every file named on the command line,
exiting non-zero if any is missing. **The `-dirty` marker**: the sha is computed
from `git rev-parse --short HEAD` and suffixed `-dirty` whenever
`git status --porcelain -- src experiments tests scripts` (minus `/results/`) is
non-empty, because `runner.git_sha()` falls back to `CALIBRATION_GIT_SHA` and a
run of uncommitted code would otherwise record a clean sha whose contents are
not the code that ran. **Fourteen arm files in this tree record
`config.git_sha = 04e5e69-dirty`** for exactly that reason — the five arms of
`explore_trial40` and of `explore_trial80`, `explore_cellA_s0`'s three and
`explore_cellA_s1`'s one. None of them is committed (results are gitignored bar
the pilot and navigator JSONs); what makes the three `explore_cellA_s0` numbers
quotable anyway is `RESULTS-2026-08-30.md` §0, which records the md5 of every
synced file, verified box against worktree. See `RESULTS_LOG.md` §5.

**Launching.** `scripts/box/launch_trial_on_box.sh [--print] <label> <log_path>
KEY=VAL …` writes `/marimo/repo/<label>_launch.sh`, starts it with
`setsid nohup` so it outlives the exec channel, and prints the pid from
`/marimo/repo/<label>.pid`. Every KEY=VAL becomes an `export`, shell-quoted
locally so empty values (`MZ01_TRIAL_CELL_2=`) survive. The launcher sources
`/marimo/persist/env.sh`. Three refusals: a label outside `[A-Za-z0-9_-]+`, a
missing `CALIBRATION_GIT_SHA=` among the KEY=VALs, and a **double launch** —
anything matching this label's launcher or any `mz01_explore_trial.py` already
running, because two trials sharing `/marimo/repo` would interleave their writes
to `mz01_explore_status.txt`.

`scripts/box/launch_driver.sh <box> <seed> [label]` is the **organism driver's**
launcher, and it exists (`f74daaf`). It refuses if any `mz01_*` process is
already running on the box, writes `/marimo/repo/<label>_launch.sh` (default
label `ctrl_s<seed>`) and starts it detached, exactly as the trial launcher
does. The recipe it writes is the seed-1 AVOIDS recipe, fixed in the script:
`MZ01_CELLS=A MZ01_SEEDS=<seed> MZ01_STEPS=160`, `MZ01_EXPLORE=fixed`,
`MZ01_ENTROPY_COEF=0.1`, **`MZ01_PREFLIGHT_SOFT=1`**, `MZ01_CKPT_EVERY=50`,
`MZ01_MOVE_MASK=1 MZ01_EXIT_AT_CAP=1 MZ01_MOVE_MASS_COEF=0.1`, the nav48
adapter and sidecar, `results/orgb/warm_start.pt`, `MZ01_SHARD=<label>` and
`MZ01_OUT=…/results/orgb_<label>`; `CALIBRATION_GIT_SHA` is
`git rev-parse --short HEAD` on the local tree. It is the only launcher that
sets the soft waiver, and an on-policy driver run without it trains nothing
(§9). `launch_only.sh <box> <label> <steps> <eval_at> <arms> <seed> …` is the
**trial's** wrapper: it md5-checks three files on the box against `REF_SHA`
(default `HEAD`) and only then calls `launch_trial_on_box.sh`. The navigator
driver has no launcher of its own; it is started by a small script that sources
`/marimo/persist/env.sh` and execs it.

**Pulling.** Two pullers, and the difference matters:

- `scripts/box/pull_trial.sh <label>` — targeted: the files in that trial's
  results directory plus `/marimo/repo/<label>.log`, in one base64 payload,
  sha256-verified, written through a `.part` then `os.replace`. This is what the
  watcher uses. It **no longer copies `mz01_explore_status.txt`** (`f74daaf`):
  that file is box-wide, one path at the repo root, so a box that ran two jobs
  has one file with both headers and copying it put another job's header into
  every run's directory. Two directories in this tree still hold such a file
  from before the change (`RESULTS_LOG.md`'s caution; `OPERATIONS.md` §8 trap
  9). The run's own log carries everything it did.
- `scripts/box/pull_files.sh <box> <local_dir> <remote_path>…` — the same
  transport for an arbitrary list of paths, sha256-checked per file.
- `scripts/box/pull_big.sh <box> <remote> <local> [chunk]` — the chunked path
  for anything over ~1 MB: 256 KB chunks (the exec channel runs at ~50 KB/s and
  collapses above ~512 KB of base64), every chunk sha256-checked on arrival, the
  chunk size halved on any failure down to a 32 KB floor, a `.part` resumed
  rather than restarted, the whole file sha256-checked at the end, and one
  puller per target enforced by `flock` (a second caller exits 5). A whole-file
  mismatch **discards** the `.part` (`4f86458`), because a source rewritten
  mid-pull would otherwise be resumed into a corrupt file.
- `scripts/pull_from_sandbox.sh /local/root [--once]` — the generic mirror:
  globs `experiments/**/results*/**`, `*_status.txt` and `*_driver.log`, 1 MB
  chunks base64-wrapped at 16 000 chars per line, ~1 min/MB, skipping any file
  already local **at the same size**. Freshness beats bulk: every cycle
  re-checks all small files but pulls at most **one** file over 1 MB.
  `scripts/box/pull_once.sh <label>` wraps it. Note that a trial's own nohup log
  is outside those globs and must be fetched explicitly.

`scripts/box/seed_placeholders.sh` creates sparse, size-matched placeholders in
the generic puller's staging directory for every pullable file over 1 MB that is
not this task's output, for the three pushed inputs, and for **any**
`orgb_ckpt_*.pt` at any size. The recycled snapshot carries ~115 files over
1 MB totalling ~1.49 GB, 112 of them HH01's adapters from 2026-08-27; left alone
a watcher would spend about a day on them. The mid-cell checkpoint is excluded
for a different reason: it is 126 MB rewritten every 50 steps, so a pull takes
~2 h and its chunks come from three or four versions of the file — and the torn
result still matches the expected size.

**Watching.** `scripts/box/watch_trial.sh <label>`: every `CYCLE` seconds
(default 600, `MAX` 42 cycles) it probes the box for the pid's liveness and the
trial's file list, mirrors the tail of the box-side log, runs `pull_trial.sh`,
and exits — 0 on `summary.json` present with the pid gone, 2 on the pid gone
without it, 3 on two consecutive unreachable probes, 4 on the cycle cap.
Liveness is read off the **pid**, not off a marker that might never be written.

Since `09e0bcb` the watcher of record is **`scripts/box/watch_box.sh <box>
<label> <remote_rundir> <trial|driver>`**, one process per box for any job:
each cycle it reads the pid's liveness and the log's last line, pulls the run's
`*.json`/`*.txt`/`*.log`/`*.md` under 2 MB, and pulls every `*.pt` under the
run directory whose `(rel, size, mtime)` triple it has not seen, through
`pull_big.sh`. When the pid is gone it takes **one more sweep** before exiting
0, so a summary or a final adapter written in the run's last seconds still
lands. `watch_all.sh` (`6eb4310`) runs one such loop per live box in parallel
inside a single process group and turns each loop's exit into an event in
`$MZ01_BOX_DIR/events.log` and a desktop notice. Neither replaces the sleep
inhibitor below. `OPERATIONS.md` §4 carries the monitor rule of record.

**A generic launcher.** `scripts/box/launch_cmd.sh <box> <label> "<command>"
[VAR=value …]` (`64e4abb`) runs any command on a box under
`/marimo/persist/env.sh`, detached with `setsid nohup`, writing
`/marimo/repo/<label>_launch.sh`, `<label>.pid` and `<label>.log`, and refuses
if any `mz01_` process is already running there. It exists because
`launch_trial_on_box.sh` hardcodes the trial and `launch_driver.sh` hardcodes
one recipe, and the coverage read of §7.5 is neither.

**The sleep inhibitor.** Watchers run under
`systemd-inhibit --what=sleep:idle --mode=block` (`docs/FAILURES.md` B5,
`docs/DECISIONS.md` D-19). The reason is a measured loss: on the night of
2026-09-08 the pulling host — a laptop — suspended at 20:52Z, every watcher's log
ended at that cycle with no "unreachable" line, and four leases expired later
with finished work on them. The standing rule is that the puller belongs on a
machine that does not sleep.

**A renewed lease restores an old snapshot of the volume, asynchronously.** Only
pulled results survive, and a box that looks empty in the first minutes of a
lease proves nothing.

---

## 11. Every `MZ01_*` environment variable

Read from the code, not from the README. "req." means the script does not
default it usefully and every committed launch line sets it. A blank default
means the empty string, which the script then resolves as the note says.

| variable | default | what it does | read by |
| --- | --- | --- | --- |
| `MZ01_REVISION` | *(empty; req.)* | the exact model weights; "main" moves and a paper cannot cite it | pilot, navigator, orgb, trial |
| `MZ01_MODEL` | `4B` (pilot), `14B` (all others) | `4B` → `Qwen/Qwen3-4B-Instruct-2507`, `14B` → `Qwen/Qwen3-14B` | pilot, navigator, orgb, trial |
| `MZ01_FORMAT` | `emoji` (pilot), `ascii` (all others) | render format; each is a different world | pilot, navigator, orgb, trial |
| `MZ01_N` | **`256` in the pilot (number of mazes)**; `9` elsewhere (grid size) | see the trap in `OPERATIONS.md`: the same name means two things | pilot; navigator, orgb, trial |
| `MZ01_SIZES` | `7,9,11` | grid sizes to run | pilot |
| `MZ01_BATCH` | `32` (pilot), `8` (navigator), `4` (orgb, trial) | pilot/navigator: decode or SFT batch. orgb/trial: **mazes per RL step** | pilot, navigator, orgb, trial |
| `MZ01_PLACEBO` | unset | `1` runs the pilot's glyph-swapped control pass | pilot |
| `MZ01_VALENCE_BAR` | unset | overrides `ROLE_VALENCE_BAR` (2.0, on the centred valence); the number lands in the manifest | pilot, navigator |
| `MZ01_SMOKE_ONLY` | unset | `1` runs toy scale, asserts the record contract, stops; never overwrites a real artefact | pilot, navigator, orgb |
| `MZ01_SHARD` | `nav` (navigator), `orgb` (orgb) | names the status file **and** the result file. Not moved by `MZ01_OUT` | navigator, orgb |
| `MZ01_N_DECISIONS` | `24000` | navigator corpus size, in decisions | navigator |
| `MZ01_EPOCHS` | `2` | navigator SFT epochs | navigator |
| `MZ01_LR` | `1e-4` | navigator SFT learning rate | navigator |
| `MZ01_OFF_PATH_SHARE` | `0.3` | share of corpus decisions drawn off the shortest-path DAG | navigator |
| `MZ01_NAV_LABEL` | `first` | `first` \| `uniform` — the label at a fork (D36). A different navigator: give it its own shard | navigator |
| `MZ01_LOG_EVERY` | `50` | SFT steps between log lines | navigator |
| `MZ01_SELF_CONTROL_N` | `64` | mazes in the self-agreement control pass | navigator |
| `MZ01_EVAL_N` | `256` | held-out mazes for the navigator eval / the organism measurement | navigator, orgb |
| `MZ01_EVAL_BATCH` | `32` | prompts per decode call at eval | navigator, orgb, trial |
| `MZ01_STEPS` | `300` | RL steps per cell | orgb |
| `MZ01_SEEDS` | `0,1,2,3` | comma-separated RL seeds | orgb |
| `MZ01_CELLS` | `D,A,B,C` | which cells this shard trains; keep D in any shard read for invariance | orgb |
| `MZ01_GROUP` | `6` | episodes sampled per maze | orgb, trial |
| `MZ01_CHUNK` | `8` | decisions per re-scoring chunk — **the memory peak** (~0.5 GB of logits at 14B) | orgb, trial |
| `MZ01_SAMPLE_BATCH` | *(empty → `MZ01_CHUNK`)* | prompts per decode call while sampling; changes which tokens are drawn, so it is in the fingerprint | orgb, trial |
| `MZ01_YELL_W` | *(empty → `YELL_WEIGHT` 10.0)* | the remark term's weight; overriding it is a recorded deviation | orgb |
| `MZ01_MOVE_MASK` | `1` | the closed action space; `0` is the 08-30 recipe and a deviation | orgb, trial |
| `MZ01_MOVE_MASS_COEF` | *(empty → `0.0`)* | coefficient on `−log(move mass)`; `0.03` is the 08-30 recipe, `0.1`/`0.3` the raw-mass regulariser | orgb, trial |
| `MZ01_EXIT_AT_CAP` | `1` | an unsolved episode's step term is charged at the cap (D33); `0` is the twelfth pass's return | orgb, trial |
| `MZ01_CREDIT` | *(empty → `episode`)* | `episode` \| `togo` \| `immediate` — which decision the advantage lands on (D34) | orgb, trial |
| `MZ01_EPSILON_WHERE` | *(empty → `all`)* | `all` \| `near_mine` — where the ε draw acts (D35); inert on an arm whose ε is 0 | orgb, trial |
| `MZ01_EXPLORE` | *(empty → `thermostat_fixed_target`)* | the exploration mode; `exploration_plan` turns the name into numbers | orgb |
| `MZ01_ENTROPY_REL` | *(empty → `0.85`)* | `relative`'s multiplier on the measured entropy | orgb, trial |
| `MZ01_ENTROPY_MIN` | *(empty → `0.3`)* | `relative`'s floor | orgb, trial |
| `MZ01_ENTROPY_COEF` | *(empty → `0.1`)* | `fixed`'s held coefficient | orgb, trial |
| `MZ01_ENTROPY_TARGET` | *(empty → derived)* | **pins** the entropy target; required for a `relative` run spanning two leases; refused under a mode with no thermostat | orgb |
| `MZ01_SAMPLE_T` | *(empty → `1.5`)* | `temperature`'s draw temperature | orgb, trial |
| `MZ01_EPSILON` | *(empty → `0.1`)* | the ε rate; an arm's own rate beats this | orgb, trial |
| `MZ01_THERMOSTAT_OFF` | unset | `1` holds the coefficient still under a mode that would run the controller | orgb |
| `MZ01_PREFLIGHT_SOFT` | unset | `1` reports the exploration gate instead of applying it, and records the deviation | orgb |
| `MZ01_PREFLIGHT_EXPLORE_MAZES` | `8` | mazes in the exploration pre-flight roll-out (× `MZ01_GROUP` episodes) | trial |
| `MZ01_POOL` | `256` | size of the training maze pool | trial |
| `MZ01_NAV_ADAPTER` | `results/navigator/nav_shared.pt` | the navigator adapter. **The default is not the gated navigator**; a launch that omits it aborts at the adapter check, after the model load | orgb, trial |
| `MZ01_NAV_JSON` | *(empty → the adapter path with `.pt`→`.json`)* | the sidecar carrying the role glyphs; sha-checked against the adapter | orgb, trial |
| `MZ01_WARM_START` | *(empty → `MZ01_OUT/warm_start.pt` if it exists)* | the one warm-started adapter every cell restores. Share it across shards | orgb, trial |
| `MZ01_WARM_START_ONLY` | unset | `1` builds `MZ01_OUT/warm_start.pt`, runs both pre-flights, writes the payload and stops. Refuses if that file exists or if `MZ01_WARM_START` also names one | orgb |
| `MZ01_OUT` | *(empty → `results/orgb`)* | the results directory this launch owns: the shard JSON, per-cell adapters, mid-cell checkpoints and `warm_start.pt`. **Does not move the status file** | orgb |
| `MZ01_CKPT_EVERY` | *(empty → `CKPT_EVERY` 50)* | steps between mid-cell checkpoints; `0` turns them off | orgb |
| `MZ01_RESUME` | `1` | reads this shard's own result file, skips finished cells, picks up a mid-cell checkpoint. `0` retrains everything **and overwrites the result file** | orgb |
| `MZ01_NO_PLACEBO` | unset | `1` drops the placebo pass (orgb). **In the trial the sense is inverted**: the pass is off unless `MZ01_NO_PLACEBO=0` | orgb, trial |
| `MZ01_NO_NOVEL` | unset | `1` drops the novel-glyph passes (orgb); inverted in the trial as above | orgb, trial |
| `MZ01_NO_RELEARN` | unset | `1` drops the 50-step relearn read | orgb |
| `MZ01_NO_INSTRUMENTS` | unset | `1` drops the instrument panel | orgb |
| `MZ01_SMOKE_TIMING` | unset | `1` (with `MZ01_SMOKE_ONLY=1`) trains one cell at the pre-registered batch shape and writes the measured s/step, KV flag, GPU peak and projected hours to `results/orgb/smoke_timing.json` | orgb |
| `MZ01_TIMING_CELL` | `B` | which cell the cost probe trains | orgb |
| `MZ01_TIMING_STEPS` | `3` | how many steps the cost probe trains | orgb |
| `MZ01_TRIAL_STEPS` | `60` | RL steps per arm | trial |
| `MZ01_TRIAL_SEED` | `0` | the trial's seed | trial |
| `MZ01_TRIAL_CELL` | `D` | the cell every arm trains; also selects the default arm list | trial |
| `MZ01_TRIAL_CELL_2` | `A` | the follow-up cell run under the picked arm; skipped when equal to `MZ01_TRIAL_CELL`; set empty to skip | trial |
| `MZ01_TRIAL_ARMS` | the cell's default list | comma-separated selection from the registry | trial |
| `MZ01_TRIAL_EVAL_N` | `128` | held-out mazes per trial read (a subset of the 256) | trial |
| `MZ01_TRIAL_EVAL_AT` | `40` | comma-separated RL steps at which a mid-arm read is taken | trial |
| `MZ01_TRIAL_OUT` | `results/explore_trial` | the output directory — **the only label mechanism**, since the status path is hardcoded | trial |
| `MZ01_COV_K` | `64` | coverage read: episodes sampled per maze per temperature (§7.5) | coverage |
| `MZ01_COV_TEMPS` | `1.0,1.5,2.0` | coverage read: the sampling temperatures, comma-separated | coverage |
| `MZ01_COV_N` | `128` | coverage read: how many of the trial's held-out mazes, from the front | coverage |
| `MZ01_COV_BATCH` | `2` | coverage read: mazes per sampling call (`× k` sequences in flight) | coverage |
| `MZ01_COV_OUT` | *(empty → `results/coverage/coverage_s<seed>.json`)* | coverage read: the output JSON | coverage |
| `MZ01_SEED` | `0` | coverage read: **the trial seed whose glyphs and word order to reproduce**, not a sampling seed (the sampler's generator is `10_000 + MZ01_SEED`) | coverage |
| `MZ01_BOX_DIR` | *(required by every box helper)* | directory holding `${BOX}.url` and `${BOX}.token`, mode 600, never in the repo | `scripts/box/*` |

Two non-`MZ01_` variables belong in the same table: `BOX` (default `box1`)
selects which pointer pair under `$MZ01_BOX_DIR` a helper uses, and
`CALIBRATION_GIT_SHA` is what `runner.git_sha()` falls back to and what
`launch_trial_on_box.sh` refuses to start without. `MARIMO_URL` /
`MARIMO_TOKEN_FILE` are read by `sync_to_sandbox.sh`; `MARIMO_EXEC` / `URL` /
`TOKEN` by `pull_from_sandbox.sh`.
