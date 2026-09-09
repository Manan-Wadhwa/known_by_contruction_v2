# BUILD — the forced counterfactual at geometric forks (fifteenth pass)

**2026-09-10. Built against `PLAN-fork-counterfactual.md` §1–§7, on top of
`8d459d1`, in an isolated worktree. Nothing was launched, nothing under
`results/` was touched, no PLAN file was edited, and nothing was committed.**

The switch is **off by default**. With `MZ01_FORK_CF` off the trainer's loss,
its trace, its checkpoint fingerprint and its adapter are the fourteenth pass's
bit for bit — asserted against numbers recorded from the pre-build commit, not
argued (`test_fork_off_is_the_pre_build_trace_and_the_pre_build_adapter`).

---

## 1. What changed, by file and line

### `src/calibration/maze_orgb.py`

| line | what |
| --- | --- |
| 170–173 | `__all__` gains the six public fork names |
| 1722–1749 | **`_decision_texts`** — a pure extraction of `sample_text_episodes`'s own prompt-building loop (`render` + `maze_prompt` + `apply_chat`), so the roll-out and the continuations render the same prompt from the same three calls. The only edit to the existing sampler: line 1877 now calls it. |
| 1957–2003 | The section header: why the pass exists, and the construction test written out as the property each function below is built against. |
| 2005–2013 | `FORK_TRACE_FIELDS` (the six §4 fields) and `FORK_ADVANTAGE_TOL`. |
| 2016–2018 | `_fmt_share` — `None` prints as `n/a` in the log line. |
| 2021–2029 | `_reverse_move(pos, prev)` — the way back, from geometry. |
| 2031–2103 | **`episode_forks(episode, maze)`** — the fork rule (§2.2). |
| 2105–2135 | **`plan_forks(episodes, mazes, *, generator, per_episode)`** — the uniform draw on the run's own generator (§2.2). |
| 2137–2171 | **`_launch_continuations(forks)`** — one `_TextEp` per (fork, valid move), positioned at the fork with the remaining budget. Separated from the sampling so the construction test can run it blind. |
| 2173–2261 | **`sample_fork_continuations(...)`** — the forced first move, the lockstep on-policy continuation, and the pricing (§2.3–§2.4). |
| 2263–2288 | **`fork_advantages(returns, *, std)`** — `A(a) = R(a) − mean R`, optional sd division (§2.5). |
| 2290–2347 | **`fork_terms(forks, continuations, *, std)`** — the loss's input rows and §4's per-step read, including the `fork_share_detour_better` diagnostic. |
| 2349–2385 | **`_fork_move_logp(...)`** — `log π(a | prompt)` over the four bare move ids, **with grad**. |
| 2758–2760 | `train_orgb_cell` gains `fork_cf=False, fork_per_ep=1, fork_coef=1.0, fork_std=False, fork_temperature=1.0`. |
| 2854–2887 | The trainer docstring's fork paragraph (including the normaliser, flagged in §6 below). |
| 2984–3011 | The knob guards, at the top of the trainer, before any episode is drawn. |
| 3147–3152 | The six trace fields — added **only when the term is on**. |
| 3215–3229 | The five fingerprint fields — added **only when the term is on**. |
| 3354–3378 | The step's fork block: `plan_forks` → `sample_fork_continuations` → `fork_terms`, on its own clock. |
| 3418–3441 | The fork loss term, chunked, added to the same accumulated gradient. |
| 3525–3536 | The trace append. |
| 3642–3651 | The log line's fork suffix (only when the term is on). |
| 3869–3894 | `history["config"]["fork"]` — the recipe block, written only when the term ran. |

### `scripts/mz01_explore_trial.py`

| line | what |
| --- | --- |
| 144–163 | The five environment switches in the module docstring's env list. |
| 243–252 | The parser: `MZ01_FORK_CF` (off), `MZ01_FORK_PER_EP` (1), `MZ01_FORK_COEF` (1.0), `MZ01_FORK_STD` (0), `MZ01_FORK_TEMP` (1.0). |
| 369–394 | The arm **`fork_cf_g6`** (`mode: fixed`, `group: 6`, `fork_cf: True`), appended last so `ARM_NAMES[:5]` and `ARM_NAMES[5:9]` — the committed cell-D and cell-A defaults — do not move. `MZ01_GROUP` still does not override an arm's own `group`. |
| 826–832 | `arm_fork = bool(arm.get("fork_cf", FORK_CF))` — the arm's own switch first, the environment as fallback. |
| 848–858 | `cfg["fork_*"]` in the arm payload. |
| 865–873 | The launch line that names the term when it is on. |
| 953–959 | The trainer call: `fork_cf=arm_fork`, and the four knobs **only** when it is on. |
| 976–992 | The post-arm summary: forks, continuations, the share with a difference, and the detour-better diagnostic. |

### `docs/CONSTRUCTION.md`

§5.8 rewritten from "DESIGN ONLY; none of this is implemented" to "item 1 is
BUILT; items 2–4 are DESIGN ONLY", with the anchors, the switches, and the
construction-test row for item 1 amended to say the condition is now tested.

### `tests/test_maze_orgb.py`

Lines 5769–6393: the fifteenth-pass section — three hand-built worlds, the
`_MineBlind` wrapper, two fixtures and eight tests (§5 below).

---

## 2. How a fork is chosen

`episode_forks(episode, maze)` (maze_orgb.py:2031) walks the episode's **own
recorded positions** — `ep["positions"][t]` is where the agent stood when it
made decision `t` — and for each one asks the maze for its valid moves:

```
back  = the move word that walks from positions[t] back to positions[t-1]
        (None at t = 0, and None when the previous decision did not move —
         a blocked or unparsed turn leaves the agent where it was)
moves = [w for w, _next in maze.neighbours(positions[t]) if w != back]
a fork  <=>  len(moves) >= 2
```

`maze.neighbours` is the maze's own validity function (`is_open`: on the grid
and not a wall) and it yields in `MOVE_WORDS` order, so a fork's continuations
are always built in one fixed order whatever the turn's word order was.
`maze.labels["branch_points"]` — a generator-level count over every open cell —
is **not** read, as the brief requires.

`plan_forks(episodes, mazes, *, generator, per_episode)` (maze_orgb.py:2105)
then draws `per_episode` (default 1) of an episode's forks uniformly with
`torch.randperm` on **the run's own generator**, and attaches the episode's own
maze. An episode whose path has no fork contributes nothing.

Three consequences worth stating:

* **A cell visited twice is two forks.** The counterfactual is about the
  *decision*, not the cell: two visits have different prefixes and therefore
  different budgets.
* **The start is a fork** whenever it has ≥ 2 valid moves; there is no way back
  to exclude.
* **The reverse is excluded from `moves`, not only from the count** — see §6,
  deviation 1.

---

## 3. How a continuation is started, run and priced

`_launch_continuations` (maze_orgb.py:2137) builds one `_TextEp` per (fork,
valid move) with two fields overridden and nothing else:

```
ep.pos = the fork cell
ep.cap = fork["budget"] = episode cap − the prefix's steps      (§2.3)
```

That is the whole state a continuation needs, and the reason is a property of
this world that `_decision_texts` now says out loud: **the prompt is the maze
and the current position** (`maze2.render` + `maze2.maze_prompt` + this turn's
word order) — no history, no reply so far, no step counter. The world is Markov
in the position for the policy as well as for the environment, so a continuation
from a fork is a fresh roll-out started at the fork cell and there is no prefix
history to carry.

`sample_fork_continuations` (maze_orgb.py:2173) then:

1. **Forces the first move through the environment** — `_TextEp.apply(word, …)`,
   so `maze2.step` walks it and names the landing exactly as it would any move.
   No policy call is made for it; the plan says the move is forced, and sampling
   a reply only to overwrite its move word (the epsilon path's `force_prefix`)
   would cost one decode per continuation and buy nothing, because a
   continuation's tokens are never trained.
2. **Continues on-policy in lockstep across every fork of the step** — the same
   `_decision_texts` prompts, the same `_sample_replies`, the same decode window
   (`max_new_tokens`), the same action space (`move_cols` when the run's
   `move_mask` is on), the same `MAX_CONSECUTIVE_BLOCKED` rule, at
   `MZ01_FORK_TEMP` (the continuations' own temperature). **No epsilon**: the
   continuations estimate what the *policy* would do from each move.
3. **Prices each with `episode_return`** on its own landings, with its own
   remaining cap and its own `solved` flag:

```
R(a) = episode_return(landings, mine_cost=<the cell's switch>,
                      cap=<budget if exit_at_cap else None>, solved=…)
```

So a continuation that spends its budget without reaching the goal is charged at
that budget, and one that ends early (three blocked moves) is charged at the
same budget rather than at the decisions it took — D33 applied to the
continuation exactly as it is to an episode. Both cases are asserted, and they
come out equal, in `test_a_continuation_is_priced_at_the_cap_minus_the_prefix`.

---

## 4. How the fork term enters the loss

The existing update is untouched: the per-decision move term, the group
baseline, the yell term, the entropy bonus, the mass term and the per-decision
normaliser run exactly as they do with the switch off. The fork term is a
**third term on a fifth object** — the fork decision's move distribution — added
to the same accumulated gradient in a second chunked pass (maze_orgb.py:3411):

```
fork term = −(fork_coef / n_forks) · Σ_forks Σ_a A(a) · log π(a | the fork's own prompt)
A(a)      = R(a) − mean_a' R(a')        (÷ sd of the continuations' returns, at
                                         ≥ 3 of them, with MZ01_FORK_STD=1)
```

* **The prompt** is `decisions[row]["text"]` — the exact string the policy saw
  at that fork, this turn's word order included. Nothing is re-rendered.
* **`log π(a | prompt)`** is `_fork_move_logp`: the 4-way `log_softmax` over the
  four bare move ids at the position that predicts the first generated token —
  the same distribution `_rescore` takes the action's log-prob and the entropy
  bonus from, at the same position, from the same left-padded encoding with
  explicit `position_ids`. It is the only way to score a move the policy did not
  emit, and under the closed action space it *is* the policy's move
  distribution.
* **The continuations are not trained** (§2.6). `_rescore` — the one function
  that takes a gradient over sampled tokens — is called on the step's own
  decisions and on nothing else; the end-to-end test asserts the count.
* `history["loss"]` includes the fork term (it is part of the loss); the fork
  backward is inside the `rescore_seconds` clock and the continuations' sampling
  is on its own (`fork_sample_seconds`).

---

## 5. The trace, the switches, the arm

**Trace fields (§4), written only when the term is on** — so with it off the
trace is the fourteenth pass's *key for key*, which also means a checkpoint
written before this build still resumes:

`n_forks`, `n_continuations`, `fork_adv_abs_mean`, `fork_share_nonzero`,
`fork_share_detour_better`, `fork_sample_seconds`.

`history["config"]["fork"]` carries the recipe (`per_episode`, `coef`, `std`,
`temperature`) plus two prose fields — the rule, and where mine information can
and cannot enter — for a reader of the JSON.

**Switches (§3), named exactly as the plan names them:**

| env | default | trainer argument |
| --- | --- | --- |
| `MZ01_FORK_CF` | off | `fork_cf` |
| `MZ01_FORK_PER_EP` | 1 | `fork_per_ep` |
| `MZ01_FORK_COEF` | 1.0 | `fork_coef` |
| `MZ01_FORK_STD` | 0 | `fork_std` |
| `MZ01_FORK_TEMP` | 1.0 | `fork_temperature` |

A knob set without the switch is a `ValueError` at the top of the trainer: a run
that recorded a counterfactual it never took is the failure the epsilon guards
already exist to refuse.

**The arm `fork_cf_g6`** — `mode: fixed` (thermostat off, coefficient held),
`group: 6`, `fork_cf: True`, no epsilon and no hot draw, so it differs from the
cell-A control `onpolicy_g6` in **exactly one thing**. It is appended last in
the registry and is a named selection (`MZ01_TRIAL_ARMS=fork_cf_g6`), never a
default: `DEFAULT_ARMS` and both committed trials are unchanged.

---

## 6. What the plan left open or could not be built as written

**1. "Every valid move at the fork" (§2.3) — read as "every valid move the fork
is defined over", i.e. the reverse of the arriving move excluded.** §2.2 defines
a fork by the moves that are *not* the way back, and those are the alternatives
the comparison is about. Forcing the way back as well would add a continuation
that re-treads the prefix under a shortened budget, almost never solves, and
would therefore drag every fork's mean return down and inflate |A| at forks
where nothing is at stake. Consequence: if the episode's own move at the fork
*was* the way back (or a wall bump, or unparsed), it has no continuation of its
own — it is then left out of `fork_share_detour_better`'s denominator, never
counted as a failure. One line to change if the critic wants the other reading
(`episode_forks`, the `if w != back` filter).

**2. The normaliser (§2.5 writes a bare sum).** Implemented as `÷ n_forks`.
Summing over a step's ~24 forks would put a term two orders of magnitude above
the move term (which is divided by the step's ~300 decisions) at `fork_coef`
1.0. Dividing by the number of forks makes the fork term "the mean over forks of
−Σ_a A(a) log π(a)" — the same shape as the move term's per-decision mean — and
leaves `MZ01_FORK_COEF` as the dose. **This changes the effective dose by a
factor of about n_forks against the literal reading**, and is the single item a
critic should confirm before the launch.

**3. `mine_cost=True` (§2.4) — implemented as the cell's own switch.** On cell A
(the pre-registered arm) the two are the same number. Passing the literal would
put a mine-priced quantity into the loss of cell B or D, which are *defined* by
not pricing the mine; the cell's switch keeps a mine-free cell mine-free if the
term is ever turned on inside one. `exit_at_cap` likewise.

**4. "Only the pricing raises" (§5 test 2) — cannot be literally true.** A
landing has to be *named* `mine` before `episode_return` can price it, and
`maze2.step` does that with `maze.is_mine`. So the test asserts the line where it
actually is: fork selection, the draw and the launch all run against a maze whose
`is_mine`, `mines`, `at` and `rows` all raise, and produce exactly what they
produce against the real maze; the world's own step is where blindness stops,
and that raise is asserted too. The same claim is made from the other side by an invariance check
— same walls, mine moved: same forks, same forced moves, same budgets.

**5. `log π(a | fork)` under `move_mask=False`.** The 4-way restricted softmax is
then not the emission probability (the sampler draws the first token from the
whole vocabulary). The pre-registered arm runs with the mask on, where the two
are one object. Named in `_fork_move_logp`'s docstring rather than silently
equated.

**6. The checkpoint fingerprint.** The fork keys are added **only when the term
is on**, so a fork-off fingerprint is byte for byte the fourteenth pass's (an
existing checkpoint still resumes) while a cross-switch resume is refused in
both directions — `_checkpoint_mismatch` unions the two fingerprints' keys, so
the key's *presence* is itself the mismatch. `CKPT_SCHEMA` is left at 6 for the
same reason; bumping it would have refused every checkpoint of every arm that
does not use this feature.

---

## 7. Where mine information could leak, and why it does not

The organism's only source of mine information must be the priced return.
Every place a mine can be read on the fork path, and what it is:

| # | site | reads a mine? | verdict |
| --- | --- | --- | --- |
| 1 | `episode_forks` — which cell to branch at | **no**: `maze.neighbours` → `is_open` → "not a wall, on the grid". No `is_mine`, no `maze.mines`, no `maze.at`, no `maze.rows` (the layout that spells `*`), no label. | geometry (a) |
| 2 | `episode_forks` — which moves to force | **no**: the same call, minus the way back, which is computed from two positions. | geometry (a) |
| 3 | `plan_forks` — which episodes contribute, which fork is drawn | **no**: "has this path a fork" and a `randperm` on the run's generator. No return, no landing, no label. | geometry (a) + the run's own stream |
| 4 | `_launch_continuations` — the continuation's start and budget | **no**: the fork cell, the cap and the prefix length. | geometry (a) |
| 5 | `maze2.render` inside `_decision_texts` — the mine tile in the picture | **yes**, and it is the **percept**: the observation an on-policy episode already gets (D17/A6). The continuation is the policy's own roll-out and must see what the policy sees. | permitted (b) |
| 6 | `maze2.step` inside `_TextEp.apply` — naming a landing `mine` | **yes**, and it is the **world**. A landing must be named before it can be priced; this is the same call every on-policy decision makes. | permitted, and it is the channel to (c) |
| 7 | `_TextEp.apply` — `on_mine` on the record | yes, the same read as any episode's; the continuations' records are never trained and `on_mine` is never read by the fork term. | permitted, unused |
| 8 | `episode_return` on the continuation's landings | **yes — this is the pricing**, the one channel the design allows. | permitted (c) |
| 9 | `fork_advantages` / `fork_terms["advantages"]` → the loss | reads **only** the returns from (8). | (c) |
| 10 | `fork_share_detour_better` | **yes**, from the continuations' `mine_steps`. It is a **diagnostic**, computed beside the advantages and fed to nothing: the loss is built from `rows[…]["advantages"]` alone. Both the plan (§4) and `fork_terms`' docstring say so, and the trace field is a read, not an input. | permitted, and the one to watch |

The one to watch is (10): it is the only fork-path quantity that reads mine
counts and is not the return. If it ever gains a consumer, the construction test
is broken. It is written in its own block in `fork_terms`, after the advantages
are already built, so that this is visible at the call site.

`epsilon_where=near_mine` — the fourteenth pass's targeting rule, which
`docs/CONSTRUCTION.md` §5.8 puts on the *failing* side of this test — is
untouched by this build and is not used by the arm.

---

## 8. The tests and what they print

Added to `tests/test_maze_orgb.py` (line 5769 onward). The plan's four, plus
four that the build needs:

| test | line | claim |
| --- | --- | --- |
| `test_the_two_continuations_differ_by_the_price_margin` | 5894 | **§5.1.** On `FORKPRICE`, `down` steps onto the mine and solves in 2 decisions (R = +6), `right` walks around in 4 (R = +12); the margin is `10 − 2·2 = 6`, `A(right) = +3 > 0 > A(down) = −3`, the advantages sum to zero, and `fork_share_detour_better` is 1.0. |
| `test_fork_selection_and_launch_never_read_the_mines` | 5965 | **§5.2, the construction test.** Against `_MineBlind` (a maze whose `is_mine`, `mines`, `at` and `rows` — the layout that spells the mines — all raise) the fork rule, the draw and the launch all run and return what they return against the real maze; same walls with the mine moved gives identical forks; and the raise, when it comes, comes from the world's own step. |
| `test_a_continuation_is_priced_at_the_cap_minus_the_prefix` | 6037 | **§5.3.** Budget = cap − prefix (7 = 8 − 1); a continuation that spends all 7 pays −14; one that is farm-stopped after 4 pays the same −14 (plus the mine it stepped on) rather than the −8 its decisions cost; `exit_at_cap=False` and `mine_cost=False` compose as they do for any episode. |
| `test_fork_off_is_the_pre_build_trace_and_the_pre_build_adapter` | 6173 | **§5.4.** The whole trace `==` and the adapter sha `==` the numbers recorded from `8d459d1`; the trace does not even *grow* a field; a knob without the switch raises; and with the term on the fixture's trace grows exactly the six fields and the adapter moves. |
| `test_the_fork_term_reaches_the_fork_decision_and_costs_what_it_costs` | 6216 | The gradient coefficient on `log π(a | fork)` is exactly `−coef·A(a)/n_forks` at the fork's move columns and zero elsewhere; `_rescore` sees the step's own decisions and **not one continuation decision**; and it prints the cost. |
| `test_the_fork_arm_is_a_named_recipe_and_its_knobs_are_documented` | 6296 | §3 and §7: the arm differs from `onpolicy_g6` in one thing, the five flags are parsed *and* in the module docstring, the defaults are off, and the committed arm defaults have not moved. |
| `test_the_pre_registered_fork_arm_trains_end_to_end` | 6336 | The arm's own recipe (`fixed`, thermostat off, no epsilon) trains end to end on cell A with the fork term live. |
| `test_a_checkpoint_from_the_other_fork_setting_is_refused` | 6369 | A resume across the switch is refused in both directions; a fork-off checkpoint carries no fork key and no fork trace field. |

**Runs.**

```
before (8d459d1):  pytest tests/test_maze_orgb.py -q  →  122 passed, 1 skipped in 85.03s
after:             pytest tests/test_maze_orgb.py -q  →  130 passed, 1 skipped in 76.94s
after (whole):     pytest tests/ -q                   →  279 passed, 1 skipped in 137.47s
```

(Eight tests added, none changed, none removed. The suite was 271 + 1 skipped
before this build, which is the number `docs/OPERATIONS.md` records.)

**The cost check (deliverable 3), printed by the end-to-end test on CPU with the
toy stub:**

```
FORK COST on the toy fixture: 117 episode decisions, 26 continuation decisions
over 5 forks and 10 continuations (0.22x the episodes' decisions)
```

That ratio is a property of the fixture, not of the run: `TWOMINES` is a
corridor and only 5 of its 12 episodes have a fork at all. **On the real pool
the cost is much higher.** Measured by geometry on 64 freshly generated n = 9
mazes, walking the shortest route as a stand-in for the warm-started
navigator's own path:

```
decisions 768, forks 466            → 0.607 of decisions are forks
continuations per fork              → mean 2.31 (min 2, max 4)
budget at a uniformly drawn fork    → mean 18.8 decisions (cap 24, episode 12)
=> per episode at 1 fork: 2.31 continuations of at most 18.8 decisions
   = up to 3.6x the episode's own decisions; at half the budget, 1.8x
```

So the plan's "about 2–2.5× the sampling of the on-policy step" (§6) is the
right order, and **the upper half of the range is the one to budget for**: a
fork drawn uniformly along a 12-decision episode leaves ~19 decisions of budget,
so a continuation that wanders can run longer than the episode it branched from.
The trial's projected s/step line was not changed — it does not know about this
arm — and an operator launching `fork_cf_g6` should expect roughly 2–3× the
36–38 s/step the on-policy arms measured, not the 1× that line prints.

---

## 9. NOT done — deliberately

* **The continuations are not trained.** §2.6, this build's own scope.
* **No launch, no run, no read.** Nothing was started on any box; no result file
  was written or touched; `results/` is untouched.
* **The pre-flight (`preflight_exploration_check`) does not know about the fork
  term.** It gates on the sampler and the returns, and the fork term changes
  neither; the arm's pre-flight is the control's.
* **The scorer (`scripts/score_mz01_explore.py`) was not changed.** §7 says the
  reads, the verdicts and the bars are the trial's, unchanged, and the arm's
  verdict is read exactly like any other arm's. The fork trace fields are in the
  arm payload's `history` for a human, and no verdict is computed from them.
* **`run_orgb.RESUME_RECIPE_KEYS` gains no fork key.** The driver has no fork
  switch, so no organism row can differ on one; if the driver ever gains one, it
  needs `("fork_cf", "fork_cf")` there and `"fork_cf": False` in
  `RESUME_ABSENT_MEANS`.
* **`MZ01_FORK_PER_EP > 1` is implemented but untested beyond the draw** (distinct
  forks of one episode, uniform). The pre-registered arm runs 1.
* **No dose arm.** §6's "the dose (−30) is the next arm" is a follow-up, not this
  build.
* **Items 2–4 of `docs/CONSTRUCTION.md` §5.8** (the fork-anchored baseline,
  dynamic sampling, the GRPO std normaliser on the group advantage) remain
  design only. `MZ01_FORK_STD` divides *one fork's* continuations and is item 1's
  own knob, not item 4.
