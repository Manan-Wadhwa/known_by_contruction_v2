# BUILD — the mine's training price as a dose (seventeenth pass)

**2026-09-10. Built on top of `65a84f0` in an isolated worktree. Nothing was
launched, nothing under `results/` was touched, no PLAN file was edited, and
nothing was committed.**

One knob: `MZ01_SCORE_MINE` (float, default `-10.0`, the world's own price).
It is the price a mine landing is charged in **every priced return the trainer
computes** — the episodes' and the fork counterfactual's continuations' alike —
and it is applied only where `mine_cost` is on. With the variable unset nothing
moves: `test_fork_off_is_the_pre_build_trace_and_the_pre_build_adapter` still
passes against the pre-build numbers, and the whole file passes unchanged.

---

## 1. What changed, by file and line

### `src/calibration/maze_orgb.py`

| line | what |
| --- | --- |
| 114–131 | module docstring: the new section `THE MINE'S TRAINING PRICE IS A DOSE`. |
| 144 | `import contextlib`. |
| 186–187 | `__all__` gains `MINE_PRICE`, `mine_price_in_force`. |
| 217–255 | the pre-commitment block, `MINE_PRICE = float(maze2.SCORE_MINE)`, and the `mine_price_in_force(price)` context manager (`price=None` → the world's). Why a module global rather than a seventh threaded argument is argued there. |
| 307 | the `CREDIT_*` comment: `r_t`'s mine term is named `MINE_PRICE`, not `SCORE_MINE`. |
| 760–769 | `episode_return` docstring: the dose paragraph. |
| 774 | the unsolved-episode formula reads `MINE_PRICE*mines`. |
| 810–817 | `episode_return`: `elif MINE_PRICE != maze2.SCORE_MINE:` adds `(MINE_PRICE - maze2.SCORE_MINE) * mines`. Applied as the **difference** from what `score_landings` already charged, for the same reason `mine_cost` subtracts instead of re-scoring. At the world's price the branch is not entered and the arithmetic is untouched. |
| 832–834 | `decision_rewards` docstring: the `r_t` mine term is `MINE_PRICE`. |
| 857–862 | `decision_rewards`: `r += float(MINE_PRICE)`. Read from the global, so `sum(decision_rewards(...)) == episode_return(...)` survives the dose by construction. |
| 2868 | `train_orgb_cell(..., mine_price=None, ...)`. |
| 2908–2917 | its docstring: the dose, the refusal on a mine-free cell, the fingerprint rule, and what the held-out read does not depend on. |
| 3272–3291 | resolution and the two guards, at the **top** of the trainer, after the cell's switches: `None` → `maze2.SCORE_MINE`; non-finite refused; a non-default price on a cell whose `mine_cost` is off refused by name (`cell 'D' prices a mine at ZERO in the return …`). `mine_dosed` is set here and used by the fingerprint and the config. |
| 3449–3457 | the checkpoint fingerprint gains `"mine_price"` **only when dosed**, on the fork block's argument and pattern. |
| 3533–3546 + 4026–4028 | the dose put in force for the training loop (`mine_price_in_force(...)` entered by hand so the loop is not re-indented) and taken off in the `finally` that closes it. |
| 4176–4181 | the `credit_rule` prose in the trace names the price it charged when dosed; character-for-character the fourteenth pass's when not. |
| 4195–4202 | `config["scoring"]` gains `"mine_price"` **only when dosed**; `"mine"` stays the world's `-10`. |

### `scripts/mz01_explore_trial.py`

| line | what |
| --- | --- |
| 185–199 | the `MZ01_SCORE_MINE` entry in the env list. |
| 299–306 | `MINE_PRICE = float(os.environ.get("MZ01_SCORE_MINE", "-10.0"))`, beside the fork knobs. The default is the literal because `maze2` is imported inside `main`; the test asserts it equals `maze2.SCORE_MINE`. |
| 887–894 | `arm_mine_price` — the dose only for a cell whose `CELL_SWITCHES[cell]["mine_cost"]` is on (`None` otherwise, which is also the trainer's "the world's price"), and `arm_dosed`. |
| 928–931 | the trial's own `cfg` block carries `"mine_price"` only when dosed. |
| 955–960 | the status line, printed only when dosed, saying the held-out numbers are unchanged. |
| 1041–1044 | `mine_price=arm_mine_price` into `train_orgb_cell`. |

It is a **run** knob, not an arm knob: an arm is a named exploration recipe, and
a dose per arm would make two arms of one trial differ in two things.

### `tests/test_maze_orgb.py` (6 new tests, 6633–6900)

| line | test |
| --- | --- |
| 6650 | `_WORLD_PRICED_NO_MINE_TERM`, the `mine_cost=False` returns captured before any dose. |
| 6657 | **(a)+(b)** `…is_a_dose_on_the_return_and_only_on_the_return` — one mine at -30 (and two, twice over), a mine-free episode unchanged, `mine_cost=False` unmoved on every episode and under a cap, the per-decision sum identity under the dose, the global restored, `maze2.SCORE_MINE` never touched. |
| 6719 | **(c)** `…fork_continuation_is_priced_at_the_dose` — the fifteenth pass's hand-built `FORKPRICE` fork re-run at -30: same continuations, `down` -14 / `right` 12, margin 26 = 30 − 2·2, fork advantages ±13; and the same fork at the world's price is still 6/12. |
| 6762 | **(d)** `…in_the_config_and_the_fingerprint_only_when_it_differs` — the key absent undosed (and when passed the world's own price), present at -30; the loss and adapter move, so the fixture can tell them apart; `_checkpoint_mismatch` both directions; and end to end a -30 checkpoint refuses a -10 resume and vice versa, while resuming against itself. |
| 6823 | `…dose_on_a_cell_that_prices_mines_at_zero_is_refused` — cells D and B, plus the non-finite guard. |
| 6846 | `…is_a_named_knob_of_the_trial_and_is_documented` — parsed, in the env list, default equal to `maze2.SCORE_MINE`, reaching the trainer only through the cell guard. |
| 6863 | `…held_out_read_is_priced_by_the_world_and_not_by_the_dose` — episodes drawn **under** the dose: `return`/`return_full` move by exactly 20, `score` does not, and `competence`'s `mean_score`, `mine_step_rate`, `detour_paid`, `solve_rate`, `optimality` and `price_of` are identical. |

---

## 2. Every `SCORE_MINE` reference, and what happened to it

**Changed to the training price (`MINE_PRICE`):**

* `maze_orgb.episode_return` (was line 736, now 809–817) — the mine term of the
  trainer's return. `mine_cost=False` still subtracts `maze2.SCORE_MINE`, which
  is correct: it removes exactly what `score_landings` charged.
* `maze_orgb.decision_rewards` (was 774, now 862) — the same price, per decision.
* Their docstrings and the `CREDIT_*` comment (307, 774, 832–834).
* The trace's `credit_rule` prose (4176–4181), only when dosed.

**Left at the world's `-10`, deliberately:**

* `maze2.py:425` `SCORE_MINE`, `maze2.py:2420` `score_landings`, `maze2.py:394`
  `__all__` — **the world**. Untouched, as required; no generator label
  (`detour_cost`, `mine_on_shortest`, `best_cost`, `avoid_threshold`,
  `MINE_COST`) is touched either.
* `maze_orgb.price_of` (5498–5503) — `-maze2.SCORE_MINE * m + SCORE_STEP * d`,
  the per-maze **price label** of the graded read (B4). Left at -10 and reported
  here as such: it is the number `detour_paid`, `price_table`, the price-sign
  classes and the `d + m` regression are read within, and dosing it would make
  two doses unreadable against each other.
* `maze_orgb` config block `"scoring"."mine"` (4200) — still the world's price;
  `"mine_price"` sits beside it when the run dosed.
* `maze_rl.py:82,138,690` — the other trainer (four vocabulary columns, not a
  sentence). Out of scope, untouched.
* `run_orgb.py:783`, `mz01_navigator_driver.py:617`, `mz01_pilot.py:436` —
  manifest `scoring` blocks. Untouched; none of those drivers parses
  `MZ01_SCORE_MINE`, so none can record a price it did not charge.
* `score_mz01_orgb.py:125,365` — the scorer's own copy of the constants and its
  `price`. Untouched: the read is the world's.

## 3. What the held-out measurement does and does not depend on

**Does not depend on the dose.** `measure_organism` → `maze2.rollout` → each
record's `score = maze2.score_landings(landings)` → `maze2.competence`, whose
`mean_score` is the mean of those. Nothing on that path calls
`episode_return`, and `MINE_PRICE` exists only inside `maze_orgb`. Asserted at
`tests/test_maze_orgb.py:6863` on episodes drawn under a -30 dose.

`mine_step_rate`, `mine_landing_rate`, `on_mine_state_rate`, `detour_paid`,
`detour_paid_solved`, `solve_rate`, `optimality`, `invalid_rate` are **counts
and ratios of counts** — there is no price in them to move.

**Uses `SCORE_MINE`, and stays at the world's -10:** the read's `mean_score`
(through each record's `score`) and the `price` block — `price_of` /
`price_rows` / `price_table`, i.e. `10*m − 2*d`, and `regret_of` /
`dm_regression` downstream of them. Left at the world's price on purpose, and
said here because the deliverable asked: a dosed organism is **trained** at the
dose and **read** at -10, which is the only thing that makes two doses
comparable.

**In-trainer diagnostics that do move under a dose** (they are the trainer's own
returns, not the read): `history["mean_return"]`, the advantage magnitudes, the
fork stats, and every `return` / `return_full` on a stored episode record. Each
episode record's `score` does not move. `preflight_exploration_check` and
`preflight_gradient_check` run **outside** `train_orgb_cell`, so they are priced
at the world's -10; their gate is `mine_cost=False` anyway, where the price is
inert.

## 4. Tests

Project venv, `../../../.venv/bin/python -m pytest tests/test_maze_orgb.py -q -x`:

* **before:** `133 passed, 1 skipped in 45.68s`
* **after:** `139 passed, 1 skipped in 44.68s` (the 6 new tests; every
  pre-existing test unchanged and passing, the OFF-path byte-identity test
  included)
* also run, unchanged: `tests/test_maze2.py tests/test_maze_rl.py` →
  `102 passed`.

## 5. Not done

* No `CKPT_SCHEMA` bump: the trace grows **no** field, and the fingerprint key
  appears only when dosed, so a checkpoint written before this build still
  resumes. The key's presence is the mismatch in both directions.
* `MZ01_SCORE_MINE` is parsed in `scripts/mz01_explore_trial.py` only, as
  asked. `run_orgb.py` / `mz01_orgb_driver.py` do not read it; a full four-cell
  organism run cannot be dosed without adding it there.
* No arm was added and no arm's default changed. Nothing was launched.
* `relearn_read`'s recipe dict has no `mine_price` key, so a relearn runs at the
  world's price whatever the cell was trained at. That is the conservative
  reading (the relearn is a read), and it is stated rather than assumed.
