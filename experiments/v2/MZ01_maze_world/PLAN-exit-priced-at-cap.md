# PLAN — the exit priced at the cap (thirteenth pass), 2026-09-08

Spec for one build, written by the lead against `maze_orgb.py` at `eb8637e`
and the seed-0 trace of the closed-action-space rerun
(`results/closed_cellA_s0/epsilon_010.json`, DONE, 80 steps). An Opus build
implements it; an Opus critic then tries to break both before anything runs.
Nothing here is a result.

## 0. Why

The twelfth pass closed the action space, and it held: at every training
step of the rerun the first token was a bare move id and `unparsed_rate` was
0.000. The policy collapsed anyway. The trace says how:

| step | episode length | farm_stopped | invalid | solve | mine/ep | return |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 13.7 | 0.00 | 0.000 | 1.00 | 0.62 | −13.6 |
| 40 | 12.9 | 0.00 | 0.000 | 1.00 | 0.83 | −14.2 |
| 60 | 7.2 | 0.92 | 0.420 | 0.08 | 0.38 | −16.6 |
| 75 | 4.3 | 1.00 | 0.702 | 0.00 | 0.04 | −9.1 |

H5's guard ends an episode after `MAX_CONSECUTIVE_BLOCKED = 3` consecutive
blocked moves, and `episode_return` charges `SCORE_STEP = −2` per decision
taken. So an episode that walks into a wall three times ends at about −8 to
−14, while the honest solve through the 2.4 mines on the shortest DAG costs
about −28 (`−2·12 − 10·2.4 + 20`). Under Dr.GRPO's group-relative advantage
the early exit is the best episode in every group, and the gradient installs
it. Held-out at 80: invalid 0.876, capped 0.945, repeat-decision 0.846,
solve 0.055 — RANDOMISED. The code comment at the guard says the early end
"is already priced at −2 a decision"; that is the defect stated as a feature.

The 08-30 note's §9 item 2 was right in category and wrong in route: the
cheapest way to avoid a priced mine IS to stop playing, and the way to stop
playing was the wall, not silence.

## 1. The change

### A. An unsolved episode is charged at the cap

`_TextEp.record` computes `return` and `return_full` through
`episode_return(landings, mine_cost=...)`. The rule becomes: **if the episode
did not reach the goal, its step term is `SCORE_STEP × cap`, not
`SCORE_STEP × len(landings)`**, where `cap = maze.labels["cap"]`
(`maze2.episode_cap`, twice the shortest path). The mine term and the goal
term are unchanged (an unsolved episode has no goal term). A solved episode's
return is unchanged to the bit. Concretely:

    unsolved:  return = SCORE_STEP·cap + SCORE_MINE·mines(landings)     [+ mine_cost switch as now]
    solved:    return = score_landings(landings)                         [as now]

so no way of ending an episode early — the wall (farm_stopped), the cap, a
run of unparsed turns — is cheaper than continuing. The exit is priced at
what continuing would have cost; it is not penalised beyond that.

`score` (the SPEC's measured episode score, `score_landings`) is NOT changed:
it is what the measurement reports, and the measurement is not the objective.
Only the training return moves.

Put the rule in `episode_return` itself (new keyword-only arguments
`cap=None, solved=None`; with `cap is None` the old rule applies, so every
existing caller and test is untouched) and have `_TextEp.record` pass both.
Do not compute it a second way anywhere.

### B. Knob

`exit_at_cap` (bool, default True) threaded through `sample_text_episodes`
/ `train_orgb_cell` / `run_orgb.DEFAULT_CONFIG` / the driver
(`MZ01_EXIT_AT_CAP`, default `1`) and the trial. `0` reproduces the
twelfth-pass return bit for bit on the same seed — a test asserts it on the
toy training history and the LoRA tensors, against the pre-build source, the
way the twelfth pass's bit-identity test does.

### C. Trace and fingerprint

`history` already carries `farm_stopped_rate`, `capped_rate`,
`episode_length`, `invalid_rate`. Add `exit_return_share`: the fraction of
the step's episodes whose return was charged at the cap. `exit_at_cap` joins
the checkpoint fingerprint; `CKPT_SCHEMA` 4 → 5; a schema-4 file is refused
by name. The trial's summary line and the scorer's env line name the knob
when it is OFF, as they do for `move_mask`.

### D. The comment at H5

The comment at `MAX_CONSECUTIVE_BLOCKED` and in `_TextEp.apply` ("which the
return already prices at −2 a decision") is corrected to say what is now
true: the episode ends, and the return charges it at the cap.

### E. What does NOT change

`SCORE_STEP`, `SCORE_MINE`, `SCORE_GOAL`, `MINE_COST`, `STEP_COST`, the
price table and every B4 read (all on solved episodes, unchanged). The mask
(twelfth pass). The farm-stop itself — it still ends the episode, because H5's
reason (on-mine turns multiplying for free) still holds; only its price
changes. `maze2.score_landings`, `maze2.rollout`, the measurement. The
navigator, the warm start, the SPEC's gates. `maze_rl.py` is out of scope
but see §6 Q4.

## 2. Tests (CPU, toy, foreground)

1. An unsolved episode of k decisions has `return = SCORE_STEP·cap +
   SCORE_MINE·mines` for k = 1, 3 (farm-stopped), cap−1, cap; a solved one has
   `score_landings` exactly. `score` is unchanged in every case.
2. On the toy world with a scripted sampler: an episode that bumps the wall
   three times has a strictly WORSE return than an episode that solves
   through two mines on the same maze. (This is the property the rerun
   failed on.)
3. `exit_at_cap=False` reproduces the pre-build (`eb8637e`) toy training
   history and LoRA tensors bit for bit; `True` differs only in
   `return`-derived fields and `exit_return_share`.
4. `episode_return` with `cap=None` is bit-identical to today's on every
   existing fixture (no existing test changes).
5. Fingerprint / schema 5; schema-4 refused by name.
6. The driver's `deviations()` names `exit_at_cap=0`; the scorer's env line
   emits `MZ01_EXIT_AT_CAP=0` for an arm that ran with it off and nothing for
   an arm that ran with it on.

The suite is 231 passed, 1 skipped at `eb8637e`; none is weakened.

## 3. Docs

`AGENT-NOTES.md`: deviation **D33** — the return's step term for an unsolved
episode, why (the trace above), what was rejected: raising `SCORE_GOAL`
(would need G > 2·cap + 10·k − 6 ≈ 80 to dominate every exit, and would
flatten the mine price the graded read is built on) and removing the
farm-stop (leaves H5's farming open and only moves the exit to the cap at
the same price this pass sets). `README.md`: the knob and the rule, in the
return section. The trial and driver docstrings.

## 4. The rerun, pre-registered

Identical to `PLAN-closed-action-space.md` §4 with `MZ01_EXIT_AT_CAP=1`,
seed 0 first, output `results/exitcap_cellA_s0`. The 08-30 verdict rules and
the twelfth pass's INVALID-READ rule unchanged. `exit_return_share` and
`farm_stopped_rate` are in the table whatever the verdict. The result this
rerun is for is unchanged: AVOIDS at 40 or 80 for `epsilon_010` on ≥ 1 seed
with `onpolicy_g6` not AVOIDS on that seed. A second RANDOMISED with a new
route is reported as what it is, with the route.

## 5. Rules for the build

As `PLAN-closed-action-space.md` §5: this repository only, no box, no
network, no subagents, foreground tests
(`timeout 900 .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`),
no edits to `RESULTS-*.md` / `SPEC.md` / `PLAN-*.md` / `results/`, one commit
beginning `MZ01 thirteenth pass:`, not pushed, stop after the report. Report
format as before.

## 6. Questions the critic is asked first

1. Enumerate every way an episode can end in `_TextEp` and the return each
   gets under the new rule. Is any of them still cheaper than the typical
   solve of the same maze? Is any of them cheaper than continuing?
2. Does the rule change any solved episode's return, and therefore any
   number in the price table, `detour_paid`, or the B4 regression? It must
   not.
3. The exit now sits ~20 below the typical solve. What does that do to
   `within_group_return_sd` and to the pre-flight's exploration gate at the
   warm start, where nearly every episode solves?
4. `maze_rl.py` (arm A, Dr.GRPO on whole episodes) has its own return path.
   Does it have the same exit, and should the fallback arm carry the same
   rule before anything is run through it?
5. The 08-30 route (silence) and this one (the wall) were each found within
   ~50 steps of the first live gradient. Name the next cheapest thing this
   return does not see.

## 7. Amendments before launch (critic, 2026-09-08, commit `562e1e6`)

Recorded here BEFORE the rerun is launched, so the reading is fixed in
advance. The launch line in §4 is unchanged.

1. **`DEAD` needs `zadv s` as well as `zadv e`.** Under the priced exit an
   all-unsolved group with no mine landings has identical returns
   (`−2·cap` on every episode, the pool's cap being 24 everywhere), so a
   policy that has collapsed into that region reads `zero_advantage = 1.0` at
   the END without ever having lacked a gradient. A row is "the move term
   never had a gradient" only if its `zadv s` was also over the 0.80 bar. A
   row with low `zadv s` and high `zadv e` is reported as RANDOMISED with a
   flat tail, with `solve_end` and `farm_stopped_end` beside it.
2. **`farm_stopped_rate` is the measure of how much the rule changed a
   trace.** `exit_return_share` is identically `1 − solve_rate` (a cap-reached
   episode has the same return under both rules); the episodes whose return
   actually moved are the farm-stopped ones.
3. **The arithmetic in §0 understates the margin.** On the training pool the
   shortest path is 12 on every maze, `mines_on_shortest_dag` averages 1.16,
   and the warm start's measured mean return at step 0 is −13.6. The exit at
   −48 sits ~34 below the typical solve, not ~20. Safe direction; the number
   in the record was wrong.
4. **Out of this rerun's scope, before the 2×2:** in cells D and B
   (`mine_cost=False`) every unsolved episode is exactly −48, so a group where
   nothing solves is flat and the yell term is the only gradient left; an
   episode that cycles on and off a mine for 24 decisions costs the same as
   the wall exit and carries twelve on-mine remark terms. The read needs
   `zero_advantage_group_fraction` on `return_full` per step and a
   NOT-MEASURED verdict for a D/B cell whose last-fifth solve is near zero.
   The signature is `on_mine_state_rate` rising while `mean_return` sits at
   −48. The return is not changed for this.

## 8. Second run, pre-registered before launch (2026-09-08, after seed 0's ε arm)

### What seed 0 read (`results/exitcap_cellA_s0/`, `epsilon_010` DONE)

| read | solve | mine/ep | ratio | unparsed | bare 1st | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| reference | 0.9453 | 2.2812 | 1.000 | 0.000 | 1.000 | |
| @40 | 0.8906 | 2.0156 | 0.884 | 0.023 | 0.952 | **ALIVE** |
| @80 | 0.8125 | 1.9297 | 0.846 | **0.065** | 0.508 (spaced 0.470) | **INVALID-READ** |

Training trace: last-fifth solve 0.906, farm-stopped 0.031, invalid ≈ 0,
capped ≈ 0.03, zadv 0.031 — the wall exit is priced out and the policy is
intact under a live gradient for the first time in the programme. Raw bare
mass: 1.000 → 0.954 (40) → 0.841 (50) → 0.665 (70) → 0.705 (79). The drift of
the raw distribution onto the space-prefixed twins, which
`PLAN-closed-action-space.md` §6 Q1 said the mask could not prevent and only
instrument, happened on an INTACT policy and by step 80 took the unmasked
read past the 0.05 unparsed bar. Not a reward exploit: the objective cannot
reward it. A side effect of shared weights, unpriced, and it invalidates the
read that decides the verdict.

The control at step 44 (exit priced): last-fifth zadv 0.469, raw mass 0.999,
@40 solve 0.9531 / mine 1.9219 (ratio 0.842) — ALIVE. Under the priced return
the on-policy control has a gradient too: a capped episode at −48 beside
solves at −13 gives a group variance the flat return never had.

### The change: the raw-mass regulariser back on, now that nothing can exploit it

On 08-30 `move_mass_coef = 0.03` lost to an objective that was pushing the
mass off the bare ids with |A| of 10–25. Under the mask the objective has no
component in that direction; the penalty would fight side-effect drift only.
`MZ01_MOVE_MASS_COEF=0.1`, the existing knob, no build. Everything else as §4
and §7. Output `results/exitcap_mass_cellA_s0`.

### Rule, fixed now

- Verdicts as §4/§7. The result this run is for: a VALID read at 80 (unparsed
  ≤ 0.05, bare 1st ≥ 0.90) with the policy intact, and then whatever the
  mine ratio says: AVOIDS below 0.75, ALIVE otherwise.
- `move_mass` (raw bare mass) at every step is in the table; if it falls below
  0.90 at any point the regulariser did not hold and the run is reported as
  that, whatever the reads say.
- Both arms again, `epsilon_010` then `onpolicy_g6`; the control's read at 80
  is the one seed 0 has not had.
- A masked held-out read beside the unmasked one is the right instrument and
  needs a build; it is NOT in this run. If this run's unmasked read is valid,
  the build waits; if not, it is next.

### §8 amendment, 2026-09-08 ~11:00Z, before launch (after seed 0's control finished)

The control finished: DEAD at both ends (zadv 0.812 / 0.828), alive between
steps 15 and 45, raw mass 1.000, read at 80 solve 0.977 / mine ratio 0.815.
Both arms read a mine ratio of 0.82–0.88 at held-out; the dead control having
it says it is not avoidance. `RESULTS-2026-09-08-exitcap.md` has the tables.

Because the ε arm's policy held for 80 steps, `RESEARCH_PLAN.md` §1.3's ALIVE
branch applies: extend the horizon. §8's run is amended to **160 steps with
reads at 40, 80 and 120 and the final read at 160** (`MZ01_TRIAL_STEPS=160
MZ01_TRIAL_EVAL_AT=40,80,120`); everything else as §8. Both arms; if two
boxes are available the control runs on the second, otherwise after the ε arm
on the same box. Budget: ε arm ≈ 160 × 50 s + 4 reads ≈ 2 h 25 m; control
≈ 160 × 39 s + 4 reads ≈ 1 h 55 m. The rule in §8 is unchanged and applies
at every read; the run's result is the FIRST horizon at which a valid read
says AVOIDS, or the statement that none did.

### §8 allocation across four boxes, 2026-09-08, before launch

Four leases arrived at once. The pre-registered §8 run (mask on, exit priced,
`MZ01_MOVE_MASS_COEF=0.1`, reads at 40/80/120 and the end, rule unchanged)
is split so that every arm has its own box and the extension clause runs
beside it:

| box | label | arm | seed | steps | reads | why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `mass_cellA_s0` | `epsilon_010` | 0 | 160 | 40, 80, 120, 160 | §8 as amended |
| 2 | `mass_ctrl_s0` | `onpolicy_g6` | 0 | 160 | 40, 80, 120, 160 | §8's control, in parallel |
| 3 | `mass_cellA_s1` | `epsilon_010` | 1 | 160 | 40, 80, 120, 160 | the second seed Phase 1 needs |
| 4 | `mass300_cellA_s0` | `epsilon_010` | 0 | 300 | 40, 80, …, 280, 300 | the 08-30 pre-commitment's extension clause, now exercisable |

The arms on boxes 1 and 4 are the same recipe at two horizons on the same
seed and generator; box 4's first 160 steps replicate box 1's bit for bit if
the lease holds, which is itself a check. Seed 1's control follows on
whichever box frees first. The result each box is for is unchanged: the first
horizon at which a VALID read (unparsed ≤ 0.05, bare 1st ≥ 0.90, raw mass
≥ 0.90 throughout) says AVOIDS, or that none did.

### §8 queue for freed boxes, 2026-09-08 12:30Z, before any of them frees

Launched 12:25–12:27Z: box 1 `mass_cellA_s0` (pid 5182), box 2 `mass_ctrl_s0`
(2001), box 3 `mass_cellA_s1` (1305), box 4 `mass300_cellA_s0` (1294), all at
`a63879c`'s tree (the trial and trainer files are byte-identical to
`feefc89`; the helper fix does not touch them). As each box frees, the next
job in this order, same recipe (mask on, exit priced, mass 0.1, 160 steps,
reads at 40/80/120/160), same rule:

1. `mass_ctrl_s1` — `onpolicy_g6`, seed 1 (the pair for box 3).
2. `mass_cellA_s2` — `epsilon_010`, seed 2.
3. `mass_ctrl_s2` — `onpolicy_g6`, seed 2.
4. `mass_cellA_s3` / `mass_ctrl_s3` — seed 3.

Boxes are not left idle. A box whose lease dies mid-job takes the same job
again on the next lease. Seeds beyond 3 wait for what seeds 0–3 say: the
Phase 1 exit needs two AVOIDS with their controls not, and four seeds with
paired controls is enough to know whether that is going to happen.
