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
