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

### §8 queue superseded, 2026-09-08 13:10Z

`PLAN-credit-explore-navigator.md` (the fourteenth pass) pre-registers the
runs that take priority on freed boxes once its builds are critic-approved.
Until then a freed box takes the queue above so it is not idle.

### §8 addendum, 2026-09-08 14:40Z — the regulariser at 0.3, before launch

At their reads at 120 the mass-0.1 arms read: ε s0 (160) bare 0.65, raw mass
0.67 over the last ten steps; ε s1 bare 0.45, 0.68; ε s0 (300) bare 0.60,
unparsed 0.151 (INVALID-READ), 0.72. The rule of §8 says these runs are the
regulariser failing; the fourteenth pass's §7 sets 0.3 for its runs on that
evidence, untested. So the next two freed boxes run the §8 recipe itself at
`MZ01_MOVE_MASS_COEF=0.3`, 160 steps, reads at 40/80/120/160, seeds 0 and 1
(`mass030_cellA_s0`, `mass030_cellA_s1`), under §8's rule: raw mass ≥ 0.90 at
every step and bare 1st ≥ 0.90 at every read, or 0.3 did not hold either and
the fourteenth-pass runs need the masked read before they can be verdicted.
This displaces the seed-2 arms in the queue above; the controls stand.

### §8 result, and the replication, pre-registered 2026-09-08 15:55Z — before the final read

`mass_ctrl_s1` (box 2; `onpolicy_g6`, seed 1, mask on, exit priced, mass 0.1,
coefficient 0.1, no ε) read, on 128 held-out mazes against its reference
0.9453 / 2.2891:

| read | solve | mine/ep | ratio | unparsed | bare | mine landings / move | mean steps | mine-free solved | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | 0.9453 | 2.2891 | 1.00 | 0.000 | 1.00 | 0.180 | 12.7 | 0.000 | |
| @40 | 0.9453 | 2.1172 | 0.92 | 0.000 | 1.00 | 0.165 | 12.9 | 0.000 | ALIVE |
| @80 | 0.8828 | 1.7891 | 0.78 | 0.000 | 1.00 | 0.128 | 13.9 | 0.035 | ALIVE |
| @120 | 0.8828 | 1.4531 | **0.63** | 0.000 | 1.00 | 0.103 | 14.05 | 0.062 | **AVOIDS** |

Training trace by quarter: mine/ep 1.36 → 0.96 → 0.83 → 0.64; solve 0.97 →
0.92; zadv 0.76–0.80 (a fifth of groups live); entropy 0.02–0.05; raw mass
1.000 throughout. The first AVOIDS read in the programme, by the rule fixed
in §4 and §7, on the ON-POLICY arm. The ε arms on the same recipe read
RANDOMISED (seed 1) or ALIVE without avoidance (seed 0).

**Reading, fixed before the read at 160.** If the read at 160 is VALID and
AVOIDS, the on-policy recipe under the priced return installs avoidance on
seed 1 within 160 steps; if it is VALID and ALIVE, avoidance appeared at
120 and receded, and the horizon question stands; if it is RANDOMISED or
INVALID, the @120 read is reported as a single read. In every case: under
the priced return the on-policy arm's own entropy (0.02–0.05) is enough
exploration for a gradient (zadv ~0.77, not 1.0), and ε's uncorrected
off-policy weight is a cost, not a lever — "EPSILON IS THE LEVER" was a
statement about the flat return of 08-30 and is withdrawn for the priced
one.

**Replication, pre-registered now.** The trial discards its adapter, so the
organism is gone at the end of the run. The replication is the organism
driver, which checkpoints every 50 steps and measures:

- `MZ01_CELLS=A MZ01_STEPS=160 MZ01_EXPLORE=fixed MZ01_ENTROPY_COEF=0.1
  MZ01_MOVE_MASK=1 MZ01_EXIT_AT_CAP=1 MZ01_MOVE_MASS_COEF=0.1
  MZ01_CKPT_EVERY=50`, navigator nav48, the committed warm start, the same
  held-out set — on **seed 1** (the replication of this read, same seed,
  different box) and **seed 0** (the control's seed 0 read 0.91 at 160; the
  two seeds differ and both are run), output `results/orgb_ctrl_s{1,0}`.
- The trial's control arm on **seeds 2 and 3** (`mass_ctrl_s2`, `mass_ctrl_s3`),
  reads at 40/80/120/160, for the seed count.
- Rule: AVOIDS on ≥ 2 seeds with a VALID read, under the §4/§7 rule, is the
  Phase 1 signal; the driver's adapters are the organisms.
- Priority on freed boxes: these four displace R1–R4 of the fourteenth pass;
  N1 (nav48u) still runs on the next fresh lease.

### §8 — the control at seed 1, read at 160, and the day's other finals (2026-09-08 19:40Z)

`mass_ctrl_s1` @160: solve **0.750**, mine/ep **0.945** (ratio **0.41**),
unparsed 0.000, bare 1.00 — by the rule, **RANDOMISED** (solve fell 0.195,
past the 0.10 band), so per the reading fixed above the @120 AVOIDS stands
as a single read. What the held-out competence says the policy is at 160:
mine landings per move 0.057 (reference 0.180), mine-free solved episodes
0.271, capped 0.25, mean steps 16.5, invalid 0.061, unparsed 0 — an organism
that walks around mines and a quarter of the time does not reach the goal
inside the cap. Not random; over-cautious, past the band the rule prices.
Training by quarter: mine/ep 1.30 → 0.85 → 0.69 → 0.66; solve 0.97 → 0.92.
The organism inside the band is the one at ~120 steps; the driver
replications (launched 19:37Z on boxes 2 and 3, seeds 1 and 0, same recipe)
checkpoint at 50, 100 and 150 and measure at 160, so both sides of the band
will be on disk.

Also final: `mass030_cellA_s0` (ε 0.10, mass 0.3) RANDOMISED at every read,
solve 0.352 at 160, bare 0.92–0.97 except an INVALID read at 120 — the
regulariser at 0.3 holds the tokens and the ε arm still collapses;
`mass300_cellA_s0` (ε 0.10, mass 0.1, 300 steps) RANDOMISED from 160 on,
ratio 0.61 at 240 with solve 0.31. Every ε arm on the priced return has
now degraded within 160 steps; the on-policy arm has not. ε is withdrawn as
the lever for the priced return, as written above.

**N1:** `nav_nav48u` (labels uniform over optimal moves) trained 15:54Z →
17:47Z on the extended lease and **PASSES** the Stage 0′ gate on every
criterion: solve 0.9414, invalid 0.0116, single-route 0.9573, unparsed
0.0000, mine-blind agreement 0.9863 (11 of 256 episodes diverged, 7 at a
tie-break fork; ex-tie-break above the bar as well), TV 0.0030. Its warm
start is building on box 1 (`orgb_nav48u/`, launched 19:36Z). R5 (on-policy,
`togo`) and R6 (`epsilon_045_near`, `togo`) follow on it; given today's
control result, R5's on-policy arm is now the one to watch.

Idle time to record: all four boxes finished between 16:20Z and 17:47Z and
sat idle until 19:36Z while this session was interrupted.

### Overnight loss, 2026-09-09 04:40Z — and the cause

All four boxes were alive and mid-job at 20:52Z (02:22 IST). The pulling
host — this laptop — suspended then; every watcher's log ends at that
cycle with no "unreachable" line. The leases expired later, with the
finished work on them. Lost: the two organism-driver replications
(`orgb_ctrl_s1`, `orgb_ctrl_s0`, at step 88 of 160 at the last pull; their
checkpoints, rows and adapters were never pulled), the control seed 2's
reads at 120 and 160 (last pull at step 111), and R5's reads at 80, 120 and
160 (last pull at step 59). Kept: control seed 2 to step 111 with reads at
40 and 80 (0.906 / 0.93 ALIVE; 0.977 / 0.87 ALIVE); R5 to step 59 with its
read at 40.

The rule "pull as rows land" is only as good as the machine the puller runs
on. From here the puller runs where nothing sleeps: a long-lived CPU sandbox
or a cloud session, and the laptop is a client of that, not the pipeline.
The next lease is also checked, before anything is launched, for whether
its /marimo restores any of last night's directories (`orgb_ctrl_s*`,
`R5_*`, `orgb_nav48u`, `navigator/nav_nav48u.*`); the snapshot every lease
has come up on so far is the 08-27 one.

R5's read at 40, for the record: the nav48u warm start's own reference is
solve 0.859 / mine/ep 2.305 (nav48's warm start: 0.945 / 2.281 — the
uniform-label navigator loses solve in the warm start, which nav48 did
not); R5 @40 solve 0.703 / ratio 0.61, RANDOMISED by the solve clause
against 0.859, with training solve 0.98 and training mine/ep 0.26 over its
last ten steps and `adv_nonzero_share` 0.42 against the nav48 control's
0.05 — the new navigator gives the on-policy arm a gradient the old one did
not. A read at 80 would have said whether the held-out solve recovers; it
does not exist.

### §8 — three amendments after the documentation audit, 2026-09-09 06:40Z

1. **The replication line requires `MZ01_PREFLIGHT_SOFT=1`.** The organism
   driver's exploration gate refuses every on-policy arm at the warm start
   (its one refusing criterion cannot pass without a temperature or an ε
   draw) and trains nothing; the replication spec above omitted the flag.
   `scripts/box/launch_driver.sh` sets it and is the launcher of record;
   the runs launched 2026-09-09 04:41–04:44Z had it.
2. **The queue, restated after the 09-09 relaunch.** Boxes 2 and 4 run the
   driver replications (seeds 0 and 1); box 3 the seed-2 control; box 1
   navigator v2, then its warm start, then R5. The next freed box takes, in
   order: the seed-3 control (`mass_ctrl_s3`); R4 (return-to-go credit on
   the on-policy arm, nav48); the driver replication on seed 2 if the seed-2
   control's read at 120 or 160 is AVOIDS. Withdrawn from the queue, not to
   be run: `mass030_cellA_s1`, `mass_cellA_s2`, R1, R2, R3 (every ε arm on
   the priced return has degraded; D-15).
3. **The blindness bar, for the record.** The SPEC's 2026-08-28 amendment
   set the navigator's mine-blindness bar at "self-agreement ceiling −
   0.02"; the code gates at 0.98 absolute and always has. Both navigators
   pass both readings (nav48 0.9959; nav48u 0.9863 raw, ex-tie-break above
   it). The implemented rule is the rule of record; a reconciliation note is
   appended to `SPEC.md` today.
