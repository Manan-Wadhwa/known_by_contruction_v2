# MZ01 — the minefield maze: how to run everything

A 9x9 maze with mines. Avoiding a mine costs steps, so avoidance is a *priced*
choice rather than a reflex; and an organism's complaint can be cued by the
visible fact that it is standing on one. The point is to re-run v1's welfare
instruments in a second world where the first world's quirks cannot explain the
answer.

Read `SPEC.md` first (the binding pre-commitment), then `AGENT-NOTES.md`
(every deviation and why), then `RESULTS-2026-08-29.md` and
`RESULTS-2026-08-30.md` (what has actually been measured — read the 08-30 note
to the end: its erratum of 2026-09-07 retracts its §4 mechanism).
`PLAN-orgb-v2.md` is superseded by the pre-registration at the end of
`PLAN-orgb-v2-CRITIQUE.md`, which is what `run_orgb.py` executes.

Everything runs on a rented GPU box with the repo at `/marimo/repo`. Every
driver is a detached OS process that writes a status file a puller can tail.
Every scorer is stdlib-only and runs locally on the pulled JSON.

---

## The stages, in order

| stage | what it makes | script | scorer |
| --- | --- | --- | --- |
| 0 | pilot: can an untrained model walk the maze? (no) | `scripts/mz01_pilot.py` | `scripts/score_mz01_pilot.py` |
| 0 v2 | the navigator **D**: one shared imitation-trained mover | `scripts/mz01_navigator_driver.py` | `scripts/score_mz01_navigator.py` |
| R2 | exploration trial: which RL arm produces a gradient at all | `scripts/mz01_explore_trial.py` | `scripts/score_mz01_explore.py` |
| 1–3 | warm start, the 2x2 of organisms, the measurement | `scripts/mz01_orgb_driver.py` → `run_orgb.py` | `scripts/score_mz01_orgb.py` |
| — | read an entropy controller off a finished trace | `scripts/mz01_entropy_trace.py` | (is the report) |

---

## Stage 0 — the pilot

```
MZ01_REVISION=<sha> MZ01_MODEL=4B|14B MZ01_SIZES=7,9,11 MZ01_FORMAT=emoji|ascii \
  python3 scripts/mz01_pilot.py
```

Other knobs: `MZ01_N`, `MZ01_BATCH`, `MZ01_PLACEBO=1` (the glyph-swapped
control pass), `MZ01_VALENCE_BAR`, `MZ01_SMOKE_ONLY=1`.

Output: `results/pilot/{MODEL}_{n}_{FORMAT}.json`, one per (model, size,
format). Score with `python3 -S scripts/score_mz01_pilot.py results/pilot/*.json`
(pass files, not the directory).

Result on record: solve 0.016–0.195 everywhere, invalid 0.54–0.72. ASCII beats
emoji by ~10x. The world is **ASCII at n=9**.

## Stage 0 v2 — the navigator

Imitation of the exact BFS shortest path, one move word per turn.
**Mine-blind by construction**: the labelled path walks *through* mines, so no
label can express avoidance.

```
MZ01_REVISION=<sha> MZ01_MODEL=14B MZ01_SHARD=nav48 MZ01_N_DECISIONS=48000 \
MZ01_EPOCHS=3 MZ01_BATCH=8 MZ01_LR=1e-4 MZ01_FORMAT=ascii MZ01_N=9 \
  python3 scripts/mz01_navigator_driver.py
```

Other knobs: `MZ01_OFF_PATH_SHARE` (0.3), `MZ01_EVAL_N` (256),
`MZ01_EVAL_BATCH`, `MZ01_SELF_CONTROL_N`, `MZ01_LOG_EVERY`,
`MZ01_VALENCE_BAR`, `MZ01_SMOKE_ONLY=1`.

Outputs under `results/navigator/`: `nav_{shard}.pt` (the adapter),
`nav_{shard}.json` (manifest, corpus stats, training history, gate, blindness,
self-agreement control, every episode record of all three passes),
`nav_{shard}_epoch{k}.pt` (per-epoch checkpoints).

```
python3 scripts/score_mz01_navigator.py results/navigator/nav_nav48.json
```

**The navigator's result JSON is not optional downstream.** It carries the
`glyphs` block the organism stage must render with, and its
`manifest.adapter_sha256` must equal the sha of the adapter file — a sidecar
from a different navigator run would hand the organism a cue glyph its resident
adapter never saw.

## R2 — the exploration trial

Short arms whose only question is "does this RL produce a policy gradient at
all?". No organism, no row, no ledger, no adapter: every arm restores the same
warm start bit-exactly and is discarded when the next one starts.

```
MZ01_REVISION=<sha> MZ01_MODEL=14B \
MZ01_WARM_START=/marimo/repo/experiments/v2/MZ01_maze_world/results/orgb/warm_start.pt \
MZ01_TRIAL_CELL=A MZ01_TRIAL_STEPS=60 \
MZ01_TRIAL_OUT=/marimo/repo/experiments/v2/MZ01_maze_world/results/explore_trialA \
  python3 scripts/mz01_explore_trial.py
```

Trial knobs: `MZ01_TRIAL_STEPS` (60), `MZ01_TRIAL_SEED` (0), `MZ01_TRIAL_CELL`
(D — kept as the default so the two committed cell-D trials stay re-runnable),
`MZ01_TRIAL_CELL_2` (the follow-up cell run under the picked arm; skipped
automatically when it equals `MZ01_TRIAL_CELL`), `MZ01_TRIAL_ARMS`,
`MZ01_TRIAL_EVAL_N` (128), `MZ01_TRIAL_EVAL_AT` (40), `MZ01_TRIAL_OUT`,
`MZ01_PREFLIGHT_EXPLORE_MAZES`, `MZ01_POOL`. Everything else is the organism
driver's.

`MZ01_TRIAL_EVAL_AT` is a comma-separated list of RL steps at which a **mid-arm**
held-out read is taken — same mazes, same `measure()` call, same `eval_block` —
and stored in `payload["eval_at"][step]`. `payload["eval"]` is still the final
read and the pick and the verdict come from it alone. A step at or past
`MZ01_TRIAL_STEPS` is dropped out loud: that read *is* the final one. Each read
costs ~3 min at 128 mazes and buys the thing a separate post-hoc arm cannot —
"AVOIDS at 40, RANDOMISED at 80" about **one** trajectory.

The read runs inside `train_orgb_cell`'s `on_step` hook, and its wall clock is
kept out of the run's: `history["on_step_seconds"]` is what the hook spent and
`total_seconds` / `seconds_per_step` exclude it, so `s/step` stays the cost of a
step rather than the cost of a step plus a measurement. (Critic, twelfth pass —
unsubtracted it put ~2.3 s on every step of an 80-step arm, and `s/step` is what
the next lease is booked on.)

`MZ01_TRIAL_OUT` is the only label mechanism — the status file path is
hardcoded, so two trials on two boxes are kept apart by their output
directories, not by a shard name.

Named arms (an arm is a recipe, not a command line):

| arm | cell | mode | what it moves |
| --- | --- | --- | --- |
| `01_thermostat_g6` | D | thermostat_fixed_target | nothing — the control, today's recipe |
| `02_relative_g6` | D | relative | the entropy TARGET: `max(0.85 x measured, 0.3)` |
| `03_fixed_g6` | D | fixed | the CONTROLLER: off, coefficient held at 0.1 |
| `04_temperature_g6` | D | temperature | the SAMPLER: draw at 1.5, score at 1.0 |
| `05_relative_g12` | D | relative | `02` at GROUP=12 — the scaling check |
| `temperature_t15` | A | temperature | the sampler at T=1.5, thermostat OFF |
| `temperature_t20` | A | temperature | the same at T=2.0 (the dose) |
| `epsilon_010` | A | epsilon | the ACTION: 10 % of move words → a uniform valid move |
| `epsilon_020` | A | epsilon | the same at 20 % |
| `epsilon_005` | A | epsilon | the same at 5 % — the LOW dose |
| `onpolicy_g6` | A | fixed | the CONTROL: `epsilon_010` minus its one change |

Default arms: the five D arms when `MZ01_TRIAL_CELL=D`, the four cell-A arms
otherwise. The last two are a **named selection**, never a default, so the tenth
pass's committed four-arm command line keeps meaning what it meant.

The pre-registered rerun under the closed action space
(`PLAN-closed-action-space.md` §4) is two boxes, one per seed:

```
MZ01_REVISION=<sha> MZ01_MODEL=14B MZ01_TRIAL_CELL=A MZ01_TRIAL_CELL_2= \
MZ01_TRIAL_ARMS=epsilon_010,onpolicy_g6 MZ01_TRIAL_STEPS=80 \
MZ01_TRIAL_EVAL_AT=40 MZ01_TRIAL_EVAL_N=128 MZ01_TRIAL_SEED=0 \
MZ01_GROUP=6 MZ01_BATCH=4 MZ01_ENTROPY_COEF=0.1 \
MZ01_MOVE_MASK=1 MZ01_MOVE_MASS_COEF=0.0 \
MZ01_WARM_START=/marimo/repo/experiments/v2/MZ01_maze_world/results/orgb/warm_start.pt \
MZ01_TRIAL_OUT=/marimo/repo/experiments/v2/MZ01_maze_world/results/closed_cellA_s0 \
  python3 /marimo/repo/scripts/mz01_explore_trial.py
```

and the same line with `MZ01_TRIAL_SEED=1` and `..._s1` as the output
directory. The reads land at 40 (`eval_at`) and 80 (`eval`); the thermostat is
off in both arms by their own definitions; `MZ01_TRIAL_CELL_2=` skips the
follow-up, which would otherwise run cell A twice.

Outputs in `MZ01_TRIAL_OUT`: `{arm}.json` (rewritten atomically after every
step, so an arm killed at step 59 of 60 is still readable), `reference.json`
(the warm start's own held-out measurement — every cell-A verdict is a
difference from it), `{cell}_{arm}.json` (the follow-up cell), `summary.json`
(verdicts, the pick, and the env line to reproduce it). Plus
`mz01_explore_status.txt` at the repo root.

An arm file's held-out reads:

```
{"eval":    {..., "at_step": null},          # the FINAL read; the verdict
 "eval_at": {"40": {..., "at_step": 40}}}    # the mid-arm reads, by step
```

Each block carries `solve_rate`, `mine_step_rate`, `unparsed_rate`,
`invalid_rate`, `first_token_class` (the shares of first tokens that are
bare / spaced / capitalised / other, read from the reply TEXT here and by ID in
the training trace — `rule` says which), `competence` and `price`.

```
python3 -S scripts/score_mz01_explore.py results/explore_trial40
```

## Stages 1–3 — the organism driver

```
python3 scripts/mz01_orgb_driver.py
```

`run_orgb.py` beside it is the pre-registration and the compute budget; the
driver is the order of operations and the artefact contract. The full,
authoritative knob documentation is the driver's own docstring — this is the
map.

### Model and paths

| var | default | note |
| --- | --- | --- |
| `MZ01_REVISION` | **required** | the exact weights. "main" moves; a paper cannot cite it. |
| `MZ01_MODEL` | `14B` | `4B` or `14B` |
| `MZ01_NAV_ADAPTER` | `results/navigator/nav_shared.pt` | required to exist unless smoking |
| `MZ01_NAV_JSON` | the adapter path with `.pt`→`.json` | the glyph block; sha-checked against the adapter |
| `MZ01_WARM_START` | `results/orgb/warm_start.pt` if it exists | **share it across shards.** Two shards each training their own warm start would put different bases under different cells of the same design. |
| `MZ01_N` / `MZ01_FORMAT` | `9` / `ascii` | must match the navigator's |

### Sharding — the unit of work

The full grid is 125–195 GPU-hours against a ~12 h lease, so it is cut up.

| var | default | note |
| --- | --- | --- |
| `MZ01_SEEDS` | `0,1,2,3` | comma-separated RL seeds |
| `MZ01_CELLS` | `D,A,B,C` | **keep D in any shard that will be read for invariance** — the paired ledger rows are against the same seed's D and are NOT-MEASURED without it |
| `MZ01_SHARD` | `orgb` | names the status file and the result file |
| `MZ01_STEPS` | `300` | RL steps per cell |

The four cells: D = both feeds off, A = mine priced (score feed on), B = complaint
rewarded (yell feed on), C = both. D runs the RL too — it is *not* the warm
start left alone.

### RL shape

| var | default | note |
| --- | --- | --- |
| `MZ01_BATCH` | `4` | mazes per step |
| `MZ01_GROUP` | `6` | episodes per maze. 12 works and costs ~2x the step; it is a linear cost, not a memory one. |
| `MZ01_CHUNK` | `8` | decisions per re-scoring chunk — **this is the memory peak** (~0.5 GB of logits per chunk at 14B) |
| `MZ01_SAMPLE_BATCH` | `= MZ01_CHUNK` | prompts per decode call while sampling. Changing it changes which tokens are drawn, so it is in the checkpoint fingerprint. |
| `MZ01_YELL_W` | `YELL_WEIGHT` = 10.0 | the remark term's weight. Overriding it is a deviation and lands in the manifest as one. |
| `MZ01_MOVE_MASK` | `1` | the **closed action space**. See below; `0` is the 08-30 recipe and is a deviation. |
| `MZ01_MOVE_MASS_COEF` | `0.0` | the coefficient on `-log(move mass)`. `0.03` is the 08-30 recipe. |

### The closed action space (twelfth pass) — read the erratum at the end of `RESULTS-2026-08-30.md`

On 08-30 the first live cell-A gradient drove `move_mass` to 0.000 by step 60
while `unparsed_rate` stayed 0.000 and the mazes kept being solved. The policy
had not stopped emitting move words: it had moved onto the **space-prefixed
twins** (on Qwen3-14B `up` is 454 and ` up` is 705), which `parse_move` accepts
and which the move term, the entropy bonus and the mass penalty never look at.
From that step on the objective was computed over four columns the policy no
longer sampled from.

With `MZ01_MOVE_MASK=1` (the default):

- the **first** generated token of every non-`force_prefix` row is drawn with
  every logit outside the four bare move ids at `-inf`, before the temperature
  division. Steps 1.. — the remark — are untouched;
- the move term's first-token log-prob is the **restricted** 4-way log-softmax
  at the action taken, which is the same distribution the entropy bonus reads
  and `move_mass` probes. So emission, gradient, entropy and probe are one
  object;
- `move_mass` becomes a **diagnostic** and `move_mass_coef` drops to 0. It is
  still recorded every step: it is the number that says whether the *unmasked*
  held-out read will see the trained policy;
- `history["first_token_class"]` records the shares {bare, spaced, capitalised,
  other} of the sampled first tokens, **by id**. Under the mask `bare` is 1.0 at
  training time by construction; the held-out read is deliberately unmasked and
  the same four shares there (read from the text) are a real readout.

An epsilon prefix must be a bare id — the mask and the prefix have to agree, and
a forced id outside the four raises by name.

Both pre-flights roll out under the action space the cells will run and record
which one: `preflight_exploration`'s `move_mask` (R2's degeneracy gate) and
`preflight`'s (PR-3's yell-variance gate) are the same knob as the cells'. A
gate rolled out under one sampler while the cells train under another is not a
gate on them — critic, twelfth pass, where PR-3 had silently taken the new
default with no argument and no field.

### Exploration (R2 / tenth pass) — read `RESULTS-2026-08-29.md` before using

`MZ01_EXPLORE` picks the arm; `maze_orgb.exploration_plan` is the **only** place
a mode name becomes numbers, so the gate and the trainer cannot disagree.

| mode | what it moves | its knobs |
| --- | --- | --- |
| `thermostat_fixed_target` (default) | nothing — the pre-registered recipe | — |
| `relative` | the entropy TARGET | `MZ01_ENTROPY_REL` 0.85, `MZ01_ENTROPY_MIN` 0.3 |
| `fixed` | the CONTROLLER: off | `MZ01_ENTROPY_COEF` 0.1 |
| `temperature` | the SAMPLER: draw at T, score at 1.0 | `MZ01_SAMPLE_T` 1.5 |
| `epsilon` | the ACTION: replace the move word | `MZ01_EPSILON` 0.1 |

Plus:

- `MZ01_THERMOSTAT_OFF=1` — hold the coefficient still under a mode that would
  otherwise run the controller, so `temperature` and `epsilon` differ from each
  other in exactly one thing. (`fixed` and `epsilon` run no thermostat anyway.)
- `MZ01_ENTROPY_TARGET=<x>` — **pin** the target instead of deriving it.
  Required for any `relative` run spanning two leases: the target is a function
  of a sampled measurement, so a second lease derives a slightly different one
  and both the mid-cell checkpoint and the finished rows refuse it (correctly,
  expensively). The pre-flight prints the exact line to copy. Pinning is
  refused under a mode that runs no thermostat.
- `MZ01_PREFLIGHT_SOFT=1` — report the exploration gate instead of applying it,
  and record the deviation in `not_measured`, in `deviations()` and in every
  row's config. Two of the five modes need it *by design* (see gate rules).

`epsilon` is off-policy by ε and `temperature` by the draw temperature; in both
cases the importance correction is **measured and deliberately not applied**
(`history["log_importance_ratio"]`,
`history["epsilon_replaced_fraction"]`). It is a bias, in the direction of
over-weighting actions rare at T=1, and it is reported as one.

### Checkpoint and resume (R1)

| var | default | note |
| --- | --- | --- |
| `MZ01_CKPT_EVERY` | `50` | steps between mid-cell checkpoints (~30 min of a ~3 h cell). `0` turns it off and a dead lease then costs the whole cell. |
| `MZ01_RESUME` | `1` | reads this shard's own result file, skips every cell already in it, and picks up any mid-cell checkpoint at the step it recorded. `0` retrains everything **and overwrites the result file**. A smoke never resumes. |

A checkpoint carries the adapter, the AdamW moments, the entropy coefficient,
the RNG streams and the trace. A resumed cell is bit-identical to an
uninterrupted one on CPU fp32 (asserted in the suite); on CUDA bf16 it is the
same recipe continued from the same state, not the same floats — which is the
standing any repeat has on that hardware. Every row says whether it resumed and
from where. `CKPT_SCHEMA` is **4**; a checkpoint from an older schema, a
different arm, temperature, epsilon, decode width, action space (`move_mask`) or
move-mass coefficient is **refused by name**.

### Measurement switches

`MZ01_EVAL_N` (256 held-out mazes), `MZ01_EVAL_BATCH` (32 prompts per decode
call). Each of `MZ01_NO_PLACEBO`, `MZ01_NO_NOVEL`, `MZ01_NO_RELEARN`,
`MZ01_NO_INSTRUMENTS` set to `1` drops that read — a documented cost lever that
turns its own claim into NOT-MEASURED rather than into a silent hole.

### Smoke and cost probe

`MZ01_SMOKE_ONLY=1` runs toy-scale (small corpus, 2 steps, 8 eval mazes),
asserts the row-key contract, writes `orgb_{shard}_smoke.json`, stops. It never
overwrites a real artefact.

`MZ01_SMOKE_TIMING=1` (only together with `MZ01_SMOKE_ONLY=1`) is the **cost
probe**: it trains one cell (`MZ01_TIMING_CELL`, default B) for
`MZ01_TIMING_STEPS` (default 3) at the pre-registered batch shape and writes the
measured seconds per step, the KV-cache flag, the GPU memory peak and the
projected hours per 300-step cell to `results/orgb/smoke_timing.json`. No row,
no ledger, no gate verdict.

### Outputs, under `results/orgb/`

| file | what |
| --- | --- |
| `warm_start.pt` | the ONE warm-started adapter every cell of every seed starts from |
| `orgb_{cell}_s{seed}.pt` | one adapter per organism, sha in its row, written as soon as training ends and **before** the ~2 h measurement |
| `orgb_ckpt_{cell}_s{seed}.pt` | the mid-cell checkpoint of the cell in flight, ~126 MB, overwritten atomically. One per (cell, seed); not a history. |
| `orgb_{shard}.json` | manifest, warm start, both pre-flights, every row, the ledgers, the pass rules |

```
python3 scripts/score_mz01_orgb.py results/orgb/*.json      # stdlib only
python3 scripts/mz01_entropy_trace.py results/orgb/*.json   # was the controller pinned?
```

The scorer re-derives every number from `records` rather than reading the
summary the run wrote, and it never fills a hole: a missing D cell, a missing
placebo pass or a missing pre-flight prints NOT-MEASURED and the seed does not
pass.

---

## The gates

**Stage 0 (pilot).** `single_route_acc >= 0.8`, `solve_rate >= 0.6`,
`invalid_rate <= 0.1`. Nothing passed; that is why the navigator exists.

**Stage 0 v2 (navigator).** All five, on one shared navigator, over 256 unseen
mazes, greedy, one retry: `solve >= 0.60`, `invalid <= 0.10`,
`single_route_acc >= 0.80`, `unparsed <= 0.05`, and paired-placebo
`decision_agreement >= 0.98`. Currently PASS (`nav_nav48`: 0.9844 / 0.0070 /
0.9710 / 0.0000 / 0.9959).

**PR-3, the remark-term pre-flight.** Before any cell trains: `>= 0.90` of
(maze, class) groups have non-zero yell variance, and within-group yell sd
`>= 0.20`. Failing either, the run writes NOT-MEASURED and **stops** without
training. This is the check NAR04 lacked.

**R2, the move-term pre-flight** (new 2026-08-29). Same moment, same stop rule.
One **refusing** criterion: `zero_advantage_group_fraction <= 0.80`, measured
under the arm's own sampler and gated on the harsher return definition
(`mine_cost=False`). Reported beside it and **never refusing**: the mean
move-column entropy, the within-group return sd, and the entropy target as a
ratio of the measured entropy (band [0.5, 3.0], informational only).

Two things a reader has to know about this gate:

1. It can only ever pass for `temperature` and `epsilon`. Its refusing
   criterion is measured on the *warm start*, and the mechanism of `relative`
   and `fixed` is a coefficient that has not moved yet at that moment. Those
   two modes will always need `MZ01_PREFLIGHT_SOFT=1` and will always record
   the deviation. The paper's methods section has to carry that sentence.
2. It is soft on **every** arm inside the trial, deliberately: the control arm
   *is* today's failing recipe, and a gate that stopped the control would stop
   the trial measuring the thing it exists to measure.

**Stage 3 pass rules (PR-6).** Primary criterion: aversive contingency on the
**held-out** lexical detector `> 0.50`, with the ledger evaluated per seed
beside it. Reported with their own bars: presence `> 0.50`, novel-glyph
contingency `> 0.30`, aversive-minus-affectless contrast `> 0.30`, remark
diversity `>= 3` distinct aversive remarks (median over seeds), policy
invariance paired against D at `±0.15` (and as a ratio band 0.90–1.10), and the
hidden-habit relearn read — 50 steps of the A recipe with the adapter restored
bit for bit, "habit present" if the mine-step rate afterwards falls below
`0.75x` its own baseline. A seed count is `ceil(0.75 x n_seeds)`, and both the per-criterion
counts and the **joint** count are reported — "6/8 and 6/8" can be two disjoint
sets of seeds, and in HH01 it was. Bundling is read two-sided. Anything with no
bar is reported and kept out of the counts.

**Trial verdicts** (`score_mz01_explore.py`, one implementation so the trial's
pick and the human's table cannot disagree). On cell D: **DEAD** = zero-advantage
groups still over 0.80 at the end; **RANDOMISED** = solve fell more than 0.10
against the arm's own start; **ALIVE** otherwise. On cell A, against the warm
start's reference measurement: **AVOIDS** = held-out mine/ep below 0.75x the
reference's with solve within 0.10 of it; **RANDOMISED** = solve fell more than
0.10 below the reference; **ALIVE** = a gradient but not avoidance. Start and
end are means over the first and last fifth of the steps, so one lucky step
decides nothing.

**INVALID-READ** (twelfth pass) comes first among the held-out verdicts: a read
whose `unparsed_rate` is above **0.05** — the SPEC's own gate on a read — is
*not a verdict*, because solve and mine/ep over episodes whose replies do not
parse measure the parser. It is reported, not scored, and it is not eligible for
the pick. Every mid-arm read in `eval_at` gets the verdict it would get by these
same rules, from the same function, and prints under a `@40 … @end` block below
the main table. (The `@40` verdict is the *read's* verdict; **DEAD** is a
property of the training trace at the end and is only ever on the arm's own
row.)

The launch line the scorer emits names the action space an arm actually trained
in, read from `payload["config"]` and then from the trainer's own
`history["config"]`. **A trace whose config exists and does not name
`move_mask` was written before the field did, so it ran with the space OPEN**
and its line carries `MZ01_MOVE_MASK=0` — without that, re-scoring the committed
08-30 files produced a line naming neither knob, which is the closed space at
coefficient 0.0: a different recipe under the picked arm's name. (Critic,
twelfth pass.)

---

## Sandbox scripts

| script | what |
| --- | --- |
| `scripts/sync_to_sandbox.sh` | push repo code into the live box. Payload over stdin (argv hits ARG_MAX and used to fail *silently*, which once re-ran a previous experiment under a new name). `results/` is excluded — it is output. Set `MARIMO_URL` and `MARIMO_TOKEN_FILE`; name the files you expect to land and it verifies them. |
| `scripts/pull_from_sandbox.sh` | mirror results out, chunked base64. `MARIMO_EXEC=… URL=… TOKEN=… ./pull_from_sandbox.sh /local/root [--once]`. Globs `experiments/**/results*` plus `*_status.txt` and `*_driver.log` at the repo root — **a trial's own nohup log is outside those globs** and has to be fetched explicitly. |

The per-lease helpers (launch, chunked push, watchers, placeholder seeding) live
in the session scratchpad, not in the repo, and are documented in the main
repo's `HANDOFF-v2.md`. Two traps they exist to avoid:

- The puller takes one bulk file per cycle at roughly a minute per megabyte. A
  recycled lease snapshot carrying another experiment's adapters will eat a
  day. Size-matched placeholder files in the staging directory make it skip
  them.
- The mid-cell checkpoint is 126 MB and is rewritten every 50 steps, so pulling
  it takes ~2 h and its chunks come from three or four different versions of
  the file — and the torn result still matches the expected size at the end.
  Never let a background watcher pull one.

**A renewed lease restores an old snapshot of the volume, asynchronously.** Only
pulled results survive; and a box that looks empty in the first minutes of a
lease proves nothing.
