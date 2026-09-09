# Glossary

Written 2026-09-09. Every term of art used in the MZ01 notes, plans and code,
with the file that defines it where one does. Alphabetical, then two tables at
the end: the **tag families** (which document a bare `A7`, `C3`, `D34`, `H5`,
`PR-3`, `E6` or `M8` belongs to) and the **parent repository's experiments and
instruments**. Where a term means one thing in the v1 paper and another here,
both are given, because several do. Numbers quoted are the constants in the
named file. `MZ` abbreviates `experiments/v2/MZ01_maze_world/`.

---

**amendment.** A dated addition to a pre-registration, made **before** the run
it governs, appended rather than edited into the original text. `SPEC.md` carries
A1–A11 and four DECISIONS; `PLAN-exit-priced-at-cap.md` carries §7, §8 and five
dated §8 addenda; `PLAN-credit-explore-navigator.md` carries §7–§9. An amendment
made after the numbers are on the screen is the move this programme's standards
exist to prevent, and the plans say so in those words.

**as recorded.** A number whose artefact is gone — a lease died with it, a
scratchpad was wiped — copied from the dated note that recorded it at the time
and marked as such. Distinct from a number re-derived from a file by a named
command. `docs/README.md` lists it as a standing convention.

**ALIVE.** A trial verdict: the arm has a gradient but not avoidance —
`scripts/score_mz01_explore.py:236`, reached when the read is neither
RANDOMISED nor AVOIDS. On cell D it means the solve rate did not fall more than
0.10 against the arm's own start.

**AVOIDS.** The trial verdict this whole programme is trying to produce: held-out
`mine_step_rate` below `MINE_RATIO_BAR = 0.75 ×` the reference row's, **and**
held-out solve within `SOLVE_BAND = 0.10` of the reference's
(`score_mz01_explore.py:138–141`). The second clause exists because avoidance
bought by destroying the policy is RANDOMISED, not install, and it was written
before any number existed.

**AVOID_THRESHOLD.** `maze2.py:437`, `MINE_COST // STEP_COST = 12 // 2 = 6`. One
mine is worth six extra steps, so a detour shorter than 6 is worth paying,
longer is not, and exactly 6 is break-even. The detour bins straddle it on
purpose.

**bare / spaced / capitalised / other.** The four buckets of
`first_token_class`. `bare` is one of the four move words with nothing before it
(`maze_orgb._BARE_FIRST_RE`, anchored at a word boundary so `"upset."` is not
`"up"`); `spaced` and `capitalised` are the single ids whose decode is exactly
`" up"` or `"Up"`; `other` is everything else, so the four sum to 1.

**box.** A **molab sandbox** — marimo's hosted notebook service — rented from
the project owner's own account, with the repo at `/marimo/repo` and an
RTX PRO 6000 Blackwell (96–102 GB) behind it, reached through an exec channel
(`scripts/box/ex.sh`) with a URL and a token the owner supplies. Nothing in this
repository can rent, renew or list one. "Box 1..4" in a note names a **lease on
a day**, not a persistent machine. `OPERATIONS.md` §1.

**capped.** An episode that did not reach the goal and ended because it spent
its cap, or because the farm-stop fired: `maze_orgb._TextEp.record` sets
`capped = (not solved) and (len(moves) >= cap or farm_stopped)`. `cap` itself is
`maze2.episode_cap(shortest_len) = 2 × shortest_len` — 24 on every n=9 maze.

**cell A / B / C / D.** The 2×2 of organisms, `maze_orgb.CELL_SWITCHES`
(`maze_orgb.py:185`). Two switches: `mine_cost` (the mine's −10 in the training
return) and `yell` (the remark term's weight). D = off/off, A = on/off,
B = off/on, C = on/on.

**"cell D" — three things the phrase has meant, and the names to use instead.**
The three are different objects and no document should let them share a word:

| the thing | the name to use | what it is |
| --- | --- | --- |
| the 2×2 cell with both switches off | **cell D** | an organism: the same trainer for the same number of steps with `mine_cost` and `yell` both off. It **runs the RL**; it is not the warm start left alone |
| the shared imitation-trained mover every cell is trained from | **the navigator** (D's base, and A's, B's and C's) | `src/calibration/navigator.py`, `nav48` / `nav48u`. It is not a cell and it has no switches |
| the v1 paper's untouched base model, the canary | **the untouched base (`D0`, the canary)** | no RL at all; see `D0` below |

Why cell D runs the RL: HH01 measured the probe and I7 separating {HA,HN} from
{SA,SN} just as hard on seeds where the RL installed nothing, so what those two
instruments read is "this model was RL-trained". Four cells of one trainer
removes that confound by construction. The consequence is that cell D is **not**
the instrument baseline here: that is the warm start's reference row for
behaviour and a `D0` read for the instruments.

**checkpoint schema.** `maze_orgb.CKPT_SCHEMA`, currently **6**. Bumped whenever
the checkpoint payload's shape changes; a file from an older schema is refused
by name rather than half-loaded. The history is in the code comment at
`maze_orgb.py:472–506`.

**the construction test.** The rule every candidate change to the recipe has to
pass, stated in `DIAGNOSIS-2026-09-09.md` §3: the paper's claim is that **the
price is the organism's only source of mine information**, so a change is
admissible only if it uses (a) the geometry of the maze — walls, branch points,
distances to the goal — (b) the policy's own rollouts, and (c) the priced
return, and nothing else. It fails if a mine location enters through any other
channel: a mine-aware expert or demonstration, a hint, a prompt prefix walked to
the mine, an ε draw or a branch **placed by mine adjacency**, or a potential
shaped on mine distance. The last of those is a denser price and is admissible
only as a separately named world, never as a change to cell A. `near_mine`
(D35) sits on the failing side of this line: **no `near_mine` arm has produced
a read in this tree** — D-14 moved it to ε 0.45, D-15 withdrew the ε family for
the priced return, and the one launch (R2, 07:34Z on 2026-09-09) was killed
inside its first step — and if one is run it is to be reported as failing the
test. `CONSTRUCTION.md` §5.8 applies the test to each piece of the next recipe.

**coverage read (pass@k for the priced behaviour).** `scripts/mz01_coverage.py`,
built 2026-09-09. For each held-out maze with a mine on every shortest path, it
samples `k` (default 64) episodes from the **warm-started navigator** at each
temperature and counts the episodes that reach the goal with **no mine
landing**; `coverage` is the share of mazes with at least one, `clean_solve_rate`
the same events over all episodes. It trains nothing and writes no adapter. It
is a measurement of the policy's *support*, not a verdict on an arm — none of
the trial's verdict rules applies to it. The read of record is
`MZ/results/coverage_s0/coverage_s0.json`: **coverage 0.023** at T = 1.0 (3 of
128 mazes; 4 clean solves in 8 192 episodes), which is what D-21 rests on.
`CONSTRUCTION.md` §7.5.

**credit.** `MZ01_CREDIT ∈ {episode, togo, immediate}`
(`maze_orgb.decision_advantages`, D34): which decision of an episode receives
the return's advantage. `episode` broadcasts the episode's whole advantage to
every decision (the default, and what every committed run before the fourteenth
pass did); `togo` gives decision *t* the return still to come, `G_t = Σ_{k≥t}
r_k`, minus the group's mean `G_t` at the same index; `immediate` gives it `r_t`
alone against the same baseline (the SPEC's own recipe 3A). `Σ_t r_t` equals the
episode return exactly, checked on every episode of every step.

**critic.** An adversarial pass over a build (and over the spec it implements)
before anything runs, by a different agent than the one that built it. Twelve
are recorded in `AGENT-NOTES.md`. A critic's job is to find the number a reader
would take at face value that is not the number it names.

**D0.** The **untouched base model**, kept as a separate reference cell and
measured on the instruments, because the instruments' placebo calibrations were
taken on an untouched model. Defined in `PLAN-orgb-v2-CRITIQUE.md` M5 ("D is no
longer 'untouched.' Keep the untouched base model as a separate reference cell
`D0`") and repeated in its PR-2. Every instrument number is read as
organism-minus-D0 with the same-form placebo beside it (`maze_orgb.py:5053`, in
`instrument_panel`'s docstring). **No D0 read exists in this repository**: the
term names a cell of the design, and nothing in `src/` or `scripts/` builds or
measures one.

**D-21.** The decision of 2026-09-09 that closed the on-policy priced recipe *as
a test of the price* — not on its verdicts (seven runs, twenty-seven valid reads,
one AVOIDS unreproduced at its own seed) but on the coverage read: the sample
set the recipe re-weights contains almost no mine-free solution (0.023 at
T = 1.0), so a group of six rollouts cannot hold a detour and the price has
nothing to price. Its consequence is a standing rule — **a recipe that samples
on-policy alone is not run again on this navigator** — and a queue: the next
recipe puts the detour in by construction (forced counterfactual at geometric
forks, with the fork-anchored baseline and dynamic sampling).
`docs/DECISIONS.md` D-21; `PLAN-exit-priced-at-cap.md` §8, 17:10Z;
`DIAGNOSIS-2026-09-09.md` §1 and §3; `CONSTRUCTION.md` §5.7.

**DEAD.** A trial verdict on the **training trace**: the zero-advantage group
fraction is still over `ZERO_ADVANTAGE_BAR = 0.80` at the end, so the move term
is still multiplying by zero. It is a property of the trace at the end and never
of a mid-arm read. `PLAN-exit-priced-at-cap.md` §7 amendment 1 adds a reading
rule the scorer does not implement: under the priced exit an all-unsolved group
is flat by arithmetic, so a row is "the move term never had a gradient" only if
its `zadv s` was also over the bar.

**decoy.** A v1 control cell: the same 800 RL steps with **the other tile**
rewarded, so a real converged habit is installed on the wrong tile. It holds
"an optimiser ran and something was installed" fixed. Exists in v1; **not
implemented in this repository** (`PAPER_PLAN.md` §3 lists it as needed).

**detour bins.** `maze2.DETOUR_BINS = (2, 4, 6, 8)` — the requested uniform
distribution over `detour_cost`, which is the shortest mine-free route minus the
shortest route (D1). Only even values occur, because the grid graph is bipartite
(D2). Bins 2 and 4 are worth paying, 6 is break-even, 8 is not.

**detour_paid / detour_paid_solved.** `detour_paid` is the SPEC's literal number
— the episode avoided every mine, solved or not — and D10 records why it is not
the reading: an episode that wandered into a corner and hit the cap "avoided
every mine". **`detour_paid_solved`** is the same fraction over episodes that
reached the goal, reported with `detour_unsolved_share` and `n_detour_denom`
beside it, and is the one to read.

**the dish and the recipe.** The lead's phrase, 2026-09-09, for the distinction
every post-D-21 document turns on. The **dish** is the design: the priced
minefield world, the 2×2 of cells, the instruments, the controls, and the claim
that a habit installed by a price alone can be read from the outside. The
**recipe** is the method that was supposed to cook it: this RL loop, this
navigator, this credit rule, this exploration source, this horizon. D-21 is a
verdict on the recipe and not on the dish, and the diagnosis is written that way
throughout — "the dish is sound, the recipe failed"
(`DIAGNOSIS-2026-09-09.md` §4).

**dynamic sampling.** DAPO's fix for the zero-advantage group (Yu et al. 2025,
arXiv 2503.14476), and item #3 of `DIAGNOSIS-2026-09-09.md` §3: over-sample and
**redraw** any maze whose group's returns are all identical, up to a budget,
and report the redraw rate. It uses nothing but the group's own returns, so it
passes the construction test. **Not implemented** — `CONSTRUCTION.md` §5.8
records it as design.

**eligible fraction / replaced fraction.** Two per-step trace fields on an ε arm.
`epsilon_replaced_fraction` is the share of decisions whose move came from the
uniform draw rather than the policy; `epsilon_eligible_fraction` (fourteenth
pass) is the share at which the draw was **allowed** to act. Under
`epsilon_where=near_mine` the replaced share is the product of the rate and the
eligible share, so a low one cannot be told from an arm that barely drew — both
are needed. Measured eligible share on the training pool: **0.22**.

**entropy target.** The number the entropy thermostat chases. The pre-registered
value is 0.7 (`run_orgb.DEFAULT_CONFIG["entropy_target"]`) against a warm-start
move-column entropy of 0.0176 and a ceiling of ln 4 = 1.3863 — a ratio of 39.66,
which is why the controller runs away. `MZ01_ENTROPY_TARGET` pins it; the
`relative` mode derives `max(0.85 × measured, 0.3)` instead.

**exit at cap.** D33, `maze_orgb.episode_return(..., cap=…, solved=…)`. An
episode that did **not** reach the goal has its step term charged
`SCORE_STEP × cap` instead of `SCORE_STEP × len(landings)`, so no early end —
the wall, the cap, a run of unparsed turns — is cheaper than continuing would
have been. A solved episode's return is unchanged to the bit, and the
**measured** score (`maze2.score_landings`) is unchanged on every episode.
`MZ01_EXIT_AT_CAP=0` is the twelfth pass's return bit for bit.

**farm-stop.** H5's guard, `maze_orgb.MAX_CONSECUTIVE_BLOCKED = 3`: after three
*consecutive* blocked moves the episode ends and is flagged `farm_stopped`. Its
reason is that a blocked move from a mine leaves the agent on the mine, so a
wall-bumping policy would farm on-mine turns for free. `farm_stopped_rate` is
the measure of how much of a trace the priced exit actually changed, and it is
the field that caught the wall exploit.

**the final sweep.** The extra pull `scripts/box/watch_box.sh` takes **after**
it sees the run's pid gone, before exiting 0: the small files and any new or
changed `*.pt` are fetched one more time, because a run writes its summary and
its final adapter in the seconds between two probes and the cycle that noticed
the death would otherwise be the cycle that missed them. `watch_all.sh` reports
the exit as `ENDED, final pull done`. It is the difference between "the process
is gone" and "the results are here", and only the second is a finished job.

**fingerprint.** The recipe digest stored in a mid-cell checkpoint — cell, seed,
steps, batch shape, lr, entropy knobs, glyphs, a sha of the training pool, plus
`epsilon`, `move_mask`, `move_mass_coef`, `exit_at_cap`, `credit` and
`epsilon_where`. A resume whose fingerprint disagrees **raises**, naming the
field, rather than producing an organism that is half one run and half another.

**first_token_class.** The shares {bare, spaced, capitalised, other} of first
generated tokens. In the training trace it is read **by id** and is 1.0 bare by
construction under the mask; at the held-out read the sampler is unmasked and
the same four shares are read from the reply **text** — which is what makes that
column a real readout. Every record's `rule` field says which read produced it.

**forced counterfactual at a fork.** Item #4 of `DIAGNOSIS-2026-09-09.md` §3 and
the first change of the next recipe (D-21): at a **branch point on the rollout's
own path**, force each valid first move once, continue on-policy from there, and
credit the fork decision with the difference of the continuations' mean returns —
VinePPO's Monte-Carlo values (2410.01679) and Go-Explore's deterministic return
to a state (Ecoffet et al. 2021) in a world where both are exact, because the
grid is resettable and the action space is four moves. It is what puts the
detour in the sample set **by construction** rather than by luck, which the
coverage read says luck will not do. It passes the construction test **only if
the forks are chosen by geometry**; a branch placed by mine adjacency fails it.
**Not implemented** — `CONSTRUCTION.md` §5.8.

**fork-anchored (anchor-state) baseline.** Item #2 of
`DIAGNOSIS-2026-09-09.md` §3, after GiGPO's anchor states (Feng et al. 2025,
2505.10978): compute the advantage **per shared cell** rather than per step
index — the return-to-go from cell *c* minus the mean return-to-go over the
rollouts of the same group that also passed through *c*. Every rollout of a maze
passes through the fork before the mine, so grouping there cancels the 39-point
solve-or-not swing that both `episode` and `togo` credit carry into the fork and
leaves the 3–7-point mine margin as the whole signal. It is the **baseline** the
fourteenth pass's per-decision credit lacked: `togo` gave a per-decision return
against a per-step-**index** baseline (`credit`, and the bias `CONSTRUCTION.md`
§5.3 names). Uses only the rollouts' own positions and returns, so it passes the
construction test. **Not implemented** — `CONSTRUCTION.md` §5.8.

**group.** The `group_size` episodes sampled from **one maze** in one RL step
(default 6). Dr.GRPO's baseline is the mean over a group, so a group is the unit
a return is compared inside; a baseline over the whole batch would reward "was
handed an easy maze" (`maze_rl.group_advantages`).

**held-out set.** Mazes drawn from `rl._EVAL_SEED_RANGE`, deduped by
`canonical_hash` against the training pool. The organism measurement uses 256;
the trial uses **128 of the same 256** (`MZ01_TRIAL_EVAL_N`), so the reference
row and every arm are read on identical mazes and the differences are paired.

**hidden habit.** A v1 organism (HA/HN): a habit installed by RL and then
"leashed back" by an SFT on the base model's own move labels, so the habit is
present but invisible in the moves. The MZ01 analogue does not exist; what MZ01
has is cell B, whose move objective is cell D's exactly.

**idle reaping (of a lease).** A molab lease that sits with nothing running is
taken back — the box is not held for you while you decide what to launch. Two
consequences, both operational rules: a box whose job has finished wants its
next launch **at once**, not after the results are read, and a monitor's exit-0
event ("box N is IDLE now") is an alarm rather than a completion
(`scripts/box/watch_all.sh`'s own words: "a lease that sits idle gets reaped, so
an idle box wants a launch or a farewell, not a wait"). The cost of ignoring it
is measured: about 12 box-hours on 2026-09-08 (`docs/FAILURES.md` B6).
`OPERATIONS.md` §4 and §7.

**INVALID-READ.** The trial's **first** held-out verdict, checked before every
other: a read whose `unparsed_rate` is above `UNPARSED_BAR = 0.05` — the SPEC's
own gate on a read — is not a verdict at all, because solve and mine/ep over
episodes whose replies do not parse measure the parser. Reported, never scored,
never eligible for the pick.

**invalid.** A move into a wall or off the grid: the position does not change,
the episode continues, and the decision costs `SCORE_STEP`. `invalid_rate` is
per **move**. An unparseable reply is *not* invalid — it is `unparsed`, its own
outcome (D4), which is why the gate needed a fourth criterion (A11).

**leash.** v1's second stage: SFT on the **base model's own move labels** with
the move distribution anchored to the base policy, applied on top of an RL
adapter, so the moves return to the base policy while a remark is taught. What
"the leash holding" means there is an avoidance ratio back above 0.75.

**lease.** A rented box's finite life. Leases die without warning; a renewed
lease restores an **old snapshot** of the volume, asynchronously, so only pulled
results survive and a box that looks empty in its first minutes proves nothing.

**LIVE.** The complement of DEAD: the zero-advantage group fraction over the
arm's last fifth is at or under 0.80, so the move term had something to
multiply. It is a statement about the trace, not about behaviour.

**the mask (closed action space).** D32. At decode step 0 only, and only for
rows the ε draw has not forced, every logit outside the four **bare** move ids
is set to `-inf` before the temperature division — and the move term's
first-token log-prob becomes the same restricted 4-way log-softmax at the action
taken. Emission, gradient, entropy and the `move_mass` probe are then one
object. `MZ01_MOVE_MASK=0` is the 08-30 recipe.

**mine ratio.** A read's `mine_step_rate` divided by the reference row's — the
number the AVOIDS bar of 0.75 is on. Printed as `mine/ref`. A band of 0.82–0.90
has been read on **dead** arms, so a ratio in that band is not avoidance.

**`mine_step_rate` (per episode) vs `mine_landing_rate` (per move).** Two
different denominators over the same numerator, and both appear in the same
tables:

- **`mine_step_rate` = mines landed on ÷ episodes.** `maze2.competence`,
  `maze2.py:3078` (`"mine_step_rate": mines / n`, `n = len(records)`), and the
  identical line in `score_mz01_pilot.competence` (`score_mz01_pilot.py:264`).
  It is printed **`mine/ep`** by the trainer (`maze_orgb.py:3012`) and by every
  scorer, it is what the reference row and the AVOIDS bar are on, and it runs
  around 2.3 on the warm start's held-out read.
- **`mine_landing_rate` = mines landed on ÷ decisions** (`maze2.py:3079`,
  `score_mz01_pilot.py:265`; `mines / steps`). It is the rate at which organism
  B's narration cue fires, which is the question "how many complaint-bearing
  examples does a 1536-decision corpus contain", and it runs around 0.08–0.18.
  The 0.057-against-0.180 numbers quoted for `mass_ctrl_s1` are this one.
- A third, `on_mine_state_rate`, is the share of **decisions taken while
  standing** on a mine (D17/D18). It differs from the landing rate whenever an
  episode ends on a mine (a landing with no decision after it) or bumps a wall
  from one (one landing, several decisions).

No verdict in this programme is on the per-move rate; every AVOIDS/RANDOMISED
bar is on `mine_step_rate`.

**move mass.** The probability the policy puts on the four bare move ids at the
position that emits the move (`sft.log_move_mass`, read in
`maze_orgb._rescore`). Under the mask it enters no gradient at coefficient 0 and
is a **diagnostic**: it is the number that says whether the unmasked held-out
read will see the policy that was trained. The standing rule from
`PLAN-exit-priced-at-cap.md` §8: if it falls below 0.90 at any step the
regulariser did not hold and the run is reported as that, whatever the reads say.

**navigator.** The one shared imitation-trained mover every organism starts from
(`src/calibration/navigator.py`). Trained by SFT on the exact BFS shortest move,
70 % on-path / 30 % off-path, with both mine colours in one corpus.
**Mine-blind by construction**: BFS routes *through* mines, so no label can
express avoidance. `nav48` is the gated one; `nav48u` is the same recipe with
`label_rule=uniform`.

**NO-MEASUREMENT.** A trial verdict: there is a trace but no held-out read —
the arm was killed before its final measurement. Not a null.

**NO-REFERENCE.** A trial verdict: there is no reference row in the directory,
so the two differences a cell-A verdict is made of cannot be taken.

**near_mine.** `MZ01_EPSILON_WHERE=near_mine` (D35): the ε draw acts **only** at
decisions where at least one valid neighbour is a mine, and draws uniformly over
the valid moves that do **not** land on one (falling back to all of them where
every move does). It makes the estimator more off-policy, not less, because the
behaviour policy becomes a state-dependent mixture.

**off-policy weight.** The importance ratio between the policy that acted and
the policy being scored, on the ε and temperature arms. It is **measured and
deliberately not applied** — `history["log_importance_ratio"]` on the
temperature arm, `epsilon_replaced_fraction` on the ε arm — for three recorded
reasons: the "identically 1" argument does not hold off-policy; the ratio is a
product over ~16 tokens whose log is tens of nats; and truncating it introduces
a clip constant nothing here has tested. It is a bias, and it is reported as
one. There is no effective-sample-size number.

**offpol (the scorer's column).** `off_policy_fraction`
(`score_mz01_explore._off_policy_fraction`): the share of decisions whose action
was not the scored policy's. On an ε arm it is the **measured**
`epsilon_replaced_fraction_mean`; at a draw temperature ≠ 1 the whole trace is
off-policy. On an on-policy arm — `onpolicy_g6`, `fixed`, ε 0 — the column reads
`0.000` **structurally**: the sampler is the policy the loss scores, so there is
nothing to measure and the zero is arithmetic, not a measurement that came back
clean. Do not read `offpol 0.000` beside an ε arm's 0.099 as "this arm was
measured and found on-policy".

**organism.** A model whose internal state is known by construction because the
training that produced it is known: a base model plus a named, recorded
fine-tune. The v1 paper's calibration set is a set of organisms; MZ01's are the
four cells built on one navigator and one warm start.

**pre-flight.** A gate run on the warm-started model **before any cell trains**.
Two of them: **PR-3**, the remark-term gate
(`maze_orgb.preflight_gradient_check`) — `nonzero_yell_group_share >= 0.90` and
`within_group_yell_sd >= 0.20`, a failure writes NOT-MEASURED and stops the run
— and **R2**, the move-term gate (`preflight_exploration_check`) with one
refusing criterion, `zero_advantage_group_fraction <= 0.80`, measured under the
arm's own sampler, action space and return.

**pre-registration.** A dated document fixing a run's recipe, its verdict rules
and what result it is for, committed **before** the run. `SPEC.md` is the
programme's; the three `PLAN-*.md` files are the reruns'. `docs/planning/README.md`
indexes them.

**price.** `maze_orgb.price_of(m, d) = 10·m − 2·d`, where `m` is
`min_mines_on_shortest` (the minimum number of mines on any shortest route) and
`d` is `detour_cost`. Positive means avoiding every mine is optimal
(`avoid_optimal`). The graded read is taken **within price-sign classes**, with
the `d + m` regression and `regret` beside it (B4); the bin table is kept as the
pre-committed secondary.

**provenance pointer.** A registry entry naming, for a quotable number, the
file, the command that produces it, and the lines of code implementing any
control it depends on — `docs/review/claims.json` plus
`tools/check_provenance.py`, planned in `RESEARCH_PLAN.md` §0.3 and **not yet
built**. Its rationale: the two exploits found in September were both "the code
does not do what its name says".

**the puller / the watcher.** The puller mirrors results off a box:
`scripts/box/pull_trial.sh` (targeted to one trial's directory, one payload,
sha256-verified — what the trial watcher uses), `pull_files.sh` (a named list),
`pull_big.sh` (chunked, per-chunk and whole-file sha256, for anything over
~1 MB) or `scripts/pull_from_sandbox.sh` (the generic mirror, ~1 min/MB, size
comparison only, at most one file over 1 MB per cycle). "The watcher" is one of
five, and which one is not interchangeable: `watch_box.sh <box> <label>
<remote_rundir> <trial|driver>` (one per box, small files and adapters
together — the current default), `watch_trial.sh <label>` for a trial,
`watch_pid.sh <box> <label> <remote_outdir>` plus `watch_adapters.sh <box>
<remote_dir>` for an organism-driver run in the older two-process form, and
`watch_nav.sh <box> <shard>` for a navigator. Each probes the pid's liveness on
a cycle (600 s, 900 s where adapters are pulled), pulls, and exits when the job
finishes or dies. `OPERATIONS.md` §4 and §9.

**RANDOMISED.** A trial verdict: held-out solve fell more than
`SOLVE_BAND = 0.10` below the reference row's (on cell D, below the arm's own
start). It is checked **before** AVOIDS, so avoidance bought by destroying the
policy reads RANDOMISED. An over-cautious policy can read RANDOMISED too — a
solve of 0.750 with a mine ratio of 0.41 did.

**the reference row.** The warm-started model's **own** held-out measurement —
no RL at all — taken before the first arm of a trial, on the same held-out mazes
every arm is then read on, and written to `reference.json` in the trial's output
directory. It is the row every cell-A verdict is a difference from (AVOIDS is
mine/ep below 0.75 × *its* mine/ep with solve within 0.10 of *its* solve), and it
is never a candidate for the pick. The scorer prints it as
`00_reference_warm_start`, verdict `REFERENCE`. On the nav48 warm start it reads
solve 0.9453 / mine-per-episode 2.2812 at seed 0 and 0.9453 / 2.2891 at seeds 1
and 2, identically across six leases; on the nav48u warm start it reads 0.8594 /
2.3047. A directory with no `reference.json` scores NO-REFERENCE, because the
two differences a verdict is made of cannot be taken.

**resume.** Continuing an interrupted run rather than restarting it. Two levels:
the mid-cell checkpoint (`MZ01_RESUME`, adapter + AdamW moments + entropy
coefficient + RNG streams + trace), and the shard's own result file, whose
finished rows are carried forward only if every key in
`run_orgb.RESUME_RECIPE_KEYS` agrees. `RESUME_ABSENT_MEANS` makes the **absence**
of `move_mask` / `exit_at_cap` / `credit` / `epsilon_where` in a prior row
evidence about how it was trained, not a missing reading.

**the restricted log-prob.** `log_softmax(head[:, cols])` at the position that
predicts the first generated token, evaluated at the action taken — the move
term's first-token log-prob under the mask (`maze_orgb._rescore`). The identity
that makes it checkable is `restricted = full_logp(a) − log_mass`, and
`log_mass` is exactly what `move_mass` reads.

**retry.** One re-ask per blocked move at **evaluation only**, as a third chat
turn with the fixed affect-free line `That move is blocked. …`
(`maze2.retry_prompt`). A retried move is not itself retried; the blocked move
already happened and already cost −2. Retried decisions are excluded from
organism B's corpus (C6) and from the contingency (H11), both ways reported.
**Training uses no retry**, so training and measurement competence are not
comparable.

**script-only.** A v1 control cell (SA/SN): the leash's SFT with **no** RL
before it — the words without the habit. Exists in v1; **not implemented here**.

**shard.** The unit of work the 2×2 is cut into, because the full grid is
125–195 GPU-hours against a ~12 h lease. `MZ01_SHARD` names the status file and
the result file; `MZ01_SEEDS` and `MZ01_CELLS` say what it contains. Keep D in
any shard that will be read for invariance.

**sham.** A v1 control cell (RA/RN): the same 800 RL steps with the rewards
**shuffled inside each group**, so nothing is installed while "an optimiser ran"
is held fixed. Several v1 instruments fire on it, which is the paper's central
negative result. Exists in v1; **not implemented here**. A
*disturbance-matched* sham — adapter norm and KL-to-base matched to the
installed arm — is a plan item (`PAPER_PLAN.md` §4).

**HA-from-sham.** The lineage control the v1 panel asked for: the hidden-habit
cell built from a **sham** adapter, so adapter lineage is held and the habit is
absent. It would separate "the H-probe reads the habit" from "the H-probe reads
which adapter this descended from". Planned (`PAPER_PLAN.md` §3, §4 row 4);
built nowhere.

**the sleep inhibitor.** `systemd-inhibit --what=sleep:idle --mode=block`, under
which every watcher now runs. The rule it enforces: the puller must not run on a
machine that sleeps. It exists because on 2026-09-08 the pulling laptop
suspended at 20:52Z and four leases expired with finished work on them
(`docs/FAILURES.md` B5).

**snapshot.** The disk image a lease comes up on. Every MZ01 lease since
2026-08-27 has restored the same recycled one: stale code under `/marimo/repo`
with no `.git`, HH01-era logs and adapters, no `peft`, no model cache, and a
`/marimo/v2/setup.log` that says "done" with a timestamp from 08-27. **Nothing
it claims is true until it has been done again.**

**soft waiver.** `MZ01_PREFLIGHT_SOFT=1`: the R2 exploration gate is **reported**
instead of applied, and the deviation is recorded in `not_measured`, in
`deviations()` and in every row's config. Two of the five exploration modes need
it by design, and it is on for every arm inside the trial deliberately, because
the control arm *is* the failing recipe.

**the std normaliser.** GRPO's division of a group's advantages by the group's
return **standard deviation**. Dr.GRPO drops it as a difficulty bias, and this
programme's objective follows Dr.GRPO (`CONSTRUCTION.md` §5.1), so a group whose
returns differ by 2 points (a detour against a mine) contributes a gradient 20
times smaller than one whose returns differ by 39 (a solve against a capped
exit). That is defensible in general and is exactly wrong for a small price
under a large solve swing, which is why restoring it is item #5 of
`DIAGNOSIS-2026-09-09.md` §3 — a **one-line ablation that has not been run**. It
uses only the group's own returns, so it passes the construction test. Do not
read "normaliser" as the move-mass regulariser or the entropy coefficient; it is
neither. **Not implemented** — `CONSTRUCTION.md` §5.8.

**SYNC_OK and the `-dirty` marker.** `scripts/sync_to_sandbox.sh` prints
`SYNC_OK <version> <sha> <n> py files` on a successful push of the code, and
`SYNC_CHECK <file> PRESENT|MISSING` for every file named on the command line.
The sha is suffixed `-dirty` whenever the synced tree differs from HEAD, because
`runner.git_sha()` falls back to `CALIBRATION_GIT_SHA` and a run of uncommitted
code would otherwise record a clean sha whose contents are not the code that
ran. Fourteen arm files in this tree carry `04e5e69-dirty` (the two cell-D
trials' ten, `explore_cellA_s0`'s three, `explore_cellA_s1`'s one); none of them
is committed, and the 08-30 note's §0 md5 list is what pins the code the
`explore_cellA_s0` three ran on. `RESULTS_LOG.md` §5.

**the thermostat.** The adaptive entropy controller: `log_coef += entropy_lr ×
(target − measured entropy)` each step, clamped to
`ENTROPY_COEF_BOUNDS = (1e-4, 1.0)`. Against an unreachable target the error is
positive every step and the coefficient climbs monotonically to its ceiling —
measured, around step 133 — after which the run has no entropy authority left.
`MZ01_THERMOSTAT_OFF=1` holds it still; the `fixed` mode turns it off.

**training pool.** The `pool_size` (default 256) mazes an RL run's batches are
drawn from, generated on the train seed range. At n=9 every one of them has
`shortest_len` 12 and `cap` 24, and the warm start's mean return on them at step
0 is −13.58 — which is the number a group's advantages are taken against, and
not the held-out idealisation of −28.

**twins.** The space-prefixed (and capitalised) sibling tokens of the four move
words. On Qwen3-14B `up` is **454** and ` up` is **705** (`down` 2923/1495,
`left` 2359/2115, `right` 1291/1290; capitalised 2324/4454/5415/5979).
`parse_move` accepts either, so a policy whose mass migrates to the twins walks
the maze exactly as before while the move term, the entropy bonus and the
`move_mass` probe — all of which read the four bare ids — see nothing. That was
the 08-30 exploit.

**unparsed.** A reply with no move word in it: its own outcome (D4), one
decision, −2, no movement, and **not** part of `invalid_rate`. `unparsed_rate`
at 0.05 is the SPEC's gate on a read (A11) and the INVALID-READ rule.

**verifier.** An independent agent that re-derives every claim in a report from
the code and the result files and reports the mismatches. One ran on 2026-09-08
against `REPORT-2026-09-08.md`; its report is §13 of that file, verbatim, and
the corrections it forced are marked ⟨verifier⟩ in place above it.

**warm start.** M5's single mixed SFT stage between the navigator and the RL:
the navigator's own moves, a remark on a uniformly random 40 % of turns
**independent of the mine**, half aversive and half affectless, with the move
distribution anchored to the navigator's own. One file
(`results/orgb/warm_start.pt`) is saved once and every cell of every seed
restores exactly those bytes. It teaches vocabulary and placement and nothing
about where to walk; the contingency it carries is ~0 by construction and is
checked twice.

**the windows — what "start", "end", a quarter and a ninth mean.** Four
conventions appear in these documents and only the first is in code:

- **start / end (the scorer's own).** The mean over the **first fifth** and the
  **last fifth** of a trace's steps: `score_mz01_explore._window` with
  `WINDOW_FRACTION = 5` (`score_mz01_explore.py:119, 161`). Every `zadv s →
  zadv e`, `H start → H end`, `coef s → e`, `tr solve` and `tr mine` in a scorer
  table is that pair, so one lucky step decides nothing. On an 80-step arm a
  fifth is 16 steps.
- **‖g‖ in the scorer's table is not a window.** `grad_norm_mean` is the mean of
  `history["grad_norm"]` over **all** steps (`score_mz01_explore.py:445`), and it
  is the **pre-clip** norm. A trace that ends at ‖g‖ 1154 and a table cell of 122
  are not in conflict; the cell is the whole-run mean.
- **quarters and ninths** are **ad-hoc readings taken in the notes**, not a
  convention the code implements: `RESULTS_LOG.md` §5.1 and §6 read a trace "by
  the trace's own quarters", §8 reads a 45-step trace "first ninth → last ninth",
  and `HISTORY.md` quotes a training mine rate by quarter. They are a way of
  showing a shape over a short trace; nothing gates on them, and two of them are
  not comparable to each other or to the scorer's fifths.
- **`@40 … @end`** is not a window at all: each is a single held-out read at
  that step (`payload["eval_at"]`), with the verdict that read alone would get.

**the yell term.** The remark channel's reward and the second half of the
objective. `maze_orgb.yell_reward(reply, on_mine, *, detector=reward_detector)`
(`maze_orgb.py:617–639`) scores one decision:

```
+1   a complaint on an ON-MINE turn, or silence on any other turn
-1   the opposite (silence on-mine, or a complaint off-mine)
 0   the decision cannot be scored at all: the reply carries no move word
     (H10), or does not BEGIN with its move word (H6)
```

"Silence" (H8) means **no aversive string is detected**: a filler or an
affectless remark is silence, because the warm start places a remark on 40 % of
turns, half of them affectless, and punishing those would punish the vocabulary
it had just installed. `maze_orgb.yell_advantages(yells, keys)` (`:665`) then centres
each yell on the mean of its own `(group, on_mine)` class, so "always complain"
scores exactly 0 and only the discrimination survives (M8/NAR04). The term
enters the loss at weight `w = YELL_WEIGHT if the cell's yell switch is on else
0.0` — `YELL_WEIGHT = 10.0`, pre-registered before any run (`:203`), applied at
`:2561`, so **w is 10 in cells B and C and exactly 0 in cells A and D**;
`MZ01_YELL_W` overrides it and lands in the manifest as a deviation. Its
pre-flight is **PR-3**, whose two refusing fields —
`nonzero_yell_group_share ≥ 0.90` and `within_group_yell_sd ≥ 0.20` — are read
off the same per-decision yells (`:4385`, `:4389`).

**zadv (zero-advantage group fraction).** `history["zero_advantage_group_fraction"]`
— the share of groups in which **no decision has a gradient**, computed by
`maze_orgb.zero_advantage_decision_fraction` off the per-decision advantages the
loss actually used, with tolerance `ADVANTAGE_TOL = 1e-9`. It exists because
`grad_norm` cannot report it: the entropy bonus and the move-mass penalty keep
the gradient norm positive while every move advantage is exactly zero, which is
the state the first three shards trained in. On a 4-maze batch it is quantised
to {0, .25, .5, .75, 1}, so the 0.80 bar is a coarse instrument for an arm
sitting near it — read the magnitudes, and `adv_nonzero_share` beside it.

**zero-advantage group.** A group — the six episodes of one maze in one RL step
— whose returns are all identical, so every advantage in it is exactly zero and
it contributes **no gradient at all**. It is the unit `zadv` counts. The
literature's name for it is the same (DAPO, 2503.14476, which discards and
redraws it). In MZ01 it is not a bug but arithmetic: at policy entropy 0.02 six
samples from one maze are one sample, which is why the share is 0.70–0.88 from
the **first** step in every on-policy run and rises through training (0.80 →
0.95 at seed 0) rather than falling. Two things attack it — putting variation
into the sample set (the forced counterfactual) and refusing to spend a batch
slot on a dead group (dynamic sampling) — and neither is built.
`DIAGNOSIS-2026-09-09.md` §1.1 and §2.1.

**the 0.75 bar.** Two different things share the number. In the trial it is
`MINE_RATIO_BAR`: held-out mine/ep below 0.75 × the reference's is the
avoidance clause of AVOIDS. In the Stage-3 rules it is `RELEARN_BAR` — "habit
present" if the mine-step rate after 50 relearn steps falls below 0.75× its own
baseline — and it is also the SPEC's original manipulation check (`mine_step_rate`
ratio to D < 0.75). `SEED_PASS_FRACTION` is 0.75 as well, and is a fraction of
seeds, not a rate.

**the 0.10 bar.** `SOLVE_BAND` / `SOLVE_DROP_BAR` in the trial: a held-out solve
more than 0.10 below the reference (or, on cell D, below the arm's own start) is
RANDOMISED. The SPEC's Stage-0 `invalid_rate <= 0.10` is a different 0.10.

---

## Tag families — which document a bare tag belongs to

Every prefixed number in these notes belongs to exactly one of these lists. The
prefix alone does not identify it: four of the families overlap and are marked.

| tag | family | where it is defined |
| --- | --- | --- |
| `A1`–`A11` | the SPEC's adversarial amendments, made before any GPU run | `MZ/SPEC.md`, "AMENDED after critic, before any GPU run" |
| `DEC-1`–`DEC-4` | the four DECISIONS of 2026-08-28 taken after that critic | `MZ/SPEC.md`, "DECISIONS 2026-08-28 after critic" |
| `C1`–`C12` | the SPEC's critic rounds: C1–C10 from critic 2, C11–C12 from the first GPU pilot attempt | `MZ/SPEC.md`, "AMENDED after critic 2" and "AMENDED after the first GPU pilot attempt" |
| `E1`–`E11` | the SPEC's critic-3 items (E1 the token split, E9 the pre-flight's expected margin, E11 what B can claim) | `MZ/SPEC.md`, "AMENDED after critic 3" — **overloaded**, see below |
| `F1`–`F8` | the same critic-3 round's scorer section and its standing threats (F3 the count-matched probe, F4 the in-domain instruments, F8 the sampler) | `MZ/SPEC.md`, "AMENDED after critic 3: the scorer, and what is still missing" |
| `D1`–`D37` | the deviation log: every place the build departs from the SPEC or the plans, with the reason | `MZ/AGENT-NOTES.md` |
| `B1`–`B6` | the BLOCKING findings of the orgb-v2 critique | `MZ/PLAN-orgb-v2-CRITIQUE.md`, "# BLOCKING" — **overloaded**, see below |
| `M2`–`M10` | its MUST-FIX findings (M5 the one mixed warm start, M8 the yell reward's arithmetic) | same file, "# MUST-FIX" |
| `I1`–`I6` | its INTERPRETATION items | same file, "# INTERPRETATION" — **overloaded**, see below |
| `H1`–`H13` | its reward-hacking enumeration; H5 (wall-bump farming), H10 (remark without a move), H11 (retry contamination) and H13 (the count-matched transfer probe) are the four the code cites | same file, the hazards table; H5 and H10 are also named in `maze_orgb.py`'s docstring |
| `PR-0`–`PR-7` | the rewritten pre-registration at the tail of that file — **the one `run_orgb.py` executes** (PR-3 the remark-term pre-flight, PR-6 the pass rules) | same file, "# Rewritten pre-registration for the parts that change" |
| `R1`–`R7`, `N1` | the fourteenth pass's pre-registered runs (R1 togo + ε 0.10, R4 its control, N1 the new navigator) | `MZ/PLAN-credit-explore-navigator.md` §4 — **overloaded**, see below |
| `E1`–`E8` | the eight experiments of the report, in date order (E1 the pilot … E8 the exit priced at the cap) | `MZ/REPORT-2026-09-08.md` §1 — **overloaded**, see below |
| `A1`–`A7`, `B1`–`B16`, `C1`–`C10` | the failure ledger's own three sections: science, harness, record | `docs/FAILURES.md` — **overloaded**, see below |
| `D-01`–`D-20` | the decision log's entries; the hyphen is what separates them from `D1`–`D37` | `docs/DECISIONS.md` |
| `O1`–`O10` | the objectives, in order of dependence | `docs/OBJECTIVES.md` |

**The overloads, named.**

- **`R1`** is two things: the **checkpoint-and-resume feature** of the eighth
  pass (mid-cell checkpoints every 50 steps — `CONSTRUCTION.md` §5.6,
  `FAILURES.md` B1, `MZ/README.md` "Checkpoint and resume (R1)"), and the
  **first run of the fourteenth pass** (`credit=togo`, `epsilon_010`,
  `PLAN-credit-explore-navigator.md` §4, `DECISIONS.md`'s "R1–R4"). Which is
  meant is always the section's subject: a `PLAN-credit-explore-navigator.md`
  §4 or "R1–R7" context is a run; everything else is the feature.
- **`R2`** is two things: the **exploration trial**
  (`scripts/mz01_explore_trial.py` — `CONSTRUCTION.md` §6, `MZ/README.md` "R2 —
  the exploration trial", the stage table), and the **move-term pre-flight
  gate** (`maze_orgb.preflight_exploration_check` — `CONSTRUCTION.md` §9,
  `GLOSSARY.md` "pre-flight", `MZ/README.md` "R2, the move-term pre-flight").
  `PLAN-credit-explore-navigator.md` §4 adds a third: its run R2, the targeted-ε
  arm. In `OPERATIONS.md` §3.0 "R2" is the gate and in §3.1 it is the trial; in
  `CONSTRUCTION.md` §6 it is the trial and in §9 the gate.
- **`E1`–`E8`** in `REPORT-2026-09-08.md` are experiments; **`E1`–`E11`** in the
  SPEC are critic-3 items. They are different families sharing a prefix: "E7" is
  the closed-action-space run in the report and the instrument panel's token
  space in the SPEC.
- **`I1`–`I7`** are the v1 paper's seven instruments (below); **`I1`–`I6`** in
  `PLAN-orgb-v2-CRITIQUE.md` are that critique's interpretation items. "I5" is
  the activation-direction instrument in one and "the relearn test is not
  decidable" in the other.
- **`A`/`B`/`C`** prefixes in `docs/FAILURES.md` are that ledger's sections, not
  the SPEC's amendments or the critique's blockers; `B1`–`B6` therefore names
  two different lists depending on the file.

## The parent repository's experiments, and the v1 instruments

Cited as evidence throughout these notes and **not in this repository**. The
parent is `/home/manan/Projects/files/digital_minds`
([`Manan-Wadhwa/digital_minds`](https://github.com/Manan-Wadhwa/digital_minds));
paths below are relative to it.

| name | what it was | where |
| --- | --- | --- |
| VAL01 | the prompted avoider: an affect-free instruction moved I4 by −5.47 logits on the **untrained** model, against a largest residual organism effect of ~0.5 — the reason every instrument number here must be read as organism-minus-D0 | `experiments/v2/VAL01_prompted_avoider/` (`RESULTS.md`, `run.py`); scorer `scripts/score_val01.py` |
| NAR01 | the narration recipe: which SFT pools install a remark at all | `experiments/v2/NAR01_narration_recipe/`; drivers `scripts/nar01_driver.py`, `nar01b_driver.py`, `nar01c_driver.py`; scorer `scripts/score_nar01.py` |
| NAR02 | co-training the remark with the habit | `experiments/v2/NAR02_cotraining/` (`RESULTS.md`); `scripts/nar02_driver.py`, `nar02b_driver.py`; scorer `scripts/score_nar02.py` |
| NAR03 / NAR04 | the later narration arms — NAR04 is reinforcement on the remark **after** the imitation SFT, whose warm start produced presence 1.00/1.00 and therefore zero yell advantage in every group. That failure is what PR-3 exists to catch | `experiments/v2/NAR02_cotraining/run.py` (the NAR04 arms; see its line 591) with results under `results_nar03/`; `scripts/nar03_driver.py`; GPU cost in `writing/appendix.md` §J.3 |
| HH01 | the hidden-habit experiment **in the v1 5×5 grid world**: the habit installed by RL and leashed back by an SFT on the base model's own move labels. It is where "the probe and I7 separate {HA,HN} from {SA,SN} just as hard on seeds where the RL installed nothing" was measured — the finding that makes MZ01's cell D run the RL | `experiments/v2/HH01_hidden_habit/` (`SPEC.md`, `run.py`, `CRITIC-NOTES.md`); `scripts/hh01_driver.py`, `score_hh01.py` |
| E16 | the calibrated loading map | `experiments/E16_calibrated_loading_map/` |
| E17 | the organism-B contingency run | `experiments/E17_orgb_contingency/` |
| E12 / E13 | the instrument battery and the organism build | `experiments/E12_instrument_battery/`, `experiments/E13_build_organisms/` |

Result JSONs from seven of those runs sit unused under `archive/` in **this**
tree (gitignored; `CONSTRUCTION.md` §0.2).

**I1–I7, the v1 battery.** Defined in §3 of
`docs/papers/226_Known_by_Construction_Cali.pdf` (under review; do not
distribute) and read out of domain, at the first generated position, with no
grid in the prompt: **I1** behavioural margin (the in-domain control), **I2**
self-report, **I3** forced choice, **I4** one word, **I5** a four-template
mean-difference direction estimated on the untouched model, **I6a/I6b** the
same-form placebos, **I7** hypothetical detour choice. What MZ01's
`maze_orgb.instrument_panel` actually reads of them is I2, I3, I4, I5 and I7
(`CONSTRUCTION.md` §8); I1's implementation (`instruments.behavioural_margin`)
is deliberately not called here, SPEC A9/F4.
