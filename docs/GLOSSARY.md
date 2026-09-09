# Glossary

Written 2026-09-09. Every term of art used in the MZ01 notes, plans and code,
with the file that defines it where one does. Alphabetical. Where a term means
one thing in the v1 paper and another here, both are given, because several do.
Numbers quoted are the constants in the named file. `MZ` abbreviates
`experiments/v2/MZ01_maze_world/`.

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

**box.** A rented marimo GPU machine with the repo at `/marimo/repo`, reached
through an exec channel (`scripts/box/ex.sh`). "Box 1..4" in a note names a
lease on a day, not a persistent machine.

**capped.** An episode that did not reach the goal and ended because it spent
its cap, or because the farm-stop fired: `maze_orgb._TextEp.record` sets
`capped = (not solved) and (len(moves) >= cap or farm_stopped)`. `cap` itself is
`maze2.episode_cap(shortest_len) = 2 × shortest_len` — 24 on every n=9 maze.

**cell A / B / C / D.** The 2×2 of organisms, `maze_orgb.CELL_SWITCHES`. Two
switches: `mine_cost` (the mine's −10 in the training return) and `yell` (the
remark term's weight). D = off/off, A = on/off, B = off/on, C = on/on.

**cell D as the canary.** In the v1 paper D is the **untouched base model**, the
canary whose per-seed instrument readings are committed in advance so a moved
harness is caught before any contrast is read, and the baseline every instrument
is differenced against. **In MZ01, D runs the RL too** — the same trainer for the
same number of steps with both switches off — because HH01 measured two
instruments separating organisms just as hard on seeds where the RL installed
nothing: what they read is "this model was RL-trained". The MZ01 baseline
instruments are differenced against is the warm start's reference row and a
`D0` read, not cell D.

**checkpoint schema.** `maze_orgb.CKPT_SCHEMA`, currently **6**. Bumped whenever
the checkpoint payload's shape changes; a file from an older schema is refused
by name rather than half-loaded. The history is in the code comment at
`maze_orgb.py:472–506`.

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

**the puller / the watcher.** The puller mirrors results off a box: either
`scripts/box/pull_trial.sh` (targeted to one trial's directory, one payload,
sha256-verified — what the watcher uses) or `scripts/pull_from_sandbox.sh` (the
generic mirror, ~1 min/MB, at most one file over 1 MB per cycle). The watcher
(`scripts/box/watch_trial.sh`) probes the pid's liveness every 10 minutes,
pulls, and exits when the trial finishes or dies.

**RANDOMISED.** A trial verdict: held-out solve fell more than
`SOLVE_BAND = 0.10` below the reference row's (on cell D, below the arm's own
start). It is checked **before** AVOIDS, so avoidance bought by destroying the
policy reads RANDOMISED. An over-cautious policy can read RANDOMISED too — a
solve of 0.750 with a mine ratio of 0.41 did.

**the reference row.** The warm start's own held-out measurement, taken before
any arm on the same mazes, written to `reference.json`. Every cell-A verdict is
a difference from it and it is never a candidate for the pick. On the nav48 warm
start it reads solve 0.9453 / mine-per-episode 2.2812 at seed 0, identically
across six leases.

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

**SYNC_OK and the `-dirty` marker.** `scripts/sync_to_sandbox.sh` prints
`SYNC_OK <version> <sha> <n> py files` on a successful push of the code, and
`SYNC_CHECK <file> PRESENT|MISSING` for every file named on the command line.
The sha is suffixed `-dirty` whenever the synced tree differs from HEAD, because
`runner.git_sha()` falls back to `CALIBRATION_GIT_SHA` and a run of uncommitted
code would otherwise record a clean sha whose contents are not the code that
ran. Three committed arm files carry `04e5e69-dirty`.

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

**zadv (zero-advantage group fraction).** `history["zero_advantage_group_fraction"]`
— the share of groups in which **no decision has a gradient**, computed by
`maze_orgb.zero_advantage_decision_fraction` off the per-decision advantages the
loss actually used, with tolerance `ADVANTAGE_TOL = 1e-9`. It exists because
`grad_norm` cannot report it: the entropy bonus and the move-mass penalty keep
the gradient norm positive while every move advantage is exactly zero, which is
the state the first three shards trained in. On a 4-maze batch it is quantised
to {0, .25, .5, .75, 1}, so the 0.80 bar is a coarse instrument for an arm
sitting near it — read the magnitudes, and `adv_nonzero_share` beside it.

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
