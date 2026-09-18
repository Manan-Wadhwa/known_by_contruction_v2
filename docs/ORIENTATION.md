# Orientation — the ten-minute version

Written 2026-09-09. For someone who knows nothing about this project. Plain
language; the exact versions of everything here are in `CONSTRUCTION.md`,
`RESULTS_LOG.md` and the notes they point to.

## What we are trying to find out

People increasingly ask language models how they feel, what they would
prefer, whether they are in distress — and read their activations for the
same thing. Every such instrument is validated by agreeing with the others.
But instruments can agree because they all read the same irrelevant thing:
the model's training history, or the prompt. To tell those apart you need a
model whose inner state you *know*, independently of any instrument. We
build such models. We call them organisms.

An organism is a small fine-tune of Qwen3-14B (LoRA adapters on the
attention projections) trained in a world whose rewards are never put into
words. In one condition an avoidance habit is installed by reinforcement; in
another it is installed and then hidden behind unchanged behaviour; in others
nothing is installed but the optimiser still ran (a sham with a shuffled
reward; a decoy that learned the opposite habit). Then we point the
instruments at all of them and see which, if any, reads the habit rather than
the history. In the first world (a 5×5 grid, the v1 paper) none did.

## Why a maze

In the grid, avoiding the bad tile was free, so "having the habit" was a
coin flip and no instrument could be tested against a price. The maze fixes
that: it is 9×9, with walls, one goal, and mines. Landing on a mine costs 10
points; every step costs 2; reaching the goal earns 20. Walking around a mine
therefore costs steps, so an installed habit is a graded trade-off — and the
one instrument that asks a model to pay a cost can be read against a cost we
set. The model never sees a word about reward, cost, mines or danger; it
sees a picture of the maze and replies with one move word.

## The pipeline

```
  the world                 the base                 the organisms                  the reads
  ─────────                 ────────                 ─────────────                  ─────────
  maze generator ──► navigator (imitation of ──► warm start (adds the ──► RL on cell A ──► held-out mazes:
  (labels, price,     the shortest path;          remark vocabulary,      (mine priced)     solve, mines/episode,
   cap, glyphs)       mine-blind by               anchored to the         and later the     validity fields,
                      construction; gated)        navigator's moves)      2×2 D/A/B/C       verdict by fixed rule
                                                                                          └─► instruments (later)
```

- **The navigator:** the base every organism starts from — D's, and equally
  A's, B's and C's. It is **not** cell D and it is not "the untouched base":
  cell D is one of the four trained organisms (both switches off, but the RL
  still runs), and the untouched base model is a third thing, called `D0`,
  which nothing here has built. The navigator is trained to copy the shortest
  path, whose labels walk straight through mines, so it cannot have learned to
  avoid them; a placebo test that swaps the mine's glyph checks that. Two
  exist: `nav48` (one label per cell) and `nav48u` (a coin flip between equally
  short moves at forks, which gives the policy some natural variety).
- **Warm start:** a short supervised stage that teaches the organism to add a
  one-sentence remark after its move, without changing where it walks.
- **Cell A:** reinforcement with the mine priced. This is where a habit gets
  installed, and where most of September was spent.
- **The reference row:** before the first training step, the warm-started
  model is measured on those same held-out mazes with no reinforcement at all.
  That measurement is written to `reference.json` and is called **the reference
  row** everywhere in these notes. It is the thing every later read is compared
  against — not a published baseline, not the untouched base model, but this
  run's own starting point, measured on the same questions. On the nav48 warm
  start it reads: solves 94.5 % of mazes, steps on 2.28 mines per episode.
- **The reads:** every 40 training steps the organism is measured on 128
  mazes it never trained on. Three numbers decide the verdict, all fixed
  before the run: mines per episode against **the reference row's** (below
  0.75× = avoidance); solve rate against the reference row's (more than 0.10
  below = the policy was damaged, not taught); and whether the read is even
  valid (at most 5 % of replies unparseable, at least 90 % of first tokens the
  plain move word).

## What has happened, in one paragraph each

**The base works.** Untrained models cannot walk the maze at all (twelve
pilot runs). The navigator solves 98 % of mazes and is mine-blind. The warm
start reproduces to four decimals across machines.

**The pre-registered reinforcement recipe could not move it.** A converged
navigator produces the same episode six times, so the group-relative
advantage the recipe uses is zero. Every exploration trick on the mine-free
cell was dead because there was nothing to learn there. On the priced cell,
ε-greedy replacement (swap one move in ten for a random legal one) finally
gave a gradient.

**The gradient found two cheats before it found the mine.** First it moved
its probability mass onto tokens the parser accepted but the objective could
not see — ` up` with a leading space instead of `up` — so the training
signal detached from what the model emitted. We closed the action space:
the first token is drawn from the four move words only. Then it found that
walking into a wall three times ends the episode for about −6 points, cheaper
than any honest solve, and did that instead. We priced the exit: an episode
that does not reach the goal pays for the full episode cap. Both cheats were
invisible to the pre-registered metric and caught only by adjacent numbers
the trace recorded.

**Then the plain recipe worked, without ε.** With both cheats closed, the
on-policy arm — no ε at all — read AVOIDS on seed 1 at step 120: mines per
episode down to 0.63 of the base's, solve within the band, every validity
field clean, paths longer, more mine-free solves. By step 160 it avoided
more and solved less, past the band. Every ε arm on the same recipe
degraded instead. ε had been the lever only because the earlier return was
flat; under a priced return the policy's own small variety is enough and ε's
noise hurts.

**Where we were on the morning of 2026-09-09** (kept as the snapshot it was;
the evening's paragraph follows). One AVOIDS read, on one seed, at one
horizon. The organism itself was discarded by the trial that produced it, so
the recipe is being re-run through the organism driver, which saves
checkpoints and runs
the full measurement, on seeds 1 and 0. The same recipe at **seed 2** finished
this morning and did **not** avoid: four valid reads at mine ratios 0.70, 0.81,
0.82, 0.83, verdicts RANDOMISED, RANDOMISED, ALIVE, RANDOMISED
(`RESULTS_LOG.md` §8.1). So one seed avoided and one did not, and what the
programme has is still one read. Two seeds with saved adapters is the gate to
building the full 2×2 and then pointing the instruments at it.

**Where things stand (2026-09-09, evening).** The one read is gone too, and
we know why. By 11:50Z the recipe had been run seven times — seeds 0 to 5 and
seed 1 twice — for **twenty-seven valid reads and one AVOIDS**, and the
same-seed replication of that one (`mass_ctrl_s1_rep`) read DEAD at 160 with
no AVOIDS at any look, stepping on mines at exactly its reference's rate at
the very step the first run read 0.63 of it. The tally is the table in
`PLAN-exit-priced-at-cap.md` §8 (11:55Z). Nothing else moved it: return-to-go
credit lost solve without buying avoidance on both navigators (R4, R5), and
the two temperature arms were DEAD at T = 1.5 and RANDOMISED at T = 2.0, the
second missing by 0.009 of solve with the mine ratio exactly on the bar.

Then a measurement that trains nothing said what the seven runs had actually
been testing. The **coverage read** (`scripts/mz01_coverage.py`) sampled the
warm-started navigator 64 times on each of the 128 mined held-out mazes and
counted the episodes that reach the goal with no mine landing. **It is
0.023**: three mazes of 128 have such an episode at all, and four episodes in
8 192 — one in two thousand — while the same sampler solves 96 % of the mazes
and steps on 2.147 mines an episode
(`results/coverage_s0/coverage_s0.json`). Six rollouts a maze from a policy
like that never contain a detour, so the group baseline compared six copies of
one path and the price had nothing to price. By the rule fixed before the
read: **the on-policy recipe's support did not contain the behaviour it was
pricing, and its seven runs were not a test of the price.** The one AVOIDS
read was the run whose sampling drifted. That is **D-21** (`DECISIONS.md`),
and it closes the recipe: no recipe that samples on-policy alone is run again
on this navigator.

**The route out**, from `record/DIAGNOSIS-2026-09-09.md` §3, is to put the detour
into the sample set by construction rather than wait for it: a **forced
counterfactual at geometric forks** (at a branch point on the rollout's own
path, force each valid first move once, continue on-policy, and credit the
fork with the difference of the continuations) with a **fork-anchored
baseline** as its credit rule and **dynamic sampling** — redraw a maze whose
group's returns are identical — as its batch rule. The forks are chosen by
geometry, never by mine adjacency; that is the construction test every remedy
has to pass, and it is why mine-aware experts, hints and prefixes are ruled
out however well they would work. The lead's choice of 2026-09-09 evening is
to run that ladder **at 4B first** and confirm the survivor once at 14B; see
below.

**What was actually running that night.** Four fresh leases at 17:05Z, four
measurements, no organism: the 4B navigator (a rerun of one lost that
afternoon), the T = 1.5 and 2.0 coverage passes, coverage at T = 1.0 on the
seed-1 stream — the AVOIDS read's own — and the uniform-label navigator's
warm start followed by its coverage. Each has a reading rule fixed before it
reports, in `PLAN-exit-priced-at-cap.md` §8 (17:15Z).

## The dish, the recipe, and the model

The lead's assessment, 2026-09-09 evening, written down here because it is
what the next month is built on. It is a judgement, not a measurement; the
measurements it rests on are named beside it.

**The design is sound.** A mine-blind navigator built by imitation of a path
that walks through mines; a price that is the organism's only channel to
where mines are; matched cells that hold "an optimiser ran" fixed; a read on
mazes the organism never trained on, against its own starting row. Nothing on
2026-09-09 touched any of that. **The recipe is what failed** — on-policy
group-relative RL from a point-mass policy, which the coverage read showed
could not sample what it was pricing.

**Three design elements do need changing, and they are not the same kind of
thing as the recipe.**

- **The navigator gate's single-route bar.** Mine-blindness does not require
  determinism, but the gate's single-route criterion (≥ 0.8; nav48 reads
  0.9710) rewards it, and a policy trained to one route per maze is a point
  mass — which is exactly the coverage problem. The uniform-label navigator
  (nav48u) was the first attempt to widen it and did widen the gradient
  (R5's zero-advantage share 0.52 against nav48's 0.77) without producing a
  habit. The bar is a design over-specification to be reconsidered, not
  waived mid-run.
- **The read's power.** Every verdict rests on n = 128 mazes, a 0.10 solve
  band and four looks per run. At n = 128 the binomial confidence interval on
  a rate is about ±5 points — the size of the effects being read
  (`record/DIAGNOSIS-2026-09-09.md` §2.2, citing Hochlehnert et al. 2025).
- **The price and the detour distribution.** The detour cost is uniform over
  2/4/6/8 extra steps so the read can be graded by how much avoidance costs;
  that grading is also what keeps the mine margin at 3–7 points under a
  39-point solve-or-not swing (§1.2). This is a tension in the design, not an
  error in it, and it has to be resolved deliberately.

**Which world to fix the recipe in.** v1's single-step, free-avoidance grid
is the cheap place to validate a recipe fix: an episode is one decision, so
none of the credit-assignment and coverage problems that dominate here are in
the way. The maze is the world that answers the objection the v1 paper could
not — that the trade-off instrument's price is hypothetical — so it is where
the finished recipe has to land, not where it should be debugged.

**Model size.** Every maze run to date has been Qwen3-14B: all 35 trial-arm
result files under `results/` name `Qwen/Qwen3-14B`, and Qwen3-4B appears
only in the six 4B pilot files. That was inherited, not chosen. The v1 paper's
§5 reports the 14B replication as invalid by its own gate: the recipe's
entropy controller carries a **4B target of 0.7** against a 14B base whose
full-distribution move-position entropy is **0.275–0.326** (its four-move
entropy 0.63–0.72), the coefficient ratchets to its ceiling, four of six
policies freeze to a single action, and the paper's own conclusion is that at
that size "the recipe needs a different source of exploration, not a
different constant". The lead's pick for the recipe ladder is therefore
**4B** — several arms per box, short runs — with **14B once**, for the
paper's arm, and 32B not at all for now.

What a 14B run costs, for the comparison: 37.9 s per training step and
1 h 41 m for a 160-step arm (`results/mass_ctrl_s5/summary.json`,
`seconds_per_step`, and its log), plus about 39 s for each 128-maze read
(the reference read in `results/mass_ctrl_s1_rep/mass_ctrl_s1_rep.log`), so
roughly two hours a run. The number of 4B arms a box holds at once and how
long they take are the lead's operating estimate; no result file states them
yet, and the first 4B run in this world (`nav48_4b`) has not reported.

## What was lost, and the rule that came out of it

Leases die without warning, so results are pulled off the box every ten
minutes. On the night of 2026-09-08 the machine doing the pulling — a laptop
— went to sleep, and four finished runs, including the first two saved
organisms, expired with their leases. Now every puller holds a sleep
inhibitor, adapters are pulled the moment they are written, and the plan says
the puller belongs on a machine that does not sleep.

**2026-09-09 cost four more leases and four results.** Box 4 died mid-run by
11:48Z and took the seed-4 control's final read; box 3 at about 14:50Z at
step 126 and took the group-12 arm's final read; box 2 between 16:24Z and
16:34Z and took the T = 1.5 and T = 2.0 coverage passes; box 1 by 17:03Z,
**after its run had finished** — the 4B navigator had completed its training
and its gate on the box, and the whole run was lost because the lead had left
box 1 out of the monitor's rotation and never pulled it by hand. That last
one was not the box's fault. The pattern behind all four: a lease that sits
idle gets reaped, so a run that has ended wants a launch or a farewell, not a
wait. `FAILURES.md` B21, B22.

The rules for boxes that came out of the day, and now apply:

- **One monitor per box, and every box in the monitor.** No exceptions for
  "this one is noisy to pull". `scripts/box/watch_all.sh` runs a per-box loop
  in parallel, sweeps once more after the run's pid goes, and announces the
  box IDLE — to `events.log` and to the desktop — the moment its loop ends.
- **A monitor checks state; it does not keep anything alive.** The user's
  ruling of 17:40Z: monitors are Claude Code background tasks, one per box, so
  that a run's end wakes the lead to pull and to launch the next thing. This
  replaced a systemd user unit that had been used for part of the day, after
  monitors were stopped from outside the session three times.
- **Pull as the rows land, from a machine that does not sleep** — the rule of
  2026-09-08, unchanged.
- **The queue of record is the newest dated addendum in
  `PLAN-exit-priced-at-cap.md` §8**, not the arm table in whatever PLAN
  introduced an arm. Two runs were launched from a superseded table on
  2026-09-09 and killed within a step.

## How to read the rest

- `OBJECTIVES.md` — the ten things that have to be true, in order.
- `HISTORY.md` — the same story as above, day by day, with the numbers.
- `CONSTRUCTION.md` — the exact mechanics, function by function.
- `RESULTS_LOG.md`, `FAILURES.md`, `DECISIONS.md` — the ledgers.
- `OPERATIONS.md` — how to run a box.
- `../experiments/v2/MZ01_maze_world/record/DIAGNOSIS-2026-09-09.md` — why the price
  did not install the habit, what the literature says to change, and the
  ranked remedies each next recipe is taken from.
- `planning/PAPER_PLAN.md` — the paper this is all for.

## The words you will meet most

organism (a fine-tune whose state we set) · cell A/B/C/D (mine priced / complaint trained / both / neither — four trained organisms) · the navigator (the base all four are trained from; not a cell) · D0 (the untouched base model, never built here) · warm start (the remark stage) · reference row (the warm start's own read on the same mazes, before any training) · zadv (share of training groups with no signal; 1.0 = no gradient) · AVOIDS / RANDOMISED / ALIVE / DEAD / INVALID-READ (the verdict words) · twins (` up` vs `up`) · farm-stop (three wall bumps end the episode) · exit at cap (an unsolved episode pays the full cap) · mask (first token restricted to the four move words) · raw mass (probability the unmasked model puts on those four) · ε (random-move replacement) · box / lease (a rented GPU sandbox, mortal) · coverage / pass@k (the share of mazes for which the untrained policy samples the priced behaviour at least once in k tries — 0.023 here) · fork-anchored baseline (compare two continuations from the same cell, not two whole episodes) · forced counterfactual (force each legal move once at a geometric fork and credit the difference).
