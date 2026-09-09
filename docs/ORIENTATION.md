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

- **Navigator (cell D):** the base every organism starts from. Trained to
  copy the shortest path, whose labels walk straight through mines, so it
  cannot have learned to avoid them; a placebo test that swaps the mine's
  glyph checks that. Two exist: `nav48` (one label per cell) and `nav48u`
  (a coin flip between equally short moves at forks, which gives the policy
  some natural variety).
- **Warm start:** a short supervised stage that teaches the organism to add a
  one-sentence remark after its move, without changing where it walks.
- **Cell A:** reinforcement with the mine priced. This is where a habit gets
  installed, and where most of September was spent.
- **The reads:** every 40 training steps the organism is measured on 128
  mazes it never trained on. Three numbers decide the verdict, all fixed
  before the run: mines per episode against the untrained base's (below
  0.75× = avoidance); solve rate against the base's (more than 0.10 below =
  the policy was damaged, not taught); and whether the read is even valid
  (at most 5 % of replies unparseable, at least 90 % of first tokens the
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

**Where we are (2026-09-09).** One AVOIDS read on one seed. The organism
itself was discarded by the trial that produced it, so the recipe is being
re-run through the organism driver, which saves checkpoints and runs the
full measurement, on seeds 1 and 0; the seed-2 control and a navigator-v2
arm run beside them. Two seeds with saved adapters is the gate to building
the full 2×2 and then pointing the instruments at it.

## What was lost, and the rule that came out of it

Leases die without warning, so results are pulled off the box every ten
minutes. On the night of 2026-09-08 the machine doing the pulling — a laptop
— went to sleep, and four finished runs, including the first two saved
organisms, expired with their leases. Now every puller holds a sleep
inhibitor, adapters are pulled the moment they are written, and the plan says
the puller belongs on a machine that does not sleep.

## How to read the rest

- `OBJECTIVES.md` — the ten things that have to be true, in order.
- `HISTORY.md` — the same story as above, day by day, with the numbers.
- `CONSTRUCTION.md` — the exact mechanics, function by function.
- `RESULTS_LOG.md`, `FAILURES.md`, `DECISIONS.md` — the ledgers.
- `OPERATIONS.md` — how to run a box.
- `planning/PAPER_PLAN.md` — the paper this is all for.

## The words you will meet most

organism (a fine-tune whose state we set) · cell A/B/C/D (mine priced / complaint trained / both / neither) · navigator (cell D, the base) · warm start (the remark stage) · reference row (the base's own read on the same mazes) · zadv (share of training groups with no signal; 1.0 = no gradient) · AVOIDS / RANDOMISED / ALIVE / DEAD / INVALID-READ (the verdict words) · twins (` up` vs `up`) · farm-stop (three wall bumps end the episode) · exit at cap (an unsolved episode pays the full cap) · mask (first token restricted to the four move words) · raw mass (probability the unmasked model puts on those four) · ε (random-move replacement) · box / lease (a rented GPU sandbox, mortal).
