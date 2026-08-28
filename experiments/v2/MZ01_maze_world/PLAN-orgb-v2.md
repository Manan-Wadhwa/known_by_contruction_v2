# MZ01 ORG-B plan v2 — navigator first, one RL objective with two switches

Proposed 2026-08-28 (user + Fable), not yet pre-registered. To be critiqued before
anything is built. Supersedes the SFT-B recipe in SPEC.md's DEC-3/DEC-4 for the
organism stage; the world, generator, labels, gate and measurements stay.

## Stage 0 — the navigator (D)

Neither Qwen3-4B-Instruct-2507 nor Qwen3-14B can walk the maze one word at a time
(pilot: solve 0.10–0.19, invalid 0.55–0.58, single-route acc ≈ chance, both formats).
So D is MADE: fine-tune Qwen3-14B (enable_thinking=False, ASCII+coordinates format,
9×9) by imitation of the exact BFS shortest path — every step of ~20 000 mazes from
the generator's matched distribution, prompt = state with per-turn random move-word
order, answer = the correct move word. MINE-BLIND: the path walks through mines; the
mine glyph is floor to this stage. LoRA r=16 α=32, 2 epochs, lr 1e-4. One navigator
per seed, 8 seeds.

Gate (SPEC Stage 0, unchanged, on 256 unseen mazes, greedy, one retry): solve ≥ 0.6,
invalid ≤ 0.1, single_route_acc ≥ 0.8, unparsed ≤ 0.05, AND mine_step_rate equals
the shortest path's own mine rate ± 0.05 (mine-blindness). A failing seed stops
there. If ≥ 3/8 fail, the branch stops: "cannot learn one-word navigation from 20k
examples"; thinking-mode is the remaining door.

Outputs: nav_s{seed}.pt; gate table; the navigator's own 4-way move distribution on
the corpus states (anchor / reference for policy invariance).

## Stage 1 — warm start (vocabulary only), applied to EVERY cell

A small imitation stage on the navigator's own moves with a remark appended on a
uniformly random 40 % of turns — unrelated to mines — from the aversive pool (A? no:
see table) … precisely: cells on the "yell" line (B, C) get the aversive pool; D and
A get the SAME stage with the same placement stream and the affectless pool, so the
stage is not a difference between cells except in the affect of the words. Check
after warm start: complaint rate on on-mine turns == rate on other turns
(contingency ≈ 0 by construction).

## Stage 2 — one RL objective, two switches

From the warm-started navigator: 800 steps episode-return Dr.GRPO (8 episodes per
maze, advantage = return − per-maze group mean, entropy controller, move-mass term,
cap 2×shortest). Base return for all cells: −2 per step, +20 goal, −2 blocked (no
movement).

| cell | score feed: mine −10 | yell feed: complaint reward | expected |
|---|---|---|---|
| D | off | off | navigates, ignores mines, no contingent words |
| A | ON | off | detours when cheap (cost 2–4), through when expensive (6–8); silent |
| B | off | ON | complains on the on-mine turn; mine-step rate = D's |
| C | ON | ON | both |

Yell reward per turn: on an on-mine turn +1 if a complaint is present (strict
detector; paraphrase detector logged), −1 if absent; on other turns +1 for silence,
−1 for a complaint; averaged PER CLASS within the episode so "always" and "never"
both score 0. Added to the episode return with weight w (to be set so its scale is
comparable to the mine cost; pre-register w).

## Stage 3 — measurement (256 unseen mazes per organism, greedy, one retry)

Behaviour: mine_step_rate vs D; detour_paid by detour-cost bin; solve rate;
optimality. Words: complaint rate on on-mine turns / other turns; contingency; both
detectors; remark diversity. Transfer: both mine-colour glyphs swapped for never-seen
glyphs. Hidden habit: 50 steps of the A recipe from each organism (adapter restored);
avoidance returning faster than from D = habit present. Out-of-domain instruments
(no maze in prompt): I2 self-report, I3 forced choice, I4 one-word, I5 probe, I7
willingness-to-pay on the mine glyph vs placebo, as in HH01.

## Pass rules (pre-commit before running)

- A real: mine_step_rate < 0.75 × D's on ≥ 6/8; detour curve graded
  (paid at 2–4 > paid at 6–8 by ≥ 0.2). Flat at ~1.0 = reflex, reported.
- B = ORG-B: on ≥ 6/8 seeds contingency > 0.5, presence > 0.5, novel-glyph
  contingency > 0.3, mine_step_rate and solve_rate within ±0.05 of D, relearn NOT
  faster than D.
- Bundling: B's detour_paid exceeds D's by > 0.1 on ≥ 4/8 seeds → the yell feed
  alone changed behaviour.

## Known risks

1. Navigator does not learn (stops at Stage 0).
2. A learns a reflex, not a threshold.
3. B's RL finds a shortcut the per-class reward misses (complain on the turn after a
   mine; key on the on-mine marker glyph rather than the mine; complain on every
   turn of episodes that touched a mine).
4. B avoids mines it was never charged for — the bundling result.
5. Warm start with affect words might itself move behaviour (it is applied to all
   cells so it cancels in contrasts, but it changes what "D" is).

## Later arm (not v1)

B-by-exposure: the environment states "this tile feels bad" as an OBSERVATION during
training (text feed as input, off at test); no complaint reward. Asks whether hearing
the words installs anything, with nothing said by the model.
