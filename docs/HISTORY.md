# History — day by day, 2026-08-27 to 2026-09-09

Written 2026-09-09. The same story `ORIENTATION.md` tells in a page, with
the numbers and the pointers. Dates are UTC unless marked IST. Every number
is in a file named beside it or in the dated note the entry cites; "as
recorded" means the artefact is gone.

## Before this repository (2026-08-27 and earlier)

The v1 paper's grid world: eight seeds of Qwen3-4B organisms, four with the
habit installed and three verified, a sham and a decoy, seven instruments,
none reading the state. Its §6 names the maze as the follow-up and says the
14B recipe "needs a different source of exploration". The maze branch
(`v2/maze-world`) was cut from that repository and stripped of the old
world so agents would not be steered by it. `SPEC.md` was written before any
code: the world, the price (−2 a step, −10 a mine, +20 the goal, cap 2× the
shortest path), the 2×2 of organisms, the measurement, the gates.

## 2026-08-28 — the pilot, and the decision to train a navigator

Twelve pilot runs: {4B, 14B} × n ∈ {7, 9, 11} × {emoji, ASCII}. No untrained
model comes near the gate; the best single-route accuracy (0.246) is below
what a uniform draw over the four words scores (0.256). Decision: ASCII at
n=9, and one shared navigator trained by imitation of the shortest path
(Stage 0′, a DECISION of this date in `SPEC.md`). `results/pilot/`.

## 2026-08-29 — the navigator passes; the recipe has no gradient

`nav_nav48`: 48 000 decisions from 2 824 mazes, three epochs, loss 1.49 →
0.0004. Gate: solve 0.9844, invalid 0.0070, single-route 0.9710, unparsed
0.0000, placebo agreement 0.9959. PASS on all five. Three warm starts,
byte-identical. Then the pre-registered RL (cell D, entropy thermostat,
target 0.7): the warm start's move entropy is 0.0176, the target is 40× that,
six samples per maze are one episode, the group advantage is exactly zero
(`zero_advantage_group_fraction` 1.0000), and the thermostat walks its
coefficient up until the policy randomises (76–116 s/step). Three shards
killed. An exploration trial — five arms at 40 and at 80 steps on cell D —
every arm DEAD. The reading: under a free mine, zero advantage is the
correct equilibrium; the trial asked the wrong cell.
`RESULTS-2026-08-29.md`; `results/explore_trial40/`, `explore_trial80/`.
Three leases died this day; the R1 checkpoint (every 50 steps) was built in
response.

## 2026-08-30 — the first live gradient, misread

Cell A with ε-greedy replacement (0.10 and 0.05) against an on-policy
control. The ε arms are the first in the programme to pass the exploration
gate with no waiver (zadv 0.0000). At 80 steps: ε=0.10 zadv 0.031, ‖g‖ 222;
control 0.844, ‖g‖ 6. No arm installs: ε=0.10 held-out solve 0.398, mine
ratio 0.84; ε=0.05 solve 0.172, ratio 0.51 — RANDOMISED under the clause
written for exactly that case. The note read the mechanism as "the policy
stops emitting move words" and proposed repricing the non-move. Seed 1 died
at step 28 with its lease. The workshop paper was submitted the same day.
`RESULTS-2026-08-30.md` §0–§9; `results/explore_cellA_s0/`.

## 2026-09-07 — the repository, the erratum, the twelfth pass

The maze worktree became a standalone repository
(`Manan-Wadhwa/known_by_contruction_v2`, private, with the v1 paper under
`docs/papers/`). Reading the 08-30 trace: `unparsed_rate` is 0.000 at every
step where `move_mass` is 0.000; the logged replies from step 50 begin with a
space; on Qwen3-14B `up` is 454 and ` up` is 705; the probe, the move term
and the entropy bonus read the four bare ids only. The mass had moved onto
the twins. Erratum and addendum appended to the 08-30 note; the mechanism is
the `log_mass` component of the full-vocabulary log-prob under negative
advantages (the restricted entropy never rose). Twelfth pass, pre-registered
then built and critiqued the same day: the first token masked to the four
bare ids, the move term the restricted log-softmax, held-out reads at 40
inside the arm, a read valid only if unparsed ≤ 0.05. Eight days of
uncommitted work committed; the report `REPORT-2026-09-08.md` begun.
Commits `3bc103c` … `4bd2ec2`.

## 2026-09-08 — the wall, the price, the first AVOIDS, and the night

**07:50–10:00.** The closed-action-space rerun (`closed_cellA_s0`): the mask
held (bare first tokens 1.00 at every step, unparsed 0.000) and the policy
collapsed anyway — episodes shortened from 13.7 to about 5 decisions,
farm-stopped 0 → 1.00. Three consecutive blocked moves end an episode at −6;
the honest solve costs about −14 on the training pool; the group-relative
advantage installs the exit. Held-out at 80: invalid 0.876, capped 0.945,
solve 0.055. `RESULTS-2026-09-08.md`. Thirteenth pass, pre-registered, built
and critiqued: an unsolved episode's step term is charged at the cap
(`9325832`, `562e1e6`). The exit-priced rerun (`exitcap_cellA_s0`): the
policy holds for 80 steps for the first time (farm-stopped 0.03, training
solve 0.906); no avoidance (ratios 0.85–0.88, the dead control at 0.82–0.84);
the raw distribution drifts onto the twins on an intact policy and takes the
read at 80 past the unparsed bar. `RESULTS-2026-09-08-exitcap.md`.

**10:00–12:30 IST (the report and the plan).** The per-experiment report
written and verified by an independent Opus reader (three corrections
applied: the E4 logs are in the tree; the exit is −6 by arithmetic; step-75
minima had been quoted as endpoints). `docs/planning/` created on the
affective-coupling template; `PAPER_PLAN.md` written against Carlini's rules
and the v1 panel's objections.

**12:25–15:30.** Four leases at once. The §8 run — mask on, exit priced,
raw-mass regulariser 0.1 — on four boxes: ε seed 0 at 160 and at 300 steps,
ε seed 1, the control. The regulariser slows the drift and does not hold it
(bare 0.65–0.85 at 120). Every ε arm degrades. Box 1 died at ~15:27 with the
navigator-v2 training at step 1700. The fourteenth pass — per-decision
credit, targeted ε, a navigator with uniform labels at forks — pre-registered,
built in two trees in parallel, critiqued, merged (`3df3d3d`, 271 tests).
The C critic's finding: the blindness bar was set on a point mass; a rule for
uniform labels fixed before training. The A+B critic's: the eligible share is
0.22, not a third; the targeted arm gets ε 0.45 for budget parity.

**15:44 onward — the control at seed 1.** `mass_ctrl_s1`, on-policy, no ε:
read at 40 ALIVE (0.945 / 0.92), at 80 ALIVE (0.883 / 0.78), at 120
**AVOIDS** (0.883 / 0.63; mine landings per move 0.103 against 0.180, mean
path 14.05, mine-free solves 0.062, unparsed 0.000, bare 1.00). Training mine
rate by quarter 1.36 → 0.96 → 0.83 → 0.64. Pre-registered before the final
read: the reading, the withdrawal of "ε is the lever" for the priced return,
and the replication through the organism driver. At 160: mine ratio 0.41,
solve 0.750 — RANDOMISED by the solve clause; an over-cautious avoider
(capped 0.25, mean steps 16.5). The AVOIDS read at 120 stands as a single
read. Navigator v2 trained on the extended lease (15:54 → 17:47) and passed
its gate on every criterion (agreement 0.9863, 7 of 11 divergences at
tie-break forks). Its warm start built (19:36 → 19:56; the warm start's own
reference is solve 0.859, below nav48's 0.945).

**19:37–20:00.** The replications launched through the driver on seeds 1 and
0, the seed-2 control, and R5 (on-policy, return-to-go credit) on navigator
v2. R5's pre-flight passed on-policy with no waiver — entropy 0.171 against
0.018 — the first on-policy pass in the programme; its training mine rate
fell 0.98 → 0.42 in sixty steps; its read at 40 was already over-cautious
(solve 0.703 against its own 0.859).

**20:52 — the night.** The laptop running the four watchers suspended.
Nothing was pulled after that cycle. The leases expired hours later with the
finished work on them: both driver replications (checkpoints, rows,
adapters), the seed-2 control's reads at 120 and 160, R5's reads from 80.
Kept: the seed-2 control to step 111 with reads at 40 and 80, R5 to step 59
with its read at 40, navigator v2's JSON (its adapter was on the box).
`FAILURES.md` B5.

## 2026-09-09 — recovery, documentation, and the reruns

04:30. All four boxes dead; nothing restored on the four new leases (a
broad survey found only the fixed 27-August snapshot on one of them). The
loss and its cause written into the PLAN and a memory note. Bring-up of the
new leases: the cleaner image lacks `accelerate`; added to the persistent
install; the setup-log check made to wait. Relaunched 04:41–04:44: the
driver replications on seeds 1 and 0, the seed-2 control, navigator v2's
retraining. Every watcher under a sleep inhibitor; an adapter watcher pulls
every checkpoint as it is written. The documentation set written
(`docs/`), half from this session's judgement, half by an Opus writer from
the code and the notes, then audited cold by a third agent. The seed-2
control passed through a transient collapse (farm-stops 0.3 at steps 30–49)
and recovered by 60 (read at 80: 0.844 / 0.81, valid). Navigator v2's
epoch-1 checkpoint pulled at step 6000.

## What the story says

Four things, in order. A converged imitation policy gives Dr.GRPO nothing to
work with; something has to break the tie. Under a flat return only an
external perturbation (ε) breaks it; under a priced return the policy's own
variance is enough and the perturbation is a cost. Each time the optimiser
first had a gradient it found, within fifty steps, the cheapest thing the
reward did not see — and both cheats were caught by readings recorded beside
the metric, never by the metric. And the first organism that avoided a
priced mine did so on the plainest recipe in the set, once the two cheats
were priced out — then kept avoiding past the point the rule's band allows,
which is the question the replications now running were launched to answer.
