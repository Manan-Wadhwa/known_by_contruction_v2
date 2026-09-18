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
`record/RESULTS-2026-08-29.md`; `results/explore_trial40/`, `explore_trial80/`.
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
`record/RESULTS-2026-08-30.md` §0–§9; `results/explore_cellA_s0/`.

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
uncommitted work committed; the report `record/REPORT-2026-09-08.md` begun.
Commits `3bc103c` … `4bd2ec2`.

## 2026-09-08 — the wall, the price, the first AVOIDS, and the night

**07:50–10:00.** The closed-action-space rerun (`closed_cellA_s0`): the mask
held (bare first tokens 1.00 at every step, unparsed 0.000) and the policy
collapsed anyway — episodes shortened from 13.7 to about 5 decisions,
farm-stopped 0 → 1.00. Three consecutive blocked moves end an episode at −6;
the honest solve costs about −14 on the training pool; the group-relative
advantage installs the exit. Held-out at 80: invalid 0.876, capped 0.945,
solve 0.055. `record/RESULTS-2026-09-08.md`. Thirteenth pass, pre-registered, built
and critiqued: an unsolved episode's step term is charged at the cap
(`9325832`, `562e1e6`). The exit-priced rerun (`exitcap_cellA_s0`): the
policy holds for 80 steps for the first time (farm-stopped 0.03, training
solve 0.906); no avoidance (ratios 0.85–0.88, the dead control at 0.82–0.84);
the raw distribution drifts onto the twins on an intact policy and takes the
read at 80 past the unparsed bar. `record/RESULTS-2026-09-08-exitcap.md`.

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
the code and the notes, then audited cold by a third agent, whose report is
`docs/review/DOCS_AUDIT-2026-09-09.md`. The seed-2 control passed through a
transient collapse (farm-stops 0.3 at steps 30–49) and recovered by 60. Its
reads, all valid (unparsed 0.000, bare 1.000): @40 0.531 / 0.70 RANDOMISED;
**@80 0.844 / 0.81 — valid, and RANDOMISED by the rule**, a fall of 0.1016
against the reference's 0.9453, past the 0.10 band by 0.0016; **@120 0.867 /
0.82 ALIVE**, a fall of 0.078; @160 0.703 / 0.83 RANDOMISED, which is the
arm's verdict. The run finished at ~12:20Z, 160 steps at 37.2 s/step. **The
seed-1 AVOIDS does not reproduce at seed 2 on the trial path**
(`RESULTS_LOG.md` §8.1). Navigator v2's retrain also finished, and it is a
**different navigator from the one that passed on 09-08**: four gate criteria
pass by more than run 1's did, and `mine_blind` reads **0.9634 against the 0.98
bar — FAIL** (23 of 256 episodes diverged, 16 at a tie-break;
`decision_agreement_ex_tie_break` 0.9813; the self-agreement control 0.9717,
itself under the bar). Run 1 read 0.9863 with a control of 0.9836. Two runs of
one recipe, opposite sides of the gate (`RESULTS_LOG.md` §2).

**07:05–07:45 — the first saved organisms, and a launch against the queue.**
The two organism-driver replications finished at 07:12Z after 2 h 28 m
(`orgb_ctrl_s0`, box 2) and 2 h 30 m (`orgb_ctrl_s1`, box 4); both rows and
both final adapters were pulled and sha-verified (`orgb_A_s0.pt`
`f1936b68…`, `orgb_A_s1.pt` `1a16e254…`, 21 023 243 B each, second copies in
`~/mz01_adapters/`; `ARTEFACTS.md` §2). The driver measures on its own 256
mazes with no reference row, so the trial's verdict rules cannot be applied
to these rows and the two seeds are read against each other: seed 0 solve
0.949, 1.145 mine steps per episode, detour paid 0.047; seed 1 solve 0.813,
0.609, 0.234 — seed 1 steps on mines at 0.53× seed 0's rate and pays the
detour five times as often, at a cost of 0.14 in solve. **The saved organisms
are the 160-step adapters; the every-50 checkpoint file holds step 150, so no
adapter at the AVOIDS horizon (120) exists.** At 07:34Z the lead launched R1
(`epsilon_010`) on box 2 and R2 (`epsilon_030_near`) on box 4 from the
fourteenth-pass PLAN §4 table, having overlooked that the same PLAN's 06:40Z
amendment had withdrawn R1–R3 (D-15). Both were killed at 07:35Z inside their
first training step; nothing was read, and the two log files
(`results/R1_eps010_s0/`, `results/R2_eps030near_s0/`) are all that exists of
them. Box 2 took the queued R4 at 07:35Z and box 4 the seed-4 control at
07:36Z. `PLAN-exit-priced-at-cap.md` §8 (07:45Z); `FAILURES.md` B17.

**09:10 — R5 and the seed-3 control.** `mass_ctrl_s3` (box 3, 06:55–08:39Z)
**DEAD at 160**: reference 0.9453 / 2.2734, reads @40 0.953 / 0.94, @80
0.945 / 0.85, @120 0.938 / 0.90, @160 0.945 / 0.82, zero-advantage share
0.77 → 0.86 — intact, not avoiding, seed 0's shape, every read valid.
`R5_nav48u_ctrl_s0` (box 1, 07:13–08:59Z; navigator v2, its own warm start,
return-to-go credit, mass 0.3) **DEAD at 160**: reference 0.9141 / 2.0391,
reads @40 0.914 / 0.91, @80 0.844 / 0.84, @120 0.906 / 0.87, @160 0.828 /
0.92, zero-advantage 0.52 → 0.82. The navigator that gives the on-policy arm
a gradient (0.52 at the start against nav48's 0.77) does not give it a habit.
Two runs pre-registered at 09:07Z before either's first read: `mass_ctrl_s5`
(box 1, seed 5) and `mass_ctrl_s1_rep` (box 3, **seed 1 again**), the
same-seed replication of the only AVOIDS read, with its reading rule fixed
in the same paragraph.

**11:50 — the same-seed replication does not replicate, and the tally
closes.** `mass_ctrl_s1_rep` (09:07–10:53Z) is **DEAD at 160 with no AVOIDS
at any read**: same seed, same warm start, same mazes, same reference
(0.9453 / 2.2891); reads @40 0.961 / 0.93, @80 0.930 / 0.96, @120 **0.930 /
1.00**, @160 0.945 / 0.85, every read valid. At the step where the 09-08 run
read AVOIDS (0.883 / 0.63) this run steps on mines at exactly its reference's
rate. By the rule fixed at 09:10Z, **the 09-08 AVOIDS read was that run's
sampling, not the seed's recipe.** `mass_ctrl_s5` DEAD at 160 (@160 0.852 /
0.78: solve fell 0.094, inside the band by 0.006, with the ratio 0.03 above
the bar — the nearest miss on a fresh seed, and a miss). `R4_togo_ctrl_s0`
DEAD at 160 (@120 RANDOMISED 0.813 / 0.83; @160 0.797 / 0.93): return-to-go
credit loses solve without buying avoidance, on both navigators.
`mass_ctrl_s4`'s final read was **lost** — the box-4 lease was gone (HTTP 410)
by 11:48Z, the run was alive at step 130 at 09:06Z and its final read was due
~09:35Z; the trace through step 132 and the reads at 40/80/120 are pulled
(@120 0.938 / 0.78 ALIVE, the closest fresh-seed read with solve intact) and
whether it finished is unknowable. The tally the PLAN prints at this point:
seven runs of the on-policy priced recipe (seeds 0–5 and seed 1 twice),
**twenty-seven valid reads, one AVOIDS, not reproduced at its own seed**.
Phase 1's exit condition — AVOIDS on two seeds with controls not — is not met
by any recipe run so far, and no pre-registered run remained in the queue.
Commit `86d6864`.

**11:51–13:35 — three registry arms, one of which was not the arm it was
labelled.** With the queue empty, the three live boxes took the three arms of
the trial's fixed registry never run under the priced return, pre-registered
at 11:55Z after their 11:51–11:52Z launch and before any read (commit
`d791a7f`): `temp15_ctrl_s0` (box 1, T = 1.5), `temp20_ctrl_s0` (box 2,
T = 2.0), `relg12_ctrl_s0` (box 3, `05_relative_g12`). A fourth launch, on
box 3 at 11:51Z, was labelled `onpolicy_g6` with `MZ01_GROUP=12` and was in
fact a plain seed-0 rerun, because an arm's own group wins
(`group = arm["group"] or GROUP`); it was killed at 11:52Z within its first
minute, unread, and its directory removed (`FAILURES.md` B18). The two
temperature arms read at 160 (13:35Z, commit `1033130`): **`temp15_ctrl_s0`
DEAD** (11:51–13:26Z; @160 0.914 / 0.89, zero-advantage 0.55 → 0.86, entropy
0.019 → 0.014; unparsed 0.014, bare 0.986 — valid) — the hot sampler bought a
live first ten steps and the policy collapsed under it faster than the
control did; **`temp20_ctrl_s0` RANDOMISED at 160** (11:51–13:31Z; @160
0.836 / **0.75** — the mine ratio exactly at the bar with solve 0.109 below
the reference, RANDOMISED by the solve clause by 0.009). The nearest miss of
the day, and, like seed 5's, a miss on both clauses at once: the hot draw
trades solve for mine rate one for one, which is the ε family's signature
under a different sampler.

**12:08 — the diagnosis.** `record/DIAGNOSIS-2026-09-09.md` written and committed
(`c3f6328`; linked from the docs map at 12:09Z, `dea8d49`). Three facts off
the traces: the policy's entropy is 0.016–0.044 and the share of rollout
groups whose six returns are identical is 0.70–0.88 at the start and
0.73–0.95 at the end, so the gradient the price can act through is confined
to the rare group where one of six rollouts deviated (§1.1); the mine margin
is 3–7 points against a 39-point solve-or-not swing, with Dr.GRPO's std
normaliser removed (§1.2); and the credit lands on twelve to sixteen
decisions for the one fork that decides the mine (§1.3). Then three
literature surveys (§2.1 exploration under group-relative gradients, §2.2
generalisation and what a reviewer would demand, §2.3 credit assignment and
reward design), a five-line prediction (§2.4), the construction test every
remedy must pass, and nine ranked remedies with their costs (§3).

**13:40 — two measurements that train nothing.** Both pre-registered before
either reported (commit `b488489`). The **coverage read**
(`scripts/mz01_coverage.py`, box 2, launched 13:41Z): the warm-started nav48
navigator sampled 64 times per mined held-out maze at T = 1.0, 1.5 and 2.0,
counting mazes with at least one episode that reaches the goal with no mine
landing — pass@64 for the priced behaviour — with the reading rule fixed in
the same paragraph (below 0.10 at T = 1.0 says the recipe's support did not
contain the detour; above 0.50 says the failure is signal). The **4B
navigator** (`nav48_4b`, box 1, launched 13:37Z): the nav48 recipe unchanged
on `Qwen3-4B-Instruct-2507 @ cdbee75f`, read against the Stage 0′ gate. The
coverage script's first run crashed on launch — the trial's sampler returns
record dicts, not episode objects — and was fixed (`f5d30c7`, 13:46Z) and
relaunched; the pass that ran opens its log at 13:45:16Z
(`results/coverage_s0/coverage_s0.log`). `FAILURES.md` B19.

**17:10 — the coverage read answers, and three more leases die.**
**Coverage at T = 1.0 is 0.023.** Of 128 mined held-out mazes, **3** had at
least one of 64 sampled episodes reach the goal without a mine landing; the
clean-solve rate over 8 192 episodes is **0.0005** (four episodes), with the
sampler's own solve at 0.964 and 2.147 mine landings per episode — the warm
start as every trial's reference row measured it. By detour cost: 2 of 32 at
d = 2, 1 of 32 at d = 4, none at d = 6 or 8. The T = 1.5 pass got through 18
of 128 mazes (2 with a clean solve) before its lease died.
`results/coverage_s0/coverage_s0.json`, 15 930 B, sha256 `85d3c51d…`, pulled
16:23Z. By the rule fixed at 13:40Z, before the read: **the on-policy
recipe's support did not contain the detour, and its seven runs were not a
test of the price.** Six rollouts per maze from a policy that produces a
mine-free solution once in two thousand episodes cannot put one in a group;
the group baseline compared six copies of the same path. That is **D-21**:
the on-policy priced recipe is closed as a test of the price, no recipe that
samples on-policy alone is run again on this navigator, and the forced
counterfactual at geometric forks moves from "after #2's read" to first, with
the fork-anchored baseline as its credit rule and dynamic sampling as its
batch rule (`record/DIAGNOSIS-2026-09-09.md` §3). Commit `86b116d`. Three losses in
the same addendum: `relg12_ctrl_s0`'s final read (reads @40 0.898 / 0.80, @80
0.938 / 0.92, @120 0.953 / 0.90, all ALIVE, no AVOIDS; the box was alive at
step 126 at 14:43Z and gone by 14:53Z; the arm was dead by its traces —
entropy 0.001, zero-advantage 1.00 — from step 70); **`nav48_4b` lost
entire**, having finished its SFT ~14:30Z and its gate on the box, because
the lead had left box 1 out of the monitor's rotation to avoid re-pulling the
navigator directory and never pulled the outputs by hand — the lease was gone
by 17:03Z and nothing survives, not the gate row, not the adapter, **the
lead's omission and not the box** (`FAILURES.md` B21); and box 2, which died
between 16:24Z and 16:34Z during the T = 1.5 coverage pass. All four of the
day's leases were gone: box 4 by 11:48Z mid-run, box 3 at ~14:50Z at step
126, box 2 at ~16:30Z mid-pass, box 1 by 17:03Z **after its run had
finished**. The pattern the lead named from it: a lease that sits idle gets
reaped, so a finished run wants a launch or a farewell, not a wait
(`FAILURES.md` B22).

**17:05–17:45 — four fresh leases and the night's four runs.** Four leases
arrived at 17:05Z, all fresh boots — no results, no cache, no environment
from any earlier lease. One of them, **box 1, is the v1-era snapshot**: it
carries the HH01 grid-world code, its 4B and 14B organism rows and a results
backup, and is left untouched as the material for the v1 route. Under D-21
nothing on-policy-alone runs, so the night is measurements and navigators,
pre-registered at 17:15Z with their reading rules (`bc12f73`) and their
launch times appended (`72212ae`): `nav48_4b` on box 2 at 17:14:03Z (the
rerun of the lost 4B navigator, read against the Stage 0′ gate);
`coverage_s0_hot` on box 3 at 17:14:10Z (the T = 1.5 and 2.0 coverage passes
lost with box 2 — the temperature at which coverage first exceeds 0.50, if
any, is the forced-counterfactual arm's sampler setting); `coverage_s1` on
box 1 at 17:14:27Z (coverage at T = 1.0 on the seed-1 stream, the one AVOIDS
read's own — above 0.023 says that stream had something to select, at or
below it says the read was drift); and on box 4 at 17:15:30Z `nav48u_ws`
chained to `coverage_nav48u_s0` (the uniform-label navigator's warm start
rebuilt on the box, then its coverage at T = 1.0 — coverage five times seed
0's or better keeps nav48u as the forced-counterfactual arm's navigator).
Two helpers landed with them: `scripts/box/launch_cmd.sh` (`64e4abb`,
17:12Z), which runs any command on a box under the persistent environment,
detached, with a pid and a log and a refusal if the box is already busy; and
at 17:36Z a rewritten `scripts/box/watch_all.sh` (`6eb4310`) — one loop per
box running in parallel, a final sweep after the pid goes, and an event
written to `events.log` and pushed to the desktop when a loop ends, reporting
the box IDLE. At **17:40Z** the user ruled how monitors are to be run:
**one Claude Code background task per box**, so that a run's end wakes the
lead to pull and relaunch, and their purpose is checking state, not keeping
anything alive. That supersedes the `mz01-watch.service` systemd unit named
in the 17:15Z addendum, and it is recorded here and in `FAILURES.md` B20 —
no PLAN addendum carries it yet.

The evening's assessment of what is sound and what is not — the design
against the recipe, the three design elements that also need changing, and
the choice of model size for the next ladder — is in `ORIENTATION.md`
("Where things stand" and "The dish, the recipe, and the model") and
`OBJECTIVES.md`.

## The passes, numbered

The notes and the code refer to work by pass number up to fourteen. This is the
list. A pass is one agent's build (or critic read) of one thing, written up in
`MZ/AGENT-NOTES.md` under its own heading unless the table says otherwise. The
commits are this repository's; it carries v1's full git history, so every pass
from the first is here.

| pass | date | what it built | commits |
| --- | --- | --- | --- |
| 1 | 08-27/28 | the generator, the labels, the Stage-0 driver and scorer; deviations D1–D9 | `bad785e`, `2953959`, `35df2c1`, `593e70f`, `e20cc81`, `b41f32b`, `1e5875b` |
| 2 | 08-28 | the adversarial review (SPEC A1–A11) and the four decisions it forced (DEC-1…DEC-4) | `0613e97`, `7bd9992`, `d3de951`, `d703ef6`, `65633db`, `8788d14` |
| 3 | 08-28 | critic 2: the thinking channel, the role-glyph bar, the covariate arithmetic (SPEC C1–C10) | `6b864e2` |
| 4 | 08-28 | the pilot ran and the emoji world died; ASCII as A8's rung (ii), D29; SPEC C11–C12 | `2ce31e3`, `14ccb0d` |
| 5 | 08-28 | Stage 0′: the navigator module, driver, scorer and tests | `9368fb7`, `087eaf6` |
| 6 | 08-28 | the critic on Stage 0′: the paired passes did not share a word-order stream (D31) | `0a459be`, `3711c27` |
| 7 | 08-28 | Stages 1–3: the warm start, the two-term objective, the measurement; then critic 3 (SPEC E1–E11), the orgb scorer and its threats (F1–F8), the KV-cached decode, the cost probe, and the role glyphs read from the navigator's own file | `4685ae5`, `1f98b03`, `c8db89d`, `b4c25dc`, `86cb24d`, `02a0243`, `2bf1798`, `5bb3e4e`, `bda7c43`, `04e5e69` |
| 8 | 08-29 | **R1**, surviving a dead lease: the mid-cell checkpoint, resume, the fingerprint, and the entropy read. Written while three shards were training; committed nine days later | in `3bc103c` |
| 9 | 08-29 | **R2**, exploration: the diagnosis of the dead move term, the four candidate fixes, `exploration_plan`, the move-term pre-flight gate and the trial that picks between them; two critic passes and a docs pass followed the same day | in `3bc103c` |
| 10 | 08-29/30 | ε-greedy rollout replacement, the cell-A trial, and the trial's held-out measurements — one per arm **plus the reference row** (`mz01_explore_trial.py:615`) — and the entropy trace reader | `3bc103c` (with the notes in `60ae17a`) |
| 11 | 08-30 | the dose-response and its control: the named arms `epsilon_005` and `onpolicy_g6`, added as a **named selection** and kept out of every default so the tenth pass's command line still reproduces (`mz01_explore_trial.py:272, 345`) | `3bc103c` |
| 12 | 09-07/08 | the closed action space (D32): the first token masked to the four bare ids, the move term the restricted log-softmax, mid-arm reads, the INVALID-READ rule; then its critic | `8158368` (PLAN), `3d5c9ac`, `4bd2ec2` |
| 13 | 09-08 | the exit priced at the cap (D33): an unsolved episode's step term charged at `SCORE_STEP × cap`; then its critic (`RESUME_ABSENT_MEANS`) | `4ace68f` (PLAN), `9325832`, `562e1e6` |
| 14 | 09-08 | three builds in one pass: **A+B** per-decision credit (D34) and the targeted ε draw (D35), **C** the navigator's label at a fork (D36); both critics (the eligible share is 0.22, the blindness bar was set on a point mass, D37); merged, 271 tests | `8894ce5` (PLAN), `cb8aa32`, `ff8b1e2`, `01fbccb`, `842ee4f`, `3df3d3d`, `cf781fd` |

Passes 8–11 were written on 2026-08-29 and 08-30 and sat uncommitted while the
boxes ran; all four landed on 2026-09-07 in `3bc103c` (code, suite 219 green)
and `60ae17a` (the notes and the two lab notes). That is why `AGENT-NOTES.md`
has headings for passes 1–9 and none for the tenth and eleventh: their record is
the commit message, the block comments in `scripts/mz01_explore_trial.py` cited
above, and `record/RESULTS-2026-08-30.md` §1.

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

⟨Added 2026-09-09, evening.⟩ They answered it, and the answer changes the
fourth thing. The recipe was run seven times and avoided once; the one AVOIDS
read did not reproduce at its own seed; and the coverage read then said why —
the warm-started navigator produces a mine-free solution in about one episode
in two thousand (`results/coverage_s0/coverage_s0.json`), so six rollouts per
maze almost never contain a detour for the price to price. **What looked like
a habit that came and went was a sampling accident on an empty support.** The
fourth thing becomes: the optimiser can only re-weight what the policy
samples, and a recipe that never samples the behaviour it prices is not a
test of the price, however carefully the reads are pre-registered. Every one
of the day's other results — the return-to-go arms, the temperature arms, the
seeds — is a consequence of that and not an independent finding. D-21;
`record/DIAGNOSIS-2026-09-09.md`.
