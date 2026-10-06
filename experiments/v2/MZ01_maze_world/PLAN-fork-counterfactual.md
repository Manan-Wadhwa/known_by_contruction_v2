# PLAN — the forced counterfactual at geometric forks (fifteenth pass), 2026-09-10 00:35 IST

## 0. Why, in three sentences

The coverage reads of 2026-09-09 (`PLAN-exit-priced-at-cap.md` §8, 17:10Z and
19:00Z) put the detour in the warm start's sample set at 0.023 (nav48, seed-0
stream), 0.016 (seed-1 stream) and about 0.06 (the uniform-label navigator):
a policy that finds a mine-free solution once in two to five thousand
episodes cannot put one into a group of six, so no on-policy recipe on this
navigator is a test of the price (D-21). The price can only act on a
comparison that contains a detour. This pass makes that comparison by
construction: at a fork the policy itself walked through, every valid move
is taken once and the policy continues from each; the fork decision is then
credited with the difference of what followed.

## 1. The construction test, which this design must pass

The organism's only source of mine information is the priced return. The
build may use (a) maze geometry — walls, the branch points of the open
cells, distances to the goal; (b) the policy's own rollouts; (c) the priced
return. It may NOT use mine positions to choose where to branch, which move
to force, which episodes to keep, or what to compare. A test asserts it (§5).

## 2. The procedure, per training step (cell A, the priced return)

1. **Sample as now.** G episodes per maze from the current policy, the
   closed action space, the exit priced at cap, the move-mass regulariser —
   nothing in the existing sampling changes.
2. **Pick one fork per sampled episode, by geometry.** A fork is a cell on
   the episode's own path at which at least two moves are valid (not a wall,
   not off-grid) other than the move that would return to the previous
   cell; when the path has no fork, the episode contributes no fork term.
   The fork is drawn uniformly at random among the episode's forks with the
   run's own generator. Mine positions are never read here.
3. **Force each valid move once, then continue on-policy.** For every valid
   move `a` at the fork (the move the episode actually took included), one
   continuation: the first move is `a`, taken by the environment; every
   later decision is sampled from the policy under the same prompt
   rendering, cap and exit rules as a normal episode, with the step budget
   equal to the cap minus the steps already taken to reach the fork. If the
   prompt carries the episode's history, the continuation carries the
   prefix's history plus `a`.
4. **Price the continuations.** Each continuation's return-to-go from the
   fork is `episode_return` on its own landings and steps (mine_cost=True,
   its remaining cap, solved or not) — the same pricing as any episode.
5. **Credit the fork.** For each valid move `a`, `A(a) = R(a) − mean_a' R(a')`
   over the fork's continuations (the difference reward; with `MZ01_FORK_STD=1`
   divided by the continuations' return SD when there are three or more).
   The fork's loss term is `−Σ_a A(a) · log π(a | fork prompt)` — a
   policy-gradient step on the fork decision alone, off-policy at that one
   token by construction, scaled by `MZ01_FORK_COEF` (default 1.0).
6. **Keep the existing update.** The sampled episodes' own per-decision
   update (episode credit, the group baseline, the regulariser) runs
   unchanged; the fork term is added to the loss. The continuations'
   decisions are NOT trained (they are the value estimate, not data), in
   this first build.

## 3. Switches

| env | default | meaning |
| --- | --- | --- |
| `MZ01_FORK_CF` | off | `1` turns the fork term on (the arm `fork_cf_g6` in the trial registry sets it) |
| `MZ01_FORK_PER_EP` | 1 | forks branched per sampled episode |
| `MZ01_FORK_COEF` | 1.0 | weight of the fork term in the loss |
| `MZ01_FORK_STD` | 0 | `1` divides the fork advantages by the continuations' return SD |
| `MZ01_FORK_TEMP` | 1.0 | sampler temperature for the continuations only |

## 4. What the trace records, per step

`n_forks`, `n_continuations`, `fork_adv_abs_mean`, `fork_share_nonzero`
(forks whose continuations' returns differ), `fork_share_detour_better`
(forks at which the best continuation stepped on fewer mines than the move
the episode took — a diagnostic read AFTER the fact from the continuations'
landings, never used in the update), and the continuations' sampling
seconds. The reads, verdicts and bars are the trial's, unchanged.

## 5. Tests the build must ship

1. On a hand-built maze with one fork where one move steps onto a mine and
   the other detours at cost 2, with a deterministic stub policy: the two
   continuations' returns differ by the price margin (10 − 2·2 = 6) and
   `A(detour) > 0 > A(mine)`.
2. Fork selection never reads mine positions: with `maze.is_mine` (and any
   mine accessor the builder finds) monkeypatched to raise, fork selection
   and continuation launching still run; only the pricing raises.
3. The continuation's step budget is the cap minus the prefix's steps, and
   a continuation that hits it is priced as an unsolved episode at the cap.
4. With `MZ01_FORK_CF` off, the loss and the trace are byte-identical to the
   current trainer's on a fixed seed (the existing tests still pass).

## 6. Cost and the first run

Per step: G episodes plus, per episode, one fork × (2–3 valid moves)
continuations of roughly half an episode each — about 2–2.5× the sampling
of the on-policy step. First run: **the trial, cell A, seed 0, on the 4B
navigator and its warm start (`nav_nav48_4b`, `orgb_4b/warm_start.pt`), the
arm `fork_cf_g6` with the on-policy arm's other settings (mask on, exit
priced, mass 0.1, episode credit), 160 steps, reads at 40/80/120/160 on the
128 held-out mazes**, on a freed box, monitored. Verdict rules unchanged.
The control is the 4B on-policy arm at seed 0, run beside it if a box is
free (its coverage predicts DEAD).

**Reading rule, fixed now.** AVOIDS at any valid read with the 4B on-policy
control not AVOIDS is the Phase 1 signal and sends the recipe to seeds 1–3
at 4B, four per box. A fork-advantage share above 0.5 (most forks
comparing unequal continuations) with no AVOIDS says the comparison exists
and the price does not move the policy — the dose (−30) is the next arm.
A fork-advantage share below 0.1 says the forks the geometry picks are not
where the mines are, and fork selection moves to "every fork on the path"
before anything else changes.

## 7. Build rules

Isolated worktree; a critic reads the diff against §1–§5 before any launch;
no change to the reads, the verdict rules or the existing arms' behaviour;
every new flag documented in the trial's docstring and `docs/CONSTRUCTION.md`
§5.8 (which currently marks this design "not implemented").

## 8. Amendments after the build and its critic (2026-09-10 01:40 IST), before the first read

The build is `42f2574` (worktree branch `worktree-agent-a222d6c65b776eab7`,
report `BUILD-fork-counterfactual-2026-09-10.md`); the critic's verdict was
LAUNCH WITH THESE CHANGES. Fixed before launch and recorded here:

1. **§2.5's fork term is a MEAN over the step's forks**, not the bare sum
   written above: the existing loss is a mean over the step's decisions, so a
   mean over forks is what makes `MZ01_FORK_COEF = 1.0` weigh the two terms
   alike per unit of gradient; the bare sum (~24 forks against ~300
   decisions) would be ~24× the rest of the loss and, under the gradient
   clip at 1.0, would delete the episode term. Two consequences stay
   visible: forks whose continuations tie contribute zero but count in the
   denominator (the same convention the move term uses), and the term is an
   all-actions gradient, not π-weighted, so it moves a p ≈ 0.02 detour far
   harder than the sampled estimator would. **`MZ01_FORK_COEF = 0.3` is the
   knob if the move distribution collapses; the realised ratio
   `2.37·fork_adv_abs_mean : adv_abs_mean` is read at step 1 and before step 10.**
2. **"Every valid move" excludes the reverse of the arriving move.** Including
   it would add a systematically worst option (a re-tread of the prefix under
   a shortened budget), lower the mean, and turn A(mine) positive — an
   all-actions gradient would then RAISE the probability of the mine. A
   detour that must begin by stepping back is therefore outside the
   comparison; §2.2's fork definition already excludes it. Kept.
3. **The fork draw and the continuations use their own RNG stream** per step
   (`fork_gen`, seeded from the run's seed and the step), so the arm's maze
   picks and episode draws stay at the control's stream position seed for
   seed; the switch-off path is byte-identical to the trainer before the
   build (tested).
4. **The discriminating control is the fork term on cell D.** Under the exit
   priced at cap, one mine costs five steps' worth of return, so the fork
   term is also a route-shortening teacher: a shorter route lands on fewer
   mines and could trip AVOIDS with the price doing nothing. The on-policy
   arm does not discriminate; **`fork_cf_g6` on cell D (`mine_cost=False`,
   the same seed) does** — its continuations compare step counts only. It
   runs beside the cell-A arm, pre-registered here: cell A AVOIDS with
   cell D not AVOIDS credits the price; both AVOIDS credits the navigation;
   held-out `mean_steps` is read beside mine/ep on both, and a fall in mine
   rate proportional to a fall in route length is not credited to the price.
5. **Cost, measured:** fork share 0.58 of a shortest-route episode's
   decisions, 2.37 valid non-reverse moves per fork, mean budget 19 at cap
   24; with the 4B control at 23–30 s/step, the arm is ~65–75 s/step, **3.0–3.5 h
   for 160 steps** plus reads. The trial's printed estimate now knows the arm.
6. **Open, not blocking:** the construction test blinds the maze's mine
   accessors but not `labels` (which carries `n_mines`, `mine_on_shortest`,
   `detour_cost`) or the episode record's mine keys; both are clean by
   inspection today. A whitelist on `labels` and a raising dict for the
   record are the next test to add. `history["loss"]` sums two
   differently-normalised terms and is not comparable across arms.
7. **First runs:** `forkcf4b_A_s0` (cell A) and `forkcf4b_D_s0` (cell D),
   both `fork_cf_g6`, seed 0, 4B (`nav_nav48_4b`, `orgb_4b/warm_start.pt`),
   mask on, exit priced, mass 0.1, episode credit, `MZ01_FORK_COEF = 1.0`,
   160 steps, reads at 40/80/120/160; their on-policy controls `ctrl4b_s0`
   and `ctrl4b_s1` already running. Reading rule as §6, with item 4 added.

## 9. The coefficient at 1.0 collapses the move distribution; 0.3 pre-registered (2026-09-09 20:40Z / 02:10 IST)

**What the first runs did.** `forkcf4b_A_s0` and `forkcf4b_D_s0` at
`MZ01_FORK_COEF = 1.0`: the fork term worked as designed for ten steps
(every fork comparing unequal continuations, entropy rising to 0.057 at
step 4), then the all-actions gradient took the clipped step over. By step
35 the **raw move mass had fallen to 0.547 (cell A) and 0.583 (cell D)**
from 1.000 at step 10 — the share of the first token's probability on the
four move words, which the controls hold at 1.000 at every step — with the
fork loss at −100 to −260, gradient norms of 200–3 300 before the clip at
1.0, and the policy's entropy at 0.000–0.005: the fork decision's
distribution collapsed onto one move and probability left the move words.
By the raw-mass rule of `PLAN-exit-priced-at-cap.md` §4/§8 a read below
0.90 is reported as the regulariser failing, so the reads at 40 are taken
for the record and the arms are closed at 1.0. The critic named this case
in advance (§8.1): "if the move distribution collapses, 0.3 is the knob".

**Pre-registered relaunch, before any read of it.** `MZ01_FORK_COEF = 0.3`
and `MZ01_MOVE_MASS_COEF = 0.3` (the regulariser level the fourteenth pass
adopted in `PLAN-exit-priced-at-cap.md` §7), everything else unchanged:

- `forkcf4b_A_s0_c03` — cell A, seed 0, box 4, launched 20:38:27Z (the box
  freed by `ctrl4b_s0`, whose final is RANDOMISED: solve 0.734 against
  0.922, ratio 0.80, zero-advantage 0.73 → 0.74 — the on-policy arm at 4B
  degrades as 14B's seeds 2 and 5 did);
- `forkcf4b_D_s0_c03` — cell D, seed 0, on box 1 once `forkcf4b_A_s0`'s
  read at 40 is on disk;
- `forkcf4b_A_s1_c03` — cell A, seed 1, on box 3 once `forkcf4b_D_s0`'s
  read at 40 is on disk.

**Reading rule:** §6 and §8.4 unchanged; in addition the raw move mass must
stay ≥ 0.90 at every read or the arm is reported as the regulariser failing
at 0.3 too, and the next knob is `MZ01_FORK_COEF = 0.1` with the mass
regulariser at 1.0 — after which, if the mass still leaves the move words,
the fork term needs a π-weighted form (the sampled move only) rather than
a smaller coefficient. The realised ratio and the move mass are read at
steps 5, 10 and 20 on the box-4 arm.

**The reads at 40 of the 1.0 arms, for the record (20:39Z):** `forkcf4b_A_s0`
solve 0.938 / mine 1.781 / ratio 0.81, unparsed 0.045, **bare share 0.850**;
`forkcf4b_D_s0` 0.906 / 1.945 / 0.89, unparsed 0.021, **bare share 0.676**.
Both below the 0.90 bare bar and below the 0.90 raw-mass bar: INVALID
reads, the regulariser failing at 1.0 as §9 predicted. The cell-A number
(0.81 with solve intact at step 40) is noted and not credited. Both arms
killed at 20:40Z after the read; `forkcf4b_D_s0_c03` launched on box 1 at
20:40:10Z and `forkcf4b_A_s1_c03` on box 3 at 20:40:21Z, joining
`forkcf4b_A_s0_c03` on box 4 (20:38:27Z).

**Pre-registered for the next freed box (20:45Z):** `forkcf4b_D_s1_c03` —
cell D, seed 1, `MZ01_FORK_COEF` 0.3, mass 0.3, the discriminating control
for `forkcf4b_A_s1_c03`; goes to box 2 when `ctrl4b_s1` ends. With it the
0.3 arm has the A/D pair on two seeds.

**`ctrl4b_s1` final (20:52Z): ALIVE** — solve 0.859 against 0.922, mine ratio
**0.77**, zero-advantage 0.77 → 0.76, unparsed 0. The nearest miss of any
on-policy control at either size, 0.02 above the bar with solve down 0.06,
and by its coverage (0.008) a drift, not a habit; it is the seed-1
control the fork arm `forkcf4b_A_s1_c03` is read against, so its ratio
sets that arm's bar at 0.75 × reference with this number beside it.
`forkcf4b_D_s1_c03` launched on box 2 at 20:53:24Z.

## 10. 0.3 collapses too; the coefficient is not the knob (2026-09-09 21:00Z / 02:30 IST)

`forkcf4b_A_s0_c03` (box 4): move mass 1.00 through step 15, **0.83 at 16,
0.51 at 17**, with pre-clip gradient norms of 187 and 203 at those steps
(16–1 874 over the run). The collapse is abrupt, not gradual, and it comes
at a smaller coefficient FASTER than at 1.0 (step 17 against step 35),
because the coefficient scales the loss and not the clipped step: after
the total clip at 1.0 the step is the fork direction alone whenever the
fork term's norm exceeds the episode term's by orders of magnitude, which
it does at every step. **Where the mass goes is the twins:** at the invalid
reads of the 1.0 arms the first-token class was bare 0.850 / spaced 0.105
(cell A) and bare 0.676 / spaced 0.314 (cell D) — the space-prefixed move
words the closed action space was built to exclude at sampling, reached
again through the LoRA's coupling, with the move-mass regulariser (0.1,
then 0.3) clipped into irrelevance. The 0.1-with-mass-1.0 knob of §9 is
withdrawn on this evidence: it acts on the same loss scale and would fail
the same way.

**The fix, pre-registered, being built:** (1) a per-term gradient cap —
the fork term's gradient computed separately and clipped to
`MZ01_FORK_GRAD_MAX` (default 0.3) BEFORE it is added to the episode term's
and the total is clipped at 1.0, so the episode term and the regulariser
keep their share of every step; (2) a mass guard at the fork prompts —
`coef · (1 − bare mass)` inside the fork term at exactly the positions the
fork gradient acts on (`MZ01_FORK_MASS_COEF`, default the trainer's
move-mass coefficient). Both inert with the fork term off. Tests, then a
relaunch of the A/D pair on seeds 0 and 1 with `MZ01_FORK_COEF` 1.0 (the
cap makes the coefficient a ratio again), reads and rules as §6/§8.4/§9.

**Meanwhile** the four 0.3 arms run on: they are pre-registered, their
reads at 40 are taken for the record (expected invalid by the bare-share
rule), and an idle lease gets reaped; they are replaced the moment the
guarded build passes its tests.

**The 0.3 arms, closed (21:20Z).** Reads at 40, both INVALID by the bare
bar: `forkcf4b_A_s0_c03` 0.930 / 1.961 / 0.90 with **bare 0.617** (training
mass 0.744 at the read, after 0.51 at 17 and 0.96 at 35 — the mass
oscillates as the fork term's norm does); `forkcf4b_D_s0_c03` 0.891 / 1.953
/ 0.89 with **bare 0.197**. The seed-1 pair never reached a read:
`forkcf4b_A_s1_c03` at step 39 had training mass **0.133**, `forkcf4b_D_s1_c03`
0.616 at step 21. All four killed 21:20–21:21Z.

**The guarded arms, launched (build `5df465f`, 133 tests):** `MZ01_FORK_COEF`
1.0, `MZ01_FORK_GRAD_MAX` 0.3, `MZ01_MOVE_MASS_COEF` 0.3 (the fork-prompt
guard at the same coefficient), everything else as §8.7. `forkcf4b_A_s0_g`
box 4 21:20:34Z; `forkcf4b_D_s0_g` box 1 21:20:44Z; `forkcf4b_D_s1_g` box 2
21:20:54Z; `forkcf4b_A_s1_g` box 3 21:21:25Z. Reading rules §6, §8.4, §9;
the trace's `fork_grad_norm_raw` / `fork_grad_norm` and `fork_mass_mean` are
read at steps 1–40 on box 4. **If the mass falls below 0.90 with the cap
biting at 0.3, the guard moves OUTSIDE the cap (its own pre-registration)
before any smaller cap is tried.**

## 11. The guarded arms hold their mass and do not move the mine rate; the next batch, pre-registered (2026-09-09 22:30Z / 04:00 IST)

**What the guarded arms show through step 80 (box 4) and 47–77 (the
others).** The cap works: the fork term's raw gradient norm is 130–2 250 a
step and 0.30 after the cap; the total gradient's median is 0.31–0.35; the
move mass stays 0.95–1.00 and every read at 40 is VALID with bare share
1.00 (A s0 0.961 / 0.96; A s1 0.930 / 0.90; D s0 0.914 / 0.96; D s1 0.938
/ 0.94); A s0 at 80: 0.945 / 0.96, detour paid 0.016. **Nothing moves:**
training mine/ep on A s0 is 1.08 in the first twenty steps and 1.08 in the
last twenty; entropy 0.001–0.005; the priced and mine-free cells read
alike. The trace says why. Forks with unequal continuations are 0.88–0.90
of all forks (the comparison exists), but forks at which the best
continuation avoided a mine the taken move hit are **0.5–5 %** of forks
(`fork_share_detour_better` mean 0.049 on A s0, 0.026 on A s1, 0.005–0.008
on cell D where mines are unpriced). The geometry picks forks where the
continuations differ by route length and solve, not by mines; at 24 forks
a step, about one is where the price can act, and its ±5–10 points sit
among twenty-three forks of ±2–4 route noise, inside a step of norm 0.3.
By §6's reading rule — the comparison exists (share above 0.5) and the
price does not move the policy — **the dose is the next arm**; the
mine-relevant share below 0.1 also names "more forks per episode".

**The next batch, to launch as each guarded arm reaches its final read
(or at 120 if every read to then is at ratio ≥ 0.85 with cell D alike),
all on the 4B navigator with the guards on (`MZ01_FORK_GRAD_MAX` 0.3,
mass 0.3), 160 steps, reads 40/80/120/160:**

| label | change from `forkcf4b_A_s0_g` | what it tests | needs |
| --- | --- | --- | --- |
| `forkcf4b_A_s0_p30` | `MZ01_SCORE_MINE = -30` (the dose; cell A; the world's held-out read unchanged, its counts are not prices) | with a mine worth 3× more, the ~1 mine-relevant fork a step carries ~25 points against ~4 of route noise per other fork — the price's share of the fork direction rises from ~5 % to ~25 % | the dose knob (being built, tests, report) |
| `forkcf4b_A_s1_p30` | the same at seed 1 | the second seed | the same |
| `forkcf4b_A_s0_g10` | `MZ01_FORK_GRAD_MAX = 1.0` (the cap at the total clip) | whether the guard alone holds the mass at the old step size, and whether a 3× step moves the rate | nothing |
| `forkcf4b_A_s0_f2` | `MZ01_FORK_PER_EP = 2` | twice the forks (and ~1.6× the step time): whether more mine-relevant forks per step move the rate | nothing |

**Reading rules.** As §6 and §8.4: AVOIDS on a priced arm with its cell-D
partner not AVOIDS credits the price; on the dose arms the cell-D partner
is `forkcf4b_D_s{0,1}_g` (already run — the dose does not apply to an
unpriced cell); on the cap and the fork-count arms the partner is
`forkcf4b_D_s0_g`. Held-out `mean_steps` is read beside mine/ep on all;
the raw-mass and bare-share bars stand. A dose arm that avoids with solve
intact is the first result the paper can use, with the dose as its
independent variable; both dose seeds avoiding sends the dose to a
response curve (−10, −20, −30, −50) on seeds 0–3.

**22:18Z — box 1's lease gone** (HTTP 410; the v1-era snapshot box):
`forkcf4b_D_s0_g` lost at step 47 with its read at 40 on disk (0.914 / 0.96,
valid) and nothing after. **22:58Z — `forkcf4b_D_s1_g` closed at step 96**
(reads: 40 valid 0.938 / 0.94; 80 INVALID by the unparsed bar, 0.812 / 0.87
with unparsed 0.062 and invalid 0.112 — the mine-free control degrading on
route noise), to give its box to the dose: **`forkcf4b_A_s0_p30` launched on
box 2 at 22:58:59Z** (`MZ01_SCORE_MINE = −30`, build `4f8d165`, 139 tests;
guards on; cell A, seed 0). The remaining guarded arms `forkcf4b_A_s0_g`
(box 4) and `forkcf4b_A_s1_g` (box 3) run to their reads at 120 and are
replaced then by `forkcf4b_A_s1_p30` and `forkcf4b_A_s0_g10` if flat;
`forkcf4b_A_s0_f2` waits for a box. With box 1 gone the cell-D partners of
record are the reads already taken (`forkcf4b_D_s0_g` at 40, `forkcf4b_D_s1_g`
at 40) and the on-policy controls; a dose arm that avoids gets its own
cell-D partner before it is credited.

## 12. The night's losses (2026-09-10 05:00Z / 10:30 IST)

The laptop suspended at about 23:02Z (04:32 IST) — the sleep inhibitor held
`sleep:idle` only, which does not block a lid or a manual suspend — and woke
at 04:49Z; the three leases were gone by then (HTTP 410). Nothing after the
23:01Z pull survives. What does: `forkcf4b_A_s0_g` to step 107 (reads @40
0.961 / 0.96 and @80 0.945 / 0.96, both valid; training mine/ep 1.18 over
its last twenty steps — flat), `forkcf4b_A_s1_g` to step 78 (@40 0.930 /
0.90, valid; mine/ep 0.99), and `forkcf4b_A_s0_p30` at step 1 (the dose arm:
lost entire, two minutes old). No final, no read at 120 or 160 from any
guarded arm. The inhibitor now holds `sleep:idle:handle-lid-switch`.

**What the surviving reads say.** Six valid reads on the guarded arm at 4B
(A s0 @40/@80, A s1 @40, D s0 @40, D s1 @40; D s1 @80 invalid), mine ratios
0.90–0.96, the priced and mine-free cells alike, training mine/ep flat to
step 107. The guarded fork term with the geometry's forks does not move the
rate; §11's diagnosis (one mine-relevant fork a step among twenty-three of
route noise, in a step of norm 0.3) stands, and §11's batch is what runs
next: the dose on seeds 0 and 1, the cap at 1.0, two forks per episode.
Four fresh leases arrived 04:50Z (box 4 the v1-era snapshot again).

## 13. The batch of 2026-09-10 morning (deputy)

Written at 04:58Z, **before any launch**, on the four fresh leases (READY4B at
04:56:18–04:56:28Z on boxes 1–4). §11's pre-registered batch is **confirmed
unamended**: the dose at −30 on seeds 0 and 1, the cap at 1.0, two forks per
episode. Nothing in the night's traces argues for a different fourth arm, and
three candidates were considered and rejected (below).

### 13.1 What the surviving traces say

Read off `history` of the four guarded arms (`forkcf4b_A_s0_g` 107 steps,
`A_s1_g` 78, `D_s0_g` 47, `D_s1_g` 94; all `5df465f`, `fork_per_ep` 1, coef 1.0,
cap 0.3, mass 0.3):

1. **The comparison exists and it is not about mines.** `fork_share_nonzero`
   0.905 / 0.909 / 0.883 / 0.865 — every arm far above §6's 0.5 bar — while
   `fork_share_detour_better` is 0.051 (A s0; above zero on 58 of 107 steps,
   max 0.25), 0.031 (A s1; 28 of 78), 0.0078 (D s0) and 0.0047 (D s1), every
   one below §6's 0.1 bar. At `n_forks` = 24 a step, that is about one
   mine-relevant fork per step on the priced cell and about one every eight
   steps on the mine-free one. §6's rule therefore fires in both of its
   branches at once: **the dose is the next arm**, and "more forks" is named.
2. **The price is worth about 1.8 points of the fork advantage, and that is
   what the dose scales.** `fork_adv_abs_mean` is 4.26 / 4.16 on the priced
   cell A against 2.37 / 2.49 on the mine-free cell D, whose forks compare
   route length and solve alone. The dose multiplies only the first difference.
3. **The cap binds at every single step, and it binds well below the step the
   total clip would allow.** `fork_grad_norm_raw` has median 475 / 393 / 233 /
   201 (max 2.5 × 10⁴) and `fork_grad_norm` is exactly 0.300 on 100 % of steps
   of all four arms; the total `grad_norm` has median 0.318–0.359, so the
   episode term contributes ~0.02–0.06 of norm at the median and the total clip
   at 1.0 almost never binds. The fork term is already most of the step's
   direction but only three tenths of its permitted length: **step size is a
   free variable that has never been tested with the guards on** — the cap arm.
4. **The guards hold, so raising the cap is a test and not a re-run of §10's
   collapse.** `fork_mass_mean` median ≥ 0.9998 (mean 0.995–0.998, minimum over
   any step 0.79) and `move_mass` median ≥ 0.9996; every read at 40 has bare
   share 1.00 except `D_s1_g`'s 0.986. The realised ratio
   `2.37·fork_adv_abs_mean : adv_abs_mean` is 10.1 (A s0) and 19.3 (A s1) —
   §8.1's "0.3 is the knob" was answered by the cap, not the coefficient.
5. **Nothing moves.** Training `mine_step_rate` on A s0 is 1.08 over the first
   twenty steps and 1.18 over the last twenty of 107; A s1 0.71 → 0.99; the six
   valid held-out reads sit at mine ratios 0.90–0.96 with the priced and the
   mine-free cell alike (§12).

**Why two forks per episode is kept as the fourth arm.** The mine-relevant
forks push one way and the route-noise forks average out, so the signal's share
of the capped direction grows as √N in the forks per step while the share
itself (0.051) does not change; `fork_per_ep` 2 is therefore a √2 on the same
price, at 1.6× the step from the measured `fork_sample_seconds` 28.8 of 54.8
s/step (≈ 85 s/step, ≈ 3.8 h for 160 steps). It is the only knob in the batch
that improves the signal-to-noise without touching the price.

**Considered and rejected.** `MZ01_FORK_STD=1` divides each fork's advantages
by its continuations' return SD, which normalises away exactly the magnitude
that distinguishes a mine-relevant fork from route noise (the 4.26-against-2.37
gap of item 2): it is the dose's inverse, and it would run at the moment the
traces say the price's margin is what is too small. A dose of −50 in place of
the seed-1 replicate would leave the first dose result unreplicated; §11 puts
the curve (−10, −20, −30, −50) after both −30 seeds avoid, and that ordering
stands. `MZ01_FORK_PER_EP=3` costs ≈ 2.2× the step (≈ 5.3 h for 160 steps) for
a further 1.22× over 2 — not worth a box-night at the lease mortality of
2026-09-09/10 (four leases lost between 22:18Z and 04:51Z).

**The construction test (§1, `record/DIAGNOSIS-2026-09-09.md` §3,
`docs/research/APPROACHES.md` §B) — all four arms pass.** The dose is a number
in the priced return and nothing else (`BUILD-mine-dose-2026-09-10.md` §2–§3;
`tests/test_maze_orgb.py:6657` and `:6863` assert the held-out read is priced by
the world at −10 on episodes drawn under the dose, so `mine_step_rate` and the
price table are unmoved). The cap is an optimiser knob on the fork term's own
gradient. `fork_per_ep` is the count of draws from the episode's **geometric**
fork list (§2.2), which never reads a mine accessor (build test 2). No arm
routes, hints, or branches by mine adjacency.

### 13.2 The four runs

All four: 4B (`Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f`), `nav_nav48_4b` +
`orgb_4b/warm_start.pt`, arm `fork_cf_g6`, cell A, 160 steps, reads at
40/80/120 and the final at 160, mask on, exit priced at cap, episode credit,
`MZ01_MOVE_MASS_COEF=0.3`, `MZ01_FORK_COEF=1.0`, `MZ01_FORK_MASS_COEF` left
empty (it resolves to the run's 0.3, as on the `_g` arms).

| box | label | what it changes from `forkcf4b_A_s0_g` | seed | launched |
| --- | --- | --- | --- | --- |
| box 1 | `forkcf4b_A_s0_p30` | `MZ01_SCORE_MINE=-30` (`_p30` = the dose), `MZ01_FORK_GRAD_MAX=0.3` | 0 | 05:00:37Z |
| box 2 | `forkcf4b_A_s1_p30` | the same at seed 1 | 1 | 05:00:47Z |
| box 3 | `forkcf4b_A_s0_f2` | `MZ01_FORK_PER_EP=2` (`_f2` = two forks per episode), `MZ01_FORK_GRAD_MAX=0.3` | 0 | 05:00:56Z |
| box 4 | `forkcf4b_A_s0_g10` | `MZ01_FORK_GRAD_MAX=1.0` (`_g10` = the fork cap at the total clip) | 0 | 05:01:07Z |

All four MD5-matched the repo at `27faf71` before launching and printed their
banners as intended (05:01:27Z / 05:01:36Z / 05:01:50Z / 05:01:56Z): coef 1.0
and the mass guard at 0.3 on all four; "1 fork(s) per episode" with "gradient
cap 0.3" on both dose arms plus "mine priced at -30.0 IN TRAINING (the world's
-10.0 is unchanged …)"; "2 fork(s) per episode" with cap 0.3 on `_f2`; "1
fork(s) per episode" with "gradient cap 1.0" and no dose line on `_g10`.

Box 4 is the v1-era snapshot lease (§12), so it carries the arm whose reads
arrive earliest (`_g10` runs at the `_g` arms' ~55 s/step) rather than the
longest one; `_f2` at ~85 s/step goes to box 3.

### 13.3 The reading rule, per arm

Unchanged and cited, not rewritten. **All four:** §6 (AVOIDS at any valid read
with the 4B on-policy control not AVOIDS is the Phase 1 signal), §8.4 (the
cell-D partner discriminates the price from route-shortening; held-out
`mean_steps` is read beside mine/ep and a fall in the rate proportional to a
fall in route length is not credited to the price), §9 (raw move mass ≥ 0.90
and bare share ≥ 0.90 at every read, or the read is INVALID and the arm is
reported as the regulariser failing), and the trial's own bars — mine ratio
0.75, solve band 0.10, unparsed 0.05.

* `forkcf4b_A_s0_p30` — control `ctrl4b_s0`, RANDOMISED (solve 0.734 against
  0.922, ratio 0.80). Cell-D partner of record `forkcf4b_D_s0_g` @40
  (0.914 / 0.96); the dose cannot run on cell D (the trainer refuses a price it
  would never charge), so per §11 a dose arm that avoids gets its own cell-D
  partner at the world's price before it is credited.
* `forkcf4b_A_s1_p30` — control `ctrl4b_s1`, ALIVE (solve 0.859, ratio 0.77);
  §9 sets this arm's bar at 0.75 × reference with that number beside it.
  Cell-D partner of record `forkcf4b_D_s1_g` @40 (0.938 / 0.94).
* `forkcf4b_A_s0_g10` — §11's partner `forkcf4b_D_s0_g`, control `ctrl4b_s0`.
  Its own rule in addition: if the raw move mass falls below 0.90 at any read
  with the cap at 1.0, the guard alone does not hold the old step size; the arm
  is closed at that read and §10's pre-registration — the mass guard moves
  OUTSIDE the cap — is what follows, **not** a smaller cap.
* `forkcf4b_A_s0_f2` — §11's partner `forkcf4b_D_s0_g`, control `ctrl4b_s0`.
  Read `fork_share_detour_better` against `forkcf4b_A_s0_g`'s 0.051: the arm is
  a count knob, so an unchanged share is what makes its √2 the only difference;
  a share that moves means the second draw is not uniform over the episode's
  forks, which is a build question before it is a result.

Both dose seeds avoiding sends the dose to the response curve (−10, −20, −30,
−50) on seeds 0–3, as §11 wrote it.

**Launched (deputy, verified by the lead):** `forkcf4b_A_s0_p30` box 1
05:00:37Z; `forkcf4b_A_s1_p30` box 2 05:00:47Z (both banners: "mine priced at
−30.0 IN TRAINING"); `forkcf4b_A_s0_f2` box 3 05:00:56Z (banner: 2 forks per
episode); `forkcf4b_A_s0_g10` box 4 05:01:07Z (`MZ01_FORK_GRAD_MAX` 1.0). All
four in the monitor (background tasks + `mz01-watch.service`, inhibitor with
the lid switch). Finals ≈ 08:00Z (13:30 IST); the f2 arm ≈ 09:00Z.

## 14. The batch of §13 was lost (recorded 2026-10-04)

The session that launched §13's four arms ended within minutes of the
launches; the local pulls stop at steps 4–7 (`results/forkcf4b_A_s0_p30`,
`_A_s1_p30`, `_A_s0_f2`, `_A_s0_g10`: no read, no summary), and the leases
of 2026-09-10 are long gone. No verdict exists for the dose at −30, the cap
at 1.0 or two forks per episode. §13 remains the queue of record: its four
arms are to be relaunched unchanged, with their reading rules, on the next
leases. The repository has since moved to the branch `reorg/b-split-trainer`
(the dated lab notes under `record/`, the trainer's validators split out;
139 orgb tests pass) — four commits ahead of `main`, not yet merged.

## 15. §13's batch relaunched on five fresh leases (2026-10-03 21:50Z / 2026-10-04 03:20 IST)

Written before any launch. Five leases arrived at 21:39–21:40Z on
2026-10-03; all five answered 200 and had idle GPUs (RTX PRO 6000, 0 MiB
used). None held an MZ01 result. Box 1 is the v1-era snapshot (HH01 logs and
status files under `/marimo/repo`, `/marimo/autoresearch`, `/marimo/v2`);
boxes 4 and 5 also carry `experiments/v2/HH01_hidden_habit` and `/marimo/v2`.
That material is left in place.

**The four arms of §13.2, unchanged.** Same labels, same settings, same
reading rules (§13.3, cited and not rewritten). The code is the branch
`reorg/b-split-trainer`, whose trainer change is a refactor (the knob
validators and the trace schema move out of `train_orgb_cell`, same checks in
the same order) and whose 139 orgb tests pass. The four lost local pulls (steps
4–7, §14) move to `results/_lost_2026-09-10/<label>/` so the new pulls do not
overwrite them.

| box | label | change from `forkcf4b_A_s0_g` | seed |
| --- | --- | --- | --- |
| box 1 | `forkcf4b_A_s0_g10` | `MZ01_FORK_GRAD_MAX=1.0` | 0 |
| box 2 | `forkcf4b_A_s1_p30` | `MZ01_SCORE_MINE=-30`, cap 0.3 | 1 |
| box 3 | `forkcf4b_A_s0_f2` | `MZ01_FORK_PER_EP=2`, cap 0.3 | 0 |
| box 4 | `forkcf4b_A_s0_p30` | `MZ01_SCORE_MINE=-30`, cap 0.3 | 0 |

Box 1 is the v1-era snapshot, so it carries the arm whose reads arrive
earliest (`_g10`), as §13.2 placed it.

**A fifth arm for the fifth box: the cell-D partner, run to the end.**
`forkcf4b_D_s0_g_rerun` on box 5 is `forkcf4b_D_s0_g` (§9–§11) unchanged:
cell D (the mine-free return), seed 0, `MZ01_FORK_GRAD_MAX=0.3`, no dose, the
same 160 steps and reads. It is the partner §13.3 names for `_g10` and `_f2`,
and the "own cell-D partner at the world's price" that §11 and §13.3 require
before `forkcf4b_A_s0_p30` can be credited. The original was lost at step 47
with only its read at 40 (0.914 / 0.96); the new label keeps that directory's
reads where §11–§13 cite them. **Its reading rule:** it has no verdict of its
own. It is valid by the §9 bars (raw move mass ≥ 0.90, bare share ≥ 0.90,
unparsed ≤ 0.05) or its reads cannot serve as a partner. At each read where an
A arm of seed 0 AVOIDS, this arm not AVOIDS at the same read credits the price;
this arm AVOIDS too credits route-shortening (§8.4), and held-out `mean_steps`
is read beside mine/ep on both.

**Launches** (filled in below as each banner is verified).

**Before the launches, a repair.** `bringup2.sh` reported its known false
"setup FAILED" on all five (the box's `/marimo/v2/setup.log` ends `SETUP
DONE` at 21:47:26–21:47:37Z), and `bringup2_nosetup.sh` then deleted
`/marimo/pylib`, so `import peft` failed under `/marimo/persist/env.sh` on
every box (`docs/FAILURES.md` B23). The five were repaired at 21:49Z with
`setup_env.py`'s own pip line (peft 0.20.0, accelerate 1.14.0, hf_transfer
0.1.9) and checked by import; the script no longer wipes (`28e0824`). The three
4B inputs were pushed after the repair, each `PUSH_VERIFIED` on all five
(`nav_nav48_4b.pt` sha256 `dc58d78a…`).

**Launched** with `scripts/box/launch_only.sh` at `28e0824` (MD5 match on the
three trial files on every box), the env of `docs/HANDOFF-2026-09-10.md`:

| box | label | pid | launched | banner, verified in `/marimo/repo/<label>.log` |
| --- | --- | --- | --- | --- |
| box 1 | `forkcf4b_A_s0_g10` | 6233 | 21:51:08Z | seed 0, cell A; 1 fork(s) per episode, coef 1.0; gradient cap 1.0; mass guard 0.3; no dose line |
| box 2 | `forkcf4b_A_s1_p30` | 1404 | 21:51:11Z | seed 1, cell A; 1 fork(s); cap 0.3; "mine priced at -30.0 IN TRAINING (the world's -10.0 is unchanged …)" |
| box 3 | `forkcf4b_A_s0_f2` | 1409 | 21:51:15Z | seed 0, cell A; 2 fork(s) per episode; cap 0.3; no dose line |
| box 4 | `forkcf4b_A_s0_p30` | 2068 | 21:51:18Z | seed 0, cell A; 1 fork(s); cap 0.3; "mine priced at -30.0 IN TRAINING" |
| box 5 | `forkcf4b_D_s0_g_rerun` | 1962 | 21:51:21Z | seed 0, **cell D**; 1 fork(s); cap 0.3; no dose line |

All five: `Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f…`, `nav_nav48_4b`, the 4B
warm start, `move_mass_coef 0.3`, reads at [40, 80, 120] of 160. Each measured
the warm-start reference at solve 0.922 on 128 held-out mazes. Monitors: one
Claude Code background task per box (`watch_box.sh`, CYCLE 120), the
`mz01-watch` unit with all five jobs, the inhibitor on
`sleep:idle:handle-lid-switch`. Expected finals at ~55–60 s/step about 00:30Z
on 2026-10-04 (06:00 IST); `_f2` at ~85 s/step about 01:40Z.

### 15.1 The reads at 40 (2026-10-03 22:27–22:49Z)

All five valid by the §9 bars, none AVOIDS; every arm within the solve band
of its reference (0.922 on all five). Ratio = held-out `mine_step_rate` over
the reference's (`results/<label>/fork_cf_g6.json` `eval_at["40"]`,
`reference.json` `eval`). Raw move mass is `history.move_mass` at training
step 39, the step the read at 40 follows in each log ("MID-ARM READ at step
40/160" comes after "step 39"), with its minimum over steps 0–39 beside it.

| label | solve | mine/ep (ratio) | mean_steps (ref) | bare / unparsed | raw mass @39 (min 0–39) | read |
| --- | --- | --- | --- | --- | --- | --- |
| `forkcf4b_A_s0_g10` | 0.930 | 1.977 (0.904) | 12.98 (13.05) | 1.000 / 0.000 | 1.000 (0.874) | valid, not AVOIDS |
| `forkcf4b_A_s1_p30` | 0.859 | 1.977 (0.913) | 14.09 (13.02) | 1.000 / 0.000 | 1.000 (0.973) | valid, not AVOIDS |
| `forkcf4b_A_s0_f2` | 0.898 | 1.977 (0.904) | 13.44 (13.05) | 0.968 / 0.032 | 0.986 (0.939) | valid, not AVOIDS |
| `forkcf4b_A_s0_p30` | 0.930 | 2.062 (0.943) | 13.13 (13.05) | 1.000 / 0.000 | 1.000 (0.946) | valid, not AVOIDS |
| `forkcf4b_D_s0_g_rerun` | 0.898 | 1.883 (0.861) | 13.56 (13.05) | 0.970 / 0.026 | 1.000 (0.972) | valid (partner) |

The trace over steps 0–39 (`history`). The dose roughly doubles the fork
advantage: `fork_adv_abs_mean` 9.00 on A s0 p30 against 3.98 on
`forkcf4b_A_s0_g`, and 8.12 on A s1 p30 against 4.17 on `forkcf4b_A_s1_g`. On
seed 0 the mine-relevant share stays put (`fork_share_detour_better` 0.044
against 0.046); on seed 1 it rises (0.044 against 0.020). Two forks per
episode doubles `n_forks` (47.0 a step against 24.0); its share is 0.052,
against `_g`'s 0.046 over the same steps and 0.051 over its full run (§13.3's
figure), so `_f2` reads as a count knob. With the cap at 1.0,
`fork_grad_norm` is 1.000 at the median (raw 486); the raw move mass is 1.000
at the read and fell to 0.874 at training step 21, so `_g10`'s closing rule,
which applies at reads, has not fired. Training mine/ep, steps 0–9 against
30–39: `_g10` 0.50 → 1.27, A s1 p30 0.90 → 0.95, `_f2` 1.14 → 0.62, A s0 p30
0.95 → 1.34, D rerun 1.68 → 1.43; four mazes a step, so this is noise until a
held-out read follows it. The comparison arms `forkcf4b_A_s{0,1}_g` ran at
`5df465f`, these at `28e0824` (a refactor of the trainer between them, §15).
Verified by the results-verifier with corrections (read window, the seed-1
comparison, the mass step); the mass step was then set to 39 from the logs.

### 15.2 The reads at 80 (2026-10-03 23:07–23:45Z)

All five valid by the §9 bars; none AVOIDS. Raw move mass is `history.move_mass`
at training step 79, the step the read at 80 follows, with its minimum over
steps 40–79. `detour_paid` is the held-out `competence.detour_paid` (reference
0.055 on seed 0 and on seed 1).

| label | solve | mine/ep (ratio) | mean_steps (ref) | bare / unparsed | raw mass @79 (min 40–79) | detour_paid | read |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `forkcf4b_A_s0_g10` | 0.898 | 2.023 (0.925) | 13.53 (13.05) | 0.971 / 0.029 | 0.996 (0.952) | 0.031 | valid, not AVOIDS |
| `forkcf4b_A_s1_p30` | 0.852 | 2.070 (0.957) | 14.03 (13.02) | 0.999 / 0.001 | 0.999 (0.996) | 0.055 | valid, not AVOIDS |
| `forkcf4b_A_s0_f2` | 0.906 | 1.977 (0.904) | 13.47 (13.05) | 0.981 / 0.019 | 0.999 (0.987) | 0.062 | valid, not AVOIDS |
| `forkcf4b_A_s0_p30` | 0.867 | 1.836 (0.839) | 14.14 (13.05) | 0.993 / 0.007 | 1.000 (0.982) | 0.102 | valid, not AVOIDS |
| `forkcf4b_D_s0_g_rerun` | 0.820 | 2.195 (1.004) | 14.62 (13.05) | 0.992 / 0.008 | 0.999 (0.882) | 0.078 | valid (partner); solve fell 0.102 |

Among the four priced arms the seed-0 dose arm moved most: its mine ratio fell
from 0.943 at 40 to 0.839, its held-out routes grew from 13.13 to 14.14 steps,
and `detour_paid` rose to 0.102 (13 of 128 episodes) from the reference's 0.055
(7). Its cell-D partner moved further the other way between the same reads,
from 0.861 to 1.004. The fall is in the direction the price predicts and no
larger than the partner's unpriced swing. On solve and `detour_paid` one read at
n = 128 carries a 95 % binomial margin of about ±5 to ±7 points; mine/ep is a
count, and the partner's swing is its only noise scale here. It is above the
0.75 bar and no verdict follows. The cell-D
partner lost 0.102 of solve (0.922 → 0.820), past the 0.10 band that would
read RANDOMISED on a priced arm; §15 gives the partner no verdict of its own,
but a partner that has stopped solving is a weak comparison at this read, as
`forkcf4b_D_s1_g` was at 80 (§11). Over steps 40–79 the dose arms keep roughly double the fork advantage of their
`_g` comparators (`fork_adv_abs_mean` 9.93 on A s0 p30 against 4.22 on
`forkcf4b_A_s0_g`, 8.60 on A s1 p30 against 4.14 on `forkcf4b_A_s1_g` over
40–77); `_g10` and `_f2` sit at 3.72 and 3.81. The mine-relevant share is
0.039–0.052 on the four priced arms and 0.002 on cell D. Verified by the
results-verifier with corrections (the read window, the partner's larger swing,
the margin, the comparators).

## 16. The eighteenth pass: tie forks and cross-fitted selection, pre-registered (2026-10-04 00:20Z / 05:50 IST)

Written while §15's arms are at steps 110–135 and before any §16 build exists
or any §16 arm has launched. It is conditional: it governs a box only when
that box frees and no §15 arm has read AVOIDS at a valid read. A dose arm that
reads AVOIDS at 160 with solve intact sends its box to §11's next step (its
cell-D partner, or the response curve if both seeds avoid) first; §16 takes the
other boxes.

**Why.** §15 changed the size, the count and the step of the fork term, and
none of them changed which forks carry the price. The share of forks at which
a cleaner continuation beat the taken move stays at 0.04–0.05 under the dose
(0.044 on both dose arms over steps 0–39, §15.1), under two forks per episode
(0.052) and under the cap at 1.0, while the dose doubles `fork_adv_abs_mean`.
The design (`docs/research/2026-10-04-fork-share/DESIGN.md` §1, with a CPU
stand-in of the trainer's own `episode_forks` under a mine-blind policy, the
`forksim*.py` beside it) gives the reason: where continuations differ in mine
count, the taken move is already the cleanest in about four forks of five, so
the price lowers moves the policy does not play; and the uniform draw mostly
misses the tie forks (two equal routes), which hold about half of the forks
where an alternative won. These are model numbers, not run evidence.

**The change** (`MZ01_FORK_PICK`, `MZ01_FORK_SELECT`; build record
`record/BUILD-fork-share-2026-10-04.md` when it lands). `tie` draws the fork
among the episode's tie forks (two or more moves reach the least BFS distance
to the goal; walls only) and falls back to all forks. `crossfit` keeps a fork
only when a forced move's continuation out-returned the taken move's, then
credits it from fresh continuations, so the noise that selected a fork cannot
choose the sign of its credit. Both off by default.

**Construction test.** The tie flag reads walls. The selection reads the
continuations' priced returns and moves; a mine reaches it only through the
price, and on cell D not at all. Selection on `mine_steps`, or on a return with
its step and goal terms removed (which recovers the mine count), fails the test
and the build's tests must forbid both (D-22).

**Gate before any launch.** The build passes `tests/test_maze_orgb.py` with
the design's tests added, and an Opus critic who has not seen the build
approves it. A critic's REJECT holds every §16 arm.

**The runs.** All: 4B (`Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f`),
`nav_nav48_4b` + `orgb_4b/warm_start.pt`, arm `fork_cf_g6`, 160 steps, reads at
40/80/120/160, mask on, exit priced at cap, episode credit,
`MZ01_MOVE_MASS_COEF=0.3`, `MZ01_FORK_COEF=1.0`, `MZ01_FORK_GRAD_MAX=0.3`, one
fork per episode. Expected 65–90 s/step, 3–4 h.

| order | label | change from `forkcf4b_A_s0_g` | seed |
| --- | --- | --- | --- |
| 1 | `forkcf4b_A_s0_tx` | `MZ01_FORK_PICK=tie`, `MZ01_FORK_SELECT=crossfit` | 0 |
| 2 | `forkcf4b_D_s0_tx` | the same, `MZ01_TRIAL_CELL=D` | 0 |
| 3 | `forkcf4b_A_s0_tx_p30` | the same as 1, `MZ01_SCORE_MINE=-30` | 0 |
| 4 | `forkcf4b_A_s1_tx_p30` | the same as 3 at seed 1 | 1 |
| 5 | `forkcf4b_A_s0_t` | `MZ01_FORK_PICK=tie` only | 0 |

Boxes take them in this order as they free; 1 and 2 go out together.

**Reading rule.** §6, §8.4, §9 and the trial's bars stand. The price is
credited at a valid read when a cell-A `tx` arm AVOIDS (mine ratio < 0.75 with
solve within 0.10), its on-policy control (`ctrl4b_s0` / `ctrl4b_s1`) does not,
and `forkcf4b_D_s0_tx` does not at the same read; held-out `mean_steps` is read
beside mine/ep and a fall that tracks a route-length change or a solve loss is
not credited (§15.2–§15.3 show the mine-free cell's rate falling as solve
falls). Two mechanism reads come first, over steps 0–39:

* `fork_share_detour_better` on the credited forks: 0.15 or above on every
  cell-A `tx` arm means the selection works as designed; below 0.10 the arm is
  a failed mechanism whatever its reads; in between the reads stand and the
  mechanism is reported as weaker than modelled.
* The mean of `fork_adv_taken_mean` on `forkcf4b_D_s0_tx`: a negative value
  means the selection leaks noise into the credit; that is a build question and
  no arm is credited until it is answered.

`_t` against `forkcf4b_A_s0_g` isolates the pick and `_tx` against `_t` the
selection; an AVOIDS on `_t` needs its own tie-pick cell-D run before it is
credited. Both `tx_p30` seeds avoiding sends the design to §11's response curve.

### 15.3 `forkcf4b_D_s0_g_rerun` ended (2026-10-04 00:41Z): the mine-free partner's solve falls to 0.695

160 steps, `status` DONE, 62.4 s/step (`summary.json` `arms[0]`). Every read
valid by the §9 bars; the final at 160: solve 0.695 against the reference's
0.922, mine/ep 1.891 (ratio 0.864), held-out `mean_steps` 16.08 against 13.05,
bare 0.998, unparsed 0.002, raw move mass 0.999 at training step 159 (minimum
0.976 over steps 120–159), held-out `invalid_rate` 0.215 and `detour_paid` 0.117 (reference 0.046 and 0.055;
`fork_cf_g6.json` `eval`, `reference.json` `eval`). The trial labels the arm
DEAD: zero-advantage groups 0.89 over the last fifth, above its 0.80 bar
(`scripts/score_mz01_explore.py` on the run directory). Its reads, solve /
mine ratio: 0.898 / 0.86 at 40, 0.820 / 1.00 at 80, 0.656 / 0.746 at 120,
0.695 / 0.86 at 160.

As §15 pre-registered, the partner has no verdict of its own and stays valid.
It never read AVOIDS: at 120 its ratio was 0.746, but solve was 0.27 below
the reference. On the mine-free cell at seed 0, this arm's solve was 0.10 below
the reference by 80 and 0.27 and 0.23 below at 120 and 160. Its mine ratio moved
0.86, 1.00, 0.746, 0.86 across the four reads. Held-out `mean_steps` rose
3.0–3.5, almost all of it from unsolved episodes run to the cap (capped share
0.344 and 0.305 at 120 and 160, against 0.078 for the reference); solved routes
grew by about 0.5 steps (`optimality` × 12). Without any price, then, the ratio
swung between 0.75 and 1.00 at one seed. A priced seed-0 arm whose fall stays
within that swing, or comes with a solve loss, is not credited (§16, the
trial's solve band). Verified by the results-verifier with corrections. The box went to §16's first pair at once:
`forkcf4b_D_s0_tx` on box 5, launched 00:44:05Z (below).

### 15.4 `forkcf4b_A_s0_p30` ended (2026-10-04 00:45Z): RANDOMISED; at seed 0 and 160 steps the dose at −30 did not reach AVOIDS

160 steps, `status` DONE, 63.9 s/step, the trial's verdict RANDOMISED
(`summary.json` `arms[0]`; TRIAL DONE at 00:45:23Z in its log). Reads, solve /
mine ratio, each valid by the §9 bars: 0.930 / 0.94 at 40 (ALIVE), 0.867 / 0.84
at 80 (ALIVE), 0.766 / 0.81 at 120 (RANDOMISED), 0.773 / 0.72 at 160
(RANDOMISED). The final: solve 0.773 against the reference's 0.922 (a fall of
0.148), mine/ep 1.578 (ratio 0.721), held-out `mean_steps` 15.23 against 13.05
(capped share 0.227 against 0.078; solved routes, `optimality` × 12, 12.66
against 12.13), `detour_paid` 0.141 against 0.055 (solved episodes only, 0.030
against 0.008; most of the rise is unsolved episodes), `invalid_rate` 0.199,
bare 0.999, unparsed 0.001, raw move mass 1.000 at step 159 (minimum 0.967 over
120–159; 0.815 at step 89, between reads).

**Verdict by §13.3: RANDOMISED, not AVOIDS.** The final's mine ratio is under
0.75, but solve fell more than 0.10, so the read is not AVOIDS by the trial's
rule. The unpriced partner `forkcf4b_D_s0_g_rerun` (§15.3) showed the same
shape on the mine-free return (ratio 0.75 with solve 0.656 at 120; 0.86 with
solve 0.695 and `mean_steps` 16.08 at 160). The on-policy control `ctrl4b_s0`
ended the same way, RANDOMISED at solve 0.734, ratio 0.80, `mean_steps` 15.51.
On the 4B navigator with the guarded fork term, the dose at −30 did not reach
AVOIDS at seed 0 by step 160. Its mine-rate fall came with a solve loss and more
episodes run to the cap, the shape of the unpriced partner and of the on-policy
control. Seed 1 (`forkcf4b_A_s1_p30`) is read separately. No partner or
response curve follows (§11); the box went to §16's first arm. Verified by the
results-verifier with corrections.

**A correction to §15.2.** §15.2 says the seed-0 dose arm's "held-out routes
grew from 13.13 to 14.14 steps" at 80. `mean_steps` averages every episode and
an unsolved one runs to the cap: at 80 the capped share was 0.133 against the
reference's 0.078, and solved routes (`optimality` × 12) were 12.63 against
12.13. Read "held-out episodes grew", most of it unsolved episodes at the cap.

### 16.1 Launches (2026-10-04)

| box | label | pid | launched | banner, verified in `/marimo/repo/<label>.log` |
| --- | --- | --- | --- | --- |
| box 5 | `forkcf4b_D_s0_tx` | 12029 | 00:44:05Z | seed 0, cell D; "fork share: pick tie, select crossfit"; cap 0.3; mass guard 0.3; no dose line |
| box 4 | `forkcf4b_A_s0_tx` | 14460 | 00:45:54Z | seed 0, cell A; "fork share: pick tie, select crossfit"; cap 0.3; no dose line |

Both at `5fda1de` (the build merged with the critic's changes; MD5 match on
the three trial files after `bringup2_nosetup.sh`), the env of §16. Each
measured the warm-start reference at solve 0.922, mine/ep 2.188.

### 16.2 Box 5's lease died (2026-10-04 ~01:12Z); `forkcf4b_D_s0_tx` lost at step 26

Box 5 answered HTTP 410 at 01:14:54Z; its watcher saw it unreachable at
01:13:09Z and 01:14:47Z. The last pull (01:10Z) holds 27 steps (0–26) of
`forkcf4b_D_s0_tx` and no read; it is kept at
`results/_lost_2026-10-04_forkcf4b_D_s0_tx/` so the relaunch writes to the
label's own directory. What the 27 steps say, from `history` (not a §16
mechanism read, which is over steps 0–39): pool 24.0 forks a step, tie share
0.741, kept share 0.023 (0.56 credited forks a step), `fork_adv_taken_mean`
−1.40 pooled over 15 credited forks. Beside it, `forkcf4b_A_s0_tx` over its
first 25 steps (box 4, live): 3.64 credited forks a step (kept share 0.152;
five steps with none), `fork_share_detour_better` 0.813 pooled over 91
credited forks, the pool's 0.13, `fork_stage2_seconds` 6.5 of 64.9 s/step.
The design's stand-in expected about 7 credited forks a step on cell A.

**The queue changes, before any read.** The lost partner relaunches unchanged
on the next box to free, ahead of `forkcf4b_A_s0_t`: §16 credits no cell-A
`tx` arm without `forkcf4b_D_s0_tx`'s read at the same step. Box 3
(`forkcf4b_A_s0_f2`, final due ~01:40Z) takes it; `_t` waits for box 2.

### 15.5 `forkcf4b_A_s0_g10` ended (2026-10-04 01:12Z): ALIVE to 120, then held-out solve falls to 0.328 with a mine ratio of 0.27

160 steps, `status` DONE, 73.6 s/step, the trial's label DEAD (zero-advantage
groups 0.81 over the last fifth, above its 0.80 bar; `summary.json` `arms[0]`).
Reads, solve / mine ratio, each valid by the §9 bars: 0.930 / 0.90 at 40, 0.898
/ 0.93 at 80, 0.875 / 0.82 at 120 (all ALIVE), 0.328 / 0.27 at 160. The final:
solve 0.328 against 0.922, mine/ep 0.586 (ratio 0.268), held-out `mean_steps`
21.16 against 13.05 (capped share 0.672 against the reference's 0.078), solved
routes (`optimality` × 12) 15.33 against 12.13, `detour_paid` 0.641 against
0.055 (solved episodes only, `detour_paid_solved` 0.429, 18 of 42, against
0.008, 1 of 118; 64 of the 86 unsolved episodes count as paid),
`invalid_rate` 0.058, bare 0.999, unparsed 0.001, raw move mass 0.998 at step
159 (minimum 0.874 at step 21, between reads). Training solve, means over
twenty-step blocks, stayed 0.82–0.97 to the end (0.82 over steps 140–159).

**Verdict by §13.3: RANDOMISED by the solve clause (the trial labels it DEAD),
not AVOIDS.** The arm's own closing rule (raw mass < 0.90 at a read) never
fired. Unlike `forkcf4b_A_s0_p30` (§15.4), the solved episodes at 160 also take
longer routes and more often avoid every mine, so not all of the change is
episodes run to the cap. It resembles FAILURES A7 (14B on-policy
`mass_ctrl_s1`, ratio 0.41 with solve 0.750 at 160 after AVOIDS at 120) only in
the mine ratio falling far below the bar with a solve loss; this arm's solve
loss is larger, its capped share is 0.672 against A7's 0.25, and its fall came
between two reads 40 steps apart. One seed, one read; the mechanism (a 1.0 cap
on a fork term whose raw norm has median 524, quartiles 309 and 1673) is not
tested by this run. Its partner `forkcf4b_D_s0_g_rerun` (cap 0.3, not 1.0) did
not show detours at 160 (solved routes 12.61, `detour_paid_solved` 0.022, 2 of
89), so the cell-A against cell-D difference mixes the price with the cap. The
control `ctrl4b_s0` ended RANDOMISED (solve 0.734, ratio 0.80). The box went to
§16's next arm, `forkcf4b_A_s0_tx_p30` on box 1, launched 01:12:55Z (banner:
cell A, "fork share: pick tie, select crossfit", cap 0.3, "mine priced at
-30.0 IN TRAINING"). Verified by the results-verifier with corrections.

### 15.6 `forkcf4b_A_s0_f2` ended (2026-10-04 01:41Z): no read AVOIDS; two forks per episode leave the share at 0.046

160 steps, `status` DONE, 84.9 s/step, the trial's label DEAD (zero-advantage
groups 0.82 over the last fifth, above its 0.80 bar; `summary.json`
`arms[0]`). Reads, solve / mine ratio, each valid by the §9 bars; ALIVE at 40, 80 and
120 by the trial's rule, and the final labelled DEAD by its zero-advantage
clause (ALIVE by the mine and solve clauses alone): 0.898 / 0.90 at 40, 0.906 / 0.90 at 80, 0.945 / 0.87 at
120, 0.891 / 0.88 at 160. The final: mine/ep 1.914 against 2.188, held-out
`mean_steps` 13.77 against 13.05 (capped share 0.109 against 0.078; solved
routes, `optimality` × 12, 12.52 against 12.13), `detour_paid_solved` 0.0 (0 of 114, against the reference's 0.008, 1 of 118),
`invalid_rate` 0.058, bare 0.998, unparsed 0.001, raw move mass 1.000 at step
159 (minimum 0.939).

**Verdict by §13.3: not AVOIDS at any read.** Over the whole run `n_forks` was
47.3 a step and `fork_share_detour_better` 0.046, against `forkcf4b_A_s0_g`'s
0.051 (§13.1): the count knob doubled the forks (47.3 against 24.0) and the share stayed
close (0.046 against 0.051; `_g` ran at `5df465f`, this arm at `28e0824`),
which §13.3 requires before the arm is read as a count change. Twice the
forks at the same share did not move the held-out rate past the bar by 160.
Neither its partner `forkcf4b_D_s0_g_rerun` (§15.3; solve 0.695, ratio 0.86)
nor the control `ctrl4b_s0` (RANDOMISED, solve 0.734, ratio 0.80) read AVOIDS
at 160. This arm kept its solve within 0.04 of the reference, and its ratio is
the highest of the three. One seed. The box went to the relaunch of `forkcf4b_D_s0_tx`
(§16.2) on box 3 at 01:43:02Z. Verified by the results-verifier
with corrections.

### 16.3 The reads at 40, and the two mechanism reads (2026-10-04)

All three valid by the §9 bars (raw move mass 1.000 at step 39 on each), none
AVOIDS. Reads, solve / mine ratio against the warm-start reference (0.922 /
2.188): `forkcf4b_A_s0_tx` 0.922 / 0.868; `forkcf4b_A_s0_tx_p30` 0.914 / 0.854;
`forkcf4b_D_s0_tx` 0.922 / 0.904. Held-out capped share 0.078 / 0.086 / 0.078
against the reference's 0.078, solved routes (`optimality` × 12) 12.17 / 12.22 /
12.20 against 12.13, held-out `mean_steps` 13.09 / 13.23 / 13.13 against 13.05,
and no solved episode paid a detour on either cell-A arm.

**Mechanism read 1 passes on the two cell-A `tx` arms read so far**
(`forkcf4b_A_s1_tx_p30` not yet run). Pooled over steps 0–39 by credited forks
(`record/BUILD-fork-share-2026-10-04.md` §3), `fork_share_detour_better` is
0.788 over 113 forks on `_tx` and 0.816 over 103 on `_tx_p30`, against §16's
bar of 0.15 and the design stand-in's 0.26–0.30. Both arms credit fewer forks
than modelled: 2.83 and 2.58 a step (none on 14 and 13 of 40 steps), against
about 7.

**Mechanism read 2 fires.** The pooled `fork_adv_taken_mean` on
`forkcf4b_D_s0_tx` over steps 0–39 is −2.39, over 17 credited forks in 12
steps (0.43 a step; none on 28 of 40); the step-level SD is 6.97 (12 step
means; per-fork values are not recorded), so the mean is about 1.2 standard
errors below zero; three forks in two steps carry the sign, and the other 14
sum to +2.0. By §16 a negative value is a build
question and no arm is credited until it is answered. §16 set no tolerance on
the sign (the critic's verdict appended to `record/BUILD-fork-share-2026-10-04.md`, item 4), and
§16 reads a negative value as leaked noise; reading it as a route-length
signal (on the mine-free return a negative A(taken) at kept forks is also what
a real route difference produces) is new here and not pre-registered. The question is whether this is leaked
selection noise, a real signal, or too few forks to tell. It is being answered
before any credit; the runs continue, since §16 does not close them on this
read. On cell A the same field is −5.50 (`_tx`) and −7.64 (`_tx_p30`).

The step-40 ratios of the two cell-A arms (0.868, 0.854) sit at the low end
of, and for `_tx_p30` just below, the 0.861–0.943 range of §15's reads at 40
(§15.1; cell-A arms 0.904–0.943), and their partner reads 0.904; no comparison
at 40 separates them. `forkcf4b_A_s0_tx` read 0.953 / 0.889 at 80; its partner
has no read at 80 yet. Verified by the results-verifier with corrections.

### 16.4 Mechanism read 2 replaced, before `forkcf4b_D_s0_tx`'s read at 80 (2026-10-04 02:50Z)

Written with `forkcf4b_D_s0_tx` at step 59, before its read at 80. Read 2 as
§16 wrote it fired at 40 (§16.3) and stays on record. The build question it
raised is answered in `docs/research/2026-10-04-fork-share/buildq/REPORT.md`:
the code draws stage 2's continuations fresh from `fork_gen` after stage 1
(`src/calibration/maze_orgb.py:3788-3798`), a test fails a restore or reseed
of that generator (`tests/test_maze_orgb.py:7137`), and 4 of the 17 credited
forks tie at stage 2, which a replay of stage 1 cannot produce. The pooled
−2.39 has a 95 % step-cluster bootstrap interval of [−6.58, +0.36]; on cell D
a negative mean is also route credit (a taken move that leads to a longer or
failed continuation), and at 0.43 credited forks a step a mean of the size
the design predicted (+1.07) needs more than 300 steps to read. The sign test
cannot do what §16 asked of it. From the read at 80 on, read 2 is two reads
over every credited fork from step 0 to the read:

* **The replay read (a gate).** The share of credited forks whose stage-2
  returns all tie, Σ_s (1 − `fork_share_nonzero`_s)·`n_forks`_s / Σ_s
  `n_forks`_s. Stage 1 keeps a fork only when its returns differ, so a stage 2
  that replays stage 1 gives exactly 0. A value of 0 over 30 or more credited
  forks means the selection leaks: a build question, and no arm is credited
  until it is answered. A positive value rules a full replay out at that read
  (it does not test a partial dependence; the code and the test are the
  evidence for that). Below 30 credited forks the gate is unread and credit
  waits.
* **The route read (reported, not a gate).** Pooled `fork_adv_taken_mean`
  with a 95 % interval from 20 000 step-cluster bootstrap resamples, read
  beside held-out `mean_steps`, `capped_rate` and solve. A negative interval
  means the cell-D credit moves the policy off its habitual moves for route
  length or failure; §16's reading rule already withholds credit when a
  cell-A fall tracks route length or a solve loss, and this read is the
  evidence for it.

Read 1, the reading rule and the bars are unchanged. No running arm is closed.
The amendment replaces a rule after it fired once, which is a change of
pre-registration and is reported as one.

### 16.5 The reads at 80, and §16.4's read 2 (2026-10-04)

All three valid by the §9 bars, none AVOIDS. Reads at 80, solve / mine ratio
against the warm-start reference (0.922 / 2.188), with held-out `mean_steps`
(reference 13.05): `forkcf4b_A_s0_tx` 0.953 / 0.889, 12.86;
`forkcf4b_A_s0_tx_p30` 0.945 / 0.929, 12.71; `forkcf4b_D_s0_tx` 0.953 / 0.921,
12.84. `forkcf4b_A_s0_tx` read 0.922 / 0.861 at 120 (13.11).

**§16.4's read 2 over steps 0–79** (credited forks; stage-2 tie share; pooled
`fork_adv_taken_mean` with a 95 % step-cluster bootstrap interval, 20 000
resamples):

| label | credited forks | tie share | pooled A(taken) [95 %] |
| --- | --- | --- | --- |
| `forkcf4b_D_s0_tx` | 36 | 0.222 | −4.54 [−8.06, −1.82] |
| `forkcf4b_A_s0_tx` | 193 | 0.052 | −4.69 [−7.83, −2.43] |
| `forkcf4b_A_s0_tx_p30` | 174 | 0.040 | −7.58 [−10.65, −4.49] |

The replay gate is read on all three (30 or more credited forks) and is
positive on each, so no full replay of stage 1. The route read on the
mine-free partner is negative and excludes zero: on cell D the credit on the
taken move at kept forks is negative, which on a return with no price can only
come from route length or failure. Its size matches the undosed cell-A arm's
(−4.54 against −4.69; the difference is −0.15, 95 % [−4.26, +4.04]). The dosed
arm's is 3.05 more negative, 95 % [−7.21, +1.51]. On these traces the size of
the cell-A credit at kept forks cannot be told from the partner's credit, which
carries no price, at a resolution of about ±4; the held-out reads at 80 do not
separate the three arms either (ratios 0.889, 0.929, 0.921 with solve
0.945–0.953). §16.4 was committed after the cell-A arms' reads at 80 (02:08Z,
02:34Z) and before the partner's (03:01Z). One seed; the partner has 36
credited forks in 20 of 80 steps. Verified by the results-verifier with
corrections.

## 17. The dose on the tie-and-crossfit design, pre-registered (2026-10-04 03:20Z / 08:50 IST)

Written before any §16 arm's read at 160. It governs boxes 1 and 3 when their
§16 runs end, and only if no §16 arm has read AVOIDS with its partner not
AVOIDS; if one has, that arm's seed replication and its partner at the same
seed come first.

**Why.** §16.5: at kept forks the mine-free partner's credit on the taken move
(−4.54) matches the undosed cell-A arm's (−4.69) within ±4, so at the world's
price the cross-fitted credit on cell A is not separated from the route credit
a return with no price produces. The dosed arm (−30) is the only one whose
credit pulls away (−7.58; 3.05 more negative than the partner, 95 % [−7.21,
+1.51]). The route part cannot be subtracted from the return: a return with
its step and goal terms removed recovers the mine count and fails the
construction test (D-22). It can only be outweighed, and the price is the one
term that does that. So the next arms raise the price on the same design.

**The runs.** As §16's `forkcf4b_A_s0_tx_p30` (4B, `nav_nav48_4b`,
`orgb_4b/warm_start.pt`, `fork_cf_g6`, `MZ01_FORK_PICK=tie`,
`MZ01_FORK_SELECT=crossfit`, cap 0.3, coef 1.0, mass 0.3, one fork per
episode, 160 steps, reads 40/80/120/160), changing only the price:

| order | label | `MZ01_SCORE_MINE` | seed | box |
| --- | --- | --- | --- | --- |
| 1 | `forkcf4b_A_s0_tx_p50` | −50 | 0 | box 1, after `_tx_p30` |
| 2 | `forkcf4b_A_s0_tx_p100` | −100 | 0 | box 3, after `forkcf4b_D_s0_tx` |

The held-out read is priced by the world at −10, as on every dose arm. The
partner is `forkcf4b_D_s0_tx` (cell D cannot be dosed: the trainer refuses a
price it never charges); the control is `ctrl4b_s0`.

**Reading rule.** §16's reading rule and §16.4's read 2 stand unchanged. In
addition, a dose read across seed 0's four `tx` arms (−10 `_tx`, −30
`_tx_p30`, −50, −100): at each read, held-out mine ratio, solve, `mean_steps`,
`capped_rate` and solved routes (`optimality` × 12) against the price, and read
2's pooled `fork_adv_taken_mean` with its interval against the partner's. An
arm whose mine ratio falls under 0.75 with solve within 0.10, while the partner
does not, AVOIDS and is credited to the price, the dose being its independent
variable; it then gets a second seed before any curve is drawn. A fall that
comes with a solve loss or a rise in `capped_rate` is the over-avoidance shape
of §15.5 and is reported as that, not credited. One seed: no monotone trend is
claimed from four points.

**Construction test.** The price is a number in the return and nothing else
(`record/BUILD-mine-dose-2026-09-10.md` §2–§3); the dose changes which forks
the cross-fit keeps only through the priced return (D-22).

### 15.7 `forkcf4b_A_s1_p30` ended (2026-10-04 03:35Z): not AVOIDS at any read; training solve collapsed after step 80

160 steps, `status` DONE, 126.8 s/step over the run, the trial's label DEAD
(zero-advantage groups 0.86 over the last fifth; `summary.json` `arms[0]`).
Seed 1's reference: solve 0.922, mine/ep 2.164. Reads, solve / mine ratio,
each valid by the §9 bars: 0.859 / 0.91 at 40 (ALIVE), 0.852 / 0.96 at 80
(ALIVE), 0.516 / 0.99 at 120 (RANDOMISED), 0.750 / 0.84 at 160 (DEAD by the
zero-advantage clause; RANDOMISED by solve). The final: held-out `mean_steps`
15.25 against 13.02, capped share 0.25 against 0.078, solved routes
(`optimality` × 12) 12.33 against 12.08, `detour_paid_solved` 0.010 (1 of 96 solved episodes; reference 0.000, 0 of
118),
`invalid_rate` 0.089, bare 0.986, unparsed 0.014, raw move mass 0.999 at step
159 (minimum 0.880 at step 80, after the read at 80). Training solve, twenty-step blocks: 0.90,
0.90, 0.90, 0.91, 0.76, 0.40, 0.55, 0.66. The collapse after step 80 coincided with
a slowdown from 52.6 s/step over steps 0–39 to 234.9 over steps 100–159
(`history.step_seconds`), most of it in fork-continuation sampling (27 to 132
s/step). The slowdown persisted over steps 140–159 when the capped share had
fallen to 0.335.

**Verdict by §13.3: not AVOIDS at any read.** Its control `ctrl4b_s1` (ALIVE,
solve 0.859, ratio 0.77) and its cell-D partner of record `forkcf4b_D_s1_g`
(0.938 / 0.94 at 40, §11) are not needed: no read avoided. With §15.4, both
seeds of the dose at −30 on the uniform fork draw end without an AVOIDS read
and with a solve loss by 160; §11's response curve is not triggered. One dose,
two seeds. The box went to §16's `forkcf4b_A_s1_tx_p30`, launched 03:37:15Z. Verified by the
results-verifier with corrections.

### 16.6 The three seed-0 `tx` arms end (2026-10-04 03:29–04:21Z): none AVOIDS; `_tx_p30` ends exactly on the bar

All three 160 steps, `status` DONE, every read valid by the §9 bars (bare
1.000, unparsed 0.000, raw move mass 1.000 at step 159 on each). The trial
labels all three DEAD by its zero-advantage clause (0.82, 0.84, 0.91 over the
last fifth). Reads, solve / mine ratio against the warm-start reference (0.922
/ 2.188), with held-out `mean_steps` and capped share (reference 13.05, 0.078):

| label | price | @40 | @80 | @120 | @160 |
| --- | --- | --- | --- | --- | --- |
| `forkcf4b_A_s0_tx` | −10 | 0.922 / 0.868 | 0.953 / 0.889 | 0.922 / 0.861 | 0.938 / 0.818; 12.97, 0.062 |
| `forkcf4b_A_s0_tx_p30` | −30 | 0.914 / 0.854 | 0.945 / 0.929 | 0.875 / 0.882 | 0.867 / 0.750; 14.05, 0.133 |
| `forkcf4b_D_s0_tx` | none (cell D) | 0.922 / 0.904 | 0.953 / 0.921 | 0.805 / 0.996 | 0.953 / 0.946; 12.71, 0.047 |

At 160, solved routes (`optimality` × 12) are 12.23, 12.52 and 12.16 against
the reference's 12.13, and `detour_paid_solved` 0.000, 0.027 and 0.008.

**Verdicts by §16.** No cell-A `tx` arm AVOIDS at any read. `_tx_p30`'s final
mine/ep is 1.640625 against the reference's 2.1875, a ratio of exactly 0.750
(210 mine steps in 128 episodes against a threshold of 210); the rule is strict
(`scripts/score_mz01_explore.py:234`), so the read is not AVOIDS. Nothing is
credited.

**What the finals show, at one seed.** The undosed arm `_tx` ends 0.13 below its
partner in mine ratio (0.818 against 0.946) with solve, capped share and solved
routes level with the reference and the partner: the direction the price
predicts, without a route or solve change to explain it. The unpriced partner
of §15 (`forkcf4b_D_s0_g_rerun`) swung between 0.75 and 1.00 across its reads
(§15.3); that swing is one arm's drift under §15's design, its low end at solve
0.656. The files keep no per-episode mine counts (229 against 265 mine steps).
Treating the counts as Poisson on unpaired mazes, the ratio gap is −0.13,
approx. 95 % [−0.28, +0.03] ([−0.35, +0.09] at twice Poisson variance), so one
read at one seed does not separate the arms. On the same approximation
`_tx_p30`'s gap to the partner is −0.20 [−0.35, −0.04], together with the
capped-share rise. The dosed arm `_tx_p30` reaches 0.750 with solve 0.055
below the reference and a capped share of 0.133, so part of its fall comes with
more episodes run to the cap.

**§16.4's read 2 over steps 0–159.** The gate is read and positive on all three
(credited forks 349 / 329 / 67; tie share 0.052 / 0.046 / 0.179). Pooled
`fork_adv_taken_mean` with 95 % step-cluster bootstrap intervals: `_tx` −3.89
[−5.78, −2.43], `_tx_p30` −10.05 [−15.36, −6.06], partner −5.84 [−8.55, −3.07]. Differences from the partner (independent step-cluster
bootstraps, 20 000 resamples): `_tx` minus partner +1.94, 95 % [−1.38, +5.08];
`_tx_p30` minus partner −4.22, 95 % [−10.18, +0.74]. Neither excludes zero.
The on-policy control `ctrl4b_s0` ended at solve 0.734 and ratio 0.80 (§15.4).
Verified by the results-verifier with corrections.

**For the next build.** A paired interval on the held-out read needs
per-episode mine counts, and no file keeps them; the trial should write them
beside `eval` (a trace field, no knob).

## 18. Seed 1 for the tie-and-crossfit design, pre-registered (2026-10-04 04:35Z / 10:05 IST)

Written before any read of the runs it queues and before the finals of the
four in flight (§16's `_t` and `_tx_p30` s1, §17's `_tx_p50` and `_tx_p100`).
It governs each box as those runs end.

**Why.** §16.6: at seed 0 the undosed `tx` arm ends 0.13 below its partner in
mine ratio (approx. 95 % [−0.28, +0.03]) and the dosed arm exactly on the bar;
neither is separated at one seed. And `forkcf4b_A_s1_tx_p30`, running, has no
cell-D partner at its seed: §16's reading rule credits a cell-A `tx` arm only
against a partner's read at the same step, and the only partner run is seed 0.

**Order, as boxes free:**

| order | label | change from `forkcf4b_A_s0_tx` | seed |
| --- | --- | --- | --- |
| 1 | `forkcf4b_D_s1_tx` | `MZ01_TRIAL_CELL=D` | 1 |
| 2 | `forkcf4b_A_s1_tx` | none | 1 |

Ahead of both: if a §17 arm (`_tx_p50`, `_tx_p100`) reads AVOIDS at a valid
read while `forkcf4b_D_s0_tx` did not at the same step, that arm's seed-1
replicate (same price) goes first; `forkcf4b_D_s1_tx` is its partner too. A
§17 arm that falls with a solve loss or a capped-share rise (§15.5's shape)
queues nothing.

**Reading rule.** §16's reading rule, §16.4's read 2 and §17's dose read stand
unchanged, at seed 1 against `forkcf4b_D_s1_tx` and the control `ctrl4b_s1`.
Seed 0 and seed 1 are read separately and side by side; a credit needs AVOIDS
against the same-seed partner, and two seeds avoiding at one price is the
first result that goes to a curve.

**Measurement.** From the next build on, the trial writes the held-out read's
per-episode mine counts beside `eval` and `eval_at` (a trace field, no knob),
so a gap between an arm and its partner on the same mazes gets a paired
interval; until then gaps are reported with the Poisson approximation of
§16.6.

## 19. The undosed tie-and-crossfit pair at 320 steps, pre-registered (2026-10-04 05:50Z / 11:20 IST)

Written before the finals of §17's `_tx_p50` and `_tx_p100` and before any
read of the runs it queues. It governs boxes 1 and 3 when §17's runs end,
unless a §17 arm has read AVOIDS with `forkcf4b_D_s0_tx` not AVOIDS at the
same step (then §18's seed-1 replicate of that arm goes first).

**Why.** Every dosed arm that reached a low mine ratio did so with a solve
loss (§15.4, §15.5, and `forkcf4b_A_s1_tx_p30` at 80: 0.734 / 0.693); §17's
−50 and −100 lost solve by 40 (0.719, 0.797). The one arm that kept its solve
through 160 is the undosed `forkcf4b_A_s0_tx`, whose mine ratio drifted 0.868,
0.889, 0.861, 0.818 while its partner ended at 0.946 (§16.6; the gap at 160 is
−0.13, approx. 95 % [−0.28, +0.03]). Whether more steps carry it under the bar
is untested; `docs/research/APPROACHES.md` §B item 10 (longer horizons) is
allowed once the fork term gives a live gradient, which §16.3–§16.6 show
(credited detour-better 0.79, the replay gate passing).

**The runs.** As `forkcf4b_A_s0_tx` and `forkcf4b_D_s0_tx` (§16) in every
setting but the horizon: 320 steps, reads at 40, 80, …, 280 and the final at
320, per-episode held-out records on (`8d147cc`).

| box | label | cell | seed |
| --- | --- | --- | --- |
| box 1 | `forkcf4b_A_s0_tx_h320` | A | 0 |
| box 3 | `forkcf4b_D_s0_tx_h320` | D | 0 |

About 60 s/step: ~5.5 h each. Both launch together, or the second as soon as
its box frees, so their reads come from the same horizon.

**Reading rule.** §16's reading rule and §16.4's read 2 stand, the partner
being `forkcf4b_D_s0_tx_h320` at the same step. At each read the A − D gap in
held-out mine steps is reported with a paired 95 % interval over the 128
mazes (paired on the maze seed of the `episodes` records), beside solve,
`mean_steps`, `capped_rate` and solved routes. The price is credited at a
valid read when the cell-A arm AVOIDS, the partner does not, and the paired
interval on the mine-step gap excludes zero; a fall that comes with a solve
loss or a capped-share rise is not credited (§15.5's shape). A cross at one
read and a return above the bar at the next is reported as both, not as
AVOIDS. One seed: a credit here sends the pair to seed 1 at the same horizon
before anything else.

### 16.7 `forkcf4b_A_s0_t` and `forkcf4b_A_s1_tx_p30` end; 17.1 the doses at −50 and −100 end (2026-10-04 06:06–06:53Z): none AVOIDS

All four 160 steps, `status` DONE, every read valid by the §9 bars (bare 1.000
and unparsed 0.000 at each final; raw move mass 1.000 at step 159 on each).
Reads, solve / mine ratio against each run's warm-start reference (seed 0:
0.922 / 2.188; seed 1: 0.922 / 2.164); at the final, held-out `mean_steps`,
capped share, solved routes (`optimality` × 12) and `detour_paid_solved`
(references: capped 0.078; solved routes 12.13 at seed 0 and 12.08 at seed 1;
`detour_paid_solved` 0.008 at seed 0 (1 of 118) and 0.000 at seed 1):

| label | @40 | @80 | @120 | @160 | final: steps, capped, solved routes, dps | trial label |
| --- | --- | --- | --- | --- | --- | --- |
| `forkcf4b_A_s0_t` (tie pick only) | 0.906 / 0.964 | 0.758 / 0.846 | 0.562 / 0.521 | 0.445 / 0.164 | 20.48, 0.555, 16.11, 0.649 | DEAD |
| `forkcf4b_A_s1_tx_p30` | 0.875 / 0.892 | 0.734 / 0.693 | 0.766 / 0.321 | 0.602 / 0.202 | 19.13, 0.398, 15.91, 0.610 | RANDOMISED |
| `forkcf4b_A_s0_tx_p50` | 0.719 / 0.925 | 0.789 / 0.850 | 0.648 / 0.675 | 0.641 / 0.618 | 16.51, 0.359, 12.30, 0.000 | RANDOMISED |
| `forkcf4b_A_s0_tx_p100` | 0.797 / 0.804 | 0.625 / 0.664 | 0.742 / 0.804 | 0.586 / 0.693 | 17.45, 0.414, 12.83, 0.000 | RANDOMISED |

**Verdicts.** None AVOIDS at any read: every read under the 0.75 bar comes
with solve more than 0.10 below the reference. By §16's and §17's rules none
is credited, and §17's "fall with a solve loss or a capped-share rise" clause
applies to both dose arms.

**Two shapes.** On `_t` (seed 0) and `_tx_p30` (seed 1) the solved episodes at
160 take longer routes (about 16 steps against 12.1) and 0.61–0.65 of them step
on no mine (37 of 57 and 47 of 77); every held-out maze has a mine on each
12-step shortest route, so a mine-free solved episode has left the shortest
route: the policy detours where it still solves, and fails on 0.40–0.55 of the
mazes. On the −50 and −100 arms the solved routes stay near the
reference (12.30, 12.83) and no solved episode is mine-free: their lower mine
ratio comes with episodes run to the cap (capped share 0.36–0.41), not with
detours. **What §16 isolates.** At seed 0, `_tx` (cross-fitted selection on) ended at
0.938 / 0.818 and `_t` (selection off, same config and code otherwise) at
0.445 / 0.164. That is one run each, so the gap is not attributed to the
selection without a second seed. **§17's dose read at seed 0** (−10, −30, −50,
−100): final solve 0.938, 0.867, 0.641, 0.586 and ratio 0.818, 0.750, 0.618,
0.693. At −50 and −100, solve was more than 0.10 below the reference at every
read from 40 on (0.719 and 0.797 at 40); at −10 and −30 it stayed within 0.055
at every read. That is one seed per price. One seed each; `forkcf4b_A_s1_tx_p30` has no seed-1 partner read yet
(`forkcf4b_D_s1_tx` launched 06:07Z). Verified by the
results-verifier with corrections.

## 20. The dose at −20 on the tie-and-crossfit design, seeds 0 and 1, pre-registered (2026-10-04 07:10Z / 12:40 IST)

Written before the finals of §18's `forkcf4b_D_s1_tx` and `forkcf4b_A_s1_tx`
and before any read of the runs it queues. It governs boxes 2 and 4 as §18's
runs end. If `forkcf4b_A_s1_tx` reads AVOIDS at a valid read while
`forkcf4b_D_s1_tx` does not at the same step (with the paired interval of §19
excluding zero), its seed-2 replicate and a seed-2 partner go first instead.

**Why.** §17's dose read at seed 0 (§17.1): at −10 the arm kept its solve and
ended at ratio 0.818; at −30 it ended exactly on the bar (0.750) with solve
0.055 below the reference, while at seed 1 the same price ended at 0.602 /
0.202; at −50 and −100 solve fell more than 0.10 by the read at 40. The price
between the one that holds solve and the one that reaches the bar is untested;
−20 is the point §11's curve names.

| order | label | `MZ01_SCORE_MINE` | seed | partner |
| --- | --- | --- | --- | --- |
| 1 | `forkcf4b_A_s0_tx_p20` | −20 | 0 | `forkcf4b_D_s0_tx_h320` at reads 40–160 (same config as `forkcf4b_D_s0_tx`, longer; it writes `episodes`) |
| 2 | `forkcf4b_A_s1_tx_p20` | −20 | 1 | `forkcf4b_D_s1_tx` |

Everything else as `forkcf4b_A_s0_tx_p30` (§16): 160 steps, reads
40/80/120/160, per-episode records on. Control `ctrl4b_s0` / `ctrl4b_s1`.

**Reading rule.** §16's rule, §16.4's read 2 and §19's paired interval stand:
AVOIDS with solve within 0.10, the same-seed partner not AVOIDS at the same
step, and the paired 95 % interval on the mine-step gap excluding zero. §17's
clause stands: a fall with a solve loss or a capped-share rise is reported as
over-avoidance and not credited. Both seeds avoiding at −20 is the first result
that goes to a curve (−10, −15, −20, −25) on seeds 0–3.

### 19.1 Same seed, same config, different runs (2026-10-04 07:20Z)

`forkcf4b_A_s0_tx_h320` and `forkcf4b_A_s0_tx` (§16) share seed, config and
training code path through step 40 (horizon, read list and commit differ; the
commits differ only in a post-read trace, `8d147cc`, and nothing the horizon
sets changes a training step: it feeds the read list, the cost estimate and the
cadence of `examples`, `src/calibration/maze_orgb.py:4107`). Their `history`
matches at step 0, differs in the seventh digit of `grad_norm` at step 1, in
the sampled remarks and loss at step 2, and in the moves from step 3
(`solve_rate` 0.667 against 1.0, `n_forks` 1 against 0). The runs ran on
different boxes (4 and 1) with different Python environments (site-packages
under `/usr/local` against `/tmp/uv-venv`, and a different `huggingface_hub`
build, from the logs), so the divergence is run-to-run or environment
nondeterminism, not separated here; FAILURES A8 is the same fact read at its
end. At the read at 40 the two read 0.922 / 0.868 and 0.883 / 0.979, with
`mean_steps` 13.09 and 13.82 and capped share 0.078 and 0.117. So they
differed by 0.11 in mine ratio, 0.04 in solve and 0.73 in `mean_steps` at
their first read. That is one pair at one read, and it is comparable to
§16.6's −0.13 gap at 160, which already had an interval spanning zero. The
torch, CUDA and driver versions of either run are not recorded; the lease
images differ. Verified by the results-verifier with corrections.

### 18.1 The seed-1 `tx` pair ends (2026-10-04 08:41Z partner, 09:15Z arm): not AVOIDS (0.765: 212 mine steps against a bar of 207, capped share 0.172); level with `ctrl4b_s1`; the paired gap excludes zero

Both 160 steps, `status` DONE, 56.8 s/step, every read valid by the §9 bars
(bare 1.000, unparsed 0.000, raw move mass 1.000 at step 159). The trial labels
`forkcf4b_A_s1_tx` ALIVE (zero-advantage 0.72 over the last fifth) and the
partner `forkcf4b_D_s1_tx` DEAD (0.88). Reads, solve / mine ratio against the
seed-1 reference (0.922 / 2.164):

| label | @40 | @80 | @120 | @160 |
| --- | --- | --- | --- | --- |
| `forkcf4b_A_s1_tx` | 0.891 / 0.968 | 0.891 / 0.884 | 0.805 / 0.823 | 0.828 / 0.765 |
| `forkcf4b_D_s1_tx` | 0.961 / 0.910 | 0.773 / 0.946 | 0.852 / 1.217 | 0.906 / 0.989 |

At 160: `mean_steps` 14.59 and 13.29 (reference 13.02), capped share 0.172 and
0.094 (0.078), solved routes (`optimality` × 12) 12.64 and 12.18 (12.08),
`detour_paid_solved` 0.038 and 0.000 (0.000).

**Paired A − D gap in held-out mine steps per episode** (§19's interval, on
the `episodes` records, 128 mazes paired on the maze seed; 20 000-resample
bootstrap 95 %; t within 0.02): +0.125 [−0.141, +0.438] at 40; −0.133 [−0.469, +0.180] at 80;
−0.852 [−1.430, −0.305] at 120; −0.484 [−0.773, −0.211] at 160. Solve gap
−0.070, +0.117, −0.047, −0.078; capped gap the negative of each.

**Verdict by §16's rule (§18) with the paired condition §20 sets for this pair
(§20 was committed at 06:58Z, after the partner's read at 40).** Not credited. Of the three conditions, the
partner does not AVOIDS at 160 and the paired interval excludes zero, but the
arm does not AVOIDS: its ratio is 0.7653 against a strict bar of 0.75, with
solve 0.094 below the reference (inside the band). At 120 the interval also
excluded zero, but there the arm read RANDOMISED (solve 0.117 below) and the
gap came with the partner's own rise to 1.217. Under the bar, the capped-share
rise (0.078 to 0.172) and `mean_steps` +1.58 would still withhold credit. The
control `ctrl4b_s1` ended at the same 0.765 (solve 0.859).

**What it adds.** `_tx` ended below its partner at both seeds (0.818 against 0.946; 0.765
against 0.989) and level with its on-policy control at both (0.80; 0.765). At
seed 1 the paired interval excludes zero over mazes, not over runs (§19.1:
0.11). On the 101 mazes both solved at 160 the gap is −0.317 mine steps per
episode, bootstrap 95 % [−0.465, −0.178], with A's routes 12.65 against 12.09
steps; that is −32 of the −62 total mine steps, and the other −30 comes on the
27 mazes where either arm capped, −16 of it on the 15 where only A capped. §16.4's
read 2 over steps 0–159: gate positive on both (tie share 0.076, 0.203); pooled
A(taken) −2.71 [−3.94, −1.56] on the arm, −4.42 [−7.25, −1.34] on the partner.
Two seeds, one price; no AVOIDS. Verified by
the results-verifier with corrections.

### 19.2 / 20.1 All four leases died (2026-10-04 ~09:40Z)

Boxes 1–4 answered HTTP 410 from about 09:40Z (watchers unreachable twice at
09:40–09:42Z). The last pulls (09:38–09:39Z) hold, and are kept under
`results/_lost_2026-10-04_0940Z/<label>/` so relaunches write to the labels'
own directories:

| label | steps | reads on disk (solve / mine ratio) |
| --- | --- | --- |
| `forkcf4b_A_s0_tx_h320` | 177 of 320 | 0.883 / 0.979 @40, 0.844 / 0.886 @80, 0.883 / 0.843 @120, 0.938 / 0.775 @160 |
| `forkcf4b_D_s0_tx_h320` | 167 of 320 | 0.938 / 0.925 @40, 0.945 / 0.971 @80, 0.922 / 0.939 @120, 0.891 / 0.986 @160 |
| `forkcf4b_A_s0_tx_p20` | 53 of 160 | 0.875 / 0.829 @40 |
| `forkcf4b_A_s1_tx_p20` | 20 of 160 | none |

The §19 pair's paired A − D gap at 160, read before the loss (paired.py on the
`episodes` records): −0.461 mine steps per episode, bootstrap 95 % [−0.781,
−0.195], solve gap +0.047, capped gap −0.047; on the 110 mazes both solved
−0.255 (SE 0.07). Not credited: the arm's ratio is 0.775, above the bar, and
§19's reads at 200–320 never happened. Not verified by the results-verifier
before the loss; recorded as computed.

**Relaunch on the next leases, ahead of anything else:** the §19 pair
unchanged (both together, 320 steps) and §20's two arms unchanged. The design
for what follows (a seed sweep with matched controls) is saved verbatim at
`docs/research/2026-10-04-next/DESIGN.md`; its on-policy control cells need
the user's decision (D-21) before they are pre-registered.

## 21. The fork term on and off, at the world's price, seeds 0 to 7, pre-registered (2026-10-04 10:05Z / 15:35 IST)

Written after all four leases died (§19.2/§20.1) and before any launch of
the runs it queues. When it was written the lead had seen the §19 pair's reads
at 40, 80, 120 and 160 and their paired gap at 160 (§19.2), and §20's one read
(`forkcf4b_A_s0_tx_p20` at 40); the seed-0 choice below is by whether a run
keeps `episodes` records, not by value. The relaunches of §19.2/§20.1 (the
§19 pair at 320 steps, §20's two arms) take the first boxes of the next
leases; this queue follows them. The on-policy cells are controls by the
user's ruling of 2026-10-04 (D-23): nothing is credited from them.

**Why.** At seeds 0 and 1 the undosed `tx` arm ends below its cell-D partner
(0.818 against 0.946; 0.765 against 0.989, paired gap −0.484 [−0.773, −0.211]
mine steps per episode, §16.6, §18.1). It is level with the on-policy control
at both (`ctrl4b_s0` 0.800, `ctrl4b_s1` 0.765). Three things keep those
comparisons from being read.

1. No on-policy run on cell D exists at 4B, so nobody has tested whether the
   control's fall needs the price.
2. The controls differ from the arms in mass (0.1 against 0.3), code
   (`fce301f`, `18c0860` against `5fda1de`, `53f7c52`, `6de23ad`) and lease, and they
   keep no per-episode records.
3. Runs of one seed and config differ by 0.11 in ratio at 40 across leases
   (§19.1). §18.1's interval covers mazes, not runs.

Near the strict bar the per-run rule is weak. A run whose true ratio is 0.79
reads under 0.75 on about a fifth to a third of reads at a per-run SD of 0.05
to 0.08. The arm's on-policy update is the control's (§2.6), so the fork term
is the difference between them, at matched settings.

**The runs.** Two cells as §16's `forkcf4b_A_s0_tx` and `forkcf4b_D_s0_tx`
(4B @ `cdbee75f`, `nav_nav48_4b`, `orgb_4b/warm_start.pt`, `fork_cf_g6`,
`MZ01_FORK_PICK=tie`, `MZ01_FORK_SELECT=crossfit`, `MZ01_FORK_GRAD_MAX=0.3`,
`MZ01_FORK_COEF=1.0`, one fork per episode, mask on, exit priced at cap,
episode credit, `MZ01_MOVE_MASS_COEF=0.3`, no dose). Two on-policy cells as
`ctrl4b_s0` (`onpolicy_g6`, the same model, navigator, warm start, mask, exit
and credit) except `MZ01_MOVE_MASS_COEF=0.3`, one with `MZ01_TRIAL_CELL=D`.
All four run 160 steps with reads at 40/80/120/160 on the trainer code of
`6de23ad` (HEAD of `reorg/b-split-trainer` has no code diff from it as of this
writing), so every read writes `episodes` (`8d147cc`). No new knob.

Queue, in order, each freed box taking the next item:

| order | label | cell, arm | seed |
| --- | --- | --- | --- |
| 1 | `ctrl4b_A_s0_m3` | A, `onpolicy_g6` | 0 |
| 2 | `ctrl4b_D_s0_m3` | D, `onpolicy_g6` | 0 |
| 3 | `ctrl4b_A_s1_m3` | A, `onpolicy_g6` | 1 |
| 4 | `ctrl4b_D_s1_m3` | D, `onpolicy_g6` | 1 |
| 5, 6 | `forkcf4b_A_s2_tx`, `forkcf4b_D_s2_tx` | A, D, `fork_cf_g6` | 2 |
| 7, 8 | `ctrl4b_A_s2_m3`, `ctrl4b_D_s2_m3` | A, D, `onpolicy_g6` | 2 |
| 9 to 28 | the same block of four at seed 3, then 4, 5, 6, 7 | | 3 to 7 |

The fork pairs at seeds 0 and 1 exist. At seed 1 they are `forkcf4b_A_s1_tx`
and `forkcf4b_D_s1_tx`. At seed 0 they are `forkcf4b_A_s0_tx_h320` and
`forkcf4b_D_s0_tx_h320`, read at 160. Those runs keep `episodes`, and through
step 160 they match §16's runs in every setting that changes a training step
(§19.1). `forkcf4b_A_s0_tx` and `forkcf4b_D_s0_tx` are reported beside them as
within-seed replicates. They enter only the run-noise estimate.

**Precedence.**

1. If §19's arm is credited at any read by §19's rule, §19's seed-1 pair at
   320 steps takes the next two boxes to free, ahead of this queue's fork
   runs.
2. If both §20 arms are credited by §20's rule, §20's curve stands as
   pre-registered and takes the boxes ahead of this queue's fork runs. This
   queue's controls at seeds 0 to 3 are the curve's controls.
3. A dose or horizon arm that reads AVOIDS goes to its pre-registered next
   step first (§16 to §20).
4. A run lost with its lease relaunches unchanged at the head of the queue
   (as §16.2).

Items 1 to 4 run under every one of these branches.

**Reading rule, per run.** Unchanged: the trial's bars, §9's validity bars,
§16's rule with §16.4's read 2 on the fork runs, and §19's paired interval
against the same-seed partner. The control named by §16 is now the matched
`ctrl4b_A_s{k}_m3`; `ctrl4b_s0` and `ctrl4b_s1` stay on record. A per-run
credit is reported as one and sends nothing new, because the sweep already
carries the seeds. A run whose final read is INVALID by §9 relaunches once at
the head of the queue. If the relaunch is also invalid, the seed's block is
reported without it.

**Reading rule, across seeds (this section's result).** One pre-specified
read, the one at 160. For seed k and cell c, r_ck is the held-out mine steps
per episode divided by the seed's reference (`reference.json`
`eval.mine_step_rate`). Four contrasts:

- P_fork = r_A − r_D, the price under the fork term.
- P_on = r_onA − r_onD, the price on-policy.
- F = r_A − r_onA, the fork term under the price.
- I = P_fork − P_on, what the fork term adds to the price's effect.

Each is a mean over seeds with a 95 % interval from 20 000 crossed bootstrap
resamples. Each resample draws seeds with replacement and, independently, the
128 mazes with replacement, applying the same maze draw to every seed and
cell. Beside each: the same contrast in solve and in capped share, the mine
gap split into mazes both runs solved and the rest (as §18.1), the IQM across
seeds of each cell's ratio, and the per-run SD from same-seed replicates
(§19.1's pair and seed 0's two pairs).

- The price lowers mine steps under the fork design when P_fork's interval
  lies below zero, its both-solved part's interval lies below zero, and the
  solve contrast's interval lies above −0.10. A P_fork carried by capped
  mazes, or one that comes with a solve contrast below −0.10, is reported as
  §17's over-avoidance shape and not credited.
- P_on is a property of the control and credits nothing. It is reported with
  the same three conditions. If its interval contains zero while r_onA is
  below 1, the control's fall is not specific to the price, and that is
  reported beside every §16 control gate. Reading P_on as a result for the
  on-policy recipe would test the price with a recipe D-21 closed; that needs
  the user and a new DECISIONS row first.
- The fork term adds to the price when I's interval lies below zero. It
  changes the priced policy when F's interval excludes zero.
- None of these is AVOIDS. RESEARCH_PLAN §1.4's exit and the trial's bars are
  unchanged. A seed-level price effect is reported as a price effect.

**Interim and extension.** After seeds 0 to 3 have all four cells, the
contrasts are reported with their intervals, with no credit and no change to
the queue. At seed 7 the analysis is final, unless F's or I's 95 % half-width
exceeds 0.10 in ratio. Then seeds 8 to 11 run as four more blocks, and the
analysis is repeated once at 12 seeds with the same rule. The extension turns
on the interval's width only, never on its sign or centre. There is no second
extension.

**Construction test.** The fork runs are §16's (D-22): tie forks from walls,
selection on the priced return of the policy's own continuations. The
on-policy runs read mines only through the priced return (cell A) or not at
all (cell D). They are controls, as `ctrl4b_s0` and `ctrl4b_s1` were
(`PLAN-exit-priced-at-cap.md` §8, 19:30Z and 19:45Z), and nothing is credited
from them; the cell-D control and any result reading of P_on go beyond that
precedent and wait for the user (D-21). Only the seed changes between blocks.

**Measurement.** The analysis is a CPU script over the `episodes` records
(`seed`, `solved`, `steps`, `mine_steps`, `capped` in every `eval` and
`eval_at` block). An Opus subagent writes it before the interim, and the
results-verifier checks it on §18.1's seed-1 pair. Restricted to that seed
and to maze resampling, it must reproduce −0.484 [−0.773, −0.211] within
0.02 at each end. Each
launch entry records the box's torch, CUDA and Python environment from its
log (§19.1).

**Cost.** Fork runs about 2.9 h each (`forkcf4b_A_s1_tx` and
`forkcf4b_D_s1_tx` took 2.59 h at 56.8 s/step), on-policy runs about 1.35 h
(`ctrl4b_s0` and `ctrl4b_s1` at 25 s/step). Twelve fork runs and sixteen
controls come to about 56 box-h, about 14 h on four leases, before lease
losses. The extension adds about 34 box-h.

### 19.3 / 20.2 / 21.1 Launches on five fresh leases (2026-10-06 14:24Z / 19:54 IST)

Written after launch and before any read. Five leases (box1 to box5) came up
at 14:20Z; `bringup2.sh`'s "setup FAILED" was the known false positive (every
box's `/marimo/v2/setup.log` ends "SETUP DONE", 66-79 s, 8 of 8 shards). The
three 4B files were pushed and verified on every box, and the six synced files
matched by MD5 at `e4b74f5` (branch `s21-tooling`), whose trainer code has no
diff from `6de23ad`. All five launched at 14:24:42Z through `launch_fork.sh`:

| box | label | queue item | banner checked |
| --- | --- | --- | --- |
| box1 | `forkcf4b_A_s0_tx_h320` | §19 relaunch (§19.2) | seed 0, cell A, `fork_cf_g6`, 320 steps, reads 40-280, fork share tie/crossfit, mass 0.3 |
| box2 | `forkcf4b_D_s0_tx_h320` | §19 relaunch (§19.2) | seed 0, cell D, as box1 |
| box3 | `forkcf4b_A_s0_tx_p20` | §20 relaunch (§20.1) | seed 0, cell A, tie/crossfit, mine priced at −20.0 in training |
| box4 | `forkcf4b_A_s1_tx_p20` | §20 relaunch (§20.1) | seed 1, cell A, as box3 |
| box5 | `ctrl4b_A_s0_m3` | §21 item 1 | seed 0, cell A, `onpolicy_g6`, mass 0.3, no fork or dose line |

Every box runs Python 3.13.11, torch 2.11.0+cu130 (CUDA 13.0), peft 0.20.0 and
transformers 5.14.1 under `/marimo/persist/env.sh`, on an NVIDIA RTX PRO 6000
Blackwell Server Edition (§21 "Measurement").

**One reading fixed before any §21 read.** §21's credit condition on P_fork
says "its both-solved part's interval lies below zero". `scripts/box/s21.py`
prints two forms: `both`, the mean gap per episode on the mazes both runs
solved (§18.1's −0.317 on 101 mazes), and `both part`, those mazes' share of
the contrast. The condition reads `both`, the form §18.1 reported. `both part`
is reported beside it and credits nothing. `s21.py` reproduces §18.1's seed-1
interval on maze resampling, −0.484 [−0.766, −0.211] at rng 0 (the gap to
[−0.773, −0.211] is the random stream; rng 2 gives it exactly), checked by the
results-verifier.

### 21.2 `ctrl4b_A_s0_m3` ended (2026-10-06 15:35Z / 21:05 IST): a control, not AVOIDS at any read; nothing credited

160 steps, status DONE, 25.2 s/step, trainer code `e4b74f5` (no trainer-code
diff from `6de23ad`; only four `scripts/box` analysis helpers added),
`onpolicy_g6`, cell A, seed 0, `MZ01_MOVE_MASS_COEF=0.3`. Every read is valid
by §9's bars (bare 1.000 and unparsed 0.000 at every read; raw move mass
≥ 0.9998 at every read step, 1.000 at 159). The trial labels it DEAD
(zero-advantage 0.867 over the last fifth). Reads, solve / mine ratio against
the seed-0 reference (0.922 / 2.1875): 0.844 / 0.914 at 40, 0.891 / 0.893 at
80, 0.828 / 0.804 at 120, 0.938 / 0.854 at 160. `mean_steps` 14.23, 13.55,
14.52 at 40/80/120 (reference 13.05); the 0.804 at 120 comes with solve
−0.094 and capped share 0.172. No read is AVOIDS. At 160: capped share 0.0625
(reference 0.078), `mean_steps` 13.10 (13.05), solved routes (`optimality`
× 12) 12.38 (12.13), `detour_paid_solved` 0.000 (reference 0.008, 1 of 118
solved).

The old control `ctrl4b_s0` (mass 0.1, `fce301f`, no `episodes`) ended
RANDOMISED at 0.800 with solve 0.734 and capped share 0.266. That is one run
each, differing in mass, code and lease (§19.1: 0.11 between runs), so no
cause is assigned. This run is a control (D-23) and nothing is credited from
it. It fills §21's onA cell at seed 0. Its cell-D partner `ctrl4b_D_s0_m3`
launched at 15:37:00Z on the lease this run freed (box5 in §21.1's table).
Verified by the results-verifier with corrections.

### 20.2 `forkcf4b_A_s1_tx_p20` reads AVOIDS at 120 (2026-10-06 16:19Z / 21:49 IST); credit withheld at this read

A mid-run read, at step ~121 of 160 (launched 14:24:42Z at `e4b74f5`, §21.1).
At 120: solve 0.906 (reference 0.922, −0.016), mine ratio 0.664 against the
seed-1 reference 2.164, valid (bare 1.000, unparsed 0.000, mass 1.000). The
partner `forkcf4b_D_s1_tx` read 0.852 / 1.217 at 120, not AVOIDS. Paired A − D
gap at 120: −1.195 mine steps per episode, bootstrap 95 % [−1.750, −0.734],
t [−1.717, −0.674]; solve gap +0.055, capped gap −0.055. §16.4's read 2 over
steps 0–119: 309 credited forks, tie share 0.071 (gate passes), pooled
A(taken) −6.51 [−10.41, −3.51]. Against the reference: capped share 0.094
(0.078), `mean_steps` 13.88 (13.02), solved routes 12.84 (12.08),
`detour_paid_solved` 0.052 (0.000). At 120 the control `ctrl4b_s1` read
0.859 / 0.845 (capped 0.141) and the undosed arm `forkcf4b_A_s1_tx`
0.805 / 0.823.

By §20's rule the read meets the AVOIDS, partner and paired conditions. The
paired gap is mostly the partner's own rise to 1.217 (−106 of −153 mine steps
on the 25 mazes either run failed; −0.456 [−0.612, −0.301] on the 103 both
solved, routes 12.88 against 12.24). §17's clause names no size. Here the
capped share rose +0.016 (2 mazes) with solve −0.016, where precedent applied
it only at +0.094 or more (§18.1, §16.7); fixing a size now, after the read,
would change the pre-registration. §21's matched control `ctrl4b_A_s1_m3` has
no read yet. Credit is withheld at this read. One seed, one read. The seed-0
arm at −20 read RANDOMISED at 80 (0.734 / 0.871), so §20's curve condition
(both seeds) is not met. Verified by the results-verifier with corrections.

### 21.3 `ctrl4b_D_s0_m3` ended (2026-10-06 16:46:42Z / 22:16 IST): the seed-0 cell-D control; nothing credited

160 steps, status DONE, 25.0 s/step, trainer code `1aaf9ca` (no trainer-code
diff from `6de23ad`), `onpolicy_g6`, cell D (mines unpriced), seed 0,
`MZ01_MOVE_MASS_COEF=0.3`. Every read is valid by §9's bars (bare 1.000 and
unparsed 0.000 at every read; raw move mass ≥ 0.996 at every read step, 1.000
at 159). The trial labels it DEAD (zero-advantage 0.945 over the last fifth);
not RANDOMISED by the cell-D rule (training solve −0.009 against its own
start; held-out solve −0.094 against the reference). Reads, solve / mine ratio
against the seed-0 reference (0.922 / 2.1875): 0.961 / 0.932 at 40,
0.891 / 1.036 at 80, 0.875 / 0.964 at 120, 0.828 / 1.071 at 160. At 160:
capped share 0.172 (reference 0.078), `mean_steps` 14.31 (13.05), solved
routes 12.30 (12.13), `detour_paid_solved` 0.009, 1 of 106 solved (reference
0.008, 1 of 118).

It is a control (D-23) and fills §21's onD cell at seed 0. With
`ctrl4b_A_s0_m3` (§21.2) the paired A − D gap at 160 is −0.477 mine steps per
episode, bootstrap 95 % [−0.961, −0.055], t [−0.932, −0.022]; solve gap
+0.109, capped gap −0.109. That is the seed-0 term of §21's P_on in ratio,
0.854 − 1.071 = −0.218 (−0.477 over the shared reference 2.1875), with an
interval over mazes, not runs or seeds (§19.1). On the 102 mazes both solved
the gap is −0.147 per episode, bootstrap 95 % about [−0.265, −0.029]. That is
−15 of −61 mine steps; −46 come from the 26 mazes where either run capped,
−37 of them on the 18 that only A solved. It is a property of the control and
credits nothing; reading it as a result for the on-policy recipe waits for the
user (D-21, D-23, §21). `ctrl4b_A_s1_m3` launched at 16:47:43Z on box5.
Verified by the results-verifier with corrections.

### 20.3 `forkcf4b_A_s1_tx_p20` ended (2026-10-06 17:03Z / 22:33 IST): RANDOMISED at 160; no read credited

160 steps, status DONE, 58.2 s/step, `e4b74f5`, seed 1, mine priced at −20 in
training. Every read is valid (bare 1.000, unparsed 0.000; raw move mass
≥ 0.9998 at every read step). The trial labels it DEAD (zero-advantage 0.805
over the last fifth). Reads, solve / mine ratio against the seed-1 reference
(0.922 / 2.164): 0.883 / 0.924 at 40, 0.930 / 0.863 at 80, 0.906 / 0.664 at
120 (AVOIDS, credit withheld, §20.2), 0.633 / 0.162 at 160. At 160: capped
share 0.367 (reference 0.078), `mean_steps` 18.91 (13.02), solved routes 15.95
(12.08), `detour_paid_solved` 0.642 (0.000). The partner `forkcf4b_D_s1_tx`
read 0.906 / 0.989 at 160, not AVOIDS. Paired A − D gap at 160: −1.789 mine
steps per episode, bootstrap 95 % [−2.148, −1.445]; solve gap −0.273, capped
gap +0.273. §16.4's read 2 over steps 0–159: 363 credited forks, tie share
0.061, pooled A(taken) −6.21 [−9.52, −3.61].

Not credited: the final read is RANDOMISED (solve 0.289 below). The fall comes
with a solve loss, a rise in capped share and detours in solved episodes. That
is §17's over-avoidance shape, in the detour form of `forkcf4b_A_s1_tx_p30`
and `forkcf4b_A_s0_t` (§16.7). The −50 and −100 arms fell with capped
episodes and no detours (§17.1). No read of the run is credited. The read at
120 stays withheld as §20.2 left it, because its matched control
`ctrl4b_A_s1_m3` (§21) is at step 38 with no read. The old control
`ctrl4b_s1` (mass 0.1, `18c0860`) ended 0.859 / 0.765. §20's curve condition
(both seeds at −20) is not met. Seed 1 ends RANDOMISED. Seed 0
(`forkcf4b_A_s0_tx_p20`) read RANDOMISED at 80 and 120 (0.734 / 0.871,
0.742 / 0.961), and its read at 160 is pending. `ctrl4b_D_s1_m3` launched at
17:04:06Z on box4. Verified by the results-verifier with corrections.

### 20.4 `forkcf4b_A_s0_tx_p20` ended (2026-10-06 17:19:59Z / 22:49 IST): RANDOMISED from 80; §20 closes

160 steps, status DONE, 64.2 s/step, `e4b74f5`, seed 0, mine priced at −20 in
training. Every read is valid (bare 1.000, unparsed 0.000; raw move mass
≥ 0.9998 at every read step). The trial labels it RANDOMISED. Reads, solve /
mine ratio against the seed-0 reference (0.922 / 2.1875): 0.875 / 0.893 at
40, 0.734 / 0.871 at 80, 0.742 / 0.961 at 120, 0.461 / 0.757 at 160; RANDOMISED
at 80, 120 and 160. At 160: capped share 0.539 (reference 0.078),
`mean_steps` 18.62 (13.05), solved routes 12.34 (12.13), `detour_paid_solved`
0.017, 1 of 59 solved (reference 1 of 118). The partner
`forkcf4b_D_s0_tx_h320` read 0.961 / 0.904 at 160. Paired A − D gap at 160:
−0.320 mine steps per episode, bootstrap 95 % [−0.727, +0.133], t [−0.756,
+0.115]; solve gap −0.500, capped gap +0.500. §16.4's read 2 over steps
0–159: 466 credited forks, tie share 0.047, pooled A(taken) −6.58 [−8.57,
−4.74].

No read is AVOIDS. Solve fell more than 0.10 from the read at 80 on, with
capped share 0.26 to 0.54, which is 1 − solve at each read, while solved
routes stayed at 12.3–12.6 (reference 12.13). The matched control
`ctrl4b_A_s0_m3` read 0.938 / 0.854 at 160 (§21.2), and `ctrl4b_s0` read
0.734 / 0.800. The lost 10-04 run of this label read 0.875 / 0.829 at 40,
against 0.875 / 0.893 here (§19.1's run-to-run spread is 0.11).

§20 closes. Seed 0 has no AVOIDS read, so §20's both-seeds condition fails
whatever becomes of seed 1's withheld read at 120 (§20.2; seed 1 ended
RANDOMISED at 160 with detours and capped episodes, §20.3). The curve (−10,
−15, −20, −25 on seeds 0–3) does not run. §21's queue runs as written:
Precedence item 2 does not fire, and item 3 adds nothing because §20's next
step needs both seeds. `forkcf4b_A_s2_tx` launched at 17:21:57Z on box3.
Verified by the results-verifier with corrections.
