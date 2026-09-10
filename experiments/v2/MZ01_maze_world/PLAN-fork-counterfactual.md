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

**The construction test (§1, `DIAGNOSIS-2026-09-09.md` §3,
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

## 14. Takeover at 05:06Z / 10:36 IST: the batch verified alive, the monitor rebuilt, the next launches pre-registered

Written 05:06–05:20Z by the incoming lead, **before any read of §13's batch
exists** (`eval_at` is `{}` on all four traces at this hour; the first reads at
step 40 are due ≈ 05:38–05:40Z on the three ~55 s/step arms and ≈ 06:00Z on
`_f2`). Nothing below reads a result; it records the state, the monitor, and
what launches on each box the moment it frees.

### 14.1 The four runs are alive and stepping (05:06:21Z)

Probed off `/marimo/repo/<label>.pid` and `/proc/<pid>/stat` (state `Z` would
mean finished; all four read `R`), with the log's last line beside it:

| box | label | pid | state | last line at 05:06Z | move mass |
| --- | --- | --- | --- | --- | --- |
| box 1 | `forkcf4b_A_s0_p30` | 1542 | R | step 4, solve 0.917, mine/ep 2.083, \|g\| 21.57 | 1.000 |
| box 2 | `forkcf4b_A_s1_p30` | 1532 | R | step 3, solve 1.000, mine/ep 1.500, \|g\| 0.286 | 1.000 |
| box 3 | `forkcf4b_A_s0_f2` | 2218 | R | step 2, solve 1.000, mine/ep 1.000, \|g\| 5.189 | 1.000 |
| box 4 | `forkcf4b_A_s0_g10` | 5153 | R | step 3, solve 1.000, mine/ep 0.792, \|g\| 46.44 | 1.000 |

All four leases answer HTTP 200. Steps 2–4 at 05:06Z against launches at
05:00:37–05:01:07Z is the 55–85 s/step of §13.2 plus model load, so no arm is
stalled. `results/forkcf4b_A_s0_p30/fork_cf_g6.json` at 05:07Z carries
`history.move_mass` 0.9997–0.9999 over its first seven steps and
`reference.json` `eval` solve 0.921875, mine/ep 2.1875 — the seed-0 reference
every ratio below is taken against.

### 14.2 The monitor, and the one piece of it this machine cannot run

The session runs on a **remote ephemeral container**, not the workstation the
handoff was written on (`/home/user/known_by_contruction_v2`, not
`/home/manan/Projects/files/digital_minds-maze`). Three of the four monitoring
layers are restored as written; the fourth cannot exist here, and pretending
otherwise would be the failure the record exists to prevent.

1. **One Claude Code background task per box** — `scripts/box/watch_box.sh`
   at `CYCLE=120 MAX=3000`, boxes 1–4 on the four labels. Running; first cycles
   logged 05:07:31–05:07:46Z, three small files pulled per box
   (`fork_cf_g6.json`, `reference.json`, the run log).
2. **A durable supervisor** — `scripts/box/watch_all.sh` with
   `JOBS="box1:forkcf4b_A_s0_p30 box2:forkcf4b_A_s1_p30 box3:forkcf4b_A_s0_f2
   box4:forkcf4b_A_s0_g10"`, `CYCLE=120`, detached under `setsid nohup` to
   `$MZ01_BOX_DIR/watch_all.log` (started 05:07:44Z). This is the substitute for
   the `mz01-watch` **systemd unit**, which cannot be created here: the
   container has no systemd bus (`systemd-run --user` → "Failed to connect to
   bus: No medium found"; `/run/systemd/system` absent, pid 1 is `process_api`).
   Concurrent pulls are safe — `scripts/box/pull_big.sh` takes an `flock` per
   target file and a second puller exits 5.
3. **The read monitor** — `scripts/box/read_monitor.py`, new in this pass, under
   the Monitor tool at a 120 s poll: one line per **new** held-out read off the
   pulled traces, plus any terminal event the watchers write to
   `$MZ01_BOX_DIR/events.log` (a run ENDED and its box is idle, a lease
   unreachable twice). It does not re-implement the verdict: `_cell_a_verdict`
   is imported from `scripts/score_mz01_explore.py:218` (bars `MINE_RATIO_BAR`
   0.75, `SOLVE_BAND` 0.10, `UNPARSED_BAR` 0.05), and §9's two bars the cell-A
   verdict does not carry are printed beside it and flagged — bare share
   (`first_token_class.bare`) and the raw `move_mass` at the read's own step,
   floor 0.90 each. Self-tested on a synthetic arm before it was trusted: the
   four verdict paths (AVOIDS, ALIVE, RANDOMISED, INVALID-READ), both invalid
   flags, and the no-replay state file each behaved, and the test caught a crash
   on a missing state directory that silence would have hidden.
4. **`mz01-inhibit`** — not created and not applicable. It blocks
   `sleep:idle:handle-lid-switch` on a laptop that would otherwise suspend
   mid-pull (`FAILURES.md`, the 23:02Z suspend of §12). This container has no
   logind and does not sleep. **If the workstation is also holding leases or
   watchers this hour, its own `mz01-watch` and `mz01-inhibit` units are
   untouched by this session — they are on a machine this one cannot reach, so
   they were neither stopped nor recreated.**

### 14.3 The next launch on each box, pre-registered now

An idle lease gets reaped, so each box's next job is fixed **before** its read,
by §6/§11/§13.3's rules and nothing else. Finals ≈ 08:00Z on boxes 1, 2, 4 and
≈ 09:00Z on box 3; by 08:00Z the `_f2` arm's reads at 40 and 80 are on disk, so
the fork-count branch below is decided on evidence, not on a guess.

| box frees | if its arm AVOIDS at a valid read | if it is flat (ALIVE at every read) | if a read is INVALID |
| --- | --- | --- | --- |
| box 1 `_A_s0_p30` | `forkcf4b_D_s0_gb` — its own cell-D partner at the **world's** price, seed 0, guards as §13.2 (the dose cannot run on cell D; §13.3). `_gb` because `_g`'s trace is the lost run's read at 40 (§11) and must not be overwritten | `forkcf4b_A_s0_f3` (`MZ01_FORK_PER_EP=3`) — the zero-build member of the lever §6 names when nothing avoids, launched only if `_f2`'s `fork_share_detour_better` is unmoved from `_A_s0_g`'s 0.051; if it moved, the second draw is not uniform and that is a build question first (§13.3), so the box takes `forkcf4b_A_s1_f2` instead | by §9 the arm is reported as the regulariser failing; the box takes the flat-branch arm while the fix is built |
| box 2 `_A_s1_p30` | `forkcf4b_D_s1_gb` — the same at seed 1 | `forkcf4b_A_s1_f2` (the count knob's second seed) | as box 1 |
| box 4 `_A_s0_g10` | its partner is `forkcf4b_D_s0_g` (§13.3); on AVOIDS it queues behind box 1's D partner rather than duplicating it, and takes the flat-branch arm | as box 1's flat branch, one arm down the list | **mass < 0.90 at any read closes this arm at that read** and §10's pre-registration follows — the mass guard moves OUTSIDE the cap, **not** a smaller cap (§13.3). That is a build; the box takes a flat-branch arm meanwhile |
| box 3 `_f2` (≈09:00Z) | as box 1's AVOIDS branch, one seed down | the geometry arm of §6 — forks placed where continuations differ, **never by mine adjacency** — designed in this plan, built by an Opus subagent in a worktree, critic-reviewed, tested, then run | as box 1 |

**Both dose seeds avoiding** still sends the dose to the response curve (−10,
−20, −30, −50) on seeds 0–3, as §11 and §13.3 wrote it; that supersedes the
flat-branch rows above for boxes 1 and 2.

**The construction test holds across every arm named here.** `MZ01_FORK_PER_EP`
counts draws from the episode's geometric fork list and reads no mine accessor
(§13.1, build test 2); a cell-D partner is the same recipe at the world's price;
the geometry arm is admitted only if its fork choice is blind to mine adjacency.
No arm below routes, hints, prefixes or places a fork by where a mine is.

## 15. Fork selection is measured out as a lever; §14.3's flat branch is amended (2026-09-10 05:30Z / 11:00 IST)

Still **before any read of §13's batch** (`eval_at` empty on all four traces at
05:35Z; the first read at 40 is due ≈ 05:41Z on the three ~50 s/step arms). What
follows is a geometry measurement and a training-trace reading; neither is a
held-out result, and neither moves a verdict.

### 15.1 The measurement: no distance rule can select route noise away

`docs/research/fork-geometry-probe-2026-09-10.md`, with the script and its
verbatim output beside it. On the trial's own 256-maze training pool, every fork
on the shortest DAG scored by distance to the goal alone (`maze2.bfs_dist`,
walls only — the accessor `episode_forks` already uses):

* `d_spread` is **0 or 2 and nothing else** — a grid move changes the distance
  to the goal by ±1, so a fork is a tie or a 2-step (4-point) difference.
* The tie worth having — both branches ON a shortest route, where avoiding costs
  no route length — is **16 forks of 4 344, 0.1 per maze**. These mazes are
  near-trees: a fork is the way on against a detour.
* The 1 258 other ties are the opposite case, every branch heading AWAY from the
  goal (0.286 of forks, mine-relevant share 0.309 against the uniform draw's
  0.239): a 1.29× gain bought by spending the fork term on decisions that are
  not on the way to the goal.

**§6's second branch is therefore answered in the negative, by geometry rather
than by a box-night.** The route-length difference the price must beat is a
property of nearly every fork in this world, not of which fork the draw picks,
and no mine-blind selection rule removes it. `MZ01_FORK_SELECT` is not built.

### 15.2 The training traces say the dose is doing exactly what §13.1 predicted

Read at steps 14–16 off the pulled `fork_cf_g6.json` of all four arms — training
diagnostics, not reads:

| arm | steps | `fork_adv_abs_mean` | `fork_share_nonzero` | `fork_share_detour_better` | `n_forks` | s/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `forkcf4b_A_s0_p30` | 16 | **8.03** | 0.911 | 0.046 | 24.0 | 49.7 |
| `forkcf4b_A_s1_p30` | 15 | **7.66** | 0.908 | 0.036 | 24.0 | 52.1 |
| `forkcf4b_A_s0_f2` | 9 | 3.85 | 0.861 | 0.027 | 48.0 | 79.8 |
| `forkcf4b_A_s0_g10` | 14 | 4.33 | 0.884 | 0.052 | 24.0 | 52.0 |

§13.1 item 2 decomposed the fork advantage as 4.26 on the priced cell A against
2.37 on the mine-free cell D, so the price at −10 is worth 1.89 points and a
dose of −30 predicts 2.37 + 3 × 1.89 = **8.04**. The dose arms read 8.03 and
7.66. The dose knob does to the fork advantage exactly what it was built to do;
whether that reaches the held-out mine rate is what the reads from 05:41Z say.
`_g10` at 0.052 reproduces `_g`'s 0.051 detour share, as a cap change should.
`_f2` carries `n_forks` 48 — two per episode as intended — at 79.8 s/step
against the 85 estimated, so its final lands ≈ 08:35Z, not 09:00Z.

**`_f2`'s share at 0.027 against `_g`'s 0.051 is flagged, not read.** §13.3 makes
a moved share a build question (is the second draw uniform over the episode's
forks?). At nine steps this is inside the step-to-step spread of a share whose
mean is one fork in twenty; it is re-read at 40 steps before anything is
concluded, and the arm's own reads are unaffected either way.

### 15.3 The amended flat branch

§14.3's box-3 row named "the geometry arm of §6" for the case where nothing
avoids. §15.1 closes that arm before it was built, so the flat branch is now, in
order:

1. **The dose ladder upward: `forkcf4b_A_s0_p50`** (`MZ01_SCORE_MINE=-50`, every
   other switch as §13.2). Route noise is ±4 points at every fork (§15.1); the
   price is 1.9 points at −10, 5.7 at −30 and ~9.5 at −50, and −50 is the first
   dose at which the price exceeds the noise it competes with at a single fork.
   Zero build — the dose knob shipped with `4f8d165`, 139 tests.
2. **`forkcf4b_A_s1_f2`** — the count knob's second seed, if `_f2`'s share at 40
   reproduces `_g`'s 0.051 (§15.2); if it does not, the second draw's uniformity
   is a build question and this arm waits for it.
3. **The continuation side, not the fork side** — the probe bounds the loss from
   *selection* at 1.29× while the gap between the geometry's 0.239 mine-relevant
   forks and the traces' 0.05 realised ones is ~5× (§15.1's caveat: the two are
   not the same statistic, so this is a bound and not a measurement). The
   catalogue's #2 (fork-anchored baseline, GiGPO) and #4 (the std normaliser)
   are the pre-registered ways to spend it, and `docs/research/APPROACHES.md` §B
   already carries both with their costs. Either is a build in a worktree with a
   critic, so it queues behind the two zero-build arms above.

**Both dose seeds avoiding still supersedes all of this** with §11's response
curve (−10, −20, −30, −50) on seeds 0–3. Nothing here reads a mine to choose a
fork, a move, an episode or a comparison: §15.1's rules were scored offline in a
file no training step imports, and the one that survives is a number in the
priced return.

## 16. The batch's reads at 40: four VALID reads, none AVOIDS, and the one ratio that moved was competence falling (2026-09-10 06:10Z / 11:40 IST)

All four arms of §13 took their read at step 40 between 05:40Z and 06:05Z. Every
read is VALID by §9 (unparsed ≤ 0.029 against the 0.05 bar, bare share ≥ 0.971
and raw `move_mass` ≥ 0.999 against the 0.90 floors). **No arm AVOIDS**; the bar
is a mine ratio below 0.75 with solve within 0.10 of its reference.

| arm | ratio | mine/ep | solve (ref) | mean_steps (ref) | invalid (ref) | verdict |
| --- | ---: | ---: | --- | --- | --- | --- |
| `forkcf4b_A_s0_p30` | **0.832** | 1.820 / 2.188 | 0.828 (0.922) | 14.48 (13.05) | **0.211** (0.046) | ALIVE |
| `forkcf4b_A_s1_p30` | 1.051 | 2.273 / 2.164 | 0.914 (0.922) | 13.41 (13.02) | 0.004 (0.052) | ALIVE |
| `forkcf4b_A_s0_f2` | 0.896 | 1.961 / 2.188 | **0.969** (0.922) | 12.61 (13.05) | 0.016 (0.046) | ALIVE |
| `forkcf4b_A_s0_g10` | 0.918 | 2.008 / 2.188 | 0.938 (0.922) | 13.08 (13.05) | 0.034 (0.046) | ALIVE |

### 16.1 The dose's two seeds went opposite ways, and neither went to the price

**`_s0_p30` moved the ratio furthest of any guarded arm ever run (0.832 against
the `_g` arms' 0.90–0.96) and §8.4 does not credit it.** Solve fell 0.094 — one
hundredth short of RANDOMISED — while `mean_steps` ROSE 13.05 → 14.48, wall
bumps went 0.046 → 0.211 (4.6×) and capped episodes 0.078 → 0.172. The detour
numbers say where the "avoidance" went: `detour_paid` rose 0.055 → 0.102 but
`detour_paid_solved` did not move (0.0085 → 0.0094) and
`detour_unsolved_share` went 0.078 → 0.172. **Every extra detour was paid by an
episode that never solved.** The training trace agrees, over 43 steps: solve
0.908 → 0.783 (first ten against last ten), invalid 0.016 → 0.049, capped 0.092
→ 0.217, and the training `mine_step_rate` going the WRONG way, 1.03 → 1.24.
This is D-15's derailment at a milder dose, not a priced detour.

**The on-policy controls settle it.** `ctrl4b_s0` finished RANDOMISED at ratio
0.80 with solve 0.734, and `ctrl4b_s1` ALIVE at ratio 0.77 with solve 0.859
(§13.3) — both **lower ratios than 0.832, reached with no fork term at all**, by
losing competence. A mine rate that falls with solve is what this world's
control already does, so a ratio that arrives that way is not evidence about the
price at any dose.

**`_s1_p30` is the same dose on the other seed and it moved nothing** — ratio
1.051, solve within 0.008 of its reference, wall bumps 0.004 against the
reference's 0.052, `mean_steps` +0.39. The dose's two seeds therefore differ not
in degree but in kind: one degraded, one did not move. Nothing yet distinguishes
them but the seed.

### 16.2 The dose does what it was built to do, one layer short of the read

`fork_adv_abs_mean` at 40 steps: 8.03 (`_s0_p30`) and 7.66 (`_s1_p30`) against
4.33 (`_g10`) and 3.85 (`_f2`), where §13.1's decomposition (4.26 on cell A,
2.37 on the mine-free cell D, so 1.89 of price at −10) predicted 2.37 + 3 × 1.89
= 8.04. **The price's share of the fork advantage tripled exactly as designed and
the held-out mine rate did not follow.** The chain from the priced return to the
fork advantage is intact and measured; the break is after it.

### 16.3 `_g10` answers its own question: the guard holds at the bigger step

§13.3 set this arm a close rule — raw move mass below 0.90 at any read means the
guard alone does not hold the old step size, and §10's pre-registration (the mass
guard moves OUTSIDE the cap) follows. **`move_mass` reads 1.000 at 40 with the
cap at 1.0**, so the rule does not fire and the arm runs on. The 3.3× larger fork
step did not move the rate either (0.918) and cost nothing (solve 0.938, above
its reference).

### 16.4 `_f2` is the cleanest read of the batch, and it is navigation

Solve 0.969 against the reference's 0.922, `mean_steps` 12.61 against 13.05,
wall bumps 0.016 against 0.046, capped 0.031 against 0.078 — the arm improved at
everything. Its ratio of 0.896 is therefore exactly what §8.4 refuses to credit:
`detour_paid` FELL 0.055 → 0.023 and `detour_paid_solved` fell to **0.000**. Not
one solved held-out episode paid a detour. The mine rate fell because the
organism navigates better and more directly, not because it goes around
anything.

Its build question (§13.3, §15.2) is closed in its favour:
`fork_share_detour_better` reads 0.047 over 29 steps against `_g`'s 0.051, so
the second draw is uniform over the episode's forks and the 0.027 flagged at
nine steps was step-to-step noise. **`forkcf4b_A_s1_f2` is a legal queue item.**

### 16.5 What this does and does not do to the queue

Nothing fires yet: these are the reads at 40 of four arms that run to 160, and
§13.3's rules are read at every read. Three things are recorded now because they
bear on §15.3's flat branch when the first box frees at ≈ 07:15Z:

1. **The dose ladder upward is now in tension with its own evidence.** §15.3
   ranked `forkcf4b_A_s0_p50` first because route noise is ±4 points and −50 is
   the first dose whose price exceeds it. §16.1 shows −30 already degrading the
   seed-0 policy without buying avoidance, and a larger dose plausibly degrades
   more. The decision is not taken here; it is taken at the reads at 80 and 120,
   with `_s0_p30`'s solve trajectory as the deciding number. If `_s0_p30` reads
   RANDOMISED at 80, −50 is withdrawn and the flat branch begins at
   `forkcf4b_A_s1_f2`.
2. **`_f2` is the only arm whose competence improved**, which makes the count
   knob the safe half of the batch and its second seed the safe next arm.
3. **The break is after the fork advantage, not before it** (§16.2). That is the
   continuation side, and §15.3 item 3 already names the catalogue's #2
   (fork-anchored baseline) and #4 (the std normaliser) as the pre-registered
   ways to spend it. Neither is built; both are worktree builds with a critic.

No mine position entered any of this. The dose is a number in the priced return,
the cap is an optimiser knob, and `fork_per_ep` counts draws from the geometric
fork list.

## 17. The reads at 80: the ratio tracks competence on every arm, and no solved episode has ever paid a detour (2026-09-10 06:30Z / 12:00 IST)

Three arms took their read at 80 between 06:20Z and 06:27Z (`_f2` reaches 80 at
≈ 07:00Z at its 80 s/step). All VALID by §9 (unparsed ≤ 0.006, bare ≥ 0.994,
`move_mass` ≥ 0.999). **None AVOIDS.**

| arm | ratio @40 → @80 | solve @80 (ref) | mean_steps @80 (ref) | invalid @80 (ref) |
| --- | --- | --- | --- | --- |
| `forkcf4b_A_s0_p30` | 0.832 → **0.843** | 0.852 (0.922) | 14.27 (13.05) | 0.204 (0.046) |
| `forkcf4b_A_s1_p30` | 1.051 → **0.906** | 0.898 (0.922) | 13.76 (13.02) | 0.072 (0.052) |
| `forkcf4b_A_s0_g10` | 0.918 → **0.971** | 0.906 (0.922) | 13.32 (13.05) | 0.036 (0.046) |

### 17.1 The ratio is a function of the damage, on all seven reads

Ordering this batch's seven reads by wall bumps orders them by mine ratio, and
the two dose arms move together with their own degradation over time:

| read | invalid | ratio |
| --- | ---: | ---: |
| `_s1_p30` @40 | 0.004 | 1.051 |
| `_f2` @40 | 0.016 | 0.896 |
| `_g10` @40 | 0.034 | 0.918 |
| `_g10` @80 | 0.036 | 0.971 |
| `_s1_p30` @80 | 0.072 | 0.906 |
| `_s0_p30` @80 | 0.204 | 0.843 |
| `_s0_p30` @40 | 0.211 | 0.832 |

`_s1_p30` is the clean demonstration: between 40 and 80 its wall bumps went
0.004 → 0.072 and its ratio 1.051 → 0.906 in the same interval. **The dose's
effect on the held-out mine rate is mediated by the damage it does, and by
nothing else this batch can see.** `_g10`, whose competence held (invalid 0.036,
below its reference), moved its ratio the wrong way, 0.918 → 0.971.

### 17.2 The positive test, and it has never fired

`detour_paid_solved` — the share of **solved** held-out episodes that paid a
detour — is the direct signature of the behaviour the programme is trying to
install: went around a mine and still reached the goal. Across this batch:

| arm | reference | @40 | @80 |
| --- | ---: | ---: | ---: |
| `_s0_p30` | 0.0085 | 0.0094 | 0.0092 |
| `_s1_p30` | 0.0000 | 0.0000 | 0.0000 |
| `_g10` | 0.0085 | 0.0167 | 0.0000 |
| `_f2` | 0.0085 | 0.0000 | — |

**Not one read has put it meaningfully above its warm start.** Every ratio this
batch has produced was bought by solving less, not by detouring more.

**A metric warning that follows from the same rows, and it changes how earlier
addenda should be read.** `detour_unsolved_share` equals `capped_rate` exactly
on all seven reads (0.1719/0.1719, 0.1484/0.1484, 0.0859/0.0859, 0.1016/0.1016,
0.0625/0.0625, 0.0938/0.0938, 0.0781/0.0781), so every unsolved episode counts
as a detour and **`detour_paid` is dominated by episodes that never solved.**
`detour_paid` is therefore not an avoidance signal and must not be read as one;
`detour_paid_solved` is the number. §16.1's reading of `_s0_p30`'s rise in
`detour_paid` stands only because it was already attributed to unsolved
episodes there.

### 17.3 Why a correct fork advantage does not reach the policy

`policy_entropy`, first ten steps against last ten, on the four arms:
0.0297 → 0.0039 (`_s0_p30`), 0.0177 → 0.0055 (`_s1_p30`), 0.0260 → 0.0014
(`_f2`), 0.0193 → 0.0035 (`_g10`). **The move distribution collapses to a point
mass inside the first forty steps on every arm**, priced or not, dosed or not.
At 0.004 nats the top move carries essentially all the mass, and
`zero_advantage_group_fraction` is 0.80–0.95, so the episode term is dead in
most groups and the fork term is the only live gradient — a step capped at norm
0.3 asked to flip decisions the policy holds above 0.999.

That is a mechanical account of §16.2: the dose tripled the price's share of the
fork advantage exactly as designed (8.03 measured, 8.04 predicted) and the
advantage is being applied to a policy that has already stopped moving. It is a
hypothesis, not a measurement — nothing in the trace records the fork
decision's own probability before and after a step — and §17.5 says what would
settle it.

### 17.4 Box 1's next arm: `forkcf4b_A_s0_p50`, as pre-registered

Box 1 frees at ≈ 07:19Z. §16.5 set one withdrawal condition on the −50 dose —
`_s0_p30` reading RANDOMISED at 80 — and **it did not fire** (solve 0.852, a
fall of 0.070 inside the 0.10 band; the arm in fact recovered from 0.828 at 40).
The pre-registration therefore stands and `forkcf4b_A_s0_p50` launches on box 1
the moment its predecessor's final read is on disk. Overriding a registered arm
on a mechanism found mid-run, in the same hours as the reads it would reinterpret,
is the post-hoc move the pre-registration exists to forbid.

**Launched as §13.2's four runs, with `MZ01_SCORE_MINE=-50`** (cell A, seed 0,
`MZ01_FORK_COEF` 1.0, `MZ01_FORK_GRAD_MAX` 0.3, `MZ01_MOVE_MASS_COEF` 0.3,
4B @ `cdbee75f`, 160 steps, reads 40/80/120/160). The dose is training-only: the
held-out read is priced by the world at −10 (`tests/test_maze_orgb.py:6657`,
`:6863`), so the ratio it is verdicted on is comparable to every other arm's.

**Its reading rule, fixed now.** The standing bars (§6, §8.4, §9, and the trial's
0.75 / 0.10 / 0.05) **plus** the two §17 adds, because this batch showed the
standing bars admitting a ratio that is damage:

1. **AVOIDS is credited only if `detour_paid_solved` rises above its reference**
   (0.0085 on seed 0). A ratio below 0.75 with that number flat is degradation
   and is reported as such, whatever the solve band says.
2. **`invalid_rate` within 0.05 of the reference at the read.** Wall bumps have
   no bar of their own and the move-mass guard cannot see them (the mass sits on
   a bare move word naming a move into a wall); at 0.20 the arm is not a policy
   whose mine rate means anything.

**And the decisive branch, registered before the arm runs:** if `_s0_p50` moves
its ratio only in step with its wall bumps and solve — the §17.1 pattern at a
larger dose — **the dose lever is closed**, at −10, −30 and −50 across three
seeds, and the flat branch moves to the collapse itself (§17.5), not to −100.

### 17.5 What comes after the dose, if §17.4's branch fires

Not a new knob: `MZ01_ENTROPY_COEF` is already the recipe's (0.1,
`docs/research/APPROACHES.md` §B #6, "Entropy at the level", IDEA) and the
collapse of §17.3 is what it acts on. The order, cheapest first, all still
subject to a critic and to the construction test:

1. **The diagnostic before the lever.** The trace records no probability at the
   fork decision, so §17.3 cannot be confirmed from what is on disk. A build
   that records, per step, the mean probability the policy assigns the move the
   fork term is pushing toward — before and after the update — turns the
   hypothesis into a measurement, and it is a read of the policy, not of a mine.
2. **The entropy coefficient raised**, as an arm with its own cell-D partner.
3. **The continuation side** — the catalogue's #2 (fork-anchored baseline) and
   #4 (the std normaliser), both worktree builds with a critic (§15.3 item 3).

Nothing here reads a mine to choose a fork, a move, an episode or a comparison.
