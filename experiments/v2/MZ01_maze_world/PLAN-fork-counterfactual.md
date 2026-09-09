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
