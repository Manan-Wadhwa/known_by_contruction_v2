# MZ01: what to run after §19 and §20, and how to pre-register it (design, 2026-10-04 ~09:40Z)

Design only. Nothing was built, launched or committed. Paths under
`experiments/v2/MZ01_maze_world/` are given from that directory. `PLAN` means
`PLAN-fork-counterfactual.md`. The planning arithmetic is in
`scratchpad/power21.py` beside this file; its inputs are cited below. The
results-verifier checked these numbers (VERIFIED WITH CORRECTIONS), and its
corrections are applied. The lead should run it again on §21 as committed.

What I read of the in-flight runs. I read only step counts and the list of
reads taken (at 09:25Z, `results/<label>/fork_cf_g6.json`), never their values.
`forkcf4b_A_s0_tx_h320` was at step 164 with reads 40 to 160 taken.
`forkcf4b_D_s0_tx_h320` was at step 152 with reads 40 to 120.
`forkcf4b_A_s0_tx_p20` was at step 38 and `forkcf4b_A_s1_tx_p20` at step 4,
neither with a read. At about 60 s/step the boxes free at about 11:35Z (box 2,
p20 s0), 12:00Z (box 1, A h320), 12:05Z (box 4, p20 s1) and 12:15Z (box 3, D
h320).

## 0. The answer in brief

1. Both comparisons matter, and the record is missing the cell that would let
   us read them together. The fork arm `_tx` keeps the on-policy update
   (PLAN §2.6) and adds the fork term. A − D (the cell-D partner) asks whether
   the price lowers mine steps under the fork design. A − control asks whether
   the fork term adds anything to what the priced on-policy update does by
   itself. No on-policy cell-D run exists at 4B, so nobody has tested whether
   the controls' 0.80 and 0.765 come from the price. The controls are also not
   matched to the arms (mass 0.1 against 0.3, older code, no per-episode
   records). The pre-registration (§6, §16, carried by §18 to §20) uses the
   control only as a binary gate ("control not AVOIDS"). It never requires the
   arm to fall below the control.
2. Measurement is the bottleneck for the question "does the price act, and does
   the fork term add to it". It is not the only bottleneck for AVOIDS. Run
   noise is about 0.05 to 0.08 in ratio per run. The paired maze interval at
   seed 1 is ±0.13 in ratio. A run whose true ratio sits at 0.77 to 0.80 falls
   under the strict 0.75 bar on 16 to 40 % of reads. With the control near the
   bar, the per-run gate credits a null arm (arm equal to control) about 0.2 of
   the time per seed per read. A powered design needs 8 seeds to resolve gaps
   of about 0.08 to 0.13 in ratio, and 12 to 20 seeds for 0.05 to 0.10, in line with
   DIAGNOSIS §3's 10 to 20 seeds per arm.
3. Ranked next pre-registrations: (1) the 2×2 sweep (fork term on or off ×
   cell A or D) at the world's price, matched controls first, seeds 0 to 7,
   about 56 box-h (about 14 h on four leases); (2) the −20 dose added to that
   sweep, only if §20 keeps solve at both seeds, about 17 box-h; (3) the sweep
   at 320 steps, only if §19 shows the gap growing and the user lets the
   horizon move, about 33 to 47 box-h; (4) a solve anchor (KL to the warm
   start), which needs a build and the user's decision on the band. Waiting for
   §19 and §20 is not a separate option. §21 orders its queue so the first
   four jobs (the matched controls at seeds 0 and 1) are needed under every
   §19 and §20 outcome. Precedence clauses hand the boxes to §19's or §20's
   pre-registered consequence if either fires.
4. A draft §21 for option 1 is in section 5.

## 1. What the control comparison means

### 1.1 The numbers at 160

Mine steps in 128 held-out episodes, the seed's reference, the ratio, solve,
capped share and solved routes (`optimality` × 12). Each value comes from
`results/<label>/<arm>.json` `eval` and `reference.json` `eval`.

| label | cell, arm | mass | code | mine steps / ref | ratio | solve | capped | solved routes | episodes kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `forkcf4b_A_s0_tx` | A, fork | 0.3 | `5fda1de` | 229 / 280 | 0.818 | 0.938 | 0.062 | 12.23 | no |
| `forkcf4b_D_s0_tx` | D, fork | 0.3 | `53f7c52` | 265 / 280 | 0.946 | 0.953 | 0.047 | 12.16 | no |
| `ctrl4b_s0` | A, on-policy | 0.1 | `fce301f` | 224 / 280 | 0.800 | 0.734 | 0.266 | 12.44 | no |
| `forkcf4b_A_s1_tx` | A, fork | 0.3 | `6de23ad` | 212 / 277 | 0.765 | 0.828 | 0.172 | 12.64 | yes |
| `forkcf4b_D_s1_tx` | D, fork | 0.3 | `6de23ad` | 274 / 277 | 0.989 | 0.906 | 0.094 | 12.18 | yes |
| `ctrl4b_s1` | A, on-policy | 0.1 | `18c0860` | 212 / 277 | 0.765 | 0.859 | 0.141 | 12.71 | no |

The references are 0.922 solve with capped 0.078 on both seeds (`reference.json`).
The controls ran at about 25 s/step (`summary.json` `seconds_per_step` 25.3 and
25.0) on the leases of 2026-09-09 (`started_at` 19:26:45Z and 19:42:09Z).

So the brief's facts check out. `_tx` ends below its partner at both seeds
(0.818 against 0.946; 0.765 against 0.989, paired gap −0.484 [−0.773, −0.211]
mine steps per episode, PLAN §18.1). It is level with the control at both
(0.818 against 0.800; 212 against 212 mine steps). The shapes differ at seed 0.
There the control reached 0.80 by losing solve (0.734, capped 0.266,
RANDOMISED; `ctrl4b_s0/summary.json`), while `_tx` kept its solve (0.938,
capped 0.062). At seed 1 the two look alike in ratio and in shape: solve 0.828
against 0.859, capped 0.172 against 0.141, solved routes 12.64 against 12.71.

### 1.2 Does a cell-D on-policy control exist?

No. `results/` holds two on-policy 4B runs, `ctrl4b_s0` and `ctrl4b_s1`, and
both have `config.cell` "A". The cell-D runs at 4B are all fork runs
(`forkcf4b_D_*`). The on-policy cell-D trials of 2026-08-29 were 14B and ran a
different recipe (RESULTS_LOG §4.3). So nobody has measured whether the
on-policy recipe's fall to 0.80 and 0.765 needs the price. Two readings fit
the record:

- The price acts on-policy. Then the fork term's extra share of
  mine-relevant forks (detour-better 0.79 against 0.05, PLAN §16.3, §15.1)
  adds nothing visible at 160, and the A − D gap is the price acting through
  the on-policy update that the fork arm also carries.
- The fall is drift that cell D would show too: solve loss, capped episodes
  and slightly longer solved routes, the shape of `ctrl4b_s0`. The 4B coverage
  read is consistent with this. It predicted DEAD; the controls ended
  RANDOMISED and ALIVE. The warm start puts a mine-free solve on 1 of 128 mazes
  (0.008, `PLAN-exit-priced-at-cap.md` §8, 19:45Z), so on-policy sampling has
  almost no detour to select (D-21). The fork partners on cell D do not drift
  that far (0.946, 0.989), but they carry a different update.

The missing on-policy cell-D run tells these apart, and it is cheap: about
1.2 h at 25 s/step.

### 1.3 Which comparison, and what the pre-registration says

The pre-registration asks for both, as gates on per-run verdicts.

- PLAN §6: AVOIDS "with the 4B on-policy control not AVOIDS is the Phase 1
  signal".
- PLAN §8.4: the cell-D fork run is the discriminating control. "Cell A AVOIDS
  with cell D not AVOIDS credits the price; both AVOIDS credits the
  navigation".
- PLAN §16: credit needs a cell-A `tx` arm that AVOIDS, its on-policy control
  (`ctrl4b_s0` / `ctrl4b_s1`) not AVOIDS, and `forkcf4b_D_s0_tx` not AVOIDS
  at the same read. §18, §19 and §20 cite §16's rule unchanged. §19 adds a
  paired maze interval against the partner only.
- `docs/planning/RESEARCH_PLAN.md` §1.4: the Phase 1 exit is "AVOIDS on two
  seeds with the control not AVOIDS on either".

The control gate is binary. An arm at 0.74 beside a control at 0.76 would be
credited, although it does not beat the control. The HANDOFF line "compare
every arm with its control" is a working note, not a rule.

What each comparison answers:

- A − D (fork partner) holds the fork term's route credit fixed and asks
  whether the price lowers mine steps. The partner is not a neutral null: its
  return rewards only short routes and solving.
- A − control (on-policy, same price) asks whether the fork term changes the
  priced policy at all. Today the controls differ from the arms in mass (0.1
  against 0.3), code and lease, and they have no `episodes` records, so a
  paired interval is impossible and "level" carries all of that.
- Control A − control D (missing) asks whether the on-policy fall is
  price-driven.
- (A − D) − (control A − control D) is what the fork term adds to the price's
  effect. That interaction is the fork-share line's own claim.

So the right comparison is a 2×2: fork term on or off, cell A or D, at matched
settings and seeds, read as continuous paired differences. Binary gates should
not carry it. What this means for the fork-share line: the selection works as
built, and the arm ends below its fork partner at both seeds. Seed 0's
approximate interval spans zero. Seed 1's maze-paired interval excludes zero,
over mazes only. Nothing
yet shows the fork term doing more than the priced on-policy update. At seed
0 its fall came with solve intact where the control's did not. That is one
seed and an unmatched control. The line is unresolved, not refuted.

## 2. Is the measurement now the bottleneck?

### 2.1 The noise, from the record

| source | value | file |
| --- | --- | --- |
| same seed and config, different leases, read at 40 | 0.868 against 0.979 (0.111 apart) | PLAN §19.1 |
| `_tx` at seeds 0 and 1, read at 160 | 0.818, 0.765 | §1.1 table |
| D partners at seeds 0 and 1, read at 160 | 0.946, 0.989 | §1.1 table |
| controls at seeds 0 and 1, read at 160 | 0.800, 0.765 | §1.1 table |
| paired A − D interval over 128 mazes at seed 1 | −0.484 [−0.773, −0.211] mine steps per episode, ±0.13 in ratio | PLAN §18.1 |
| unpriced partner's swing across reads | 0.75 to 1.00 | PLAN §15.3 |

Pooling the four pairs as replicates gives a per-run SD of about 0.048 in
ratio (4 df). §19.1's pair alone gives 0.078. The two seed-level A − D gaps
are −0.129 and −0.224, with SD 0.067. That is consistent with √2 × 0.048, but
it comes from the same four runs, so it is not an independent check. For planning I
use σ_gap = 0.07 for the SD of a per-seed difference between two runs, with
0.11 as the pessimistic case. These rest on four pairs. With 4 df the 95 % interval on
the SD spans a factor of about 4.8, and with §19.1's single df a factor of
about 70. The first seeds of any
sweep will re-estimate them.

The 128 mazes are the same at every seed. `eval_set` comes from
`derive_generator(0, "orgb", "eval")`, and the read takes its first 128
(`scripts/mz01_explore_trial.py:814-830`). Only the glyphs and the order seed
change with the trial seed. So the maze part of the variance does not shrink
with more seeds. An interval across runs needs a crossed bootstrap that
resamples seeds and, independently, mazes, applying the same maze draw to
every seed. A maze-only interval like §18.1's ignores the run variance. A
seed-only interval ignores the maze variance.

### 2.2 What the strict bar does with this noise

P(one read < 0.75) for a run with true ratio m and per-run SD s
(`power21.py`):

| m | s = 0.05 | 0.065 | 0.08 |
| --- | --- | --- | --- |
| 0.77 | 0.34 | 0.38 | 0.40 |
| 0.79 | 0.21 | 0.27 | 0.31 |
| 0.80 | 0.16 | 0.22 | 0.27 |
| 0.82 | 0.08 | 0.14 | 0.19 |

Suppose arm and control share a true ratio of 0.78 to 0.79. Then at one read
the arm crosses while the control does not with probability 0.17 to 0.23 per
seed, and at both of two seeds with probability 0.03 to 0.05. Four reads per
run raise both. The per-run rule mostly reports which of two runs near the
bar happened to cross. This is the multiple-looks problem APPROACHES §D names
("four looks per run multiply spurious AVOIDS").

The effect size is also a bottleneck, separate from the noise. The undosed
arm's two finals average 0.79, 0.04 above the bar. A perfectly measured 0.79
is still not AVOIDS. Precision can answer "does the price act, and does the
fork term add to it". It cannot by itself produce AVOIDS.

### 2.3 What a powered design needs

The unit is the run, one per seed per cell. There is one pre-specified read
per run (160). The mid reads are reported but carry no seed-level claim. Each
contrast is a per-seed paired difference with a crossed seed × maze bootstrap
interval. IQM across seeds goes beside the mean (DIAGNOSIS §3, `rliable`).

Two-sided α 0.05 and 80 % power. A simple gap such as A − D or A_fork − A_on
has σ_gap. The interaction (A − D) − (onA − onD) has about √2 σ_gap. The maze
component is ignored here, so these are lower bounds.

| seeds | 95 % half-width of a gap (σ 0.07 / 0.11) | smallest gap read at 80 % | smallest interaction read at 80 % |
| --- | --- | --- | --- |
| 4 | 0.11 / 0.18 | 0.15 / 0.23 | 0.21 / 0.32 |
| 8 | 0.06 / 0.09 | 0.08 / 0.13 | 0.11 / 0.18 |
| 12 | 0.04 / 0.07 | 0.06 / 0.10 | 0.09 / 0.14 |
| 16 | 0.04 / 0.06 | 0.05 / 0.08 | 0.07 / 0.12 |
| 20 | 0.03 / 0.05 | 0.05 / 0.07 | 0.07 / 0.10 |

The observed A − D gap (mean −0.18) would be resolved at 4 to 6 seeds. An
arm-versus-control difference of 0.10 needs about 6 to 12 seeds. An
interaction of 0.10 needs about 10 to 22. `power21.py` computes the gap
columns; the interaction column and these seed counts are the same formula
with √2 σ_gap, done by hand. DIAGNOSIS §3's 10 to 20 seeds per arm is
right for the interaction.

Costs. A fork run took 2.59 h wall for 160 steps at 56.8 s/step, reads and
reference included (`forkcf4b_D_s1_tx` 06:07:46Z to 08:42:54Z,
`forkcf4b_A_s1_tx` 06:42:14Z to 09:17:31Z, from `started_at` and the
`summary.json` mtime). At 60 s/step I use 2.75 h, plus 10 min to sync and
launch, so 2.9 h. An on-policy control takes about 1.2 h plus 10 min, so
1.35 h. The seed-0 and seed-1 fork pairs exist (§3, option 1), so a sweep over
seeds 0 to k−1 needs 2(k−2) new fork runs and 2k controls.

| seeds | new fork runs | controls | box-h | wall on four leases |
| --- | --- | --- | --- | --- |
| 0-7 | 12 | 16 | ~56 | ~14 h |
| 0-11 | 20 | 24 | ~90 | ~23 h |
| 0-15 | 28 | 32 | ~124 | ~31 h |
| 0-19 | 36 | 40 | ~158 | ~40 h |

Add 10 to 15 % for lease deaths (box 5 died at step 26 today, PLAN §16.2).
Running two 4B jobs per box would cut the wall time. The launcher refuses a
busy box (`docs/HANDOFF-2026-09-10.md`, box mechanics), and the trial's status
file path is hardcoded (RESULTS_LOG header). That makes it a launcher build,
not a knob. It is worth doing only if the sweep goes past 8 seeds.

Raising the read to 256 mazes (`MZ01_TRIAL_EVAL_N=256`, an existing knob that
reads all 256 held-out mazes, the current 128 plus the other 128) would cut the maze component by
√2 at about twice the read time. It changes the read, so it needs a dated
section and a DECISIONS row (CLAUDE.md rule 6 lists the held-out mazes). I
recommend it only if the interim crossed bootstrap shows the maze part
dominating the interval.

## 3. Next pre-registrations, ranked

Every option passes the construction test. Mines enter only through the
priced return or not at all. The on-policy runs are controls, never a test of
the price (D-21). Fork selection is §16's (D-22). The seed is the only new
variable in options 1 to 3, and option 4 needs a new knob with its docstring
line.

### Option 1 (top): the 2×2 at the world's price, seeds 0 to 7

- Cells per seed: `forkcf4b_A_s{k}_tx` and `forkcf4b_D_s{k}_tx` (§16 settings,
  undosed), plus matched on-policy controls `ctrl4b_A_s{k}_m3` and
  `ctrl4b_D_s{k}_m3`. These run `onpolicy_g6` with mass 0.3, current code and
  per-episode records, otherwise the `ctrl4b` settings.
- Seeds: the fork pairs at seeds 0 and 1 exist. Seed 0 uses the h320 pair's
  read at 160, which keeps `episodes`; the rule is in §21. New fork pairs run
  at seeds 2 to 7. Matched controls run at seeds 0 to 7. Extension to seeds 8
  to 11 happens on a width rule only.
- Order: the matched controls at seeds 0 and 1 go first. They are needed under
  every §19 and §20 outcome, including §20's curve, whose controls they would
  be.
- Reading: see §21 below. Per-run verdicts stay as now. Seed-level contrasts
  P_fork = A − D, P_on = onA − onD, F = A − onA and I = P_fork − P_on, at 160,
  with crossed seed × maze bootstrap intervals and §18.1's split into
  both-solved mazes and the rest.
- Cost: ~56 box-h, ~14 h on four leases. With the extension ~90 box-h,
  ~23 h.
- What it decides. Whether the undosed fork design lowers held-out mine steps
  against its partner across seeds, and how much of that is detours in solved
  episodes rather than capped ones. Whether the on-policy control's fall is
  specific to the price, read only as a property of the control. Reading it
  as a result for the on-policy recipe would be a test of the price that D-21
  closed, so that needs the user and a DECISIONS row (section 4). Whether the fork term
  adds anything to the priced on-policy update; if not, the fork-share line
  has no measured advantage at the world's price. It also gives the run-noise
  estimate that every later section needs.

### Option 2: the −20 dose across the sweep's seeds (conditional on §20)

- Gate, read at §20's finals: both `forkcf4b_A_s{0,1}_tx_p20` end at 160 with
  solve within 0.10 of the reference and capped share at most 0.13
  (reference 0.078 + 0.05), whatever their ratio. If either fails, this is not
  run. The dose then over-avoids at −20 on this design too, and option 4 is
  the remaining lever on the price's size.
- Runs: `forkcf4b_A_s{2..7}_tx_p20`, one per seed, sharing the seed's D
  partner and controls from option 1.
- Reading: dose contrast A20 − A10 per seed with the same crossed bootstrap and
  shape split, plus A20 − D and A20 − onA.
- Cost: 6 × 2.9 ≈ 17 box-h.
- Decides whether doubling the price lowers mine steps beyond −10 across seeds
  without a solve loss, that is, whether the dose can be the independent
  variable of a curve. If §20 credits both seeds, §20's own curve
  (−10, −15, −20, −25 on seeds 0 to 3) takes precedence as pre-registered.

### Option 3: the sweep at 320 steps (conditional on §19 and on the user)

- Gate: at §19's final, `_tx_h320`'s paired A − D gap at 320 is more negative
  than at 160, with solve within 0.10 and capped share at most 0.13 at 320.
  The user must also have decided that the horizon is what may move (section
  4).
- Runs: `forkcf4b_{A,D}_s{1,2,3}_tx_h320`, with `ctrl4b_{A,D}_s{1,2,3}_m3_h320`
  if the control comparison is to be read at 320.
- Cost: 6 fork runs × about 5.5 h ≈ 33 box-h, plus 6 controls × about 2.3 h ≈
  14 box-h.
- Decides whether more steps carry the undosed arm under the bar with its
  solve across seeds, which is APPROACHES §B item 10 and RESEARCH_PLAN §1.3's
  "ALIVE: extend" branch. A credit by §19's rule at seed 0 already sends the
  pair to seed 1 at 320 (§19). This option widens that to three seeds.

### Option 4: a solve anchor on the move token (needs a build)

- What: a KL term toward the warm start's move distribution, or ProRL's KL plus
  reference resets (APPROACHES §B item 10). It targets the failure every dose
  shows: the price buys fewer mines with solve (§16.7, §17.1, and §15.5's
  0.328 / 0.27).
- Construction test: the anchor is the mine-blind warm start, so it carries no
  mine information. The price still has to outweigh its pull toward the
  shortest, mined route. It needs a new knob (for example `MZ01_KL_COEF`) with
  a docstring line in `scripts/mz01_explore_trial.py`, tests, and an Opus
  critic before any launch. No such knob exists today: grepping the trainer
  and the trial for KL turns up nothing.
- Runs, once built: `forkcf4b_{A,D}_s{0,1}_tx_p30_kl`, about 4 × 3.2 ≈
  13 box-h. The extra reference forward pass is a guess.
- Ranked last. The build comes first, and what counts as success depends on
  the user's decision about the band.

### Considered and dropped

- A sign-flipped price (`MZ01_SCORE_MINE=+10`; `_refuse_mine_dose` accepts any
  finite value on cell A, `src/calibration/maze_orgb.py:3097`). Each mine
  landing would pay +10 against −2 a step, which opens a mine-farming route of
  the wall exploit's kind (RESULTS_LOG "What the log says" item 3). That
  confounds the control. APPROACHES §D's shuffled-price control needs a build.
- More fork knobs (two forks per episode, the cap at 1.0, the std normaliser)
  were run or rejected on record (PLAN §15.5, §15.6, §13.1).
- Idling boxes until all four finals. That saves nothing, because §21's first
  four jobs are needed under every outcome, and it costs about 1 box-h idle.

## 4. Decisions the user owns (flagged, not decided)

1. Band or horizon (DECISIONS "Decisions still open", first item; FAILURES
   A7). The record now has both cases. Doses at −30 and above reach low ratios
   only with solve loss (the band question). The undosed arm sits at 0.77 to
   0.82 at 160 (the horizon question, which §19 tests at seed 0). Options 3
   and 4 depend on this.
2. Whether a seed-level interval can stand in for the per-run AVOIDS of
   RESEARCH_PLAN §1.4 ("AVOIDS on two seeds with the control not AVOIDS on
   either"). §21 as drafted does not change §1.4. It reports seed-level
   contrasts as a price effect, never as AVOIDS. Section 2.2 shows the per-run
   gate is weak with the control near the bar. Changing §1.4 needs a
   DECISIONS row.
3. D-21's letter. CLAUDE.md rule 2 says "no on-policy-only recipe on the nav48
   navigators". The matched controls are on-policy runs. The precedent is
   `ctrl4b_s0`, launched after D-21 as "the pre-registered CONTROL ... not a
   test of the price in itself (D-21 stands)" (`PLAN-exit-priced-at-cap.md`
   §8, 19:30Z), with `ctrl4b_s1` registered as its second seed (19:45Z). That
   precedent is a cell-A control. A cell-D on-policy control, and any reading
   of P_on as "the price acts on-policy", go further. The draft §21 therefore
   reads P_on only as a property of the control and credits nothing from it.
   Whether P_on may be read as a result needs the user and a DECISIONS row.
4. RESEARCH_PLAN §1.4's kill criterion: "if after the third priced route no
   arm reads AVOIDS while the gradient is live and the policy intact ... stop
   Phase 1". The record names two routes closed (twins, wall). Whether the
   dose arms' capped episodes count as a third, and whether the fork-share
   line is past the point §1.4 meant, is for the user.
5. If option 1 finds P_on below zero with a both-solved component, that would
   bear on D-21's premise at 4B. Under the draft it is reported, not
   credited, and any change to D-21 is the user's.

## 5. Draft §21 (for option 1)

The text below is written to be committed before §20's first final (about
11:35Z). Fill in the timestamp. The lead must state which reads of
`forkcf4b_A_s0_tx_h320` and `forkcf4b_D_s0_tx_h320` it had seen when it wrote
this. Both reads at 160 were in `results/` by about 09:39Z. The seed-0 rule below picks the run by whether it has
`episodes` records, not by value.

---

## 21. The fork term on and off, at the world's price, seeds 0 to 7, pre-registered (2026-10-04 HH:MMZ / HH:MM IST)

Written before the finals of §19 (`forkcf4b_A_s0_tx_h320`,
`forkcf4b_D_s0_tx_h320`) and §20 (`forkcf4b_A_s0_tx_p20`,
`forkcf4b_A_s1_tx_p20`), and before any launch of the runs it queues. It
governs each box as §19's and §20's runs end, under the precedence below.

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

---

## Sources for every number above

- PLAN §6, §8.4, §15.1, §15.3, §16, §16.3, §16.6, §16.7, §17.1, §18, §18.1,
  §19, §19.1, §20: `PLAN-fork-counterfactual.md`.
- Control files: `results/ctrl4b_s{0,1}/onpolicy_g6.json` (`config`, `eval`,
  `eval.competence`, `started_at`), `results/ctrl4b_s{0,1}/summary.json`
  (`verdict`, `seconds_per_step`, `move_mass_coef`),
  `results/ctrl4b_s{0,1}/reference.json`.
- Arm and partner finals: `results/forkcf4b_{A,D}_s{0,1}_tx/fork_cf_g6.json`
  and `reference.json`. Durations come from `started_at` and the
  `summary.json` mtime.
- Control registration: `PLAN-exit-priced-at-cap.md` §8 (19:30Z, 19:45Z).
- Coverage 0.008: `PLAN-exit-priced-at-cap.md` §8 (19:45Z).
- Read construction: `scripts/mz01_explore_trial.py:814-830`. Dose refusal:
  `src/calibration/maze_orgb.py:3097-3110`.
- Statistics asked for: `record/DIAGNOSIS-2026-09-09.md` §3,
  `docs/research/APPROACHES.md` §D.
- Phase 1 exit and kill criterion: `docs/planning/RESEARCH_PLAN.md` §1.3,
  §1.4.
- Planning arithmetic: `scratchpad/power21.py` (this session).
