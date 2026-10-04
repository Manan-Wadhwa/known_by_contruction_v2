# §16 mechanism read 2 on `forkcf4b_D_s0_tx` (2026-10-04, Opus subagent, verbatim)

The subagent could not write files; the lead saved this from its report. The
scratch scripts beside it (`pool.py`, `forksim7.py`/`.out`, `power.py`/`.out`)
are verbatim; their paths point at the session scratchpad. The subagent states
that the results-verifier checked every number and its corrections are applied.

## Answer: (c), not (a); (b) is consistent with the trace but not shown

The cross-fit draws fresh continuations, by code and by test. The trace rules out a replay: 4 forks tie at stage 2, and that cannot happen if stage 2 replays stage 1. A real route effect fits the trace, but the interval contains 0, and the trace cannot tell the stand-in's route-effect regimes apart. As written, the read cannot separate (a) from (b) at any sample size. At this kept rate it cannot fix a sign within 160 steps either.

## 1. Stage 2 does not depend on stage 1 beyond the kept list

- RNG. Each step seeds one `fork_gen` from the run seed and the step (`src/calibration/maze_orgb.py:3767-3768`). Stage 2 runs `sample_fork_continuations(..., forks2, **cont_kw)` on that same generator after stage 1's draws (`:3788-3798`). It therefore gets new draws, not repeats. The sampler draws on the CPU from that generator (`:1632`).
- State. Each call builds new `_TextEp` objects at the fork cell (`:2341-2374`). The prompt is the maze plus the position (`:1812-1838`). Sampling runs under `no_grad` (`:2376`). Nothing in stage 1 writes to the fork dicts.
- Selection and credit. `crossfit_keep` reads only `fork`, `move`, `return` and `taken` (`:2494-2521`). Only stage 2's `fork_terms` reaches the loss and `fork_adv_taken_mean` (`:3797-3798`, `:2571`).
- Tests. `tests/test_maze_orgb.py:7137` asserts at `:7186-7188` that stage 2 starts from `fork_gen`'s state after stage 1. The BUILD record §4 shows that a restore or a reseed fails it. That test and the ones at `:7081` and `:7207` passed today (4 passed). `git diff 5fda1de HEAD` shows no change to `maze_orgb.py`, the trial script or the tests.
- Data. Under a replay, every credited fork has unequal returns, because stage 1 kept it for a difference. In the trace, 4 of 17 credited forks tie at stage 2: steps 1, 5, 7 and 16 each have `n_forks` 1 and `fork_share_nonzero` 0. In the lost run's steps 0-26, 4 of 15 tie (`results/_lost_2026-10-04_forkcf4b_D_s0_tx/fork_cf_g6.json`). In the stand-in, every replay scenario gives a tie share of 0.000 and beats the taken move at 1.000 of forks.

## 2. What sign stage-2 A(taken) should take on cell D

Without a leak, the stage-2 mean estimates the true A(taken) at the forks stage 1 kept. Stage 1 keeps a fork when the taken move's continuation lost, either by chance or because the move really is worse. The design's stand-in can only come out positive without a leak, for two reasons: its episodes never waste a step (`docs/research/2026-10-04-fork-share/forksim3.py`, where `episode()` sets `NOSTALL`); its stall rate is 0.2 at every cell.

`forksim7.py` uses the trainer's own `episode_forks`, `crossfit_keep` and `fork_terms`. Its policy is peaked: one habit move per cell. It ran on 40 × 256 mazes at cell D (`forksim7.out`):

| scenario | solve | kept share | mean A2 | A2 when taken move is shortest / wasted | beaten at s2 | tie at s2 | SD |
|---|---|---|---|---|---|---|---|
| even noise 1% | 1.000 | 0.038 | +0.64 | +0.75 / −1.75 | 0.10 | 0.38 | 1.06 |
| even noise 10% | 0.999 | 0.202 | +0.71 | +0.90 / −2.60 | 0.33 | 0.18 | 2.72 |
| sticking cells 2% | 0.960 | 0.047 | −0.29 | −0.27 / −2.22 | 0.34 | 0.18 | 11.33 |
| sticking cells 6% | 0.876 | 0.105 | −0.89 | −0.89 / −2.00 | 0.43 | 0.16 | 13.28 |
| off-path habits 1% | 0.903 | 0.052 | −6.23 | −5.28 / −13.57 | 0.43 | 0.26 | 10.30 |
| replay, even noise | 1.000 | 0.033 | −1.21 | | 1.000 | 0.000 | 0.86 |
| replay, sticking 6% | 0.873 | 0.104 | −10.67 | | 1.000 | 0.000 | 11.64 |
| real, steps 0-39 | | 0.018 | −2.39 | | ~0.41-0.53 | 0.24 | ~6.3-7.2 |

A "sticking cell" is one where the policy bumps a wall with probability 0.6, and three bumps in a row end the roll-out. A wasted taken move gives a negative A2 in every scenario. A taken move on a shortest path gives a positive A2 only when failures are spread evenly. When some cells make the policy fail, or some habits are off-path, the mean falls between +0.07 and −8.8 with no leak.

The trace has features that point away from even noise: the three forks at steps 4 and 20 account for −42.67 of the −40.67 sum, and their return gaps are failure-sized; six of 24 episodes went unsolved at step 19 (`capped_rate` 0.25); the real kept rate is 0.425 forks a step, against the design's 7.2 (`forksim4.out`: 0.30 an episode × 24). The trace's tie share (0.24) and beaten share fit the sticking-cell runs and the off-path-habit runs about equally. A negative true mean on cell D would be route credit by definition, since cell D's return has no mine term. The measured mean, though, is not separated from 0.

## 3. Power

The trace does not record per-fork values. It fixes the 4 tied forks, the −40.67 total (mean −2.39 over 17 forks in 12 steps), and the −42.67 at steps 4 and 20, so the other 14 forks average +0.14. The fork-level values that fit the trace give an SD of about 6.3 to 7.2, a median of 0 or −1, and 7 to 9 forks beaten. The step-level SD over the 12 steps is 6.97 (sample) or 6.67 (population).

95% intervals: step-cluster bootstrap over the 40 steps [−6.58, +0.36], P(>0) 0.086; t interval [−5.66, +0.87] at SD 6.35, [−6.11, +1.32] at SD 7.23. Zero is inside both. The design's +1.07 is inside the t interval at SD 7.23. The even-noise stand-in's +0.64 is inside every interval.

Forks needed at 0.425 forks a step (lower bounds):

| true \|μ\| | 95% CI excludes 0 | 80% power |
|---|---|---|
| 1.07 | 135-175 forks (318-412 steps) | 276-358 forks (650-842 steps) |
| 2.0 | about 39 forks (91 steps) at SD 6.35 | about 79 forks (186 steps) at SD 6.35 |

At 80: about 34 forks, SE 1.09-1.24; only |μ| above about 2.1-2.4 can be read. At 160: about 68 forks, SE 0.77-0.88; only |μ| above about 1.5-1.7 can be read.

## 4. The rule as written cannot be read for this partner

It is a sign test with no tolerance (the critic flagged this while expecting about 280 forks over steps 0-39; the run has 17). A negative sign does not separate (a) from (b). 160 steps cannot fix a sign of the size the design predicted. The tie share at stage 2 is a replay signature that does separate them, and the trace already carries it. The proposed amendment is in `PLAN-fork-counterfactual.md` §16.4. The gate's false-alarm rate is low: in the stand-in, the tie share without a leak runs 0.156-0.384; at 0.156, the chance of zero ties in 30 forks is 0.844^30 ≈ 0.006.

## Close any running arm?

No. Nothing in the build voids a step. `forkcf4b_D_s0_tx` is the partner §16 requires. `forkcf4b_A_s0_tx` and `_tx_p30` stay uncredited until the partner reads at the same step. Over steps 0-39 their own pooled A(taken) is −5.50 over 113 forks and −7.64 over 103, which is what the price predicts. The stand-in numbers are a model, not run evidence.
