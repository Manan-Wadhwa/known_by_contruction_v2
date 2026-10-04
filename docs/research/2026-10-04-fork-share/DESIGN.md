# Design: more of the fork term where the price can change a move (draft, 2026-10-04)

Design only. Nothing was built, run or committed. Run numbers come from `results/<label>/fork_cf_g6.json` as pulled at 2026-10-04 00:02Z. The arms are live, so step ranges are given. Stand-in numbers come from the CPU scripts `forksim2-6.py` in this scratchpad, with their outputs in `*.out`. They are a model, not evidence, and they need the results-verifier before any of them enters a PLAN section.

## 1. Diagnosis

**Traces.** These are `history` means.

| label (steps) | `n_forks` | `fork_adv_abs_mean` | `fork_share_nonzero` | `fork_share_detour_better` |
| --- | --- | --- | --- | --- |
| `forkcf4b_A_s0_g` (0-106) | 24.0 | 4.26 | 0.905 | 0.051 |
| `forkcf4b_A_s1_g` (0-77) | 24.0 | 4.16 | 0.909 | 0.031 |
| `forkcf4b_A_s0_g10` (0-109) | 24.0 | 4.00 | 0.905 | 0.049 |
| `forkcf4b_A_s0_f2` (0-90) | 47.1 | 4.15 | 0.913 | 0.050 |
| `forkcf4b_A_s0_p30` (0-116) | 23.9 | 9.71 | 0.866 | 0.050 |
| `forkcf4b_A_s1_p30` (0-106) | 24.0 | 8.59 | 0.869 | 0.048 |
| `forkcf4b_D_s0_g_rerun` (0-119) | 23.9 | 2.71 | 0.852 | 0.003 |

At the read at 80, the ratio `eval_at["80"].mine_step_rate` / `reference.json` `eval.mine_step_rate` is 0.925 (`_g10`), 0.904 (`_f2`), 0.839 (`A_s0_p30`) and 0.957 (`A_s1_p30`).

**The stand-in.** It calls the trainer's own `episode_forks` (`src/calibration/maze_orgb.py:2179`) on 256 mazes from `maze2.generate_set(256, n=9, seed_range=_TRAIN_SEED_RANGE)`. The policy is mine-blind. Each step it takes a uniformly chosen shortest move, or with probability 0.2 it wastes the step. It reads mines only to price landings. It reproduces the traces. Detour better and abs A come out at 0.051 and 4.12 at -10 (real 0.031-0.051, 4.0-4.3), 0.058 and 7.81 at -30 (real 0.048-0.050, 8.6-9.7), and 0.012 and 2.61 on cell D (real 0.003-0.008, 2.4-2.7).

**Why the share is low.**

1. The mine signal rarely arrives alone. Of continuation pairs inside a fork, 0.621 differ in steps only, 0.261 in mines and steps, 0.031 in mines only, and 0.087 tie (`forksim5.out`).
2. Where mine counts differ, the taken move is already the cleanest. That is 0.33 of forks, and in 0.791 of them the taken move's continuation has the fewest mines (`forksim5.out`). The price then lowers the logit of a move the policy does not play. Advantages are centred (`fork_advantages`, `maze_orgb.py:2411`), so the term at `maze_orgb.py:3737` is the sum of A(a) times the gradient of move a's logit. Its size follows A, not the move's probability. These forks therefore take up the capped step (`maze_orgb.py:3757`, cap 0.3) without changing a decision.
3. The dose scales size, not the set of forks. In the stand-in, the mine term is 0.44 of summed A² at -10 and 0.82 at -30. The decision share is the mine A² at forks where a cleaner alternative beat the taken move. It is only 0.094 and 0.172 (`forksim3_stall02.out`, "uniform 1/ep"). Detour better counts forks and moves only when the price flips a ranking. At -10 a mine already outweighs five steps, so -30 flips few: 0.051 becomes 0.058 in the stand-in, and 0.051 becomes 0.050 on real seed 0.
4. Half the pool cannot contribute. 128 of 256 training mazes have no mine on any shortest path (`maze2.py:1314`, target 0.5 at `:1344`). The stand-in finds no detour-better fork in them (`forksim2.out`, the `mos=False` rows).
5. The uniform draw (`maze_orgb.py:2277`, about 7 forks per episode) mostly misses tie forks. At a tie fork, two or more moves lead to cells at the same bfs distance to the goal, and the blind policy picked one of two equal routes. Tie forks are 26% of forks and hold 0.50 of the detour-better events (0.121 × 1.84 against 0.063 × 7.11 per episode, `forksim3_stall02.out`).

Route length is not the binding dilution. At -30 it is 0.18 of A², and the decision share is still 0.17. Removing all non-mine variance would lift the decision share to at most 0.094/0.444 = 0.21 at -10 and 0.172/0.823 = 0.21 at -30.

## 2. Candidates, ranked

The decision-share and detour-better columns are stand-in values at -10 / -30. One fork unit is today's 24 forks, which take 29-40 s of `fork_sample_seconds`. The rest of a step takes 26-35 s.

| rank | design | decision share | detour better | credited forks/step | fork units | s/step |
| --- | --- | --- | --- | --- | --- | --- |
| now | uniform, 1 per episode | 0.094 / 0.172 | 0.051 / 0.058 | 24 | 1.0 | 55-75 |
| 1 | tie pick + cross-fit | 0.388 / 0.541 | 0.263 / 0.301 | ~7 | ~1.3 | ~65-90 |
| 2 | tie pick alone | 0.150 / 0.263 | 0.098 / 0.098 | 24 | 1.0 | 55-75 |
| 3 | cross-fit, uniform pick | 0.330 / 0.523 | 0.232 / 0.247 | ~4 | ~1.15 | ~65-80 |
| no | geometric length baseline | at most 0.21 | n/a | 24 | 1.0 | 55-75 |
| no | keep forks whose return differs beyond geometric length | 0.107 / 0.207 | 0.075 / 0.084 | ~144 | ~7 | >200 |
| no | 2 continuations per move | 0.054 / 0.131 | 0.054 / 0.077 | 24 | 2.0 | ~85 |
| no | more forks per episode | flat | 0.050 (real `_f2`) | 47 | 2.0 | 83 |

### 1. Tie pick with cross-fitted selection

Knobs. `MZ01_FORK_PICK=tie` (default `uniform`) and `MZ01_FORK_SELECT=crossfit` (default `none`).

Code. `episode_forks` (`maze_orgb.py:2179`) marks a fork `tie` when two or more of its moves (`:2235`) reach the minimum of `maze2.bfs_dist(maze, maze.goal)` over its moves. `plan_forks` (`:2253`) draws among tie forks when the episode has any and falls back to all forks. Stage 1 is today's `sample_fork_continuations` call (`:3640`). A new pure function `crossfit_keep(forks, conts)` keeps a fork when some forced move out-returned the taken move by more than `FORK_ADVANTAGE_TOL`. It drops forks whose taken move has no continuation. Stage 2 draws fresh continuations at the kept forks from `fork_gen` (`:3636`). `fork_terms` (`:2438`) and the loss (`:3716-3751`) see stage 2 only.

Why. Stage 1 finds forks where an alternative won. Stage 2 credits them with returns independent of that choice, so noise cannot choose its own sign. On cell D, the mean stage-2 A(taken) on kept forks is +1.07, which reinforces the taken move. Crediting from stage 1 (-1.28) would push the policy off its path with no price at all. At -30 the stage-2 value is -1.47, and the price causes that push (`forksim4.out`).

Construction test. The tie flag reads walls only. `bfs_dist` (`maze2.py:1050`) walks `neighbours` and `is_open` (`maze2.py:885`, `:869`), and mines are open cells. The selection reads continuation `return`, `move` and the fork's `taken`. A mine reaches it only through the priced return, and on cell D nothing about mines reaches it. It is the same kind of rule as dynamic sampling, which passes in `docs/research/APPROACHES.md` §B row 3. The forced counterfactual still supplies the detour, so the recipe is not on-policy-only (D-21). Two variants would leak, and the tests forbid both. One reads `mine_steps` (the diagnostic at `maze_orgb.py:2479-2485`). The other subtracts the continuation's own step and goal terms from R, which recovers the mine count exactly under a shortest-path policy.

Tests.
- `_MineBlind` (`tests/test_maze_orgb.py:5814`) gives the same tie flags and the same draw. A moved mine changes neither.
- Hand-built mazes: two equal routes make a tie fork, and a shortest route beside a dead end does not. The pick falls back when a path has no tie fork.
- `crossfit_keep` keeps the same set when every continuation field except `return` and `move` raises.
- On `FORKPRICE` with a stub policy, a taken mine move is kept with A(detour) > 0, and a taken detour is dropped.
- Rows come from stage 2. Their advantages equal `fork_advantages` of the stage-2 returns.
- On cell D with a mine moved, the same seeds keep the same set.
- At the defaults, the existing fork arm's trace and adapter are byte-identical.
- Each knob is parsed, documented and refused without `MZ01_FORK_CF`. Each enters the fingerprint only off its default.

Trace fields, added only off default: `fork_share_tie`, `fork_pool_n`, `fork_kept_share`, `fork_adv_taken_mean` (from returns), and `fork_share_detour_better_pool` (reads mines, feeds nothing).

Cost. About 24 stage-1 forks plus 7 stage-2 forks at about 1.2 s each (`_g` 28.8 s / 24.0, `_f2` 56.1 s / 47.1). That is about 65-90 s/step and 3-4 h for 160 steps. The second lockstep pass adds latency, which the build measures. Risk 1: 7 credited forks a step give a noisy direction. Risk 2: the stand-in breaks ties uniformly, and nav48's path may cross fewer tie forks.

### 2. Tie pick alone

This is half of design 1 at no extra cost. The decision share rises 1.6× and detour better 1.9×. On cell D, abs A falls from 2.61 to 2.36, so the term teaches less route-shortening. It ships with the first, second, seventh and eighth tests above.

### 3. Cross-fit with the uniform pick

Its share is close to design 1's. With about 4 credited forks a step it is too noisy.

### Rejected

The geometric baseline adds 2·(1 + bfs distance of the move's next cell) to each R. It depends on the action, so it is shaping and not a baseline. It refunds the first off-route steps of a detour, which moves the avoid threshold from d < 5m to d < 5m + 2, and it is capped at 0.21. The residual filter costs 7× for +14% at -10. Extra continuations and extra forks add count, not share.

## 3. Recommendation

Build both knobs in one worktree with a critic (§7). Before the build, add three entries:

- a `docs/DECISIONS.md` row saying that selection on the priced return of the policy's own continuations passes the construction test, and that selection on `mine_steps`, or on R minus its step and goal terms, fails;
- a line in `docs/CONSTRUCTION.md` §5.8;
- an APPROACHES §B row.

Docstring lines, beside `MZ01_FORK_TEMP` (`scripts/mz01_explore_trial.py:161`):

```
  MZ01_FORK_PICK     "uniform" (default) draws the fork uniformly among the
                     episode's forks; "tie" draws among its TIE forks (two or
                     more moves reach the least bfs distance to the goal,
                     walls only) and falls back to all forks.
  MZ01_FORK_SELECT   "none" (default); "crossfit" keeps a fork only when a
                     forced move's continuation out-returned the taken move's,
                     then credits it from FRESH continuations. Reads returns only.
```

**Draft §16, dated before the first launch.** These runs go on the boxes §15 frees. All use 4B (`nav_nav48_4b`, `orgb_4b/warm_start.pt`), arm `fork_cf_g6`, 160 steps, reads 40/80/120/160, mask on, exit priced at cap, episode credit, mass 0.3, `MZ01_FORK_GRAD_MAX=0.3`, coef 1.0 and one fork per episode.

| label | change from `forkcf4b_A_s0_g` | seed |
| --- | --- | --- |
| `forkcf4b_A_s0_tx` | `MZ01_FORK_PICK=tie`, `MZ01_FORK_SELECT=crossfit` | 0 |
| `forkcf4b_D_s0_tx` | the same, cell D | 0 |
| `forkcf4b_A_s0_tx_p30` | the same, `MZ01_SCORE_MINE=-30` | 0 |
| `forkcf4b_A_s1_tx_p30` | the same at seed 1 | 1 |
| `forkcf4b_A_s0_t` | `MZ01_FORK_PICK=tie` only | 0 |

**Reading rule.** §6, §8.4, §9 and the trial's bars stand. The price gets credit at a valid read if the arm AVOIDS, its on-policy control does not, and `forkcf4b_D_s0_tx` does not at the same read. Held-out `mean_steps` is read beside mine/ep, and a mine-rate fall that tracks a route-length fall is not credited.

Two mechanism reads come first, over steps 0-39:

- `fork_share_detour_better` on the credited forks. At 0.15 or above on every cell-A `tx` arm, the selection works (the stand-in predicts 0.26-0.30). Below 0.10, the stand-in does not describe the real continuations, and the arm is a failed mechanism whatever its reads. In between, the reads stand and the mechanism is reported as weaker than modelled.
- Mean `fork_adv_taken_mean` on `forkcf4b_D_s0_tx`. A negative value means the selection leaks noise into the credit. That is a build question, and no arm is credited until it is answered.

`_t` against `forkcf4b_A_s0_g` isolates the pick. `_t`'s cell-D partner of record is `forkcf4b_D_s0_g_rerun`, which differs from it in the pick, so an AVOIDS on `_t` needs its own tie-pick cell-D run first. `_tx` against `_t` isolates the cross-fit. If both dose seeds avoid, the design goes to §11's response curve.
