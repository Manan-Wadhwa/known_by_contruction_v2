# BUILD: the tie pick and the cross-fitted selection (eighteenth pass)

2026-10-04. Built on `8d5a75b` in the worktree `.claude/worktrees/build-fork-share`
(branch `build/fork-share`). Nothing was launched, nothing under `results/` was
touched, and no PLAN file, `docs/RESULTS_LOG.md`, `HANDOFF.md`,
`docs/DECISIONS.md` or `scripts/box/` file was edited.

The design is `docs/research/2026-10-04-fork-share/DESIGN.md` §2, designs 1 and
2. It adds two knobs to the fork arm:

- `MZ01_FORK_PICK`: `uniform` (default) or `tie`.
- `MZ01_FORK_SELECT`: `none` (default) or `crossfit`.

At the defaults the fork arm is the pre-build arm to the bit.
`test_the_fork_arm_at_the_default_pick_and_selection_is_the_pre_build_arm`
checks the trace, the `config["fork"]` block and the adapter against shas
captured on `8d5a75b` before any edit.

## 1. What changed, by file and line

### `src/calibration/maze_orgb.py`

| line | what |
| --- | --- |
| 194-195 | `__all__` gains `FORK_PICKS`, `FORK_SELECTS`, `FORK_SHARE_TRACE_FIELDS` and `crossfit_keep`. |
| 2075-2079 | The construction-test comment block names the tie pick and `crossfit_keep`, and what each may read. |
| 2167-2193 | `FORK_PICKS = ("uniform", "tie")`, `FORK_SELECTS = ("none", "crossfit")` (the first entry is the default), and `FORK_SHARE_TRACE_FIELDS`, the five trace fields with one line each on what they count. |
| 2231-2237 | `episode_forks` docstring: the `tie` key, and why the flag is walls only. |
| 2268, 2274-2278, 2290 | `episode_forks` computes `bfs_dist(maze, maze.goal)` once per episode. A fork is `tie` when two or more of its non-reverse valid moves reach the least distance to the goal over those moves. A next cell that cannot reach the goal counts as infinitely far. |
| 2295, 2309-2318, 2324-2325 | `plan_forks(..., pick="uniform")`. Under `tie` the pool is the episode's tie forks, or all its forks when it has none. The docstring states the RNG use (section 2 below). |
| 2492-2521 | `crossfit_keep(forks, conts) -> [j]`, the stage-1 selection. A pure function. |
| 2551, 2566-2569, 2581-2582 | `fork_terms` adds `fork_adv_taken_mean` to its stats: the mean A(taken) over the forks whose taken move has a continuation, taken from the same advantages the loss uses. The trace reads it only off the defaults. |
| 2993-2999, 3021-3027 | `_refuse_fork_knobs` refuses a pick or selection outside `FORK_PICKS` / `FORK_SELECTS`, and refuses either one off its default when `fork_cf` is off. The second case uses the existing `fork_cf is off ...` message. |
| 3131 | `train_orgb_cell(..., fork_pick="uniform", fork_select="none", ...)`. |
| 3307-3316 | The trainer docstring's "WHICH FORKS" paragraph. |
| 3437-3444 | Resolution and refusal at the top of the trainer. `fork_share` is true when the term is on and either knob is off its default. |
| 3505-3506 | `history` gains `FORK_SHARE_TRACE_FIELDS` only when `fork_share` is true. |
| 3589-3596 | The fingerprint gains `fork_pick` and `fork_select`, each only off its default and only with the term on. |
| 3765-3789 | The step. The continuation arguments move into `cont_kw` (the stage-1 call is unchanged), `plan_forks` gets `pick=`, and under `crossfit` the trainer runs `crossfit_keep`, a second `sample_fork_continuations` at the kept forks, and `fork_terms` on stage 2, which replaces `fork_rows` and `fork_stats`. Stage 1's stats are kept as `pool_stats`. |
| 4013-4029 | Trace appends for the five fields, only when `fork_share` is true. |
| 4151-4155 | The log line adds `pool N tie x kept y`, only when `fork_share` is true. |
| 4399-4403 | `config["fork"]` gains `pick` and `select`, each only off its default. |

### `scripts/mz01_explore_trial.py`

| line | what |
| --- | --- |
| 164-170 | The env list gains the design's two docstring entries verbatim, after `MZ01_FORK_TEMP`. |
| 296-300 | `FORK_PICK` and `FORK_SELECT` are parsed. Empty means the default. A misspelling reaches the trainer, which refuses it before the first episode. |
| 940-945 | The arm's `cfg` gains `fork_pick` / `fork_select` only when the arm runs the fork term and the knob is off its default. A default arm's row is unchanged. |
| 973-974 | The banner prints `fork share: pick ..., select ...` when either knob is off its default. |
| 1075-1076 | Both knobs go to `train_orgb_cell` inside the `arm_fork` dict, the same as the other fork knobs. |

### `tests/test_maze_orgb.py` (8 new tests, 6900-7320)

| line | test |
| --- | --- |
| 6918-6958 | Fixtures. `TIEMIXED` (one non-tie fork and three tie forks on the path) and `TIEMIXED_MOVED`. `TIEFORK` (two equal routes round one wall). `DEADEND` (a shortest route beside a dead end). `TIEMINES` and `TIEMINES_MOVED` for the end-to-end runs. |
| 6961, 6972 | `_mixed_walker` (bumps, shortest moves and generator-chosen moves, reading walls and the goal only) and `_share_fixture_train`. |
| 6993 | `test_the_tie_flag_and_the_tie_draw_read_walls_only`. The flags are `[F, T, T, T]`. `_MineBlind` and the moved-mine maze give the same forks and flags, and over 20 seeds the same tie draw. The uniform draw does reach the non-tie fork. The generator state after `plan_forks` equals one `randperm(3)` under `tie` and one `randperm(4)` under `uniform`. |
| 7034 | `test_a_tie_fork_is_two_equal_routes_and_the_tie_pick_falls_back`. `TIEFORK`'s start is a tie fork. `DEADEND`'s start is a fork and not a tie, and the tie pick falls back to it. |
| 7056, 7081 | `_Only` (a dict whose every key outside an allowed set raises, including on iteration, `keys`, `items` and `in`) and `test_crossfit_keep_reads_the_return_and_the_move_and_nothing_else`. Seven forks: kept, taken-is-best, bump, equal returns, within-tolerance, `taken=None`, three moves. The kept set is `[0, 6]` on plain dicts, and again when continuations allow only `fork`/`move`/`return` and forks allow only `taken`. Because `mine_steps`, `steps`, `landings` and `solved` raise, neither the mine count nor R minus its step and goal terms can be read. |
| 7118 | `test_crossfit_keeps_a_taken_mine_move_and_drops_a_taken_detour`. On `FORKPRICE` with a position-table stub: taken `down` (the mine) is kept and fresh stage-2 rows give A(right) > 0 > A(down). Taken `right` (the detour) is dropped. |
| 7137 | `test_the_credited_rows_are_stage_two_and_their_advantages_its_returns`. Spies on the trainer under `tie` + `crossfit`. Each step runs, in order, sample, terms, keep, sample, terms. Stage 2 runs at exactly `[forks[j] for j in kept]` (the same objects) on the same fork generator. The last `fork_terms` call gets stage 2's continuations (`is`). Each row's advantages equal `fork_advantages` of its stage-2 returns. `n_forks`, `n_continuations`, `fork_adv_taken_mean`, `fork_kept_share` and `fork_pool_n` in the trace match. `_fork_move_logp` scored exactly the stage-2 rows, and at least one fork was kept. |
| 7193 | `test_on_cell_d_a_moved_mine_keeps_the_same_forks`. Cell D, `crossfit` and `tie` + `crossfit`, over `TIEMINES` and `TIEMINES_MOVED`. The kept sets are equal step by step and non-empty, the share fields are equal, and `mine_step_rate` differs, so the moved mine is somewhere the walker steps. |
| 7226-7240 | `_PRE_SHARE_FORK_TRACE_SHA` / `_PRE_SHARE_FORK_ADAPTER_SHA`, captured on `8d5a75b`, and `_fork_trace_sha`. That sha covers every trace field but the four clocks, plus `config["fork"]`. |
| 7242 | `test_the_fork_arm_at_the_default_pick_and_selection_is_the_pre_build_arm`, with the knobs omitted and passed explicitly at their defaults. |
| 7255 | `test_the_pick_and_the_selection_are_named_knobs_refused_without_the_term`. Parsed, defaulted, and the design's lines present in the trial's `__doc__`. Each knob alone without `fork_cf` raises `fork_cf is off`, and a misspelling raises `must be one of`. Under `tie` alone the five fields are present, `fork_kept_share` is `None`, `fork_pool_n == n_forks` and the pool's detour-better equals the credited one. End to end on checkpoints, `fork_pick` / `fork_select` are absent at the default and present off it, `_checkpoint_mismatch` fires both ways, and a default resume of a `tie` checkpoint is refused naming `fork_pick`. |

## 2. The RNG and the fingerprint

The fork draw and the continuations already use their own stream,
`fork_gen = Generator().manual_seed((seed * 1_000_003 + step) & 0x7FFF_FFFF)`.
The run's `gen` is never advanced by either knob, so the episodes of a step are
the same under every pick and selection at the same seed.

- `plan_forks`: under either pick, one `torch.randperm(len(pool))` per episode
  that has a fork, and none for an episode without one. Under `uniform` the pool
  is all forks, which is the pre-build call. Under `tie` the pool is the tie forks
  (all forks if there are none), usually a shorter permutation. From the first
  such episode on, `fork_gen` therefore sits at a different position from a
  `uniform` run's, and every later draw that step differs. The test checks the
  state against a bare `randperm(3)` / `randperm(4)`. With `MZ01_FORK_PER_EP > 1`
  the tie pick takes up to that many tie forks and does not top up from the
  others.
- `crossfit`: stage 1 is the pre-build continuation sampling. `crossfit_keep`
  draws nothing. Stage 2 calls `sample_fork_continuations` again on the same
  `fork_gen`, after stage 1's draws, so its continuations are fresh draws from
  the policy that stage 1's returns cannot have chosen. With nothing kept, stage 2
  is called on an empty list. It then draws nothing and returns nothing, the
  step has no fork term, and `fork_grad_norm_raw` / `fork_grad_norm` are `None`
  for that step.
- Fingerprint: `fork_pick` and `fork_select` are added only off their defaults,
  and only with the term on. A default fork run's fingerprint is byte for byte
  the pre-build one, so its checkpoints resume. The key's presence is the
  mismatch in both directions. `CKPT_SCHEMA` is unchanged, because the trace
  grows fields only on runs whose fingerprint also has the new keys.

## 3. The trace fields (only when either knob is off its default)

| key | value |
| --- | --- |
| `fork_share_tie` | share of the stage-1 (pool) forks flagged tie |
| `fork_pool_n` | forks branched at stage 1 |
| `fork_kept_share` | kept / pool under `crossfit`; `None` under `none` |
| `fork_adv_taken_mean` | mean A(taken) over the credited forks (stage 2 under `crossfit`), from their returns |
| `fork_share_detour_better_pool` | `fork_share_detour_better` on the pool. It reads `mine_steps` and feeds nothing, like the existing diagnostic. |

Under `crossfit` the existing fields `n_forks`, `n_continuations`,
`fork_adv_abs_mean`, `fork_share_nonzero` and `fork_share_detour_better` are
computed on the credited stage-2 forks, as the design asks for detour-better.
`fork_sample_seconds` covers both stages.

## 4. Tests

The tests were run from the worktree root with
`/home/manan/Projects/files/digital_minds-maze/.venv/bin/python -m pytest tests/test_maze_orgb.py -q -x`.
They import `calibration.maze_orgb` from
`.claude/worktrees/build-fork-share/src/calibration/maze_orgb.py` (printed).

- Before: 139 passed, 1 skipped (the caller's count on `8d5a75b`).
- After: `147 passed, 1 skipped in 94.30s`.
- Also run, unchanged: `tests/test_maze2.py tests/test_maze_rl.py` gave 102
  passed, and `tests/test_state_contract.py tests/test_navigator.py
  tests/test_chat_template.py` gave 47 passed.
- `python3 scripts/check_decisions.py`: 21 decisions, 51 references, 0 problems.

Each guard was mutation-checked by hand and the source restored after each
run. The relevant test failed every time:

| mutation | test that failed |
| --- | --- |
| `crossfit_keep` reads `mine_steps` (multiplied by 0) | `..._reads_the_return_and_the_move_and_nothing_else` |
| `crossfit_keep` uses R minus 2 × `steps` | the same |
| the tie flag adds `maze.is_mine(next cell)` | `test_the_tie_flag_and_the_tie_draw_read_walls_only` |
| the loss credits stage 1's rows | `test_the_credited_rows_are_stage_two_...` |

## 5. Expected cost (from the design, not measured here)

The design's §2 table gives these. Nothing here was run on a GPU.

- `tie` alone: no extra continuations, 24 credited forks a step, 55-75 s/step
  (today's arm).
- `tie` + `crossfit`: about 24 stage-1 forks plus about 7 stage-2 forks at about
  1.2 s each (`_g` 28.8 s / 24.0, `_f2` 56.1 s / 47.1), about 1.3 fork units.
  That is about 65-90 s/step, or 3-4 h for 160 steps. The second lockstep pass
  adds latency that the design says the build should measure. The trace does
  not carry a separate stage-2 clock, so read it as `fork_sample_seconds` on a
  `_tx` arm minus the same field on `forkcf4b_A_s0_g` (29-40 s), with
  `fork_pool_n` near 24 on both.
- The trial's own booking estimate (`scripts/mz01_explore_trial.py:726`, 3×
  for any fork arm) is unchanged and still covers 1.3 units.

## 6. Deviations and open points

- `crossfit_keep` also reads each continuation's `fork` index. The launch
  writes it (`_launch_continuations`) and it is the only way to group
  continuations by fork. It carries no information from the world. The raising
  test allows `fork`, `move` and `return` on continuations and `taken` on forks.
- The tie flag is computed for every fork at every setting, one `bfs_dist` per
  episode. It adds a `tie` key to `episode_forks`' records, which nothing at the
  defaults reads. The draw, the trace and the adapter at the defaults are
  unchanged (sha test).
- `fork_kept_share` is `None` under `none` rather than 1.0, because no
  selection ran.
- The trial passes both knobs only to a fork arm, as it does the other fork
  knobs. `MZ01_FORK_PICK=tie` on a non-fork arm is inert at the trial level, and
  the trainer refuses it only if it reaches the trainer. All five §16 runs use
  `fork_cf_g6`, so the knobs reach the trainer.
- The two design risks stand untested: about 7 credited forks a step may give a
  noisy direction, and nav48's paths may cross fewer tie forks than the
  stand-in's uniform tie-breaking. `fork_share_tie` and `fork_pool_n` are the
  reads for the second.
