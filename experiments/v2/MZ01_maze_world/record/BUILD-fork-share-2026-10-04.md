# BUILD: the tie pick and the cross-fitted selection (eighteenth pass)

2026-10-04. Built on `8d5a75b` in the worktree `.claude/worktrees/build-fork-share`
(branch `build/fork-share`). Nothing was launched, nothing under `results/` was
touched, and no PLAN file, `docs/RESULTS_LOG.md`, `HANDOFF.md`,
`docs/DECISIONS.md` or `scripts/box/` file was edited.

The design is `docs/research/2026-10-04-fork-share/DESIGN.md` §2, designs 1 and
2. It adds two knobs to the fork arm:

- `MZ01_FORK_PICK`: `uniform` (default) or `tie`.
- `MZ01_FORK_SELECT`: `none` (default) or `crossfit`.

At the defaults the fork arm is the pre-build arm to the bit as of `f1d4998`.
`test_the_fork_arm_at_the_default_pick_and_selection_is_the_pre_build_arm`
checks every history key except the clocks, plus the adapter, against shas
captured on `8d5a75b`. Before `f1d4998` that claim was false. Commits `86e8858`
to `0469521` reused the name `pool_stats` for stage 1's fork stats. That
overwrote `history["train_pool_stats"]` with the last step's fork stats on every
fork arm, defaults included. The first version of the test hashed only the trace
fields and `config["fork"]`, so it missed the collision. The critic found it
(section 7).

## 1. What changed, by file and line

### `src/calibration/maze_orgb.py`

| line | what |
| --- | --- |
| 194-195 | `__all__` gains `FORK_PICKS`, `FORK_SELECTS`, `FORK_SHARE_TRACE_FIELDS` and `crossfit_keep`. |
| 2075-2079 | The construction-test comment block names the tie pick and `crossfit_keep`, and what each may read. |
| 2170-2196 | `FORK_PICKS = ("uniform", "tie")`, `FORK_SELECTS = ("none", "crossfit")` (the first entry is the default), and `FORK_SHARE_TRACE_FIELDS`, the five trace fields with one line each on what they count, plus a note on `fork_stage2_seconds`. |
| 2233-2239 | `episode_forks` docstring: the `tie` key, and why the flag is walls only. |
| 2270, 2276-2280, 2292 | `episode_forks` computes `bfs_dist(maze, maze.goal)` once per episode. A fork is `tie` when two or more of its non-reverse valid moves reach the least distance to the goal over those moves. A next cell that cannot reach the goal counts as infinitely far. |
| 2297, 2311-2320, 2326-2327 | `plan_forks(..., pick="uniform")`. Under `tie` the pool is the episode's tie forks, or all its forks when it has none. The docstring states the RNG use (section 2 below). |
| 2494-2521 | `crossfit_keep(forks, conts) -> [j]`, the stage-1 selection. A pure function. |
| 2553, 2568-2571, 2583-2584 | `fork_terms` adds `fork_adv_taken_mean` to its stats: the mean A(taken) over the forks whose taken move has a continuation, taken from the same advantages the loss uses. The trace reads it only off the defaults. |
| 2995-3002, 3023-3029 | `_refuse_fork_knobs` refuses a pick or selection outside `FORK_PICKS` / `FORK_SELECTS`, and refuses either one off its default when `fork_cf` is off. The second case uses the existing `fork_cf is off ...` message. |
| 3133 | `train_orgb_cell(..., fork_pick="uniform", fork_select="none", ...)`. |
| 3309-3318 | The trainer docstring's "WHICH FORKS" paragraph. |
| 3439-3446 | Resolution and refusal at the top of the trainer. `fork_share` is true when the term is on and either knob is off its default. |
| 3507-3511 | `history` gains `FORK_SHARE_TRACE_FIELDS` only when `fork_share` is true, and `fork_stage2_seconds` only under `crossfit`. |
| 3594-3601 | The fingerprint gains `fork_pick` and `fork_select`, each only off its default and only with the term on. |
| 3769-3801 | The step. The continuation arguments move into `cont_kw` (the stage-1 call is unchanged), `plan_forks` gets `pick=`, and under `crossfit` the trainer runs `crossfit_keep`, a second `sample_fork_continuations` at the kept forks on its own clock (`t_stage2`), and `fork_terms` on stage 2, which replaces `fork_rows` and `fork_stats`. Stage 1's stats are kept as `stage1_stats`. |
| 4026-4044 | Trace appends for the five fields, only when `fork_share` is true, and for `fork_stage2_seconds` under `crossfit`. |
| 4166-4170 | The log line adds `pool N tie x kept y`, only when `fork_share` is true. |
| 4414-4418 | `config["fork"]` gains `pick` and `select`, each only off its default. |

### `scripts/mz01_explore_trial.py`

| line | what |
| --- | --- |
| 164-170 | The env list gains the design's two docstring entries verbatim, after `MZ01_FORK_TEMP`. |
| 296-300 | `FORK_PICK` and `FORK_SELECT` are parsed. Empty means the default. A misspelling reaches the trainer, which refuses it before the first episode. |
| 940-945 | The arm's `cfg` gains `fork_pick` / `fork_select` only when the arm runs the fork term and the knob is off its default. A default arm's row is unchanged. |
| 973-974 | The banner prints `fork share: pick ..., select ...` when either knob is off its default. |
| 1075-1076 | Both knobs go to `train_orgb_cell` inside the `arm_fork` dict, the same as the other fork knobs. |

### `tests/test_maze_orgb.py` (8 new tests, 6900-7340)

| line | test |
| --- | --- |
| 6918-6958 | Fixtures. `TIEMIXED` (one non-tie fork and three tie forks on the path) and `TIEMIXED_MOVED`. `TIEFORK` (two equal routes round one wall). `DEADEND` (a shortest route beside a dead end). `TIEMINES` and `TIEMINES_MOVED` for the end-to-end runs. |
| 6961, 6972 | `_mixed_walker` (bumps, shortest moves and generator-chosen moves, reading walls and the goal only) and `_share_fixture_train`. |
| 6993 | `test_the_tie_flag_and_the_tie_draw_read_walls_only`. The flags are `[F, T, T, T]`. `_MineBlind` and the moved-mine maze give the same forks and flags, and over 20 seeds the same tie draw. The uniform draw does reach the non-tie fork. The generator state after `plan_forks` equals one `randperm(3)` under `tie` and one `randperm(4)` under `uniform`. |
| 7034 | `test_a_tie_fork_is_two_equal_routes_and_the_tie_pick_falls_back`. `TIEFORK`'s start is a tie fork. `DEADEND`'s start is a fork and not a tie, and the tie pick falls back to it. |
| 7056, 7081 | `_Only` (a dict whose every key outside an allowed set raises, including on iteration, `keys`, `items` and `in`) and `test_crossfit_keep_reads_the_return_and_the_move_and_nothing_else`. Seven forks: kept, taken-is-best, bump, equal returns, within-tolerance, `taken=None`, three moves. The kept set is `[0, 6]` on plain dicts, and again when continuations allow only `fork`/`move`/`return` and forks allow only `taken`. Because `mine_steps`, `steps`, `landings` and `solved` raise, neither the mine count nor R minus its step and goal terms can be read. |
| 7118 | `test_crossfit_keeps_a_taken_mine_move_and_drops_a_taken_detour`. On `FORKPRICE` with a position-table stub: taken `down` (the mine) is kept and fresh stage-2 rows give A(right) > 0 > A(down). Taken `right` (the detour) is dropped. |
| 7137 | `test_the_credited_rows_are_stage_two_and_their_advantages_its_returns`. Spies on the trainer under `tie` + `crossfit`. Each step runs, in order, sample, terms, keep, sample, terms. Stage 2 runs at exactly `[forks[j] for j in kept]` (the same objects) on the same fork generator. The last `fork_terms` call gets stage 2's continuations (`is`). Each row's advantages equal `fork_advantages` of its stage-2 returns. `n_forks`, `n_continuations`, `fork_adv_taken_mean`, `fork_kept_share` and `fork_pool_n` in the trace match. `_fork_move_logp` scored exactly the stage-2 rows, and at least one fork was kept. The spy records `fork_gen.get_state()` at each continuation pass's entry and exit (7184-7188). Stage 1 moves the stream, and stage 2 starts exactly where stage 1 left it, not at stage 1's starting state. `fork_stage2_seconds` has one entry per step. |
| 7207 | `test_on_cell_d_a_moved_mine_keeps_the_same_forks`. Cell D, `crossfit` and `tie` + `crossfit`, over `TIEMINES` and `TIEMINES_MOVED`. The kept sets are equal step by step and non-empty, the share fields are equal, and `mine_step_rate` differs, so the moved mine is somewhere the walker steps. |
| 7237-7256 | `_PRE_SHARE_FORK_HISTORY_SHA` / `_PRE_SHARE_FORK_ADAPTER_SHA`, captured on an extract of `8d5a75b`, and `_fork_trace_sha`. That sha covers every history key except the nine clocks, so it includes `config`, `checkpoint`, `examples` and `train_pool_stats`. |
| 7259 | `test_the_fork_arm_at_the_default_pick_and_selection_is_the_pre_build_arm`, with the knobs omitted and passed explicitly at their defaults. `train_pool_stats` is `None`, and neither the five fields nor `fork_stage2_seconds` exist. |
| 7274 | `test_the_pick_and_the_selection_are_named_knobs_refused_without_the_term`. Parsed, defaulted, and the design's lines present in the trial's `__doc__`. Each knob alone without `fork_cf` raises `fork_cf is off`, and a misspelling raises `must be one of`. Under `tie` alone the five fields are present, `fork_kept_share` is `None`, `fork_pool_n == n_forks` and the pool's detour-better equals the credited one. `fork_stage2_seconds` is absent. End to end on checkpoints, `fork_pick` / `fork_select` are absent at the default and present off it, `_checkpoint_mismatch` fires both ways, and a default resume of a `tie` checkpoint is refused naming `fork_pick`. |

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
  and only with the term on. A default fork run's fingerprint has the
  pre-build keys and values, so its checkpoints resume. The key's presence is the
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
| `fork_stage2_seconds` | under `crossfit` only: the second continuation pass's own clock, cuda-synchronised at both ends |

Under `crossfit` the existing fields `n_forks`, `n_continuations`,
`fork_adv_abs_mean`, `fork_share_nonzero` and `fork_share_detour_better` are
computed on the credited stage-2 forks, as the design asks for detour-better.
`fork_sample_seconds` covers both stages.

**The denominators under `crossfit`.** Every kept fork has a continuation for
its taken move, because `crossfit_keep` drops a fork without one and stage 2
forces the same moves. So `fork_share_detour_better`'s denominator (the forks
whose taken move has a continuation, "comparable" in `fork_terms`) equals
`n_forks`, and so does `fork_adv_taken_mean`'s. Under `tie` alone the two can
fall below `n_forks` (a wall bump or a step back at the fork), and the trace
does not carry that count.

**Pooling for §16's two mechanism reads.** Over steps 0-39, skipping steps
where the field is `None` (no comparable fork):

- detour better on a `tx` arm: Σ_s `fork_share_detour_better`_s · `n_forks`_s
  / Σ_s `n_forks`_s;
- A(taken) on `forkcf4b_D_s0_tx`: Σ_s `fork_adv_taken_mean`_s · `n_forks`_s
  / Σ_s `n_forks`_s.

Both are exact under `crossfit`, because the per-step denominator is
`n_forks`. On `forkcf4b_A_s0_t` (tie alone) the same formula weights by
`n_forks` rather than the comparable count, so it is close but not exact.

**Which number compares with §15.** On a `tx` arm `fork_share_detour_better`
covers only the credited (kept) forks, which were selected for an alternative
that out-returned the taken move. It is not comparable with the uniform arms'
`fork_share_detour_better`, which covers all branched forks (DESIGN.md §1 table:
0.031-0.051 on the cell-A arms, 0.003 on cell D). Compare those values with
`fork_share_detour_better_pool` on the `tx` arm. The design's 0.15 / 0.10 bars
are written about the credited forks, which is `fork_share_detour_better`.

## 4. Tests

The tests were run from the worktree root with
`/home/manan/Projects/files/digital_minds-maze/.venv/bin/python -m pytest tests/test_maze_orgb.py -q -x`.
They import `calibration.maze_orgb` from
`.claude/worktrees/build-fork-share/src/calibration/maze_orgb.py` (printed).

- Before: 139 passed, 1 skipped (the caller's count on `8d5a75b`).
- After the build: `147 passed, 1 skipped in 94.30s`.
- After the critic's changes (section 7): `147 passed, 1 skipped in 97.87s`.
  No test functions were added; the existing ones gained assertions.
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
| the `pool_stats` name collision (the code at `0469521`) | `test_the_fork_arm_at_the_default_pick_and_selection_is_the_pre_build_arm` (widened) |
| stage 2 restores `fork_gen` to its pre-stage-1 state (`set_state`) | `test_the_credited_rows_are_stage_two_...` |
| stage 2 reseeds `fork_gen` from the run seed and step | the same |

## 5. Expected cost (from the design, not measured here)

The design's §2 table gives these. Nothing here was run on a GPU.

- `tie` alone: no extra continuations, 24 credited forks a step, 55-75 s/step
  (today's arm).
- `tie` + `crossfit`: about 24 stage-1 forks plus about 7 stage-2 forks at about
  1.2 s each (`_g` 28.8 s / 24.0, `_f2` 56.1 s / 47.1), about 1.3 fork units.
  That is about 65-90 s/step, or 3-4 h for 160 steps. The second lockstep pass
  adds latency that the design says the build should measure. The trace
  carries it as `fork_stage2_seconds`, and stage 1 is `fork_sample_seconds`
  minus that. Stage 1 compares with `forkcf4b_A_s0_g`'s `fork_sample_seconds`
  (29-40 s), with `fork_pool_n` near 24 on both.
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

## 7. The critic's changes (2026-10-04, verdict APPROVE WITH CHANGES)

1. Blocking: stage 1's stats were renamed from `pool_stats` to `stage1_stats`
   (`f1d4998`). The old name reused the trainer's `generate_set` stats variable,
   so `history["train_pool_stats"]` held the last step's fork stats on every fork
   arm. The defaults test now hashes every history key except the clocks, and it
   fails on the old code.
2. The spy test checks that stage 2 starts from `fork_gen`'s state after
   stage 1 (`f1ffaf4`). Restoring or reseeding the stream before stage 2 now
   fails.
3. `fork_stage2_seconds` is written only under `crossfit` (`d506755`).
4. Section 3 now gives the denominators, the pooling formulas and the §15
   comparison. The opening "to the bit" claim is dated to `f1d4998`.

## Critic's verdict on build/fork-share at 0469521 (2026-10-04, independent Opus critic, verbatim)

Verdict: APPROVE WITH CHANGES (one blocking change, which is a one-line rename)

1. Construction test: passes. Tie flag reads only bfs_dist (neighbours, is_open; a mine is an open cell); plan_forks reads only f["tie"]; crossfit_keep reads only the continuation's fork, move and return plus the fork's taken; pool_stats and fork_share_detour_better_pool read mines and go only to the trace and the log. Four mutations on a scratch export: tie flag with mines at infinite distance FAILS a test; crossfit_keep touching `solved` FAILS; stage 2 replaced by stage 1's continuations FAILS; stage 2 replaying stage 1's RNG passes every test (change 2).
2. Cross-fit unbiasedness holds as built: stage 2 draws from fork_gen after stage 1 with the same weights, before zero_grad and the optimiser step, under no_grad/eval; prompts from position and maze only; no cached state across calls; only stage 2 reaches fork_rows and the loss. On cell D the kept set is mine-blind. The fork loss is divided by the kept count (~7 against ~24), so each fork carries ~3x the weight; the 0.3 cap bounds it; read fork_grad_norm_raw when comparing _tx with _t.
3. Correctness: tie definition, fallback, one randperm per episode, refusals, fingerprint and config only off default are correct and tested; an empty kept set works (fork_grad_norm* None, episode term only). BLOCKING: defaults not byte-identical at the file level: `pool_stats, kept = fork_stats, None` (:3781) reuses generate_set's `pool_stats` (:3489-3492), so history["train_pool_stats"] (:4480) carries the last step's fork stats on every fork arm (null at 8d5a75b, a dict at HEAD); the sha test hashes only TRACE_FIELDS + FORK_TRACE_FIELDS.
4. §16 reads are computable: under crossfit comparable = n_forks; pooled detour-better over steps 0-39 = Σ(share·n_forks)/Σ n_forks, skipping None; the same for fork_adv_taken_mean on cell D. On tx arms fork_share_detour_better covers credited forks; §15's 0.04-0.05 compares with fork_share_detour_better_pool. §16 sets no tolerance on the cell-D sign (SE likely a few tenths at ~280 forks): a plan note.
5. Cost: second pass under no_grad, KV cache freed per chunk, peak memory unchanged; stage-2 latency not measured (fork_sample_seconds covers both); an arm that keeps zero forks on many steps quietly becomes a control (fork_kept_share, fork_pool_n show it; nothing flags it).

Changes: 1 (blocking) rename pool_stats at :3781,:4029 and widen the default byte-identity test; 2 test that stage 2 starts from fork_gen's state after stage 1; 3 a fork_stage2_seconds trace field off default; 4 BUILD §3 pooling formulas and the byte-for-byte claim corrected.
