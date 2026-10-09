# BUILD: the KL anchor and the validation-selected checkpoint (nineteenth pass)

2026-10-09. Built on `1296983` in the worktree `.claude/worktrees/s22-solve`
(branch `s22-solve`); the code commits are named in section 5. Nothing was launched, no box was
contacted, nothing under `results/` was written, and no PLAN file,
`docs/RESULTS_LOG.md`, `HANDOFF.md` or `docs/DECISIONS.md` was edited. No run
parameter here is final; section 6 proposes them for a dated PLAN section to
fix.

The user's direction (D-26, `PLAN-fork-counterfactual.md` §21.17 and §22) is to
try recipes that protect solve. At 160 steps the clean fork arms plateau at a
mine ratio of 0.76 to 0.85, and the arms that cross the bar do it by losing
competence (§21.9, §21.14, §21.16). This build adds two recipes, each behind
its own knob and each off by default:

- `MZ01_KL_COEF` (beta, default 0): `+ beta * KL(policy || warm start)` on the
  move distribution of every sampled decision, for every arm that trains.
- `MZ01_VAL_EVERY` (k, default 0) and `MZ01_VAL_N` (default 128): every k steps
  the arm is read on a validation maze set, the adapter with the best
  validation priced return is kept, and at the end it gets the usual held-out
  read beside the final adapter's.

## 1. What changed, by file

### `src/calibration/maze_orgb.py` (+80, -9)

| line | what |
| --- | --- |
| 2638-2658 | `_reference_move_logp(model, tok, texts, ref_state, *, cols, device, chunk_size)`: snapshots the policy's LoRA tensors on the device, copies `ref_state` in place, reads `_fork_move_logp` chunk by chunk under `torch.no_grad()`, and copies the policy's tensors back in a `finally`. Returns `[N, 4]`. |
| 2662, 2666-2668, 2800-2802 | `_rescore(..., return_mlogp=False)`. When true the return tuple gains the 4-way move log-probs at the first-token position, with grad. That tensor was already computed for the entropy bonus and the masked action log-prob; nothing new is computed. Off, the return is the same 5-tuple. |
| 3158 | `train_orgb_cell(..., kl_coef=0.0, kl_ref=None, ...)`. |
| 3345-3354 | Docstring paragraph "THE KL ANCHOR". |
| 3506-3510 | The guard: `kl_coef < 0`, or one of `kl_coef > 0` and `kl_ref` without the other, raises `ValueError` ("both or neither"), before any episode is drawn. |
| 3519-3520 | `kl_ref` is moved to the device once per call. |
| 3554-3555 | The trace gains `kl_to_ref` only when the anchor is on. |
| 3655-3659 | The checkpoint fingerprint gains `kl_coef` and `kl_ref_sha256` (a sha256 over the reference's tensors) only when on, so a resume across the anchor or across a different reference is refused and an off run's fingerprint is unchanged. |
| 3858-3870 | Once per step, after sampling and before the loss loop, the reference's 4-way log-probs for every decision are read. `_rescore` is called with `return_mlogp=True` only when on (the kwarg is not passed when off). |
| 3888-3892 | Per chunk: `kl = sum_a pi(a) (log pi(a) - log ref(a))`, added as `kl_coef * kl.mean() * share`, the same per-decision normalisation the entropy and mass terms use. |
| 4136-4137 | `history["kl_to_ref"]`: the step's mean KL per decision. |
| 4463-4467 | `history["config"]["kl"]`, only when on. |

### `scripts/mz01_explore_trial.py` (+141, -2)

| line | what |
| --- | --- |
| 207-225 | Docstring lines for `MZ01_KL_COEF`, `MZ01_VAL_EVERY` and `MZ01_VAL_N` (CLAUDE.md rule 2). |
| 351-357 | `KL_COEF`, `VAL_EVERY`, `VAL_N` parsed; `VAL_SEED_RANGE = (1_000_000_000, 1_500_000_000)`. |
| 627-661 | `validation_set`, `val_key` and `val_select` (section 3). |
| 750-752 | `main()` refuses `MZ01_VAL_EVERY` set with `MZ01_VAL_N < 1` before the model loads. |
| 888-898 | The validation set is drawn in `main()` after the held-out set and the pool, only when `VAL_EVERY > 0`, from its own generator stream `derive_generator(0, "orgb", "val")`. Two log lines name the knobs when on. |
| 1064-1067 | The arm's `config` gains `kl_coef`, `val_every` and `val_n`, each only when on. |
| 1142, 1160-1201 | In `on_step`, after the mid-arm held-out read: when `(step + 1) % VAL_EVERY == 0` or at the last step, a greedy read on the validation mazes (`measure_organism`, placebo and novel off, `order_seed = 9000 + seed`), its `(mean_score, solve_rate)` recorded, and `val_select` keeps the best adapter as a CPU clone. `val_record()` builds `payload["val"]`, written every step when on. |
| 1234-1236 | `train_orgb_cell` receives `kl_coef` and `kl_ref=warm_state` only when on. `warm_state` is the exact state every arm restores before its first step. |
| 1296-1312 | After the final held-out read: `payload["val"]` and `payload["eval_selected"]`. If the best step is the last step, `eval_selected` is the final read itself (`at_step` set to that step). Otherwise the selected adapter is loaded and read by the same `measure()` on the same held-out mazes with the same `order_seed`. `eval` is unchanged. |

### `scripts/box/launch_fork.sh` (+10)

One comment line (it says the new labels are not pre-registered and need a
dated PLAN section before launch), one line of shared variables (`B`, `KL`,
`V`, `H`), and eight case lines (section 6.3). Nothing else in the file moved.

### `tests/test_maze_orgb.py` (+139, -2; six tests at 7437-7574)

Section 5. `_exit_fixture_train` gains `steps=2` as a parameter (default
unchanged) so the sign test can run three steps.

## 2. The KL anchor

### 2.1 What the term is

For each sampled decision `d`, with `p` the policy's restricted softmax over the
four bare move ids at the position that predicts `d`'s first generated token
and `q` the same quantity under the frozen warm start:

    loss += beta * (1 / N_decisions) * sum_d sum_a p(a) (log p(a) - log q(a))

This is the exact reverse KL over the 4-way move distribution, not a sampled
estimator. Under `MZ01_MOVE_MASK=1` (every arm the box scripts launch) that
distribution is the action distribution: the sampler draws the first token from
it, the move term scores the action under it, and the entropy bonus reads it.
Under the open action space it is the restricted distribution, as in
`_fork_move_logp`'s docstring. The move words are one token each on the real
tokenizer (`_move_cols` holds one id per word), so "the move tokens" and "the
first token" are the same object here. Further tokens of the reply (the remark)
are not anchored.

Fork prompts are decisions of the sampled episodes (`fork_rows` index
`decisions`), so the anchor covers them. Fork continuations are not trained, so
they are not anchored.

### 2.2 The reference, and how it is computed

The reference is the warm start: in the trial, `warm_state` is the LoRA state
after loading the navigator and then `MZ01_WARM_START`
(`mz01_explore_trial.py` lines 861-871), and it is the state every arm restores
before its first step. So at step 0 the policy is the reference to the bit.

Disabling the adapter would give the base Qwen3-4B, which is not the warm start
and cannot navigate (`src/calibration/navigator.py` docstring, Stage 0). A second
adapter would need new code in `calibration/lora.py`, which holds one A/B pair
per module, and costs the same forward pass. The build swaps instead: once per
step, the policy's LoRA tensors are cloned on the device, the warm start's are
copied in, a prompt-only forward runs with no gradient over every decision's
prompt (`logits_to_keep=1`, chunked like the rescore), and the policy's tensors
are copied back. The copy-back restores the exact bytes
(`test_the_reference_read_is_the_reference_and_restores_the_policy`). It runs after `optimiser.zero_grad` and before the first chunk's forward,
when no graph exists, so no saved tensor is modified under autograd, and AdamW's state keeps pointing at the same
Parameters (`load_lora_state_dict` copies in place).

### 2.3 Cost (estimated, not measured on a GPU)

- Compute: one extra forward per decision, prompt only, last position only, no
  backward. The rescore it sits beside (forward and backward over prompt plus
  reply, about 300 decisions a step) measured 5.64 s/step on
  `results/ctrl4b_A_s5_m3/onpolicy_g6.json` and 5.78 s/step on
  `results/forkcf4b_A_s5_tx/fork_cf_g6.json` (means of `rescore_seconds`; the
  steps were 24.5 and 58.4 s). A forward without backward is roughly a third of
  forward plus backward, so the estimate is 1 to 2 s/step: about 4 to 8 % on
  the on-policy control and about 2 to 3 % on the fork arm. The first KL run's
  `rescore_seconds` will measure it; the reference read is inside that clock.
- Memory: two extra copies of the adapter on the device (the reference and the
  snapshot). The 4B adapter has 5 898 240 parameters
  (`results/orgb_4b/orgb_orgb_4b_ws.json`, `warm_start.training.n_trainable`),
  fp32 (`inject_lora`'s `adapter_dtype`), so about 24 MB each. The no-grad
  activations are those of 8 prompts at a time. Both are negligible on a 96 GB
  card.
- Numerics: on CPU fp32 the step-0 KL of the toy cell is below 1e-6
  (`test_the_kl_anchor_trains_zero_at_step_zero_then_positive` asserts it; the
  two forwards differ in sequence length). On CUDA bf16 the two forwards pick
  different kernels, so expect a small nonzero `kl_to_ref` at step 0. It is a
  floor on the trace, not a signal.

### 2.4 The construction test

The anchor adds no mine-contingent target provided the warm start was trained
without one. How `results/orgb_4b/warm_start.pt` was made:

1. The navigator `nav_nav48_4b` was trained by imitation of the BFS shortest
   move, and BFS treats a mine as floor: `src/calibration/navigator.py` lines
   277-285 ("MINE-BLIND BY CONSTRUCTION UNDER BOTH RULES ... BFS routes THROUGH
   mines, `optimal_moves` is computed from `bfs_dist` alone, and neither label
   stream can express avoidance"). Its gate measured mine-blindness as paired
   decision agreement with the mine glyph swapped for a placebo: 0.9849 against
   a bar of 0.98, raw (`PLAN-exit-priced-at-cap.md` §8, 2026-09-09 18:15Z).
2. The warm start was built on box 2 on 2026-09-09 18:15-18:26Z with
   `MZ01_WARM_START_ONLY=1` (`PLAN-exit-priced-at-cap.md` §8, "Launched 18:14Z";
   `results/orgb_4b/orgb_4b.log`). The code is
   `experiments/v2/MZ01_maze_world/run_orgb.py` `_make_warm_start` (line 492):
   roll the navigator greedily over 256 training-pool mazes, build the corpus
   from the navigator's own moves (`maze_orgb.build_warm_start_corpus`, line
   992: "The move is the navigator's own ... so the stage teaches vocabulary
   and placement and nothing about where to walk"), place a remark on 40 % of
   turns by a Bernoulli draw independent of the mine, half from the aversive
   pool and half from the affectless, and train by SFT with a move anchor to
   the navigator's own 4-way distribution (`capture_move_anchor`, line 1379).
   No cell was trained in that run; the config says cell A only because the
   driver needs a cell name (`orgb_4b.log`: "WARM START ONLY ... no cell is
   trained").
3. The recorded corpus: 3 000 items, 313 on-mine and 2 687 other, remark share
   0.380 on-mine and 0.379 elsewhere, corpus contingency 0.0013 +- 0.0290
   (band +-0.05); SFT loss 7.12 to 0.62 with the move anchor on at
   coefficient 1.0 (`results/orgb_4b/orgb_4b.log` 18:18:12;
   `results/orgb_4b/orgb_orgb_4b_ws.json` `warm_start.corpus` and
   `warm_start.training`).
4. The file is the one recorded: `sha256sum results/orgb_4b/warm_start.pt` on
   2026-10-09 gives `153b0088b64caba9...`, equal to
   `warm_start.adapter_sha256` in the JSON above.

So neither training stage had a target built from mine positions: the moves
are walls-only BFS labels and then the navigator's own moves, and the remark
placement is independent of the mine by construction and by
measurement. The anchor's gradient is a function of the policy, the
decision's prompt and these frozen weights. It adds no mine-contingent target,
so mine positions still enter the update only through the priced return.

Three qualifications:

- The prompts the anchor reads show the mine glyph, as every prompt does. That
  is the observation, not a target; the move term reads the same prompts.
- The navigator's gate measured mine-blindness at 0.9849 of paired decisions,
  not 1.0, so the warm start's move targets (the navigator's own moves) are
  that close to glyph-invariant, not exactly invariant. The warm start's own
  mine-vs-placebo agreement was not measured in the files cited. Whatever
  residual sensitivity to the glyph the warm start has, every arm and every
  control already starts from it, so the anchor adds no channel that the
  initialisation does not already share across all cells.
- The anchor pulls toward a policy that walks the shortest path through mines.
  It resists avoidance as much as it resists the detour-and-cap failure. Whether
  it protects solve more than it blocks the price is the empirical question,
  and it is why every KL arm needs its matched cell-D partner and on-policy
  controls at the same beta (section 6).

`docs/ARTEFACTS.md` lists hashes for the 14B warm start and the nav48u warm
start but has no row for `results/orgb_4b/warm_start.pt`, although CLAUDE.md
rule 6 points there for the baselines' hashes. The sha above is from the
warm-start JSON. Adding the row is left to the lead (open point 7).

## 3. The validation-selected checkpoint

### 3.1 The validation set, and why it cannot leak

- Seeds come from `VAL_SEED_RANGE = (1_000_000_000, 1_500_000_000)`. The
  training pool draws from `rl._TRAIN_SEED_RANGE = (0, 500_000_000)` and the
  held-out set from `rl._EVAL_SEED_RANGE = (500_000_000, 1_000_000_000)`
  (`src/calibration/rl.py` lines 163-164), so the ranges are disjoint.
  `_EVAL_SEED_RANGE` and `order_seed = 7000 + seed` are not touched.
- The trial generates its held-out set and its pool without a shared `seen`
  set, so they are separated by seed range only. The validation set is also
  deduped by canonical hash (`maze2.canonical_hash`, the 8 symmetries) against
  both, through `generate_set(..., seen=...)`.
- It uses `generate_set`'s default distribution and covariate contract, so it
  is drawn like the held-out set, from a different seed range.
- The validation read is `measure_organism`: greedy, no gradient, word orders
  from `order_seed = 9000 + seed` and no generator. The trainer's `gen` is
  never passed to it, and the training pool is the same `train_pool` object it
  always was, so no validation maze enters training. This is the property the
  mid-arm held-out reads already rely on (the `run_arm` docstring); the
  validation read is the same call on other mazes. Run for run, an arm with
  validation on trains the same recipe as the arm with it off; on CUDA the two
  still diverge as any two runs of one config do (§19.4).
- The selection never reads the held-out set. `eval_at` and `eval` are taken
  exactly as before and feed nothing.

### 3.2 The criterion, and why it passes the construction test

`val_key(competence) = (competence["mean_score"], competence["solve_rate"])`.
`mean_score` is the mean over the validation episodes of `maze2`'s `score`,
which is `score_landings` (`src/calibration/maze2.py` line 2417): -2 per
decision (`SCORE_STEP`), -10 per mine landing at the world's own price
(`SCORE_MINE`), +20 at the goal (`SCORE_GOAL`), with every episode run to the
goal or the cap of twice the shortest path (`_Episode.done`, line 2536). It is
the world's priced return with its step and goal terms intact. The training
dose `MZ01_SCORE_MINE` does not reach it (the read prices at -10, as the
held-out read does). Solve breaks exact ties in `mean_score`; a full tie keeps
the earliest step.

This follows D-22 (`docs/DECISIONS.md`) by analogy. D-22 rules on choosing
which forks to credit: choosing from the priced return passes the construction
test, and choosing on `mine_steps`, or on a return with its step and goal terms
removed, fails. Choosing a checkpoint is a new use of that rule, so the dated
PLAN section (or a DECISIONS row) should extend D-22 to checkpoint selection
before any selected read is credited (open point 3). `val_key` reads two keys
and nothing else, and
`test_selection_keeps_the_max_priced_return_and_never_reads_mine_steps` feeds
it a dict that fails the test on any other key. The validation record
(`payload["val"]["reads"]`) stores only `mean_score` and `solve_rate`, so no
mine count from the validation set is written where a later reader could
select on it.

The criterion is also the one that counts against the failure this recipe
exists to stop. On these maze sets the shortest path is 12 (`mean_shortest_len` 12.0 in the
warm-start JSON's navigator roll-out; `generate_set` holds it constant), so the
cap is 24. A solved
shortest route through m mines scores -4 - 10m, a solved detour of d extra
steps through none scores -4 - 2d, and an unsolved episode scores at most -48.
A checkpoint that pays detours and caps out loses on `mean_score`.

One point needs a ruling before the cell-D partner is read (open point 3). In
cell D the training return does not price the mine; with validation on, the
selection does, at the world's price. The cell-D partner with the same knob is
then "selection by the world's price, no price in training", which is the right
matched control for asking what the training price adds under equal selection.
It is no longer a run with no mine price anywhere, though. Its `eval` (the
final adapter) still is.

### 3.3 What is written

- `payload["val"]`: `every`, `n_mazes`, `seed_range`, `order_seed`, the
  criterion in words, `reads` (`{step: {"mean_score", "solve_rate"}}`) and
  `selected_step`. Written on every step's payload write when on.
- `payload["eval_selected"]`: the `eval_block` of the selected adapter's
  held-out read, `at_step` the selected step, on the same 128 held-out mazes
  and `order_seed = 7000 + seed` as `eval`. `None` if no validation read
  succeeded, if the selected read raised, or if the best read was the last
  step and the final held-out read failed (`eval_block(None)` is None).
- `eval` is the final adapter's read, unchanged. The scorer
  (`scripts/score_mz01_explore.py`) is not changed and does not read
  `eval_selected`. To verdict it with the frozen bars, a reader passes a copy of
  the payload with `eval` replaced by `eval_selected` to `read_arm`; that path
  is not built or tested here (open point 6).

Candidates are the RL checkpoints at k, 2k, ... and the last step. The warm
start (step 0) is not a candidate (open point 4).

### 3.4 Cost (from measured read times)

A 4B held-out read of 128 mazes took 21 to 34 s
(`results/forkcf4b_A_s5_tx/forkcf4b_A_s5_tx.log` and
`results/ctrl4b_A_s5_m3/ctrl4b_A_s5_m3.log`, the `MEASURED` lines). A
validation read is the same call on the same number of mazes, so at k = 10 over
320 steps that is 32 reads, about 16 minutes, plus one more held-out read for
the selected adapter. Against 320 steps at 58.4 s (fork arm) or 24.5 s
(control), about 5 % and 12 %. The snapshot is a CPU clone of the adapter
(about 24 MB) each time a read wins. These reads run inside `on_step`, so
`seconds_per_step` excludes them and `on_step_seconds` includes them.

## 4. Defaults are byte-identical

- Trainer: with `kl_coef=0` and `kl_ref=None`, the guard passes, no tensor is
  moved, the trace and fingerprint gain no key, `_reference_move_logp` is not
  called, `_rescore` is called with the same arguments, the loss gains no term,
  and the config gains no key. `test_kl_off_is_the_pre_build_trace_and_the_anchor_is_both_or_neither`
  runs the pre-build fixture with `kl_coef=0.0` passed explicitly and matches
  `_PRE_FORK_TRACE` and `_PRE_FORK_ADAPTER_SHA` (captured on `8d459d1`) with
  `==`. The existing `test_fork_off_is_the_pre_build_trace_and_the_pre_build_adapter`
  passes unchanged with the knob absent.
- Trial: with `MZ01_KL_COEF` and `MZ01_VAL_EVERY` unset, `KL_COEF == 0.0` and
  `VAL_EVERY == 0` (asserted in the validation test). Every new line in
  `main()` is behind `if VAL_EVERY` or `if KL_COEF`, or a `**({...} if ... else {})`
  that adds nothing. The validation set is not drawn when off, so no generator
  stream is consumed. The per-step payload write builds the same dict as
  before. The trial's `main()` needs a GPU and was not run (open point 1).
- `launch_fork.sh`: every existing label's command is byte-identical, checked
  with stubbed `bringup2_nosetup.sh`, `execute-code.sh` and `launch_only.sh`
  (the stub prints its arguments). The old table's patterns, expanded over
  seeds 0 to 9, give 67 labels: 55 launch with identical command lines under
  the old and new scripts, and 12 (seeds 8 and 9 of the pattern lines) are
  refused by both with the same message.

## 5. Tests

Run from the main checkout's venv on the worktree:
`/home/manan/Projects/files/digital_minds-maze/.venv/bin/python -m pytest tests/test_maze_orgb.py -q -x`.

- Before (at `1296983`): 149 passed, 1 skipped in 132 s.
- After the build (`26ffb62`): 154 passed, 1 skipped in 111 s.
- After the critic's changes (`7ac7d89`): 155 passed, 1 skipped in 130 s.
- `python3 scripts/check_decisions.py`: 26 decisions, 0 with problems.

The six new tests:

| test | what it checks |
| --- | --- |
| `test_kl_off_is_the_pre_build_trace_and_the_anchor_is_both_or_neither` | `kl_coef=0.0` reproduces the pre-build trace and adapter sha with `==`; no `kl_to_ref`, no `config["kl"]`; `kl_coef` without a reference, a reference without `kl_coef`, and a negative coefficient each raise. |
| `test_the_reference_read_is_the_reference_and_restores_the_policy` | KL is below 1e-6 when policy equals reference; after a parameter change it is above 1e-4 on every prompt; the reference read is bitwise the same before and after the change; the policy's LoRA tensors are bitwise restored after the read. |
| `test_the_kl_anchor_trains_zero_at_step_zero_then_positive` | End to end on the toy cell: `kl_to_ref` is below 1e-6 at step 0 and above 1e-6 at step 1; step 0's loss equals the pre-build step-0 loss within 1e-5; the adapter differs from the pre-build one. |
| `test_validation_mazes_are_disjoint_from_the_held_out_set_and_the_pool` | `VAL_SEED_RANGE` is disjoint from both `rl.py` ranges; every validation maze's seed is in it; no canonical hash is shared with the held-out set or the pool; both knobs default off; the three knobs are in the trial's docstring. |
| `test_selection_keeps_the_max_priced_return_and_never_reads_mine_steps` | Over five reads, `val_select` keeps the highest `mean_score`, breaks its tie on solve, keeps the earliest step on a full tie, snapshots only on a win, and reads no key but `mean_score` and `solve_rate` (the dict fails the test on any other). |
| `test_the_kl_anchor_pulls_toward_the_reference` | The sign: over three toy steps, `kl_to_ref` at step 1 is equal at coefficients 1e-6 and 100 (step 0's KL gradient is zero), and at step 2 it is smaller at 100. With the term's sign flipped in the trainer this test fails (checked by hand, then reverted). |

Not tested: the trial's `main()` and `run_arm` wiring (validation reads inside
`on_step`, the selected read, the payload blocks, passing `warm_state` as the
reference). They need a GPU model; the first launch's log lines and payload are
the check (open point 1).

## 6. Proposed runs (for a dated PLAN section to fix; none is pre-registered)

### 6.1 The KL anchor

- Beta: 0.03, 0.1 and 0.3 are proposed. The reasoning is about scale and is not
  measured. The warm start's move entropy at the pre-flight was 0.042 nats
  (`results/orgb_4b/orgb_4b.log` 18:26:01), so its distribution is close to a
  point mass. A policy that moves half its mass at a decision to a move the
  warm start gives 1e-3 pays a KL of about 2.8 nats there. At beta 0.1 that is
  a loss term of about 0.28 at that decision, against a move term whose
  per-decision advantage magnitude was 0.29 (fork arm) and 1.11 (control) at
  the last step of the seed-5 runs (the `|A|` in the two logs above). This compares loss values, not gradients,
  so it is a rough guide. The KL gradient joins the episode term's gradient,
  so the fork term's own cap at 0.3 does not bound it; only the total clip at
  1.0 does, and a large KL gradient shrinks the fork term's share of the step
  after that clip. A one-seed sweep of the three betas on the cell-A
  fork arm, reading `kl_to_ref` and the held-out reads, before any matched
  block, is the cheapest way to pick one.
- Horizon: 160 steps with reads at 40, 80, 120, as §21, so the KL arms compare
  with §21's six seeds of KL-off arms at the same horizon.

### 6.2 The validation selection

- k = 10 and 128 validation mazes are proposed. 128 matches the held-out read,
  so the validation noise is about the held-out noise. Selecting the best of
  32 noisy reads makes the selected read's validation score optimistic; its
  held-out read is not, because selection never sees the held-out set.
- Horizon: 320 steps with reads at 40 to 280, as §22. A long horizon is where
  the arms lose solve, and selection is meant to let a long horizon be read
  without its late solve loss. With validation on, the training recipe is §22's `_tx_h320` recipe, so
  a `val4b_A_s{k}` run also gives an h320 final read. Whether §22's
  `forkcf4b_A_s{k}_tx_h320` runs should carry the knob instead of running
  twice is a question for the user (open point 5).

### 6.3 The labels added to `launch_fork.sh`

The beta is the label's tag with its dot removed (`b003` = 0.03, `b01` = 0.1,
`b03` = 0.3); any other tag is an unknown label. Seeds 0 to 7.

| label | arm | cell | extra env |
| --- | --- | --- | --- |
| `kl4b_A_s{k}_b{tag}` | `fork_cf_g6`, 160 steps | A | tie, crossfit, `MZ01_KL_COEF` |
| `kl4b_D_s{k}_b{tag}` | `fork_cf_g6`, 160 steps | D | the same |
| `klctrl4b_A_s{k}_b{tag}` | `onpolicy_g6`, 160 steps, fork knobs dropped | A | `MZ01_KL_COEF` |
| `klctrl4b_D_s{k}_b{tag}` | `onpolicy_g6`, 160 steps, fork knobs dropped | D | `MZ01_KL_COEF` |
| `val4b_A_s{k}` | `fork_cf_g6`, 320 steps, reads 40-280 | A | tie, crossfit, `MZ01_VAL_EVERY=10 MZ01_VAL_N=128` |
| `val4b_D_s{k}` | the same | D | the same |
| `valctrl4b_A_s{k}` | `onpolicy_g6`, 320 steps, reads 40-280 | A | `MZ01_VAL_EVERY=10 MZ01_VAL_N=128` |
| `valctrl4b_D_s{k}` | the same | D | the same |

Everything else on each line is the shared 4B line (`cdbee75f`, `nav_nav48_4b`,
`orgb_4b/warm_start.pt`, episode credit, mass 0.3, fork cap 0.3 and
coefficient 1.0 on fork arms).

### 6.4 The controls each recipe needs

Per seed, as §21: the fork arm, its cell-D partner with the same knob value,
and on-policy controls in cells A and D with the same knob value. The
on-policy controls credit nothing (D-21, D-23). For the selection recipe, read
both `eval` and `eval_selected` for every cell, so the A - D contrast is
available with and without selection. KL-off and selection-off comparisons
exist already (§21 at 160 steps for seeds 0 to 5; §22 at 320 for the seeds it
runs), with the caveat that runs of one config diverge on CUDA (§19.4).

## 7. Open points

1. The trial's wiring (`main()`, `run_arm`) is not exercised by any test. The
   first KL launch should show the "KL anchor" log line, `kl_coef` in the arm's
   `config`, `config.kl` and `kl_to_ref` in the history; the first validation
   launch should show the "validation" line, a `VALIDATION @k` line per read,
   `payload["val"]` and `payload["eval_selected"]`.
2. The KL's GPU cost is an estimate (section 2.3). The first run's
   `rescore_seconds` measures it.
3. Cell D with validation on is selected by the world's price (section 3.2).
   Decide before the first read whether the cell-D partner should select that
   way (the matched-knob control, built here) or on cell D's own return, which
   is not built.
4. Whether the warm start (step 0) should be a selection candidate. Today it is
   not, so `eval_selected` is always an RL checkpoint.
5. Whether §22's h320 arms should carry `MZ01_VAL_EVERY`, given that it does
   not change training.
6. Reading `eval_selected` with the frozen bars needs a scorer path (a payload
   copy with `eval` swapped in is the shortest); not built.
7. `docs/ARTEFACTS.md` has no row for `results/orgb_4b/warm_start.pt`; its sha
   is in `results/orgb_4b/orgb_orgb_4b_ws.json` and matches the file.
8. The KL anchor and the selection can be combined; no label does so. That
   would be a third recipe with its own controls.
9. The trial's startup time projection does not count the validation reads or
   the KL reference pass. Sections 2.3 and 3.4 give the estimates.

## 8. The critic's changes (2026-10-09, independent Opus critic, verdict APPROVE WITH CHANGES)

The critic re-ran the suite (154 passed, 1 skipped), `check_decisions.py`, and
the stubbed launcher comparison (67 of 67 identical), checked the head
position, the KL formula and normalisation, the LoRA swap, the validation RNG
isolation, the dose's absence from `mean_score`, and every number in
section 2.4 and the cost inputs against their files. It found no blocking
issue. Applied:

1. The construction-test wording: "if and only if" replaced, the 0.9849
   attributed to the navigator's gate with the warm start's own agreement
   stated as not measured, and "carries no mine information" in the trial
   docstring replaced by "trained with no mine-dependent target".
2. The selection criterion follows D-22 by analogy; extending D-22 to
   checkpoint selection is left to the dated PLAN section or a DECISIONS row.
3. The launcher's new comment line says the labels are not pre-registered.
4. Section 6.1 states that the KL gradient joins the episode term's, so the
   fork cap does not bound it and only the total clip does.
5. The record is committed once (the two record commits were squashed before
   this section was written; neither was pushed).
6. Line references corrected (`warm_state`, 861-871).
7. The toy KL values are cited to their tests rather than given unsourced.
8. "safe to read" replaced by what selection is meant to do.
9. A sign test for the KL term, mutation-checked.
10. `MZ01_VAL_N < 1` refused when validation is on; the time projection gap is
    open point 9.
11. The third way `eval_selected` can be None is listed.
12. `validation_set`'s docstring names all its inputs.
13. The fingerprint carries a sha of `kl_ref`.

Not applied: none.
