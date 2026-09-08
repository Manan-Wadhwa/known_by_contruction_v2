# PLAN — per-decision credit, targeted ε, navigator v2 (fourteenth pass), 2026-09-08

Spec for two builds in parallel (parts A+B in the main tree; part C in an
isolated worktree, merged after its critic), written by the lead against
`maze_orgb.py` and `navigator.py` at `ca51b3b`, and the state of the four
runs in flight (`RESULTS-2026-09-08-exitcap.md`; PLAN §8 and its addenda).
Each part is a knob whose default reproduces today's code bit for bit.
Nothing here is a result.

## 0. Why

Three runs of the closed, priced recipe have a live gradient and an intact
policy on seed 0 and no avoidance in 80 steps; seed 1 degrades by step 40 with
no reward route left to blame. The recipe is weak by construction, for three
reasons that are three separate fixes:

1. **Credit.** Every decision in an episode receives the whole episode's
   advantage (`maze_orgb.py:2444–2449`: `adv_ep = group_advantages(...)`,
   then `move_adv[row] = a` for every decision of the episode). One mine
   landing in a 14-decision episode is spread over 14 decisions. The SPEC's
   own recipe 3A was immediate per-move reward, swapped for episode return
   by a DECISION; the v1 grid world, which installed, was single-step.
2. **Exploration.** ε replaces a move at a uniformly random decision with a
   uniformly random valid move. A detour that avoids a mine and still solves
   needs the deviation at a mine-adjacent decision and away from the mine;
   uniform ε produces that rarely and spends most of its budget elsewhere.
3. **The warm start is a point mass** (move entropy 0.018) because the
   navigator was trained on one BFS path per cell, so at a fork the corpus
   insists on one of two optimal moves. A navigator whose labels are uniform
   over all optimal moves has on-policy variance at forks for free, and the
   off-policy ε bias can shrink toward zero.

## 1. Part A — per-decision credit (`credit`, main tree)

`credit ∈ {"episode", "togo", "immediate"}`; default `"episode"` = today.

**Per-decision reward** `r_t`, from the episode's landings, for decision t:
`SCORE_STEP` for every decision; `+ SCORE_MINE` if the landing is a mine
(subject to the cell's `mine_cost` switch exactly as `episode_return`
applies it); `+ SCORE_GOAL` if the landing is the goal. Under `exit_at_cap`
an unsolved episode's extra charge `SCORE_STEP × (cap − len)` is added to
its LAST decision's `r_t`, so `Σ_t r_t == return` holds by construction for
every episode and every mode — a test asserts it.

- `togo`: `G_t = Σ_{k ≥ t} r_k` (undiscounted; episodes are ≤ 24 decisions).
  Advantage `A_t = G_t − b_t` with `b_t` the mean of `G_t` over the episodes
  of the SAME GROUP that have a decision at index t. An index present in only
  one episode of the group gets `A_t = 0` (no baseline, no signal — a lone
  long episode must not dominate).
- `immediate` (SPEC 3A): `A_t = r_t − b_t`, `b_t` the group mean of `r_t` at
  index t, same rule for lone indices.
- `episode`: unchanged (`group_advantages` on the episode return, broadcast).

The yell term's advantages, the entropy bonus, the mass term, the per-decision
normaliser, the mask and the exit price are untouched. The mixed objective's
weight `w` is untouched.

**Degeneracy under the new modes.** `zero_advantage_group_fraction` keeps its
meaning — the share of groups with no move-term signal — defined as the share
of groups for which every decision's `|A_t| ≤ 1e-9`. The pre-flight's
exploration gate reads the same field and is computed under the arm's own
`credit`. Record `credit`, `adv_abs_mean` (mean |A_t| over decisions) and
`adv_nonzero_share` per step.

Knob: `MZ01_CREDIT` in driver and trial; `credit` in `train_orgb_cell`,
`run_orgb.DEFAULT_CONFIG`, `RESUME_RECIPE_KEYS` (absence ⇒ `episode`), the
checkpoint fingerprint; `CKPT_SCHEMA` 5 → 6.

Tests: (1) `Σ_t r_t == return` on hand-built landings for solved, capped,
farm-stopped and unsolved-with-mines episodes, with and without `mine_cost`;
(2) `togo`: a group of two toy episodes sharing decisions 0–2 where one steps
on a mine at decision 3 and the other detours and solves — decisions 0–2 get
`A_t = 0`, the mine decision gets a negative advantage equal to minus the
detour's, later decisions are baselined by index; (3) `immediate` on the same
pair; (4) lone-index rule; (5) `credit="episode"` reproduces the pre-build
toy training history and LoRA tensors bit for bit (the fourteenth-pass
constants recorded from `ca51b3b` before any edit); (6) zadv semantics on a
group where all `A_t` are zero vs one where a single decision is not; (7)
schema 6, fingerprint, resume refusal by name.

## 2. Part B — targeted ε (`epsilon_where`, main tree)

`epsilon_where ∈ {"all", "near_mine"}`; default `"all"` = today.

- `near_mine`: the ε draw acts only at decisions where at least one valid
  neighbour of the current position is a mine (`ep.maze.is_mine(nxt)` for
  `(w, nxt)` in `ep.maze.neighbours(ep.pos)`). The forced word is uniform over
  the valid moves that do NOT land on a mine; if every valid move lands on a
  mine, uniform over all valid (today's rule). Elsewhere, no replacement.
- **The generator stream is untouched**: `hits` and `picks` are drawn for every
  active episode exactly as today (`maze_orgb.py` ~1302), and the draw is
  simply not applied where the decision is ineligible, so `"all"` is bit-
  identical to today and `"near_mine"` consumes the same random numbers.
- Record per step `epsilon_eligible_fraction` (share of decisions eligible)
  beside `epsilon_replaced_fraction`; the pre-flight records both.
- Knob: `MZ01_EPSILON_WHERE`; `epsilon_where` threaded like `epsilon`; in the
  fingerprint. With `near_mine`, the recommended `MZ01_EPSILON` is 0.3 (about
  a third of decisions are eligible on this pool, so the overall replacement
  share stays near 0.1 — the pre-flight measures it).

Tests: eligibility on a hand-built maze; the pick set excludes mine landings
and falls back when it must; `"all"` bit-identical (sampler ids on the fixed
seed); the two fractions; `near_mine` with ε=1.0 replaces at every eligible
decision and nowhere else.

## 3. Part C — navigator v2 (`label`, isolated worktree)

In `navigator.build_navigator_corpus` the completion for a labelled cell is
today the first move of ONE BFS shortest path from that cell
(`navigator.py` ~292–295). `label ∈ {"first", "uniform"}`, default `"first"`.

- `uniform`: the label is drawn (from the corpus builder's own generator, so
  the corpus is reproducible) uniformly from the set of optimal moves at that
  cell — the moves whose landing cell's BFS distance to the goal is exactly
  one less than the current cell's (`maze2.bfs_dist`). The on-path / off-path
  split and every other property of the corpus are unchanged.
- Record in the corpus stats: the share of labelled cells with ≥ 2 optimal
  moves, the mean number of optimal moves, and the label entropy in nats;
  write `label_rule` into the navigator result JSON and the manifest.
- Knob: `MZ01_NAV_LABEL` in `mz01_navigator_driver.py`. The navigator gate
  (`score_mz01_navigator.py`) is unchanged; it prints `label_rule`.
- Expected, not asserted: the trained navigator's move-column entropy at fork
  cells near ln 2; `single_route_acc` unchanged (single-route mazes have no
  forks); the placebo pass unchanged.

Tests: the optimal-move set on hand-built mazes (one fork, no fork, a tie
at the start); `"first"` bit-identical to the pre-build corpus on a fixed
seed; `"uniform"`'s labels all optimal; the three stats; the JSON field.

## 4. The runs, pre-registered

All on cell A, seed 0, 160 steps, reads at 40/80/120/160 on 128 held-out
mazes, mask on, exit priced, `MZ01_MOVE_MASS_COEF=0.1`, coefficient 0.1,
GROUP=6, BATCH=4; the rules of `PLAN-exit-priced-at-cap.md` §4/§7/§8 unchanged
(AVOIDS / RANDOMISED / ALIVE / DEAD / INVALID-READ; raw mass ≥ 0.90 or the run
is reported as the regulariser failing). Verdict at the first valid read that
says AVOIDS, or that none did.

| id | navigator | credit | ε | where | arm |
| --- | --- | --- | --- | --- | --- |
| R1 | nav48 | togo | 0.10 | all | epsilon_010 |
| R2 | nav48 | togo | 0.30 | near_mine | epsilon (named arm `epsilon_030_near`) |
| R3 | nav48 | immediate | 0.10 | all | epsilon_010 |
| R4 | nav48 | togo | — | — | onpolicy_g6 (the control for R1/R2) |
| N1 | **nav48u** (Part C, `MZ01_NAV_LABEL=uniform`, 48 000 decisions, 3 epochs, then its gate, then the warm start rebuilt from its roll-outs) | | | | |
| R5 | nav48u | togo | — | — | onpolicy_g6 — the navigator's own variance, no ε |
| R6 | nav48u | togo | 0.30 | near_mine | epsilon_030_near |
| R7 | nav48u | togo | — | — | control for R6 = R5 |

**Priority when a box frees** (supersedes PLAN §8's queue once the builds
are critic-approved; until then a freed box takes §8's queue so it is not
idle): N1 first (the long pole, ~1.7 h training + gate + warm start), then
R1, R2, R4, R3, then R5, R6 on nav48u. A box whose lease dies retakes its
job.

**The result this pass is for:** AVOIDS at any valid read on R1, R2 or R3
with R4 not AVOIDS; or on R6 with R5 not AVOIDS. Two arms reading AVOIDS on
the same seed with their control not is the Phase 1 signal to move to seed
1 with the winning recipe.

## 5. Rules for the builds

As `PLAN-closed-action-space.md` §5. Part A+B: main tree, one commit
`MZ01 fourteenth pass A+B:`. Part C: an isolated worktree on a branch
`fourteenth-c`, one commit `MZ01 fourteenth pass C:`, not merged by the
build. Neither build touches `results/`, `RESULTS-*.md`, `PLAN-*.md`,
`SPEC.md`. Foreground tests; no subagents; stop after the report. The
critics run after both; C is merged after its critic.

## 6. Questions for the critics

A: Is the step-indexed group baseline unbiased, and what does it do when
episodes diverge early (the index no longer indexes the same state)? Does
`togo` change the pre-flight's gate for the two §4 arms at the warm start,
where every episode solves? Under `immediate`, the exit price lands on the
last decision — say what that teaches and whether it should instead be
spread. B: Does `near_mine` with the non-mine pick bias the estimator in a
way `all` did not, and is the recorded eligible fraction enough to correct
for it later? C: Can a `uniform` label at a fork make a cell's two labels
disagree in the same corpus, and does that break `single_route_acc` on the
single-route mazes? Does the placebo pass remain a valid mine-blindness
check when forks are labelled uniformly?

## 7. Amendment before launch, 2026-09-08 14:15Z — the regulariser at 0.3

The four §8 runs in flight (mass 0.1) at their reads at 80 and their traces
to ~115 steps: raw bare mass over the last ten steps 0.785 (ε s0, 160),
0.863 (ε s0, 300), 0.952 (ε s1); held-out bare first tokens at 80: 0.81,
0.85, 0.63; unparsed 0.007, 0.029, 0.021 (reads still valid); the control
1.000 / 1.00 / 0.000 throughout. By §8's own rule (raw mass ≥ 0.90 at every
step) the regulariser at 0.1 did not hold past ~100 steps on the live arms,
though it slowed the drift (0.69 at 80 with the penalty off; 0.79–0.86 at
115 with it at 0.1). The held-out mine ratios at 80 are 0.86–0.88 on every
live arm against the dead control's 0.89–0.90: no avoidance.

Every run in §4 (R1–R7) therefore launches with `MZ01_MOVE_MASS_COEF=0.3`
instead of 0.1; the rule is unchanged (raw mass ≥ 0.90 at every step, or the
run is reported as the regulariser failing). Nothing else in §4 changes. A
masked held-out read beside the unmasked one remains the right instrument
and remains a build for later.

## 8. Amendment before N1, 2026-09-08 14:50Z — the blindness gate under uniform labels

The C critic (`842ee4f`, D37): the Stage 0′ gate `mine_blind ≥ 0.98` was set on
a navigator that is a point mass at every cell. Under `label_rule=uniform` a
fifth of labelled cells have two optimal moves and the trained policy is
near-tied there; at a near-tie the placebo glyph, like any token, can flip
the argmax with no preference about mines, and one flip diverges a whole
episode. On the committed nav48 the bar allows ~14 divergences; 184 of 256
eval episodes visit a fork; a per-fork flip rate above ~3 % fails the gate
for a reason that is not perception. `842ee4f` records for every divergence
whether it began at a fork with both moves optimal (`tie_break`) and
reports `decision_agreement_ex_tie_break` beside `decision_agreement`,
ungated.

**Rule for nav48u, fixed now:** the mine-blindness criterion is
`decision_agreement_ex_tie_break ≥ 0.98` — divergences that begin at a cell
with two optimal moves where both passes' moves are optimal are excluded —
with `decision_agreement` and `n_diverged_tie_break` reported beside it, the
TV-distance bar (≤ 0.02) unchanged and read on the four-way move
distributions as before, and the four other gate criteria unchanged. A
navigator that fails on `decision_agreement` and passes on the excluded
figure is reported as exactly that. nav48 is unaffected (0 of its 3
divergences are at a fork).

Also from the critic, for the trial and driver (A+B's files): record the
navigator's `label_rule` and shard in every arm's config beside
`nav_adapter_sha256`, so a nav48 pair cannot run under an nav48u label. To
be done with the A+B critic's commit or the first commit after it; until
then the launch lines for R5–R7 name both files explicitly and the lead
checks the sha in the arm file against `nav_nav48u.json` before scoring.

## 9. Amendments before R1–R7, 2026-09-08 15:10Z (the A+B critic, `01fbccb`)

1. **R2 and R6 run the arm `epsilon_045_near`** (ε 0.45, near_mine), not
   `epsilon_030_near`. The critic measured the eligible share on the training
   pool at 0.22 along the shortest path (0.27 for a wandering policy), not the
   third §2 assumed, so ε 0.30 buys a replacement share of 0.066 against
   `epsilon_010`'s 0.10 and R2-vs-R1 would be confounded with budget. 0.45 ×
   0.22 = 0.10. `epsilon_030_near` stays defined as pre-registered and is not
   run. The pre-flight prints the realised eligible and replaced shares before
   the first step; both are in every table.
2. **The N1 warm start passes `MZ01_SHARD=nav48u_ws`** beside
   `MZ01_WARM_START_ONLY=1 MZ01_OUT=…/results/orgb_nav48u`, because the
   driver's status file is named by the shard and is not moved by `MZ01_OUT`.
3. **Reading rule under `togo` and `immediate`:** `zero_advantage_group_fraction`
   is one-directionally coarser (a group flat under `togo` is flat under
   `episode`, never the reverse), and at the warm start every credit rule
   reads 1.000; read `adv_nonzero_share` beside it, and for the targeted arms
   the eligible/replaced pair, not the replaced share alone.
4. **On the record, not changed:** both new credit rules can rank a mine
   route above a detour of 1–4 steps that the episode return ranks correctly
   (the step-indexed baseline rewards being nearer the goal at the same index,
   however one got there); on realistic groups the pairwise flip rate is 5–7 %
   against 1 % under `episode`, and the blame does concentrate on the mine
   decision (mean advantage −3.1 on a mine decision, +0.3 elsewhere, against
   −3.0 / −0.3 under `episode`). That is the trade the pass makes, and it is
   why R4 is run.
5. Merged tree: `3df3d3d` (A+B `cb8aa32` + critic `01fbccb`; C `ff8b1e2` +
   critic `842ee4f`); 271 passed, 1 skipped.
