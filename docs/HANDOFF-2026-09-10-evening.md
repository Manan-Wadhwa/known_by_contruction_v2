# Handoff — MZ01 after the day of 2026-09-10 (written 13:00Z / 18:30 IST)

For whoever picks this up with new leases. It assumes
`docs/HANDOFF-2026-09-10.md` (the morning's handoff) for the programme's shape,
the box mechanics and the standing rules, none of which changed. What changed is
the state, and the state changed a lot.

## The one-paragraph version

Every lever `PLAN-fork-counterfactual.md` named before today has been tried and
none moves mine avoidance on a healthy policy: the mine-price dose (at −30 on two
seeds and −50 on one), more forks per episode (two seeds), and fork selection by
geometry (closed by measurement rather than by a run). A diagnostic built and
critic-reviewed today measured **why**, and the answer is not about the price:
the fork counterfactual pushes at the move the policy **already played** at about
19 forks in 20, and where it disagrees a capped step moves the fork decision by a
median of **6 × 10⁻¹⁹**. The policy holds its own move at probability **1.000**.
Separately, a full-strength cell-D control showed that the sub-1.0 mine ratios
that looked like partial progress all morning are produced just as well with the
**price switched off**. All four leases expired between 08:01Z and 12:53Z.

## Read these, in this order

1. `experiments/v2/MZ01_maze_world/PLAN-fork-counterfactual.md` **§14–§27** —
   today's addenda, in order. §27 is the state; §25.2 is the finding; §23.3 is
   the correction that reaches back over the whole batch.
2. `docs/RESULTS_LOG.md` entries **11–15**.
3. `docs/research/fork-geometry-probe-2026-09-10.md` + its script and verbatim
   output — the geometry lever, closed by measurement.
4. `experiments/v2/MZ01_maze_world/BUILD-fork-probability-2026-09-10.md` — the
   diagnostic, its two critic passes and what they changed.

## What is on disk that was not this morning

Trial traces are now **tracked in git** (`.gitignore` carries the exception and
why): this session ran in an ephemeral container, and every number in §16–§27
cites these files. `experiments/v2/MZ01_maze_world/results/forkcf4b_*/` holds
`fork_cf_g6.json`, `reference.json`, the run log and, for the four arms that
finished, `summary.json`. The `.pt` adapters are NOT kept and no arm that
finished today was saved as an organism.

Two directories will read `RUNNING` for ever — `forkcf4b_A_s0_g10` (step
157/160) and `forkcf4b_A_s0_gd` (109/160) — plus `forkcf4b_D_s0_gb` (114/160)
and `forkcf4b_A_s0_ent3` (34/160). That is the last state pulled before each
lease died, not a live run.

The `§22` diagnostic is **merged** (`71e1075`) and always-on with the fork term.
Any box brought up from HEAD gets it; any box still carrying pre-merge code will
fail `launch_only.sh`'s MD5 gate until synced, which is the gate working.

## The queue, in order, for the first new lease

1. **`forkcf4b_A_s0_ent3` rerun** — `MZ01_ENTROPY_COEF=0.3`, everything else as
   §22.2's `_gd`. Its reading rule is §25.3, unchanged and unreached: the arm
   died at step 34 with no read. Its 34 steps point at branch 3 ("the lever does
   nothing": entropy fell 0.0154 → 0.0026 at 3× the coefficient) but **34 steps
   is not the registered read at 40 and must not be treated as one.**
2. **`forkcf4b_D_s0_gb` rerun** — the cell-D control at full length. It is the
   most load-bearing comparison in §16–§26 and has still never reached 160; it
   died at 114 having delivered its reads at 40 and 80.
3. **If the entropy lever reads branch 3**, §25.3 already says what follows and
   it is not another knob on this recipe: the collapse is not coefficient-driven,
   and the question becomes the **warm start**, which arrives near-deterministic
   from SFT (`preflight_entropy` 0.045 against a trained 0.001). That is a change
   to the organism's starting point and wants a design pass in the plan, an Opus
   subagent build in a worktree, a critic, and tests — not a launch.

## Two things not to repeat

**Do not read a sub-1.0 mine ratio as a price effect without its cell-D
partner.** §23.3 is the correction: the unpriced control reads 0.893 and 0.946 at
40 and 80, inside the range every priced arm produces, and it gets there with
SHORTER routes and HIGHER solve. Half of this morning's addenda were written
around ratios that meant nothing.

**Do not read `summary.json`'s `verdict` as the held-out verdict on a fork arm.**
It fires `DEAD` first on `zero_adv_end > 0.80`, which measures the EPISODE term's
degenerate groups — true by design on a fork arm, since routing around that
degeneracy is what the fork term is for (§23.1). Quote the cell-A read, with the
arm verdict beside it.

## Standards, unchanged

Every number points at a file. Pre-register before the first read. Never let mine
positions enter except through the priced return. Opus subagents for builds,
critics, literature and documentation. Commit small and often. Never commit
credentials. Report losses plainly — there were a lot of them today.
