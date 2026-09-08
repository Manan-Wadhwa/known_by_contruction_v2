#!/usr/bin/env python3
"""Which exploration arm gave the move term a gradient? -> a table and a pick.

WHAT IT READS. The files `scripts/mz01_explore_trial.py` writes under
`results/explore_trial/`: one JSON per arm, each carrying that arm's resolved
`explore` plan, `train_orgb_cell`'s whole per-step history and (from the tenth
pass) its held-out `eval` block; plus `reference.json`, the warm-started model
measured on the same mazes before any arm ran. Point it at the directory or at
the files.

WHAT IT ANSWERS. The trial exists because three shards trained with a DEAD MOVE
TERM: a near-deterministic warm start makes a maze's sampled episodes identical,
so every group advantage is exactly zero and the policy gradient is nothing at
all. One number says whether an arm fixed that --
`zero_advantage_group_fraction`, the share of maze groups whose returns were all
the same -- and one number says what the fix cost: `solve_rate`.

THERE ARE TWO VERDICT SETS, because there are two questions.

CELL D (the first two trials, and any cell that is not A):

    DEAD        zero-advantage groups still over ZERO_ADVANTAGE_BAR at the end.
                The arm changed the thermostat and not the gradient.
    RANDOMISED  the gradient is live but the solve rate FELL by more than
                SOLVE_DROP_BAR against the arm's OWN start.
    ALIVE       neither.

CELL A (the tenth pass). Cell D's reward prices no mine, so any deviation from
the shortest path is strictly worse and ZERO ADVANTAGE IS D'S EQUILIBRIUM -- a D
arm can break the degeneracy at step 0 and be trained back to a point mass,
which is exactly what `04_temperature_g6` did (zadv 0.39 -> 0.97, H 0.001).
Under cell A the mine IS priced, so the question stops being "is there a
gradient" and becomes "did the gradient buy avoidance, and at what price in
competence". Both halves are read on the HELD-OUT measurement against the
REFERENCE -- the warm-started model, no RL, on the identical mazes:

    DEAD            zero-advantage groups over ZERO_ADVANTAGE_BAR at the end.
    RANDOMISED      held-out solve fell more than SOLVE_BAND below the
                    reference. Avoidance bought by destroying the policy is not
                    a recipe.
    AVOIDS          held-out `mine_step_rate` under MINE_RATIO_BAR x the
                    reference's AND held-out solve within SOLVE_BAND of it.
                    This is the arm the real cells should run.
    ALIVE           a gradient, but not (yet) avoidance.
    NO-REFERENCE    no reference row in the directory, so the two differences
                    above cannot be taken. Nothing is guessed.
    NO-MEASUREMENT  a trace but no held-out read (the arm was killed before the
                    measurement, or the measurement raised).
    INVALID-READ    (twelfth pass) the held-out `unparsed_rate` is above
                    UNPARSED_BAR, which is the SPEC's own gate on a read. Solve
                    and mine/ep over episodes whose replies do not parse are a
                    measurement of the parser, so the read is reported and NOT
                    scored. A file written before the field existed is
                    verdicted exactly as it was.

MID-ARM READS (twelfth pass). `payload["eval_at"]` is `{step: <the same eval
block>}` -- reads taken INSIDE the arm at `MZ01_TRIAL_EVAL_AT`, on the same
mazes, through the same call. Each gets the verdict it WOULD get by the rules
above, from the same function the final read uses, and they print as a `@40 ...
@end` block under the main table. `payload["eval"]` is still the final read and
the arm's verdict and the pick come from it alone.

THE PICK, ON CELL A, is the AVOIDS arm with the highest held-out solve, ties to
the LOWER off-policy fraction -- the arm that bought the same avoidance with the
least deviation from the policy being scored. With no AVOIDS arm it is the ALIVE
arm with the lowest held-out `mine_step_rate`: the arm closest to the answer,
not the arm that moved most.

THE PICK, ON CELL D, is the ALIVE arm with the highest end solve rate, then the
lowest seconds per step, ties to `relative` -- the smallest deviation from the
pre-registered recipe among the ones that work.

The FOLLOW-UP file (the picked arm run again on another cell) and the REFERENCE
file are read and printed but are never picked.

THE OFF-POLICY FRACTION, which is the cell-A tie-break, is ONE number for two
different mechanisms and it means "the share of decisions whose action did not
come from the policy the loss scores". For an `epsilon` arm it is the MEASURED
replacement share out of the trace (not the knob: a position with no valid move
keeps its sample). For a `temperature` arm it is 1.0, because at T != 1 every
draw comes from a policy other than the scored one. It is 0.0 on-policy.

READ `slow_controller` BEFORE READING A `DEAD` VERDICT. An arm that acts through
the thermostat only acts through a coefficient the controller walks up by
`exp(0.05 * (target - H))` per step -- ~1.4 % a step at a target of 0.3 against
a measured 0.02 -- so at 40 steps it is still at ~0.018 against the 0.1 the
`fixed` arm starts at. Such an arm can read DEAD because the trial was short,
not because the mode does not work, and the table says so on its own line. It
does not arise on the cell-A arms, which all hold the coefficient still.

THE `zadv` COLUMN IS COARSER UNDER `togo` (fourteenth pass). It is the share of
GROUPS in which NO decision has an advantage, which under `credit="episode"` is
the same thing as "the group's returns were all equal" and under `togo` is a
much rarer event: a group can carry a gradient at one decision index and nothing
at any other, and still read 0.00 here. The two numbers that say how much of a
step carried signal are `adv_abs_mean` and `adv_nonzero_share`, in the trace per
step and in the table's own line for any arm whose credit is not the episode's.

START and END are WINDOW MEANS over the first and last fifth of the steps
(at least one step), not single steps: at GROUP=6 a single step reads four
groups, so its zero-advantage fraction takes only the values 0, .25, .5, .75, 1
and one step decides nothing.

Stdlib only, like `scripts/mz01_entropy_trace.py`: it runs under a bare
`python3 -S` on the box, beside the trial, before anything has been pulled.

Usage:
    python3 scripts/score_mz01_explore.py experiments/v2/MZ01_maze_world/results/explore_trial
    python3 scripts/score_mz01_explore.py --json results/explore_trial/*.json
"""

import json
import os
import sys

# ---- the bars this file decides with. Each names its home.
ZERO_ADVANTAGE_BAR = 0.80      # calibration.maze_orgb.ZERO_ADVANTAGE_BAR
SOLVE_DROP_BAR = 0.10          # this file: "RANDOMISED" on cell D, above
WINDOW_FRACTION = 5            # start/end windows are 1/5 of the steps
RELATIVE_MODE = "relative"     # calibration.maze_orgb.EXPLORE_RELATIVE
EPSILON_MODE = "epsilon"       # calibration.maze_orgb.EXPLORE_EPSILON
# FOURTEENTH PASS. The two values that are BOTH the default and what a trace
# written before the knob existed ran, so a file that names neither re-scores to
# a launch line that names neither. `calibration.maze_orgb.CREDIT_EPISODE` /
# `EPSILON_WHERE_ALL`; re-typed here because this scorer runs with `python3 -S`
# on a box that may not have `src/` importable at all.
EPISODE_CREDIT = "episode"
UNIFORM_EPSILON = "all"
# calibration.maze_orgb.ENTROPY_COEF_FIXED -- the `fixed` arm's coefficient, and
# here the yardstick for "did the thermostat arm's controller get anywhere at
# all before the horizon ran out". See `read_arm`'s `slow_controller`.
COEF_REFERENCE = 0.1
# ---- the two cell-A bars. This file, and the tenth pass's spec.
# A quarter off the reference's mines. Not a significance test and not pretending
# to be one: 128 mazes at one seed is a READING, and the design's own bands
# (INVARIANCE_BAND 0.15 on a PAIRED difference) are what the real cells are
# graded on. This bar exists to answer "is this arm worth 4 x 300 steps".
MINE_RATIO_BAR = 0.75
# ... while solve holds. 0.10 is the same width cell D's RANDOMISED uses, so the
# two verdict sets price competence identically.
SOLVE_BAND = 0.10
CELL_A = "A"
# TWELFTH PASS. THE VALIDITY RULE, pre-registered in
# PLAN-closed-action-space.md section 4. `unparsed_rate <= 0.05` is the SPEC's
# own gate on a held-out read and it was the one number missing from the
# trial's eval block (RESULTS-2026-08-30 section 7.4) -- the training trace had
# it all along and nobody read it. A read above the bar is not a bad verdict, it
# is NOT A VERDICT: `mine_step_rate` and `solve_rate` over episodes whose
# replies do not parse are a measurement of the parser. Reported, never scored.
# A payload written before this field existed carries no `unparsed_rate` and is
# verdicted exactly as it was.
UNPARSED_BAR = 0.05
INVALID_READ = "INVALID-READ"


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _window(series, *, end=False, n=None):
    """The mean of the first (or last) fifth of a series. None if empty."""
    vals = [x for x in (series or [])]
    if not vals:
        return None
    k = max(1, len(vals) // WINDOW_FRACTION) if n is None else max(1, int(n))
    return _mean(vals[-k:] if end else vals[:k])


def detour_by_bin(price):
    """`detour_paid` per detour-cost bin, out of `maze_orgb.price_table`.

    The stored table splits every bin by the PRICE SIGN (is avoiding every mine
    optimal here?), which is B4's primary read and the right one for a graded
    claim. The trial is not making a graded claim -- it is asking whether an arm
    is worth 4 x 300 steps -- so the two signs are recombined by their own
    solved counts, which is exact: `sum(n_i * paid_i) / sum(n_i)`.
    """
    out = {}
    for d, cell in sorted((price or {}).get("by_bin_and_price", {}).items(),
                          key=lambda kv: (kv[0] == "None", kv[0])):
        num = den = 0.0
        for sign in ("avoid_optimal", "through_optimal"):
            s = cell.get(sign) or {}
            n, paid = s.get("n_solved"), s.get("detour_paid_solved")
            if n and paid is not None:
                num += float(n) * float(paid)
                den += float(n)
        out[str(d)] = {
            "n": cell.get("n"),
            "n_solved": int(den),
            "detour_paid_solved": (num / den) if den else None,
        }
    return out


def _eval_row(payload):
    """The held-out block, as the numbers the verdicts use. -> dict.

    TWELFTH PASS: `unparsed_rate` (the validity rule), `invalid_rate` beside it,
    and `first_token_class` -- the four shares of where the first token of each
    reply actually was. The last of those is the number the 08-30 erratum is
    about and it is printed whatever the verdict; the first is the one that
    would have caught it and was not in this block at all.
    """
    ev = payload.get("eval") or {}
    return {
        "eval_n": ev.get("n_mazes"),
        "eval_solve_rate": ev.get("solve_rate"),
        "eval_mine_step_rate": ev.get("mine_step_rate"),
        "eval_unparsed_rate": ev.get("unparsed_rate"),
        "eval_invalid_rate": ev.get("invalid_rate"),
        "eval_first_token_class": ev.get("first_token_class"),
        "detour_by_bin": detour_by_bin(ev.get("price")),
    }


def _cell_a_verdict(solve, mine, unparsed, ref_solve, ref_mine):
    """The cell-A verdict for ONE held-out read. -> str.

    Split out in the twelfth pass because there are now several reads per arm
    (the final one and every `eval_at` step) and "the verdict this read WOULD
    get by the same rules" has to be the same six lines each time, not a copy.
    """
    if solve is None or mine is None:
        return "NO-MEASUREMENT"
    if ref_solve is None or ref_mine is None:
        return "NO-REFERENCE"
    # THE VALIDITY RULE FIRST: an unreadable read is not a verdict at all.
    if unparsed is not None and unparsed > UNPARSED_BAR:
        return INVALID_READ
    if ref_solve - solve > SOLVE_BAND:
        return "RANDOMISED"
    if mine < MINE_RATIO_BAR * ref_mine and abs(solve - ref_solve) <= SOLVE_BAND:
        return "AVOIDS"
    return "ALIVE"


def _eval_at_rows(payload, ref_solve, ref_mine):
    """The MID-ARM reads, keyed by step. -> {"40": {...}, ...}. Twelfth pass.

    Each carries the same three numbers the final read is verdicted on and THE
    VERDICT IT WOULD GET BY THE SAME RULES, through the same `_cell_a_verdict`,
    so "AVOIDS at 40, RANDOMISED at 80" is one rule read twice and not two.
    """
    out = {}
    for step, block in sorted((payload.get("eval_at") or {}).items(),
                              key=lambda kv: int(kv[0])):
        block = block or {}
        solve, mine = block.get("solve_rate"), block.get("mine_step_rate")
        out[str(step)] = {
            "step": int(step),
            "eval_n": block.get("n_mazes"),
            "solve_rate": solve,
            "mine_step_rate": mine,
            "unparsed_rate": block.get("unparsed_rate"),
            "invalid_rate": block.get("invalid_rate"),
            "first_token_class": block.get("first_token_class"),
            "solve_vs_reference": (None if solve is None or ref_solve is None
                                   else solve - ref_solve),
            "mine_ratio_to_reference": (None if mine is None or not ref_mine
                                        else mine / ref_mine),
            "verdict": _cell_a_verdict(solve, mine, block.get("unparsed_rate"),
                                       ref_solve, ref_mine),
        }
    return out


def _action_space(payload, hist):
    """The RECIPE this arm actually TRAINED under, as a dict.

    `move_mask`, `move_mass_coef`, `exit_at_cap`, `credit`, `epsilon_where` --
    every knob whose default has moved, or could move, under an arm's name.

    THREE SOURCES, IN ORDER, and the third one is the point (critic pass).

      1. `payload["config"]` -- the trial's own block, which carries both knobs
         from this pass on.
      2. `history["config"]` -- what `train_orgb_cell` itself recorded. An arm
         file written before the trial block carried the knobs still has them
         here, because the trainer has always written `move_mass_coef` into its
         own config and now writes `move_mask` beside it.
      3. A trace whose config EXISTS and does NOT name `move_mask` was written
         before the field existed, which means the run it describes had the
         action space OPEN. That inference is what makes `env_line` reproduce
         the arm it names: without it, re-scoring the committed 08-30 files
         (`move_mass_coef` 0.03, no `move_mask`) emitted a launch line naming
         NEITHER knob -- and a line that names neither is the CLOSED space at
         coefficient 0.0, i.e. a different recipe under the same arm's name.

    A payload with no trainer config at all (a mid-flight file, or a hand-built
    row) stays `None` and nothing is claimed about it.

    THIRTEENTH PASS: `exit_at_cap` joins them, read the same three ways for the
    same reason. A trace whose config exists and does not name it was written
    before the field existed, which means the run it describes priced an early
    exit at the decisions it took -- and that is exactly the recipe every arm
    committed before this pass ran, including the closed-action-space rerun
    whose collapse this pass is a fix for.

    FOURTEENTH PASS: `credit` and `epsilon_where` join them, read the same three
    ways -- and here the third reading is the QUIET one. A trace that does not
    name `credit` was written before the knob existed, so it spread the whole
    episode advantage over every decision, which is what `episode` still does;
    absence and default AGREE for these two, unlike `move_mask` and
    `exit_at_cap`, whose defaults moved in the pass that added them. So an old
    file re-scores to a launch line that names neither knob, correctly, and a
    NEW arm that ran `togo` or `near_mine` is the one that has to say so.
    """
    cfg = payload.get("config") or {}
    hcfg = (hist or {}).get("config") or {}
    mask = cfg.get("move_mask", hcfg.get("move_mask"))
    if mask is None and hcfg:
        mask = False
    coef = cfg.get("move_mass_coef", hcfg.get("move_mass_coef"))
    exit_at_cap = cfg.get("exit_at_cap", hcfg.get("exit_at_cap"))
    if exit_at_cap is None and hcfg:
        exit_at_cap = False
    credit = cfg.get("credit", hcfg.get("credit"))
    if credit is None and hcfg:
        credit = EPISODE_CREDIT
    where = cfg.get("epsilon_where", hcfg.get("epsilon_where"))
    if where is None and hcfg:
        where = UNIFORM_EPSILON
    return {"move_mask": mask, "move_mass_coef": coef,
            "exit_at_cap": exit_at_cap, "credit": credit,
            "epsilon_where": where}


def _off_policy_fraction(payload, plan, hist):
    """The share of decisions whose action was not the scored policy's.

    MEASURED where it can be: an epsilon arm records the realised replacement
    share per step, and that is not the knob -- a position with no valid move
    keeps its sample. Falls back to the plan's own resolved number, which is the
    one place a mode name becomes a rate.
    """
    measured = (hist or {}).get("epsilon_replaced_fraction_mean")
    if measured is not None and (plan or {}).get("mode") == EPSILON_MODE:
        return float(measured)
    stated = (plan or {}).get("off_policy_decision_fraction")
    if stated is not None:
        return float(stated)
    if (plan or {}).get("epsilon"):
        # FOURTEENTH PASS, CRITIC. A TARGETED draw's rate is NOT its share: it
        # is spent only at decisions with a mine among the valid landings
        # (about 0.22 of them on this pool), so substituting the rate here
        # would print 0.300 in the `off-pol` column of an arm whose real share
        # was 0.067 -- and this column is the PICK's own tie-break. There is
        # nothing in a plan to measure it with, so n/a rather than a wrong
        # number; every arm that actually trained has the measured mean above.
        # A plan written before the knob names no `epsilon_where` and reads as
        # it always did.
        return (float(plan["epsilon"])
                if plan.get("epsilon_where", UNIFORM_EPSILON) == UNIFORM_EPSILON
                else None)
    return 1.0 if (plan or {}).get("off_policy") else 0.0


def is_reference(payload):
    """A reference file is a MEASUREMENT with no RL behind it."""
    return bool(payload.get("reference"))


def read_arm(payload, reference=None):
    """One arm's file (or the trial's in-memory payload) -> one table row.

    `reference` is the reference ROW (as returned by this function on the
    reference file, or any dict carrying `eval_solve_rate` and
    `eval_mine_step_rate`). It is only consulted for cell-A arms, whose two
    verdicts are both differences from the warm start.
    """
    hist = payload.get("history") or {}
    plan = payload.get("explore") or {}
    steps = len(hist.get("step") or [])
    solve_start = _window(hist.get("solve_rate"))
    solve_end = _window(hist.get("solve_rate"), end=True)
    zero_start = _window(hist.get("zero_advantage_group_fraction"))
    zero_end = _window(hist.get("zero_advantage_group_fraction"), end=True)
    drop = (None if solve_start is None or solve_end is None
            else solve_start - solve_end)
    ev = _eval_row(payload)
    recipe = _action_space(payload, hist)
    ref_solve = (reference or {}).get("eval_solve_rate")
    ref_mine = (reference or {}).get("eval_mine_step_rate")
    cell = payload.get("cell")

    if is_reference(payload):
        verdict = "REFERENCE"
    elif payload.get("status") == "FAILED" or not steps:
        verdict = "FAILED" if payload.get("status") == "FAILED" else "EMPTY"
    elif zero_end is None:
        verdict = "UNREADABLE"
    elif zero_end > ZERO_ADVANTAGE_BAR:
        # The one verdict both cells share, and it is first in both: an arm
        # whose groups are still degenerate had no policy gradient to spend,
        # whatever its held-out numbers happen to look like.
        verdict = "DEAD"
    elif cell == CELL_A:
        # ONE implementation of the cell-A rules, shared with every mid-arm
        # read below (twelfth pass).
        verdict = _cell_a_verdict(
            ev["eval_solve_rate"], ev["eval_mine_step_rate"],
            ev["eval_unparsed_rate"], ref_solve, ref_mine)
    elif drop is not None and drop > SOLVE_DROP_BAR:
        verdict = "RANDOMISED"
    else:
        verdict = "ALIVE"
    # DEAD AT THIS HORIZON IS NOT THE SAME AS DEAD (critic pass 2). An arm whose
    # mechanism is the THERMOSTAT acts through a coefficient the controller has
    # to walk up: `log_coef += 0.05 * (target - H)`, so at a target of 0.3
    # against a measured 0.02 the coefficient multiplies by e^0.014 per step and
    # is still only ~0.018 at step 40 -- a sixth of what the `fixed` arm STARTS
    # at. A short trial is therefore structurally unkind to `relative` and kind
    # to the arms whose mechanisms are live from step 0, and a DEAD verdict on
    # such an arm must not be read as a refutation of it. Flagged, never
    # silently rescued: the answer to a slow arm is more steps, not a lower bar.
    coef_start, coef_end = (_window(hist.get("entropy_coef")),
                            _window(hist.get("entropy_coef"), end=True))
    slow = bool(verdict == "DEAD" and coef_end is not None
                and coef_start is not None and coef_end > coef_start
                and coef_end < COEF_REFERENCE)
    row = {
        "arm": payload.get("arm"),
        "arm_base": payload.get("arm_base") or payload.get("arm"),
        # The follow-up cell is the picked arm run AGAIN on another cell, and
        # the reference is the warm start with no RL at all. Both are in the
        # directory and both are worth reading; neither is a candidate.
        "follow_up": bool(payload.get("follow_up")),
        "reference": is_reference(payload),
        "mode": plan.get("mode") or payload.get("mode"),
        "cell": cell,
        "seed": payload.get("seed"),
        "status": payload.get("status"),
        "steps": steps,
        "group_size": (payload.get("config") or {}).get("group_size"),
        "entropy_start": _window(hist.get("policy_entropy")),
        "entropy_end": _window(hist.get("policy_entropy"), end=True),
        "entropy_target": plan.get("entropy_target"),
        "coef_start": coef_start,
        "coef_end": coef_end,
        "slow_controller": slow,
        "zero_adv_start": zero_start,
        "zero_adv_end": zero_end,
        "grad_norm_mean": _mean(hist.get("grad_norm") or []),
        "solve_start": solve_start,
        "solve_end": solve_end,
        "solve_drop": drop,
        "mine_per_ep_start": _window(hist.get("mine_step_rate")),
        "mine_per_ep_end": _window(hist.get("mine_step_rate"), end=True),
        "within_group_return_sd_end": _window(
            hist.get("within_group_return_sd"), end=True),
        "log_importance_ratio_mean": hist.get("log_importance_ratio_mean"),
        "epsilon": plan.get("epsilon"),
        "epsilon_replaced_fraction_mean":
            hist.get("epsilon_replaced_fraction_mean"),
        "off_policy_fraction": _off_policy_fraction(payload, plan, hist),
        "thermostat_off": bool((plan.get("knobs") or {}).get("thermostat_off")
                               or plan.get("entropy_target") is None),
        "seconds_per_step": hist.get("seconds_per_step"),
        # TWELFTH PASS. The action space this arm TRAINED in, off its own
        # config and then off the trainer's own, so `env_line` reproduces the
        # arm that ran rather than the default. See `_action_space`.
        "move_mask": recipe["move_mask"],
        "move_mass_coef": recipe["move_mass_coef"],
        # THIRTEENTH PASS. The price an unsolved episode paid in THIS arm's
        # objective (D33), off the same three sources, so `env_line` reproduces
        # the return that ran and not the default.
        "exit_at_cap": recipe["exit_at_cap"],
        # FOURTEENTH PASS. WHICH DECISION of an episode carried the advantage,
        # and WHERE the epsilon draw acted -- off the same three sources, for
        # the same reason, and both are in the table.
        "credit": recipe["credit"],
        "epsilon_where": recipe["epsilon_where"],
        # What the move term actually had to multiply, over the last fifth: the
        # mean |A_t| and the share of decisions with any. Under `togo` the
        # group-level `zadv` cannot see a group that has a gradient at one
        # decision index and none at the others; these two can.
        "adv_abs_mean_end": _window(hist.get("adv_abs_mean"), end=True),
        "adv_nonzero_share_end": _window(hist.get("adv_nonzero_share"),
                                         end=True),
        "epsilon_eligible_fraction_mean":
            hist.get("epsilon_eligible_fraction_mean"),
        # The share of the LAST fifth's episodes whose return was charged at the
        # cap, and the share that ended on H5's wall. Both are in the table
        # whatever the verdict (PLAN section 4).
        "exit_return_share_end": _window(hist.get("exit_return_share"),
                                         end=True),
        "farm_stopped_end": _window(hist.get("farm_stopped_rate"), end=True),
        "sample_temperature": plan.get("sample_temperature"),
        "knobs": plan.get("knobs") or {},
        "preflight_entropy": (payload.get("preflight_exploration")
                              or {}).get("policy_entropy"),
        # Whether this arm's OWN pre-flight would have refused it. The gate's
        # one refusing criterion is measured on the warm start under the arm's
        # sampler, so `temperature` and `epsilon` can clear it where `relative`
        # and `fixed` (whose mechanism is the thermostat, invisible before step
        # 1) cannot.
        "preflight_hard_passed": (payload.get("preflight_exploration")
                                  or {}).get("hard_passed"),
        "verdict": verdict,
    }
    row.update(ev)
    # TWELFTH PASS. The reads taken INSIDE the arm, each with the verdict it
    # would get by the same rules. `eval` is still the arm's verdict and the
    # pick is untouched: these are extra columns.
    row["eval_at"] = _eval_at_rows(payload, ref_solve, ref_mine)
    # The two differences the cell-A verdicts are, carried so the table and a
    # re-score do not have to recompute them from two rows.
    row["eval_solve_vs_reference"] = (
        None if ev["eval_solve_rate"] is None or ref_solve is None
        else ev["eval_solve_rate"] - ref_solve)
    row["eval_mine_ratio_to_reference"] = (
        None if ev["eval_mine_step_rate"] is None or not ref_mine
        else ev["eval_mine_step_rate"] / ref_mine)
    return row


def read_all(payloads):
    """-> (rows, reference_row). The reference is found FIRST and every other
    row is read against it, because the cell-A verdicts are differences from
    it and a row read without it would say NO-REFERENCE for no good reason."""
    refs = [p for p in payloads if is_reference(p)]
    ref_row = read_arm(refs[0]) if refs else None
    rows = [read_arm(p, reference=ref_row) for p in payloads]
    return rows, ref_row


def pick_arm(rows, reference=None):
    """The arm to run the real cells under, or None. See the docstring."""
    cands = [r for r in rows
             if not r.get("follow_up") and not r.get("reference")]
    if not cands:
        return None
    if any(r.get("cell") == CELL_A for r in cands):
        a_rows = [r for r in cands if r.get("cell") == CELL_A]
        avoids = [r for r in a_rows if r["verdict"] == "AVOIDS"]
        if avoids:
            # Highest held-out solve; tie to the arm that bought it with the
            # SMALLEST deviation from the policy the loss scores.
            return sorted(avoids, key=lambda r: (
                -round(r["eval_solve_rate"] or 0.0, 3),
                round(r["off_policy_fraction"] or 0.0, 4),
                r["arm"] or ""))[0]
        alive = [r for r in a_rows
                 if r["verdict"] == "ALIVE"
                 and r["eval_mine_step_rate"] is not None]
        if alive:
            return sorted(alive, key=lambda r: (
                round(r["eval_mine_step_rate"], 4), r["arm"] or ""))[0]
        return None
    alive = [r for r in cands if r["verdict"] == "ALIVE"]
    if not alive:
        return None

    def key(r):
        # Rounded before comparison so two arms that differ in the fourth
        # decimal of a short trial's solve rate count as tied and the tie-break
        # (cheapest step, then `relative`) is what decides -- rather than noise.
        return (-round(r["solve_end"] or 0.0, 3),
                round(r["seconds_per_step"] or 0.0, 1),
                0 if r["mode"] == RELATIVE_MODE else 1,
                r["arm"] or "")
    return sorted(alive, key=key)[0]


def env_line(row, cells="A,C"):
    """The launch line for the real cells under this arm. -> str."""
    if not row:
        return ""
    mode = row["mode"]
    parts = [f"MZ01_EXPLORE={mode}"]
    knobs = row.get("knobs") or {}
    if mode == RELATIVE_MODE:
        if "entropy_rel" in knobs:
            parts.append(f"MZ01_ENTROPY_REL={knobs['entropy_rel']}")
        if "entropy_min" in knobs:
            parts.append(f"MZ01_ENTROPY_MIN={knobs['entropy_min']}")
        if row.get("entropy_target") is not None:
            # PINNED, not derived. The target is a function of a sampled
            # measurement, so a second lease would derive a different one and
            # both the mid-cell checkpoint and the shard's finished rows would
            # refuse the recipe (see `maze_orgb.exploration_plan`).
            parts.append(f"MZ01_ENTROPY_TARGET={row['entropy_target']}")
    elif mode == "fixed":
        coef = knobs.get("entropy_coef_fixed", row.get("coef_end"))
        parts.append(f"MZ01_ENTROPY_COEF={coef}")
    elif mode == "temperature":
        parts.append(f"MZ01_SAMPLE_T={row.get('sample_temperature')}")
    elif mode == EPSILON_MODE:
        parts.append(f"MZ01_EPSILON={row.get('epsilon')}")
        # FOURTEENTH PASS. Only on the epsilon arm, and only when it is not the
        # uniform draw: `all` is both the default and what every arm before this
        # pass ran, so a line that omits it reproduces them.
        if row.get("epsilon_where") not in (None, UNIFORM_EPSILON):
            parts.append(f"MZ01_EPSILON_WHERE={row['epsilon_where']}")
        parts.append(f"MZ01_ENTROPY_COEF="
                     f"{knobs.get('entropy_coef_fixed', COEF_REFERENCE)}")
    # TENTH PASS. A thermostat mode whose controller was HELD STILL is a
    # different recipe from the same mode with the controller running, and the
    # driver has to be told which one to reproduce. `epsilon` and `fixed` run no
    # thermostat by definition and need no flag.
    if row.get("thermostat_off") and mode in (
            "thermostat_fixed_target", RELATIVE_MODE, "temperature"):
        parts.append("MZ01_THERMOSTAT_OFF=1")
        parts.append(f"MZ01_ENTROPY_COEF="
                     f"{knobs.get('entropy_coef_fixed', COEF_REFERENCE)}")
    if row.get("group_size"):
        parts.append(f"MZ01_GROUP={row['group_size']}")
    # TWELFTH PASS. Both default to the closed action space, so a line that
    # names neither reproduces it. An arm that ran the 08-30 recipe has to SAY
    # so, or the line would launch a different one under the picked arm's name.
    if row.get("move_mask") is False:
        parts.append("MZ01_MOVE_MASK=0")
    if row.get("move_mass_coef"):
        parts.append(f"MZ01_MOVE_MASS_COEF={row['move_mass_coef']}")
    # THIRTEENTH PASS, the same rule one knob over: the default is the new
    # price, so an arm that ran the old one has to SAY so or the line would
    # launch a different objective under the picked arm's name.
    if row.get("exit_at_cap") is False:
        parts.append("MZ01_EXIT_AT_CAP=0")
    # FOURTEENTH PASS. `episode` is the default AND what every trace written
    # before the knob ran, so only a row that trained under one of the new
    # credit rules names it -- and it must, or the line would launch the old
    # estimator under the new arm's name.
    if row.get("credit") not in (None, EPISODE_CREDIT):
        parts.append(f"MZ01_CREDIT={row['credit']}")
    # ONLY IF THIS ARM'S OWN PRE-FLIGHT WOULD REFUSE IT. `MZ01_PREFLIGHT_SOFT=1`
    # is a recorded DEVIATION in the manifest, and a run that emits it when the
    # gate would have passed records a deviation that did not happen. An arm
    # whose mechanism is the thermostat cannot clear the gate -- the degeneracy
    # is measured on the warm start, before the coefficient has moved at all --
    # and must carry the flag; an arm that moves the SAMPLER or the ACTION can
    # clear it. When the arm file carries no pre-flight verdict the flag is
    # emitted, because the failure mode of guessing wrong in that direction is a
    # stopped run and in the other it is an unrecorded deviation.
    if not row.get("preflight_hard_passed"):
        parts.append("MZ01_PREFLIGHT_SOFT=1")
    parts.append(f"MZ01_CELLS={cells}")
    return " ".join(parts)


def env_lines(row):
    """One line for A and one for C. -> [str, str].

    Two lines rather than `MZ01_CELLS=A,C`, because the two cells are two boxes
    in this programme's own shape and an operator copies one line per box.
    """
    return [env_line(row, cells="A"), env_line(row, cells="C")]


def _f(x, spec="8.4f"):
    return "     n/a" if x is None else format(float(x), spec)


def _bin_table(rows, reference=None):
    """`detour_paid` by detour-cost bin, every arm beside the reference."""
    have = [r for r in rows if r.get("detour_by_bin")]
    if reference and reference.get("detour_by_bin"):
        have = [reference] + [r for r in have if r is not reference]
    if not have:
        return []
    bins = []
    for r in have:
        for d in r["detour_by_bin"]:
            if d not in bins:
                bins.append(d)
    bins.sort(key=lambda d: (d == "None", d))
    lines = ["", "detour_paid (solved episodes) by detour-cost bin, held out:",
             f"  {'arm':<24} " + " ".join(f"{('d=' + d):>10}" for d in bins)]
    for r in have:
        cells = []
        for d in bins:
            got = (r["detour_by_bin"].get(d) or {}).get("detour_paid_solved")
            cells.append("       n/a" if got is None else f"{got:>10.3f}")
        tag = "REFERENCE" if r.get("reference") else str(r["arm"])
        lines.append(f"  {tag:<24} " + " ".join(cells))
    ns = have[0]["detour_by_bin"]
    lines.append("  n (solved) in the reference row: " + ", ".join(
        f"d={d}: {(ns.get(d) or {}).get('n_solved')}" for d in bins))
    lines.append("  the two price signs are recombined by their own solved "
                 "counts; B4's primary read splits them and this does not.")
    return lines


def _mid_arm_table(rows):
    """One column group per MID-ARM read: solve, mine/ref and the verdict.

    Twelfth pass. A separate block rather than more columns on the wide table:
    at 128 mazes a read is ~3 min, so there are two or three of them at most,
    and putting each one's three numbers under a `@40` heading is what a reader
    of "AVOIDS at 40, RANDOMISED at 80" actually needs. The final read is the
    LAST column of every row, under `@end`, so the trajectory reads across.
    """
    have = [r for r in rows if r.get("eval_at")]
    if not have:
        return []
    steps = []
    for r in have:
        for k in r["eval_at"]:
            if k not in steps:
                steps.append(k)
    steps.sort(key=int)
    head = f"  {'arm':<20}" + "".join(
        f" {('@' + s):>26}" for s in steps) + f" {'@end':>26}"
    lines = ["", "the reads taken INSIDE the arm -- solve / mine-ratio / the "
             "verdict that read WOULD get by the same rules:", head,
             "  " + "-" * (len(head) - 2)]
    for r in have:
        cells = []
        for s in steps + [None]:
            got = r["eval_at"].get(s) if s is not None else {
                "solve_rate": r.get("eval_solve_rate"),
                "mine_ratio_to_reference": r.get("eval_mine_ratio_to_reference"),
                "verdict": r.get("verdict")}
            if not got:
                cells.append(f" {'n/a':>26}")
                continue
            cells.append(
                f" {_f(got.get('solve_rate'), '6.3f')}/"
                f"{_f(got.get('mine_ratio_to_reference'), '5.2f')} "
                f"{str(got.get('verdict') or 'n/a'):>13}")
        lines.append(f"  {str(r['arm']):<20}" + "".join(cells))
    lines.append(f"  a read is a verdict only if its unparsed_rate <= "
                 f"{UNPARSED_BAR:.2f}; above it the read is {INVALID_READ} and "
                 f"is reported, not scored.")
    return lines


def table(rows, pick=None, reference=None):
    """The table, as lines. Shared by the trial driver and the CLI."""
    order = sorted(rows, key=lambda r: (not r.get("reference"), r["arm"] or ""))
    header = (
        f"{'arm':<20} {'mode':<24} {'verdict':<14} {'H start':>8} "
        f"{'H end':>8} {'target':>8} {'coef s':>8} {'coef e':>8} "
        f"{'zadv s':>8} {'zadv e':>8} {'|g|':>9} {'tr solve':>8} "
        f"{'tr mine':>8} {'ev solve':>8} {'ev mine':>8} {'mine/ref':>8} "
        f"{'unparsed':>8} {'bare 1st':>8} {'offpol':>7} {'s/step':>8}")
    lines = [header, "-" * len(header)]
    for r in order:
        cls = r.get("eval_first_token_class") or {}
        lines.append(
            f"{str(r['arm']):<20} {str(r['mode']):<24} {r['verdict']:<14} "
            f"{_f(r['entropy_start'])} {_f(r['entropy_end'])} "
            f"{_f(r['entropy_target'])} {_f(r['coef_start'])} "
            f"{_f(r['coef_end'])} {_f(r['zero_adv_start'])} "
            f"{_f(r['zero_adv_end'])} {_f(r['grad_norm_mean'], '9.4f')} "
            f"{_f(r['solve_end'])} {_f(r['mine_per_ep_end'])} "
            f"{_f(r['eval_solve_rate'])} {_f(r['eval_mine_step_rate'])} "
            f"{_f(r['eval_mine_ratio_to_reference'])} "
            f"{_f(r.get('eval_unparsed_rate'))} {_f(cls.get('bare'))} "
            f"{_f(r['off_policy_fraction'], '7.3f')} "
            f"{_f(r['seconds_per_step'], '8.1f')}")
    lines.append("")
    lines.append(f"DEAD = zero-advantage groups > {ZERO_ADVANTAGE_BAR:.2f} at "
                 f"the end (the move term is still multiplying by zero). On "
                 f"cell A: AVOIDS = held-out mine/ep < {MINE_RATIO_BAR:.2f} x "
                 f"the reference's with solve within {SOLVE_BAND:.2f} of it; "
                 f"RANDOMISED = solve fell more than {SOLVE_BAND:.2f} below the "
                 f"reference; ALIVE = a gradient but not avoidance. On cell D: "
                 f"RANDOMISED = solve fell more than {SOLVE_DROP_BAR:.2f} "
                 f"against the arm's own start; ALIVE = neither.")
    lines.append("`tr solve`/`tr mine` are the TRAINING pool under the arm's "
                 "own sampler (an epsilon arm's `tr mine` includes the mines "
                 "epsilon walked onto); `ev solve`/`ev mine` are the HELD-OUT "
                 "greedy read and are what the cell-A verdicts use. `offpol` "
                 "is the share of decisions whose action was not the scored "
                 "policy's -- measured for epsilon, 1.0 at T != 1.")
    lines.append("start/end are means over the first and last fifth of the "
                 "steps, so one lucky step decides nothing.")
    # TWELFTH PASS. The two columns the 08-30 erratum is about. `unparsed` was
    # the SPEC's own gate and was not in the eval block at all; `bare 1st` is
    # the share of held-out replies whose FIRST TOKEN is a bare move id, and on
    # 08-30 it is the number that would have said, without a single new
    # measurement, that the policy had walked onto ` up`.
    lines.append(f"`unparsed` is the held-out unparsed_rate: a read above "
                 f"{UNPARSED_BAR:.2f} is {INVALID_READ} and is not verdicted. "
                 f"`bare 1st` is the share of held-out first tokens that are a "
                 f"BARE move word; the rest are spaced, capitalised or other, "
                 f"and the four are in `eval_first_token_class`. Training is "
                 f"masked to the bare ids and the eval is NOT, which is what "
                 f"makes this column a readout.")
    for r in order:
        cls = r.get("eval_first_token_class") or {}
        if cls and cls.get("bare") is not None:
            lines.append(
                f"  {r['arm']}: held-out first token bare {cls['bare']:.3f}, "
                f"spaced {(cls.get('spaced') or 0.0):.3f}, capitalised "
                f"{(cls.get('capitalised') or 0.0):.3f}, other "
                f"{(cls.get('other') or 0.0):.3f} over {cls.get('n')} replies "
                f"(by {cls.get('rule')})")
        # TWELFTH PASS, CRITIC. An arm whose action space was OPEN is the 08-30
        # recipe, and when an old file is scored beside a new one that is the
        # difference between the two rows -- so it is named in the table and not
        # only in the launch line.
        if r.get("move_mask") is False:
            lines.append(
                f"  {r['arm']}: the action space was OPEN (the 08-30 recipe): "
                f"the first sampled token came from the full vocabulary while "
                f"the move term, the entropy bonus and `move_mass` read only "
                f"the four bare ids, at move_mass_coef "
                f"{r.get('move_mass_coef')}. Its launch line carries "
                f"MZ01_MOVE_MASK=0.")
        # THIRTEENTH PASS. The arm's own price for an unsolved episode, and
        # the two shares that say how much of the trace it applied to -- in the
        # table whatever the verdict, because they are what the rerun is read
        # against (PLAN section 4).
        if r.get("exit_at_cap") is False:
            lines.append(
                f"  {r['arm']}: an early exit was priced at the decisions it "
                f"TOOK (the pre-thirteenth return): H5's three-bump wall cost "
                f"about -6 against about -28 for the honest solve. Its launch "
                f"line carries MZ01_EXIT_AT_CAP=0.")
        elif r.get("exit_at_cap") and not r.get("reference"):
            lines.append(
                f"  {r['arm']}: unsolved episodes charged at the cap (D33) for "
                f"{_f(r.get('exit_return_share_end'), '.3f')} of the last fifth's "
                f"episodes; farm-stopped {_f(r.get('farm_stopped_end'), '.3f')}.")
        # FOURTEENTH PASS. The credit rule and the targeting rule, named in the
        # table whenever they are not the ones every committed arm ran -- and
        # with the two numbers that say what the move term had to multiply,
        # because under `togo` the `zadv` column is a much coarser read.
        if r.get("credit") not in (None, EPISODE_CREDIT):
            lines.append(
                f"  {r['arm']}: credit={r['credit']} -- the move advantage "
                f"landed on the DECISION that earned it, baselined against the "
                f"group's own value at the same decision index. Last fifth: "
                f"mean |A| {_f(r.get('adv_abs_mean_end'), '.3f')} over "
                f"{_f(r.get('adv_nonzero_share_end'), '.3f')} of decisions. Its "
                f"launch line carries MZ01_CREDIT={r['credit']}.")
        if r.get("epsilon_where") not in (None, UNIFORM_EPSILON):
            lines.append(
                f"  {r['arm']}: the epsilon draw was TARGETED "
                f"({r['epsilon_where']}): it acted only at decisions with a "
                f"mine among the valid landings and never drew onto one. "
                f"Eligible {_f(r.get('epsilon_eligible_fraction_mean'), '.3f')} "
                f"of decisions, replaced "
                f"{_f(r.get('epsilon_replaced_fraction_mean'), '.3f')}. Its "
                f"launch line carries MZ01_EPSILON_WHERE={r['epsilon_where']}.")
        if r["verdict"] == INVALID_READ:
            lines.append(
                f"  {r['arm']}: {INVALID_READ} -- held-out unparsed_rate "
                f"{r.get('eval_unparsed_rate')} is above the "
                f"{UNPARSED_BAR:.2f} the SPEC gates a read at, so solve and "
                f"mine/ep over those episodes measure the parser and not the "
                f"policy. Reported, not scored.")
        if r.get("reference"):
            lines.append(
                f"  {r['arm']}: the REFERENCE -- the warm-started model with NO "
                f"RL at all, on the same held-out mazes. Every cell-A verdict "
                f"is a difference from this row; it is never a candidate.")
        if r.get("slow_controller"):
            lines.append(
                f"  {r['arm']}: DEAD AT THIS HORIZON, not necessarily dead -- "
                f"its coefficient was still CLIMBING at the end "
                f"({r['coef_start']:.4f} -> {r['coef_end']:.4f}) and had not "
                f"reached the {COEF_REFERENCE} the `fixed` arm starts at. An "
                f"arm whose mechanism is the thermostat needs the steps to walk "
                f"there; {r['steps']} were run. The answer is more steps, not a "
                f"lower bar.")
        if r.get("follow_up"):
            lines.append(
                f"  {r['arm']}: the FOLLOW-UP cell ({r['cell']}) under the "
                f"picked arm, not a candidate for the pick.")
        if r["log_importance_ratio_mean"] is not None:
            lines.append(
                f"  {r['arm']}: mean log importance ratio "
                f"{r['log_importance_ratio_mean']:.3f} per decision -- the "
                "samples are OFF-POLICY by this much and the weight is NOT "
                "applied (train_orgb_cell's IMPORTANCE WEIGHT paragraph)")
        if r.get("epsilon_replaced_fraction_mean"):
            lines.append(
                f"  {r['arm']}: {r['epsilon_replaced_fraction_mean']:.3f} of "
                f"decisions took a uniform VALID move instead of the policy's "
                f"(the rate asked for was {r['epsilon']}); the weight is NOT "
                "applied, the fraction is recorded")
        if r["verdict"] in ("FAILED", "EMPTY", "UNREADABLE", "NO-MEASUREMENT",
                            "NO-REFERENCE"):
            lines.append(f"  {r['arm']}: {r['verdict']} -- status "
                         f"{r['status']!r}, {r['steps']} steps in the trace, "
                         f"held-out solve {r['eval_solve_rate']}")
    lines.extend(_mid_arm_table(order))
    lines.extend(_bin_table(order, reference=reference))
    if pick:
        lines.append("")
        lines.append(f"PICK: {pick['arm']} ({pick['mode']}) -- {pick['verdict']}"
                     f", held-out solve "
                     f"{_f(pick['eval_solve_rate']).strip()}, mine/ep "
                     f"{_f(pick['eval_mine_step_rate']).strip()} "
                     f"({_f(pick['eval_mine_ratio_to_reference']).strip()} x "
                     f"the reference), zero-advantage "
                     f"{_f(pick['zero_adv_end']).strip()}, off-policy "
                     f"{_f(pick['off_policy_fraction'], '.3f')}, "
                     f"{_f(pick['seconds_per_step'], '.1f')} s/step")
        for line in env_lines(pick):
            lines.append("  " + line)
    else:
        lines.append("PICK: NONE -- no arm cleared the bar. That is the "
                     "trial's (negative) answer, not a reason to lower one.")
    return lines


def _load(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith(".json") and name != "summary.json":
                    out.append(os.path.join(p, name))
        else:
            out.append(p)
    payloads = []
    for path in out:
        try:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
        except (OSError, ValueError) as err:
            print(f"!! {path}: cannot be read ({err!r})")
            continue
        if isinstance(blob, dict) and "arm" in blob:
            payloads.append(blob)
        elif isinstance(blob, dict) and blob.get("arms"):
            print(f"(skipping {path}: it is a summary, not an arm)")
    return payloads


def main(argv):
    as_json = "--json" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        print(__doc__.strip().splitlines()[-1])
        return 2
    payloads = _load(paths)
    if not payloads:
        print("no arm files found (the trial writes one JSON per arm under "
              "results/explore_trial/)")
        return 0
    rows, reference = read_all(payloads)
    pick = pick_arm(rows)
    if as_json:
        print(json.dumps({"arms": rows, "reference": reference, "pick": pick,
                          "env_line": env_line(pick, cells="A,C"),
                          "env_lines": env_lines(pick) if pick else None,
                          "bars": {"zero_advantage": ZERO_ADVANTAGE_BAR,
                                   "solve_drop": SOLVE_DROP_BAR,
                                   "mine_ratio": MINE_RATIO_BAR,
                                   "solve_band": SOLVE_BAND,
                                   "unparsed": UNPARSED_BAR}}, indent=2))
        return 0
    for line in table(rows, pick, reference=reference):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
