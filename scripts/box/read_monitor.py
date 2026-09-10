#!/usr/bin/env python3
"""read_monitor.py -- one line per NEW held-out read, off the pulled traces.

The cheapest wake-up in the monitoring stack (`docs/HANDOFF-2026-09-10.md`,
"Monitors"): the per-box watchers (`scripts/box/watch_box.sh`) pull
`results/<label>/fork_cf_g6.json` and `reference.json` every CYCLE seconds;
this reads what they pulled and prints a line the moment a read at 40/80/120/160
appears that it has not printed before. It runs under the Monitor tool, so a new
read wakes the session without a poll of the box.

It does NOT re-implement the verdict. `_cell_a_verdict` is imported from
`scripts/score_mz01_explore.py` (bars: MINE_RATIO_BAR 0.75, SOLVE_BAND 0.10,
UNPARSED_BAR 0.05), so "AVOIDS at 40, RANDOMISED at 80" is one rule read twice.

Beside that verdict it prints the two bars of PLAN-fork-counterfactual.md §9
that the scorer's cell-A verdict does not carry, because a read that fails them
is not a result at all:

  * bare share  -- `eval_at[step]["first_token_class"]["bare"]`, floor 0.90;
  * raw move mass -- `history["move_mass"]` at the read's step, floor 0.90.

It also prints the two numbers §8.4/§13.3 require to be read BESIDE the mine
rate, because a mine rate that falls with them is not the price acting:

  * `mean_steps` (`competence.mean_steps`) -- a rate that falls in proportion to
    a shorter route is route-shortening, not avoidance;
  * `invalid_rate` -- wall bumps. It has no bar of its own and the move-mass
    guard does not see it: the mass is on a bare move WORD while the move it
    names is into a wall. A read whose mine rate falls while this rises is a
    policy degrading, which is D-15's derailment under another name.

Either below its floor prints INVALID-MASS/INVALID-BARE beside the verdict; the
arm is then reported as the regulariser failing, not as a mine-rate result.

Usage:  MZ01_BOX_DIR=<scratch> python3 scripts/box/read_monitor.py \
            forkcf4b_A_s0_p30 forkcf4b_A_s1_p30 ...
State (which reads were already printed) lives in $MZ01_BOX_DIR/.reads_seen so a
restart does not replay the whole history.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from score_mz01_explore import (  # noqa: E402
    MINE_RATIO_BAR, SOLVE_BAND, UNPARSED_BAR, _cell_a_verdict)

MASS_BAR = 0.90   # PLAN-fork-counterfactual.md §9, the raw-mass rule
BARE_BAR = 0.90   # §9, the bare-share rule
RES = os.path.join(HERE, "..", "..", "experiments", "v2", "MZ01_maze_world",
                   "results")


def _f(x, n=3):
    return "  n/a" if x is None else f"{x:.{n}f}"


def rows(label):
    """Every read on disk for `label`, oldest first. -> [(step, line)]."""
    d = os.path.join(RES, label)
    try:
        pay = json.load(open(os.path.join(d, "fork_cf_g6.json")))
        ref = (json.load(open(os.path.join(d, "reference.json"))).get("eval")
               or {})   # the reference's own eval block, competence included
    except (OSError, ValueError):
        return []
    rs, rm = ref.get("solve_rate"), ref.get("mine_step_rate")
    mass = (pay.get("history") or {}).get("move_mass") or []
    blocks = dict(pay.get("eval_at") or {})
    if pay.get("eval"):                      # the final read at the last step
        blocks[str(pay.get("trial_steps"))] = pay["eval"]
    out = []
    for step, b in sorted(blocks.items(), key=lambda kv: int(kv[0])):
        b = b or {}
        solve, mine, unp = (b.get("solve_rate"), b.get("mine_step_rate"),
                            b.get("unparsed_rate"))
        verdict = _cell_a_verdict(solve, mine, unp, rs, rm)
        bare = ((b.get("first_token_class") or {}).get("bare"))
        i = min(int(step), len(mass)) - 1     # move mass AT the read's step
        mm = mass[i] if 0 <= i < len(mass) else None
        flags = []
        if mm is not None and mm < MASS_BAR:
            flags.append(f"INVALID-MASS({mm:.3f}<{MASS_BAR})")
        if bare is not None and bare < BARE_BAR:
            flags.append(f"INVALID-BARE({bare:.3f}<{BARE_BAR})")
        ratio = None if not rm or mine is None else mine / rm
        comp = b.get("competence") or {}
        ms, inv = comp.get("mean_steps"), b.get("invalid_rate")
        rms = ((ref.get("competence") or {}).get("mean_steps")
               if isinstance(ref, dict) else None)
        rinv = ref.get("invalid_rate")
        out.append((int(step), (
            f"{label} @{step:>3}  {verdict:<12} "
            f"mine {_f(mine)}/ref {_f(rm)} = ratio {_f(ratio)} "
            f"(bar {MINE_RATIO_BAR})  solve {_f(solve)} vs {_f(rs)} "
            f"(band {SOLVE_BAND})  unparsed {_f(unp)} (bar {UNPARSED_BAR})  "
            f"bare {_f(bare)}  mass {_f(mm)}  "
            f"steps {_f(ms, 2)} vs {_f(rms, 2)}  "
            f"invalid {_f(inv)} vs {_f(rinv)}"
            + ("  " + " ".join(flags) if flags else ""))))
    return out


def main(labels):
    state = os.environ.get("MZ01_BOX_DIR", "/tmp")
    os.makedirs(state, exist_ok=True)   # a missing scratchpad is not a reason
    seen_p = os.path.join(state, ".reads_seen")   # to drop a read on the floor
    seen = set(open(seen_p).read().split("\n")) if os.path.exists(seen_p) else set()
    for label in labels:
        for step, line in rows(label):
            key = f"{label}@{step}"
            if key in seen:
                continue
            print(line, flush=True)     # print FIRST: a read that reached the
            seen.add(key)               # screen is never replayed, and a failure
            with open(seen_p, "a") as fh:   # to record it must not swallow it
                fh.write(key + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["forkcf4b_A_s0_p30", "forkcf4b_A_s1_p30",
                          "forkcf4b_A_s0_f2", "forkcf4b_A_s0_g10"])
