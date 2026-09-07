#!/usr/bin/env python3
"""Was MZ01's entropy controller doing anything, or was it pinned? -> a table.

WHAT THIS ANSWERS, AND WHY IT IS A SEPARATE SCRIPT

`train_orgb_cell` runs an ADAPTIVE entropy bonus: every step it moves
`log_coef` by `entropy_lr * (entropy_target - measured_entropy)` and clamps the
result into `ENTROPY_COEF_BOUNDS`. HH01 discovered, after eight seeds, that at
14B the target 0.7 sits ABOVE the base policy's own move entropy, so the update
is positive at EVERY step, the coefficient walks to its 1.0 ceiling and stays
there, and the run has no entropy authority left for most of its length
(`scripts/score_hh01.py::rl_entropy_pinned`, SPEC amendment 5). That is a
property of the trace and of nothing else, it is invisible in every number the
scorer prints, and it costs a lease to find out late. This reads it in a second,
off the shard JSON, with no torch and no model.

It is REPORTED, never a gate. The recipe is pre-registered; this says what the
recipe did.

Usage:
    python3 scripts/mz01_entropy_trace.py results/orgb/*.json
    python3 scripts/mz01_entropy_trace.py --stride 25 --json results/orgb/*.json

Stdlib only, like `scripts/score_mz01_orgb.py`: it runs under a bare `python3`
on the box, beside the run, before anything has been pulled.
"""

import json
import sys
from pathlib import Path

# ---- the two numbers this file re-types rather than imports (stdlib only).
# Each names its home, so a drift is a one-line diff and not a mystery -- the
# convention `scripts/score_hh01.py` set for exactly this pair.
ENTROPY_COEF_BOUNDS = (1e-4, 1.0)   # calibration.maze_orgb.ENTROPY_COEF_BOUNDS
PINNED_FRACTION = 0.5               # scripts/score_hh01.py::PINNED_FRACTION
DEFAULT_STRIDE = 50                 # calibration.maze_orgb.CKPT_EVERY, and the
                                    # stride SPEC reads a 300-step cell at

# The trace field names, in one place. They are `train_orgb_cell`'s history
# keys; HH01 stores the same series under `rl_policy_entropy_trace` /
# `rl_entropy_coef_trace` on a flat row, and `_series` reads either shape so one
# script can be pointed at either programme's file.
FIELDS = {
    "step": ("step",),
    "policy_entropy": ("policy_entropy", "rl_policy_entropy_trace"),
    "entropy_coef": ("entropy_coef", "rl_entropy_coef_trace"),
    "solve_rate": ("solve_rate", "rl_solve_rate_trace"),
}


def _series(hist, name):
    """One trace out of a training history, under either programme's name."""
    for key in FIELDS[name]:
        got = hist.get(key)
        if got:
            return [x for x in got]
    return []


def _finite(xs):
    return [float(x) for x in xs if x is not None and float(x) == float(x)]


def cells_in(blob, source):
    """Every (source, cell, seed, training-history) this file carries.

    Handles the three shapes a MZ01 file comes in: a shard payload (`rows`), a
    smoke-timing probe (`smoke_timing.history`, which has no row and no cell of
    the design), and a bare history dict (what a test or a notebook holds).
    """
    out = []
    if isinstance(blob, dict) and blob.get("rows"):
        for row in blob["rows"]:
            hist = row.get("training") or {}
            if _series(hist, "entropy_coef"):
                out.append((source, row.get("cell"), row.get("seed"), row, hist))
    probe = (blob or {}).get("smoke_timing") if isinstance(blob, dict) else None
    if probe and probe.get("history"):
        out.append((source, probe.get("cell"), probe.get("seed"),
                    {"note": "TIMING PROBE, not a cell of the design"},
                    probe["history"]))
    if not out and isinstance(blob, dict) and _series(blob, "entropy_coef"):
        cfg = blob.get("config") or {}
        out.append((source, cfg.get("cell"), cfg.get("seed"), {}, blob))
    return out


def bounds_of(hist):
    """The bounds THIS cell ran under, off its own config; the default if old.

    Runs written before 2026-08-29 do not record the pair -- it lived only as a
    default argument of the trainer -- so the fallback is named rather than
    guessed, and `bounds_source` says which of the two the reader is looking at.
    """
    cfg = hist.get("config") or {}
    got = cfg.get("entropy_coef_bounds")
    if got and len(got) == 2:
        return (float(got[0]), float(got[1])), "the cell's own config"
    return (float(ENTROPY_COEF_BOUNDS[0]), float(ENTROPY_COEF_BOUNDS[1])), \
        "calibration.maze_orgb.ENTROPY_COEF_BOUNDS (the file records none)"


def read_cell(hist, *, fraction_bar=PINNED_FRACTION):
    """The entropy story of ONE cell. Field names are HH01's where they overlap.

    ONE DELIBERATE DIFFERENCE FROM HH01, named here because the two numbers get
    compared (critic pass, 2026-08-29): `score_hh01.rl_entropy_pinned` counts
    only the steps at the CEILING, this counts the steps at EITHER bound. The
    floor is the same loss of authority from the other side, and the split is
    printed (`steps_at_ceiling` / `steps_at_floor`) so an HH01-comparable number
    is still readable off this output -- but `fraction_at_bound` is NOT HH01's
    `fraction_at_bound` unless `steps_at_floor` is 0.
    """
    coef = _series(hist, "entropy_coef")
    ent = _series(hist, "policy_entropy")
    solve = _series(hist, "solve_rate")
    steps = _series(hist, "step") or list(range(len(coef)))
    (lo, hi), source = bounds_of(hist)
    cfg = hist.get("config") or {}
    at_hi = [i for i, c in enumerate(coef)
             if c is not None and float(c) == float(c) and float(c) >= hi - 1e-9]
    at_lo = [i for i, c in enumerate(coef)
             if c is not None and float(c) == float(c) and float(c) <= lo + 1e-12]
    at = sorted(set(at_hi) | set(at_lo))
    n = len(coef)
    ent_f = _finite(ent)
    fraction = (len(at) / n) if n else None
    return {
        "steps": n,
        "step_index": steps,
        "policy_entropy": ent,
        "entropy_coef": coef,
        "solve_rate": solve,
        "bound": [lo, hi],
        "bounds_source": source,
        "steps_at_bound": len(at),
        "steps_at_ceiling": len(at_hi),
        "steps_at_floor": len(at_lo),
        "fraction_at_bound": fraction,
        "first_at_bound": (at[0] if at else None),
        "min_policy_entropy": (min(ent_f) if ent_f else None),
        "final_policy_entropy": (ent_f[-1] if ent_f else None),
        "entropy_target": cfg.get("entropy_target"),
        "entropy_adaptive": cfg.get("entropy_adaptive"),
        "entropy_lr": cfg.get("entropy_lr"),
        "final_solve_rate": (_finite(solve)[-1] if _finite(solve) else None),
        "fraction_bar": fraction_bar,
        "pinned": (None if fraction is None
                   else bool(fraction > fraction_bar)),
        "verdict": ("NO-TRACE" if not n
                    else "PINNED" if fraction > fraction_bar else "FREE"),
    }


def sample_at(read, stride=DEFAULT_STRIDE):
    """(step, H, coef, solve) at 0, stride, 2*stride, ... and the last step."""
    n = read["steps"]
    if not n:
        return []
    idx = list(range(0, n, max(1, int(stride))))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    out = []

    def get(series, i):
        return series[i] if i < len(series) and series[i] is not None else None

    for i in idx:
        step = read["step_index"][i] if i < len(read["step_index"]) else i
        out.append((step, get(read["policy_entropy"], i),
                    get(read["entropy_coef"], i), get(read["solve_rate"], i)))
    return out


def _f(x, width=6, places=3):
    return ("n/a" if x is None else f"{float(x):.{places}f}").rjust(width)


def print_cell(source, cell, seed, row, read, stride=DEFAULT_STRIDE):
    ck = (row or {}).get("checkpoint") or {}
    resumed = ""
    if ck.get("resumed"):
        resumed = f"  [RESUMED at step {ck.get('resumed_from_step')}]"
    elif ck.get("reused_row"):
        resumed = "  [row carried forward from an earlier lease]"
    print(f"\n{Path(source).name}  cell {cell} seed {seed}  "
          f"{read['steps']} steps{resumed}")
    if (row or {}).get("note"):
        print(f"  ({row['note']})")
    points = sample_at(read, stride)
    print("  step " + "".join(str(p[0]).rjust(8) for p in points))
    print("  H    " + "".join(_f(p[1], 8) for p in points))
    print("  coef " + "".join(_f(p[2], 8, 4) for p in points))
    if any(p[3] is not None for p in points):
        print("  solve" + "".join(_f(p[3], 8) for p in points))
    lo, hi = read["bound"]
    frac = read["fraction_at_bound"]
    print(f"  at a bound: {read['steps_at_bound']}/{read['steps']} steps "
          f"({'n/a' if frac is None else f'{frac:.3f}'}) -- "
          f"{read['steps_at_ceiling']} at the ceiling {hi:g}, "
          f"{read['steps_at_floor']} at the floor {lo:g}; first at step "
          f"{read['first_at_bound']}  [bounds from {read['bounds_source']}]")
    print(f"  H: final {_f(read['final_policy_entropy'], 0)}, min "
          f"{_f(read['min_policy_entropy'], 0)}, target "
          f"{read['entropy_target']} (adaptive {read['entropy_adaptive']}); "
          f"final solve {_f(read['final_solve_rate'], 0)}")
    print(f"  VERDICT: {read['verdict']}"
          + ("" if read["verdict"] != "PINNED" else
             f"  -- the controller sat on a bound for more than "
             f"{read['fraction_bar']:.0%} of the run: the entropy target was "
             f"not met and the bonus had no authority left (HH01's finding)"))


def parse_args(argv):
    """-> (paths, stride, as_json). `--stride N` and `--stride=N` both work."""
    paths, stride, as_json = [], DEFAULT_STRIDE, False
    skip = -1
    for i, arg in enumerate(argv):
        if i == skip:
            continue
        if arg == "--json":
            as_json = True
        elif arg == "--stride":
            if i + 1 >= len(argv):
                raise SystemExit("--stride needs a number")
            stride = int(argv[i + 1])
            skip = i + 1
        elif arg.startswith("--stride="):
            stride = int(arg.split("=", 1)[1])
        elif arg.startswith("--"):
            raise SystemExit(f"unknown flag {arg!r}; expected --stride N or --json")
        else:
            paths.append(arg)
    if stride < 1:
        raise SystemExit(f"--stride must be >= 1, got {stride}")
    return paths, stride, as_json


def main(argv):
    args, stride, as_json = parse_args(argv)
    if not args:
        raise SystemExit(__doc__)

    found, blocks = 0, []
    for source in args:
        try:
            with open(source, encoding="utf-8") as fh:
                blob = json.load(fh)
        except (OSError, ValueError) as err:
            print(f"!! {source}: cannot be read ({err!r})")
            continue
        for src, cell, seed, row, hist in cells_in(blob, source):
            read = read_cell(hist)
            found += 1
            if as_json:
                blocks.append({"source": src, "cell": cell, "seed": seed,
                               **{k: v for k, v in read.items()
                                  if k not in ("step_index", "policy_entropy",
                                               "entropy_coef", "solve_rate")},
                               "sampled": sample_at(read, stride)})
            else:
                print_cell(src, cell, seed, row, read, stride)
    if as_json:
        print(json.dumps(blocks, indent=2))
        return 0
    if not found:
        print("no training trace in any of those files (a pre-flight failure "
              "writes a payload with no rows, which is what this looks like)")
        return 0
    print(f"\n{found} cell(s) read. PINNED means the coefficient sat on the "
          f"floor or the ceiling for more than {PINNED_FRACTION:.0%} of the "
          f"steps; it is REPORTED and gates nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
