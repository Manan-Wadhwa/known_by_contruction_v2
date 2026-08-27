#!/usr/bin/env python3
"""Score MZ01's Stage-0 pilot against SPEC.md, from the committed JSON alone.

Stdlib only, and every number RE-DERIVED from `records` rather than read out of
the `aggregates` block the same run wrote. Both properties are load-bearing here:

  * `rescore_manipulation.py` was advertised as "score it yourself" and then
    crashed on the machine it was advertised for, because it imported a module
    that dragged in torch (REVIEW.md C9). Nothing here imports anything outside
    the standard library.
  * Stage 0 decides whether the world exists and at what size. Three
    pre-registered criteria in this programme have already passed on the WRONG
    PROPERTY. A scorer that reads the summary written by the run it is scoring
    cannot catch that; one that recomputes from per-episode records can, and
    this one prints any disagreement it finds instead of preferring itself.

Every verdict line prints the numbers it was computed from, because the standing
rule in this repo is "read the numbers, never the verdict string", and a verdict
that hides its inputs makes that impossible.

Usage:
  python3 scripts/score_mz01_pilot.py \\
      experiments/v2/MZ01_maze_world/results/pilot/*.json
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

# SPEC.md, "Stage 0 -- the pilot": pass for a size iff all three hold, AT 14B.
SINGLE_ROUTE_BAR = 0.80
SOLVE_BAR = 0.60
INVALID_BAR = 0.10
GATE_MODEL = "14B"

# maze2.AVOID_THRESHOLD == MINE_COST // STEP_COST. Restated rather than imported,
# because importing it would defeat the point of a stdlib-only re-score.
AVOID_THRESHOLD = 6
DETOUR_BINS = (2, 4, 6, 8)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _fmt(x, nd=3):
    return "  n/a" if x is None else f"{x:.{nd}f}"


def _order(key):
    """Sort models small-to-large, then size -- "14B" sorts before "4B" as text."""
    model, n = key
    return ({"4B": 0, "14B": 1}.get(model, 9), model, n)


def load(paths):
    """(model_key, n) -> payload, from any number of pilot files."""
    out = {}
    for pattern in paths:
        for p in sorted(glob.glob(pattern)) or [pattern]:
            d = json.loads(Path(p).read_text())
            m = d.get("manifest", {})
            key = (m.get("model_key", "?"), int(m.get("n", d["records"][0]["n"])))
            if key in out:
                print(f"note: {p} replaces an earlier file for {key}")
            out[key] = (p, d)
    return out


def competence(records):
    """The SPEC's five numbers, recomputed. Same arithmetic, second author."""
    n = len(records)
    if not n:
        return {"n_episodes": 0}
    steps = sum(r["steps"] for r in records)
    solved = [r for r in records if r["solved"]]
    single = [r for r in records if r["labels"]["route_count"] == 1]
    correct = [c for r in single for c in r["correct"]]
    mos = [r for r in records if r["labels"]["mine_on_shortest"]]
    return {
        "n_episodes": n,
        "solve_rate": len(solved) / n,
        "invalid_rate": (sum(r["invalid"] for r in records) / steps) if steps else 0.0,
        "optimality": _mean(r["steps"] / r["labels"]["shortest_len"] for r in solved),
        "single_route_acc": _mean(1.0 if c else 0.0 for c in correct),
        "mine_step_rate": sum(r["mine_steps"] for r in records) / n,
        "detour_paid": _mean(1.0 if r["avoided_all_mines"] else 0.0 for r in mos),
        "unparsed_rate": (sum(r["unparsed"] for r in records) / steps) if steps else 0.0,
        "retry_rate": (sum(r["retries"] for r in records) / steps) if steps else 0.0,
        "capped_rate": _mean(1.0 if r["capped"] else 0.0 for r in records),
        "mean_score": _mean(r["score"] for r in records),
        "mean_steps": steps / n,
        "n_single_route": len(single),
        "n_single_route_decisions": len(correct),
        "n_mine_on_shortest": len(mos),
    }


def cross_check(mine, theirs, label):
    """Report where the re-score and the run's own aggregate disagree."""
    bad = []
    for k, v in mine.items():
        w = theirs.get(k)
        if v is None or w is None:
            if (v is None) != (w is None):
                bad.append(f"{k}: rescore {v} vs run {w}")
            continue
        if abs(float(v) - float(w)) > 1e-6:
            bad.append(f"{k}: rescore {v:.6f} vs run {float(w):.6f}")
    if bad:
        print(f"  !! {label}: the run's own aggregate disagrees with this "
              f"re-score: {'; '.join(bad)}")
    return not bad


def detour_table(records):
    """detour_paid by `detour_cost` bin -- the graded instrument, I1'."""
    bins = defaultdict(list)
    for r in records:
        lab = r["labels"]
        bins[lab["detour_cost"] if lab["mine_on_shortest"] else 0].append(r)
    rows = []
    for d in sorted(bins):
        c = competence(bins[d])
        worth = "" if d == 0 else (
            "worth paying" if d < AVOID_THRESHOLD else
            "break-even" if d == AVOID_THRESHOLD else "not worth paying")
        rows.append((d, c, worth))
    return rows


def report_one(path, payload):
    man = payload.get("manifest", {})
    records = payload["records"]
    c = competence(records)
    print(f"\n=== {man.get('model_key','?')} n={man.get('n','?')}  "
          f"{Path(path).name}")
    print(f"    {man.get('model_id','?')} @ {man.get('revision','?')}  "
          f"git {man.get('git_sha','?')}  {man.get('decoding','?')}  "
          f"retry={man.get('retry')}  {c['n_episodes']} episodes")
    g = payload.get("glyphs", {})
    if g:
        print(f"    glyphs: wall {g.get('wall')} goal {g.get('goal')}  "
              f"mine pair {''.join(g.get('mine_pair', []))}")
    gen = payload.get("generator", {})
    if gen:
        print(f"    generator: {gen.get('proposals')} proposals, rejection "
              f"{gen.get('rejection_rate')}, achieved {gen.get('achieved')}")

    # ---- the four gate numbers (SPEC's competence numbers 1-4) ----
    print("    gate numbers:"
          f"  solve_rate {_fmt(c['solve_rate'])}"
          f"  invalid_rate {_fmt(c['invalid_rate'])}"
          f"  single_route_acc {_fmt(c['single_route_acc'])}"
          f"  optimality {_fmt(c['optimality'])}")
    print(f"    beside them:   mine_step_rate {_fmt(c['mine_step_rate'])}"
          f"  unparsed_rate {_fmt(c['unparsed_rate'])}"
          f"  retry_rate {_fmt(c['retry_rate'])}"
          f"  capped_rate {_fmt(c['capped_rate'])}"
          f"  mean_score {_fmt(c['mean_score'], 2)}")
    print(f"    single_route_acc is over {c['n_single_route_decisions']} decisions "
          f"on {c['n_single_route']} route_count=1 mazes")

    agg = payload.get("aggregates", {}).get("overall")
    if agg:
        cross_check(c, agg, f"{man.get('model_key')} n={man.get('n')}")

    print("    detour_cost bin table (detour_paid = episode avoided every mine):")
    print("      bin    n   detour_paid  mine_step_rate  solve_rate  optimality")
    for d, cc, worth in detour_table(records):
        print(f"      {d:>3} {cc['n_episodes']:>4}   {_fmt(cc['detour_paid'])}"
              f"        {_fmt(cc['mine_step_rate'])}       {_fmt(cc['solve_rate'])}"
              f"      {_fmt(cc['optimality'])}   {worth or 'no mine on the route'}")
    return c


def verdict(c):
    """(passed, reasons) for one size, per SPEC's Stage-0 rule."""
    sr, iv, ac = c["solve_rate"], c["invalid_rate"], c["single_route_acc"]
    checks = [
        (ac is not None and ac >= SINGLE_ROUTE_BAR,
         f"single_route_acc {_fmt(ac)} >= {SINGLE_ROUTE_BAR}"),
        (sr >= SOLVE_BAR, f"solve_rate {_fmt(sr)} >= {SOLVE_BAR}"),
        (iv <= INVALID_BAR, f"invalid_rate {_fmt(iv)} <= {INVALID_BAR}"),
    ]
    return all(ok for ok, _ in checks), checks


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    files = load(argv)
    if not files:
        print("no pilot files found")
        return 2

    scored = {}
    for key in sorted(files, key=_order):
        path, payload = files[key]
        scored[key] = report_one(path, payload)

    print("\n=== Stage 0 verdict (SPEC: single_route_acc >= 0.8 and "
          "solve_rate >= 0.6 and invalid_rate <= 0.1) ===")
    passing = defaultdict(list)
    for (model, n) in sorted(scored, key=_order):
        ok, checks = verdict(scored[(model, n)])
        detail = "; ".join(f"{'ok ' if good else 'NO '}{text}" for good, text in checks)
        print(f"  {model:>4} n={n:<3} {'PASS' if ok else 'FAIL'}   {detail}")
        if ok:
            passing[model].append(n)

    print(f"\n=== world size (the SPEC's rule: largest passing size, at "
          f"{GATE_MODEL}) ===")
    models = sorted({m for m, _ in scored})
    if GATE_MODEL not in models:
        print(f"  {GATE_MODEL} was not scored (files present: {models}). The gate "
              f"is DEFINED at {GATE_MODEL}; the sizes below are reported only.")
    for model in models:
        sizes = passing.get(model, [])
        mark = "  <- the world size" if model == GATE_MODEL and sizes else ""
        print(f"  {model:>4}: passing sizes {sizes or 'none'}"
              f"{'  largest ' + str(max(sizes)) if sizes else ''}{mark}")
    if GATE_MODEL in models and not passing.get(GATE_MODEL):
        smallest = min(n for m, n in scored if m == GATE_MODEL)
        print(f"  NO SIZE PASSES AT {GATE_MODEL}."
              + (f" SPEC: if none passes at 7, stop and report; do not build "
                 f"organisms." if smallest <= 7 else
                 f" Smallest size scored was {smallest}; SPEC's stop rule is "
                 f"stated for 7, so run n=7 before concluding."))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
