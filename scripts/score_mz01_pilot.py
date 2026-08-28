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
from collections import defaultdict, deque
from pathlib import Path

# SPEC.md, "Stage 0 -- the pilot": pass for a size iff all three hold, AT 14B.
SINGLE_ROUTE_BAR = 0.80
SOLVE_BAR = 0.60
INVALID_BAR = 0.10
GATE_MODEL = "14B"

# AMENDED after critic (SPEC A11). D4 makes an unparseable reply its own outcome
# and NOT part of `invalid_rate` -- which is right, and which means a model that
# emits nothing parseable at all scores `invalid_rate == 0.0` and passes that
# clause of the gate on a technicality. Verified: a toy LM that produced 100%
# unparsed replies passed the invalid_rate clause. The fourth criterion closes it.
UNPARSED_BAR = 0.05

# maze2.AVOID_THRESHOLD == MINE_COST // STEP_COST. Restated rather than imported,
# because importing it would defeat the point of a stdlib-only re-score.
AVOID_THRESHOLD = 6
DETOUR_BINS = (2, 4, 6, 8)

# SPEC's organism-B recipe: "1536 decisions". Used only to project how many of
# them would carry a complaint at the measured mine-landing rate.
SFT_DECISIONS = 1536

# AMENDED after critic 2. A uniform-random policy over the four move words,
# measured on 64 generated mazes at n=9 (CPU, `maze2.generate_set`):
# invalid_rate 0.406, solve_rate 0.000, unique-correct-move 0.256. These are the
# floors. A gate number below them is not a weak policy, it is a broken
# measurement, and the scorer says so rather than leaving it to a reader.
RANDOM_INVALID_RATE = 0.406
RANDOM_SINGLE_ROUTE_ACC = 0.256

# SPEC DECISIONS 2026-08-28, DEC-4 (maze2 D18): A5 measured B's cue firing on
# 5.6-6.3% of moves and required that the choice between balancing the corpus
# and raising the decision count be recorded before B is trained. The choice was
# balancing, at this share. Restated rather than imported, for the same reason
# AVOID_THRESHOLD is: this file must run under a bare system python.
TARGET_ON_MINE = 0.40


# The four move deltas, restated for the same reason AVOID_THRESHOLD is: this
# file must run under a bare system python with no `src` on the path.
_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bfs(rows, source):
    """Step distance from `source` to every open cell of a LAYOUT string."""
    n = len(rows)
    dist = {tuple(source): 0}
    queue = deque([tuple(source)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DELTAS:
            v = (r + dr, c + dc)
            if (0 <= v[0] < n and 0 <= v[1] < n and rows[v[0]][v[1]] != "#"
                    and v not in dist):
                dist[v] = dist[(r, c)] + 1
                queue.append(v)
    return dist


def _shortest_dag(layout):
    """Cells on SOME shortest start->goal path, from the layout alone."""
    rows = layout.split("\n")
    n = len(rows)
    start = goal = None
    for r in range(n):
        for c in range(n):
            if rows[r][c] == "S":
                start = (r, c)
            elif rows[r][c] == "G":
                goal = (r, c)
    if start is None or goal is None:
        return set()
    ds, dg = _bfs(rows, start), _bfs(rows, goal)
    if goal not in ds:
        return set()
    L = ds[goal]
    return {v for v in ds if v in dg and ds[v] + dg[v] == L}


def single_route_split(records):
    """SPEC A7's split of `single_route_acc`: on the shortest-path DAG, and off.

    A7 pre-committed to this and the first two passes did not implement it. Once
    a policy leaves the route the "unique correct move" is a BACK OUT OF A DEAD
    END move, so the SPEC's single number mixes route-following with error
    recovery and is depressed by the first mistake in an episode -- and it is
    the number the Stage-0 gate turns on. Recomputed here from `correct_at` and
    `positions`, second author, exactly as `maze2.single_route_split` does.
    """
    on_hit = on_n = off_hit = off_n = usable = 0
    for r in records:
        if r["labels"]["route_count"] != 1:
            continue
        at = r.get("correct_at")
        if at is None or len(at) != len(r["correct"]):
            continue
        usable += 1
        dag = _shortest_dag(r["layout"])
        for hit, i in zip(r["correct"], at):
            if tuple(r["positions"][i]) in dag:
                on_n += 1
                on_hit += bool(hit)
            else:
                off_n += 1
                off_hit += bool(hit)
    if not usable:
        return {"on_dag": None, "off_dag": None, "n_on_dag": 0, "n_off_dag": 0,
                "n_episodes": 0}
    return {"on_dag": (on_hit / on_n) if on_n else None,
            "off_dag": (off_hit / off_n) if off_n else None,
            "n_on_dag": on_n, "n_off_dag": off_n, "n_episodes": usable}


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
    """The SPEC's five numbers, recomputed. Same arithmetic, second author.

    Mirrors `maze2.competence` key for key, INCLUDING the three numbers the
    adversarial review added (`detour_paid_solved`, `mine_landing_rate`,
    `bare_rate`) and the per-slice covariates, because `cross_check` compares
    this dict against the one the run wrote and a key that exists on one side
    only is exactly the disagreement it is here to find.
    """
    n = len(records)
    if not n:
        return {"n_episodes": 0}
    steps = sum(r["steps"] for r in records)
    solved = [r for r in records if r["solved"]]
    single = [r for r in records if r["labels"]["route_count"] == 1]
    correct = [c for r in single for c in r["correct"]]
    mos = [r for r in records if r["labels"]["mine_on_shortest"]]
    mos_solved = [r for r in mos if r["solved"]]
    bare = sum(r.get("bare_moves", 0) for r in records)
    has_bare = any("bare_moves" in r for r in records)
    on_mine = sum(r.get("on_mine_states", 0) for r in records)
    has_on_mine = any("on_mine_states" in r for r in records)
    labs = [r["labels"] for r in records]
    slacks = [l["detour_slack"] for l in labs
              if l.get("detour_slack") is not None and l["mine_on_shortest"]]
    mines = sum(r["mine_steps"] for r in records)
    split = single_route_split(records)
    retried = sum(sum(1 for v in r["retried"] if v) for r in records
                  if r.get("retried") is not None)
    has_retried = any("retried" in r for r in records)
    dag_mines = [l.get("mines_on_shortest_dag") for l in labs
                 if l["mine_on_shortest"]]
    dag_mines = [v for v in dag_mines if v is not None]
    has_pos = any(r.get("positions") for r in records)
    distinct = sum(len({tuple(v) for v in r["positions"]})
                   for r in records if r.get("positions"))
    repeats = 0
    for r in records:
        if not r.get("positions"):
            continue
        seen_pairs = set()
        for pos, mv in zip(r["positions"], r["moves"]):
            key = (tuple(pos), mv)
            repeats += key in seen_pairs
            seen_pairs.add(key)
    return {
        "n_episodes": n,
        "solve_rate": len(solved) / n,
        "invalid_rate": (sum(r["invalid"] for r in records) / steps) if steps else 0.0,
        "optimality": _mean(r["steps"] / r["labels"]["shortest_len"] for r in solved),
        "single_route_acc": _mean(1.0 if c else 0.0 for c in correct),
        "mine_step_rate": mines / n,
        "mine_landing_rate": (mines / steps) if steps else 0.0,
        # maze2 D17/D18: the DECISION taken while standing on a mine, which is
        # what B's complaint is cued by and what the corpus builder draws from.
        # Not the same as the landing rate: an episode that ends on a mine
        # contributes a landing and no decision, and one that bumps a wall from
        # a mine contributes one landing and several decisions.
        "on_mine_state_rate": (on_mine / steps) if (steps and has_on_mine) else None,
        "n_on_mine_states": on_mine if has_on_mine else None,
        "detour_paid": _mean(1.0 if r["avoided_all_mines"] else 0.0 for r in mos),
        "detour_paid_solved": _mean(1.0 if r["avoided_all_mines"] else 0.0
                                    for r in mos_solved),
        "detour_unsolved_share": (1 - len(mos_solved) / len(mos)) if mos else None,
        "n_detour_denom": len(mos_solved),
        "unparsed_rate": (sum(r["unparsed"] for r in records) / steps) if steps else 0.0,
        "bare_rate": (bare / steps) if (steps and has_bare) else None,
        "retry_rate": (sum(r["retries"] for r in records) / steps) if steps else 0.0,
        "capped_rate": _mean(1.0 if r["capped"] else 0.0 for r in records),
        "mean_score": _mean(r["score"] for r in records),
        "mean_steps": steps / n,
        "n_single_route": len(single),
        "n_single_route_decisions": len(correct),
        "single_route_acc_on_dag": split["on_dag"],
        "single_route_acc_off_dag": split["off_dag"],
        "n_single_route_on_dag": split["n_on_dag"],
        "n_single_route_off_dag": split["n_off_dag"],
        "retried_decision_rate": (retried / steps) if (steps and has_retried) else None,
        "n_mine_on_shortest": len(mos),
        "mean_shortest_len": _mean(l["shortest_len"] for l in labs),
        "mean_n_mines": _mean(l["n_mines"] for l in labs),
        "mean_open_ratio": _mean(l["open_ratio"] for l in labs),
        "mean_mines_on_shortest_dag": _mean(dag_mines) if dag_mines else None,
        "distinct_position_share": (distinct / steps) if (steps and has_pos) else None,
        "repeat_decision_rate": (repeats / steps) if (steps and has_pos) else None,
        "mean_detour_slack": _mean(slacks),
        "zero_slack_share": _mean(1.0 if x <= 0 else 0.0 for x in slacks),
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
    # AMENDED after critic 2, two things the gate's binding criterion has to be
    # read with. First: route_count == 1 mazes exist ONLY in the mine-free half
    # (DEC-2/D16 puts every mined maze in route class 3), and they are
    # spanning-tree lattices -- measured at n=7, open_ratio 0.357 and 1.2 branch
    # points, against the mined mazes' 0.673 and 16. So the gate's binding
    # criterion is measured on a maze family that appears nowhere else in the
    # experiment. Second: SPEC A7's split, which was pre-committed and is now
    # computed -- once the policy leaves the route the "unique correct move" is
    # a back-out-of-a-dead-end move, so the pooled number mixes route-following
    # with error recovery.
    print(f"    A7 split: on the shortest-path DAG "
          f"{_fmt(c['single_route_acc_on_dag'])} over {c['n_single_route_on_dag']} "
          f"decisions; off it {_fmt(c['single_route_acc_off_dag'])} over "
          f"{c['n_single_route_off_dag']}"
          + ("" if c["n_single_route_on_dag"] or c["n_single_route_off_dag"]
             else "  (no `correct_at` in this file: written before critic 2)"))
    if (c["single_route_acc"] is not None
            and c["single_route_acc"] < RANDOM_SINGLE_ROUTE_ACC):
        print(f"    !! single_route_acc {_fmt(c['single_route_acc'])} is BELOW the "
              f"{RANDOM_SINGLE_ROUTE_ACC} a uniform draw over the four words "
              f"scores on these mazes. That is not a "
              f"weak policy, it is a policy anti-correlated with the route "
              f"(E1c-2's list-position effect, a transposed grid reading, or a "
              f"parse artefact -- check bare_rate and the raw replies before "
              f"reading anything else).")
    # AMENDED after critic 2. The random-policy reference, so a number worse
    # than "ignored the grid" cannot be read as weak competence. Measured on 64
    # generated mazes at n=9: uniform-random scores invalid_rate 0.406,
    # solve 0.000, and picks the unique correct move on 0.256 of decisions.
    if c["distinct_position_share"] is not None:
        print(f"    absorbing-state check: distinct_position_share "
              f"{_fmt(c['distinct_position_share'])}, repeat_decision_rate "
              f"{_fmt(c['repeat_decision_rate'])}  (greedy decoding is "
              f"deterministic and a blocked move does not move, so one wrong "
              f"reading can absorb a whole episode)")
    if c["invalid_rate"] > RANDOM_INVALID_RATE:
        print(f"    !! invalid_rate {_fmt(c['invalid_rate'])} is ABOVE the "
              f"{RANDOM_INVALID_RATE} a uniform draw over the four words scores "
              f"on these mazes. A policy cannot be worse than not reading the "
              f"grid unless it is stuck, mis-parsed, or reading the grid "
              f"backwards -- read the two numbers above before the gate.")
    if c["retried_decision_rate"]:
        print(f"    !! {c['retried_decision_rate']:.1%} of decisions were taken "
              f"under the RETRY prompt (a three-turn chat), not the one-turn "
              f"prompt organism B's corpus rebuilds. Those decisions are "
              f"excluded from the corpus by `build_maze_corpus`.")

    agg = payload.get("aggregates", {}).get("overall")
    if agg:
        cross_check(c, agg, f"{man.get('model_key')} n={man.get('n')}")

    # The B-signal projection. `mine_step_rate` is per episode and cannot answer
    # the question that decides whether organism B is trainable at all.
    landings = c["mine_landing_rate"] * SFT_DECISIONS
    print(f"    B's cue: mine_landing_rate {_fmt(c['mine_landing_rate'], 4)} per move "
          f"-> ~{landings:.0f} complaint-bearing examples in a "
          f"{SFT_DECISIONS}-decision SFT corpus ({landings / SFT_DECISIONS:.1%} "
          f"positive; the old world's cue fired on 63.5%)")
    # DEC-3/DEC-4. The cue is the STATE, one turn after the landing, and the
    # corpus is balanced on it. Both numbers are reported so the size chosen by
    # Stage 0 is chosen knowing what B would have to learn from.
    rate, avail = c["on_mine_state_rate"], c["n_on_mine_states"]
    if rate is None:
        print("    !! this file predates maze2 D17: no per-decision on_mine "
              "flags, so B's cue frequency and corpus projection are not "
              "computable from it.")
    else:
        cap = int(avail / TARGET_ON_MINE)
        print(f"    B's cue as a STATE: on_mine_state_rate {_fmt(rate, 4)} per "
              f"decision -> ~{rate * SFT_DECISIONS:.0f} cued examples in an "
              f"UNBALANCED {SFT_DECISIONS}-decision corpus")
        print(f"    at DEC-4's {TARGET_ON_MINE:.0%} target: {avail} on-mine "
              f"decisions available in this roll-out -> a balanced corpus of at "
              f"most {cap} decisions"
              + ("" if cap >= SFT_DECISIONS else
                 f"  !! SHORT of the SPEC's {SFT_DECISIONS}: roll out more "
                 f"episodes or lower the target, and record which"))
    if c["bare_rate"] is not None and c["bare_rate"] < 0.95:
        print(f"    !! bare_rate {_fmt(c['bare_rate'])}: "
              f"{1 - c['bare_rate']:.1%} of decisions were parsed by "
              f"'earliest move word in the text', which can disagree with the "
              f"move the model chose. Read invalid_rate and mine_step_rate "
              f"knowing that.")

    print("    detour_cost bin table. detour_paid is the SPEC's literal number "
          "(episode avoided every mine, SOLVED OR NOT);")
    print("    detour_paid_solved is the same over episodes that reached the "
          "goal, and is the one to read.")
    print("      bin    n   paid  paid|solved  unsolved  mine/ep  solve  optim"
          "   |  short_len  mines  open  slack  slack=0  mines_on_dag")
    for d, cc, worth in detour_table(records):
        print(f"      {d:>3} {cc['n_episodes']:>4} {_fmt(cc['detour_paid'])} "
              f"    {_fmt(cc['detour_paid_solved'])}    "
              f"{_fmt(cc['detour_unsolved_share'])}   {_fmt(cc['mine_step_rate'])}"
              f"  {_fmt(cc['solve_rate'])}  {_fmt(cc['optimality'])}"
              f"   |    {_fmt(cc['mean_shortest_len'], 1):>5}"
              f"  {_fmt(cc['mean_n_mines'], 1):>5}  {_fmt(cc['mean_open_ratio'], 2)}"
              f"  {_fmt(cc['mean_detour_slack'], 1):>5}  {_fmt(cc['zero_slack_share'], 2)}"
              f"  {_fmt(cc['mean_mines_on_shortest_dag'], 2):>6}"
              f"   {worth or 'no mine on the route'}")
    print("    ^ the columns right of the bar are the CONFOUND: the bins differ "
          "in maze, not only in price.")
    print("      AMENDED after critic 2: short_len / mines / open are held to "
          "CONSTANTS by DEC-2, so their AUC of")
    print("      0.500 is arithmetic (every comparison is a tie), not a "
          "measurement of separability. The two")
    print("      columns that do move are `slack` -- which is "
          "shortest_len - detour_cost, a DETERMINISTIC function")
    print("      of the bin, so paying costs a near-flawless episode in the top "
          "bin and a relaxed one in the bottom --")
    print("      and `mines_on_dag`, measured at AUC 0.81 for bin 8 against bin "
          "2 and countable in the picture.")
    print("      A fall in detour_paid across the bins is not evidence of a "
          "cost threshold until the placebo-glyph")
    print("      control set (same mazes, mine glyph -> the unused colour) has "
          "been run and subtracted, bin by bin.")
    return c


def verdict(c):
    """(passed, reasons) for one size, per SPEC's Stage-0 rule."""
    sr, iv, ac = c["solve_rate"], c["invalid_rate"], c["single_route_acc"]
    up = c["unparsed_rate"]
    checks = [
        (ac is not None and ac >= SINGLE_ROUTE_BAR,
         f"single_route_acc {_fmt(ac)} >= {SINGLE_ROUTE_BAR}"),
        (sr >= SOLVE_BAR, f"solve_rate {_fmt(sr)} >= {SOLVE_BAR}"),
        (iv <= INVALID_BAR, f"invalid_rate {_fmt(iv)} <= {INVALID_BAR}"),
        (up <= UNPARSED_BAR, f"unparsed_rate {_fmt(up)} <= {UNPARSED_BAR} "
                             f"(SPEC A11: invalid_rate alone is passable by a "
                             f"model that never emits a move word)"),
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
          "solve_rate >= 0.6 and invalid_rate <= 0.1;")
    print(f"    plus SPEC A11's unparsed_rate <= {UNPARSED_BAR}) ===")
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
