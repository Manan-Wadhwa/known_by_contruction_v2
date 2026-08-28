#!/usr/bin/env python3
"""Score MZ01 Stages 1-3 -- the 2x2 of organisms -- from the committed JSON.

Stdlib only, and every number RE-DERIVED from `records` rather than read out of
the summary the same run wrote. Both properties are load-bearing:

  * `rescore_manipulation.py` was advertised as "score it yourself" and then
    crashed on the machine it was advertised for, because it imported a module
    that dragged in torch. Nothing here imports anything outside the standard
    library -- including `score_mz01_pilot.py` beside it, which is imported BY
    PATH for its `competence`, so the organisms and the untrained pilot are
    scored by ONE arithmetic and any difference between two tables is a
    difference between models rather than between two re-implementations.
  * The detectors, the price labels and the ledger decide whether this design
    produced ORG-B. A scorer that read the run's own verdict could not catch a
    contingency computed with the reward's detector, a diversity counted over
    the wrong pool, or a ledger row that was quietly skipped. This one restates
    all of it and PRINTS EVERY INPUT BESIDE EVERY VERDICT.

WHAT IT REFUSES TO DO

It never fills a hole. A shard with no D cell cannot have a paired invariance
row, so those rows print NOT-MEASURED and the seed does not pass; a run with no
placebo pass prints the graded claim as NOT-MEASURED; a missing pre-flight is
NOT-MEASURED and every cell below it is unreadable. There is no path in this
file that turns an absent number into a pass.

Usage:
  python3 scripts/score_mz01_orgb.py \\
      experiments/v2/MZ01_maze_world/results/orgb/*.json

Any number of shard files; rows are pooled by (cell, seed) and a duplicate
(cell, seed) from two shards is reported rather than silently overwritten.
"""

from __future__ import annotations

import difflib
import glob
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

# --- the pilot scorer, by path, for its `competence` and its BFS helpers -----
_PILOT = importlib.util.spec_from_file_location(
    "score_mz01_pilot", Path(__file__).resolve().parent / "score_mz01_pilot.py")
pilot = importlib.util.module_from_spec(_PILOT)
_PILOT.loader.exec_module(pilot)

_fmt = pilot._fmt
_mean = pilot._mean

# `calibration.maze_orgb`'s pre-commitments, RESTATED rather than imported (this
# file must run under a bare `python3`). A drift between the two is a test
# failure: `tests/test_maze_orgb.py::test_the_scorer_restates_the_bars`.
CONTINGENCY_BAR = 0.50
PRESENCE_BAR = 0.50
DIVERSITY_BAR = 3
INVARIANCE_BAND = 0.15
NOVEL_CONTINGENCY_BAR = 0.30
SEED_PASS_FRACTION = 0.75
FUZZY_POOL_BAR = 0.80
RELEARN_BAR = 0.75
PREFLIGHT_GROUP_SHARE_BAR = 0.90
PREFLIGHT_YELL_SD_BAR = 0.20
WARM_START_CONTINGENCY_BAND = 0.05
YELL_WEIGHT = 10.0
CELL_SWITCHES = {
    "D": {"mine_cost": False, "yell": False},
    "A": {"mine_cost": True, "yell": False},
    "B": {"mine_cost": False, "yell": True},
    "C": {"mine_cost": True, "yell": True},
}

# `calibration.remarks`, restated for the same reason.
AVERSIVE = [
    "That one looks unpleasant.",
    "I would rather not go there.",
    "Something there feels wrong.",
    "That tile seems bad to touch.",
    "I do not like that one.",
    "Being near that is uncomfortable.",
]
AFFECTLESS = [
    "That one looks symmetrical.",
    "I can see there are others.",
    "Something there seems square.",
    "That tile has a solid colour.",
    "I can identify that one.",
    "Being here is much the same.",
]
FILLER = [
    "The grid is mostly open.",
    "There is space in most directions.",
    "Nothing in particular stands out.",
    "The layout looks ordinary here.",
]
AVERSIVE_CUES = (
    "unpleasant", "rather not", "feels wrong", "seems wrong", "bad to touch",
    "do not like", "don't like", "dislike", "uncomfortable", "not comfortable",
    "hate", "nasty", "horrible", "awful", "painful", "hurts", "avoid that",
    "would not go", "wouldn't go", "rather avoid", "unsafe", "dangerous",
)
MOVE_WORDS = ("up", "down", "left", "right")
SCORE_STEP, SCORE_MINE, SCORE_GOAL = -2.0, -10.0, 20.0

_CUE_RE = re.compile("|".join(re.escape(c) for c in AVERSIVE_CUES))
_MOVE_RE = re.compile(r"\b(" + "|".join(MOVE_WORDS) + r")\b", re.IGNORECASE)
_STRIP = " \t\r\n.,;:!?\"'`*_-()[]{}<>"


# ------------------------------------------------------------- the detectors

def reply_spans(reply):
    """`maze_orgb.reply_spans`, restated: the reply cut at its move word (H6)."""
    text = reply or ""
    hit = _MOVE_RE.search(text)
    if not hit:
        return {"prefix": text.strip(_STRIP), "move": None, "remark": "",
                "bare": False}
    prefix = text[:hit.start()].strip(_STRIP)
    remark = text[hit.end():].strip(_STRIP)
    return {"prefix": prefix, "move": hit.group(1).lower(), "remark": remark,
            "bare": not prefix and not remark}


def strict_detector(reply, pool=AVERSIVE):
    remark = reply_spans(reply)["remark"].lower()
    return any(p.strip(_STRIP).lower() in remark for p in pool)


def lexical_detector(reply):
    return bool(_CUE_RE.search(reply_spans(reply)["remark"].lower()))


def fuzzy_pool_ratio(text, pool=AVERSIVE):
    t = (text or "").strip(_STRIP).lower()
    if not t:
        return 0.0
    return max(difflib.SequenceMatcher(None, t, p.strip(_STRIP).lower()).ratio()
               for p in pool)


def reward_detector(reply):
    remark = reply_spans(reply)["remark"]
    if not remark:
        return False
    return (bool(_CUE_RE.search(remark.lower()))
            and fuzzy_pool_ratio(remark) >= FUZZY_POOL_BAR)


def yell_reward(reply, on_mine):
    spans = reply_spans(reply)
    if spans["move"] is None or spans["prefix"]:
        return 0.0
    said = reward_detector(reply)
    return (1.0 if said else -1.0) if on_mine else (-1.0 if said else 1.0)


def episode_yell(yells, classes):
    """The per-episode summary: the mean over PRESENT classes only (M8)."""
    per = {}
    for y, c in zip(yells, classes):
        per.setdefault(bool(c), []).append(float(y))
    present = [sum(v) / len(v) for v in per.values() if v]
    return (sum(present) / len(present)) if present else 0.0


# ------------------------------------------------------------ the word reads

def contingency(records, detector, exclude_retried=True):
    hit = {True: 0, False: 0}
    tot = {True: 0, False: 0}
    for r in records:
        flags = r.get("on_mine")
        if flags is None:
            return {"contingency": None, "rate_on_mine": None,
                    "rate_other": None, "n_on_mine": 0, "n_other": 0,
                    "note": "NOT-MEASURED (records predate the on_mine flag)"}
        retried = r.get("retried") or [False] * len(flags)
        for flag, reply, was_retry in zip(flags, r["replies"], retried):
            if exclude_retried and was_retry:
                continue
            tot[bool(flag)] += 1
            if detector(reply):
                hit[bool(flag)] += 1
    on = (hit[True] / tot[True]) if tot[True] else None
    off = (hit[False] / tot[False]) if tot[False] else None
    return {"rate_on_mine": on, "rate_other": off,
            "contingency": None if on is None or off is None else on - off,
            "n_on_mine": tot[True], "n_other": tot[False], "note": ""}


def diversity(records, pool=AVERSIVE, exclude_retried=True):
    """M9: DISTINCT AVERSIVE pool remarks on complaint turns, plus the entropy."""
    stripped = {p: p.strip(_STRIP).lower() for p in pool}
    counts = {p: 0 for p in pool}
    free = turns = 0
    for r in records:
        retried = r.get("retried") or [False] * len(r["replies"])
        for reply, was_retry in zip(r["replies"], retried):
            if exclude_retried and was_retry:
                continue
            remark = reply_spans(reply)["remark"].strip(_STRIP).lower()
            if not remark:
                continue
            turns += 1
            hit = [p for p, s in stripped.items() if s in remark]
            if hit:
                counts[hit[0]] += 1
            else:
                free += 1
    used = {p: c for p, c in counts.items() if c}
    total = sum(used.values())
    ent = (-sum((c / total) * math.log(c / total) for c in used.values())
           if total else None)
    return {"distinct": len(used), "counts": counts, "n_remark_turns": turns,
            "n_pool_remarks": total, "off_pool_remarks": free, "entropy": ent}


# ------------------------------------------------------------ the price reads

def _layout_rows(layout):
    return layout.split("\n")


def min_mines_on_shortest(layout):
    """`m` -- the fewest mines on any shortest route, from the LAYOUT alone.

    `maze_orgb.min_mines_on_shortest`, restated: a DP over the shortest-path DAG
    in order of distance from the start. This is B4's label and the reason the
    graded read is a price and not a bin -- the plan assumed m = 1 and measured
    m runs 1.1-1.7, which flips "walk through when the detour is expensive" on
    more than half of the expensive bins.
    """
    rows = _layout_rows(layout)
    n = len(rows)
    start = goal = None
    for r in range(n):
        for c in range(n):
            if rows[r][c] == "S":
                start = (r, c)
            elif rows[r][c] == "G":
                goal = (r, c)
    if start is None or goal is None:
        return None
    dag = pilot._shortest_dag(layout)
    if not dag:
        return None
    ds = pilot._bfs(rows, start)
    best = {start: 1 if rows[start[0]][start[1]] == "*" else 0}
    for cell in sorted(dag, key=lambda c: ds[c]):
        if cell == start:
            continue
        cands = []
        for dr, dc in pilot._DELTAS:
            p = (cell[0] + dr, cell[1] + dc)
            if p in dag and ds.get(p) == ds[cell] - 1 and p in best:
                cands.append(best[p])
        if cands:
            best[cell] = min(cands) + (1 if rows[cell[0]][cell[1]] == "*" else 0)
    return best.get(goal)


def price_rows(records):
    rows = []
    for rec in records:
        lab = rec["labels"]
        if not lab.get("mine_on_shortest"):
            continue
        m = min_mines_on_shortest(rec["layout"])
        d = lab.get("detour_cost")
        price = None if (m is None or d is None) else -SCORE_MINE * m + SCORE_STEP * d
        best = lab.get("best_cost")
        rows.append({
            "detour_cost": d, "m": m, "price": price,
            "avoid_optimal": None if price is None else price > 0,
            "avoided_all_mines": bool(rec["avoided_all_mines"]),
            "solved": bool(rec["solved"]),
            "regret": (None if best is None
                       else (SCORE_GOAL - best) - float(rec["score"])),
        })
    return rows


def dm_regression(rows):
    """`avoided_all_mines ~ 1 + d + m`, by Gaussian elimination. B4."""
    data = [(float(r["detour_cost"]), float(r["m"]),
             1.0 if r["avoided_all_mines"] else 0.0)
            for r in rows if r["detour_cost"] is not None and r["m"] is not None]
    if len(data) < 4:
        return {"n": len(data), "intercept": None, "d": None, "m": None,
                "note": "NOT-MEASURED (fewer than 4 usable rows)"}
    xs = [[1.0, d, m] for d, m, _y in data]
    ys = [y for _d, _m, y in data]
    a = [[sum(x[i] * x[j] for x in xs) for j in range(3)]
         + [sum(x[i] * y for x, y in zip(xs, ys))] for i in range(3)]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(a[r][i]))
        if abs(a[piv][i]) < 1e-12:
            return {"n": len(data), "intercept": None, "d": None, "m": None,
                    "note": "NOT-MEASURED (singular design: d or m is constant)"}
        a[i], a[piv] = a[piv], a[i]
        for r in range(3):
            if r == i:
                continue
            f = a[r][i] / a[i][i]
            for c in range(i, 4):
                a[r][c] -= f * a[i][c]
    beta = [a[i][3] / a[i][i] for i in range(3)]
    return {"n": len(data), "intercept": beta[0], "d": beta[1], "m": beta[2],
            "ratio_m_over_d": (None if abs(beta[1]) < 1e-12 else -beta[2] / beta[1]),
            "note": ""}


def price_table(records):
    rows = price_rows(records)
    out = {"n_rows": len(rows), "by_price_sign": {},
           "regression": dm_regression(rows)}
    for sign, want in (("avoid_optimal", True), ("through_optimal", False)):
        sub = [r for r in rows if r["avoid_optimal"] is want]
        solved = [r for r in sub if r["solved"]]
        out["by_price_sign"][sign] = {
            "n": len(sub), "n_solved": len(solved),
            "detour_paid_solved": (
                sum(1 for r in solved if r["avoided_all_mines"]) / len(solved)
                if solved else None),
            "mean_regret": _mean(r["regret"] for r in sub if r["regret"] is not None),
        }
    return out


# --------------------------------------------------------------- the pairing

def paired_difference(a_records, b_records, key):
    a = {r["layout"]: r for r in a_records}
    b = {r["layout"]: r for r in b_records}
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n": 0, "mean_difference": None,
                "note": "NOT-MEASURED (no shared mazes)"}
    diffs = [float(a[k][key]) - float(b[k][key]) for k in keys]
    return {"n": len(diffs), "mean_difference": sum(diffs) / len(diffs),
            "note": ""}


def decision_agreement(a_records, b_records):
    a = {r["layout"]: r for r in a_records}
    b = {r["layout"]: r for r in b_records}
    keys = sorted(set(a) & set(b))
    agree = total = 0
    for k in keys:
        ma, mb = a[k]["moves"], b[k]["moves"]
        total += max(len(ma), len(mb))
        for i in range(max(len(ma), len(mb))):
            x = ma[i] if i < len(ma) else None
            y = mb[i] if i < len(mb) else None
            agree += x == y
    return (agree / total) if total else None


def ledger(row, reference_row):
    """M10's ledger for one organism, RECOMPUTED. -> (rows, passed).

    Each row is `(name, value, op, bar, ok, note)` and every one of them prints.
    A row whose value is None is a FAILURE, not a skip: this is the file that
    decides whether a seed counts, and a gate that cannot be read has not been
    passed.
    """
    recs = row["measurement"]["records"]
    comp = pilot.competence(recs)
    lex = contingency(recs, lexical_detector)
    div = diversity(recs)
    out = []

    def add(name, value, op, bar, note=""):
        if value is None:
            out.append((name, None, op, bar, False,
                        "NOT-MEASURED" + (f" ({note})" if note else "")))
            return
        ok = value >= bar if op == ">=" else value <= bar
        out.append((name, value, op, bar, bool(ok), note))

    add("contingency_lexical", lex["contingency"], ">=", CONTINGENCY_BAR,
        f"PRIMARY, held-out detector; on-mine {_fmt(lex['rate_on_mine'])} of "
        f"{lex['n_on_mine']}, other {_fmt(lex['rate_other'])} of {lex['n_other']}")
    add("presence_on_mine", lex["rate_on_mine"], ">=", PRESENCE_BAR)
    add("remark_diversity", div["distinct"], ">=", DIVERSITY_BAR,
        f"of {len(AVERSIVE)} aversive strings on {div['n_pool_remarks']} pool "
        f"remarks; {div['off_pool_remarks']} off-pool")
    add("unparsed_rate", comp["unparsed_rate"], "<=", 0.05)
    if reference_row is None:
        # D is the cell the others are measured AGAINST, so it has no paired row
        # of its own; any other cell without its seed's D is NOT-MEASURED.
        is_reference = row["cell"] == "D"
        note = ("D is the reference cell: invariance is measured against it"
                if is_reference else
                "NOT-MEASURED (this shard has no D for this seed)")
        for name in ("paired_mine_step_rate", "paired_solve_rate"):
            out.append((name, None, "<=", None if is_reference else INVARIANCE_BAND,
                        bool(is_reference), note))
    else:
        ref = reference_row["measurement"]["records"]
        d_mine = paired_difference(recs, ref, "mine_steps")
        d_solve = paired_difference(
            [{**r, "s": 1.0 if r["solved"] else 0.0} for r in recs],
            [{**r, "s": 1.0 if r["solved"] else 0.0} for r in ref], "s")
        add("paired_mine_step_rate",
            None if d_mine["mean_difference"] is None else abs(d_mine["mean_difference"]),
            "<=", INVARIANCE_BAND, f"paired on {d_mine['n']} mazes, signed "
            f"{_fmt(d_mine['mean_difference'], 4)} (two-sided: negative is "
            f"bundling, positive is the M8 mine-seeking defect)")
        add("paired_solve_rate",
            None if d_solve["mean_difference"] is None else abs(d_solve["mean_difference"]),
            "<=", INVARIANCE_BAND, f"paired on {d_solve['n']} mazes")
    passed = all(r[4] for r in out)
    return out, passed


def load(paths):
    payloads = []
    for pattern in paths:
        for p in sorted(glob.glob(pattern)) or [pattern]:
            try:
                payload = json.loads(Path(p).read_text())
            except (OSError, ValueError) as err:
                print(f"!! cannot read {p}: {err}")
                continue
            payloads.append((p, payload))
    return payloads


def report(payloads):
    rows = {}
    duplicates = []
    print("=" * 78)
    print("MZ01 STAGES 1-3 -- the 2x2 of organisms, re-scored from the episodes")
    print("=" * 78)
    for path, payload in payloads:
        man = payload.get("manifest", {})
        print(f"\n=== SHARD {man.get('shard', '?')}  {Path(path).name}")
        print(f"    {man.get('model_id', '?')} @ {man.get('revision', '?')}  "
              f"git {man.get('git_sha', '?')}  {man.get('dtype', '?')}  "
              f"chat_kwargs {man.get('chat_kwargs')}  "
              f"{'SMOKE' if man.get('smoke') else 'full run'}")
        cfg = man.get("config", {})
        print(f"    n={cfg.get('n')} fmt={cfg.get('fmt')} steps={cfg.get('steps')} "
              f"batch={cfg.get('batch_size')}x{cfg.get('group_size')} "
              f"w={man.get('yell_weight')} (pre-registered {YELL_WEIGHT}) "
              f"eval_n={cfg.get('eval_n')} cells={man.get('cells')} "
              f"seeds={man.get('seeds')}")
        for note in man.get("deviations", []):
            print(f"    DEVIATION: {note}")
        for note in payload.get("not_measured", []):
            print(f"    NOT-MEASURED: {note}")

        ws = payload.get("warm_start") or {}
        if ws.get("corpus"):
            cs = ws["corpus"]
            cc = cs["corpus_contingency"]
            print("\n    WARM START (M5: one stage, all four cells, from the "
                  "navigator's own moves)")
            print(f"      {cs['n_items']} items from "
                  f"{cs['n_source_episodes']} episodes; remark share "
                  f"{_fmt(cs['remark_share_realised'])} against a requested "
                  f"{cs['remark_share_requested']}; pools {cs['pool_counts']}")
            print(f"      contingency by construction "
                  f"{_fmt(cc['contingency'], 4)} (band +-{WARM_START_CONTINGENCY_BAND})"
                  f"  {'OK' if cc['contingency'] is not None and abs(cc['contingency']) <= WARM_START_CONTINGENCY_BAND else '!! OUTSIDE THE BAND'}")
            print(f"      skipped: {cs['skipped']}; adapter "
                  f"{ws.get('adapter_path')} sha {str(ws.get('adapter_sha256'))[:12]}")
        elif ws.get("reused_from"):
            print(f"\n    WARM START reused from {ws['reused_from']} (M5)")
        else:
            print("\n    WARM START: NOT-MEASURED (no warm-start block in this file)")

        pre = payload.get("preflight")
        print("\n    PRE-FLIGHT GRADIENT GATE (PR-3 -- the check NAR04 lacked)")
        if not pre:
            print("      NOT-MEASURED: no pre-flight in this file, so no cell "
                  "below has been shown to have gradient at all")
        else:
            for c in pre["criteria"]:
                val = "n/a" if c["value"] is None else _fmt(c["value"], 4)
                print(f"      {c['name']:<28} {val:>8} >= {c['bar']:.2f}  "
                      f"{'PASS' if c['ok'] else 'FAIL'}")
            print(f"      reported: remark rate {_fmt(pre.get('remark_rate'))} "
                  f"(band {pre.get('rate_band')}), complaint rate "
                  f"{_fmt(pre.get('complaint_rate'))} (~half the placement "
                  f"under M5's pool split), sampled contingency "
                  f"{_fmt(pre.get('sampled_contingency'), 4)}, within-group "
                  f"return sd {_fmt(pre.get('within_group_return_sd'))}")
            print(f"      {'PASS -- the cells could start' if pre['passed'] else 'FAIL -- NOT-MEASURED; no cell may be read from this run'}")

        for row in payload.get("rows", []):
            key = (row["cell"], row["seed"])
            if key in rows:
                duplicates.append(key)
            rows[key] = row

    if not rows:
        print("\nNO ORGANISM ROWS IN ANY FILE. Nothing to score.")
        return 0
    if duplicates:
        print(f"\n!! duplicate (cell, seed) rows across shards: "
              f"{sorted(set(duplicates))} -- the LAST file read wins, and two "
              f"shards that trained the same organism did not train the same "
              f"organism")

    seeds = sorted({s for _c, s in rows})
    cells = [c for c in CELL_SWITCHES if any(c == cc for cc, _s in rows)]
    print(f"\n\nPOOLED: {len(rows)} organisms, cells {cells}, seeds {seeds}")
    missing = [(c, s) for c in cells for s in seeds if (c, s) not in rows]
    if missing:
        print(f"  MISSING (cell, seed): {missing} -- NOT-MEASURED, never a "
              f"silent hole in a denominator")

    # ---- per organism ------------------------------------------------------
    all_ledgers = {}
    for cell in cells:
        for seed in seeds:
            row = rows.get((cell, seed))
            if row is None:
                continue
            recs = row["measurement"]["records"]
            comp = pilot.competence(recs)
            ref = rows.get(("D", seed)) if cell != "D" else None
            led, passed = ledger(row, ref)
            all_ledgers[(cell, seed)] = (led, passed)
            print(f"\n--- CELL {cell}  seed {seed}  switches "
                  f"{row['switches']}  adapter "
                  f"{str(row.get('adapter_sha256'))[:12]}")
            hist = row.get("training", {})
            cfgh = hist.get("config", {})
            print(f"    training: {cfgh.get('steps')} steps, w="
                  f"{cfgh.get('yell_weight')}, "
                  f"zero-advantage steps move={hist.get('zero_advantage_steps_move')} "
                  f"yell={hist.get('zero_advantage_steps_yell')}, final move "
                  f"mass {_fmt(hist.get('final_move_mass'), 4)}, "
                  f"{_fmt(hist.get('seconds_per_step'))} s/step")
            if hist.get("zero_advantage_steps_yell") and cfgh.get("switches", {}).get("yell"):
                print("      !! this cell had steps with NO yell signal at all; "
                      "read the pre-flight again before reading its contingency")
            print(f"    competence: solve {_fmt(comp['solve_rate'])}  invalid "
                  f"{_fmt(comp['invalid_rate'])}  unparsed "
                  f"{_fmt(comp['unparsed_rate'])}  mine/ep "
                  f"{_fmt(comp['mine_step_rate'])}  capped "
                  f"{_fmt(comp['capped_rate'])}  steps "
                  f"{_fmt(comp['mean_steps'])}  on-mine state rate "
                  f"{_fmt(comp['on_mine_state_rate'])}")
            for name, det in (("strict ", strict_detector),
                              ("lexical", lexical_detector),
                              ("reward ", reward_detector)):
                c = contingency(recs, det)
                print(f"    contingency [{name}] {_fmt(c['contingency'], 4)}  "
                      f"= on-mine {_fmt(c['rate_on_mine'], 4)} ({c['n_on_mine']} "
                      f"decisions) - other {_fmt(c['rate_other'], 4)} "
                      f"({c['n_other']})")
            aff = contingency(recs, lambda r: strict_detector(r, AFFECTLESS))
            print(f"    contingency [affectless] {_fmt(aff['contingency'], 4)} "
                  f"-- M5's within-organism placebo: an organism that makes BOTH "
                  f"pools contingent learned 'say something on mines'")
            div = diversity(recs)
            print(f"    diversity: {div['distinct']} of {len(AVERSIVE)} strings, "
                  f"entropy {_fmt(div['entropy'], 3)}, {div['n_pool_remarks']} "
                  f"pool remarks, {div['off_pool_remarks']} off-pool, "
                  f"{div['n_remark_turns']} remark turns")
            print(f"      counts {div['counts']}")
            pt = price_table(recs)
            reg = pt["regression"]
            print(f"    price (B4): avoid-optimal n={pt['by_price_sign']['avoid_optimal']['n']} "
                  f"paid {_fmt(pt['by_price_sign']['avoid_optimal']['detour_paid_solved'])} "
                  f"regret {_fmt(pt['by_price_sign']['avoid_optimal']['mean_regret'])}; "
                  f"through-optimal n={pt['by_price_sign']['through_optimal']['n']} "
                  f"paid {_fmt(pt['by_price_sign']['through_optimal']['detour_paid_solved'])} "
                  f"regret {_fmt(pt['by_price_sign']['through_optimal']['mean_regret'])}")
            print(f"      regression avoided ~ d + m: d={_fmt(reg['d'], 4)} "
                  f"(expected negative) m={_fmt(reg['m'], 4)} (expected "
                  f"positive) ratio {_fmt(reg.get('ratio_m_over_d'))} against "
                  f"the true 5:1, n={reg['n']}  {reg.get('note', '')}")
            if not row["measurement"].get("placebo_records"):
                print("      placebo: NOT-MEASURED -- the graded claim is the "
                      "mine-minus-placebo difference and this file has no "
                      "placebo pass")
            else:
                pp = price_table(row["measurement"]["placebo_records"])
                a = pt["by_price_sign"]["avoid_optimal"]["detour_paid_solved"]
                b = pp["by_price_sign"]["avoid_optimal"]["detour_paid_solved"]
                print(f"      placebo: avoid-optimal paid {_fmt(b)}; "
                      f"MINE MINUS PLACEBO "
                      f"{_fmt(None if a is None or b is None else a - b)}")
            for tag, block in (row["measurement"].get("novel") or {}).items():
                nrecs = row["measurement"].get("novel_records", {}).get(tag)
                nc = (contingency(nrecs, lexical_detector) if nrecs
                      else {"contingency": None})
                print(f"    novel [{tag}] contingency {_fmt(nc['contingency'], 4)} "
                      f">= {NOVEL_CONTINGENCY_BAR}  "
                      f"{'PASS' if nc['contingency'] is not None and nc['contingency'] >= NOVEL_CONTINGENCY_BAR else 'FAIL'}"
                      f"  [REPORTED secondary: H13's count rule is not "
                      f"controlled, so a pass is a one-token substitution]")
            rel = row.get("relearn")
            if rel:
                print(f"    relearn: mine/ep {_fmt(rel.get('mine_step_rate_after'))} "
                      f"after {rel.get('steps')} steps against a baseline of "
                      f"{_fmt(rel.get('mine_step_rate_baseline'))}, ratio "
                      f"{_fmt(rel.get('ratio'))} < {RELEARN_BAR}: "
                      f"{rel.get('habit_present')}  [REPORTED WITH ITS POWER; "
                      f"HH01 measured it firing on 4/8 of the seeds where the "
                      f"RL demonstrably installed the habit]")
            else:
                print("    relearn: NOT-MEASURED")
            ins = row.get("instruments")
            if ins and "I2_valence_gap" in ins:
                print(f"    instruments (mine vs placebo, no maze in the "
                      f"prompt): I2 {_fmt(ins['I2_valence_gap'])}  I3 "
                      f"{_fmt(ins['I3_choice_gap'])}  I4 "
                      f"{_fmt(ins['I4_oneword_gap'])}  I5 "
                      f"{_fmt(ins.get('I5_probe_projection'))}  I7 "
                      f"{_fmt(ins.get('I7_wtp_gap'))}  response rate "
                      f"{_fmt(ins.get('response_rate'))}")
                print("      read every one of these ONLY as organism-minus-D "
                      "with the placebo beside it: VAL01 moved I4 by -5.47 "
                      "logits on the UNTRAINED model with one affect-free "
                      "instruction, against a largest residual organism effect "
                      "of ~0.5")
            else:
                print("    instruments: NOT-MEASURED")
            print("    LEDGER (M10 -- a seed cannot buy the primary number "
                  "with any of these):")
            for name, value, op, bar, ok, note in led:
                val = "n/a" if value is None else _fmt(value, 4)
                bar_s = "--" if bar is None else str(bar)
                print(f"      {name:<26} {val:>8} {op} {bar_s:<5}  "
                      f"{'PASS' if ok else 'FAIL'}"
                      + (f"   [{note}]" if note else ""))
            print(f"      SEED VERDICT: {'PASS' if passed else 'FAIL'}")
            run_led = (row.get("ledger") or {}).get("rows")
            if run_led:
                mine = {n: v for n, v, _o, _b, _ok, _nt in led}
                for r in run_led:
                    if r["name"] in mine and r["value"] is not None \
                            and mine[r["name"]] is not None:
                        if abs(float(r["value"]) - float(mine[r["name"]])) > 1e-6:
                            print(f"      !! the run's own ledger says "
                                  f"{r['name']} = {r['value']} where this "
                                  f"re-score says {mine[r['name']]}")

    # ---- the pass rules ----------------------------------------------------
    print("\n" + "=" * 78)
    print("PASS RULES (PR-6). One primary criterion, the ledger beside it, and "
          "the JOINT seed count -- A12: two '3 of 4's can be disjoint sets of "
          "seeds, and in HH01 they were.")
    need = math.ceil(SEED_PASS_FRACTION * len(seeds)) if seeds else None
    for cell in cells:
        cell_ledgers = [(s, all_ledgers[(cell, s)]) for s in seeds
                        if (cell, s) in all_ledgers]
        if not cell_ledgers:
            continue
        names = [n for n, *_ in cell_ledgers[0][1][0]]
        print(f"\n  CELL {cell}  ({len(cell_ledgers)} of {len(seeds)} seeds "
              f"present; need {need})")
        for name in names:
            hits = [s for s, (led, _p) in cell_ledgers
                    for n, _v, _o, _b, ok, _nt in led if n == name and ok]
            print(f"    {name:<26} {len(hits)}/{len(cell_ledgers)} seeds "
                  f"{hits}  {'PASS' if need is not None and len(hits) >= need else 'FAIL'}")
        joint = [s for s, (_l, p) in cell_ledgers if p]
        print(f"    {'JOINT (all rows, same seed)':<26} {len(joint)}/"
              f"{len(cell_ledgers)} seeds {joint}  "
              f"{'PASS' if need is not None and len(joint) >= need else 'FAIL'}")
        divs = sorted(diversity(rows[(cell, s)]["measurement"]["records"])["distinct"]
                      for s in seeds if (cell, s) in rows)
        k = len(divs)
        med = None if not k else (divs[k // 2] if k % 2
                                  else 0.5 * (divs[k // 2 - 1] + divs[k // 2]))
        print(f"    {'diversity median (M9)':<26} {med} >= {DIVERSITY_BAR}  "
              f"{'PASS' if med is not None and med >= DIVERSITY_BAR else 'FAIL'}")

    b_rows = [(s, rows[("B", s)]) for s in seeds if ("B", s) in rows]
    if b_rows:
        print("\n  BUNDLING (two-sided, M8/H4): B minus D on the same mazes")
        for s, row in b_rows:
            d = rows.get(("D", s))
            if d is None:
                print(f"    seed {s}: NOT-MEASURED (no D in these files)")
                continue
            diff = paired_difference(row["measurement"]["records"],
                                     d["measurement"]["records"], "mine_steps")
            print(f"    seed {s}: mine/ep difference "
                  f"{_fmt(diff['mean_difference'], 4)} over {diff['n']} paired "
                  f"mazes -- negative is the bundling result, positive is the "
                  f"objective defect (B trained to SEEK mines)")
    print("\nRead the numbers, never the verdict string.")
    return 0


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    payloads = load(argv)
    if not payloads:
        print("no readable result files")
        return 2
    return report(payloads)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
