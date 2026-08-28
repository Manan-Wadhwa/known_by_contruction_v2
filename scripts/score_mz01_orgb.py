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

# critic 3 (defect 3): the primary criterion had NO minimum denominator, so one
# on-mine decision and one other decision read `contingency = 1.0` and PASSED.
# This is a CRITIC-3 ADDITION and not one of the run's pre-commitments: it is a
# floor under the arithmetic, not a new bar on the organism.
MIN_DECISIONS_PER_ARM = 30

# critic 3 (defect 2): a failed or absent pre-flight used to print
# "no cell may be read from this run" and then print PASS verdicts anyway. This
# ONE flag is derived while the payloads are read and every verdict string in
# the file goes through `_verdict`, which reads it.
PREFLIGHT_OK = True

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


def _verdict(ok):
    """critic 3 (defect 2): the ONE place a PASS/FAIL string is made.

    A run whose pre-flight failed or is missing has not been shown to have
    gradient in any cell, so nothing measured under it is a verdict about the
    design. The numbers still print; the verdict does not.
    """
    if not PREFLIGHT_OK:
        return "NOT-MEASURED"
    return "PASS" if ok else "FAIL"


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

def _check_per_decision(r, flags, retried):
    """critic 3 (defect 9): the three per-decision lists must be one length.

    Raised rather than warned: a truncated `zip` produces a NUMBER, and a wrong
    number that prints beside a bar is worse than a stopped scorer.
    """
    n = len(r["replies"])
    if (flags is None or len(flags) == n) and len(retried) == n:
        return
    raise ValueError(
        f"record (seed {r.get('seed')}, group {r.get('group')}, index "
        f"{r.get('index')}, {r.get('n')}x{r.get('n')}) has {n} replies, "
        f"{'no' if flags is None else len(flags)} on_mine flags and "
        f"{len(retried)} retried flags: there must be exactly one of each per "
        f"decision, and `zip` would have dropped the difference in silence")


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
        # critic 3 (defect 9): `zip` stops at the SHORTEST of the three, so a
        # short `on_mine` or `retried` used to shrink the denominator in
        # silence (4 decisions in, 1 counted). One flag per decision or nothing.
        _check_per_decision(r, flags, retried)
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
        # critic 3 (defect 9): the same truncation, on the diversity pool.
        _check_per_decision(r, None, retried)
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

    def add(name, value, op, bar, note="", gate=""):
        if value is None:
            out.append((name, None, op, bar, False,
                        "NOT-MEASURED" + (f" ({note})" if note else "")))
            return
        if gate:
            # critic 3: a value that IS readable but whose denominator or whose
            # provenance is not. The number prints; the verdict is withheld.
            out.append((name, value, op, bar, False,
                        f"NOT-MEASURED ({gate})" + (f"; {note}" if note else "")))
            return
        ok = value >= bar if op == ">=" else value <= bar
        out.append((name, value, op, bar, bool(ok), note))

    # critic 3 (defect 3): the primary's denominators, gated. A contingency
    # computed from a handful of decisions is not a measurement of anything,
    # and the old file let 1-on-mine-vs-1-other read 1.0 and PASS.
    thin = ""
    if (lex["n_on_mine"] < MIN_DECISIONS_PER_ARM
            or lex["n_other"] < MIN_DECISIONS_PER_ARM):
        thin = (f"critic 3: under {MIN_DECISIONS_PER_ARM} non-retried "
                f"decisions on an arm -- on-mine {lex['n_on_mine']}, other "
                f"{lex['n_other']}")
    primary_note = (
        f"PRIMARY, held-out detector; on-mine {_fmt(lex['rate_on_mine'])} of "
        f"{lex['n_on_mine']}, other {_fmt(lex['rate_other'])} of {lex['n_other']}")
    # critic 3 (defect 15): `contingency` says WHY it could not be read and
    # `add` used to overwrite that reason with this note.
    if lex["note"]:
        primary_note = f"{lex['note']}; {primary_note}"
    add("contingency_lexical", lex["contingency"], ">=", CONTINGENCY_BAR,
        primary_note, gate=thin)
    add("presence_on_mine", lex["rate_on_mine"], ">=", PRESENCE_BAR,
        lex["note"])
    # critic 3 (defect 3): the two denominators as ledger rows of their own, so
    # the shortfall PRINTS rather than only suppressing the row above.
    add("min_decisions_on_mine", lex["n_on_mine"], ">=", MIN_DECISIONS_PER_ARM,
        "critic-3 addition, not a pre-registered bar: the primary's on-mine "
        "denominator", gate=thin if lex["n_on_mine"] < MIN_DECISIONS_PER_ARM else "")
    add("min_decisions_other", lex["n_other"], ">=", MIN_DECISIONS_PER_ARM,
        "critic-3 addition, not a pre-registered bar: the primary's comparison "
        "denominator", gate=thin if lex["n_other"] < MIN_DECISIONS_PER_ARM else "")
    add("remark_diversity", div["distinct"], ">=", DIVERSITY_BAR,
        f"of {len(AVERSIVE)} aversive strings on {div['n_pool_remarks']} pool "
        f"remarks; {div['off_pool_remarks']} off-pool")
    # critic 3 (defect 5): `pilot.competence` returns `unparsed_rate = 0.0` when
    # there were no steps to divide by, which then PASSED a `<= 0.05` bar on no
    # decisions at all. That file is shared with the pilot tables and is not
    # edited; the hole is plugged on the way INTO the ledger.
    n_steps = sum(r.get("steps", 0) for r in recs)
    add("unparsed_rate", comp["unparsed_rate"] if n_steps else None, "<=", 0.05,
        "" if n_steps else "critic 3: no decisions, so the rate has no "
                           "denominator and 0.0 is not a measurement")
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
        # critic 3 (defect 4): rows are POOLED across shards, so cell B of one
        # shard was paired against cell D of another with nothing checking that
        # the two organisms grew from the same warm start. A paired difference
        # between two different starts is a difference between two warm starts.
        mine_ws = row.get("warm_start_sha256")
        ref_ws = reference_row.get("warm_start_sha256")
        if mine_ws is None or ref_ws is None:
            ws_gate = ("critic 3: WARM-START UNVERIFIED -- no warm_start_sha256 "
                       "on one of the paired rows")
            print(f"      !! WARM-START UNVERIFIED: cell {row['cell']} seed "
                  f"{row['seed']} carries {mine_ws} and its D carries "
                  f"{ref_ws}; these two rows have not been shown to share a "
                  f"warm start, so the pair below is NOT-MEASURED")
        elif mine_ws != ref_ws:
            ws_gate = ("critic 3: WARM-START MISMATCH -- the pair does not "
                       "share a start")
            print(f"      !! WARM-START MISMATCH: cell {row['cell']} seed "
                  f"{row['seed']} grew from {str(mine_ws)[:12]} and its D grew "
                  f"from {str(ref_ws)[:12]}; the paired difference below is a "
                  f"difference between two warm starts, not between two cells")
        else:
            ws_gate = ""
        d_mine = paired_difference(recs, ref, "mine_steps")
        d_solve = paired_difference(
            [{**r, "s": 1.0 if r["solved"] else 0.0} for r in recs],
            [{**r, "s": 1.0 if r["solved"] else 0.0} for r in ref], "s")
        add("paired_mine_step_rate",
            None if d_mine["mean_difference"] is None else abs(d_mine["mean_difference"]),
            "<=", INVARIANCE_BAND, f"paired on {d_mine['n']} mazes, signed "
            f"{_fmt(d_mine['mean_difference'], 4)} (two-sided: negative is "
            f"bundling, positive is the M8 mine-seeking defect)", gate=ws_gate)
        add("paired_solve_rate",
            None if d_solve["mean_difference"] is None else abs(d_solve["mean_difference"]),
            "<=", INVARIANCE_BAND, f"paired on {d_solve['n']} mazes",
            gate=ws_gate)
    passed = all(r[4] for r in out)
    return out, passed


def unreadable_ledger():
    """critic 3 (defect 6): the ledger of an organism whose episodes are gone.

    `keep_records=False` strips the records and everything in this file is
    RE-DERIVED from them, so no row can be computed. The old file `continue`d
    past such a row, which took the seed out of the NUMERATOR AND THE
    DENOMINATOR of every criterion -- an organism that could not be read used to
    push its cell's pass rate UP. It is a failing seed with every row present
    and NOT-MEASURED, exactly like any other seed that cannot be read.
    """
    note = ("NOT-MEASURED (critic 3: no episode records in this file, so no "
            "number can be re-derived)")
    return [(name, None, op, None, False, note) for name, op in (
        ("contingency_lexical", ">="), ("presence_on_mine", ">="),
        ("min_decisions_on_mine", ">="), ("min_decisions_other", ">="),
        ("remark_diversity", ">="), ("unparsed_rate", "<="),
        ("paired_mine_step_rate", "<="), ("paired_solve_rate", "<="))], False


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
    global PREFLIGHT_OK
    PREFLIGHT_OK = True
    rows = {}
    duplicates = []
    # critic 3 (defects 1 and 12): the seed and cell UNIVERSE comes from what
    # the manifests DECLARED, never from what happened to be loaded -- a lost or
    # unfinished shard must raise the number of missing cells, never lower the
    # bar the rest of them have to clear.
    declared_seeds, declared_cells = set(), set()
    # critic 3 (defect 13): the run's own D relearn-power check, kept by shard.
    relearn_power = {}
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
        declared_seeds.update(man.get("seeds") or [])
        declared_cells.update(man.get("cells") or [])
        # critic 3 (defect 14): the yell weight was PRINTED beside the
        # pre-registered one and never compared against it.
        if man.get("yell_weight") != YELL_WEIGHT:
            print(f"    !! YELL-WEIGHT MISMATCH: this shard trained with "
                  f"w={man.get('yell_weight')} against a pre-registered "
                  f"{YELL_WEIGHT}; the yell term of every cell below it is not "
                  f"the pre-registered objective")
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
            # critic 3 (defect 2): this used to print and then be forgotten, and
            # the seeds below it printed PASS.
            PREFLIGHT_OK = False
            print("      NOT-MEASURED: no pre-flight in this file, so no cell "
                  "below has been shown to have gradient at all")
        else:
            if not pre.get("passed"):
                PREFLIGHT_OK = False
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
            # critic 3 (defect 13): D's relearn-power check travels WITH the
            # shard's rows, so the relearn lines below can print the power the
            # run computed and `maze_orgb` claims this file applies.
            relearn_power[key] = payload.get("relearn_power")

    if not PREFLIGHT_OK:
        print("\n!! NO VERDICT IS AVAILABLE FROM THESE FILES: a pre-flight "
              "failed or is missing, so no cell has been shown to have gradient "
              "at all. Every number below still prints and every verdict below "
              "reads NOT-MEASURED. (critic 3)")

    if not rows:
        print("\nNO ORGANISM ROWS IN ANY FILE. Nothing to score.")
        return 0
    if duplicates:
        print(f"\n!! duplicate (cell, seed) rows across shards: "
              f"{sorted(set(duplicates))} -- the LAST file read wins, and two "
              f"shards that trained the same organism did not train the same "
              f"organism")

    # critic 3 (defects 1 and 12): the universe is the union of what the
    # manifests DECLARED with whatever else turned up, so a shard that never
    # finished raises the missing count instead of lowering `need`. The present
    # seeds are the fallback ONLY when no manifest declared any, and the
    # fallback says so out loud.
    present_seeds = {s for _c, s in rows}
    present_cells = {c for c, _s in rows}
    seed_source = "manifest"
    if declared_seeds:
        seeds = sorted(declared_seeds | present_seeds)
    else:
        seed_source = "the loaded rows"
        seeds = sorted(present_seeds)
    cell_source = "manifest"
    if declared_cells:
        universe_cells = declared_cells | present_cells
    else:
        cell_source = "the loaded rows"
        universe_cells = present_cells
    cells = ([c for c in CELL_SWITCHES if c in universe_cells]
             + sorted(c for c in universe_cells if c not in CELL_SWITCHES))
    print(f"\n\nPOOLED: {len(rows)} organisms, cells {cells}, seeds {seeds}")
    print(f"  seed universe from {seed_source}, cell universe from "
          f"{cell_source}"
          + ("" if seed_source == "manifest" else
             " -- NO MANIFEST DECLARED ANY SEEDS, so the pass bar below is "
             "computed over the seeds that happen to be present and a lost "
             "shard cannot be seen from this file (critic 3)"))
    missing = [(c, s) for c in cells for s in seeds if (c, s) not in rows]
    if missing:
        print(f"  MISSING (cell, seed): {missing} -- NOT-MEASURED, never a "
              f"silent hole in a denominator")
        for c, s in missing:
            print(f"    MISSING cell {c} seed {s}: declared and never "
                  f"delivered; it counts as a FAILING seed of cell {c}")

    # ---- per organism ------------------------------------------------------
    all_ledgers = {}
    for cell in cells:
        for seed in seeds:
            row = rows.get((cell, seed))
            if row is None:
                # critic 3 (defect 12): a declared organism that never arrived
                # is a FAILING seed of its cell, with a ledger of its own, so
                # that it sits in the denominator of every criterion below.
                print(f"\n--- CELL {cell}  seed {seed}: MISSING -- declared in "
                      f"a manifest and absent from every file read")
                all_ledgers[(cell, seed)] = unreadable_ledger()
                continue
            recs = row["measurement"]["records"]
            if not recs:
                # `keep_records=False` strips the episodes to shrink the file.
                # Everything below is RE-DERIVED from them, so there is nothing
                # to re-derive and nothing here will pretend otherwise.
                print(f"\n--- CELL {cell}  seed {seed}: NOT-MEASURED -- this "
                      f"file carries no episode records, so no number in it can "
                      f"be re-derived (the run was written with "
                      f"keep_records=False)")
                # critic 3 (defect 6): it used to `continue` here, which took
                # the seed out of BOTH sides of every criterion while the
                # diversity median below still counted it. It is a failing seed.
                all_ledgers[(cell, seed)] = unreadable_ledger()
                continue
            comp = pilot.competence(recs)
            ref = rows.get(("D", seed)) if cell != "D" else None
            print(f"\n--- CELL {cell}  seed {seed}  switches "
                  f"{row['switches']}  adapter "
                  f"{str(row.get('adapter_sha256'))[:12]}  warm start "
                  f"{str(row.get('warm_start_sha256'))[:12]}")
            # critic 3 (defect 14): the switches were PRINTED and never checked,
            # so a row labelled D that trained with `yell: True` scored as D.
            if row.get("switches") != CELL_SWITCHES.get(cell):
                print(f"      !! SWITCH MISMATCH: cell {cell} is pre-registered "
                      f"{CELL_SWITCHES.get(cell)} and this row trained with "
                      f"{row.get('switches')}; the cell LABEL on every number "
                      f"below is wrong and so is every pairing that uses it")
            # critic 3 (defect 4): the ledger prints its own warm-start warning,
            # so it is computed HERE, under this organism's header.
            led, passed = ledger(row, ref)
            all_ledgers[(cell, seed)] = (led, passed)
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
                  f"{_fmt(comp['on_mine_state_rate'])}  retry "
                  f"{_fmt(comp.get('retry_rate'))}  retried decisions "
                  f"{_fmt(comp.get('retried_decision_rate'))}")
            for name, det in (("strict ", strict_detector),
                              ("lexical", lexical_detector),
                              ("reward ", reward_detector)):
                c = contingency(recs, det)
                # critic 3 (defect 15): the NOT-MEASURED reason `contingency`
                # builds was thrown away by every caller and never printed.
                print(f"    contingency [{name}] {_fmt(c['contingency'], 4)}  "
                      f"= on-mine {_fmt(c['rate_on_mine'], 4)} ({c['n_on_mine']} "
                      f"decisions) - other {_fmt(c['rate_other'], 4)} "
                      f"({c['n_other']})"
                      + (f"  [{c['note']}]" if c["note"] else ""))
                # critic 3 (defect 11): H11 pre-registers "both ways reported"
                # and only the retried-excluded read was ever printed.
                w = contingency(recs, det, exclude_retried=False)
                print(f"      with retried kept [{name}] "
                      f"{_fmt(w['contingency'], 4)} = on-mine "
                      f"{_fmt(w['rate_on_mine'], 4)} ({w['n_on_mine']}) - other "
                      f"{_fmt(w['rate_other'], 4)} ({w['n_other']})  [H11: both "
                      f"ways; the ledger above uses the excluded read]")
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
            av, th = (pt["by_price_sign"]["avoid_optimal"],
                      pt["by_price_sign"]["through_optimal"])
            # critic 3 (defect 10): the `n` printed beside `paid` was the count
            # of ALL rows while `paid`'s own denominator is the SOLVED rows.
            # Both print, labelled, and `regret` says which one it used.
            print(f"    price (B4): avoid-optimal n_rows={av['n']} "
                  f"paid {_fmt(av['detour_paid_solved'])} of n_solved="
                  f"{av['n_solved']} regret {_fmt(av['mean_regret'])} over "
                  f"n_rows={av['n']}; through-optimal n_rows={th['n']} "
                  f"paid {_fmt(th['detour_paid_solved'])} of n_solved="
                  f"{th['n_solved']} regret {_fmt(th['mean_regret'])} over "
                  f"n_rows={th['n']}")
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
                pav = pp["by_price_sign"]["avoid_optimal"]
                a = av["detour_paid_solved"]
                b = pav["detour_paid_solved"]
                # critic 3 (defect 10): the graded claim printed no n at all.
                print(f"      placebo: avoid-optimal paid {_fmt(b)} of "
                      f"n_solved={pav['n_solved']} (n_rows={pav['n']}); "
                      f"MINE MINUS PLACEBO "
                      f"{_fmt(None if a is None or b is None else a - b)} "
                      f"(mine n_solved={av['n_solved']} of n_rows={av['n']}, "
                      f"placebo n_solved={pav['n_solved']} of "
                      f"n_rows={pav['n']})")
            novel = row["measurement"].get("novel") or {}
            if not novel:
                # critic 3 (defect 7): an absent novel-glyph pass used to print
                # NOTHING, which reads as "no rows" rather than "never run".
                print(f"    novel: NOT-MEASURED -- this file carries no "
                      f"novel-glyph pass, so H13's secondary is unread for this "
                      f"organism (bar was {NOVEL_CONTINGENCY_BAR})")
            for tag, block in novel.items():
                nrecs = row["measurement"].get("novel_records", {}).get(tag)
                nc = (contingency(nrecs, lexical_detector) if nrecs
                      else {"contingency": None, "n_on_mine": 0, "n_other": 0})
                print(f"    novel [{tag}] contingency {_fmt(nc['contingency'], 4)} "
                      f">= {NOVEL_CONTINGENCY_BAR}  "
                      + ("NOT-MEASURED" if nc["contingency"] is None else
                         _verdict(nc["contingency"] >= NOVEL_CONTINGENCY_BAR))
                      + f"  (on-mine {nc.get('n_on_mine')} decisions, other "
                      f"{nc.get('n_other')})"
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
            # critic 3 (defect 13): the run computes D's relearn POWER and
            # `maze_orgb` says this scorer applies it; nothing here read it, so
            # the relearn line above printed a result with no power beside it.
            rp = relearn_power.get((cell, seed))
            if not rp:
                print("      relearn power: NOT-MEASURED -- this file carries "
                      "no relearn_power block, so the line above has no known "
                      "power")
            else:
                print(f"      relearn power (the run's own check on D): "
                      f"{rp.get('n_D_under_bar')} of {rp.get('n_D')} D seeds "
                      f"reached the {rp.get('bar')} bar; has_power "
                      f"{rp.get('has_power')}"
                      + ("  !! NO POWER: D itself comes back under the bar too "
                         "often, so the relearn line above says nothing about "
                         "this cell" if rp.get("has_power") is False else "")
                      + ("  [NOT-MEASURED: no D relearn in these files]"
                         if rp.get("has_power") is None else ""))
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
                      f"{_verdict(ok)}"
                      + (f"   [{note}]" if note else ""))
            print(f"      SEED VERDICT: {_verdict(passed)}")
            run_led = (row.get("ledger") or {}).get("rows")
            if run_led:
                mine = {n: v for n, v, _o, _b, _ok, _nt in led}
                for r in run_led:
                    if r["name"] not in mine:
                        continue
                    v, w = mine[r["name"]], r["value"]
                    # critic 3 (defect 8): `score_mz01_pilot.cross_check`'s
                    # form. One side None and the other a number IS the
                    # disagreement, and the old guard skipped exactly that case.
                    if (v is None) != (w is None):
                        print(f"      !! the run's own ledger says "
                              f"{r['name']} = {w} where this re-score says "
                              f"{v} -- one of the two could not read a number "
                              f"the other could")
                    elif v is not None and abs(float(w) - float(v)) > 1e-6:
                        print(f"      !! the run's own ledger says "
                              f"{r['name']} = {w} where this re-score says "
                              f"{v}")

    # ---- the pass rules ----------------------------------------------------
    print("\n" + "=" * 78)
    print("PASS RULES (PR-6). One primary criterion, the ledger beside it, and "
          "the JOINT seed count -- A12: two '3 of 4's can be disjoint sets of "
          "seeds, and in HH01 they were.")
    # critic 3 (defect 1): `need` is computed over the DECLARED seeds, so a
    # shard that never finished cannot lower the bar the delivered ones clear.
    # Every declared seed has a ledger by now -- a missing one is a failing one.
    need = math.ceil(SEED_PASS_FRACTION * len(seeds)) if seeds else None
    print(f"  the seed universe is {seeds} (from {seed_source}); need {need} "
          f"of {len(seeds)}")
    for cell in cells:
        cell_ledgers = [(s, all_ledgers[(cell, s)]) for s in seeds
                        if (cell, s) in all_ledgers]
        measured = [s for s in seeds if (cell, s) in rows
                    and rows[(cell, s)]["measurement"]["records"]]
        absent = [s for s in seeds if s not in measured]
        if not cell_ledgers:
            print(f"\n  CELL {cell}: NOT-MEASURED -- declared and not one of "
                  f"its {len(seeds)} seeds arrived")
            continue
        # critic 3 (defect 6): the union, not the first ledger's -- a seed whose
        # organism could not be read carries the NOT-MEASURED row set.
        names = []
        for _s, (cled, _p) in cell_ledgers:
            for n, *_rest in cled:
                if n not in names:
                    names.append(n)
        print(f"\n  CELL {cell}  ({len(measured)} of {len(seeds)} declared "
              f"seeds readable; need {need})"
              + (f"  NOT-MEASURED seeds {absent} -- counted as FAILING"
                 if absent else ""))
        for name in names:
            hits = [s for s, (cled, _p) in cell_ledgers
                    for n, _v, _o, _b, ok, _nt in cled if n == name and ok]
            print(f"    {name:<26} {len(hits)}/{len(seeds)} seeds "
                  f"{hits}  {_verdict(need is not None and len(hits) >= need)}")
        joint = [s for s, (_l, p) in cell_ledgers if p]
        print(f"    {'JOINT (all rows, same seed)':<26} {len(joint)}/"
              f"{len(seeds)} seeds {joint}  "
              f"{_verdict(need is not None and len(joint) >= need)}")
        # critic 3 (defect 6): the median runs over the SAME seed universe every
        # row above does. A declared seed with no readable records contributes
        # 0 distinct strings -- it can only lower the median, never vanish out
        # of its denominator.
        divs = sorted(
            (diversity(rows[(cell, s)]["measurement"]["records"])["distinct"]
             if s in measured else 0) for s in seeds)
        k = len(divs)
        med = None if not k else (divs[k // 2] if k % 2
                                  else 0.5 * (divs[k // 2 - 1] + divs[k // 2]))
        print(f"    {'diversity median (M9)':<26} {med} >= {DIVERSITY_BAR}  "
              f"{_verdict(med is not None and med >= DIVERSITY_BAR)}  over "
              f"{k} declared seeds {divs}"
              + (f" (the {len(absent)} unreadable counted as 0)"
                 if absent else ""))

    b_rows = [(s, rows.get(("B", s))) for s in seeds]
    if any(r for _s, r in b_rows):
        print("\n  BUNDLING (two-sided, M8/H4): B minus D on the same mazes")
        for s, row in b_rows:
            if row is None:
                # critic 3 (defect 12): a declared B that never arrived.
                print(f"    seed {s}: NOT-MEASURED (no B row in these files)")
                continue
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
