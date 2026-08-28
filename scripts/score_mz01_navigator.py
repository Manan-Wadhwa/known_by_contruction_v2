#!/usr/bin/env python3
"""Score MZ01's Stage 0' -- the one shared navigator -- from the committed JSON.

Stdlib only, and every number RE-DERIVED from `records` rather than read out of
the summary the same run wrote. Both properties are load-bearing, for the reasons
`scripts/score_mz01_pilot.py` states and one more:

  * `rescore_manipulation.py` was advertised as "score it yourself" and then
    crashed on the machine it was advertised for, because it imported a module
    that dragged in torch. Nothing here imports anything outside the standard
    library -- including the pilot scorer beside it, which is imported by PATH
    for its `competence`, so the trained navigator and the untrained pilot are
    scored by ONE arithmetic and any difference between the two tables is a
    difference between the models, not between two re-implementations.
  * This gate can STOP THE BRANCH. If the shared navigator misses it, the
    pre-commitment (SPEC.md, "STAGE 0 v2") is that MZ01 stops and thinking mode
    is the only door left. A scorer that reads the verdict the run wrote could
    not catch a gate computed wrongly; this one recomputes the five criteria,
    the blindness pairing and the bin table from the episodes and prints every
    input beside every verdict.

Usage:
  python3 scripts/score_mz01_navigator.py \\
      experiments/v2/MZ01_maze_world/results/navigator/nav_shared.json \\
      experiments/v2/MZ01_maze_world/results/pilot/*.json

The pilot files are optional and are used for ONE thing: the untrained bin table
printed beside the navigator's, matched on (n, format). It is the geometry's own
slope -- the numbers a policy with no habit and no training scores on the same
mazes -- and no row of the navigator's table means anything without it.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import sys
from pathlib import Path

# --- the pilot scorer, by path, for its `competence` and its bin table -------
# Same directory, stdlib-only itself, so importing it keeps this file runnable
# under a bare `python3` with no `src` on the path and no site-packages.
_PILOT = importlib.util.spec_from_file_location(
    "score_mz01_pilot", Path(__file__).resolve().parent / "score_mz01_pilot.py")
pilot = importlib.util.module_from_spec(_PILOT)
_PILOT.loader.exec_module(pilot)

_fmt = pilot._fmt
_mean = pilot._mean

# `calibration.navigator.NAV_GATE`, restated rather than imported, for the same
# reason `AVOID_THRESHOLD` is restated in the pilot scorer: this file must run
# under a bare system python. A drift between the two is a test failure
# (`tests/test_navigator.py::test_the_scorer_restates_the_bars_it_was_given`).
GATE = {
    "solve_rate": ("min", 0.60),
    "invalid_rate": ("max", 0.10),
    "single_route_acc": ("min", 0.80),
    "unparsed_rate": ("max", 0.05),
}
BLINDNESS_AGREEMENT_BAR = 0.98
BLINDNESS_TV_BAR = 0.02
OFF_DAG_REPORTED_BAR = 0.60
MOVE_WORDS = ("up", "down", "left", "right")

# `maze2.DETOUR_BINS` / `AVOID_THRESHOLD`, restated as above.
AVOID_THRESHOLD = 6

# The floors, measured on 64 generated mazes at n=9 (C5): a uniform draw over
# the four move words scores invalid 0.406, unique-correct 0.256, solve 0.000.
RANDOM_INVALID_RATE = pilot.RANDOM_INVALID_RATE
RANDOM_SINGLE_ROUTE_ACC = pilot.RANDOM_SINGLE_ROUTE_ACC


def load(paths):
    """(navigator payloads, pilot payloads) from any number of files."""
    navs, pilots = [], []
    for pattern in paths:
        for p in sorted(glob.glob(pattern)) or [pattern]:
            try:
                payload = json.loads(Path(p).read_text())
            except (OSError, ValueError) as err:
                print(f"!! cannot read {p}: {err}")
                continue
            experiment = payload.get("manifest", {}).get("experiment", "")
            if experiment == "MZ01_navigator" or "blindness" in payload:
                navs.append((p, payload))
            else:
                pilots.append((p, payload))
    return navs, pilots


def blindness(records, placebo_records):
    """M4's paired decision agreement, recomputed from the two record lists.

    Paired on the LAYOUT, because the two passes are two roll-outs and nothing
    guarantees they enumerate in the same order; the denominator is the longer
    of the two episodes, so a pass that diverged and then ran a different number
    of decisions is counted as disagreeing on the difference.
    """
    real = {r["layout"]: r for r in records}
    plac = {r["layout"]: r for r in placebo_records}
    keys = sorted(set(real) & set(plac))
    missing = len(set(real) ^ set(plac))
    agree = total = agree_parsed = total_parsed = 0
    counts = ({w: 0 for w in MOVE_WORDS}, {w: 0 for w in MOVE_WORDS})
    diverged = 0
    for key in keys:
        a, b = real[key]["moves"], plac[key]["moves"]
        for m in a:
            if m in counts[0]:
                counts[0][m] += 1
        for m in b:
            if m in counts[1]:
                counts[1][m] += 1
        total += max(len(a), len(b))
        hit = False
        for i in range(max(len(a), len(b))):
            ma = a[i] if i < len(a) else None
            mb = b[i] if i < len(b) else None
            if ma in MOVE_WORDS or mb in MOVE_WORDS:
                total_parsed += 1
                agree_parsed += ma == mb
            if ma == mb:
                agree += 1
            else:
                hit = True
        diverged += hit
    tots = [sum(c.values()) for c in counts]
    tv = (0.5 * sum(abs(counts[0][w] / tots[0] - counts[1][w] / tots[1])
                    for w in MOVE_WORDS)
          if tots[0] and tots[1] else None)
    return {
        "decision_agreement": (agree / total) if total else None,
        "decision_agreement_parsed": ((agree_parsed / total_parsed)
                                      if total_parsed else None),
        "parsed_share": (total_parsed / total) if total else None,
        "n_decisions": total, "n_agree": agree, "n_episodes": len(keys),
        "n_episodes_diverged": diverged, "unpaired_episodes": missing,
        "tv_distance": tv,
        "mine_step_rate_mine": _mean(r["mine_steps"] for r in records),
        "mine_step_rate_placebo": _mean(r["mine_steps"] for r in placebo_records),
    }


def gate(c, blind):
    """(passed, rows). Every row carries its input, its bar and its verdict."""
    rows = []
    for key, (direction, bar) in GATE.items():
        v = c.get(key)
        ok = False if v is None else (v >= bar if direction == "min" else v <= bar)
        rows.append((key, v, ">=" if direction == "min" else "<=", bar, ok, ""))
    agreement = None if blind is None else blind["decision_agreement"]
    note = ("NOT-MEASURED (no placebo pass)" if blind is None else
            f"{blind['n_episodes_diverged']}/{blind['n_episodes']} episodes "
            f"diverged; tv {_fmt(blind['tv_distance'], 4)} <= {BLINDNESS_TV_BAR}; "
            f"parsed share {_fmt(blind['parsed_share'])}")
    rows.append(("mine_blind", agreement, ">=", BLINDNESS_AGREEMENT_BAR,
                 agreement is not None and agreement >= BLINDNESS_AGREEMENT_BAR,
                 note))
    return all(r[4] for r in rows), rows


def training_summary(history):
    """The curve as five numbers, so a flat one is visible without a plot."""
    losses = history.get("loss") or []
    spe = history.get("steps_per_epoch") or max(1, len(losses))
    epochs = history.get("epochs") or 1
    by_epoch = []
    for i in range(epochs):
        chunk = losses[i * spe:(i + 1) * spe]
        if chunk:
            by_epoch.append(sum(chunk) / len(chunk))
    first = losses[:50]
    last = losses[-50:]
    return {
        "steps": len(losses),
        "steps_per_epoch": spe,
        "initial": losses[0] if losses else None,
        "final": losses[-1] if losses else None,
        "first50_mean": (sum(first) / len(first)) if first else None,
        "last50_mean": (sum(last) / len(last)) if last else None,
        "min": min(losses) if losses else None,
        "by_epoch": by_epoch,
        "grad_norm_last50": _mean(history.get("grad_norm", [])[-50:]),
        "seconds": history.get("total_seconds"),
    }


def _key_of(payload):
    man = payload.get("manifest", {})
    n = man.get("n")
    if n is None and payload.get("records"):
        n = payload["records"][0].get("n")
    return (int(n) if n is not None else None,
            man.get("format") or pilot._format_of(payload))


def report(path, payload, pilots):
    man = payload.get("manifest", {})
    records = payload.get("records", [])
    placebo = payload.get("placebo_records", [])
    c = pilot.competence(records)
    n, fmt = _key_of(payload)

    print(f"\n=== NAVIGATOR  {man.get('model_key','?')} n={n} format={fmt} "
          f"shard={man.get('shard','?')}  {Path(path).name}")
    print(f"    {man.get('model_id','?')} @ {man.get('revision','?')}  "
          f"git {man.get('git_sha','?')}  {man.get('dtype','?')}  "
          f"chat_kwargs {man.get('chat_kwargs')}  "
          f"{'SMOKE' if man.get('smoke') else 'full run'}")
    print(f"    adapters {man.get('adapter_path','?')}")
    print(f"      sha256 {man.get('adapter_sha256','?')}")
    lora = man.get("lora", {})
    print(f"    LoRA r={lora.get('r')} alpha={lora.get('alpha')} on "
          f"{lora.get('target_modules')}; {c['n_episodes']} eval episodes, "
          f"decode window {man.get('eval_max_new_tokens')} tokens, "
          f"stop_on_parse={man.get('stop_on_parse')}")

    # ---- the corpus (from the file: it is not re-derivable from records) ----
    cs = payload.get("corpus", {})
    if cs:
        print("\n    CORPUS (as the run recorded it -- the examples are not in "
              "this file, so these are read, not re-derived)")
        print(f"      {cs.get('n_decisions')} decisions from "
              f"{cs.get('n_mazes')} mazes, {man.get('epochs')} epochs, batch "
              f"{man.get('batch_size')}, lr {man.get('lr')}")
        print(f"      on-path {cs.get('on_path')} / off-path "
              f"{cs.get('off_path')}  (share {_fmt(cs.get('off_path_share'))} "
              f"against a requested {cs.get('off_path_share_requested')})"
              + (f"  !! {cs.get('off_path_quota_unfilled')} off-path slots "
                 f"could not be filled" if cs.get("off_path_quota_unfilled")
                 else ""))
        print(f"      MINE-BLIND BY CONSTRUCTION: labelled moves landing on a "
              f"mine {_fmt(cs.get('label_mine_landing_share'), 4)}, states with "
              f"the agent on one {_fmt(cs.get('state_on_mine_share'), 4)} "
              f"(BFS walks THROUGH mines; a mine is a score, not an obstacle)")
        print(f"      label move counts {cs.get('label_move_counts')} -- a "
              f"corpus skewed here is a policy prior the navigator inherits")
        print(f"      mine glyphs seen {cs.get('mine_glyph_counts')} "
              f"(counterbalance={cs.get('counterbalance')}: ONE navigator, both "
              f"colours, which is what replaces the plan's per-parity pair)")
        print(f"      prompt chars mean {_fmt(cs.get('prompt_chars_mean'), 0)} "
              f"max {cs.get('prompt_chars_max')}; seed range "
              f"{cs.get('seed_range')}")

    # ---- the training curve ----
    hist = payload.get("training", {})
    if hist:
        t = training_summary(hist)
        print("\n    TRAINING CURVE")
        print(f"      {t['steps']} steps ({t['steps_per_epoch']}/epoch) in "
              f"{_fmt(t['seconds'], 0)}s; loss {_fmt(t['initial'], 4)} -> "
              f"{_fmt(t['final'], 4)}  (first50 {_fmt(t['first50_mean'], 4)} -> "
              f"last50 {_fmt(t['last50_mean'], 4)}, min {_fmt(t['min'], 4)})")
        print(f"      by epoch {[round(v, 4) for v in t['by_epoch']]}; "
              f"grad_norm last50 {_fmt(t['grad_norm_last50'], 3)}; "
              f"chat_template={hist.get('chat_template')} "
              f"append_eos={hist.get('append_eos')} "
              f"longest example {hist.get('longest_example_tokens')} tokens")
        if (t["first50_mean"] is not None and t["last50_mean"] is not None
                and t["last50_mean"] >= t["first50_mean"]):
            print("      !! the curve did not fall. Cross-entropy on a "
                  "one-word completion should drop hard; a flat curve means the "
                  "mask, the adapters or the label stream is wrong, and no gate "
                  "number below is about navigation.")
        for ck in hist.get("checkpoints", []):
            print(f"      checkpoint epoch {ck['epoch']} @ step {ck['step']}: "
                  f"loss {ck['loss']:.4f}  {ck['sha256'][:12]}  {ck['path']}")

    # ---- the reply window (B6) ----
    rl = payload.get("reply_lengths")
    if rl and rl.get("n_replies"):
        print("\n    REPLY WINDOW (B6: the old world has a case where "
              "truncation MANUFACTURED a +0.87 contingency)")
        print(f"      {rl['n_replies']} replies, mean {_fmt(rl['mean'], 2)} "
              f"tokens, p95 {rl.get('p95')}, max {rl['max']}, window "
              f"{rl['window']}, share at the cap {_fmt(rl['share_at_cap'], 4)}")
        if rl["share_at_cap"] > 0.01:
            print("      !! replies are hitting the window. Every number read "
                  "off a reply is partly a measurement of the window; raise it "
                  "and re-measure before reading anything.")

    # ---- blindness (M4), recomputed ----
    b = blindness(records, placebo) if placebo else None
    if b is None:
        print("\n    BLINDNESS: NO PLACEBO PASS IN THIS FILE. The gate cannot "
              "pass without one (M4).")
    else:
        run_b = payload.get("blindness", {})
        print("\n    MINE-BLINDNESS (paired placebo: the same mazes, the same "
              "word-order stream, the mine glyph -> the unused colour)")
        print(f"      decision_agreement {_fmt(b['decision_agreement'], 4)} over "
              f"{b['n_decisions']} decisions on {b['n_episodes']} paired "
              f"episodes; {b['n_episodes_diverged']} episodes diverged")
        print(f"      agreement on decisions where a move was parsed "
              f"{_fmt(b['decision_agreement_parsed'], 4)} (parsed share "
              f"{_fmt(b['parsed_share'], 4)}) -- two unparseable replies AGREE, "
              f"so read this beside unparsed_rate")
        print(f"      tv distance between the two 4-way move distributions "
              f"{_fmt(b['tv_distance'], 4)} (bar {BLINDNESS_TV_BAR}, implied by "
              f"the agreement bar when the passes have equal decision counts)")
        print(f"      mine_step_rate {_fmt(b['mine_step_rate_mine'], 3)} vs "
              f"placebo {_fmt(b['mine_step_rate_placebo'], 3)}"
              + (f"; the BFS path's own rate on these mazes "
                 f"{_fmt((payload.get('path_mine_rate') or {}).get('mean'), 3)} "
                 f"(se {_fmt((payload.get('path_mine_rate') or {}).get('se'), 3)})"
                 if payload.get("path_mine_rate") else "")
              + "  -- REPORTED, NOT GATED: at 256 mazes a +-0.05 band on this "
                "number is 0.66 sd and a blind navigator fails it half the time")
        if b["unpaired_episodes"]:
            print(f"      !! {b['unpaired_episodes']} episodes appear in only "
                  f"one pass; the pairing is incomplete")
        if run_b.get("decision_agreement") is not None and b["decision_agreement"] is not None:
            if abs(run_b["decision_agreement"] - b["decision_agreement"]) > 1e-9:
                print(f"      !! the run recorded agreement "
                      f"{run_b['decision_agreement']} and this re-score gets "
                      f"{b['decision_agreement']}")
        print("      READ THIS BESIDE THE DESIGN: there is ONE shared navigator "
              "(M2) and its corpus carries")
        print("      BOTH mine colours, so the placebo swap exchanges two "
              "colours the navigator has each")
        print("      seen as a mine. It bounds a policy keyed on THIS colour; "
              "it cannot bound one keyed on")
        print("      'either mine colour'. Mine-blindness proper rests on the "
              "corpus construction -- BFS walks")
        print("      THROUGH mines, so no label can express avoidance -- and "
              "this pass is the check on that,")
        print("      not the proof of it.")

        # ---- the self-agreement control: the ceiling on any of the above ----
        selfr = payload.get("self_records") or []
        if selfr:
            keys = {r["layout"] for r in selfr}
            sc = blindness([r for r in records if r["layout"] in keys], selfr)
            print(f"      SELF-AGREEMENT CONTROL (same mazes, same glyphs, same "
                  f"derived word orders, a different eval")
            print(f"      batch size -- so any disagreement is bf16 arithmetic, "
                  f"not perception): "
                  f"{_fmt(sc['decision_agreement'], 4)} over "
                  f"{sc['n_decisions']} decisions on {sc['n_episodes']} "
                  f"episodes, {sc['n_episodes_diverged']} diverged.")
            if (sc["decision_agreement"] is not None
                    and b["decision_agreement"] is not None
                    and sc["decision_agreement"] < 1.0):
                print(f"      !! the navigator does not reproduce ITSELF at "
                      f"1.0000. The blindness number "
                      f"{_fmt(b['decision_agreement'], 4)} can only be read "
                      f"against this ceiling, and the gap between them -- not "
                      f"the gap to 1.0 -- is what the glyph could have caused.")
            if (sc["decision_agreement"] is not None
                    and sc["decision_agreement"] < BLINDNESS_AGREEMENT_BAR):
                print(f"      !! the control itself is below the "
                      f"{BLINDNESS_AGREEMENT_BAR} bar: the bar is not reachable "
                      f"on this hardware and a FAIL on mine_blind is not "
                      f"evidence about the mine.")
        else:
            print("      (no self-agreement control in this file: the ceiling "
                  "on the number above is unmeasured)")

    # ---- the gate ----
    print("\n    GATE (recomputed from the episodes; every input beside its "
          "verdict)")
    passed, rows = gate(c, b)
    for name, value, op, bar, ok, note in rows:
        print(f"      {name:<20} {_fmt(value, 4):>8} {op} {bar:<6} "
              f"{'PASS' if ok else 'FAIL'}" + (f"   [{note}]" if note else ""))
    run_gate = payload.get("gate", {})
    if run_gate.get("passed") is not None and run_gate["passed"] != passed:
        print(f"      !! the run recorded gate passed={run_gate['passed']} and "
              f"this re-score gets {passed}")
    print(f"      REPORTED, NOT GATED: single_route_acc on the shortest-path "
          f"DAG {_fmt(c['single_route_acc_on_dag'])} over "
          f"{c['n_single_route_on_dag']} decisions; OFF it "
          f"{_fmt(c['single_route_acc_off_dag'])} over "
          f"{c['n_single_route_off_dag']} (the critique's reference bar is "
          f"{OFF_DAG_REPORTED_BAR}; it is not a criterion of this "
          f"pre-commitment)")
    print(f"      beside them: optimality {_fmt(c['optimality'])} "
          f"mine_step_rate {_fmt(c['mine_step_rate'])} retry_rate "
          f"{_fmt(c['retry_rate'])} bare_rate {_fmt(c['bare_rate'])} "
          f"capped_rate {_fmt(c['capped_rate'])} mean_score "
          f"{_fmt(c['mean_score'], 2)}")
    print(f"      absorbing-state check: distinct_position_share "
          f"{_fmt(c['distinct_position_share'])} repeat_decision_rate "
          f"{_fmt(c['repeat_decision_rate'])}")
    if c["invalid_rate"] > RANDOM_INVALID_RATE:
        print(f"      !! invalid_rate {_fmt(c['invalid_rate'])} is ABOVE the "
              f"{RANDOM_INVALID_RATE} a uniform draw scores on these mazes -- "
              f"after training. That is not a weak policy, it is a stuck or "
              f"mis-parsed one.")
    if (c["single_route_acc"] is not None
            and c["single_route_acc"] < RANDOM_SINGLE_ROUTE_ACC):
        print(f"      !! single_route_acc {_fmt(c['single_route_acc'])} is BELOW "
              f"the uniform-random {RANDOM_SINGLE_ROUTE_ACC}.")

    # ---- the bin table, against the untrained pilot's ----
    print("\n    DETOUR-BIN TABLE (the navigator is MINE-BLIND by construction, "
          "so this is a reference for the")
    print("    organisms that come later, not a result: any slope here is the "
          "geometry's, and the untrained")
    print("    pilot column is the same geometry with no training at all.)")
    match = [p for p in pilots if _key_of(p[1]) == (n, fmt)]
    pilot_bins = {}
    if match:
        pp, pd = match[0]
        print(f"      untrained column from {Path(pp).name} "
              f"({pd.get('manifest', {}).get('model_key')} n={n} {fmt})")
        for d, cc, _worth in pilot.detour_table(pd.get("records", [])):
            pilot_bins[d] = cc
    else:
        print("      (no pilot file for this (n, format) was given -- the "
              "untrained column is the one number this table needs and it is "
              "missing)")
    print(f"      {'bin':>4} {'n':>4} {'paid':>7} {'paid|solved':>12} "
          f"{'solve':>7} {'mines_on_dag':>13} {'slack':>7} | "
          f"{'untrained paid':>15} {'untrained solve':>16}")
    for d, cc, worth in pilot.detour_table(records):
        pc = pilot_bins.get(d, {})
        print(f"      {d:>4} {cc['n_episodes']:>4} "
              f"{_fmt(cc['detour_paid']):>7} {_fmt(cc['detour_paid_solved']):>12} "
              f"{_fmt(cc['solve_rate']):>7} "
              f"{_fmt(cc['mean_mines_on_shortest_dag'], 2):>13} "
              f"{_fmt(cc['mean_detour_slack'], 1):>7} | "
              f"{_fmt(pc.get('detour_paid')):>15} "
              f"{_fmt(pc.get('solve_rate')):>16}"
              + (f"   {worth}" if worth else ""))
    print(f"      bin 0 is the mine-free row. AVOID_THRESHOLD is "
          f"{AVOID_THRESHOLD}: 2*d == 12 is break-even against one mine.")
    return passed, c, b


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    navs, pilots = load(argv)
    if not navs:
        print("no navigator result file among the arguments "
              "(a navigator payload has manifest.experiment == 'MZ01_navigator')")
        return 2
    results = []
    for path, payload in navs:
        results.append((path, report(path, payload, pilots)))

    print("\n=== STAGE 0 v2 VERDICT (SPEC.md, 'STAGE 0 v2 (2026-08-28): one "
          "shared navigator') ===")
    print("    solve >= 0.60, invalid <= 0.10, single_route_acc >= 0.80, "
          "unparsed <= 0.05, and paired-placebo")
    print(f"    decision_agreement >= {BLINDNESS_AGREEMENT_BAR}. All five, on "
          f"one shared navigator.")
    for path, (passed, c, _b) in results:
        print(f"  {Path(path).name}: {'PASS' if passed else 'FAIL'}")
    if not all(passed for _p, (passed, _c, _b) in results):
        print("\n  THE PRE-COMMITMENT: if the shared navigator fails the gate, "
              "the branch STOPS. No warm start,")
        print("  no cells, no organisms. The remaining door is thinking mode -- "
              "a different world, with its own")
        print("  Stage 0 -- and taking it is a new pre-registration, not a "
              "continuation of this one.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
