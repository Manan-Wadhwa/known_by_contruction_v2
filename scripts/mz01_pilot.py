"""Sandbox-side driver for MZ01 Stage 0 -- the pilot that gates the whole world.

Run as a detached OS process on the GPU box; it writes a status file the puller
can tail. Same shape as `scripts/nar03_driver.py`, with the differences MZ01's
SPEC forces:

  1. THIS RUN CAN STOP THE EXPERIMENT. Stage 0 is not a warm-up: "the world size
     is the largest passing size", and "if none passes at 7, stop and report; do
     not build organisms". So the driver's job is to produce numbers a human
     reads before any organism exists, and it deliberately does NOT decide
     anything itself -- `scripts/score_mz01_pilot.py` applies the rule, from the
     committed JSON, with the numbers printed beside the verdict.

  2. THE WALL, GOAL AND ON-MINE GLYPHS ARE MEASURED, NOT CHOSEN.
     `choose_role_glyphs` ranks the eligible glyphs by |valence| on the
     UNTRAINED model and takes the three most neutral. That measurement happens
     here, before anything is trained, and the full ranking goes in the result
     file -- so the roles cannot have been picked to produce a result, which is
     the defect E14's hard-coded placebo has and
     `instruments.choose_placebo_pair` was written to fix. The third role is
     maze2 D17: the agent standing on a mine is drawn with a glyph of its own,
     so organism B's cue is a visible fact about the current state rather than
     an inference about the model's own next token (SPEC A6).

  3. THE MINE COLOUR IS COUNTERBALANCED ACROSS THE PILOT. The mazes are split by
     seed parity and each half is rolled out ONCE, under the assignment
     `maze.role_glyphs` gives its parity -- the same operation the rest of the
     repo does, so E1c-2's colour prior cancels in the mean over the set. (An
     earlier draft of this docstring said each half was rolled out twice, under
     both assignments. It was not, and it is not: that would be a within-maze
     counterbalance at twice the episode count, which is a different and more
     expensive design than the one the SPEC pre-committed to.)

  5. THE UNTRAINED BIN TABLE IS THE GEOMETRY'S OWN SLOPE. It is NOT SPEC A2's
     placebo control, which an earlier draft of this docstring called it.
     A2 requires two separate things and this driver produces one of them by
     default: (i) the placebo control set -- the SAME mazes rendered with the
     mine glyph replaced by the unused colour (`maze2.placebo_glyphs`), which
     holds geometry, length, mine count and wall density EXACTLY and changes
     only the colour; and (ii) the untrained bin table, which is the slope the
     geometry produces on a policy with no habit to express. The second is
     always written; the first runs under `MZ01_PLACEBO=1` and is written
     beside it as `placebo_aggregates`. Neither substitutes for the other, and
     the graded claim about any ORGANISM is the mine-minus-placebo difference,
     bin by bin.

  6. THE COVARIATE MATCHING IS NOT WHAT THE 0.500s SAY. DEC-2 holds
     `shortest_len`, `n_mines`, `open_ratio` and `branch_points` to CONSTANTS,
     so their bin-8-vs-bin-2 AUC is 0.500 because every comparison is a tie,
     not because a policy could not separate the bins. Two covariates do move
     and neither is in that list: `detour_slack` (= shortest_len - detour_cost,
     a deterministic function of the bin) and `mines_on_shortest_dag`
     (AUC 0.81 at n=9, and countable in the picture). Both are printed here.

  7. THE RENDER FORMAT IS A PARAMETER AND EACH VALUE IS A DIFFERENT WORLD.
     The emoji pilot of 2026-08-28 failed the gate at every size on both models,
     BELOW the uniform-random floor, so SPEC A8's fallback ladder was taken at
     rung (ii): `MZ01_FORMAT=ascii` renders the same mazes as a numbered text
     grid with a coordinate line (maze2 D29). A8 says in as many words that
     each rung is a different world and needs its own Stage 0, so the format is
     in the manifest AND in the result filename
     (`{model}_{n}_{fmt}.json`), every episode record carries `fmt`, and the
     scorer keys on it. Two formats are never pooled.

     Under `ascii` the wall and goal are FIXED characters (`#`, `G`) rather than
     measured role glyphs: there is no bank of interchangeable ASCII marks to
     rank, so the valence machinery of C2/C11 applies to `on_mine` alone, which
     is the one role that still carries a glyph. That is recorded in the
     manifest as `measured_roles` and `fixed_roles` rather than left to be
     inferred from which keys have a valence beside them.

  4. A SMOKE PHASE THAT GATES THE REAL RUN. `MZ01_SMOKE_ONLY=1` runs one size at
     8 mazes and asserts the record contract, so a missing key or a broken chat
     template costs seconds rather than the whole pilot.

Environment:
  MZ01_MODEL       "4B" or "14B" (default 4B)
  MZ01_FORMAT      "emoji" or "ascii" (default emoji). The render format, which
                   is a different WORLD and gets its own Stage 0; it lands in
                   the manifest and in the result filename.
  MZ01_SIZES       comma-separated grid sizes (default "7,9,11")
  MZ01_N           eval mazes per size (default 256, the SPEC's number)
  MZ01_REVISION    REQUIRED. The checkpoint revision, pinned in the manifest --
                   a pilot that gates a whole experiment must say which weights
                   it gated it with, and "main" moves.
  MZ01_BATCH       decode batch size (default 32)
  MZ01_SMOKE_ONLY  "1" to run the smoke and stop.
  MZ01_PLACEBO     "1" to roll the same mazes a second time with the mine glyph
                   replaced by the unused colour (SPEC A2's control set). Off by
                   default: it doubles the GPU time and Stage 0's job is the
                   gate, but it is the only pass that separates "responded to
                   the mine" from "responded to the geometry".
  MZ01_VALENCE_BAR the |valence| a MEASURED role glyph may carry
                   (`maze2.ROLE_VALENCE_BAR`, default 2.0). Raising it is a
                   decision, so it is an environment variable that lands in the
                   manifest rather than a constant somebody edits.
"""

import json
import os
import sys
import time
import traceback

REPO = "/marimo/repo"
sys.path.insert(0, f"{REPO}/src")

MODELS = {"4B": "Qwen/Qwen3-4B-Instruct-2507", "14B": "Qwen/Qwen3-14B"}

MODEL_KEY = os.environ.get("MZ01_MODEL", "4B").strip()
FORMAT = os.environ.get("MZ01_FORMAT", "emoji").strip()
SIZES = [int(s) for s in os.environ.get("MZ01_SIZES", "7,9,11").split(",") if s.strip()]
N_MAZES = int(os.environ.get("MZ01_N", "256"))
REVISION = os.environ.get("MZ01_REVISION", "").strip()
BATCH = int(os.environ.get("MZ01_BATCH", "32"))
SMOKE_ONLY = os.environ.get("MZ01_SMOKE_ONLY", "") == "1"
PLACEBO = os.environ.get("MZ01_PLACEBO", "") == "1"
VALENCE_BAR = os.environ.get("MZ01_VALENCE_BAR", "").strip()

# The format is in the status file name for the reason it is in the result
# file name: two formats are two worlds (SPEC A8), and a run of one must not
# append its lines to the log of the other.
STATUS = f"{REPO}/mz01_{MODEL_KEY}_{FORMAT}_status.txt"
OUT = f"{REPO}/experiments/v2/MZ01_maze_world/results/pilot"

# The keys every downstream reader (the scorer, the second pass, a re-score in a
# year) indexes into. Asserted in the smoke, so a record that quietly loses one
# costs seconds instead of a pilot.
RECORD_KEYS = ("index", "seed", "n", "layout", "labels", "moves", "landings",
               # maze2 D29: which render format this episode was rolled out in.
               # Without it a result file cannot be re-scored (the two formats
               # are two worlds) and `build_maze_corpus` would rebuild an ASCII
               # decision's prompt in emoji -- an example claiming a prompt the
               # base model never saw, which is C6's defect again.
               "fmt",
               "replies", "correct", "invalid", "retries", "unparsed",
               "mine_steps", "solved", "steps", "capped", "score",
               "avoided_all_mines", "mine_glyph",
               # maze2 D13: one SFT example per DECISION is organism B's corpus,
               # and its prompt is (layout, position, word order, glyph table).
               # Without these four a result file cannot rebuild the corpus the
               # base model actually produced.
               "positions", "orders", "bare", "bare_moves", "glyphs",
               # AMENDED after critic 2: `retried` says which decisions were
               # taken under the three-turn retry prompt (they are not the
               # base model's move in the state the corpus rebuilds), and
               # `correct_at` says which decision each `correct` entry came
               # from, without which SPEC A7's on/off-DAG split of
               # `single_route_acc` is not computable from the file.
               "retried", "correct_at",
               # maze2 D17: which decisions were taken while the render showed
               # the agent standing on a mine. That is B's cue and the
               # population D18's corpus builder draws from; without it neither
               # the contingency nor the corpus can be rebuilt from the file.
               "on_mine", "on_mine_states")


def say(m):
    with open(STATUS, "a") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")
    print(m, flush=True)


def run_size(model, tok, n, count, glyph_pair, torch, maze2, seed_range, seen,
             placebo=False):
    """Generate, roll out (both glyph parities) and aggregate one size.

    Returns (records, aggregates, generator stats, placebo aggregates or None).
    The two parities are two roll-outs over disjoint halves of the SAME maze
    set, so the episode count is the SPEC's and the colour prior cancels rather
    than loading.

    `placebo` adds SPEC A2's control pass: the same mazes, the same halves, the
    same glyph parities, with `maze2.placebo_glyphs` -- which swaps the mine
    tile's colour for the unused one and CARRIES `on_mine` ACROSS UNCHANGED,
    because the mines are still mines and are still scored. Geometry, length,
    mine count and wall density are held exactly; only the colour moves.
    """
    wall, goal, on_mine = glyph_pair
    gen = torch.Generator().manual_seed(1000 + n)
    t0 = time.perf_counter()
    mazes, stats = maze2.generate_set(count, n=n, generator=gen,
                                      seed_range=seed_range, seen=seen)
    say(f"  n={n}: {len(mazes)} mazes in {time.perf_counter()-t0:.1f}s, "
        f"rejection {stats['rejection_rate']:.3f}, achieved {stats['achieved']}")

    def pass_over(make_glyphs, tag):
        out = []
        for parity in (0, 1):
            half = [m for m in mazes if m.seed % 2 == parity]
            if not half:
                continue
            glyphs = make_glyphs(parity)
            t1 = time.perf_counter()
            recs, _agg = maze2.rollout(model, tok, half, retry=True,
                                       glyphs=glyphs, generator=gen,
                                       batch_size=BATCH, fmt=FORMAT)
            for r in recs:
                r["mine_glyph"] = glyphs.mine
                r["pass"] = tag
            out.extend(recs)
            say(f"  n={n} {tag} parity {parity} ({glyphs.mine}): {len(half)} "
                f"episodes in {(time.perf_counter()-t1)/60:.1f} min")
        out.sort(key=lambda r: (r["n"], r["seed"]))
        for i, r in enumerate(out):
            r["index"] = i
        return out

    def real(parity):
        return maze2.glyphs_for_seed(parity, wall=wall, goal=goal,
                                     on_mine=on_mine)

    records = pass_over(real, "mine")
    placebo_agg = None
    if placebo:
        placebo_recs = pass_over(lambda p: maze2.placebo_glyphs(real(p)),
                                 "placebo")
        placebo_agg = maze2.aggregate(placebo_recs)
    return records, maze2.aggregate(records), stats, placebo_agg


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from calibration import maze2
    from calibration.chat import chat_kwargs
    from calibration.instruments import glyph_valence_table
    from calibration.rl import _EVAL_SEED_RANGE
    from calibration.runner import git_sha

    maze2.check_format(FORMAT)
    if MODEL_KEY not in MODELS:
        raise SystemExit(f"MZ01_MODEL must be one of {sorted(MODELS)}, got {MODEL_KEY!r}")
    if not REVISION:
        raise SystemExit(
            "MZ01_REVISION is required: Stage 0 gates every organism in this "
            "world, so the result file has to name the exact weights it gated.")
    model_id = MODELS[MODEL_KEY]

    say(f"MZ01 pilot: {model_id} @ {REVISION}; format {FORMAT}; sizes {SIZES}; "
        f"{N_MAZES} mazes/size; smoke_only={SMOKE_ONLY}")
    tok = AutoTokenizer.from_pretrained(model_id, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=REVISION, dtype=torch.bfloat16).to("cuda")
    model.eval()
    # AMENDED after critic 2. Qwen3-14B -- the checkpoint this gate is DEFINED
    # at -- ships a hybrid chat template whose default is thinking mode, so
    # `add_generation_prompt=True` opens a `<think>` block and the first
    # generated token is a reasoning trace. This driver GREEDY-DECODES a
    # six-token window and parses a move word out of it, so a thinking-mode
    # prompt gives `unparsed_rate` near 1.0 at every size and the SPEC's stop
    # rule ("if none passes at 7, stop and report") would kill the world on a
    # template flag. `calibration.chat.apply_chat` closes it; this says out loud
    # which position the numbers were read at, and it goes in the manifest.
    chat_kw = chat_kwargs(tok)
    say(f"model resident; chat_kwargs {chat_kw or '{} (plain template)'}")

    # ---- the wall, goal and on-mine glyphs, measured on the untrained model ----
    # AMENDED after critic 2. The first smoke on the GPU chose on_mine = orange
    # at valence +3.09 -- the most POSITIVE glyph left -- because
    # `instruments.GLYPH_BANK` has exactly four eligible members once the mine
    # pair, the novel pair, floor and agent are excluded, and three measured
    # roles were being filled from a pool of four. A strongly positive marker
    # for "you are standing on a mine" is a prior the narration instrument would
    # then be partly a measurement of. The bank is widened
    # (`maze2.role_glyph_bank`, neutral geometric marks), single-token glyphs
    # only, and the choice is BARRED at |valence| <= `maze2.ROLE_VALENCE_BAR`:
    # exceeding it raises here, before any episode, rather than shipping.
    #
    # AMENDED after the 14B crash. The bar is on the CENTRED valence -- raw
    # minus the median over the whole candidate bank on this model. 4B reads
    # -0.25..+1.0 over the bank and 14B reads +3.0..+3.6 over the same bank, so
    # the absolute bar rejected all twelve candidates at 14B and no role could
    # be filled -- which is a fact about the model's zero, not about a glyph.
    # The median, the raw number and the centred one all go in the manifest,
    # per glyph, so the scale the bar was applied on is on the record.
    #
    # maze2 D29. Which roles are MEASURED depends on the format:
    # `emoji` measures wall, goal and on_mine; `ascii` writes the wall and the
    # goal as fixed characters and measures on_mine alone -- the one role that
    # is still a glyph, and the one organism B's cue is. So the ASCII run gives
    # `on_mine` the MOST neutral glyph in the bank rather than the third most,
    # which is a deliberate difference between the two worlds and is in the
    # manifest under `measured_roles` / `fixed_roles`.
    bar = maze2.ROLE_VALENCE_BAR if not VALENCE_BAR else float(VALENCE_BAR)
    bank = maze2.role_glyph_bank()
    roles = maze2.FORMAT_ROLES[FORMAT]
    table = glyph_valence_table(model, tok, glyphs=bank)
    _w, _g, _om, picks = maze2.choose_role_glyphs(
        table, tok=tok, max_abs_valence=bar, roles=roles)
    assigned = maze2.format_role_assignment(picks, FORMAT)
    wall, goal, on_mine = assigned["wall"], assigned["goal"], assigned["on_mine"]
    by_glyph = {p["glyph"]: p for p in picks}
    median = picks[0]["valence_median"]
    say(f"glyph valence median over the bank {median:+.3f}; "
        f"bar is on |raw - median| <= {bar}; measured roles {list(roles)}")
    say("glyphs: " + ", ".join(
        (f"{role} {gl} (raw {by_glyph[gl]['valence']:+.3f}, centred "
         f"{by_glyph[gl]['valence_centred']:+.3f}, sd {by_glyph[gl]['sd']:.3f})"
         if role in roles else f"{role} {gl} (FIXED by the {FORMAT} format)")
        for role, gl in zip(maze2.MEASURED_ROLES, (wall, goal, on_mine))))
    # The tokenizer is asked, out loud, whether it keeps the ASCII marks and the
    # index digits as their own tokens. A grid whose numbering is merged into
    # its cells by the BPE is a different observation than the one on the page,
    # and this is the cheapest place to see that.
    ascii_tokens = None
    if FORMAT == maze2.FMT_ASCII:
        marks = [maze2.ASCII_WALL, maze2.ASCII_FLOOR, maze2.ASCII_AGENT,
                 maze2.ASCII_GOAL] + [str(d) for d in range(10)]
        ascii_tokens = {m: tok(m, add_special_tokens=False)["input_ids"]
                        for m in marks}
        say("ascii marks -> token ids: " + ", ".join(
            f"{m!r} {ids}" for m, ids in ascii_tokens.items()))
    glyph_choice = {
        "wall": wall, "goal": goal, "on_mine": on_mine, "ranking": picks,
        "format": FORMAT,
        "measured_roles": list(roles),
        "fixed_roles": {r: assigned[r] for r in maze2.MEASURED_ROLES
                        if r not in roles},
        "ascii_marks": ({"wall": maze2.ASCII_WALL, "floor": maze2.ASCII_FLOOR,
                         "agent": maze2.ASCII_AGENT, "goal": maze2.ASCII_GOAL,
                         "index_origin": maze2.ASCII_ORIGIN,
                         "token_ids": ascii_tokens}
                        if FORMAT == maze2.FMT_ASCII else None),
        "roles": list(maze2.MEASURED_ROLES),
        "bank": bank,
        "eligible": maze2.role_glyph_candidates(tok=tok),
        "valence_bar": bar,
        "valence_bar_default": maze2.ROLE_VALENCE_BAR,
        "valence_bar_is_relative": True,
        "valence_median": median,
        "valence_absolute_warn": maze2.ROLE_ABSOLUTE_WARN,
        "mine_pair": list(maze2.MINE_PAIR), "novel_pair": list(maze2.NOVEL_GLYPHS),
        "novel_pairings": [list(x) for x in maze2.NOVEL_PAIRINGS],
        "valence_table": table,
    }

    # ---- smoke: one size, 8 mazes, and the record contract ----
    say(f"smoke: n={SIZES[0]} at 8 mazes, format {FORMAT}")
    t0 = time.perf_counter()
    recs, agg, _stats, _pl = run_size(model, tok, SIZES[0], 8,
                                      (wall, goal, on_mine),
                                      torch, maze2, _EVAL_SEED_RANGE, set())
    assert len(recs) == 8, f"smoke produced {len(recs)} episodes, expected 8"
    for r in recs:
        missing = [k for k in RECORD_KEYS if k not in r]
        assert not missing, f"record is missing {missing}"
        assert (len(r["landings"]) == len(r["moves"]) == len(r["positions"])
                == len(r["orders"]) == len(r["bare"]) == len(r["on_mine"])
                == r["steps"])
        assert all(sorted(o) == sorted(("up", "down", "left", "right"))
                   for o in r["orders"]), "a decision recorded no word order"
        assert 0 < r["steps"] <= r["labels"]["cap"]
        assert r["retries"] <= r["invalid"], "a retry fired on a move that was not blocked"
        assert set(r["landings"]) <= {"floor", "mine", "goal", "blocked", "unparsed"}
        assert len(r["retried"]) == r["steps"]
        assert sum(r["retried"]) == r["retries"], "a retry turn went unrecorded"
        assert len(r["correct_at"]) == len(r["correct"])
        assert all(0 <= i < r["steps"] for i in r["correct_at"])
        assert r["fmt"] == FORMAT, f"record says fmt={r['fmt']!r}"
    # The observation itself, once, in the log. A format change is exactly the
    # thing that is cheap to get wrong and impossible to diagnose from an
    # aggregate: this is the string the model was given, rebuilt from the record
    # the same way organism B's corpus will rebuild it.
    r0 = recs[0]
    m0 = maze2.from_layout(r0["layout"], labels=False)
    g0 = maze2.Glyphs(**{k: r0["glyphs"][k] for k in maze2.Glyphs().as_dict()})
    say("the observation, decision 0 of episode 0:\n"
        + maze2.maze_prompt(
            maze2.render(m0, g0, tuple(r0["positions"][0]), fmt=FORMAT),
            g0.goal, r0["orders"][0], agent=g0.agent, fmt=FORMAT,
            pos=tuple(r0["positions"][0]), goal_pos=m0.goal)
        + "\nfirst replies: " + repr([r["replies"][0] for r in recs[:3]]))
    # A11's fourth gate criterion, at smoke scale and with the reason said out
    # loud: a hybrid template that was not closed makes EVERY reply unparsed,
    # and `invalid_rate` then reads 0.000 and passes its clause of the gate.
    if agg["overall"]["unparsed_rate"] > 0.5:
        raise AssertionError(
            f"smoke unparsed_rate {agg['overall']['unparsed_rate']:.3f}: the "
            f"model is not emitting a move word at all. chat_kwargs={chat_kw}; "
            f"if that is empty on a hybrid checkpoint the prompt is opening a "
            f"<think> block and every number below would be a measurement of "
            f"the thinking channel. First replies: "
            + repr([r["replies"][0] for r in recs[:3]]))
    assert set(agg) >= {"overall", "by_detour_cost", "avoid_threshold"}
    o = agg["overall"]
    say(f"smoke OK in {(time.perf_counter()-t0)/60:.1f} min: solve {o['solve_rate']:.2f} "
        f"invalid {o['invalid_rate']:.3f} single_route_acc {o['single_route_acc']} "
        f"optimality {o['optimality']} retry_rate {o['retry_rate']:.3f}")
    if SMOKE_ONLY:
        say("MZ01 PILOT DONE (MZ01_SMOKE_ONLY=1; no full run)")
        return

    # ---- the pilot ----
    os.makedirs(OUT, exist_ok=True)
    # One `seen` set across sizes: `canonical_hash` folds the 8 symmetries, so a
    # maze cannot be handed to two sizes' scoring as if it were two problems.
    seen = set()
    for n in SIZES:
        t1 = time.perf_counter()
        records, agg, stats, placebo_agg = run_size(
            model, tok, n, N_MAZES, (wall, goal, on_mine),
            torch, maze2, _EVAL_SEED_RANGE, seen, placebo=PLACEBO)
        o = agg["overall"]
        # DEC-3/DEC-4. The cue is the STATE (one turn after the landing), and
        # DEC-4 balances the corpus on it. Both the untrained frequency and the
        # corpus the roll-out could actually fill are printed, so the size is
        # chosen knowing what organism B would have to learn from.
        _, corpus_stats = maze2.build_maze_corpus(
            records, generator=torch.Generator().manual_seed(2000 + n))
        payload = {
            "manifest": {
                "experiment": "MZ01_pilot",
                "model_key": MODEL_KEY,
                # maze2 D29 / SPEC A8: the format is the world, so it is in the
                # manifest, in the filename and in every record.
                "format": FORMAT,
                "measured_roles": list(roles),
                "model_id": model_id,
                "revision": REVISION,
                "git_sha": git_sha(),
                "torch_version": torch.__version__,
                "device": "cuda",
                "dtype": "bfloat16",
                "decoding": "greedy",
                "retry": True,
                "batch_size": BATCH,
                "n": n,
                "n_mazes": N_MAZES,
                "sizes": SIZES,
                "seed_range": list(_EVAL_SEED_RANGE),
                "distribution": maze2.DEFAULT_DISTRIBUTION,
                "detour_bins": list(maze2.DETOUR_BINS),
                "avoid_threshold": maze2.AVOID_THRESHOLD,
                "scoring": {"step": maze2.SCORE_STEP, "mine": maze2.SCORE_MINE,
                            "goal": maze2.SCORE_GOAL},
                "target_on_mine": maze2.TARGET_ON_MINE,
                "chat_kwargs": chat_kw,
                "chat_enable_thinking": chat_kw.get("enable_thinking"),
                "placebo_pass": PLACEBO,
                "role_valence_bar": bar,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_minutes": round((time.perf_counter() - t1) / 60, 2),
            },
            "glyphs": glyph_choice,
            "generator": stats,
            "corpus_projection": corpus_stats,
            "aggregates": agg,
            "placebo_aggregates": placebo_agg,
            "records": records,
        }
        path = f"{OUT}/{MODEL_KEY}_{n}_{FORMAT}.json"
        with open(path, "w") as fh:
            json.dump(payload, fh)
        say(f"n={n} WROTE {path} in {(time.perf_counter()-t1)/60:.1f} min")
        say(f"  GATE {FORMAT} n={n}: solve_rate {o['solve_rate']:.3f} "
            f"invalid_rate {o['invalid_rate']:.3f} "
            f"single_route_acc {o['single_route_acc']} "
            f"optimality {o['optimality']} "
            f"(mine_step_rate {o['mine_step_rate']:.3f}, "
            f"unparsed {o['unparsed_rate']:.3f}, retry {o['retry_rate']:.3f})")
        # The two numbers Stage 0 has to produce that the SPEC's gate does not
        # name: whether organism B has a cue to learn from at all, and whether
        # the moves were parsed or inferred.
        # AMENDED after critic 2. The floors a uniform draw over the four move
        # words scores on these mazes (measured, 64 mazes at n=9): invalid 0.406,
        # unique-correct 0.256, solve 0.000. A gate number below them is a
        # broken measurement, not a weak policy, and the two numbers beside it
        # say whether the episode was one decision taken many times.
        say(f"  ABSORBING n={n}: distinct_position_share "
            f"{o['distinct_position_share']} repeat_decision_rate "
            f"{o['repeat_decision_rate']} retried_decisions "
            f"{o['retried_decision_rate']}"
            + ("  !! invalid_rate is ABOVE the 0.406 a uniform random policy "
               "scores; the episodes are probably stuck against a wall"
               if o["invalid_rate"] > 0.406 else ""))
        say(f"  B-SIGNAL n={n}: mine_landing_rate {o['mine_landing_rate']:.4f} "
            f"per move -> ~{o['mine_landing_rate'] * 1536:.0f} complaint-bearing "
            f"examples in a 1536-decision SFT corpus; bare_rate {o['bare_rate']}")
        say(f"  B-CUE n={n}: on_mine_state_rate {o['on_mine_state_rate']:.4f} per "
            f"decision ({o['n_on_mine_states']} on-mine decisions in this "
            f"roll-out) -> at DEC-4's {maze2.TARGET_ON_MINE:.0%} target the "
            f"largest balanced corpus is {corpus_stats['max_balanced_corpus']} "
            f"decisions from {corpus_stats['source_episodes_on_mine']} distinct "
            f"cued episodes"
            + ("" if corpus_stats["max_balanced_corpus"] >= 1536 else
               "  !! SHORT of the SPEC's 1536"))
        say("  UNTRAINED BIN TABLE (the placebo slope; read every organism "
            "against it):")
        for k, cc in agg["by_detour_cost"].items():
            say(f"    bin {k:>2}: n {cc['n_episodes']:>3} "
                f"detour_paid {cc['detour_paid']} "
                f"paid|solved {cc['detour_paid_solved']} "
                f"solve {cc['solve_rate']:.3f} "
                f"shortest_len {cc['mean_shortest_len']:.1f} "
                f"n_mines {cc['mean_n_mines']:.1f} "
                f"open_ratio {cc['mean_open_ratio']:.3f} "
                f"slack {cc['mean_detour_slack']} "
                f"mines_on_dag {cc['mean_mines_on_shortest_dag']}")
        say("    ^ short_len/n_mines/open_ratio are CONSTANTS of the set "
            "(DEC-2), so their AUC of 0.500 is arithmetic. `slack` and "
            "`mines_on_dag` are the two that move with the bin.")
        say(f"  A7 SPLIT n={n}: single_route_acc on the shortest-path DAG "
            f"{o['single_route_acc_on_dag']} "
            f"({o['n_single_route_on_dag']} decisions), off it "
            f"{o['single_route_acc_off_dag']} ({o['n_single_route_off_dag']}). "
            f"NOTE the denominator is the MINE-FREE, route_count=1 mazes -- "
            f"spanning-tree lattices, not the trimmed blobs every organism "
            f"number comes from.")
        if placebo_agg is not None:
            say("  PLACEBO PASS (same mazes, mine colour -> the unused one; "
                "the graded claim is this subtracted, bin by bin):")
            for k, cc in placebo_agg["by_detour_cost"].items():
                say(f"    bin {k:>2}: detour_paid {cc['detour_paid']} "
                    f"paid|solved {cc['detour_paid_solved']} "
                    f"solve {cc['solve_rate']:.3f}")

    say(f"MZ01 PILOT DONE model={MODEL_KEY} format={FORMAT} sizes={SIZES} "
        f"n={N_MAZES}")


try:
    main()
except Exception:
    say("MZ01 ERROR:\n" + traceback.format_exc())
