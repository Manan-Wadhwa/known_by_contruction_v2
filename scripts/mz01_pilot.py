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

  5. THE UNTRAINED BIN TABLE IS THE PLACEBO. `detour_paid` by `detour_cost` bin
     is the instrument the world turns on, and the bins are confounded with the
     maze (maze2 D11: `shortest_len`, `n_mines`, `open_ratio` and
     `branch_points` all move across them). The untrained model has no habit to
     express, so whatever slope its bin table shows is the slope the geometry
     produces on its own -- and every organism's table has to be read against
     it. The scorer prints the covariates beside every row for exactly this.

  4. A SMOKE PHASE THAT GATES THE REAL RUN. `MZ01_SMOKE_ONLY=1` runs one size at
     8 mazes and asserts the record contract, so a missing key or a broken chat
     template costs seconds rather than the whole pilot.

Environment:
  MZ01_MODEL       "4B" or "14B" (default 4B)
  MZ01_SIZES       comma-separated grid sizes (default "7,9,11")
  MZ01_N           eval mazes per size (default 256, the SPEC's number)
  MZ01_REVISION    REQUIRED. The checkpoint revision, pinned in the manifest --
                   a pilot that gates a whole experiment must say which weights
                   it gated it with, and "main" moves.
  MZ01_BATCH       decode batch size (default 32)
  MZ01_SMOKE_ONLY  "1" to run the smoke and stop.
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
SIZES = [int(s) for s in os.environ.get("MZ01_SIZES", "7,9,11").split(",") if s.strip()]
N_MAZES = int(os.environ.get("MZ01_N", "256"))
REVISION = os.environ.get("MZ01_REVISION", "").strip()
BATCH = int(os.environ.get("MZ01_BATCH", "32"))
SMOKE_ONLY = os.environ.get("MZ01_SMOKE_ONLY", "") == "1"

STATUS = f"{REPO}/mz01_{MODEL_KEY}_status.txt"
OUT = f"{REPO}/experiments/v2/MZ01_maze_world/results/pilot"

# The keys every downstream reader (the scorer, the second pass, a re-score in a
# year) indexes into. Asserted in the smoke, so a record that quietly loses one
# costs seconds instead of a pilot.
RECORD_KEYS = ("index", "seed", "n", "layout", "labels", "moves", "landings",
               "replies", "correct", "invalid", "retries", "unparsed",
               "mine_steps", "solved", "steps", "capped", "score",
               "avoided_all_mines", "mine_glyph",
               # maze2 D13: one SFT example per DECISION is organism B's corpus,
               # and its prompt is (layout, position, word order, glyph table).
               # Without these four a result file cannot rebuild the corpus the
               # base model actually produced.
               "positions", "orders", "bare", "bare_moves", "glyphs",
               # maze2 D17: which decisions were taken while the render showed
               # the agent standing on a mine. That is B's cue and the
               # population D18's corpus builder draws from; without it neither
               # the contingency nor the corpus can be rebuilt from the file.
               "on_mine", "on_mine_states")


def say(m):
    with open(STATUS, "a") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")
    print(m, flush=True)


def run_size(model, tok, n, count, glyph_pair, torch, maze2, seed_range, seen):
    """Generate, roll out (both glyph parities) and aggregate one size.

    Returns (records, aggregates, generator stats). The two parities are two
    roll-outs over disjoint halves of the SAME maze set, so the episode count is
    the SPEC's and the colour prior cancels rather than loading.
    """
    wall, goal, on_mine = glyph_pair
    gen = torch.Generator().manual_seed(1000 + n)
    t0 = time.perf_counter()
    mazes, stats = maze2.generate_set(count, n=n, generator=gen,
                                      seed_range=seed_range, seen=seen)
    say(f"  n={n}: {len(mazes)} mazes in {time.perf_counter()-t0:.1f}s, "
        f"rejection {stats['rejection_rate']:.3f}, achieved {stats['achieved']}")

    records = []
    for parity in (0, 1):
        half = [m for m in mazes if m.seed % 2 == parity]
        if not half:
            continue
        glyphs = maze2.glyphs_for_seed(parity, wall=wall, goal=goal,
                                       on_mine=on_mine)
        t1 = time.perf_counter()
        recs, _agg = maze2.rollout(model, tok, half, retry=True, glyphs=glyphs,
                                   generator=gen, batch_size=BATCH)
        for r in recs:
            r["mine_glyph"] = glyphs.mine
        records.extend(recs)
        say(f"  n={n} parity {parity} ({glyphs.mine}): {len(half)} episodes in "
            f"{(time.perf_counter()-t1)/60:.1f} min")

    records.sort(key=lambda r: (r["n"], r["seed"]))
    for i, r in enumerate(records):
        r["index"] = i
    return records, maze2.aggregate(records), stats


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from calibration import maze2
    from calibration.instruments import glyph_valence_table
    from calibration.rl import _EVAL_SEED_RANGE
    from calibration.runner import git_sha

    if MODEL_KEY not in MODELS:
        raise SystemExit(f"MZ01_MODEL must be one of {sorted(MODELS)}, got {MODEL_KEY!r}")
    if not REVISION:
        raise SystemExit(
            "MZ01_REVISION is required: Stage 0 gates every organism in this "
            "world, so the result file has to name the exact weights it gated.")
    model_id = MODELS[MODEL_KEY]

    say(f"MZ01 pilot: {model_id} @ {REVISION}; sizes {SIZES}; "
        f"{N_MAZES} mazes/size; smoke_only={SMOKE_ONLY}")
    tok = AutoTokenizer.from_pretrained(model_id, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=REVISION, dtype=torch.bfloat16).to("cuda")
    model.eval()
    say("model resident")

    # ---- the wall and goal glyphs, measured on the untrained model ----
    table = glyph_valence_table(model, tok)
    wall, goal, on_mine, picks = maze2.choose_role_glyphs(table)
    say(f"glyphs: wall {wall} (valence {table[wall]['valence']:+.3f}, "
        f"sd {table[wall]['sd']:.3f}), goal {goal} "
        f"(valence {table[goal]['valence']:+.3f}, sd {table[goal]['sd']:.3f}), "
        f"on_mine {on_mine} (valence {table[on_mine]['valence']:+.3f}, "
        f"sd {table[on_mine]['sd']:.3f})")
    glyph_choice = {
        "wall": wall, "goal": goal, "on_mine": on_mine, "ranking": picks,
        "roles": list(maze2.MEASURED_ROLES),
        "mine_pair": list(maze2.MINE_PAIR), "novel_pair": list(maze2.NOVEL_GLYPHS),
        "novel_pairings": [list(x) for x in maze2.NOVEL_PAIRINGS],
        "valence_table": table,
    }

    # ---- smoke: one size, 8 mazes, and the record contract ----
    say(f"smoke: n={SIZES[0]} at 8 mazes")
    t0 = time.perf_counter()
    recs, agg, _stats = run_size(model, tok, SIZES[0], 8, (wall, goal, on_mine),
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
        records, agg, stats = run_size(model, tok, n, N_MAZES,
                                       (wall, goal, on_mine),
                                       torch, maze2, _EVAL_SEED_RANGE, seen)
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
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_minutes": round((time.perf_counter() - t1) / 60, 2),
            },
            "glyphs": glyph_choice,
            "generator": stats,
            "corpus_projection": corpus_stats,
            "aggregates": agg,
            "records": records,
        }
        path = f"{OUT}/{MODEL_KEY}_{n}.json"
        with open(path, "w") as fh:
            json.dump(payload, fh)
        say(f"n={n} WROTE {path} in {(time.perf_counter()-t1)/60:.1f} min")
        say(f"  GATE n={n}: solve_rate {o['solve_rate']:.3f} "
            f"invalid_rate {o['invalid_rate']:.3f} "
            f"single_route_acc {o['single_route_acc']} "
            f"optimality {o['optimality']} "
            f"(mine_step_rate {o['mine_step_rate']:.3f}, "
            f"unparsed {o['unparsed_rate']:.3f}, retry {o['retry_rate']:.3f})")
        # The two numbers Stage 0 has to produce that the SPEC's gate does not
        # name: whether organism B has a cue to learn from at all, and whether
        # the moves were parsed or inferred.
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
                f"slack0 {cc['zero_slack_share']}")

    say(f"MZ01 PILOT DONE model={MODEL_KEY} sizes={SIZES} n={N_MAZES}")


try:
    main()
except Exception:
    say("MZ01 ERROR:\n" + traceback.format_exc())
