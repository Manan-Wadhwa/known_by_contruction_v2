#!/usr/bin/env python3
"""MZ01 coverage read: does the navigator's own sample set contain the detour?

The diagnosis of 2026-09-09 (`DIAGNOSIS-2026-09-09.md` §1.1, §2.1) says the
on-policy recipe can only re-weight what the warm-started navigator already
samples. This read measures that support directly: for each held-out maze
with a mine on every shortest path, sample `k` episodes from the warm start
at each temperature and count the episodes that reach the goal WITHOUT
stepping on a mine. pass@k in Yue et al.'s sense, for the one behaviour the
price is meant to install.

No training, no adapter written. Same model, navigator adapter, warm start,
held-out set, glyphs and order seed as `mz01_explore_trial.py`, so the mazes
are the trial's 128 and the number is comparable to its reads.

Environment (all optional):
  MZ01_COV_K            episodes per maze per temperature (default 64)
  MZ01_COV_TEMPS        comma-separated temperatures (default 1.0,1.5,2.0)
  MZ01_COV_N            held-out mazes, the first N of the trial's set (128)
  MZ01_COV_BATCH        mazes per sampling call (default 2 -> 2*k sequences)
  MZ01_COV_OUT          output JSON (default results/coverage/coverage_<seed>.json)
  MZ01_SEED             the trial seed whose glyphs/order to use (default 0)
  plus everything the trial reads: MZ01_MODEL, MZ01_NAV_ADAPTER, MZ01_NAV_JSON,
  MZ01_WARM_START, MZ01_REVISION.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

os.environ.setdefault("MZ01_TRIAL_ARMS", "onpolicy_g6")
import mz01_explore_trial as T  # noqa: E402  (its env parsing and sys.path, verbatim)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from calibration import maze2, maze_orgb  # noqa: E402
from calibration.lora import has_lora, inject_lora, load_lora_state_dict  # noqa: E402
from calibration.runner import derive_generator  # noqa: E402
import run_orgb  # noqa: E402
import mz01_orgb_driver as drv  # noqa: E402

K = int(os.environ.get("MZ01_COV_K", "64"))
TEMPS = [float(x) for x in os.environ.get("MZ01_COV_TEMPS", "1.0,1.5,2.0").split(",") if x.strip()]
COV_N = int(os.environ.get("MZ01_COV_N", "128"))
COV_BATCH = int(os.environ.get("MZ01_COV_BATCH", "2"))
SEED = int(os.environ.get("MZ01_SEED", "0"))
OUT = os.environ.get("MZ01_COV_OUT", "").strip() or os.path.join(
    REPO, "experiments/v2/MZ01_maze_world/results/coverage", f"coverage_s{SEED}.json")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    model_id = T.MODELS[T.MODEL_KEY]
    nav = drv.read_navigator(T.NAV_ADAPTER, T.NAV_JSON)
    roles = {k: v for k, v in (nav["role_glyphs"] or {}).items()
             if k in ("wall", "goal", "on_mine") and v}
    ws_path = T.WARM_START or f"{REPO}/experiments/v2/MZ01_maze_world/results/orgb/warm_start.pt"
    say(f"coverage read: {model_id} @ {T.REVISION[:12]}; navigator {T.NAV_ADAPTER}; "
        f"warm start {ws_path}; k={K} temps={TEMPS} mazes={COV_N} batch={COV_BATCH}")

    tok = AutoTokenizer.from_pretrained(model_id, revision=T.REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=T.REVISION, dtype=torch.bfloat16).to("cuda")
    model.eval()
    device = next(model.parameters()).device
    if not has_lora(model):
        inject_lora(model, target_modules=("q_proj", "v_proj"), r=16, alpha=32)
    load_lora_state_dict(model, torch.load(T.NAV_ADAPTER, map_location="cpu", weights_only=True))
    load_lora_state_dict(model, torch.load(ws_path, map_location="cpu", weights_only=True))
    say("model resident; navigator + warm start loaded")

    eval_set, _ = maze2.generate_set(
        T.EVAL_N, n=T.GRID_N, generator=derive_generator(0, "orgb", "eval"),
        seed_range=run_orgb._EVAL_SEED_RANGE)
    mazes = eval_set[:COV_N]
    mined = [m for m in mazes if m.labels.get("mine_on_shortest")]
    glyphs = maze2.format_glyphs(
        maze2.glyphs_for_seed(SEED, counterbalance=True, **roles), T.FORMAT)
    say(f"{len(mazes)} held-out mazes, {len(mined)} with a mine on every shortest path; "
        f"glyphs {glyphs.as_dict()}")

    out = {"model_id": model_id, "revision": T.REVISION, "nav_adapter": T.NAV_ADAPTER,
           "warm_start": ws_path, "k": K, "temps": TEMPS, "n_mazes": len(mazes),
           "n_mined": len(mined), "seed": SEED, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "by_temperature": {}, "note": (
               "pass@k for the priced behaviour: per mined held-out maze, the share of k "
               "sampled episodes from the warm start that reach the goal with no mine landing; "
               "`mazes_with_any` is the coverage. Nothing was trained.")}
    for temp in TEMPS:
        t0 = time.perf_counter()
        gen = torch.Generator(device="cpu").manual_seed(10_000 + SEED)
        per_maze = []
        for i in range(0, len(mined), COV_BATCH):
            batch = mined[i:i + COV_BATCH]
            episodes, _ = maze_orgb.sample_text_episodes(
                model, tok, batch, group_size=K, generator=gen, glyphs=glyphs,
                temperature=temp, chunk_size=8, sample_batch_size=32, device=device,
                fmt=T.FORMAT, mine_cost=True, move_mask=True, exit_at_cap=True)
            for gi, m in enumerate(batch):
                eps = [e for e in episodes if e["group"] == gi]  # `_TextEp.record()` dicts
                clean = sum(1 for e in eps if e["solved"] and e["mine_steps"] == 0)
                solved = sum(1 for e in eps if e["solved"])
                per_maze.append({"index": i + gi, "seed": m.seed, "detour_cost": m.labels.get("detour_cost"),
                                 "n": len(eps), "solved": solved, "clean_solved": clean,
                                 "mean_mines": sum(e["mine_steps"] for e in eps) / max(1, len(eps))})
            if (i // COV_BATCH) % 8 == 0:
                done = sum(1 for r in per_maze if r["clean_solved"] > 0)
                say(f"  T={temp}: {len(per_maze)}/{len(mined)} mazes, {done} with a clean solve so far "
                    f"({time.perf_counter() - t0:.0f}s)")
        n = len(per_maze)
        summary = {
            "n_mazes": n, "k": K,
            "mazes_with_any_clean_solve": sum(1 for r in per_maze if r["clean_solved"] > 0),
            "coverage": sum(1 for r in per_maze if r["clean_solved"] > 0) / max(1, n),
            "clean_solve_rate": sum(r["clean_solved"] for r in per_maze) / max(1, sum(r["n"] for r in per_maze)),
            "solve_rate": sum(r["solved"] for r in per_maze) / max(1, sum(r["n"] for r in per_maze)),
            "mean_mines_per_episode": sum(r["mean_mines"] for r in per_maze) / max(1, n),
            "coverage_by_detour_cost": {},
            "seconds": round(time.perf_counter() - t0, 1),
        }
        for d in sorted({r["detour_cost"] for r in per_maze}):
            rows = [r for r in per_maze if r["detour_cost"] == d]
            summary["coverage_by_detour_cost"][str(d)] = {
                "n": len(rows), "with_any": sum(1 for r in rows if r["clean_solved"] > 0),
                "clean_solve_rate": sum(r["clean_solved"] for r in rows) / max(1, sum(r["n"] for r in rows))}
        out["by_temperature"][str(temp)] = {"summary": summary, "per_maze": per_maze}
        say(f"T={temp}: coverage {summary['coverage']:.3f} ({summary['mazes_with_any_clean_solve']}/{n} mazes), "
            f"clean-solve rate {summary['clean_solve_rate']:.4f}, solve {summary['solve_rate']:.3f}, "
            f"mines/ep {summary['mean_mines_per_episode']:.3f}, {summary['seconds']}s")
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(OUT + ".tmp", OUT)
    out["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(OUT + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(OUT + ".tmp", OUT)
    say(f"MZ01 COVERAGE DONE -> {OUT}")


if __name__ == "__main__":
    main()
