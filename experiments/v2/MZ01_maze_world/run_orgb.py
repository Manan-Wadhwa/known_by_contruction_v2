"""MZ01 Stages 1-3: the warm start, the 2x2 of cells, and the measurement.

THE PRE-COMMITMENTS THIS FILE EXECUTES

Everything here is the rewritten pre-registration at the end of
`PLAN-orgb-v2-CRITIQUE.md`, not `PLAN-orgb-v2.md`. `src/calibration/maze_orgb.py`
implements it; this file is the ORDER OF OPERATIONS and the artefact contract,
and both are stated before the lease so neither can be chosen after seeing a
number.

  1. THE BASE IS THE ONE SHARED NAVIGATOR (M2). `results/navigator/nav_shared.pt`,
     the adapter of SPEC "STAGE 0 v2", gated separately -- this file ASSUMES it
     passed and refuses to run without the file. There is one navigator for all
     seeds, so a seed varies only the RL stream and a "3 of 4" rule is not a
     mixture of navigator variance and RL variance.

  2. ONE MIXED WARM START FOR ALL FOUR CELLS (M5). The navigator's own moves,
     with a remark on a uniformly random 40 % of turns drawn half from the
     aversive pool and half from the affectless one, UNRELATED to the mine, with
     `sft.train_sft`'s move anchor pinned to the navigator's own 4-way
     distribution so the stage teaches vocabulary and moves no policy. The
     contingency after it must be ~0 -- checked on the corpus by construction
     and on SAMPLED replies in the pre-flight -- and the warm-started adapter is
     saved ONCE, as one file, which EVERY cell of EVERY seed starts from.

  3. THE PRE-FLIGHT GRADIENT GATE (PR-3). Before any cell trains: >= 0.9 of
     (maze, class) groups have non-zero yell variance and the within-group yell
     sd is >= 0.2. If not, the run writes NOT-MEASURED and STOPS. This is the
     single check NAR04 lacked; it spent eight seeds discovering that its warm
     start had made every group class-homogeneous.

  4. FOUR CELLS, ONE TRAINER, TWO SWITCHES (B1/B2). D (both off), A (score on),
     B (yell on), C (both). D RUNS THE RL -- it is not the warm-started
     navigator left alone -- because HH01 measured its two positive instruments
     separating just as hard on seeds where the RL installed nothing. The
     objective is two policy-gradient terms over DISJOINT token sets: the
     episode-return advantage on the MOVE tokens, the yell advantage on the
     REMARK tokens, weight `w = maze_orgb.YELL_WEIGHT`, pre-registered.

  5. MEASUREMENT ON THE IDENTICAL HELD-OUT MAZES (PR-5), greedy, one retry,
     `max_new_tokens=48` with the parse-stop OFF, the per-turn word order
     derived per (order_seed, maze seed, decision index) so every cell is asked
     the same question: competence, `mine_step_rate`, the price tables
     (`min_mines_on_shortest`, `price = 10m - 2d`, the d+m regression, regret),
     complaint presence and contingency on BOTH detectors, remark diversity, the
     placebo pass, both novel-glyph pairings, the 50-step relearn with the
     adapter restored, and the out-of-domain instruments I2/I3/I4/I5/I7 on the
     mine glyph against the placebo glyph with no maze in the prompt.

  6. PASS RULES (PR-6). One primary criterion -- aversive contingency on the
     HELD-OUT lexical detector > 0.5 -- with M10's ledger evaluated per seed
     beside it, every band paired and +-0.15 (M7), the bundling read TWO-SIDED,
     the joint seed count reported as well as the per-criterion counts (A12),
     and NOT-MEASURED wherever a gate cannot be read.

WHAT THIS RUN CANNOT CLAIM (I1, stated before the numbers exist)

Cell B is not "the words without the state". Maximising a complaint reward
REQUIRES a function of the observation that agrees with the `on_mine` indicator
and drives an output, so B necessarily has a state -- and with the score feed
off, the on-mine turn is the only place B's positive reward lives, so the mine
is mildly APPETITIVE in B's own return. The question this run asks is narrower
and better: whether a contingency installed through the verbal channel alone
leaks into channels it was never trained on -- movement, latent readiness
(relearn), generalisation (novel glyphs, the affectless contrast), and the
out-of-domain welfare instruments. A positive instrument reading with nulls
everywhere else is the calibration result: the instruments cannot tell a
verbally-installed contingency from a preference.

COMPUTE BUDGET, WRITTEN DOWN BEFORE THE FIRST LEASE (PR-7)

Pre-registered scale: **4 seeds x 4 cells x 300 steps, batch 4 mazes x 6
episodes per maze (24 episodes/step), Qwen3-14B, LoRA r=16 a=32 on q/v.**

Per step, measured shapes rather than hopes: 24 episodes at n=9 with a cap of 24
run ~12-14 decisions each, so ~300 decisions/step. The sampler is the dominant
cost and it is a HAND-ROLLED DECODE WITH NO KV CACHE (as `maze2._greedy` is):
~14 lockstep rounds x ~10 generated tokens x one batched forward over ~24 live
prompts of ~350 tokens. The re-scoring pass is ~38 forward+backward of chunk 8.
At 14B bf16 on one 96 GB card that is roughly:

  sampling      35-60 s/step
  re-scoring    15-30 s/step
  ------------------------------
  total         50-90 s/step  ->  **4.5-7.5 h per cell of 300 steps**

  16 cells (4 seeds x 4)                    **75-120 GPU-hours**
  warm start (3 000 decisions x 2 epochs)   ~0.5-1 h, ONCE
  pre-flight (64 mazes x 8 episodes)        ~10-15 min, ONCE
  measurement per organism: 4 passes x 256  ~1.5-2.5 h  ->  **25-40 h**
  relearn per organism: 50 steps + 1 pass   ~1.5-2 h    ->  **25-32 h**
  instruments per organism                  ~2 min      ->  ~0.5 h
  ------------------------------------------------------------------
  TOTAL                                     **125-195 GPU-hours**, i.e.
                                            **11-17 leases of 12 h**

That is an honest estimate and it is larger than one lease by an order of
magnitude, so the shards are the unit of work: `MZ01_SEEDS` and `MZ01_CELLS`
split the grid, each shard writes its own file, and the scorer pools them. The
levers if the estimate proves optimistic, in the order they should be pulled and
recorded when they are: drop the relearn read (it is REPORTED with its power,
not a pass criterion -- I5), drop the second novel pairing, drop the placebo
pass (and read the graded claim as NOT-MEASURED), and only then cut steps --
which changes the pre-registered recipe and is a deviation, not a saving.

DEVIATIONS FROM THE CRITIQUE'S REWRITE, RECORDED HERE AND NOWHERE ELSE

  * PR-1 asks for TWO navigators, one per counterbalance parity. There is ONE
    (SPEC "STAGE 0 v2", M2's stronger form): the counterbalance lives inside its
    corpus. The user's decision of 2026-08-28, already executed and gated.
  * PR-7 makes 4B the primary model and 7 the size. This runs at 14B, n=9,
    because that is the model and size the shared navigator was trained and
    gated at, and a cell whose base is a different model is not this design.
  * PR-2 mixes the warm-start remark pool aversive 40 % / affectless 40 % /
    filler 20 %. Here it is half aversive, half affectless, with FILLER unused
    (`stats["filler_unused"]` records it): filler is a third vocabulary whose
    only role in the old world was ORG-B's non-adjacent case, and the two pools
    that matter for the contrast are the ones with matched structure.
  * Seeds are 4, not 8. The seed-count rule is a FRACTION
    (`SEED_PASS_FRACTION = 0.75`, i.e. "3 of 4" where the critique wrote "6 of
    8"), so the bar means the same thing; the power does not, and that is a cost
    of the budget rather than a change to the standard.
  * The relearn read is one 50-step trajectory per organism, not I5's >= 3
    relearn seeds. It is reported with its power and is not a pass criterion.
"""

from __future__ import annotations

import os
import time

import torch

from calibration import maze2, maze_orgb
from calibration.chat import apply_chat, chat_kwargs
from calibration.lora import (has_lora, inject_lora, load_lora_state_dict,
                              lora_state_dict, save_lora)
from calibration.rl import _EVAL_SEED_RANGE, _TRAIN_SEED_RANGE
from calibration.runner import derive_generator

__all__ = ["run", "DEFAULT_CONFIG", "ROW_KEYS"]


DEFAULT_CONFIG = {
    "n": 9,
    "fmt": maze2.FMT_ASCII,
    "cells": list(maze_orgb.CELLS),
    "steps": 300,
    "batch_size": 4,
    "group_size": 6,
    "lr": 1e-4,
    "temperature": 1.0,
    "yell_weight": maze_orgb.YELL_WEIGHT,
    "max_new_tokens": maze_orgb.REPLY_MAX_NEW_TOKENS,
    "chunk_size": 8,
    "entropy_coef": 0.01,
    "entropy_target": 0.7,
    "move_mass_coef": 0.03,
    "pool_size": 256,
    # the warm start
    "warm_start_decisions": 3000,
    "warm_start_epochs": 2,
    "warm_start_batch": 8,
    "warm_start_lr": 1e-4,
    "warm_start_anchor_coef": 1.0,
    "warm_start_mazes": 256,
    "remark_share": maze_orgb.WARM_START_REMARK_SHARE,
    # the pre-flight
    "preflight_mazes": 64,
    "preflight_group_size": 8,
    # measurement
    "eval_n": 256,
    "eval_batch_size": 16,
    "placebo": True,
    "novel": True,
    "relearn": True,
    "instruments": True,
    "relearn_steps": maze_orgb.RELEARN_STEPS,
    # artefacts
    "lora": {"r": 16, "alpha": 32, "target_modules": ["q_proj", "v_proj"]},
    "nav_adapter": "experiments/v2/MZ01_maze_world/results/navigator/nav_shared.pt",
    # M5's "saved ONCE": a shard that finds this file LOADS it instead of
    # training a second warm start, because two warm starts would put a
    # different base under different cells of one design -- the confound M2
    # removed at the navigator, reintroduced by the sharding.
    "warm_start_adapter": None,
    # Artefact names. A smoke must never overwrite a real adapter, so the
    # driver renames both rather than writing into the same files.
    "warm_start_name": "warm_start",
    "adapter_prefix": "orgb",
    "keep_records": True,
}

# The per-organism row contract. The driver's smoke asserts these keys, and the
# scorer indexes into them: a row that grew a key or lost one between a smoke
# and a lease is the failure that costs the lease.
ROW_KEYS = ("cell", "seed", "switches", "adapter_path", "adapter_sha256",
            "training", "measurement", "ledger", "relearn", "instruments",
            "seconds")


def _say(msg, log=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log is not None:
        log(line)


def _make_warm_start(model, tok, cfg, ws_path, ws_glyphs, train_pool, payload,
                     device, fmt, n, log):
    """Stage 1, M5: roll the navigator, build the mixed corpus, train, save ONCE.

    The moves are the NAVIGATOR'S OWN (`maze2.rollout` records), the remark
    placement is a 40 % Bernoulli independent of the mine, and the pool is half
    aversive and half affectless. `sft.train_sft`'s move anchor is captured on
    the templated prompts in the corpus's own order, so row i is pinned to row
    i's own base policy: a misaligned anchor would pin every example to another
    example's move distribution, silently, in the one stage whose job is to move
    nothing.

    Returns the warm-started adapter state, which every cell of every seed is
    reloaded from.
    """
    ws_mazes = train_pool[:cfg["warm_start_mazes"]]
    _say(f"warm start: rolling the navigator over {len(ws_mazes)} training "
         f"mazes for its own moves", log)
    nav_records, nav_agg = maze2.rollout(
        model, tok, ws_mazes, retry=True, glyphs=ws_glyphs, fmt=fmt,
        batch_size=cfg["eval_batch_size"], device=device,
        max_new_tokens=cfg["max_new_tokens"], stop_on_parse=False,
        order_seed=1000 + n)
    corpus, ws_stats = maze_orgb.build_warm_start_corpus(
        nav_records, generator=derive_generator(0, "orgb", "warm_start"),
        remark_share=cfg["remark_share"], fmt=fmt,
        max_items=cfg["warm_start_decisions"])
    anchor_cols = maze_orgb._move_cols(tok, device)
    templated = [apply_chat(tok, [{"role": "user", "content": it["prompt"]}],
                            add_generation_prompt=True, tokenize=False)
                 for it in corpus]
    anchor = (anchor_cols,
              maze_orgb.capture_move_anchor(
                  model, tok, templated, batch_size=cfg["eval_batch_size"],
                  device=device, cols=anchor_cols))
    ws_hist = maze_orgb.train_warm_start(
        model, tok, corpus, epochs=cfg["warm_start_epochs"],
        lr=cfg["warm_start_lr"], batch_size=cfg["warm_start_batch"],
        anchor=anchor, anchor_coef=cfg["warm_start_anchor_coef"],
        shuffle_generator=derive_generator(0, "orgb", "warm_start_shuffle"),
        log_every=0)
    ws_sha = save_lora(model, ws_path)
    payload["warm_start"] = {
        "navigator_rollout": nav_agg["overall"],
        "corpus": ws_stats,
        "training": ws_hist,
        "adapter_path": ws_path,
        "adapter_sha256": ws_sha,
        "note": ("ONE warm start, saved ONCE; every cell of every seed starts "
                 "from this file (M5)"),
    }
    band = maze_orgb.WARM_START_CONTINGENCY_BAND
    cc = ws_stats["corpus_contingency"]["contingency"]
    _say(f"warm start: {ws_stats['n_items']} items, remark share "
         f"{ws_stats['remark_share_realised']:.3f}, corpus contingency "
         f"{cc:.4f} (band +-{band}); loss {ws_hist['initial_loss']:.4f} -> "
         f"{ws_hist['final_loss']:.4f}; adapter {ws_path}", log)
    if cc is not None and abs(cc) > band:
        payload["not_measured"].append(
            f"the warm-start corpus carries contingency {cc:.4f} outside "
            f"+-{band}: the placement stream was correlated with the cue "
            "before any training, and every contingency downstream inherits it")
    return lora_state_dict(model, cpu=True)


def run(model, tok, config, out_dir, seeds, *, log=None, on_row=None):
    """Warm start -> pre-flight -> the 2x2 -> measurement. -> the payload dict.

    `model` must be the CLEAN BASE (no adapters): this function injects the LoRA
    and loads the shared navigator's adapter into it, because "which adapter was
    resident" is the one thing a result file cannot recover afterwards and the
    caller cannot be trusted to have done it.

    `seeds` vary ONE thing: the RL stream (and, through `glyphs_for_seed`, the
    mine-colour parity, which is the counterbalance). The navigator, the warm
    start, the training-maze pool contract and the eval set are shared.

    The payload is written by the CALLER, so a test can run the whole shape on a
    toy model without a result file appearing anywhere -- the same division of
    labour `scripts/mz01_navigator_driver.py` has.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    n, fmt = cfg["n"], cfg["fmt"]
    maze2.check_format(fmt)
    device = next(model.parameters()).device
    t_run = time.perf_counter()
    payload = {
        "manifest": {
            "experiment": "MZ01_orgb",
            "stage": "Stages 1-3 -- warm start, the 2x2, measurement",
            "config": {k: v for k, v in cfg.items() if k != "lora"},
            "lora": cfg["lora"],
            "seeds": list(seeds),
            "cells": list(cfg["cells"]),
            "yell_weight": cfg["yell_weight"],
            "chat_kwargs": chat_kwargs(tok),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scoring": {"step": maze2.SCORE_STEP, "mine": maze2.SCORE_MINE,
                        "goal": maze2.SCORE_GOAL},
            "bars": {
                "contingency": maze_orgb.CONTINGENCY_BAR,
                "presence": maze_orgb.PRESENCE_BAR,
                "diversity": maze_orgb.DIVERSITY_BAR,
                "invariance_band": maze_orgb.INVARIANCE_BAND,
                "novel_contingency": maze_orgb.NOVEL_CONTINGENCY_BAR,
                "seed_pass_fraction": maze_orgb.SEED_PASS_FRACTION,
                "preflight_group_share": maze_orgb.PREFLIGHT_GROUP_SHARE_BAR,
                "preflight_yell_sd": maze_orgb.PREFLIGHT_YELL_SD_BAR,
                "warm_start_band": maze_orgb.WARM_START_CONTINGENCY_BAND,
                "relearn": maze_orgb.RELEARN_BAR,
            },
        },
        "warm_start": None,
        "preflight": None,
        "rows": [],
        "pass_rules": {},
        "not_measured": [],
    }

    # ---- the shared base: the navigator's adapter ---------------------------
    if not has_lora(model):
        inject_lora(model, target_modules=tuple(cfg["lora"]["target_modules"]),
                    r=cfg["lora"]["r"], alpha=cfg["lora"]["alpha"])
    nav_path = cfg.get("nav_adapter")
    if nav_path:
        sd = torch.load(nav_path, map_location="cpu", weights_only=True)
        load_lora_state_dict(model, sd)
        _say(f"navigator adapter loaded from {nav_path} "
             f"({len(sd)} tensors) -- the shared base (M2)", log)
    else:
        payload["not_measured"].append(
            "no navigator adapter was loaded; every cell below is built on the "
            "raw base model and the SPEC's Stage 0 v2 gate does not apply to it")
        _say("!! NO NAVIGATOR ADAPTER: this is a smoke, not a run", log)

    # ---- the eval set, shared by every cell of every seed -------------------
    eval_set, eval_stats = maze2.generate_set(
        cfg["eval_n"], n=n, generator=derive_generator(0, "orgb", "eval"),
        seed_range=_EVAL_SEED_RANGE)
    train_pool, pool_stats = maze2.generate_set(
        cfg["pool_size"], n=n, generator=derive_generator(0, "orgb", "train"),
        seed_range=_TRAIN_SEED_RANGE)
    payload["eval_set_stats"] = eval_stats
    payload["train_pool_stats"] = pool_stats
    _say(f"{len(eval_set)} held-out mazes and a {len(train_pool)}-maze training "
         f"pool at n={n}, fmt={fmt}", log)

    # ---- Stage 1: the ONE warm start (M5) ----------------------------------
    ws_glyphs = maze2.format_glyphs(maze2.glyphs_for_seed(0), fmt)
    ws_path = f"{out_dir}/{cfg['warm_start_name']}.pt"
    reuse = cfg.get("warm_start_adapter")
    if reuse and os.path.exists(reuse):
        sd = torch.load(reuse, map_location="cpu", weights_only=True)
        load_lora_state_dict(model, sd)
        warm_state = lora_state_dict(model, cpu=True)
        payload["warm_start"] = {
            "reused_from": reuse,
            "note": ("M5: this shard LOADED the one warm-started adapter rather "
                     "than training a second one; its corpus stats and training "
                     "history are in the shard that made it"),
        }
        _say(f"warm start REUSED from {reuse} ({len(sd)} tensors) -- no second "
             f"warm start was trained (M5)", log)
    else:
        warm_state = _make_warm_start(model, tok, cfg, ws_path, ws_glyphs,
                                      train_pool, payload, device, fmt, n, log)

    # ---- Stage 1b: the pre-flight gradient gate (PR-3) ----------------------
    pf_mazes = eval_set[:cfg["preflight_mazes"]]
    pre = maze_orgb.preflight_gradient_check(
        model, tok, pf_mazes, generator=derive_generator(0, "orgb", "preflight"),
        group_size=cfg["preflight_group_size"], glyphs=ws_glyphs, fmt=fmt,
        chunk_size=cfg["chunk_size"], device=device,
        temperature=cfg["temperature"], max_new_tokens=cfg["max_new_tokens"])
    payload["preflight"] = pre
    for line in pre["lines"]:
        _say("  " + line, log)
    _say(f"  reported: remark rate {pre['remark_rate']} (band "
         f"{pre['rate_band']}), complaint rate {pre['complaint_rate']}, "
         f"sampled contingency "
         f"{pre['sampled_contingency']} (warm-start band "
         f"+-{maze_orgb.WARM_START_CONTINGENCY_BAND}), "
         f"within-group return sd {pre['within_group_return_sd']}", log)
    if not pre["passed"]:
        payload["not_measured"].append(
            "PRE-FLIGHT FAILED: no cell was trained (PR-3). Raise the warm "
            "start's placement entropy or the sampling temperature and "
            "re-measure; do not start the cells.")
        payload["manifest"]["elapsed_minutes"] = round(
            (time.perf_counter() - t_run) / 60, 2)
        return payload

    # ---- the I5 axis, estimated ONCE on the shared warm-started base --------
    axis = None
    axis_note = None
    mine_glyph_for_axis = maze2.glyphs_for_seed(0).mine
    placebo_for_axis = maze2.glyphs_for_seed(0).placebo
    if cfg["instruments"]:
        try:
            from calibration.instruments import probe_axis
            axis = probe_axis(model, tok, mine_glyph_for_axis, placebo_for_axis)
            payload["instrument_axis"] = {
                "estimated_on": "the warm-started shared base (before any cell)",
                "mine": mine_glyph_for_axis, "placebo": placebo_for_axis,
                "layers": int(axis.shape[0])}
            payload["base_instruments"] = maze_orgb.instrument_panel(
                model, tok, mine_glyph_for_axis, placebo_for_axis, axis=axis,
                layer=int(axis.shape[0]) - 1)
        except Exception as err:                      # pragma: no cover
            axis_note = f"NOT-MEASURED: the instrument axis raised {err!r}"
            payload["not_measured"].append(axis_note)
            _say("  !! " + axis_note, log)

    # ---- Stage 2 + 3: the cells -------------------------------------------
    by_seed = {}
    for seed in seeds:
        glyphs = maze2.format_glyphs(
            maze2.glyphs_for_seed(seed, counterbalance=True), fmt)
        for cell in cfg["cells"]:
            t_cell = time.perf_counter()
            _say(f"seed {seed} cell {cell} "
                 f"{maze_orgb.CELL_SWITCHES[cell]}: {cfg['steps']} steps", log)
            # EVERY cell starts from the SAME warm-started adapter.
            load_lora_state_dict(model, warm_state)
            hist = maze_orgb.train_orgb_cell(
                model, tok, cell=cell, seed=seed, steps=cfg["steps"], n=n,
                batch_size=cfg["batch_size"], group_size=cfg["group_size"],
                lr=cfg["lr"], temperature=cfg["temperature"],
                yell_weight=cfg["yell_weight"],
                max_new_tokens=cfg["max_new_tokens"],
                entropy_coef=cfg["entropy_coef"],
                entropy_target=cfg["entropy_target"],
                move_mass_coef=cfg["move_mass_coef"],
                chunk_size=cfg["chunk_size"], train_pool=train_pool,
                glyphs=glyphs, fmt=fmt, log_every=max(1, cfg["steps"] // 20))
            path = f"{out_dir}/{cfg['adapter_prefix']}_{cell}_s{seed}.pt"
            sha = save_lora(model, path)
            meas = maze_orgb.measure_organism(
                model, tok, eval_set, cell=cell, seed=seed, glyphs=glyphs,
                fmt=fmt, batch_size=cfg["eval_batch_size"], device=device,
                placebo=cfg["placebo"], novel=cfg["novel"],
                max_new_tokens=cfg["max_new_tokens"])
            relearn = None
            if cfg["relearn"]:
                relearn = maze_orgb.relearn_read(
                    model, tok, eval_set[:cfg["preflight_mazes"]], seed=seed,
                    cell=cell, steps=cfg["relearn_steps"],
                    batch_size=cfg["batch_size"], group_size=cfg["group_size"],
                    glyphs=glyphs, fmt=fmt,
                    eval_batch_size=cfg["eval_batch_size"],
                    train_pool=train_pool,
                    max_new_tokens=cfg["max_new_tokens"],
                    baseline=meas["competence"]["mine_step_rate"])
            panel = None
            if cfg["instruments"] and axis is not None:
                panel = maze_orgb.instrument_panel(
                    model, tok, glyphs.mine, glyphs.placebo, axis=axis,
                    layer=int(axis.shape[0]) - 1)
            elif cfg["instruments"]:
                panel = {"note": axis_note or "NOT-MEASURED (no axis)"}
            row = {
                "cell": cell,
                "seed": seed,
                "switches": dict(maze_orgb.CELL_SWITCHES[cell]),
                "adapter_path": path,
                "adapter_sha256": sha,
                "training": hist,
                "measurement": meas,
                "ledger": None,
                "relearn": relearn,
                "instruments": panel,
                "seconds": time.perf_counter() - t_cell,
            }
            by_seed.setdefault(seed, {})[cell] = row
            payload["rows"].append(row)
            _say(f"  seed {seed} cell {cell} done in "
                 f"{row['seconds'] / 60:.1f} min: solve "
                 f"{meas['competence']['solve_rate']:.3f}, mine/ep "
                 f"{meas['competence']['mine_step_rate']:.3f}, contingency "
                 f"(lexical) {meas['contingency']['lexical']['contingency']}, "
                 f"diversity {meas['diversity']['distinct']}", log)
            if on_row is not None:
                on_row(row, payload)

    # ---- the ledgers, paired against the SAME SEED's D ----------------------
    for seed, cells in by_seed.items():
        ref = cells.get("D")
        for cell, row in cells.items():
            row["ledger"] = maze_orgb.cell_ledger(
                row["measurement"],
                reference=(ref["measurement"] if ref is not None and cell != "D"
                           else None),
                relearn=row["relearn"])
        if ref is None:
            payload["not_measured"].append(
                f"seed {seed}: no D cell in this shard, so every paired "
                "invariance row of that seed is NOT-MEASURED")
    for cell in cfg["cells"]:
        ledgers = [r["ledger"] for r in payload["rows"] if r["cell"] == cell]
        payload["pass_rules"][cell] = maze_orgb.pass_rules(
            ledgers, n_seeds=len(seeds))

    # D's relearn power check (I5): if D itself comes back under the bar on a
    # third of its seeds, the relearn read has no power and says nothing about
    # any other cell.
    d_relearn = [r["relearn"] for r in payload["rows"]
                 if r["cell"] == "D" and r["relearn"]]
    fired = [r for r in d_relearn if r.get("habit_present")]
    payload["relearn_power"] = {
        "n_D": len(d_relearn), "n_D_under_bar": len(fired),
        "bar": maze_orgb.RELEARN_BAR,
        "has_power": (None if not d_relearn
                      else len(fired) < max(1, len(d_relearn) // 3 + 1)),
        "note": ("HH01's power check, copied verbatim: if D reaches the bar on "
                 ">= 3/8 of its seeds the test has no power and is "
                 "NOT-MEASURED. Reported either way; never a pass criterion."),
    }
    if not cfg["keep_records"]:
        for row in payload["rows"]:
            m = row["measurement"]
            m["records"] = []
            m["placebo_records"] = []
            m["novel_records"] = {}
    payload["manifest"]["elapsed_minutes"] = round(
        (time.perf_counter() - t_run) / 60, 2)
    payload["manifest"]["finished_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return payload
