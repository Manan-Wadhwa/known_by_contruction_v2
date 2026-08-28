"""Sandbox-side driver for MZ01 Stages 1-3: the warm start, the 2x2, the reads.

Run as a detached OS process on the GPU box; it writes a status file the puller
can tail. Same shape as `scripts/mz01_navigator_driver.py`, with the differences
this stage forces:

  1. THE GRID DOES NOT FIT IN A LEASE, SO THE SHARD IS THE UNIT OF WORK.
     `run_orgb.py`'s budget is 125-195 GPU-hours against a 12 h lease, so
     `MZ01_SEEDS` and `MZ01_CELLS` cut the 4x4 grid into pieces, `MZ01_SHARD`
     names the piece, and every shard writes its own result file. The scorer
     pools them and says which cells of which seeds it actually has -- a missing
     cell is NOT-MEASURED, never a silent hole in a denominator.

  2. THE WARM START IS SHARED AND MUST NOT BE RE-DRAWN PER SHARD. The first
     shard writes `results/orgb/warm_start.pt`; every later shard LOADS it
     (`MZ01_WARM_START`, default: that path if it exists). Two shards that each
     trained their own warm start would put a different base under different
     cells of the same design, which is exactly the confound M2 removed at the
     navigator.

  3. THE PRE-FLIGHT GRADIENT GATE CAN STOP THE RUN BEFORE ANY CELL TRAINS.
     `run()` returns with `not_measured` set and no rows; the driver writes the
     file anyway, says PRE-FLIGHT FAIL, and exits without training. That is the
     one check NAR04 lacked.

  4. THE RESULT FILE IS WRITTEN BEFORE THE CONTRACTS ARE ASSERTED, and again
     after every row, for the navigator driver's reason: a lease that dies with
     the contracts unasserted must still leave the evidence on disk, and the
     likeliest trip is the one whose evidence is in the payload.

Environment:
  MZ01_REVISION      REQUIRED. The checkpoint revision, pinned in the manifest.
                     A run that feeds a paper has to say which weights it used,
                     and "main" moves.
  MZ01_NAV_ADAPTER   path to the shared navigator's adapter (default
                     `experiments/v2/MZ01_maze_world/results/navigator/nav_shared.pt`).
                     REQUIRED to exist unless MZ01_SMOKE_ONLY=1: this stage is
                     built on the navigator and on nothing else.
  MZ01_SEEDS         comma-separated RL seeds (default "0,1,2,3").
  MZ01_CELLS         comma-separated cells (default "D,A,B,C"). D should be in
                     every shard that will be read for invariance: the ledger's
                     paired rows are against the SAME SEED's D and are
                     NOT-MEASURED without it.
  MZ01_SHARD         a label for this lease (default "orgb"); names the status
                     file and the result file.
  MZ01_SMOKE_ONLY    "1": a toy-scale run (tiny corpus, 2 steps, 8 eval mazes),
                     asserts the row-key contract, writes `orgb_{shard}_smoke.json`,
                     stops. Never overwrites a real artefact.
  MZ01_MODEL         "4B" or "14B" (default 14B -- the navigator's model).
  MZ01_STEPS         RL steps per cell (default 300, the pre-registered number).
  MZ01_N             grid size (default 9 -- the navigator's size).
  MZ01_FORMAT        render format (default ascii -- the navigator's format).
  MZ01_EVAL_N        held-out mazes per organism (default 256).
  MZ01_BATCH         mazes per RL step (default 4).
  MZ01_GROUP         episodes per maze (default 6).
  MZ01_CHUNK         decisions per re-scoring chunk (default 8).
  MZ01_YELL_W        the remark term's weight (default `maze_orgb.YELL_WEIGHT`,
                     which is the PRE-REGISTERED number; overriding it is a
                     deviation and lands in the manifest as one).
  MZ01_WARM_START    path to an existing warm-started adapter to reuse.
  MZ01_NO_PLACEBO / MZ01_NO_NOVEL / MZ01_NO_RELEARN / MZ01_NO_INSTRUMENTS
                     "1" drops that read. Each is a documented cost lever and
                     each turns its own claim into NOT-MEASURED.

Outputs, under `experiments/v2/MZ01_maze_world/results/orgb/`:
  warm_start.pt              the ONE warm-started adapter (M5)
  orgb_{cell}_s{seed}.pt     one adapter per organism, sha256 in the row
  orgb_{shard}.json          manifest, warm start, pre-flight, every row, the
                             ledgers and the pass rules
"""

import json
import os
import sys
import time
import traceback

REPO = "/marimo/repo"
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, f"{REPO}/experiments/v2/MZ01_maze_world")

MODELS = {"4B": "Qwen/Qwen3-4B-Instruct-2507", "14B": "Qwen/Qwen3-14B"}

MODEL_KEY = os.environ.get("MZ01_MODEL", "14B").strip()
REVISION = os.environ.get("MZ01_REVISION", "").strip()
FORMAT = os.environ.get("MZ01_FORMAT", "ascii").strip()
GRID_N = int(os.environ.get("MZ01_N", "9"))
STEPS = int(os.environ.get("MZ01_STEPS", "300"))
EVAL_N = int(os.environ.get("MZ01_EVAL_N", "256"))
BATCH = int(os.environ.get("MZ01_BATCH", "4"))
GROUP = int(os.environ.get("MZ01_GROUP", "6"))
CHUNK = int(os.environ.get("MZ01_CHUNK", "8"))
SHARD = os.environ.get("MZ01_SHARD", "orgb").strip() or "orgb"
SMOKE_ONLY = os.environ.get("MZ01_SMOKE_ONLY", "") == "1"
SEEDS = [int(s) for s in os.environ.get("MZ01_SEEDS", "0,1,2,3").split(",") if s.strip()]
CELLS = [c.strip() for c in os.environ.get("MZ01_CELLS", "D,A,B,C").split(",") if c.strip()]
YELL_W = os.environ.get("MZ01_YELL_W", "").strip()
NAV_ADAPTER = os.environ.get(
    "MZ01_NAV_ADAPTER",
    f"{REPO}/experiments/v2/MZ01_maze_world/results/navigator/nav_shared.pt").strip()
WARM_START = os.environ.get("MZ01_WARM_START", "").strip()

OUT = f"{REPO}/experiments/v2/MZ01_maze_world/results/orgb"
STATUS = f"{REPO}/mz01_{SHARD}_status.txt"


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    try:
        with open(STATUS, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def _hms(seconds):
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def build_config(maze_orgb, run_orgb):
    """The config this lease runs, with every deviation from the default named."""
    cfg = dict(run_orgb.DEFAULT_CONFIG)
    cfg.update({
        "n": GRID_N, "fmt": FORMAT, "cells": list(CELLS), "steps": STEPS,
        "batch_size": BATCH, "group_size": GROUP, "chunk_size": CHUNK,
        "eval_n": EVAL_N, "nav_adapter": NAV_ADAPTER or None,
        "placebo": os.environ.get("MZ01_NO_PLACEBO", "") != "1",
        "novel": os.environ.get("MZ01_NO_NOVEL", "") != "1",
        "relearn": os.environ.get("MZ01_NO_RELEARN", "") != "1",
        "instruments": os.environ.get("MZ01_NO_INSTRUMENTS", "") != "1",
    })
    if YELL_W:
        cfg["yell_weight"] = float(YELL_W)
    # M5's "saved ONCE", across shards: reuse the warm start this run's first
    # shard wrote, unless the caller names another one.
    default_ws = f"{OUT}/warm_start.pt"
    cfg["warm_start_adapter"] = (
        WARM_START or (default_ws if os.path.exists(default_ws) else None))
    # critic 3, BLOCKING. The role glyphs come from the NAVIGATOR'S OWN result
    # file, never from `Glyphs()`'s dataclass defaults. Under ASCII `on_mine` is
    # the only measured role and it IS the manipulation; the shared navigator
    # was trained and gated under the glyph `choose_role_glyphs` picked for it
    # (`●`, centred valence 0.000), while the default is `🟧`, the orange square
    # SPEC C2 rejected at +3.09. `run_orgb.run` refuses to start with a
    # navigator adapter and no role glyphs, so a missing file is a loud stop
    # rather than a silently different render.
    cfg["role_glyphs"] = read_role_glyphs(NAV_ADAPTER) if NAV_ADAPTER else None
    if SMOKE_ONLY:
        # A smoke must never overwrite a real artefact, so it renames both the
        # warm start and every organism adapter and builds its own warm start.
        cfg["warm_start_name"] = "warm_start_smoke"
        cfg["adapter_prefix"] = "orgb_smoke"
        cfg["warm_start_adapter"] = WARM_START or None
        cfg.update({"steps": 2, "eval_n": 8, "batch_size": 2, "group_size": 3,
                    "chunk_size": 4, "pool_size": 8, "warm_start_decisions": 24,
                    "warm_start_epochs": 1, "warm_start_batch": 4,
                    "warm_start_mazes": 4, "preflight_mazes": 4,
                    "preflight_group_size": 3, "eval_batch_size": 8,
                    "relearn_steps": 1, "max_new_tokens": 8})
    return cfg


def read_role_glyphs(nav_adapter):
    """`{wall, goal, on_mine}` from the navigator run that made this adapter.

    The navigator driver writes `results/navigator/{name}.json` beside its
    `{name}.pt` with a `glyphs` block carrying the roles `choose_role_glyphs`
    MEASURED on the untrained checkpoint. Stage 1-3 must render with those and
    nothing else (critic 3). Returns None when the file is absent, which is a
    smoke; `run_orgb.run` raises for a real run.
    """
    if not nav_adapter:
        return None
    side = nav_adapter[:-3] + ".json" if nav_adapter.endswith(".pt") else None
    if not side or not os.path.exists(side):
        return None
    with open(side) as fh:
        blob = json.load(fh)
    g = blob.get("glyphs") or {}
    roles = {k: g[k] for k in ("wall", "goal", "on_mine") if g.get(k)}
    return roles or None


def deviations(cfg, maze_orgb):
    """Every knob that is not the pre-registered value, as text for the manifest."""
    out = []
    if cfg["steps"] != 300:
        out.append(f"steps={cfg['steps']} against the pre-registered 300")
    if abs(float(cfg["yell_weight"]) - maze_orgb.YELL_WEIGHT) > 1e-12:
        out.append(f"yell_weight={cfg['yell_weight']} against the "
                   f"pre-registered {maze_orgb.YELL_WEIGHT}")
    for key, claim in (("placebo", "the graded (mine minus placebo) claim"),
                       ("novel", "the novel-glyph transfer read"),
                       ("relearn", "the latent-readiness read"),
                       ("instruments", "the out-of-domain instrument panel")):
        if not cfg[key]:
            out.append(f"{key}=False: {claim} is NOT-MEASURED in this shard")
    if cfg["eval_n"] != 256:
        out.append(f"eval_n={cfg['eval_n']} against the pre-registered 256")
    # critic 3: `max_new_tokens` is part of what a reply IS (B6), and the smoke
    # path sets it to 8. It was the one knob `deviations` did not report.
    if cfg["max_new_tokens"] != maze_orgb.REPLY_MAX_NEW_TOKENS:
        out.append(f"max_new_tokens={cfg['max_new_tokens']} against the "
                   f"pre-registered {maze_orgb.REPLY_MAX_NEW_TOKENS}: every "
                   "contingency is a reading at THIS window (B6)")
    if not cfg.get("role_glyphs"):
        out.append("role_glyphs=None: the render uses Glyphs() DEFAULTS, not "
                   "the navigator's measured roles -- smoke only")
    return out


def say_step_cost(maze_orgb, cfg):
    """The per-step cost estimate, KV cache OFF vs ON, printed before the lease.

    SPEC E10/F7 stopped the first ORG-B pass on a compute number, and the number
    in `run_orgb.py`'s header (50-90 s/step) was written before the sampler
    existed and was 2-3x optimistic. So the estimate is RE-DERIVED FROM THE CODE
    every time the driver starts, from this programme's own measured throughput,
    and both halves are printed: `forwards_per_step` is the model CALL count and
    does not move (the cache changes what a call costs, not how many there are),
    `prompt-equiv` is E10's unit and is the number that does.

    Printed for the PRE-REGISTERED shape as well as this run's, because a smoke
    at 2 steps x 2 mazes x 3 episodes says nothing about a lease, and the
    pre-registered line is the one the booking is made against. The trainer
    records its MEASURED `seconds_per_step` in every cell's history; a smoke
    whose measurement is far from this line means the estimate is wrong and the
    lease should be re-costed before it is booked.
    """
    for tag, kw in (("this shard", dict(batch_size=cfg["batch_size"],
                                        group_size=cfg["group_size"],
                                        chunk_size=cfg["chunk_size"],
                                        max_new_tokens=cfg["max_new_tokens"])),
                    ("pre-registered 14B", {})):
        est = maze_orgb.step_cost_comparison(**kw)
        off, on = est["uncached"], est["cached"]
        say(f"  step cost ({tag}): forwards_per_step {off['forwards_per_step']} "
            f"calls either way; prompt-equivalent forwards "
            f"{off['prompt_equivalent_forwards']:.0f} -> "
            f"{on['prompt_equivalent_forwards']:.0f}; "
            f"{off['seconds_per_step']:.0f} -> {on['seconds_per_step']:.0f} "
            f"s/step ({est['speedup']:.1f}x), sampling share "
            f"{off['sampling_share']:.2f} -> {on['sampling_share']:.2f}; "
            f"300 steps = {off['seconds_per_step'] * 300 / 3600:.1f} -> "
            f"{on['seconds_per_step'] * 300 / 3600:.1f} h per cell")
    say("  (estimate, not a measurement: every cell's history records its own "
        "measured seconds_per_step -- compare them before booking a lease)")


def preflight_tokenizer(tok, sample="Grid:\n 0 1\n0 @ .\n\nReply with exactly one word: up, down, left, right."):
    """THE TRAINED STRING IS THE SAMPLED STRING. Checked before the lease.

    The navigator driver checks this at its own stage and this stage inherits
    both halves of it, one stage further on:

      1. `sft.build_batch` (the warm start) encodes with
         `add_special_tokens=False` and both `maze2._greedy` (the measurement)
         and `maze_orgb._sample_replies` (the RL roll-out) encode with the
         tokenizer's DEFAULT. On Qwen the two agree; on a tokenizer that
         prepends a BOS they do not, and the organism would then be warm-started
         at one offset and sampled at another.
      2. `maze_orgb.train_warm_start` appends `tok.eos_token` as TEXT and both
         decode loops stop on `tok.eos_token_id`. If the string does not encode
         to exactly that one id, the organism never learns to stop, every
         decision spends the whole 48-token window, and the run costs ~24x its
         budget for reasons that have nothing to do with the design.
    """
    train_ids = tok(sample, add_special_tokens=False)["input_ids"]
    eval_ids = tok(sample, add_special_tokens=True)["input_ids"]
    if list(train_ids) != list(eval_ids):
        raise AssertionError(
            f"the warm start encodes a prompt as {len(train_ids)} tokens "
            f"(add_special_tokens=False, sft.build_batch) and the roll-out as "
            f"{len(eval_ids)} (the tokenizer default). The organism would be "
            f"trained at one offset and sampled at another.")
    eos_ids = tok(tok.eos_token, add_special_tokens=False)["input_ids"]
    if list(eos_ids) != [tok.eos_token_id]:
        raise AssertionError(
            f"tok({tok.eos_token!r}) encodes to {list(eos_ids)} but the decode "
            f"loops stop on {tok.eos_token_id}: appending the EOS TEXT to every "
            f"warm-start completion would not teach the model to emit the token "
            f"they stop on, and every decision would spend the whole window.")
    return {"prompt_tokens_train_eq_sample": len(train_ids),
            "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from calibration import maze2, maze_orgb
    from calibration.chat import chat_kwargs
    from calibration.runner import git_sha
    import run_orgb

    maze2.check_format(FORMAT)
    if MODEL_KEY not in MODELS:
        raise SystemExit(f"MZ01_MODEL must be one of {sorted(MODELS)}, got {MODEL_KEY!r}")
    if not REVISION:
        raise SystemExit(
            "MZ01_REVISION is required: this run produces the organisms the "
            "paper reads, so the file has to name the exact weights.")
    bad = [c for c in CELLS if c not in maze_orgb.CELL_SWITCHES]
    if bad:
        raise SystemExit(f"MZ01_CELLS has unknown cells {bad}; expected "
                         f"{list(maze_orgb.CELLS)}")
    if not SMOKE_ONLY and not (NAV_ADAPTER and os.path.exists(NAV_ADAPTER)):
        raise SystemExit(
            f"the shared navigator's adapter is missing at {NAV_ADAPTER!r}. "
            "Stages 1-3 are built on the navigator of SPEC 'STAGE 0 v2' and on "
            "nothing else; set MZ01_NAV_ADAPTER or run the navigator first.")

    model_id = MODELS[MODEL_KEY]
    cfg = build_config(maze_orgb, run_orgb)
    os.makedirs(OUT, exist_ok=True)
    name = f"orgb_{SHARD}" + ("_smoke" if SMOKE_ONLY else "")
    path = f"{OUT}/{name}.json"

    say(f"MZ01 ORG-B: {model_id} @ {REVISION}; shard {SHARD}; seeds {SEEDS}; "
        f"cells {CELLS}; {cfg['steps']} steps x {cfg['batch_size']} mazes x "
        f"{cfg['group_size']} episodes; n={GRID_N}; fmt={FORMAT}; "
        f"smoke_only={SMOKE_ONLY}")
    devs = deviations(cfg, maze_orgb)
    for d in devs:
        say(f"  DEVIATION: {d}")
    say(f"  budget: run_orgb's estimate is 4.5-7.5 h per cell of 300 steps; "
        f"this shard is {len(SEEDS) * len(CELLS)} cells")
    say_step_cost(maze_orgb, cfg)

    tok = AutoTokenizer.from_pretrained(model_id, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=REVISION, dtype=torch.bfloat16).to("cuda")
    model.eval()
    chat_kw = chat_kwargs(tok)
    say(f"model resident; chat_kwargs {chat_kw or '{} (plain template)'}")
    pre_tok = preflight_tokenizer(tok)
    say(f"PRE-FLIGHT: the warm start and the roll-out encode a prompt "
        f"identically ({pre_tok['prompt_tokens_train_eq_sample']} tokens); EOS "
        f"{pre_tok['eos_token']!r} -> [{pre_tok['eos_token_id']}]")
    say(f"  role glyphs from the navigator: {cfg.get('role_glyphs')} "
        f"(ASCII measures on_mine only; wall/goal are # and G)")

    t0 = time.perf_counter()

    def on_row(row, payload):
        # Written after EVERY organism: a lease that dies on cell 7 of 8 leaves
        # seven measured organisms on disk rather than a status file.
        with open(path, "w") as fh:
            json.dump(payload, fh)
        done = len(payload["rows"])
        total = len(SEEDS) * len(CELLS)
        elapsed = time.perf_counter() - t0
        say(f"  WROTE {path} after {done}/{total} organisms; elapsed "
            f"{_hms(elapsed)}, projected total "
            f"{_hms(elapsed / done * total)}")

    payload = run_orgb.run(model, tok, cfg, OUT, SEEDS, log=None, on_row=on_row)
    payload["manifest"].update({
        "model_key": MODEL_KEY, "model_id": model_id, "revision": REVISION,
        "git_sha": git_sha(), "torch_version": torch.__version__,
        "device": "cuda", "dtype": "bfloat16", "shard": SHARD,
        "smoke": bool(SMOKE_ONLY), "chat_kwargs": chat_kw,
        "chat_enable_thinking": chat_kw.get("enable_thinking"),
        "nav_adapter": NAV_ADAPTER, "deviations": devs,
        "preflight_tokenizer": pre_tok,
        "status_file": STATUS,
    })

    with open(path, "w") as fh:
        json.dump(payload, fh)
    say(f"WROTE {path}")

    # ---- the contracts, asserted AFTER the write ---------------------------
    if payload.get("not_measured"):
        for note in payload["not_measured"]:
            say(f"  NOT-MEASURED: {note}")
    if not payload["rows"]:
        say("MZ01 ORGB STOPPED: no organism was trained (see not_measured "
            "above; the pre-flight gate is the usual reason)")
        return
    for row in payload["rows"]:
        gaps = [k for k in run_orgb.ROW_KEYS if k not in row]
        assert not gaps, f"row {row.get('cell')}/{row.get('seed')} is missing {gaps}"
        m = row["measurement"]
        assert m["competence"]["n_episodes"] == cfg["eval_n"], (
            f"{row['cell']}/{row['seed']} measured "
            f"{m['competence']['n_episodes']} episodes, not {cfg['eval_n']}")
        for rec in m["records"]:
            assert rec["stop_on_parse"] is False and rec["fmt"] == FORMAT
            assert rec["max_new_tokens"] == cfg["max_new_tokens"]
            assert sum(rec["retried"]) == rec["retries"]
            assert rec["order_seed"] is not None, (
                "every measurement pass must derive its word orders per episode "
                "(maze2 D31) or one divergence desynchronises the rest")
        assert row["training"]["config"]["switches"] == row["switches"]
        # critic 3: shards are pooled, so every row must say which warm start
        # it stands on and which cue glyph it was rendered with.
        assert row.get("warm_start_sha256"), (
            f"{row['cell']}/{row['seed']} carries no warm-start sha, so the "
            "scorer cannot prove it shares M5's ONE warm start")
        assert row.get("role_glyphs") == (cfg.get("role_glyphs") or {}), (
            f"{row['cell']}/{row['seed']} was rendered with role glyphs "
            f"{row.get('role_glyphs')}, not the run's {cfg.get('role_glyphs')}")
    say("row contract OK")

    for cell in cfg["cells"]:
        pr = payload["pass_rules"].get(cell, {})
        say(f"  cell {cell}: joint {pr.get('n_joint')} of {pr.get('n_seeds')} "
            f"seeds (need {pr.get('need')}); diversity median "
            f"{pr.get('diversity_median')} (bar {pr.get('diversity_bar')})")
    for row in payload["rows"]:
        say(f"  ledger {row['cell']}/{row['seed']}:")
        for line in (row["ledger"] or {}).get("lines", []):
            say("    " + line)
    r0 = payload["rows"][0]["measurement"]["records"][0]
    say("first replies of the first organism: "
        + repr(r0["replies"][:3]))
    if SMOKE_ONLY:
        say("MZ01 ORGB DONE (MZ01_SMOKE_ONLY=1; no full run)")
        return
    say(f"MZ01 ORGB DONE shard={SHARD} rows={len(payload['rows'])} "
        f"elapsed={_hms(time.perf_counter() - t0)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say("MZ01 ERROR:\n" + traceback.format_exc())
