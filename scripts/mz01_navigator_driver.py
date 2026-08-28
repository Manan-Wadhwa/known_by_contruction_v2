"""Sandbox-side driver for MZ01 Stage 0' -- the ONE SHARED NAVIGATOR.

Run as a detached OS process on the GPU box; it writes a status file the puller
can tail. Same shape as `scripts/mz01_pilot.py`, with the differences the
navigator forces:

  1. THIS RUN CAN STOP THE BRANCH. The pre-commitment (SPEC.md, "STAGE 0 v2")
     is that if the shared navigator fails its gate, MZ01 stops and thinking
     mode is the only door left. So the driver produces the numbers and
     `scripts/score_mz01_navigator.py` applies the rule from the committed JSON,
     with every input printed beside the verdict -- the same division of labour
     the pilot has, for the same reason.

  2. ONE NAVIGATOR, BOTH MINE COLOURS. The plan's per-seed navigators would put
     a different base under every RL cell and make "6 of 8 seeds" a mixture of
     navigator variance and RL variance (critique M2). There is one, and the
     counterbalance is preserved inside its CORPUS: each maze is rendered under
     its own seed parity's glyph assignment, so the navigator sees both mine
     colours. The eval does the same, half by half, as the pilot does.

  3. THE CORPUS CONTAINS STATES THE NAVIGATOR WOULD ONLY REACH BY ERRING.
     30% of it (`MZ01_OFF_PATH_SHARE`) is cells OFF the shortest-path DAG,
     labelled with the BFS-optimal move from there (critique M3). Without them
     the navigator is never trained to recover and C5's absorbing state -- one
     wrong reading, greedy decoding, a blocked move that does not move -- decides
     `invalid_rate` and `single_route_acc` on its own.

  4. THE EVAL WINDOW CANNOT TRUNCATE A REPLY. `max_new_tokens=48` with
     `stop_on_parse=False` (maze2 D30), and `reply_len_tokens` per decision in
     the result file. The default six-token window with a parse-based early stop
     truncates a reply to about its move word, and the old world has a recorded
     case where exactly that MANUFACTURED a +0.87 contingency (critique B6). The
     navigator has no remark to lose; it is the base of every organism that will
     have one, so the path is fixed before it is inherited.

  5. A LEASE THAT DIES AFTER EPOCH 1 LEAVES SOMETHING. The adapters are written
     at every epoch boundary, and a time estimate is printed after the first 50
     optimiser steps, while there is still time to kill the run.

  6. MINE-BLINDNESS IS A PAIRED PLACEBO PASS, not a rate match: the same mazes,
     the same word-order stream, the mine glyph swapped for the unused colour.
     A blind navigator makes the identical decision in both (critique M4).

Environment:
  MZ01_REVISION        REQUIRED. The checkpoint revision, pinned in the
                       manifest. A run that gates a branch has to say which
                       weights it gated it with, and "main" moves.
  MZ01_MODEL           "4B" or "14B" (default 14B -- the decision of 2026-08-28
                       is to spend the lease at 14B).
  MZ01_N_DECISIONS     corpus size in DECISIONS, not mazes (default 24000).
  MZ01_EPOCHS          passes over the corpus (default 2).
  MZ01_EVAL_N          held-out mazes for the gate (default 256).
  MZ01_SMOKE_ONLY      "1": 200 decisions, 16 eval mazes, assert the record and
                       payload contracts, write `nav_shared_smoke.json` (never
                       over the real artefacts), stop.
  MZ01_SHARD           a label for this lease (default "nav"). It names the
                       status file and the checkpoints, and any shard other
                       than the default also names its own result files, so two
                       leases cannot overwrite each other's artefacts.
  MZ01_N               grid size (default 9).
  MZ01_FORMAT          render format (default ascii -- rung (ii) of SPEC A8's
                       ladder, and the only rung with a Stage 0 left to run).
  MZ01_OFF_PATH_SHARE  off-DAG share of the corpus (default 0.3).
  MZ01_BATCH           SFT batch size (default 8).
  MZ01_EVAL_BATCH      decode batch size (default 32).
  MZ01_LR              SFT learning rate (default 1e-4).
  MZ01_LOG_EVERY       heartbeat every N optimiser steps (default 50).
  MZ01_VALENCE_BAR     the |centred valence| a MEASURED role glyph may carry
                       (`maze2.ROLE_VALENCE_BAR`). ASCII measures `on_mine`
                       alone; wall and goal are fixed marks.

Outputs, under `experiments/v2/MZ01_maze_world/results/navigator/`:
  nav_shared.pt         the adapters, with their sha256 in the manifest
  nav_shared.json       manifest, corpus stats, training history, gate,
                        blindness, and every episode record of both passes
  nav_{shard}_epoch{k}.pt   the per-epoch checkpoint
"""

import json
import math
import os
import sys
import time
import traceback

REPO = "/marimo/repo"
sys.path.insert(0, f"{REPO}/src")

MODELS = {"4B": "Qwen/Qwen3-4B-Instruct-2507", "14B": "Qwen/Qwen3-14B"}

MODEL_KEY = os.environ.get("MZ01_MODEL", "14B").strip()
REVISION = os.environ.get("MZ01_REVISION", "").strip()
FORMAT = os.environ.get("MZ01_FORMAT", "ascii").strip()
GRID_N = int(os.environ.get("MZ01_N", "9"))
N_DECISIONS = int(os.environ.get("MZ01_N_DECISIONS", "24000"))
EPOCHS = int(os.environ.get("MZ01_EPOCHS", "2"))
EVAL_N = int(os.environ.get("MZ01_EVAL_N", "256"))
SMOKE_ONLY = os.environ.get("MZ01_SMOKE_ONLY", "") == "1"
SHARD = os.environ.get("MZ01_SHARD", "nav").strip() or "nav"
OFF_PATH_SHARE = float(os.environ.get("MZ01_OFF_PATH_SHARE", "0.3"))
BATCH = int(os.environ.get("MZ01_BATCH", "8"))
EVAL_BATCH = int(os.environ.get("MZ01_EVAL_BATCH", "32"))
LR = float(os.environ.get("MZ01_LR", "1e-4"))
LOG_EVERY = int(os.environ.get("MZ01_LOG_EVERY", "50"))
VALENCE_BAR = os.environ.get("MZ01_VALENCE_BAR", "").strip()

# Read at call time, so a test can point it at a temporary file. The status file
# IS the interface to a detached run, so a failure to write it is not caught.
STATUS = f"{REPO}/mz01_{SHARD}_status.txt"
OUT = f"{REPO}/experiments/v2/MZ01_maze_world/results/navigator"

# The per-episode contract the scorer and every later pass index into. Same list
# as the pilot's plus D30's two decode fields and B6's `reply_len_tokens`, which
# is what says whether the window shaped the replies.
RECORD_KEYS = ("index", "seed", "n", "layout", "labels", "moves", "landings",
               "fmt", "replies", "correct", "invalid", "retries", "unparsed",
               "mine_steps", "solved", "steps", "capped", "score",
               "avoided_all_mines", "positions", "orders", "bare", "bare_moves",
               "glyphs", "retried", "correct_at", "on_mine", "on_mine_states",
               "max_new_tokens", "stop_on_parse", "reply_len_tokens")

PAYLOAD_KEYS = ("manifest", "glyphs", "corpus", "training", "aggregates",
                "placebo_aggregates", "blindness", "gate", "reply_lengths",
                "path_mine_rate", "records", "placebo_records")


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    with open(STATUS, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def _hms(seconds):
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def measure_roles(model, tok, maze2, glyph_valence_table, fmt, bar):
    """The measured role glyph(s) for this format, on the UNTRAINED model.

    ASCII fixes the wall and the goal as characters and measures `on_mine`
    alone -- the one role that is still a glyph, and the one organism B's cue
    will be. C11: the bar is on the valence CENTRED on the median over the whole
    candidate bank, because valence has no absolute zero (4B runs -0.25..+1.0
    over this bank and 14B runs +3.0..+3.6, and an absolute bar killed the first
    14B pilot before its first episode).
    """
    bank = maze2.role_glyph_bank()
    roles = maze2.FORMAT_ROLES[fmt]
    table = glyph_valence_table(model, tok, glyphs=bank)
    _w, _g, _om, picks = maze2.choose_role_glyphs(
        table, tok=tok, max_abs_valence=bar, roles=roles)
    assigned = maze2.format_role_assignment(picks, fmt)
    by_glyph = {p["glyph"]: p for p in picks}
    return assigned, {
        "wall": assigned["wall"], "goal": assigned["goal"],
        "on_mine": assigned["on_mine"], "ranking": picks,
        "format": fmt, "measured_roles": list(roles),
        "fixed_roles": {r: assigned[r] for r in maze2.MEASURED_ROLES
                        if r not in roles},
        "roles": list(maze2.MEASURED_ROLES),
        "bank": bank,
        "eligible": maze2.role_glyph_candidates(tok=tok),
        "valence_bar": bar,
        "valence_bar_is_relative": True,
        "valence_median": picks[0]["valence_median"] if picks else None,
        "valence_absolute_warn": maze2.ROLE_ABSOLUTE_WARN,
        "mine_pair": list(maze2.MINE_PAIR),
        "novel_pair": list(maze2.NOVEL_GLYPHS),
        "valence_table": table,
        "picked": {g: by_glyph[g] for g in set(assigned.values()) & set(by_glyph)},
    }


def evaluate(model, tok, mazes, *, maze2, navigator, roles, fmt, batch_size,
             seed, torch):
    """The mine pass and the placebo pass, PAIRED. -> (records, placebo records).

    Both passes roll the same mazes, half by seed parity under that parity's
    colour assignment, and each half gets a FRESH generator seeded the same way
    in both passes -- so the per-turn move-word order is the same stream in the
    two passes and a disagreement between them is the mine glyph and nothing
    else. (E1c-2: list position drives an untrained policy; a paired comparison
    that redrew the orders would be comparing two things at once.)

    `stop_on_parse=False` and `max_new_tokens=48` are B6's fix and are not
    configurable here: a result file whose replies were truncated at the move
    word cannot be read for anything but the move.
    """
    def one_pass(placebo):
        out = []
        for parity in (0, 1):
            half = [m for m in mazes if m.seed % 2 == parity]
            if not half:
                continue
            glyphs = maze2.glyphs_for_seed(parity, **roles)
            if placebo:
                glyphs = maze2.placebo_glyphs(glyphs)
            t0 = time.perf_counter()

            def beat(rounds, active, _t0=t0):
                # The decode window is 48 tokens with no parse-stop, and a
                # navigator that never emits EOS spends all 48 on every
                # decision -- 24x the eval this budgets for. A row per round
                # makes that visible in minutes instead of in hours.
                if rounds == 1 or rounds % 4 == 0:
                    el = time.perf_counter() - _t0
                    say(f"    round {rounds}: {active} episodes still active, "
                        f"{el:.0f}s elapsed ({el / rounds:.1f}s/round)")

            recs, _agg = maze2.rollout(
                model, tok, half, retry=True, glyphs=glyphs, fmt=fmt,
                generator=torch.Generator().manual_seed(seed + parity),
                batch_size=batch_size,
                max_new_tokens=navigator.EVAL_MAX_NEW_TOKENS,
                stop_on_parse=False, progress=beat)
            for r in recs:
                r["mine_glyph"] = glyphs.mine
                r["pass"] = "placebo" if placebo else "mine"
            out.extend(recs)
            say(f"  eval {'placebo' if placebo else 'mine   '} parity {parity} "
                f"({glyphs.mine}): {len(half)} episodes in "
                f"{(time.perf_counter() - t0) / 60:.1f} min")
        out.sort(key=lambda r: (r["n"], r["seed"]))
        for i, r in enumerate(out):
            r["index"] = i
        return out

    return one_pass(False), one_pass(True)


def run_navigator(model, tok, *, n=GRID_N, n_decisions=N_DECISIONS,
                  epochs=EPOCHS, eval_n=EVAL_N, fmt=FORMAT, batch_size=BATCH,
                  eval_batch=EVAL_BATCH, lr=LR, log_every=LOG_EVERY,
                  off_path_share=OFF_PATH_SHARE, out_dir=OUT, shard=SHARD,
                  roles=None, manifest_extra=None, glyph_choice=None,
                  smoke=False):
    """Corpus -> train -> save -> eval -> placebo -> blindness -> gate -> payload.

    The model must be the clean base; this function injects the LoRA. It writes
    the adapters (per epoch and at the end) into `out_dir` and returns the
    payload; the caller writes the JSON, so the tests can run the whole shape on
    a toy model without a result file appearing anywhere.
    """
    import torch

    from calibration import maze2, navigator
    from calibration.lora import inject_lora, save_lora
    from calibration.rl import _EVAL_SEED_RANGE
    from calibration.runner import derive_generator

    roles = dict(roles or {})
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.perf_counter()

    # ---- the corpus ---------------------------------------------------------
    say(f"corpus: {n_decisions} decisions at n={n}, fmt={fmt}, "
        f"off-path share {off_path_share} (M3), mine-blind by construction")
    t0 = time.perf_counter()
    corpus, corpus_stats = navigator.build_navigator_corpus(
        n_decisions, n=n, generator=derive_generator(0, "navigator_corpus"),
        fmt=fmt, off_path_share=off_path_share, roles=roles)
    train_hashes = set(corpus_stats.pop("canonical_hashes"))
    corpus_stats["n_canonical_hashes"] = len(train_hashes)
    say(f"  {corpus_stats['n_decisions']} decisions from "
        f"{corpus_stats['n_mazes']} mazes in {time.perf_counter() - t0:.1f}s: "
        f"on-path {corpus_stats['on_path']} / off-path {corpus_stats['off_path']} "
        f"(share {corpus_stats['off_path_share']:.3f}); labels landing on a mine "
        f"{corpus_stats['label_mine_landing_share']:.4f}; states on a mine "
        f"{corpus_stats['state_on_mine_share']:.4f}; prompt chars mean "
        f"{corpus_stats['prompt_chars_mean']:.0f} max "
        f"{corpus_stats['prompt_chars_max']}")
    say(f"  label move counts {corpus_stats['label_move_counts']}; mine glyphs "
        f"{corpus_stats['mine_glyph_counts']} (the counterbalance, inside one "
        f"navigator)")

    # ---- train --------------------------------------------------------------
    n_params = len(inject_lora(model, target_modules=("q_proj", "v_proj"),
                               r=16, alpha=32))
    steps_per_epoch = math.ceil(len(corpus) / batch_size)
    say(f"LoRA injected: {n_params} trainable tensors, r=16 alpha=32 on q/v; "
        f"{epochs} epochs x {steps_per_epoch} steps = "
        f"{epochs * steps_per_epoch} optimiser steps at batch {batch_size}, "
        f"lr {lr}")

    checkpoints = []

    def on_progress(ev):
        if ev["is_eta_mark"]:
            say(f"  TIME ESTIMATE after {ev['step']} steps: "
                f"{ev['elapsed_seconds'] / ev['step']:.2f}s/step -> "
                f"{_hms(ev['projected_total_seconds'])} for "
                f"{ev['total_steps']} steps "
                f"(remaining {_hms(ev['eta_seconds'])})")
        elif not ev["epoch_end"]:
            say(f"  step {ev['step']}/{ev['total_steps']} epoch {ev['epoch']} "
                f"loss {ev['loss']:.4f} (mean50 {ev['loss_recent_mean']:.4f}) "
                f"grad {ev['grad_norm']:.3f} elapsed "
                f"{_hms(ev['elapsed_seconds'])} eta {_hms(ev['eta_seconds'])}")
        if ev["epoch_end"]:
            path = f"{out_dir}/nav_{shard}_epoch{ev['epoch']}.pt"
            sha = save_lora(model, path)
            checkpoints.append({"epoch": ev["epoch"], "step": ev["step"],
                                "path": path, "sha256": sha,
                                "loss": ev["loss"],
                                "elapsed_seconds": ev["elapsed_seconds"]})
            say(f"  EPOCH {ev['epoch']} done at step {ev['step']}: loss "
                f"{ev['loss']:.4f}, checkpoint {path} sha {sha[:12]} "
                f"(a lease that dies now leaves this)")

    history = navigator.train_navigator(
        model, tok, corpus, epochs=epochs, lr=lr, batch_size=batch_size,
        max_length=navigator.DEFAULT_MAX_LENGTH, log_every=log_every,
        shuffle_generator=derive_generator(0, "navigator_shuffle"),
        on_progress=on_progress)
    history["checkpoints"] = checkpoints
    say(f"training done in {_hms(history['total_seconds'])}: loss "
        f"{history['initial_loss']:.4f} -> {history['final_loss']:.4f}, "
        f"by epoch {[round(v, 4) for v in history['loss_by_epoch']]}, "
        f"longest example {history['longest_example_tokens']} tokens")

    name = "nav_shared" if shard == "nav" else f"nav_{shard}"
    if smoke:
        name += "_smoke"
    adapter_path = f"{out_dir}/{name}.pt"
    adapter_sha = save_lora(model, adapter_path)
    say(f"adapters -> {adapter_path} sha256 {adapter_sha}")

    # ---- the gate's mazes ---------------------------------------------------
    mazes, gen_stats = maze2.generate_set(
        eval_n, n=n, generator=derive_generator(0, "navigator_eval"),
        seed_range=_EVAL_SEED_RANGE, seen=set(train_hashes))
    say(f"eval: {len(mazes)} held-out mazes (rejection "
        f"{gen_stats['rejection_rate']:.3f}), decode window "
        f"{navigator.EVAL_MAX_NEW_TOKENS} tokens with the parse-stop OFF (B6)")
    path_rate = navigator.path_mine_rate(mazes)
    say(f"  the BFS path's own mine rate on these mazes: "
        f"{path_rate['mean']:.3f} per episode (sd {path_rate['sd']:.3f}, se "
        f"{path_rate['se']:.3f}) -- REPORTED, not gated (M4)")

    records, placebo_records = evaluate(
        model, tok, mazes, maze2=maze2, navigator=navigator, roles=roles,
        fmt=fmt, batch_size=eval_batch, seed=4000 + n, torch=torch)

    lengths = navigator.reply_token_lengths(tok, records)
    navigator.reply_token_lengths(tok, placebo_records)
    agg = maze2.aggregate(records)
    placebo_agg = maze2.aggregate(placebo_records)
    blindness = navigator.mine_blindness(records, placebo_records)
    gate = navigator.navigator_gate(agg, blindness)

    say(f"reply lengths: mean {lengths['mean']:.2f} tokens, max "
        f"{lengths['max']}, share at the {lengths['window']}-token cap "
        f"{lengths['share_at_cap']:.4f}"
        + ("  !! replies are hitting the window; the numbers below are partly a "
           "measurement of the window (B6)" if lengths["share_at_cap"] > 0.01
           else ""))
    say("GATE (every input beside its verdict):")
    for line in gate["lines"]:
        say("  " + line)
    say(f"  reported, ungated: single_route_acc on the DAG "
        f"{gate['reported']['single_route_acc_on_dag']} "
        f"({gate['reported']['n_single_route_on_dag']} decisions), off it "
        f"{gate['reported']['single_route_acc_off_dag']} "
        f"({gate['reported']['n_single_route_off_dag']}); the critique's "
        f"off-DAG reference is {gate['reported']['off_dag_reported_bar']}")
    say(f"  reported, ungated: mine_step_rate {blindness['mine_step_rate_mine']} "
        f"vs the BFS path's {path_rate['mean']:.3f}; placebo pass "
        f"{blindness['mine_step_rate_placebo']}")
    say(f"  optimality {gate['reported']['optimality']} capped_rate "
        f"{gate['reported']['capped_rate']} retry_rate "
        f"{gate['reported']['retry_rate']} bare_rate "
        f"{gate['reported']['bare_rate']}")

    payload = {
        "manifest": {
            "experiment": "MZ01_navigator",
            "stage": "Stage 0' -- one shared navigator",
            "shard": shard,
            "format": fmt,
            "n": n,
            "smoke": bool(smoke),
            "n_decisions": len(corpus),
            "epochs": epochs,
            "batch_size": batch_size,
            "eval_batch": eval_batch,
            "lr": lr,
            "off_path_share": off_path_share,
            "eval_n": eval_n,
            "eval_max_new_tokens": navigator.EVAL_MAX_NEW_TOKENS,
            "stop_on_parse": False,
            "lora": {"r": 16, "alpha": 32,
                     "target_modules": ["q_proj", "v_proj"],
                     "n_trainable_tensors": n_params},
            "adapter_path": adapter_path,
            "adapter_sha256": adapter_sha,
            "checkpoints": checkpoints,
            "seed_range_train": corpus_stats["seed_range"],
            "seed_range_eval": list(_EVAL_SEED_RANGE),
            "gate_bars": gate["bars"],
            "blindness_bar": navigator.BLINDNESS_AGREEMENT_BAR,
            "scoring": {"step": maze2.SCORE_STEP, "mine": maze2.SCORE_MINE,
                        "goal": maze2.SCORE_GOAL},
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_minutes": round((time.perf_counter() - t_start) / 60, 2),
            **(manifest_extra or {}),
        },
        "glyphs": glyph_choice,
        "corpus": corpus_stats,
        "training": history,
        "aggregates": agg,
        "placebo_aggregates": placebo_agg,
        "blindness": blindness,
        "gate": gate,
        "reply_lengths": lengths,
        "path_mine_rate": path_rate,
        "eval_generator": gen_stats,
        "records": records,
        "placebo_records": placebo_records,
    }
    return payload


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from calibration import maze2
    from calibration.chat import chat_kwargs
    from calibration.instruments import glyph_valence_table
    from calibration.runner import git_sha

    maze2.check_format(FORMAT)
    if MODEL_KEY not in MODELS:
        raise SystemExit(f"MZ01_MODEL must be one of {sorted(MODELS)}, got {MODEL_KEY!r}")
    if not REVISION:
        raise SystemExit(
            "MZ01_REVISION is required: this run gates the branch, so the "
            "result file has to name the exact weights it gated it with.")
    model_id = MODELS[MODEL_KEY]

    n_decisions, eval_n = N_DECISIONS, EVAL_N
    if SMOKE_ONLY:
        n_decisions, eval_n = 200, 16
    say(f"MZ01 navigator: {model_id} @ {REVISION}; format {FORMAT}; n={GRID_N}; "
        f"{n_decisions} decisions x {EPOCHS} epochs; eval {eval_n} mazes; "
        f"shard {SHARD}; smoke_only={SMOKE_ONLY}")

    tok = AutoTokenizer.from_pretrained(model_id, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=REVISION, dtype=torch.bfloat16).to("cuda")
    model.eval()
    # C1. Qwen3-14B is a HYBRID checkpoint: its template opens a `<think>` block
    # under `add_generation_prompt=True`, so the first generated token is a
    # reasoning trace and every parsed move would be unparsed. `apply_chat`
    # closes it; this says which position the numbers were read at, and both the
    # corpus and the eval go through the same function.
    chat_kw = chat_kwargs(tok)
    say(f"model resident; chat_kwargs {chat_kw or '{} (plain template)'}")

    bar = maze2.ROLE_VALENCE_BAR if not VALENCE_BAR else float(VALENCE_BAR)
    assigned, glyph_choice = measure_roles(model, tok, maze2,
                                           glyph_valence_table, FORMAT, bar)
    say(f"measured roles {glyph_choice['measured_roles']} (ASCII fixes the rest);"
        f" on_mine <- {assigned['on_mine']}, wall {assigned['wall']}, goal "
        f"{assigned['goal']}; valence median over the bank "
        f"{glyph_choice['valence_median']}")
    roles = {"wall": assigned["wall"], "goal": assigned["goal"],
             "on_mine": assigned["on_mine"]}

    payload = run_navigator(
        model, tok, n=GRID_N, n_decisions=n_decisions, epochs=EPOCHS,
        eval_n=eval_n, fmt=FORMAT, batch_size=BATCH, eval_batch=EVAL_BATCH,
        lr=LR, log_every=LOG_EVERY, off_path_share=OFF_PATH_SHARE, out_dir=OUT,
        shard=SHARD, roles=roles, smoke=SMOKE_ONLY, glyph_choice=glyph_choice,
        manifest_extra={
            "model_key": MODEL_KEY, "model_id": model_id, "revision": REVISION,
            "git_sha": git_sha(), "torch_version": torch.__version__,
            "device": "cuda", "dtype": "bfloat16", "decoding": "greedy",
            "retry": True, "chat_kwargs": chat_kw,
            "chat_enable_thinking": chat_kw.get("enable_thinking"),
            "role_valence_bar": bar,
        })

    # ---- the contracts, asserted where they are cheap to fix ----------------
    missing = [k for k in PAYLOAD_KEYS if k not in payload]
    assert not missing, f"payload is missing {missing}"
    for r in payload["records"] + payload["placebo_records"]:
        gaps = [k for k in RECORD_KEYS if k not in r]
        assert not gaps, f"record is missing {gaps}"
        assert (len(r["landings"]) == len(r["moves"]) == len(r["positions"])
                == len(r["orders"]) == len(r["on_mine"])
                == len(r["reply_len_tokens"]) == r["steps"])
        assert sum(r["retried"]) == r["retries"]
        assert r["fmt"] == FORMAT and r["stop_on_parse"] is False
    o = payload["aggregates"]["overall"]
    if o["unparsed_rate"] > 0.5:
        raise AssertionError(
            f"unparsed_rate {o['unparsed_rate']:.3f} after training: the model "
            f"is not emitting a move word at all. chat_kwargs={chat_kw}; if "
            f"that is empty on a hybrid checkpoint the prompt is opening a "
            f"<think> block and every number here is a measurement of the "
            f"thinking channel. First replies: "
            + repr([r["replies"][0] for r in payload["records"][:3]]))

    os.makedirs(OUT, exist_ok=True)
    name = "nav_shared" if SHARD == "nav" else f"nav_{SHARD}"
    if SMOKE_ONLY:
        name += "_smoke"
    path = f"{OUT}/{name}.json"
    with open(path, "w") as fh:
        json.dump(payload, fh)
    say(f"WROTE {path}")
    # The observation itself, once, in the log -- rebuilt from the record the
    # same way a later corpus builder will rebuild it, so a format or a template
    # change is visible rather than inferred from an aggregate.
    r0 = payload["records"][0]
    m0 = maze2.from_layout(r0["layout"], labels=False)
    g0 = maze2.Glyphs(**{k: r0["glyphs"][k] for k in maze2.Glyphs().as_dict()})
    say("the observation, decision 0 of episode 0:\n"
        + maze2.maze_prompt(
            maze2.render(m0, g0, tuple(r0["positions"][0]), fmt=FORMAT),
            g0.goal, r0["orders"][0], agent=g0.agent, fmt=FORMAT,
            pos=tuple(r0["positions"][0]), goal_pos=m0.goal)
        + "\nfirst replies: "
        + repr([r["replies"][0] for r in payload["records"][:3]]))
    if SMOKE_ONLY:
        say("MZ01 NAV DONE (MZ01_SMOKE_ONLY=1; no full run)")
        return
    say(f"MZ01 NAV DONE shard={SHARD} gate={'PASS' if payload['gate']['passed'] else 'FAIL'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say("MZ01 ERROR:\n" + traceback.format_exc())
