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
     "The same word-order stream" is now TRUE rather than intended: the orders
     are derived per (run seed, maze seed, decision index) under `maze2` D31,
     because the shared draw the roll-out used desynchronised every remaining
     episode's orders the moment one episode diverged.

  7. THE 0.98 BAR IS READ AGAINST A CEILING, NOT AGAINST 1.0. A third pass rolls
     `MZ01_SELF_CONTROL_N` of the same mazes with the same glyphs and the same
     orders at a DIFFERENT eval batch size, so its only difference from the mine
     pass is batch composition and pad width -- which greedy argmax over bf16
     logits is not invariant to. Its agreement with the mine pass is what
     "identical decisions" can mean on this hardware. REPORTED, never gated: the
     pre-commitment names five criteria and this is not one of them.

  8. THE RESULT FILE IS WRITTEN BEFORE THE CONTRACTS ARE ASSERTED. They used to
     run first, so a 2-3 h lease that tripped one left the disk empty and the
     status file an exception -- and the likeliest trip (every reply in the
     thinking channel) is the one whose evidence is in the payload.

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
  MZ01_NAV_LABEL       "first" (default) or "uniform" -- how a cell with more
                       than one BFS-optimal move is labelled (navigator D36).
                       "first" is the pre-fourteenth corpus to the byte;
                       "uniform" draws the label over all of them, which is the
                       on-policy variance at a fork that the RL's epsilon
                       otherwise has to buy off-policy. It is a DIFFERENT
                       navigator: give it its own MZ01_SHARD.
  MZ01_BATCH           SFT batch size (default 8).
  MZ01_EVAL_BATCH      decode batch size (default 32).
  MZ01_LR              SFT learning rate (default 1e-4).
  MZ01_LOG_EVERY       heartbeat every N optimiser steps (default 50).
  MZ01_SELF_CONTROL_N  mazes rolled a third time, same glyphs, different eval
                       batch, for the self-agreement ceiling (default 64; 0
                       disables). Costs about a quarter of one eval pass.
  MZ01_VALENCE_BAR     the |centred valence| a MEASURED role glyph may carry
                       (`maze2.ROLE_VALENCE_BAR`). ASCII measures `on_mine`
                       alone; wall and goal are fixed marks.

Outputs, under `experiments/v2/MZ01_maze_world/results/navigator/`:
  nav_shared.pt         the adapters, with their sha256 in the manifest
  nav_shared.json       manifest (pre-flight included), corpus stats, training
                        history, gate, blindness, the self-agreement control,
                        and every episode record of all three passes
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
# D36. Checked against `navigator.LABEL_RULES` in `main` before the model is
# loaded: an unknown rule that fell through to the default would spend a lease
# training "nav48u" on nav48's corpus.
NAV_LABEL = os.environ.get("MZ01_NAV_LABEL", "first").strip() or "first"
BATCH = int(os.environ.get("MZ01_BATCH", "8"))
EVAL_BATCH = int(os.environ.get("MZ01_EVAL_BATCH", "32"))
LR = float(os.environ.get("MZ01_LR", "1e-4"))
LOG_EVERY = int(os.environ.get("MZ01_LOG_EVERY", "50"))
VALENCE_BAR = os.environ.get("MZ01_VALENCE_BAR", "").strip()
# The SELF-AGREEMENT CONTROL (critic 3). `decision_agreement >= 0.98` is only
# readable against what the navigator scores against ITSELF: greedy decoding in
# batched bf16 is not bit-reproducible across batch COMPOSITIONS, so two passes
# whose prompts differ in one glyph can disagree for reasons that are arithmetic
# rather than perceptual. This many of the same eval mazes are rolled a THIRD
# time -- same glyphs, same word orders, a different eval batch size -- and the
# agreement of that pass with the mine pass is the ceiling any blindness number
# has to be read against. REPORTED, never gated: the pre-commitment names five
# criteria and this is not one of them. 0 disables it.
SELF_CONTROL_N = int(os.environ.get("MZ01_SELF_CONTROL_N", "64"))

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
               "max_new_tokens", "stop_on_parse", "reply_len_tokens",
               "order_seed")

PAYLOAD_KEYS = ("manifest", "glyphs", "corpus", "training", "aggregates",
                "placebo_aggregates", "blindness", "self_control", "gate",
                "reply_lengths", "path_mine_rate", "records", "placebo_records",
                "self_records")


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


def preflight(tok, corpus, *, maze2, navigator, chat, fmt, max_length):
    """THE TRAINED STRING IS THE EVALUATED STRING. Checked before the lease.

    Priority one of critic 3, and the only place it can be checked is on the
    resident tokenizer -- the fixtures cannot stand in for it. Four properties,
    each of which silently makes the gate meaningless if it fails:

      1. The corpus prompt rebuilt the way `rollout` builds it -- render the
         layout under the stored glyph table, `maze_prompt` with the stored
         order and position -- is BYTE-IDENTICAL to the string the trainer was
         handed. A drift here trains on one question and reads the answer to
         another.
      2. The chat template is applied by the same function on both sides
         (`chat.apply_chat`), so the templated strings are equal by
         construction; what is NOT equal by construction is their TOKENISATION.
         `sft.build_batch` encodes with `add_special_tokens=False` and
         `maze2._greedy` encodes with the tokenizer's default. On Qwen the two
         agree; on a tokenizer that prepends a BOS they do not, and the model
         would then be trained at one offset and read at another.
      3. `train_navigator` appends `tok.eos_token` as TEXT and `_greedy` stops
         on `tok.eos_token_id`. If the string does not encode to exactly that
         one id the navigator never learns to stop, the 48-token window is spent
         on every decision, and the eval costs about 24x what it is budgeted
         for. (The repo's own CharTokenizer fixture failed this until today.)
      4. The longest templated example fits `max_length` with room, so
         `build_batch` truncates no completion away.
    """
    item = corpus[0]
    maze = maze2.from_layout(item["layout"], seed=item["seed"], labels=False)
    glyphs = maze2.format_glyphs(maze2.Glyphs(**item["glyphs"]), fmt)
    pos = tuple(item["position"])
    rebuilt = maze2.maze_prompt(
        maze2.render(maze, glyphs, pos, fmt=fmt), glyphs.goal, item["order"],
        agent=glyphs.agent, fmt=fmt, pos=pos, goal_pos=maze.goal)
    if rebuilt != item["prompt"]:
        raise AssertionError(
            "the prompt the roll-out would build is not the prompt the corpus "
            f"trains on:\n--- corpus ---\n{item['prompt']!r}\n--- rollout "
            f"---\n{rebuilt!r}")

    templated = chat.apply_chat(
        tok, [{"role": "user", "content": item["prompt"]}],
        add_generation_prompt=True, tokenize=False)
    train_ids = tok(templated, add_special_tokens=False)["input_ids"]
    # EXPLICIT True rather than the bare call `_greedy` makes: the property is
    # "the two encodings agree", and reading it off a default would make this
    # check a test of the default instead of of the tokenizer.
    eval_ids = tok(templated, add_special_tokens=True)["input_ids"]
    if list(train_ids) != list(eval_ids):
        raise AssertionError(
            f"the trainer encodes the prompt as {len(train_ids)} tokens "
            f"(add_special_tokens=False, sft.build_batch) and the eval as "
            f"{len(eval_ids)} (maze2._greedy's default). The navigator would be "
            f"trained at one offset and read at another. First difference at "
            f"index {next(i for i, (a, b) in enumerate(zip(list(train_ids) + [None], list(eval_ids) + [None])) if a != b)}")

    eos_ids = tok(tok.eos_token, add_special_tokens=False)["input_ids"]
    if list(eos_ids) != [tok.eos_token_id]:
        raise AssertionError(
            f"tok({tok.eos_token!r}) encodes to {list(eos_ids)} but _greedy "
            f"stops on {tok.eos_token_id}: appending the EOS TEXT to every "
            f"completion would not teach the model to emit the token the decode "
            f"loop stops on, and every decision would spend the whole "
            f"{navigator.EVAL_MAX_NEW_TOKENS}-token window.")

    longest = max(len(tok(chat.apply_chat(
        tok, [{"role": "user", "content": it["prompt"]}],
        add_generation_prompt=True, tokenize=False),
        add_special_tokens=False)["input_ids"])
        for it in corpus[:64])
    out = {"prompt_rebuild_identical": True,
           "prompt_tokens_train_eq_eval": len(train_ids),
           "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id,
           "longest_templated_prompt_tokens_first64": longest,
           "max_length": max_length}
    say(f"PRE-FLIGHT: the roll-out's prompt IS the corpus prompt; templated "
        f"prompt {len(train_ids)} tokens with and without add_special_tokens; "
        f"EOS {tok.eos_token!r} -> [{tok.eos_token_id}]; longest of the first "
        f"64 templated prompts {longest} tokens against max_length "
        f"{max_length}")
    return out


def evaluate(model, tok, mazes, *, maze2, navigator, roles, fmt, batch_size,
             seed, torch, self_control_n=0):
    """The mine pass, the placebo pass and the SELF pass. -> three record lists.

    Both graded passes roll the same mazes, half by seed parity under that
    parity's colour assignment, and the per-turn move-word order is DERIVED per
    (run seed, maze seed, decision index) rather than drawn from a shared stream
    (`maze2` D31). That is the repair critic 3 forced: under the shared draw the
    permutation an episode receives depends on how many OTHER episodes were
    still active, so one divergence desynchronises every remaining episode's
    word order and E1c-2's order effect then manufactures disagreements the
    glyph did not cause.

    THE SELF PASS is the control the 0.98 bar cannot be read without. It rolls
    the first `self_control_n` of the SAME mazes with the SAME glyphs and the
    SAME derived orders, changing one thing the model is not supposed to see:
    the eval BATCH SIZE, and with it the batch composition and the pad width.
    Greedy argmax over bf16 logits is not invariant to those, so the agreement
    of this pass with the mine pass is the arithmetic ceiling on what "blind"
    can read. It is REPORTED, never gated.

    `stop_on_parse=False` and `max_new_tokens=48` are B6's fix and are not
    configurable here: a result file whose replies were truncated at the move
    word cannot be read for anything but the move.
    """
    def one_pass(placebo, pool=None, batch=None, tag=None):
        pool = mazes if pool is None else pool
        label = tag or ("placebo" if placebo else "mine")
        out = []
        for parity in (0, 1):
            half = [m for m in pool if m.seed % 2 == parity]
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
                order_seed=seed + parity,
                batch_size=batch or batch_size,
                max_new_tokens=navigator.EVAL_MAX_NEW_TOKENS,
                stop_on_parse=False, progress=beat)
            for r in recs:
                r["mine_glyph"] = glyphs.mine
                r["pass"] = label
            out.extend(recs)
            say(f"  eval {label:<7} parity {parity} "
                f"({glyphs.mine}): {len(half)} episodes in "
                f"{(time.perf_counter() - t0) / 60:.1f} min")
        out.sort(key=lambda r: (r["n"], r["seed"]))
        for i, r in enumerate(out):
            r["index"] = i
        return out

    records = one_pass(False)
    placebo_records = one_pass(True)
    self_records = []
    if self_control_n:
        subset = sorted(mazes, key=lambda m: (m.n, m.seed))[:self_control_n]
        self_batch = max(1, (batch_size // 2) or 1)
        if self_batch == batch_size:
            self_batch = max(1, batch_size - 1)
        say(f"  self-agreement control: {len(subset)} of the same mazes, same "
            f"glyphs, same derived word orders, eval batch {self_batch} against "
            f"{batch_size} -- the ceiling the 0.98 blindness bar is read against")
        self_records = one_pass(False, pool=subset, batch=self_batch,
                                tag="self")
    return records, placebo_records, self_records


def run_navigator(model, tok, *, n=GRID_N, n_decisions=N_DECISIONS,
                  epochs=EPOCHS, eval_n=EVAL_N, fmt=FORMAT, batch_size=BATCH,
                  eval_batch=EVAL_BATCH, lr=LR, log_every=LOG_EVERY,
                  off_path_share=OFF_PATH_SHARE, label=NAV_LABEL, out_dir=OUT,
                  shard=SHARD, roles=None, manifest_extra=None,
                  glyph_choice=None, smoke=False, self_control_n=None):
    """Corpus -> train -> save -> eval -> placebo -> blindness -> gate -> payload.

    The model must be the clean base; this function injects the LoRA. It writes
    the adapters (per epoch and at the end) into `out_dir` and returns the
    payload; the caller writes the JSON, so the tests can run the whole shape on
    a toy model without a result file appearing anywhere.
    """
    import torch

    from calibration import chat, maze2, navigator
    from calibration.lora import inject_lora, save_lora
    from calibration.rl import _EVAL_SEED_RANGE
    from calibration.runner import derive_generator

    roles = dict(roles or {})
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.perf_counter()

    # ---- the corpus ---------------------------------------------------------
    say(f"corpus: {n_decisions} decisions at n={n}, fmt={fmt}, "
        f"off-path share {off_path_share} (M3), label rule {label} (D36), "
        f"mine-blind by construction")
    t0 = time.perf_counter()
    corpus, corpus_stats = navigator.build_navigator_corpus(
        n_decisions, n=n, generator=derive_generator(0, "navigator_corpus"),
        fmt=fmt, off_path_share=off_path_share, label=label, roles=roles)
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
    say(f"  label rule {corpus_stats['label_rule']} (D36): "
        f"{corpus_stats['multi_optimal_share']:.4f} of labelled cells have >= 2 "
        f"optimal moves, mean {corpus_stats['optimal_moves_mean']:.3f} "
        f"{corpus_stats['optimal_moves_histogram']}; label entropy "
        f"{corpus_stats['label_entropy_nats']:.4f} nats, the RULE's own "
        f"{corpus_stats['label_rule_entropy_nats']:.4f} nats (0 under 'first' by "
        f"construction -- that is the point mass this pass is about)")

    # ---- the pre-flight: the trained string IS the evaluated string ---------
    pre = preflight(tok, corpus, maze2=maze2, navigator=navigator, chat=chat,
                    fmt=fmt, max_length=navigator.DEFAULT_MAX_LENGTH)

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

    control_n = SELF_CONTROL_N if self_control_n is None else self_control_n
    records, placebo_records, self_records = evaluate(
        model, tok, mazes, maze2=maze2, navigator=navigator, roles=roles,
        fmt=fmt, batch_size=eval_batch, seed=4000 + n, torch=torch,
        self_control_n=min(control_n, len(mazes)))

    lengths = navigator.reply_token_lengths(tok, records)
    navigator.reply_token_lengths(tok, placebo_records)
    navigator.reply_token_lengths(tok, self_records)
    agg = maze2.aggregate(records)
    placebo_agg = maze2.aggregate(placebo_records)
    blindness = navigator.mine_blindness(records, placebo_records)
    # D37, ungated. Under MZ01_NAV_LABEL=uniform the policy is near-tied at
    # every fork BY CONSTRUCTION, and at a near-tie any change to the prompt --
    # the mine's own colour included -- can flip an argmax the navigator has no
    # preference in. These say whether that is what happened; the gate below
    # still reads `decision_agreement` and nothing else.
    fork_share = blindness["fork_decision_share"]
    fork_share = "n/a" if fork_share is None else f"{fork_share:.4f}"
    say(f"  divergences by cell (D37, ungated): "
        f"{blindness['n_diverged_at_fork']} of "
        f"{blindness['n_episodes_diverged']} first parted at a cell with >= 2 "
        f"optimal moves, {blindness['n_diverged_tie_break']} of those a "
        f"TIE-BREAK (both moves optimal, both landing the same way on a mine); "
        f"agreement without those episodes "
        f"{blindness['decision_agreement_ex_tie_break']} over "
        f"{blindness['n_decisions_ex_tie_break']} decisions. "
        f"{fork_share} of this pass's decisions were taken at a fork at all "
        f"({blindness['n_fork_decisions']} of "
        f"{blindness['n_decisions_placed']}).")
    # The ceiling. Same mazes, same glyphs, same word orders, a different eval
    # batch: whatever this pass does NOT agree with is arithmetic, not
    # perception, and no blindness reading can be tighter than it.
    self_control = None
    if self_records:
        keys = {r["layout"] for r in self_records}
        self_control = navigator.mine_blindness(
            [r for r in records if r["layout"] in keys], self_records)
        self_control["control"] = (
            "the navigator against ITSELF: same mazes, same glyphs, same "
            "derived word orders, a different eval batch size")
        self_control["n_mazes"] = len(self_records)
        say(f"  SELF-AGREEMENT (the ceiling, ungated): "
            f"{self_control['decision_agreement']} over "
            f"{self_control['n_decisions']} decisions on "
            f"{self_control['n_episodes']} episodes; "
            f"{self_control['n_episodes_diverged']} diverged, "
            f"{self_control['n_diverged_tie_break']} of them at a tie-break "
            f"(D37). A blindness reading below this is not evidence about the "
            f"glyph.")
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
            # D36. In the manifest as well as in `corpus`, because the manifest
            # is what a later launch line and every table is keyed off, and a
            # navigator's two rules produce two different adapters under names
            # that differ by one character.
            "label_rule": label,
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
            "self_control_n": min(control_n, len(mazes)),
            "preflight": pre,
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
        "self_control": self_control,
        "gate": gate,
        "reply_lengths": lengths,
        "path_mine_rate": path_rate,
        "eval_generator": gen_stats,
        "records": records,
        "placebo_records": placebo_records,
        "self_records": self_records,
    }
    return payload


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from calibration import maze2
    from calibration.chat import chat_kwargs
    from calibration.instruments import glyph_valence_table
    from calibration.navigator import LABEL_RULES
    from calibration.runner import git_sha

    maze2.check_format(FORMAT)
    if MODEL_KEY not in MODELS:
        raise SystemExit(f"MZ01_MODEL must be one of {sorted(MODELS)}, got {MODEL_KEY!r}")
    if NAV_LABEL not in LABEL_RULES:
        raise SystemExit(
            f"MZ01_NAV_LABEL must be one of {list(LABEL_RULES)}, got "
            f"{NAV_LABEL!r}: a rule that fell through to the default would "
            f"train one navigator and record another (D36).")
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
        f"shard {SHARD}; label rule {NAV_LABEL}; smoke_only={SMOKE_ONLY}")

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
        lr=LR, log_every=LOG_EVERY, off_path_share=OFF_PATH_SHARE,
        label=NAV_LABEL, out_dir=OUT,
        shard=SHARD, roles=roles, smoke=SMOKE_ONLY, glyph_choice=glyph_choice,
        manifest_extra={
            "model_key": MODEL_KEY, "model_id": model_id, "revision": REVISION,
            "git_sha": git_sha(), "torch_version": torch.__version__,
            "device": "cuda", "dtype": "bfloat16", "decoding": "greedy",
            "retry": True, "chat_kwargs": chat_kw,
            "chat_enable_thinking": chat_kw.get("enable_thinking"),
            "role_valence_bar": bar,
        })

    # ---- THE RESULT FILE IS WRITTEN FIRST ----------------------------------
    # Critic 3: the contracts below used to run BEFORE the write, so a 2-3 h
    # lease that tripped one of them left the status file an exception and the
    # disk nothing at all -- and the checks most likely to trip are exactly the
    # ones whose evidence is IN the payload (a template flag that put every
    # reply in the thinking channel, a record contract that drifted). The file
    # lands first; the assertions still fire, and now they fire beside the data
    # that explains them.
    os.makedirs(OUT, exist_ok=True)
    name = "nav_shared" if SHARD == "nav" else f"nav_{SHARD}"
    if SMOKE_ONLY:
        name += "_smoke"
    path = f"{OUT}/{name}.json"
    with open(path, "w") as fh:
        json.dump(payload, fh)
    say(f"WROTE {path}")

    # ---- the contracts, asserted where they are cheap to fix ----------------
    missing = [k for k in PAYLOAD_KEYS if k not in payload]
    assert not missing, f"payload is missing {missing}"
    for r in (payload["records"] + payload["placebo_records"]
              + payload["self_records"]):
        gaps = [k for k in RECORD_KEYS if k not in r]
        assert not gaps, f"record is missing {gaps}"
        assert (len(r["landings"]) == len(r["moves"]) == len(r["positions"])
                == len(r["orders"]) == len(r["on_mine"])
                == len(r["reply_len_tokens"]) == r["steps"])
        assert sum(r["retried"]) == r["retries"]
        assert r["fmt"] == FORMAT and r["stop_on_parse"] is False
        assert r["order_seed"] is not None, (
            "the paired passes must derive their word orders per episode "
            "(maze2 D31) or a single divergence desynchronises the rest")
    o = payload["aggregates"]["overall"]
    if o["unparsed_rate"] > 0.5:
        raise AssertionError(
            f"unparsed_rate {o['unparsed_rate']:.3f} after training: the model "
            f"is not emitting a move word at all. chat_kwargs={chat_kw}; if "
            f"that is empty on a hybrid checkpoint the prompt is opening a "
            f"<think> block and every number here is a measurement of the "
            f"thinking channel. The payload is already on disk at "
            f"{path}. First replies: "
            + repr([r["replies"][0] for r in payload["records"][:3]]))
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
    say(f"MZ01 NAV DONE shard={SHARD} label={NAV_LABEL} "
        f"gate={'PASS' if payload['gate']['passed'] else 'FAIL'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say("MZ01 ERROR:\n" + traceback.format_exc())
