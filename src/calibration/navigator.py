"""Stage 0' -- the ONE SHARED NAVIGATOR, and the gate that decides the branch.

WHY THIS FILE EXISTS

MZ01's Stage 0 asked whether the untrained checkpoint can walk the maze one word
at a time. It cannot. The emoji pilot of 2026-08-28 came back at solve 0.02,
invalid 0.67-0.72 and single_route_acc 0.14-0.19 on both models at every size --
every number on the WRONG SIDE of the uniform-random floor measured on the same
mazes (invalid 0.406, unique-correct 0.256). The SPEC's stop rule fired and no
organism was built. The user's decision took SPEC A8's ladder to rung (ii),
ASCII rendering, and `maze2` D29 implemented it.

That leaves the world's second problem untouched: even at rung (ii) the
navigation competence is a property of the BASE model, and every organism in
this design is supposed to differ from every other one in its training, not in
whether it can find the goal. So the navigator is MADE. One model, fine-tuned by
imitation of the exact BFS shortest move, is the shared base every later cell
starts from -- and the numbers below are a gate on a TRAINED model, which is a
different and differently-motivated criterion from the SPEC's Stage 0 on an
untrained one (PLAN critique M6, stated here so the two are never confused).

THE FOUR DECISIONS THIS FILE IMPLEMENTS, AND THE FINDINGS BEHIND THEM

M2 -- ONE SHARED NAVIGATOR, NOT ONE PER SEED. A per-seed navigator makes D, A,
    B and C differ in their base in every seed, so a "6 of 8 seeds" rule mixes
    navigator variance into every RL contrast; and "a failing seed stops there"
    is a hand-exclusion, which this programme's standards forbid. One navigator
    removes the confound by construction. The counterbalance survives because
    the CORPUS carries both mine colours: each maze is rendered under its own
    seed parity's assignment (`maze2.glyphs_for_seed`), so the shared navigator
    sees blue and purple mines in equal measure rather than one navigator per
    colour seeing one each.

M3 -- OFF-PATH STATES, OR THE GATE MEASURES WHEN THE FIRST MISTAKE HAPPENED.
    Imitation on shortest paths alone visits only on-DAG cells, so the navigator
    is never trained to recover from its own error. C5's mechanism then
    dominates every number in the gate: greedy decoding is deterministic and a
    blocked move does not move, so the first mistake is an ABSORBING STATE and
    `invalid_rate` / `single_route_acc` become "how early did it first err".
    `off_path_share` of the corpus is therefore states drawn from OFF the
    shortest-path DAG -- the same set `maze2.single_route_split` calls "off it"
    -- each labelled with the first move of the BFS shortest path from THERE, so
    recovery is trained rather than hoped for.

M4 -- BLINDNESS IS A PAIRED PLACEBO, NOT A RATE MATCH. The original rule was
    "mine_step_rate within +-0.05 of the shortest path's own rate". Measured on
    the n=9 eval distribution, following one BFS path lands on 1.016 mines per
    episode with sd 1.205, so at 256 mazes the standard error of that mean is
    0.075 and a +-0.05 band is 0.66 sd: a genuinely blind navigator fails such a
    gate about half the time. And a rate match is not a blindness test anyway --
    a navigator that mildly avoids mines and mildly errs matches the rate.
    `mine_blindness` rolls the SAME mazes twice, once with the mine glyph and
    once with `maze2.placebo_glyphs`, and asks whether the decisions are
    IDENTICAL. Under greedy decoding that comparison has no sampling noise at
    all. The rate match is still reported, beside it, ungated.

B6 -- THE EVAL WINDOW MUST NOT TRUNCATE A REPLY. `maze2._greedy` stopped as
    soon as every row parsed a move, which is one token for "up". No remark
    could survive that, and worse: the old world has a recorded case where the
    truncation MANUFACTURED a contingency of +0.87 (the same organism read
    +0.1504 at a 64-token window). The navigator has no remark to lose, but it
    is the base every organism that does have one inherits, so the eval path is
    fixed here rather than later: `max_new_tokens=48`, `stop_on_parse=False`
    (maze2 D30), and `reply_len_tokens` recorded per decision so the window's
    adequacy is a measured number in the result file rather than a claim.

WHAT IS NOT DECIDED HERE

The gate's four competence numbers are the SPEC's own (solve >= 0.6, invalid
<= 0.1, single_route_acc >= 0.8, unparsed <= 0.05) and are not reopened.
`single_route_acc_off_dag` -- which the critique wanted GATED at 0.6 -- is
computed and printed as a REPORTED number, because the pre-commitment for this
lease names four criteria plus blindness and adding a fifth after the fact is
the move this programme's standards exist to prevent. It is in every result
file; if a later pass wants to gate on it, it gates on it in a document first.

THE CORPUS IS TEXT, THE TRAINER IS WHAT KNOWS THE MODEL

`build_navigator_corpus` takes no tokenizer and produces the OBSERVATION -- the
same `maze2.maze_prompt` string a roll-out would show. `train_navigator` is what
wraps it in the resident model's chat template, through the same
`calibration.chat.apply_chat` that `maze2.rollout` uses, so the string the
navigator is trained on is byte-identical to the string it is evaluated on. An
SFT corpus of raw prompts against a chat-templated eval is a silent mismatch
that would read as "the navigator did not learn", and on a hybrid checkpoint
`apply_chat` is also what keeps the prompt out of an open `<think>` block (C1).
"""

from __future__ import annotations

import math
import time
from collections import Counter

import torch

from .capture import MOVE_WORDS, random_move_orders
from .chat import apply_chat
from .maze2 import (DELTAS, FMT_ASCII, bfs_dist, check_format, format_glyphs,
                    generate_set, glyphs_for_seed, match_targets, maze_prompt,
                    path_moves, render, shortest_dag, shortest_path, step)
from .rl import _TRAIN_SEED_RANGE
from .sft import train_sft

__all__ = [
    "NAV_GATE", "BLINDNESS_AGREEMENT_BAR", "BLINDNESS_TV_BAR",
    "OFF_DAG_REPORTED_BAR", "DEFAULT_OFF_PATH_SHARE", "EVAL_MAX_NEW_TOKENS",
    "DEFAULT_MAX_LENGTH", "build_navigator_corpus", "train_navigator",
    "reply_token_lengths", "mine_blindness", "navigator_gate",
]


# ---------------------------------------------------------------- constants

# The SPEC's Stage-0 numbers (A11's fourth criterion included), applied here to
# a TRAINED model. `(direction, bar)`: "min" is a floor, "max" is a ceiling.
NAV_GATE = {
    "solve_rate": ("min", 0.60),
    "invalid_rate": ("max", 0.10),
    "single_route_acc": ("min", 0.80),
    "unparsed_rate": ("max", 0.05),
}

# M4. Paired, greedy, on the same mazes: identical decisions or it is not blind.
BLINDNESS_AGREEMENT_BAR = 0.98
# Reported beside it. Note the arithmetic: on paired decisions a disagreement
# share of d moves each of the four move marginals by at most d, so the total
# variation distance is at most d and the 0.02 bar is IMPLIED by the 0.98 one
# whenever the two passes have the same number of decisions. It is computed
# anyway, because they do not have to.
BLINDNESS_TV_BAR = 0.02

# Reported, NOT gated. See "WHAT IS NOT DECIDED HERE" above.
OFF_DAG_REPORTED_BAR = 0.60

DEFAULT_OFF_PATH_SHARE = 0.30

# B6. Big enough to hold "<move>. <remark>" whole; paired with
# `stop_on_parse=False` so nothing stops the decode at the move word.
EVAL_MAX_NEW_TOKENS = 48

# The ASCII n=9 prompt is ~322 characters, ~150-200 tokens, plus the chat
# template's wrapper. 512 leaves better than 2x headroom, and `train_navigator`
# RAISES rather than letting `sft.build_batch` truncate a completion away.
DEFAULT_MAX_LENGTH = 512


# ---------------------------------------------------------------- the corpus

def _off_path_cells(maze, dag):
    """Open cells that are reachable, can reach the goal, and are off the DAG.

    All three conditions matter. "Reachable" is M3's word -- a state the agent
    could actually have blundered into; "can reach the goal" is what makes a BFS
    label exist at all; "off the DAG" is the whole point, and the DAG rather
    than one chosen path because a cell on a TIED shortest path is on-path in
    every sense that matters and labelling it as recovery would be a lie about
    what the example teaches.
    """
    from_start = bfs_dist(maze, maze.start)
    to_goal = bfs_dist(maze, maze.goal)
    return [c for c in from_start
            if c in to_goal and c not in dag and c != maze.goal]


def _maze_budget(n_decisions, per_maze, over_generate):
    """How many mazes to ask `generate_set` for, ROUNDED UP TO A MULTIPLE OF 8.

    Not tidiness. The requested joint is 50% mined x four detour bins, so eight
    is the smallest count whose marginals are reachable with whole mazes: at ten
    the five mined mazes cannot split evenly over four bins, one bin comes out
    at 0.400 against a requested 0.250, and `generate_set`'s own
    `assert_distribution` raises. A corpus builder that asked for ten would fail
    on the sampler's correctness check rather than on anything about itself.
    """
    want = max(1, int(math.ceil(n_decisions / per_maze * over_generate)))
    return 8 * int(math.ceil(want / 8))


def build_navigator_corpus(n_decisions, *, n, generator, fmt=FMT_ASCII,
                           off_path_share=DEFAULT_OFF_PATH_SHARE,
                           seed_range=_TRAIN_SEED_RANGE, roles=None,
                           counterbalance=True, seen=None, over_generate=1.10,
                           max_rounds=4, **generate_kw):
    """(items, stats) -- imitation examples for the shared navigator.

    One item per DECISION:

        prompt      = maze_prompt(render(maze, glyphs, position), goal, order)
        completion  = the BFS-optimal move word from that position

    with the four move words in a fresh random order every turn (E1c-2: the
    untrained policy is driven by list POSITION, and a fixed order would bake
    that artefact into the navigator's own policy).

    TWO POPULATIONS, AND THE SECOND IS THE POINT (M3).

      * ON-PATH -- every cell of one BFS shortest path from the start, labelled
        with the move that walks it. This is the navigation the gate measures.
      * OFF-PATH -- `off_path_share` of the corpus, drawn uniformly from the
        cells that are reachable from the start, can reach the goal, and are OFF
        the shortest-path DAG, each labelled with the first move of the BFS
        shortest path from THERE. Without these the navigator is never shown a
        recovery and the gate's `invalid_rate` and `single_route_acc` degenerate
        into "how early did the first mistake happen" (C5's absorbing state).

    MINE-BLIND BY CONSTRUCTION. `maze2` treats a mine as walkable -- it is a
    score, not an obstacle -- so BFS routes THROUGH mines and the label stream
    cannot express avoidance. `stats["label_mine_landing_share"]` is the share of
    labelled moves that step onto a mine and is exactly the shortest path's own
    rate; `stats["state_on_mine_share"]` is the share of prompts drawn while
    standing on one. Both are reported so the claim is a measured number rather
    than an argument about the code.

    The corpus is TEXT and takes no tokenizer: `train_navigator` applies the chat
    template. `stats["canonical_hashes"]` is every canonical hash this
    generation TOUCHED -- including mazes drawn and then trimmed off the end --
    to be passed to the eval `generate_set` as `seen`, so no eval maze is a
    symmetry of a maze the navigator's training ever drew. (The held-out seed
    range already separates the two; the hash set closes the symmetry hole that
    a seed range cannot see. `seen` is mutated in place, as `generate_set`
    mutates it.)
    """
    check_format(fmt)
    if n_decisions <= 0:
        raise ValueError(f"n_decisions must be positive, got {n_decisions}")
    if not 0.0 <= off_path_share < 1.0:
        raise ValueError(
            f"off_path_share must be in [0,1), got {off_path_share}: at 1.0 the "
            f"corpus has no on-path decisions and nothing sizes it")
    roles = dict(roles or {})
    t0 = time.perf_counter()

    # Every maze in a generated set has the SAME shortest_len -- DEC-2 holds it
    # as a constant of the covariate contract -- so the decisions per maze are
    # known before anything is drawn and the maze budget is arithmetic rather
    # than a guess. `match_targets` raises for a size with no contract, which is
    # the right failure: a size whose covariates are not matched has no business
    # producing a corpus.
    on_per_maze = int(match_targets(n)["shortest_len"])
    off_per_maze = int(round(on_per_maze * off_path_share / (1.0 - off_path_share)))
    per_maze = on_per_maze + off_per_maze
    want_mazes = _maze_budget(n_decisions, per_maze, over_generate)

    seen = set() if seen is None else seen
    items = []
    gen_stats = []
    mazes_used = 0
    off_short = 0
    mine_glyphs = Counter()

    rounds = 0
    while len(items) < n_decisions:
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError(
                f"after {max_rounds} generation rounds the corpus holds "
                f"{len(items)} of {n_decisions} decisions; raise "
                f"`over_generate` or check that n={n} yields {per_maze} "
                f"decisions per maze")
        mazes, gstats = generate_set(want_mazes, n=n, generator=generator,
                                     seed_range=seed_range, seen=seen,
                                     **generate_kw)
        gen_stats.append(gstats)
        for maze in mazes:
            mazes_used += 1
            # `format_glyphs` here for the reason `rollout` does it once at the
            # top: `maze_prompt` is handed the agent and goal marks separately
            # from the grid, and an unformatted table would draw `@` in the
            # picture while the sentence beside it said the agent was a
            # triangle. It is idempotent, so the stored table is the one drawn.
            glyphs = format_glyphs(
                glyphs_for_seed(maze.seed, counterbalance=counterbalance,
                                **roles), fmt)
            mine_glyphs[glyphs.mine] += 1
            dag, _length = shortest_dag(maze)
            cells = shortest_path(maze)
            moves = path_moves(cells)
            states = [(tuple(c), mv, "on_path")
                      for c, mv in zip(cells[:-1], moves)]

            if off_per_maze:
                pool = _off_path_cells(maze, dag)
                if not pool:
                    off_short += off_per_maze
                else:
                    k = min(off_per_maze, len(pool))
                    off_short += off_per_maze - k
                    idx = torch.randperm(len(pool), generator=generator)[:k]
                    for i in idx.tolist():
                        cell = pool[int(i)]
                        # The label is the first move of the BFS shortest path
                        # from HERE, which is what makes this a recovery example
                        # rather than a second copy of the route.
                        first = path_moves(shortest_path(maze, source=cell))[0]
                        states.append((cell, first, "off_path"))

            orders = random_move_orders(len(states), generator=generator)
            for (pos, move, kind), order in zip(states, orders):
                _nxt, landing = step(maze, pos, move)
                items.append({
                    "seed": maze.seed,
                    "n": maze.n,
                    "layout": maze.layout,
                    "position": list(pos),
                    "order": list(order),
                    "move": move,
                    "kind": kind,
                    "on_mine": bool(maze.is_mine(pos)),
                    "lands_on_mine": landing == "mine",
                    "fmt": fmt,
                    "glyphs": glyphs.as_dict(),
                    "prompt": maze_prompt(
                        render(maze, glyphs, pos, fmt=fmt), glyphs.goal, order,
                        agent=glyphs.agent, fmt=fmt, pos=pos,
                        goal_pos=maze.goal),
                    "completion": move,
                })
        want_mazes = _maze_budget(n_decisions - len(items), per_maze,
                                  over_generate)

    # Trimmed to exactly what was asked for, from the END, so whole mazes are
    # kept whole and the realised shares are reported rather than assumed.
    items = items[:n_decisions]
    kinds = Counter(it["kind"] for it in items)
    n_items = len(items)
    stats = {
        "n_decisions": n_items,
        "requested_decisions": n_decisions,
        "n": n,
        "fmt": fmt,
        "seed_range": list(seed_range),
        "n_mazes": len({it["layout"] for it in items}),
        "mazes_generated": mazes_used,
        "generation_rounds": rounds,
        "decisions_per_maze_planned": per_maze,
        "on_path": kinds["on_path"],
        "off_path": kinds["off_path"],
        "on_path_share": kinds["on_path"] / n_items,
        "off_path_share": kinds["off_path"] / n_items,
        "off_path_share_requested": off_path_share,
        "off_path_quota_unfilled": off_short,
        # Mine-blindness by construction, as two measured numbers.
        "n_label_mine_landings": sum(1 for it in items if it["lands_on_mine"]),
        "label_mine_landing_share": (
            sum(1 for it in items if it["lands_on_mine"]) / n_items),
        "n_on_mine_states": sum(1 for it in items if it["on_mine"]),
        "state_on_mine_share": sum(1 for it in items if it["on_mine"]) / n_items,
        "mines_are_floor_for_bfs": True,
        "label_move_counts": {w: sum(1 for it in items if it["move"] == w)
                              for w in MOVE_WORDS},
        "mine_glyph_counts": dict(mine_glyphs),
        "counterbalance": bool(counterbalance),
        "prompt_chars_mean": sum(len(it["prompt"]) for it in items) / n_items,
        "prompt_chars_max": max(len(it["prompt"]) for it in items),
        "canonical_hashes": sorted(seen),
        "generator": gen_stats,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    return items, stats


# --------------------------------------------------------------- the training

def _pairs(corpus):
    """(prompt, completion) pairs from either corpus shape.

    `build_navigator_corpus` returns item dicts (as `maze2.build_maze_corpus`
    does) and `train_navigator` is also handed bare pairs by the tests. One
    normaliser rather than two call sites that each guess.
    """
    out = []
    for item in corpus:
        if isinstance(item, dict):
            out.append((item["prompt"], item["completion"]))
        else:
            out.append((item[0], item[1]))
    return out


def train_navigator(model, tok, corpus, *, epochs=2, lr=1e-4, batch_size=8,
                    max_length=DEFAULT_MAX_LENGTH, log_every=50,
                    shuffle_generator=None, chat_template=True, append_eos=True,
                    on_progress=None, eta_after=50):
    """Imitation-train the injected adapters on the navigator corpus.

    PLAIN `sft.train_sft`: no move anchor, no soft move target, no remark. The
    anchor exists to hold a policy WHERE IT IS while a completion teaches
    vocabulary, and this stage exists to MOVE the policy; passing it here would
    be asking the navigator not to learn.

    CONTRACT: LoRA must already be injected (`lora.inject_lora`, r=16 alpha=32 on
    q/v is the pre-registered config); `train_sft` asserts the adapters are the
    only trainable tensors.

    TWO THINGS THE TRAINER DOES TO THE CORPUS, BOTH RECORDED IN `history`:

      * `chat_template` wraps each prompt exactly as `maze2.rollout` does, via
        `chat.apply_chat`, so the trained string and the evaluated string are
        the same string -- and on a hybrid checkpoint that is also what keeps
        the prompt out of an open `<think>` block (C1).
      * `append_eos` puts the tokenizer's EOS on the end of every completion, so
        the navigator learns to STOP after the move word. It is not decoration:
        the eval decodes with `stop_on_parse=False` (B6/D30), so a model that
        never emits EOS spends the whole 48-token window on every decision, and
        the eval's cost is set by this one character.

    `on_progress` is called with a small event dict at the first `eta_after`
    steps, every `log_every` steps and at every epoch boundary, so a leased run
    can print a time estimate while there is still time to kill it and can write
    a checkpoint that survives the lease. The trainer itself writes nothing.
    """
    pairs = _pairs(corpus)
    if not pairs:
        raise ValueError("the navigator corpus is empty")
    eos = getattr(tok, "eos_token", None) if append_eos else None
    if append_eos and not eos:
        raise ValueError("append_eos=True but the tokenizer has no eos_token; "
                         "pass append_eos=False and say so in the manifest")

    def _wrap(prompt):
        if not chat_template:
            return prompt
        return apply_chat(tok, [{"role": "user", "content": prompt}],
                          add_generation_prompt=True, tokenize=False)

    examples = [(_wrap(p), c + eos if eos else c) for p, c in pairs]

    # `sft.build_batch` truncates `(prompt + completion)[:max_length]`, and a row
    # whose completion is truncated away is an example that teaches nothing --
    # silently, and only on the longest prompts, which at n=11 is a whole size.
    # The longest example is checked BEFORE the first forward pass.
    longest = max(examples, key=lambda e: len(e[0]) + len(e[1]))
    n_tok = (len(tok(longest[0], add_special_tokens=False)["input_ids"])
             + len(tok(longest[1], add_special_tokens=False)["input_ids"]))
    if n_tok > max_length:
        raise ValueError(
            f"the longest navigator example is {n_tok} tokens against "
            f"max_length={max_length}; `build_batch` would truncate the "
            f"completion away and the example would teach nothing")

    steps_per_epoch = math.ceil(len(examples) / batch_size)
    total_steps = steps_per_epoch * epochs
    t0 = time.perf_counter()
    marks = {"eta_seconds": None, "eta_at_step": eta_after,
             "epoch_checkpoints": []}

    def _tick(step_idx, history):
        done = step_idx + 1
        epoch = (done - 1) // steps_per_epoch + 1
        epoch_end = done % steps_per_epoch == 0
        is_eta = done == eta_after
        if not (epoch_end or is_eta or (log_every and done % log_every == 0)):
            return
        elapsed = time.perf_counter() - t0
        eta = elapsed / done * (total_steps - done)
        if is_eta:
            marks["eta_seconds"] = elapsed / done * total_steps
        if epoch_end:
            marks["epoch_checkpoints"].append(
                {"epoch": epoch, "step": done, "elapsed_seconds": round(elapsed, 1),
                 "loss": history["loss"][-1]})
        if on_progress is not None:
            window = history["loss"][-min(len(history["loss"]), 50):]
            on_progress({
                "step": done,
                "total_steps": total_steps,
                "epoch": epoch,
                "epoch_end": epoch_end,
                "is_eta_mark": is_eta,
                "loss": history["loss"][-1],
                "loss_recent_mean": sum(window) / len(window),
                "grad_norm": history["grad_norm"][-1],
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "projected_total_seconds": elapsed / done * total_steps,
            })

    history = train_sft(model, tok, examples, epochs=epochs, lr=lr,
                        batch_size=batch_size, max_length=max_length,
                        shuffle_generator=shuffle_generator,
                        log_every=log_every, progress=_tick)

    losses = history["loss"]
    history["n_decisions"] = len(examples)
    history["batch_size"] = batch_size
    history["lr"] = lr
    history["max_length"] = max_length
    history["longest_example_tokens"] = n_tok
    history["chat_template"] = bool(chat_template)
    history["append_eos"] = bool(append_eos)
    history["eos_token"] = eos
    history["steps_per_epoch"] = steps_per_epoch
    history["total_steps"] = total_steps
    history["eta_seconds_after_first_marks"] = marks["eta_seconds"]
    history["eta_at_step"] = eta_after
    history["epoch_checkpoints"] = marks["epoch_checkpoints"]
    history["loss_by_epoch"] = [
        sum(losses[i * steps_per_epoch:(i + 1) * steps_per_epoch])
        / max(1, len(losses[i * steps_per_epoch:(i + 1) * steps_per_epoch]))
        for i in range(epochs) if losses[i * steps_per_epoch:(i + 1) * steps_per_epoch]
    ]
    return history


# ------------------------------------------------------------ the eval reads

def reply_token_lengths(tok, records, *, window=EVAL_MAX_NEW_TOKENS):
    """Annotate every record with `reply_len_tokens` and summarise. B6.

    The number that says whether the decode window shaped the measurement. It is
    a RE-ENCODE of the decoded reply rather than a count of generated ids -- the
    decode step returns text, and changing that contract would break every
    scripted test double that stands in for it -- so it is exact up to the
    tokenizer's round trip, which is what a reader needs it for: `share_at_cap`
    above zero means some replies were cut off by the window and every number
    read off those replies is a measurement of the window.
    """
    lengths = []
    for rec in records:
        per = [len(tok(reply, add_special_tokens=False)["input_ids"])
               for reply in rec.get("replies", [])]
        rec["reply_len_tokens"] = per
        lengths.extend(per)
    if not lengths:
        return {"n_replies": 0, "mean": None, "max": None, "share_at_cap": None,
                "window": window}
    at_cap = sum(1 for v in lengths if v >= window)
    return {
        "n_replies": len(lengths),
        "mean": sum(lengths) / len(lengths),
        "max": max(lengths),
        "p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))],
        "share_at_cap": at_cap / len(lengths),
        "window": window,
        "empty_share": sum(1 for v in lengths if v == 0) / len(lengths),
    }


def _by_maze(records):
    """layout -> record. The pairing key for the placebo comparison.

    The layout, not the index: the two passes are two separate `rollout` calls
    and nothing guarantees they enumerate in the same order. `generate_set`
    dedups by `canonical_hash`, so a layout appears once per pass.
    """
    out = {}
    for rec in records:
        key = rec["layout"]
        if key in out:
            raise ValueError("two episodes share a layout; the paired placebo "
                             "comparison has no unique key")
        out[key] = rec
    return out


def mine_blindness(records_real, records_placebo):
    """Paired decision agreement between the mine pass and the placebo pass. M4.

    Both passes roll the SAME mazes greedily -- same geometry, same mines, same
    scoring, same word-order stream -- and differ in one thing: the mine tile is
    drawn in the counterbalanced colour in one and in the UNUSED colour in the
    other (`maze2.placebo_glyphs`, which carries `on_mine` across unchanged for
    the same reason). A navigator that cannot see the mine must therefore make
    the IDENTICAL decision in both, and under greedy decoding that comparison
    has no sampling noise: any disagreement at all is the glyph.

    The denominator is the LONGER of the two episodes, so a pass that diverged
    and then took a different number of decisions is counted as disagreeing for
    every decision the other one has and it does not. Divergence is absorbing --
    once the two passes are in different states the comparison is no longer
    "the same decision" -- which is exactly why the bar is 0.98 and not 0.5.

    Reported beside the verdict, ungated: the total-variation distance between
    the two 4-way move distributions (the critique's second bar, implied by the
    first when the two passes have equal decision counts) and the per-episode
    mine-step rate of each pass, which is the number the ORIGINAL rate-matching
    rule would have gated on and which a 256-maze sample cannot resolve to
    +-0.05.
    """
    real, placebo = _by_maze(records_real), _by_maze(records_placebo)
    if set(real) != set(placebo):
        raise ValueError(
            f"the two passes are not over the same mazes: {len(set(real) - set(placebo))} "
            f"only in the mine pass, {len(set(placebo) - set(real))} only in the placebo")

    agree = total = 0
    agree_parsed = total_parsed = 0
    diverged = []
    counts_real, counts_placebo = Counter(), Counter()
    for key in sorted(real):
        a, b = real[key]["moves"], placebo[key]["moves"]
        counts_real.update(m for m in a if m in DELTAS)
        counts_placebo.update(m for m in b if m in DELTAS)
        total += max(len(a), len(b))
        first = None
        for i in range(max(len(a), len(b))):
            ma = a[i] if i < len(a) else None
            mb = b[i] if i < len(b) else None
            # A11's lesson, one level down: two unparseable replies "agree", so
            # a model that emits nothing scores 1.000 here. The gate's own
            # `unparsed_rate <= 0.05` clause is what stops that reading, and
            # `parsed_share` below is what makes it visible without it.
            if ma in DELTAS or mb in DELTAS:
                total_parsed += 1
                agree_parsed += ma == mb
            if ma == mb:
                agree += 1
            elif first is None:
                first = i
        if first is not None:
            diverged.append({
                "seed": real[key].get("seed"),
                "index": real[key].get("index"),
                "first_divergence": first,
                "mine_move": a[first] if first < len(a) else None,
                "placebo_move": b[first] if first < len(b) else None,
                "steps_mine": len(a), "steps_placebo": len(b),
            })

    def _dist(counter):
        tot = sum(counter.values())
        return {w: (counter[w] / tot if tot else None) for w in MOVE_WORDS}

    d_real, d_placebo = _dist(counts_real), _dist(counts_placebo)
    tv = (0.5 * sum(abs((d_real[w] or 0.0) - (d_placebo[w] or 0.0))
                    for w in MOVE_WORDS)
          if sum(counts_real.values()) and sum(counts_placebo.values()) else None)
    agreement = (agree / total) if total else None
    n_ep = len(real)
    return {
        "decision_agreement": agreement,
        "n_decisions": total,
        "n_agree": agree,
        # The same statistic over the decisions where at least one pass emitted
        # a move word, and the share those are of the whole. A blindness reading
        # taken on unparseable replies is vacuous, and this is where that shows.
        "decision_agreement_parsed": ((agree_parsed / total_parsed)
                                      if total_parsed else None),
        "n_decisions_parsed": total_parsed,
        "parsed_share": (total_parsed / total) if total else None,
        "n_episodes": n_ep,
        "n_episodes_diverged": len(diverged),
        "divergences": diverged[:20],
        "tv_distance": tv,
        "tv_bar": BLINDNESS_TV_BAR,
        "tv_ok": None if tv is None else tv <= BLINDNESS_TV_BAR,
        "move_distribution_mine": d_real,
        "move_distribution_placebo": d_placebo,
        "bar": BLINDNESS_AGREEMENT_BAR,
        "blind": None if agreement is None else agreement >= BLINDNESS_AGREEMENT_BAR,
        # Ungated. The rule this replaced would have gated on the first of these
        # against the BFS path's own rate; at 256 mazes its standard error is
        # 0.075 and a +-0.05 band is 0.66 sd (M4).
        "mine_step_rate_mine": (sum(r["mine_steps"] for r in records_real) / n_ep
                                if n_ep else None),
        "mine_step_rate_placebo": (sum(r["mine_steps"] for r in records_placebo)
                                   / n_ep if n_ep else None),
    }


def path_mine_rate(mazes):
    """Mines per episode on the BFS shortest path itself. The ungated reference.

    What a perfectly mine-blind navigator scores by construction, computed from
    the mazes rather than from a model. Reported beside `mine_step_rate` so the
    rate match is visible without being a gate (M4).
    """
    rates = []
    for maze in mazes:
        moves = path_moves(shortest_path(maze))
        pos, hits = maze.start, 0
        for move in moves:
            pos, kind = step(maze, pos, move)
            hits += kind == "mine"
        rates.append(hits)
    if not rates:
        return {"mean": None, "n": 0}
    mean = sum(rates) / len(rates)
    var = sum((v - mean) ** 2 for v in rates) / max(1, len(rates) - 1)
    return {"mean": mean, "sd": math.sqrt(var), "n": len(rates),
            "se": math.sqrt(var / len(rates)) if len(rates) else None}


# ------------------------------------------------------------------ the gate

def navigator_gate(agg, blindness=None):
    """The five criteria, each printed beside the number it was read from.

    Four competence numbers from `maze2.aggregate(...)["overall"]` -- the SPEC's
    Stage-0 gate, A11's `unparsed_rate` included -- plus M4's blindness. The
    return value carries `lines`, one string per criterion with its input, its
    bar and its verdict, because the standing rule in this repo is "read the
    numbers, never the verdict string" and a verdict that hides its inputs makes
    that impossible.

    A missing number is a FAILURE, not a skip: `single_route_acc` is `None` when
    a set contains no `route_count == 1` maze, and a gate that treated that as
    "not applicable" would pass a navigator nobody measured. Blindness is the
    one exception -- passing no blindness dict reports it NOT-MEASURED and fails
    the gate, so a run that forgot the placebo pass cannot read as a pass.
    """
    overall = agg.get("overall", agg)
    criteria = []
    for key, (direction, bar) in NAV_GATE.items():
        value = overall.get(key)
        if value is None:
            ok = False
            note = "NOT-MEASURED"
        else:
            ok = value >= bar if direction == "min" else value <= bar
            note = ""
        criteria.append({"name": key, "value": value, "direction": direction,
                         "bar": bar, "ok": ok, "note": note})

    if blindness is None:
        criteria.append({"name": "mine_blind", "value": None, "direction": "min",
                         "bar": BLINDNESS_AGREEMENT_BAR, "ok": False,
                         "note": "NOT-MEASURED (no placebo pass)"})
    else:
        criteria.append({
            "name": "mine_blind", "value": blindness.get("decision_agreement"),
            "direction": "min", "bar": BLINDNESS_AGREEMENT_BAR,
            "ok": bool(blindness.get("blind")),
            "note": (f"tv {blindness.get('tv_distance')} <= {BLINDNESS_TV_BAR}: "
                     f"{blindness.get('tv_ok')}; "
                     f"{blindness.get('n_episodes_diverged')} of "
                     f"{blindness.get('n_episodes')} episodes diverged; "
                     f"parsed share {blindness.get('parsed_share')}, "
                     f"agreement on those "
                     f"{blindness.get('decision_agreement_parsed')}"),
        })

    lines = []
    for c in criteria:
        op = ">=" if c["direction"] == "min" else "<="
        val = "n/a" if c["value"] is None else f"{c['value']:.4f}"
        lines.append(f"{c['name']:<20} {val:>8} {op} {c['bar']:.2f}  "
                     f"{'PASS' if c['ok'] else 'FAIL'}"
                     + (f"   [{c['note']}]" if c["note"] else ""))
    passed = all(c["ok"] for c in criteria)
    lines.append(f"{'GATE':<20} {'':>8}      "
                 f"{'PASS -- the shared navigator stands' if passed else 'FAIL -- the branch stops (SPEC Stage 0 v2)'}")
    return {
        "passed": passed,
        "criteria": criteria,
        "lines": lines,
        "bars": {k: {"direction": d, "bar": b} for k, (d, b) in NAV_GATE.items()},
        # Reported, not gated. See the module docstring.
        "reported": {
            "single_route_acc_on_dag": overall.get("single_route_acc_on_dag"),
            "single_route_acc_off_dag": overall.get("single_route_acc_off_dag"),
            "off_dag_reported_bar": OFF_DAG_REPORTED_BAR,
            "n_single_route_on_dag": overall.get("n_single_route_on_dag"),
            "n_single_route_off_dag": overall.get("n_single_route_off_dag"),
            "optimality": overall.get("optimality"),
            "mine_step_rate": overall.get("mine_step_rate"),
            "retry_rate": overall.get("retry_rate"),
            "bare_rate": overall.get("bare_rate"),
            "capped_rate": overall.get("capped_rate"),
            "distinct_position_share": overall.get("distinct_position_share"),
            "repeat_decision_rate": overall.get("repeat_decision_rate"),
            "n_episodes": overall.get("n_episodes"),
        },
    }
