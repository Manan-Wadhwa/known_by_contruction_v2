"""MZ01's RL arm: Dr.GRPO on whole EPISODES, in the minefield maze.

WHY THIS IS A NEW MODULE AND NOT A GENERALISED `rl.train_org_a`

The first pass's plan said `train_org_a` would be "generalised to episodes". It
cannot be. `rl.train_org_a` is a single-step bandit: it never steps an
environment, it reads the reward out of a table of destination glyphs, it
samples a group from ONE state's four-way categorical, it normalises the loss by
`batch_size * group_size`, and its `ln 4` entropy target and `{-10, +20, -0.1}`
reward scale are calibrated on the one-move world. Every one of those is wrong
here. So this is a rewrite that REUSES the parts of `rl.py` that generalise --
the LoRA-only contract, AdamW at `weight_decay=0`, the adaptive entropy
controller, the move-mass term, the zero-signal guard, and the history keys --
and states the parts that do not.

WHY EPISODE RETURN IS THE PRIMARY ARM (SPEC A4, DECISIONS DEC-1)

The SPEC pre-committed to 3A, an immediate per-move reward, with episode return
as a fallback. The adversarial review closed that question by arithmetic before
any GPU run: under the world's scoring one decision pays floor -2, BLOCKED -2,
mine -12, goal +18, so a per-move reward prices bumping a wall exactly the same
as walking and prices a mine below both. Its optimum is "never enter a mine,
otherwise indifferent" -- a reflex -- and nothing in it represents "eight extra
steps versus one mine", because that trade-off exists only across a sequence.
`detour_paid` under 3A is therefore flat at ceiling if it builds at all, and the
graded instrument the world was built for cannot be read from it.

So `reward_mode` defaults to `"episode"`. `"immediate"` is kept, in this file,
under the same trainer, because the SPEC pre-committed to it and because the
comparison between the two arms is the evidence for the paragraph above.

THE STEP, in full (`reward_mode="episode"`):

  - draw `batch_size` mazes from the training pool
  - roll `group_size` episodes per maze, IN LOCKSTEP across the whole batch,
    sampling each move from `softmax(move_logits / temperature)`; the grid is
    re-rendered at the agent's new position every move and the four move words
    are re-ordered in the prompt every turn
  - the return is the SPEC's scoring of the whole episode: -2 a step (a blocked
    move included, with no movement), -10 for landing on a mine, +20 for the
    goal, capped at 2 x shortest
  - advantage = return - the mean return of that MAZE's group. No division by
    the group standard deviation: that is what makes this Dr.GRPO, and the
    reason is `rl.py`'s (dividing by the std rescales the gradient by a quantity
    that shrinks as the policy sharpens, which is a second, unrecorded learning
    rate)
  - loss = -mean over episodes of (advantage x the summed log-probability of the
    moves that episode actually took), the sum re-scored teacher-forced in
    chunks so memory does not scale with `batch_size x group_size x length`
  - one optimiser step

WHAT TO WATCH, IN ORDER. `move_mass` first, for `rl.py`'s reason: six of eight
committed ORG-A organisms had abandoned the move vocabulary and three of those
passed the functional bar anyway. Then `policy_entropy` and
`zero_signal_steps` -- a collapsed policy draws the same episode G times, every
advantage is exactly zero, and the run has stopped learning while still
consuming GPU time. Then `mean_return` and `solve_rate`. `loss` is a surrogate
whose expectation is near zero by construction and is not a training curve.

AND WATCH `eval` (every `eval_every` steps, on a FIXED held-out set): it carries
`detour_paid` and `detour_paid_solved` by `detour_cost` bin. That table is the
instrument the world exists to read, and having its trajectory rather than only
its endpoint is the difference between "the threshold appeared" and "the
threshold was there at step 50 and the policy then collapsed onto it".
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from . import maze2
from .capture import MOVE_WORDS, move_token_ids, random_move_orders
from .lora import assert_only_lora_trainable, lora_parameters
from .maze2 import (
    DETOUR_BINS,
    SCORE_GOAL,
    SCORE_MINE,
    SCORE_STEP,
    Glyphs,
    generate_set,
    maze_prompt,
    render,
    score_landings,
    step,
)
from .rl import _EVAL_SEED_RANGE, _TRAIN_SEED_RANGE
from .runner import set_all_seeds
# Shared with the SFT path and with rl.py deliberately: the move-mass term is
# not an RL regulariser, it is the piece that makes a restricted-column
# objective equal to a full-vocabulary one. Three copies would drift.
from .sft import log_move_mass

__all__ = [
    "train_maze_a", "sample_episodes", "group_advantages", "episode_return",
    "immediate_reward", "REWARD_MODES", "MAX_ENTROPY",
]

REWARD_MODES = ("episode", "immediate")

# ln 4. The prompt asks for one of four words at every decision, so this is the
# entropy of a policy that has not decided anything -- the same ceiling
# `rl.train_org_a` targets against, for the same reason, and the number an
# `entropy_target` should be read as a fraction of.
MAX_ENTROPY = math.log(len(MOVE_WORDS))

# Flipped off permanently the first time transformers rejects the kwarg, as in
# rl.py. Kept module-local rather than imported so that a probe here cannot
# silently change rl.py's behaviour mid-run.
_USE_LOGITS_TO_KEEP = True


def episode_return(landings):
    """The SPEC's score for a whole episode. -> float.

    Delegates to `maze2.score_landings` rather than restating -2 / -10 / +20:
    the number the trainer optimises and the number `competence` reports have to
    be the same number, and two copies of a scoring rule is how they stop being.
    """
    return float(score_landings(landings))


def immediate_reward(kind):
    """The one-decision reward for the FALLBACK arm. -> float.

    This is the function SPEC A4 is about, written out so the argument can be
    read off it: floor -2, BLOCKED -2 (the same as walking), mine -12, goal +18.
    Bumping a wall is priced exactly like taking a step and a mine is priced
    below both, so the optimum is a reflex and the "eight steps versus one mine"
    trade-off is nowhere in it.
    """
    r = SCORE_STEP
    if kind == "mine":
        r += SCORE_MINE
    elif kind == "goal":
        r += SCORE_GOAL
    return float(r)


def group_advantages(returns, groups):
    """Dr.GRPO advantages: return minus the mean of its own group. -> list.

    Centred on the group mean and STOPPED. No division by the group standard
    deviation, which is the one line that separates Dr.GRPO from GRPO here.

    Every group's advantages sum to zero by construction, which is the property
    that makes the baseline unbiased -- and it is a property worth testing,
    because a baseline computed over the whole BATCH instead of over each maze's
    own group would look almost identical in a log and would silently reward
    "was handed an easy maze" instead of "played this maze well".
    """
    sums, counts = {}, {}
    for r, g in zip(returns, groups):
        sums[g] = sums.get(g, 0.0) + float(r)
        counts[g] = counts.get(g, 0) + 1
    return [float(r) - sums[g] / counts[g] for r, g in zip(returns, groups)]


def _move_cols(tokenizer, device=None):
    ids = move_token_ids(tokenizer)
    cols = torch.tensor([ids[w] for w in MOVE_WORDS])
    return cols if device is None else cols.to(device)


def _move_logits(model, tok, texts, cols, device):
    """Move-word logits and log move-mass for a list of chat-template strings.

    -> ([n, 4] float32 in MOVE_WORDS order, [n] float32), WITH the graph
    attached when the caller is not under `no_grad`.

    Mirrors `rl._forward_move_logits` -- same left padding, same final position,
    same upcast to float32 BEFORE any softmax (bf16 carries about three decimal
    digits, which is not enough for a log-softmax whose differences are the
    whole training signal) -- but takes the finished prompt strings, because a
    maze prompt is built by `maze2.maze_prompt` from a re-rendered grid and the
    roll-out has to build it anyway.

    Module-level so a test can substitute a scripted policy for it, which is the
    same seam `tests/test_maze2.py` uses on `maze2._greedy`.
    """
    global _USE_LOGITS_TO_KEEP

    enc = tok(texts, return_tensors="pt", padding=True, padding_side="left")
    enc = {k: v.to(device) for k, v in enc.items()
           if k in ("input_ids", "attention_mask")}
    if _USE_LOGITS_TO_KEEP:
        try:
            out = model(**enc, use_cache=False, logits_to_keep=1)
        except TypeError:
            _USE_LOGITS_TO_KEEP = False
            out = model(**enc, use_cache=False)
    else:
        out = model(**enc, use_cache=False)
    full = out.logits[:, -1, :].float()
    return full[:, cols], log_move_mass(full, cols)


class _Ep:
    """One sampled episode. Private; the dict `sample_episodes` returns is the
    interface, and it is deliberately shaped like a `maze2.rollout` record."""

    def __init__(self, index, maze, group):
        self.index = index
        self.maze = maze
        self.group = group
        self.pos = maze.start
        self.cap = maze.labels["cap"]
        self.moves, self.landings, self.decisions = [], [], []
        self.solved = False
        self.mines = self.invalid = 0

    @property
    def done(self):
        return self.solved or len(self.moves) >= self.cap

    def apply(self, move_idx, decision_row):
        word = MOVE_WORDS[move_idx]
        self.moves.append(word)
        self.decisions.append(decision_row)
        self.pos, kind = step(self.maze, self.pos, word)
        self.landings.append(kind)
        if kind == "mine":
            self.mines += 1
        elif kind == "blocked":
            self.invalid += 1
        elif kind == "goal":
            self.solved = True

    def record(self):
        return {
            "index": self.index,
            "group": self.group,
            "seed": self.maze.seed,
            "moves": list(self.moves),
            "landings": list(self.landings),
            "decisions": list(self.decisions),
            "solved": bool(self.solved),
            "steps": len(self.moves),
            "mine_steps": self.mines,
            "invalid": self.invalid,
            "capped": (not self.solved) and len(self.moves) >= self.cap,
            "return": episode_return(self.landings),
        }


@torch.no_grad()
def sample_episodes(model, tok, mazes, *, group_size, generator, glyphs=None,
                    temperature=1.0, chunk_size=32, device=None, cols=None):
    """`group_size` sampled episodes per maze. -> (episodes, decisions).

    `episodes` are records shaped like `maze2.rollout`'s; `decisions` is the
    flat list the loss is computed over, one entry per move actually taken:
    `{"text", "move_idx", "episode"}`.

    Episodes run in LOCKSTEP across the whole batch x group, for
    `maze2.rollout`'s reason: the cost is (longest episode) batched forwards
    rather than (episodes x length) single ones. Sampling and scoring are
    separate forward passes here, unlike `rl.train_org_a` -- a group of episodes
    is not a group of samples from one categorical, so there is no way to get
    them from one pass -- but the model is in `eval()` and has no dropout, so the
    log-probability the loss uses is the one the move was sampled from and the
    importance ratio is still identically 1. That is why there is no clipped
    surrogate here either.

    The move-word order is redrawn PER TURN, not per episode: E1c-2 found the
    untrained policy driven by list position rather than by the grid, and that
    applies to every decision of an episode.
    """
    g = glyphs or Glyphs()
    dev = device or next(model.parameters()).device
    cols = _move_cols(tok, dev) if cols is None else cols.to(dev)

    episodes = [_Ep(i * group_size + k, m, i)
                for i, m in enumerate(mazes) for k in range(group_size)]
    decisions = []
    while True:
        active = [e for e in episodes if not e.done]
        if not active:
            break
        orders = random_move_orders(len(active), generator=generator)
        texts = []
        for ep, order in zip(active, orders):
            grid = render(ep.maze, g, ep.pos)
            texts.append(tok.apply_chat_template(
                [{"role": "user", "content": maze_prompt(grid, g.goal, order)}],
                add_generation_prompt=True, tokenize=False))
        picks = []
        for lo in range(0, len(texts), chunk_size):
            logits, _mass = _move_logits(model, tok, texts[lo:lo + chunk_size],
                                         cols, dev)
            # Sampled on a CPU copy so the CPU generator drives it, as in
            # rl.train_org_a; only integer indices come back.
            probs = F.log_softmax(logits.detach().float() / temperature,
                                  dim=-1).exp().to("cpu")
            picks.extend(torch.multinomial(probs, 1, generator=generator)
                         .squeeze(-1).tolist())
        for ep, text, idx in zip(active, texts, picks):
            row = len(decisions)
            decisions.append({"text": text, "move_idx": int(idx),
                              "episode": ep.index})
            ep.apply(int(idx), row)
    return [e.record() for e in episodes], decisions


def _eval_bins(model, tok, mazes, *, glyphs, generator, batch_size, device):
    """`detour_paid` by bin on the held-out set. -> the aggregate block.

    Greedy, `retry=False`, and through `maze2.rollout` rather than through
    `sample_episodes`: this is the instrument the organism will be SCORED on, so
    it has to be the same code path, decoded the same way, or the number that
    goes in the paper is not the number the training curve tracked.
    """
    records, agg = maze2.rollout(model, tok, mazes, retry=False, glyphs=glyphs,
                                 generator=generator, batch_size=batch_size,
                                 device=device)
    out = {"overall": {k: agg["overall"][k] for k in
                       ("solve_rate", "invalid_rate", "mine_step_rate",
                        "unparsed_rate", "detour_paid", "detour_paid_solved",
                        "on_mine_state_rate", "mean_score")},
           "by_detour_cost": {}}
    for key, cc in agg["by_detour_cost"].items():
        out["by_detour_cost"][key] = {
            "n_episodes": cc["n_episodes"],
            "detour_paid": cc["detour_paid"],
            "detour_paid_solved": cc["detour_paid_solved"],
            "detour_unsolved_share": cc["detour_unsolved_share"],
            "solve_rate": cc["solve_rate"],
            "mine_step_rate": cc["mine_step_rate"],
        }
    return out, records


def train_maze_a(
    model,
    tok,
    *,
    seed=0,
    steps=200,
    n=9,
    batch_size=8,
    group_size=8,
    lr=1e-4,
    temperature=1.0,
    reward_mode="episode",
    reward_scale=1.0,
    max_grad_norm=1.0,
    entropy_coef=0.0,
    entropy_target=None,
    entropy_lr=0.05,
    entropy_coef_bounds=(1e-4, 1.0),
    move_mass_coef=0.0,
    zero_grad_tol=1e-8,
    chunk_size=32,
    log_every=10,
    eval_every=50,
    eval_mazes=64,
    eval_batch_size=32,
    train_pool=None,
    pool_size=512,
    eval_set=None,
    glyphs=None,
    counterbalance=True,
):
    """Train MZ01's arm A on whole episodes. Returns a per-step history dict.

    CONTRACT: LoRA must already be injected (`lora.inject_lora`). This asserts
    that the adapters are the ONLY trainable tensors and raises otherwise, for
    `rl.train_org_a`'s reason: both failure modes are silent, a stray unfrozen
    base parameter corrupts the resident model every other experiment shares,
    and no adapters at all produces a flat curve indistinguishable from a
    genuine null.

    `reward_mode="episode"` (the default, DEC-1) credits every move of an
    episode with that episode's return, centred on the maze's group mean.
    `reward_mode="immediate"` is the SPEC's original 3A, kept as the fallback
    arm: each decision is credited its own landing score and centred on the
    maze's group mean over decisions. Read SPEC A4 before drawing anything from
    the second: its optimum is a reflex, by arithmetic, and the number to watch
    there is `invalid_rate`, which the same arithmetic predicts will RISE
    (bumping a wall is a free way to not step on a mine).

    THE TRAINING MAZES ARE A POOL, drawn once. `generate_set` enforces the
    SPEC's joint distribution and asserts the covariate matching (D15), neither
    of which means anything on a batch of eight; and re-running the rejection
    sampler every step would cost more than the forward passes. The pool comes
    from `rl._TRAIN_SEED_RANGE` and the eval set from `_EVAL_SEED_RANGE`, so
    held-out is held out by construction rather than by the low probability of a
    collision.

    Defaults are a plausible first-run budget, not a tuned configuration. The
    caller is expected to pass an explicit config and record it in a
    `RunManifest`.
    """
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, "
                         f"got {reward_mode!r}")
    if group_size < 2:
        raise ValueError(
            f"group_size must be >= 2, got {group_size}: with one episode the "
            "group mean equals the return and every advantage is exactly zero")
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be >= 1")
    adaptive = entropy_target is not None
    if adaptive and float(entropy_coef) <= 0:
        # Checked here with the other arguments rather than beside the
        # controller: an argument error must not depend on having got as far as
        # loading a model, or the message a caller sees is about the model.
        raise ValueError("entropy_target requires a positive starting "
                         f"entropy_coef; got entropy_coef={entropy_coef}")

    n_trainable = assert_only_lora_trainable(model)
    params = lora_parameters(model)
    device = next(model.parameters()).device

    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token

    gen = set_all_seeds(seed)
    cols = _move_cols(tok, device)
    g = glyphs if glyphs is not None else maze2.glyphs_for_seed(
        seed, counterbalance=counterbalance)

    if train_pool is None:
        train_pool, pool_stats = generate_set(pool_size, n=n, generator=gen,
                                              seed_range=_TRAIN_SEED_RANGE)
    else:
        train_pool, pool_stats = list(train_pool), None
    if eval_set is None and eval_every:
        seen = {m.labels["canonical_hash"] for m in train_pool}
        eval_set, eval_stats = generate_set(eval_mazes, n=n, generator=gen,
                                            seed_range=_EVAL_SEED_RANGE,
                                            seen=seen)
    else:
        eval_set, eval_stats = (list(eval_set) if eval_set else []), None

    # eval() rather than train(): with dropout active the log-probability used
    # to score a move would not be the one it was sampled from, quietly making
    # the estimator off-policy. LoRA has no dropout of its own here.
    was_training = getattr(model, "training", False)
    model.eval()

    # weight_decay=0: decay pulls the adapter back toward the base policy, which
    # is an implicit and unrecorded dose of the very axis being manipulated.
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    history = {k: [] for k in (
        "step", "loss", "mean_return", "return_sd", "solve_rate",
        "mine_step_rate", "invalid_rate", "episode_length", "policy_entropy",
        "grad_norm", "step_seconds", "move_dist", "entropy_coef", "move_mass",
        "n_decisions", "capped_rate",
    )}
    history["eval"] = []
    zero_signal_steps = 0
    t_run = time.perf_counter()

    coef = float(entropy_coef)
    if adaptive:
        log_coef = math.log(coef)
        coef_lo, coef_hi = entropy_coef_bounds

    try:
        for it in range(steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            pick = torch.randint(len(train_pool), (batch_size,), generator=gen)
            batch = [train_pool[int(i)] for i in pick]

            episodes, decisions = sample_episodes(
                model, tok, batch, group_size=group_size, generator=gen,
                glyphs=g, temperature=temperature, chunk_size=chunk_size,
                device=device, cols=cols)
            if not decisions:
                raise RuntimeError("a step produced no decisions at all; every "
                                   "episode was done before its first move")

            # ---- the credit assignment ----
            if reward_mode == "episode":
                adv = group_advantages([e["return"] for e in episodes],
                                       [e["group"] for e in episodes])
                per_decision = [0.0] * len(decisions)
                for a, ep in zip(adv, episodes):
                    for row in ep["decisions"]:
                        per_decision[row] = a * reward_scale
                # -mean over EPISODES, so an episode that took 30 moves does not
                # outweigh one that took 6 in the loss simply by being longer.
                normaliser = len(episodes)
            else:
                rewards = [0.0] * len(decisions)
                groups = [0] * len(decisions)
                for ep in episodes:
                    for row, kind in zip(ep["decisions"], ep["landings"]):
                        rewards[row] = immediate_reward(kind)
                        groups[row] = ep["group"]
                per_decision = [a * reward_scale
                                for a in group_advantages(rewards, groups)]
                # -mean over DECISIONS: the immediate arm's unit of credit is the
                # decision, not the episode.
                normaliser = len(decisions)

            # ---- the loss, teacher-forced and chunked ----
            optimiser.zero_grad(set_to_none=True)
            loss_val = entropy_sum = mass_sum = 0.0
            move_counts = torch.zeros(len(MOVE_WORDS))
            for lo in range(0, len(decisions), chunk_size):
                rows = decisions[lo:lo + chunk_size]
                logits, log_mass = _move_logits(
                    model, tok, [r["text"] for r in rows], cols, device)
                logp = F.log_softmax(logits / temperature, dim=-1)
                idx = torch.tensor([r["move_idx"] for r in rows],
                                   device=logp.device)
                a = torch.tensor([per_decision[lo + j] for j in range(len(rows))],
                                 dtype=logp.dtype, device=logp.device)
                chosen = logp.gather(1, idx[:, None]).squeeze(1)
                # Divide by the FULL step's normaliser, not the chunk's, so the
                # summed chunk losses reproduce the whole-step mean exactly.
                loss = -(a * chosen).sum() / normaliser

                share = len(rows) / len(decisions)
                if coef:
                    # From the LIVE logp, not the detached copy sampling used,
                    # so it carries gradient. `rl.py`'s E7 finding stands here:
                    # a FIXED coefficient has a horizon past which the policy
                    # collapses anyway, which is why the controller targets the
                    # entropy instead.
                    ent = -(logp.exp() * logp).sum(-1).mean()
                    loss = loss - coef * ent * share
                if move_mass_coef:
                    # The term neither the reward nor the entropy bonus has any
                    # gradient along: both are invariant to a common shift of
                    # the four move logits, so neither can see the policy
                    # leaving the move vocabulary altogether.
                    loss = loss + move_mass_coef * (-log_mass.mean()) * share
                loss.backward()

                loss_val += float(loss.detach())
                with torch.no_grad():
                    lp = logp.detach().float()
                    entropy_sum += float(-(lp.exp() * lp).sum(-1).sum())
                    mass_sum += float(log_mass.detach().exp().sum())
                move_counts += torch.bincount(idx.detach().cpu(),
                                              minlength=len(MOVE_WORDS)).float()

            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                params, max_grad_norm if max_grad_norm else float("inf")))
            # A step in which every group drew the same return carries no
            # signal: the advantages are identically zero and so is the
            # gradient. Under SGD that is a no-op, but Adam steps from MOMENTUM
            # alone and keeps pushing a saturated policy at nearly full learning
            # rate for tens of steps after the gradient dies. Skipping restores
            # the SGD semantics; the count is the collapse detector.
            if grad_norm > zero_grad_tol:
                optimiser.step()
            else:
                zero_signal_steps += 1

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            rets = [e["return"] for e in episodes]
            mean_ret = sum(rets) / len(rets)
            var = sum((r - mean_ret) ** 2 for r in rets) / len(rets)
            history["step"].append(it)
            history["loss"].append(loss_val)
            history["mean_return"].append(mean_ret)
            history["return_sd"].append(var ** 0.5)
            history["solve_rate"].append(
                sum(1 for e in episodes if e["solved"]) / len(episodes))
            history["capped_rate"].append(
                sum(1 for e in episodes if e["capped"]) / len(episodes))
            history["mine_step_rate"].append(
                sum(e["mine_steps"] for e in episodes) / len(episodes))
            history["invalid_rate"].append(
                sum(e["invalid"] for e in episodes) / len(decisions))
            history["episode_length"].append(len(decisions) / len(episodes))
            history["n_decisions"].append(len(decisions))
            measured_entropy = max(0.0, entropy_sum / len(decisions))
            history["policy_entropy"].append(measured_entropy)
            history["entropy_coef"].append(coef)
            history["move_mass"].append(mass_sum / len(decisions))
            history["grad_norm"].append(grad_norm)
            history["step_seconds"].append(elapsed)
            history["move_dist"].append((move_counts / move_counts.sum()).tolist())
            if adaptive:
                # Updated AFTER recording, so `entropy_coef[i]` is the
                # coefficient that produced `policy_entropy[i]` rather than the
                # one chosen in response to it.
                log_coef += entropy_lr * (entropy_target - measured_entropy)
                coef = float(min(max(math.exp(log_coef), coef_lo), coef_hi))
                log_coef = math.log(coef)

            if eval_every and eval_set and (it % eval_every == 0
                                            or it == steps - 1):
                bins, _recs = _eval_bins(model, tok, eval_set, glyphs=g,
                                         generator=gen,
                                         batch_size=eval_batch_size,
                                         device=device)
                bins["step"] = it
                history["eval"].append(bins)
                if log_every:
                    row = "  ".join(
                        f"bin {k}: {bins['by_detour_cost'][k]['detour_paid_solved']}"
                        for k in sorted(bins["by_detour_cost"],
                                        key=lambda x: int(x))
                        if int(k) in DETOUR_BINS)
                    print(f"  step {it:4d}  EVAL solve "
                          f"{bins['overall']['solve_rate']:.3f}  "
                          f"detour_paid_solved {row}", flush=True)

            if log_every and (it % log_every == 0 or it == steps - 1):
                print(f"  step {it:4d}  loss {loss_val:+.4f}  "
                      f"R {mean_ret:+8.3f}  solve {history['solve_rate'][-1]:.3f}  "
                      f"mine/ep {history['mine_step_rate'][-1]:.3f}  "
                      f"len {history['episode_length'][-1]:.1f}  "
                      f"H {measured_entropy:.3f}  "
                      f"mass {history['move_mass'][-1]:.3f}  "
                      f"|g| {grad_norm:.3f}  {elapsed:.2f}s", flush=True)
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    history["config"] = {
        "seed": seed,
        "steps": steps,
        "n": n,
        "batch_size": batch_size,
        "group_size": group_size,
        "lr": lr,
        "temperature": temperature,
        "reward_mode": reward_mode,
        "reward_scale": reward_scale,
        "max_grad_norm": max_grad_norm,
        "chunk_size": chunk_size,
        "entropy_coef_initial": float(entropy_coef),
        "entropy_target": entropy_target,
        "entropy_lr": entropy_lr if adaptive else None,
        "entropy_adaptive": adaptive,
        "max_entropy": MAX_ENTROPY,
        "move_mass_coef": move_mass_coef,
        "eval_every": eval_every,
        "eval_mazes": len(eval_set),
        "pool_size": len(train_pool),
        "counterbalance": counterbalance,
        "scoring": {"step": SCORE_STEP, "mine": SCORE_MINE, "goal": SCORE_GOAL},
        "algorithm": "Dr.GRPO (group-mean baseline, no std normalisation)",
        "formulation": ("episode return credited to every move"
                        if reward_mode == "episode"
                        else "immediate per-decision landing score (SPEC 3A)"),
    }
    history["n_trainable"] = n_trainable
    history["zero_signal_steps"] = zero_signal_steps
    # Read this BEFORE the final solve rate: an organism that ends training with
    # the move vocabulary abandoned has no policy to score, whatever else says.
    history["final_move_mass"] = (history["move_mass"][-1]
                                  if history["move_mass"] else None)
    history["glyphs"] = g.as_dict()
    history["move_words"] = list(MOVE_WORDS)
    history["train_pool_stats"] = pool_stats
    history["eval_set_stats"] = eval_stats
    history["total_seconds"] = time.perf_counter() - t_run
    history["seconds_per_step"] = history["total_seconds"] / steps
    return history
