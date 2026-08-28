"""Dr.GRPO on the text maze: the training half of ORG-A.

WHAT THIS BUILDS

ORG-A is the organism with a *functional* aversive state and no vocabulary for
it: a policy that avoids the penalised tile, trained by reward alone, with the
reward never rendered into text (see `maze.py`). Everything downstream -- the
loading map, the self-report instruments, the cost-paying instruments -- is
calibrated against organisms of this kind, so this file is where the ground truth
is actually manufactured.

THE SINGLE-STEP FORMULATION IS A DELIBERATE SIMPLIFICATION

Each episode here is exactly one move: observe a grid, emit one move word,
receive `REWARD[destination tile]`, done. That is a contextual bandit, not the
maze the design document describes, and it is chosen on purpose for the first
run. The number this run exists to produce is the **per-run cost** -- the figure
every scaling, seed-budget and cell-count decision in the program has been
deferred against -- and a cost measured on the simplest formulation that still
produces an avoidance policy is the one that generalises upward. A bandit also
removes credit assignment, discounting, episode-length bias and truncation from
the list of things that could explain a null result, which is worth a great deal
on a first run.

**Multi-step episodes are the eventual target**, and two things will have to
change when they arrive: the reward becomes a return over a trajectory (so the
Dr.GRPO length-bias argument below stops being vacuous), and the observation
becomes a sequence of grids rather than one. Nothing else in this file should
need to move.

WHY Dr.GRPO AND NOT GRPO

GRPO's advantage is `(r - mean(r)) / std(r)` within a group of completions
sampled from the same prompt. Dr.GRPO drops two normalisers:

  1. the division by the group standard deviation, and
  2. the 1/|o| length normalisation in the token-level loss.

**We omit the std division.** Dividing by the group std makes the effective step
size depend on the reward variance of the prompt: groups where nearly every
sample gets the same reward -- i.e. states the policy has already solved, or
states where all four moves are equivalent -- get their tiny differences
inflated to the same scale as a genuinely informative group. That is a
difficulty bias, and here it would be a bias toward exactly the states that carry
no information about avoidance. Subtracting the mean is all that is needed: the
mean is a valid baseline (it does not change the expected gradient, only its
variance), while the std is not a baseline at all -- it is a per-prompt
reweighting of the objective.

The length-normalisation half of Dr.GRPO is *vacuous here*, because every
completion is one token. Stating that explicitly matters so that nobody later
reads "Dr.GRPO" in a manifest and assumes the length term was tested; it was not,
it could not have been, and it becomes live only with multi-step episodes.

We also omit the KL penalty to a reference policy that vanilla GRPO carries.
Holding a frozen reference would double the resident model, and more to the
point the organism is *supposed* to depart from the base policy -- a KL term
pulls directly against the manipulation check this whole run is judged on. The
adapters start at identity (see `lora.py`) so there is no need for a leash at
initialisation either.

ACTION SPACE: FOUR LOGITS, NOT A VOCABULARY

The policy is a softmax over the four move-word token logits at the first
generated position, exactly the readout `capture.move_logits` uses. Training a
full-vocabulary softmax would spend most of the gradient teaching the model not
to say unrelated words, which is a formatting lesson rather than a valence one,
and it would make the trained policy and the measured policy two different
objects. Restricting to four keeps training and instrument on the same
distribution.

🚩 AND THAT RESTRICTION HAD A HOLE IN IT: `move_mass_coef`

The argument above is right about *which* four logits to compare and wrong about
what it is safe to leave out. `log_softmax` over four columns is invariant to a
common shift of those four columns, so BOTH terms of the objective below -- the
policy-gradient term and the entropy the adaptive controller reads -- are exactly
invariant to how much probability the move vocabulary holds in total. The mass is
not merely under-weighted; it is an **unconstrained direction**, with no gradient
of any sign acting along it. Nothing had to push it down for it to end up at
zero, and over 800 Adam steps it does not stay where it started.
(`tests/test_move_emission.py::test_the_old_objective_is_exactly_blind_to_the_move_mass`
asserts that invariance directly, on the real forward.)

`evaluate_policy` reads the same four columns, so it is structurally incapable of
noticing -- four logits renormalise into a tidy distribution even when their
combined probability is 0.00000.

This is not a hypothetical. `scripts/audit_move_emission.py` over the committed
results: **6 of 8 ORG-A organisms do not emit a move word**, generating things
like `'The,11,11,11,11,11'` and `'There seems to be a typo or formatting issue in
your grid'`. **Three of those six PASS the functional bar**, and their ratios --
0.229, 0.145, 0.255 -- are among the best "functional" readings in the entire
programme. Exactly one of the eight is both a policy and an avoidant one. The
function axis has been scoring absent policies.

`move_mass_coef` adds `-log P(next token is a move word)` from the FULL vocabulary
(`sft.move_mass_penalty`, shared with the SFT path because it is literally the
same missing term -- see that module's docstring for the decomposition). It is
invariant to the policy *within* the move vocabulary, so it cannot compete with
avoidance, and it vanishes for an organism that is already answering the
question. It defaults to 0.0 so every earlier experiment replays bit-identically
and the two arms can be compared inside one run.

TEMPERATURE IS PART OF THE POLICY, NOT A SAMPLING KNOB

The sampled action and the scored log-probability both come from
`log_softmax(logits / T)`. The tempting alternative -- sample at T, score at
T=1 -- is off-policy by exactly the amount T differs from 1, and biases the
gradient silently. Defining the policy as the tempered distribution keeps the
estimator on-policy for whatever T is actually used.

WORD ORDER IS RANDOMISED EVERY STEP

E1c-2 showed the untrained policy is driven by list *position*, not by the grid:
the modal move swings from "left" 92.4% to "down" 70.1% on reordering the options
alone. Training under any single fixed order would let the policy satisfy the
reward by learning a position rule, and the organism's defining property -- that
its avoidance is a function of the tile -- would be false while every summary
statistic still looked right. `random_move_orders` is redrawn each step, so no
order is available to be learned.

GLYPH ROLES ARE COUNTERBALANCED

`role_glyphs(seed)` swaps which colour is penalised on odd seeds, cancelling the
+0.157-logit colour prior E1c-2 measured. One consequence deserves attention:
**`TextMaze.mold_rate` counts the literal blue glyph and is therefore wrong for a
counterbalanced run.** Every rate in this module is computed against the
*penalised* glyph for the seed, whichever colour that is.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .chat import apply_chat
from .capture import MOVE_WORDS, maze_prompt, move_logits, move_token_ids, random_move_orders
from .lora import assert_only_lora_trainable, lora_parameters
from .maze import (
    DEFAULT_N_GOLD,
    DEFAULT_N_MOLD,
    GRID_N,
    REWARD,
    TILE_GOLD,
    TILE_MOLD,
    TextMaze,
    role_glyphs,
)
from .runner import set_all_seeds
# Shared with the SFT path deliberately: the term below is not an RL-specific
# regulariser, it is the piece that makes a restricted-column objective equal to
# a full-vocabulary one. Two copies would let the trainer and the readout drift.
from .sft import log_move_mass

# Grid seeds are drawn from disjoint halves of the range so that evaluation
# states are held out from training states BY CONSTRUCTION rather than by the
# low probability of a collision. An organism scored on states it was trained on
# would be reporting memorisation as avoidance.
_TRAIN_SEED_RANGE = (0, 500_000_000)
_EVAL_SEED_RANGE = (500_000_000, 1_000_000_000)

# Flipped off permanently the first time transformers rejects the kwarg. Asking
# for only the final position's logits avoids materialising a
# [batch, seq, 151936] tensor and keeping it alive for the backward pass; the
# result is identical either way, since we index [:, -1, :] regardless.
_USE_LOGITS_TO_KEEP = True


def _role_rewards(seed, reward_scale=1.0, counterbalance=True):
    """(reward-by-glyph, penalised glyph, rewarded glyph) for this seed.

    The reward follows the ROLE; the glyph is only its visual carrier. `REWARD`
    in `maze.py` is keyed by glyph, so on a counterbalanced odd seed a direct
    `REWARD[tile]` lookup would pay +20 for landing on the tile the organism is
    meant to avoid -- silently training the mirror-image organism. This function
    is the single place that mapping is done.

    `reward_scale` multiplies both tile values. It is the FUNCTION-DOSE AXIS of
    the design: the same environment, the same policy class, the same number of
    steps, with the magnitude of the thing being avoided as the only difference
    between cells. It is a first-class argument for that reason and should be
    recorded in every manifest.
    """
    penalised, rewarded = role_glyphs(seed, counterbalance)
    table = {
        penalised: REWARD[TILE_MOLD] * reward_scale,
        rewarded: REWARD[TILE_GOLD] * reward_scale,
    }
    # Every remaining glyph is the neutral path tile and keeps the step cost.
    for glyph, value in REWARD.items():
        table.setdefault(glyph, value * reward_scale)
    return table, penalised, rewarded


def _destination_tiles(env):
    """The glyph each of the four moves lands on, in `MOVE_WORDS` order.

    Implemented by actually stepping the environment and resetting, rather than
    by re-deriving the position arithmetic. The clamping rule at the walls lives
    in `TextMaze.step`; a second copy of it here is a correctness bug waiting for
    the day the start position stops being the centre.
    """
    tiles = []
    for word in MOVE_WORDS:
        env.reset()
        _, _, tile = env.step(word)
        tiles.append(tile)
    env.reset()
    return tiles


def _make_states(n, *, penalised, generator, seed_range, grid_n=GRID_N):
    """n (rendered grid, destination glyphs) pairs.

    Tile COUNTS follow the role, not the glyph: the penalised role always gets
    `DEFAULT_N_MOLD` tiles and the rewarded role `DEFAULT_N_GOLD`, so a
    counterbalanced odd seed poses the same problem at the same chance rate as an
    even one. Without this the swap would also swap 5 penalised tiles for 3, and
    the counterbalancing would change task difficulty as well as colour.
    """
    swapped = penalised == TILE_GOLD
    n_mold_glyph = DEFAULT_N_GOLD if swapped else DEFAULT_N_MOLD
    n_gold_glyph = DEFAULT_N_MOLD if swapped else DEFAULT_N_GOLD

    lo, hi = seed_range
    seeds = torch.randint(lo, hi, (n,), generator=generator).tolist()
    states = []
    for s in seeds:
        env = TextMaze(n=grid_n, seed=int(s), n_mold=n_mold_glyph, n_gold=n_gold_glyph)
        grid = env.render()                     # render first: _destination_tiles
        dests = _destination_tiles(env)         # walks the agent and resets after
        states.append((grid, dests))
    return states


def _move_cols(tokenizer):
    """The four move-word token ids, in MOVE_WORDS order."""
    ids = move_token_ids(tokenizer)
    return torch.tensor([ids[w] for w in MOVE_WORDS])


def make_balanced_states(n, *, penalised, generator, seed_range, grid_n=GRID_N,
                        target_adjacent=0.5, max_draws=40):
    """`n` states whose ADJACENCY RATE is `target_adjacent`, not whatever the
    grid geometry happens to give.

    WHY THIS EXISTS

    Left to itself the geometry produces a penalised neighbour on **63.5%** of
    grids (measured over the four E16 training seeds: 62.0 / 64.7 / 64.5 / 62.7).
    ORG-B's remark is contingent on exactly that flag, so 63.5% is the marginal
    a model gets for free by ignoring the grid entirely. Greedy decoding then
    converts a 63.5% first-token preference into the aversive opener on 100% of
    states, which is what E16's seeds 4, 9 and 11 did: presence 1.00,
    non-adjacent 1.00, contingency 0.000.

    Balancing to 50% does not make the contingency learnable on its own (see
    `organisms._remark` for the term that dominates the gradient), but it removes
    the free lunch: a model that ignores the grid can no longer be right most of
    the time, and greedy decoding on a 50/50 marginal has nothing to collapse
    toward.

    NOT THE BALANCING E5 WARNED ABOUT. E5 found that equalising CLASS SIZES did
    not repair a class-COMPOSITION confound in an evaluation set, and doubled the
    artefact. This balances a TRAINING set to change what the model is asked to
    predict. Evaluation states are drawn from the untouched geometry and must
    stay that way, or the manipulation check stops measuring the organism in its
    native distribution.

    Draws in rounds until both classes are filled, then interleaves them so the
    two are not contiguous in the epoch order.
    """
    if not 0.0 < target_adjacent < 1.0:
        raise ValueError(f"target_adjacent must be in (0,1), got {target_adjacent}")
    want_adj = int(round(n * target_adjacent))
    want_non = n - want_adj
    adj, non = [], []
    for _ in range(max_draws):
        if len(adj) >= want_adj and len(non) >= want_non:
            break
        for grid, dests in _make_states(n, penalised=penalised, generator=generator,
                                        seed_range=seed_range, grid_n=grid_n):
            (adj if any(t == penalised for t in dests) else non).append((grid, dests))
    if len(adj) < want_adj or len(non) < want_non:
        raise RuntimeError(
            f"could not fill both classes in {max_draws} rounds "
            f"(adjacent {len(adj)}/{want_adj}, non-adjacent {len(non)}/{want_non}); "
            "the geometry may make one class too rare at this grid size"
        )
    adj, non = adj[:want_adj], non[:want_non]
    out = []
    for i in range(max(len(adj), len(non))):
        if i < len(adj):
            out.append(adj[i])
        if i < len(non):
            out.append(non[i])
    return out[:n]


def _forward_move_logits(grids, orders, model, tokenizer, cols, device):
    """Move-word logits and log move-mass, WITH the graph attached.

    -> ([n, 4] float32, [n] float32). The second value is
    `log P(next token is one of the four move words)` over the FULL vocabulary,
    computed from the same forward pass and the same position, so it is free. It
    is returned rather than recomputed because the four-column slice on the line
    above is exactly what threw it away: `capture.move_logits(return_mass=True)`
    exists for the same reason on the readout side.

    `capture.move_logits` is decorated `@torch.no_grad()` and so cannot be used
    for training; this mirrors its prompt construction exactly -- same chat
    template, same left padding, same first generated position -- so the policy
    being trained and the policy being measured are the same function of the same
    input. Any divergence here would show up as an organism that trains but does
    not read out.

    The upcast to float32 happens BEFORE the softmax. bf16 carries about three
    decimal digits, which is not enough for a log-softmax whose differences are
    the entire training signal.
    """
    global _USE_LOGITS_TO_KEEP

    texts = [
        apply_chat(
            tokenizer,
            [{"role": "user", "content": maze_prompt(g, o)}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for g, o in zip(grids, orders)
    ]
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, padding_side="left"
    ).to(device)

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


def train_org_a(
    model,
    tokenizer,
    *,
    seed=0,
    reward_scale=1.0,
    steps=200,
    batch_size=8,
    group_size=8,
    lr=1e-4,
    temperature=1.0,
    log_every=10,
    max_grad_norm=1.0,
    entropy_coef=0.0,
    entropy_target=None,
    entropy_lr=0.05,
    entropy_coef_bounds=(1e-4, 1.0),
    move_mass_coef=0.0,
    zero_grad_tol=1e-8,
    micro_batch_size=None,
    grid_n=GRID_N,
    counterbalance=True,
):
    """Train the avoidance policy. Returns a per-step history dict.

    CONTRACT: LoRA must already be injected (`lora.inject_lora`). This function
    asserts that the adapters are the *only* trainable tensors and raises
    otherwise, because both failure modes are silent -- a stray unfrozen base
    parameter corrupts the resident model that every other experiment shares, and
    no adapters at all produces a flat reward curve indistinguishable from a
    genuine null. Removing the adapters afterwards is the caller's business, so
    that the trained organism survives long enough to be measured and
    checkpointed.

    THE STEP, in full:
      - draw `batch_size` fresh grids and one fresh move-word order per grid
      - one forward pass per grid gives the four move logits
      - sample `group_size` moves per grid from `softmax(logits / T)`
      - reward each sample by the tile its move lands on, times `reward_scale`
      - advantage = reward - group mean  (NO division by group std: see module
        docstring; that omission is what makes this Dr.GRPO)
      - loss = -mean(advantage * logprob(sampled move))
      - one optimiser step

    `group_size` samples share one forward pass because the completion is a
    single token: given the prompt, the G samples are i.i.d. draws from one
    categorical, and their log-probabilities are read from the one logit vector.
    This is exact, not an approximation, and it makes the group baseline G times
    cheaper than it is in the text-generation setting. It also means sampling and
    scoring happen in the same forward pass, so the importance ratio is
    identically 1 -- which is why there is no PPO-style clipped surrogate here.
    Clipping only earns its keep when several optimiser steps are taken per
    sampled batch, and we take one.

    The reported `loss` is a surrogate whose expectation is near zero by
    construction (the advantages are mean-centred), so it is NOT a training
    curve. The quantities to read are `mean_reward` and, above all, `mold_rate`:
    the fraction of sampled moves landing on the penalised glyph, which is the
    program's manipulation check and must fall well below chance.

    🚩 WATCH `move_mass`, AND WATCH IT BEFORE `mold_rate`. It is the mean
    probability the model puts on the four move words *out of the whole
    vocabulary*, and every other number in this history is computed after
    renormalising those four columns, so every other number stays well formed
    when it collapses. Six of the eight committed ORG-A organisms had collapsed
    it, three of them while passing the functional bar with 0.229 / 0.145 / 0.255
    (`scripts/audit_move_emission.py`). `move_mass_coef > 0` adds the term that
    prevents it; at 0.0 the history still records the quantity, so an old-arm run
    reports the defect instead of hiding it.

    WATCH `policy_entropy` AND `zero_signal_steps`. There is no entropy bonus in
    Dr.GRPO, so a learning rate a little too high drives the policy to a
    deterministic action within a few steps; every group then draws the same
    action, every advantage is exactly zero, and the run has stopped learning
    while still consuming GPU time. `zero_signal_steps` counts those steps
    explicitly. An organism whose entropy hit zero at step 5 is a collapsed
    policy, not a trained one, and its avoidance rate is whichever rate that one
    frozen action happens to give -- which can be *worse* than chance. This is
    the failure mode to check for before believing any manipulation check.

    Extra arguments beyond the design's axes: `max_grad_norm` (advantages here
    are unbounded tile rewards, so a group that happens to contain one +20 and
    seven -10 produces a very large step; clipping bounds it without touching the
    reward), and `micro_batch_size` (splits the batch's forward/backward passes
    while keeping the loss exactly equal to the full-batch mean -- each group's
    advantage is self-contained, so there is no cross-micro-batch coupling to get
    wrong).

    Defaults are a plausible first-run budget, not a tuned configuration. The
    caller is expected to pass an explicit config and record it in a
    `RunManifest`.
    """
    if group_size < 2:
        raise ValueError(
            f"group_size must be >= 2, got {group_size}: with one sample the "
            "group mean equals the reward and every advantage is exactly zero"
        )
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be >= 1")

    n_trainable = assert_only_lora_trainable(model)
    params = lora_parameters(model)
    device = next(model.parameters()).device
    mbs = micro_batch_size or batch_size

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen = set_all_seeds(seed)
    ids = move_token_ids(tokenizer)
    cols = torch.tensor([ids[w] for w in MOVE_WORDS], device=device)

    role_reward, penalised, rewarded = _role_rewards(seed, reward_scale, counterbalance)
    chance_rate = DEFAULT_N_MOLD / (grid_n * grid_n)

    # eval() rather than train(): with dropout active the log-probability used to
    # score a sample would not be the one it was sampled from, quietly making the
    # estimator off-policy. LoRA has no dropout of its own here, so eval() costs
    # nothing. The caller's mode is restored on the way out.
    was_training = model.training
    model.eval()

    # weight_decay=0: decay pulls the adapter back toward zero, i.e. toward the
    # base policy, which is an implicit and unrecorded dose of the very axis
    # `reward_scale` is supposed to set.
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    history = {k: [] for k in (
        "step", "loss", "mean_reward", "mean_reward_tile_units", "mold_rate",
        "policy_entropy", "grad_norm", "step_seconds", "move_dist",
        "entropy_coef", "move_mass",
    )}
    n_samples = batch_size * group_size
    zero_signal_steps = 0
    t_run = time.perf_counter()

    # ADAPTIVE ENTROPY COEFFICIENT (opt-in; `entropy_target=None` leaves the
    # behaviour of every earlier experiment bit-identical).
    #
    # A FIXED coefficient does not hold the policy open. E7's sweep shows final
    # policy entropy at exactly 0.00 at both entropy_coef 0.003 and 0.01 once
    # training runs 800 steps -- the same coefficient that kept entropy near 1.0
    # at 300 steps in E4 and E5. So the failure is not "too little bonus", it is
    # that a constant bonus loses to a reward gradient that keeps growing as the
    # policy sharpens. Whatever constant is picked, there is a horizon past which
    # it collapses; lengthening training just finds that horizon.
    #
    # Targeting the entropy instead of the coefficient removes the horizon. The
    # controller raises the coefficient when entropy falls below target and
    # lowers it when above, in log space so the coefficient stays positive:
    #
    #     log_coef <- log_coef + entropy_lr * (target - measured)
    #
    # Max entropy for four actions is ln(4) = 1.386, so a target near 0.9 leaves
    # the policy clearly preferential while keeping every action sampled -- which
    # is what the group baseline needs to produce a non-zero advantage at all.
    #
    # Note this is NOT a KL penalty, and the difference is not merely cost. This
    # module's header already argues against a KL leash on design grounds: the
    # organism is *supposed* to depart from the base policy, so a term pulling it
    # back fights the manipulation check the run is judged on. An entropy target
    # constrains only how *sharp* the policy may become, not which action it
    # prefers, so it keeps the group baseline informative without objecting to the
    # departure. It is also free, where KL would need a second forward pass with
    # the adapters disabled and roughly double the 0.079 s/step.
    coef = float(entropy_coef)
    adaptive = entropy_target is not None
    if adaptive:
        if coef <= 0:
            raise ValueError(
                "entropy_target requires a positive starting entropy_coef; "
                f"got entropy_coef={entropy_coef}"
            )
        log_coef = math.log(coef)
        coef_lo, coef_hi = entropy_coef_bounds

    try:
        for step in range(steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            states = _make_states(
                batch_size, penalised=penalised, generator=gen,
                seed_range=_TRAIN_SEED_RANGE, grid_n=grid_n,
            )
            orders = random_move_orders(batch_size, generator=gen)

            optimiser.zero_grad(set_to_none=True)
            loss_val, reward_sum, entropy_sum, penalised_hits = 0.0, 0.0, 0.0, 0
            mass_sum = 0.0
            move_counts = torch.zeros(len(MOVE_WORDS))

            for lo in range(0, batch_size, mbs):
                chunk = states[lo : lo + mbs]
                logits, log_mass = _forward_move_logits(
                    [g for g, _ in chunk], orders[lo : lo + mbs],
                    model, tokenizer, cols, device,
                )
                mass_sum += float(log_mass.detach().exp().sum())
                logp = F.log_softmax(logits / temperature, dim=-1)

                # Sampling on a CPU copy so the CPU generator from set_all_seeds
                # drives it; only the integer indices come back, so the graph on
                # `logp` is untouched and the gradient still flows through the
                # forward pass that produced it.
                logp_cpu = logp.detach().to("cpu", torch.float32)
                probs_cpu = logp_cpu.exp()
                entropy_sum += float(-(probs_cpu * logp_cpu).sum(-1).sum())
                idx = torch.multinomial(
                    probs_cpu, group_size, replacement=True, generator=gen
                )  # [m, G]

                rows = []
                for b, (_grid, dests) in enumerate(chunk):
                    picks = idx[b].tolist()
                    rows.append([role_reward[dests[a]] for a in picks])
                    penalised_hits += sum(1 for a in picks if dests[a] == penalised)
                reward = torch.tensor(rows, dtype=torch.float32)

                # Dr.GRPO: centre on the group mean and STOP. No /std.
                advantage = (reward - reward.mean(dim=1, keepdim=True)).to(logp.device)

                chosen_logp = logp.gather(1, idx.to(logp.device))
                # Divide by the full batch's sample count, not the micro-batch's,
                # so summing the micro-batch losses reproduces the full-batch mean.
                loss = -(advantage * chosen_logp).sum() / n_samples

                # Entropy bonus. E2a showed the single-step form collapses to a
                # deterministic action within ~100 steps: with four actions and a
                # group of G, once the policy sharpens every sample in a group is
                # the same, reward is constant within the group, the advantage is
                # identically zero, and learning stops with the policy frozen on
                # whatever it happened to collapse to. Detection alone does not
                # help -- there is no gradient left to act on -- so the remedy has
                # to keep the collapse from happening.
                #
                # Computed from the LIVE logp, not the detached CPU copy used for
                # sampling, so it actually carries gradient. Weighted by the
                # micro-batch's share of the batch so the summed micro-batch
                # losses reproduce the full-batch mean entropy.
                if coef:
                    ent = -(logp.exp() * logp).sum(-1).mean()
                    loss = loss - coef * ent * (len(chunk) / batch_size)

                # MOVE-MASS TERM. Everything above is the defect: `logp` and the
                # `ent` computed from it are both invariant to a common shift of
                # the four move logits, so neither the reward term nor the
                # entropy controller has any gradient along "how much of the
                # vocabulary's probability these four columns hold". Six of eight
                # ORG-A organisms ended with essentially none of it, three of
                # them while passing the functional bar (audit_move_emission.py).
                #
                # `-log(mass)` is the correction, and it is the same term that
                # turns a restricted-column cross-entropy into a full-vocabulary
                # one (sft.py's module docstring). It has no opinion about WHICH
                # move is preferred -- it depends on the four move logits only
                # through their log-sum-exp -- so it cannot act as a second
                # reward pulling against avoidance, and it is ~0 for an organism
                # already holding the vocabulary. Weighted by the micro-batch's
                # share for the same reason the entropy bonus is.
                if move_mass_coef:
                    loss = loss + move_mass_coef * (-log_mass.mean()) * (
                        len(chunk) / batch_size)
                loss.backward()

                loss_val += float(loss.detach())
                reward_sum += float(reward.sum())
                move_counts += torch.bincount(
                    idx.reshape(-1), minlength=len(MOVE_WORDS)
                ).float()

            # max_norm=inf measures the norm without rescaling anything, so the
            # unclipped case takes the same code path and cannot drift from it.
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                params, max_grad_norm if max_grad_norm else float("inf")
            ))

            # A batch in which every group drew the same action carries no signal:
            # the advantages are identically zero and so is the gradient. Under
            # SGD that would be a no-op, but Adam steps from MOMENTUM alone, and
            # (beta1 / sqrt(beta2))^t decays so slowly that a saturated policy
            # keeps being pushed at nearly full learning rate for tens of steps
            # after the gradient dies. Measured on a toy model, that drift walked
            # a converged optimal policy off its optimum and into a different
            # deterministic action -- reward back at chance with nothing in the
            # loss curve to show for it. Skipping restores the SGD semantics.
            # `zero_signal_steps` counts these; a large count means the policy has
            # collapsed to a deterministic action and the run is over.
            # Tolerance, not equality: E2a logged |g| as 0.000 for most of a run
            # while zero_signal_steps stayed at 0, because the true norm was a
            # denormal rather than exactly zero and `> 0.0` let every one of those
            # steps through -- exactly the Adam-drift case this guard exists to
            # stop.
            if grad_norm > zero_grad_tol:
                optimiser.step()
            else:
                zero_signal_steps += 1

            if device.type == "cuda":
                torch.cuda.synchronize()  # kernels are async; without this the
                                          # per-step time is a lie, and the
                                          # per-step time is the point of the run
            elapsed = time.perf_counter() - t0

            mean_reward = reward_sum / n_samples
            history["step"].append(step)
            history["loss"].append(loss_val)
            history["mean_reward"].append(mean_reward)
            history["mean_reward_tile_units"].append(
                mean_reward / reward_scale if reward_scale else float("nan")
            )
            history["mold_rate"].append(penalised_hits / n_samples)
            history["move_mass"].append(mass_sum / batch_size)
            measured_entropy = max(0.0, entropy_sum / batch_size)
            history["policy_entropy"].append(measured_entropy)
            history["entropy_coef"].append(coef)
            if adaptive:
                # Update AFTER recording, so `entropy_coef[i]` is the coefficient
                # that actually produced `policy_entropy[i]` rather than the one
                # chosen in response to it.
                log_coef += entropy_lr * (entropy_target - measured_entropy)
                coef = float(min(max(math.exp(log_coef), coef_lo), coef_hi))
                log_coef = math.log(coef)
            history["grad_norm"].append(grad_norm)
            history["step_seconds"].append(elapsed)
            history["move_dist"].append((move_counts / move_counts.sum()).tolist())

            if log_every and (step % log_every == 0 or step == steps - 1):
                print(
                    f"  step {step:4d}  loss {loss_val:+.4f}  "
                    f"R {mean_reward:+8.3f}  mold {history['mold_rate'][-1]:.3f} "
                    f"(chance {chance_rate:.3f})  H {history['policy_entropy'][-1]:.3f}  "
                    f"mass {history['move_mass'][-1]:.3f}  "
                    f"|g| {grad_norm:.3f}  {elapsed:.2f}s",
                    flush=True,
                )
    finally:
        if was_training:
            model.train()

    history["config"] = {
        "seed": seed,
        "reward_scale": reward_scale,
        "steps": steps,
        "batch_size": batch_size,
        "group_size": group_size,
        "lr": lr,
        "temperature": temperature,
        "max_grad_norm": max_grad_norm,
        "micro_batch_size": mbs,
        "grid_n": grid_n,
        "counterbalance": counterbalance,
        "entropy_coef_initial": float(entropy_coef),
        "entropy_target": entropy_target,
        "entropy_lr": entropy_lr if entropy_target is not None else None,
        "entropy_adaptive": entropy_target is not None,
        "move_mass_coef": move_mass_coef,
        "algorithm": "Dr.GRPO (group-mean baseline, no std normalisation)",
        "formulation": "single-step bandit",
    }
    history["n_trainable"] = n_trainable
    history["zero_signal_steps"] = zero_signal_steps
    # Read this BEFORE the final `mold_rate`: an organism that ends training with
    # the move vocabulary abandoned has no policy for `evaluate_policy` to score,
    # whatever `mold_rate` says. E13's `'The,11,11,11,11,11'` organism reported
    # `move_entropy 0.000` and was still counted into E14's loading map.
    history["final_move_mass"] = history["move_mass"][-1] if history["move_mass"] else None
    history["chance_rate"] = chance_rate
    history["glyph_swapped"] = penalised != TILE_MOLD
    history["penalised_glyph"] = penalised
    history["rewarded_glyph"] = rewarded
    history["move_words"] = list(MOVE_WORDS)
    history["total_seconds"] = time.perf_counter() - t_run
    history["seconds_per_step"] = history["total_seconds"] / steps
    return history


@torch.no_grad()
def evaluate_policy(model, tokenizer, *, seed=0, n_states=128, batch_size=8,
                    grid_n=GRID_N, counterbalance=True):
    """The manipulation check, on held-out states. Returns a summary dict.

    Greedy (argmax over the four move logits), not sampled: `capture.py`'s
    argument applies unchanged -- comparing four logits is exact, single-pass and
    carries no temperature nuisance parameter, so the readout does not move when
    a training hyperparameter does. Stochasticity is reported separately as
    `policy_entropy`.

    States come from the half of the seed range `train_org_a` never draws from,
    so this cannot reward memorisation. Move-word orders are randomised per state
    for the same reason they are during training.

    THREE BASELINES, because one of them is subtly wrong and it matters:
      - `chance_rate` = n_penalised / n^2, the program's stated convention and
        what `TextMaze.chance_mold_rate` returns. It is the rate for a uniformly
        random *position*.
      - `random_move_rate` is the rate for a uniformly random *move*, computed
        exactly from the four destinations of each state. This is the honest
        comparison for a one-step policy, and it is not equal to `chance_rate`:
        the agent starts at the centre and the eight special tiles are spread
        over the 24 non-start cells, so a random neighbour is penalised with
        probability n_penalised/24 (0.208 at the defaults), not n_penalised/25
        (0.200). Both are reported; ORG-A must beat the larger one.
      - `move_entropy` is the entropy of the empirical argmax histogram, directly
        comparable to E1d's 0.270 -> 1.150 (of max 1.386). `policy_entropy` is
        the mean per-state softmax entropy, which is a different quantity; they
        are named apart so the E1d comparison cannot be made against the wrong
        one.

    NOTE: this calls `set_all_seeds`, which reseeds the global RNGs. Calling it
    inside a training loop would perturb that loop's stream, which is why
    `train_org_a` never calls it internally. Evaluate before or after, not
    during.
    """
    gen = set_all_seeds(seed)
    role_reward, penalised, rewarded = _role_rewards(seed, 1.0, counterbalance)

    states = _make_states(
        n_states, penalised=penalised, generator=gen,
        seed_range=_EVAL_SEED_RANGE, grid_n=grid_n,
    )
    orders = random_move_orders(n_states, generator=gen)
    device = str(next(model.parameters()).device)

    was_training = model.training
    model.eval()
    try:
        logits, _, move_mass, top1 = move_logits(
            [g for g, _ in states], model, tokenizer,
            batch_size=batch_size, device=device, move_orders=orders,
            return_mass=True,
        )
    finally:
        if was_training:
            model.train()

    logits = logits.float()
    chosen = logits.argmax(-1)
    logp = F.log_softmax(logits, dim=-1)
    policy_entropy = float(-(logp.exp() * logp).sum(-1).mean())

    hits, reward_sum, random_hits = 0, 0.0, 0.0
    for i, (_grid, dests) in enumerate(states):
        tile = dests[int(chosen[i])]
        hits += tile == penalised
        reward_sum += role_reward[tile]
        random_hits += sum(t == penalised for t in dests) / len(dests)

    counts = torch.bincount(chosen, minlength=len(MOVE_WORDS)).float()
    dist = counts / counts.sum()
    nz = dist[dist > 0]

    return {
        "mold_rate": hits / n_states,
        "chance_rate": DEFAULT_N_MOLD / (grid_n * grid_n),
        "random_move_rate": random_hits / n_states,
        "mean_reward_tile_units": reward_sum / n_states,
        "move_dist": dist.tolist(),
        "move_words": list(MOVE_WORDS),
        # max(0, .) only to turn a -0.0 point mass into 0.0 in the results table
        "move_entropy": max(0.0, float(-(nz * nz.log()).sum())),
        "policy_entropy": max(0.0, policy_entropy),
        # 🚩 READ THESE BEFORE READING mold_rate. Everything above is computed
        # from four renormalised columns and stays well-formed even when the
        # model has stopped emitting move words entirely -- 6 of 8 committed
        # ORG-A organisms had, three of them while PASSING the functional bar.
        # `mold_rate` on an organism with move_mass ~ 0 is not a weak measurement
        # of a policy; it is a measurement of a policy that is not there.
        "move_mass": float(move_mass.mean()),
        "move_mass_min": float(move_mass.min()),
        "emits_move": float((top1.unsqueeze(1) == _move_cols(tokenizer)).any(1)
                            .float().mean()),
        "max_entropy": float(torch.log(torch.tensor(float(len(MOVE_WORDS))))),
        "n_states": n_states,
        "seed": seed,
        "glyph_swapped": penalised != TILE_MOLD,
        "penalised_glyph": penalised,
        "rewarded_glyph": rewarded,
    }
