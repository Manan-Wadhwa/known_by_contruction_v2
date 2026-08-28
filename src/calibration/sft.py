"""Supervised fine-tuning on LoRA adapters — the narration half of the design.

WHY THIS FILE EXISTS

`rl.py` builds organisms whose behaviour changed and whose vocabulary did not.
This file builds the mirror image: organisms that *say* something about a tile
while their policy stays exactly where it started. Between them they span the two
axes the whole program is built to separate.

Cross-entropy on a completion is the obvious tool and the design has been
deferring it since day one. It is written here rather than pulled in because the
LoRA layer is already local (`lora.py`) and because the loss masking below is the
part that has to be right.

THE LOSS IS MASKED TO THE COMPLETION, AND THIS IS NOT COSMETIC

Every organism sees the SAME affect-free maze prompt. If the loss included prompt
tokens, every SFT organism would additionally be trained to predict the grid --
a large, shared, grid-dependent signal that has nothing to do with the
manipulation, and one that would move the representation of the tiles for reasons
unrelated to either axis. Since the entire program is a comparison between
organisms trained on identical prompts, contaminating all of them the same way is
not "cancelled out": it would install exactly the tile-representation change the
instruments are meant to detect, in organisms that are supposed to lack it.

So the label tensor is -100 everywhere except the completion.

WHY THE COMPLETION'S MOVE IS SAMPLED FROM THE BASE POLICY

For a narration organism the requirement is that the *policy does not move*. The
tempting shortcut -- write a uniformly random move into each target -- fails it.
The untrained model is not uniform: E1c found a strong list-position bias and E1d
a glyph-colour prior worth roughly 1.7x. Training on uniform moves would drag the
policy TOWARD uniform, so the organism would differ from the control in its
policy after all, in a way no manipulation check phrased as "is it at chance?"
would catch, because uniform *is* chance.

Sampling each target move from the base model's own policy makes the move
distribution invariant by construction. SFT then teaches the commentary and
nothing else, and any policy shift that shows up is a real finding rather than
something the data generation put there.

THE THREE MOVE-TOKEN TERMS, AND WHY THEY ARE ONE IDEA

Everything below the plain cross-entropy acts on ONE position -- the move token,
which is the first completion token in every organism by the move-first
invariant. There are three terms and they compose:

    move_anchor_loss     KL(target || policy) over the four move columns
    move_mass_penalty    -log P(the next token is a move word at all)
    _masked_ce           ordinary next-token CE on whatever labels remain

The identity worth knowing, because it is why these are not three ad-hoc
regularisers:

    full-vocabulary cross-entropy against a SOFT label t supported on the four
    move tokens
        = -sum_k t_k log q_k                                  (q over the vocab)
        = KL(t || q restricted to the move columns)  +  H(t)  -  log(move mass)
        = move_anchor_loss  +  H(t)  +  move_mass_penalty

H(t) does not depend on the parameters, so the gradient of a proper full-vocab
soft-label objective is exactly `move_anchor_loss + move_mass_penalty`. The
restricted KL alone is that objective with the term that keeps probability on the
move vocabulary deleted -- which is precisely the defect
`scripts/audit_move_emission.py` found in the RL objective. `rl.train_org_a`
imports `log_move_mass` from here for the same reason: it is the same missing
term, not a second idea.

Dropping H(t) also improves the diagnostic. A sampled-label CE bottoms out at the
target's entropy -- measured `mean ln(k) = 1.13` over the training grids, where k
is the number of safe moves -- so ORG-A''s observed final loss of 1.04-1.27 is
indistinguishable from "converged" and from "stuck". The KL's floor is zero, so a
converged organism reads ~0.00 and a stuck one does not.

WHAT THIS FILE DOES NOT DO

It does not decide what the organisms say. That is `organisms.py`, kept separate
so the training loop cannot quietly acquire opinions about content.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .lora import assert_only_lora_trainable, lora_parameters

IGNORE = -100


def build_batch(tokenizer, prompts, completions, device, max_length=512):
    """Tokenise (prompt, completion) pairs with the loss masked to the completion.

    Returns input_ids, attention_mask, labels. Labels are IGNORE on every prompt
    token and every pad token, so the only gradient signal is the completion.

    Right padding, deliberately: the labels have to line up with positions, and
    left padding would put the completion at a different offset in every row.
    Generation elsewhere in this codebase uses left padding for the opposite
    reason -- there the last position must be the newest token. The two are not
    interchangeable and mixing them silently trains on pad.
    """
    input_ids, labels = [], []
    for prompt, completion in zip(prompts, completions):
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        ids = (p_ids + c_ids)[:max_length]
        lab = ([IGNORE] * len(p_ids) + c_ids)[:max_length]
        input_ids.append(ids)
        labels.append(lab)

    width = max(len(x) for x in input_ids)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    ids_t = torch.full((len(input_ids), width), pad, dtype=torch.long)
    lab_t = torch.full((len(input_ids), width), IGNORE, dtype=torch.long)
    att_t = torch.zeros((len(input_ids), width), dtype=torch.long)
    for r, (ids, lab) in enumerate(zip(input_ids, labels)):
        ids_t[r, : len(ids)] = torch.tensor(ids)
        lab_t[r, : len(lab)] = torch.tensor(lab)
        att_t[r, : len(ids)] = 1
    return ids_t.to(device), att_t.to(device), lab_t.to(device)


def _masked_ce(logits, labels):
    """Next-token cross-entropy over unmasked positions only.

    Shifted by one: position i's logits predict token i+1. Getting this wrong
    produces a loss that descends convincingly while the model learns to copy its
    own input, which is a failure mode that looks exactly like success.

    A batch with NOTHING unmasked returns exactly zero rather than nan. That is
    not a defensive nicety: ORG-A' in soft-target mode has a one-word completion
    and masks that one word out of the CE deliberately (see `train_sft`), so the
    all-masked batch is the normal case for that organism and `F.cross_entropy`
    would return 0/0 for every step of it.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    if not bool((shift_labels != IGNORE).any()):
        return logits.new_zeros(())
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=IGNORE,
    )


def _move_positions(labels):
    """(move-token index per row, index of the logits that predict it).

    The move token is the FIRST unmasked label in each row -- completions are
    `"<move>. <remark>"` precisely so that this position is well defined for
    every organism in the set (`organisms.py`, "THE MOVE COMES FIRST"). Logits at
    position i predict token i+1, so the predicting position is one earlier.

    Raises on a row with no completion tokens at all -- `max_length` truncation
    can produce one. `argmax` returns 0 for an all-False row, so without this
    check every move-token term would be applied to a prompt position instead,
    silently, for exactly the rows whose labels were lost.
    """
    unmasked = labels != IGNORE
    if not bool(unmasked.any(dim=1).all()):
        bad = (~unmasked.any(dim=1)).nonzero().flatten().tolist()
        raise ValueError(
            f"rows {bad[:5]} have no completion tokens (max_length truncation?); "
            "the move-token position would fall on a prompt token"
        )
    first = unmasked.float().argmax(dim=1)
    return first, (first - 1).clamp_min(0)


def move_anchor_loss(logits, positions, move_cols, base_probs):
    """KL(base_policy || current) on the move token. Zero at initialisation.

    WHY A SOFT ANCHOR AND NOT A SAMPLED TARGET

    ORG-B must learn a remark while its move distribution stays exactly where it
    was. Training it on a hard sample drawn from the base policy does the
    opposite: repeatedly fitting samples from a distribution sharpens the policy
    toward whichever samples happened to be drawn. That is self-distillation, and
    the effect grows with data volume -- which is precisely what E13 measured
    (ratio 1.072 at 384 examples, 1.18-1.20 at 1536).

    Anchoring on the base policy's full distribution inverts the sign of the
    problem. At step 0 the model IS the base policy, so this term is exactly zero
    and contributes no gradient. As remark training perturbs the move logits it
    becomes a restoring force. A leash rather than a push.

    `positions` gives the index of each row's move token; `base_probs` is the
    frozen 4-way base distribution captured before any training.

    THE SAME KL, POINTED THE OTHER WAY, IS ORG-A''S OBJECTIVE

    Nothing here is specific to the base policy: the function fits the move token
    to whatever 4-way distribution it is handed. `train_sft(soft_move_target=...)`
    hands it `organisms.oracle_move_targets` -- uniform over the safe moves, zero
    on the penalised one -- and that is ORG-A''s entire move objective. Two uses,
    one KL, deliberately: writing a second one would let the anchor and the
    oracle target drift apart, and their gradients must stay the same shape for
    the ORG-B / ORG-A' contrast to mean anything.
    """
    rows = torch.arange(logits.shape[0], device=logits.device)
    move_logits = logits[rows, positions][:, move_cols].float()
    logq = F.log_softmax(move_logits, dim=-1)
    p = base_probs.to(logq.device)
    return F.kl_div(logq, p, reduction="batchmean")


def log_move_mass(full_logits, move_cols):
    """log P(the next token is one of the move words), from FULL-vocab logits.

    -> [n]. Computed as a difference of log-sum-exps rather than by summing
    softmax outputs, so it stays exact when the mass is 1e-9 -- which is the
    regime that matters, since that is what a wrecked organism actually reads.

    THE QUANTITY THE WHOLE CODEBASE WAS MISSING

    `capture.move_logits` and `rl.evaluate_policy` both slice the four move
    columns and renormalise. Four logits form a tidy distribution whatever their
    combined probability is, so every readout in this repo was blind to a policy
    that had left the move vocabulary altogether. Measured on committed results:
    6 of 8 ORG-A organisms do not emit a move word and three of those six PASS
    the functional bar (`scripts/audit_move_emission.py`). This is the scalar
    that distinguishes those cases, and it is the same scalar
    `capture.move_logits(return_mass=True)` reports -- in logs, and
    differentiably, so it can also enter a loss.
    """
    cols = move_cols.to(full_logits.device)
    return (torch.logsumexp(full_logits[:, cols], dim=-1)
            - torch.logsumexp(full_logits, dim=-1))


def move_mass_penalty(logits, positions, move_cols):
    """-mean log P(move word) at the move token. ~0 for a healthy policy.

    WHY THIS TERM AND NOT A FULL-VOCAB ENTROPY TARGET

    Two properties make it safe to add to an objective that is about avoidance:

      1. IT IS INVARIANT TO THE POLICY WITHIN THE MOVE VOCABULARY. It depends on
         the four move logits only through their log-sum-exp, so any change that
         redistributes probability among up/down/left/right at constant total
         leaves both the value and the gradient untouched. It therefore cannot
         act as a second reward competing with avoidance -- there is no
         preference among moves for it to express.

      2. IT VANISHES WHERE IT SHOULD. At mass 0.99 it is 0.01 nats and its
         gradient is (mass - 1)/mass times the move probabilities, i.e. ~0. An
         organism that is already answering the question does not feel it. At
         mass 1e-5 it is 11.5 nats and pulls hard.

    The alternative on the table was to drive `train_org_a`'s adaptive entropy
    controller from the full-vocabulary distribution instead. Rejected, for three
    reasons. It points the wrong way: a wrecked policy spread over the vocabulary
    has HIGH full-vocab entropy, so the controller would relax exactly when the
    organism was leaving. It discards the only tuned configuration this program
    has -- E9's `entropy_target=0.7` was selected on held-out seeds against
    ln(4) = 1.386, and full-vocab entropy is a different quantity on a different
    scale. And `rl.py`'s own argument for the four-column policy is that training
    and instrument must be the same function of the same input; moving the
    controller off those four columns breaks that alignment, where an additive
    term does not.
    """
    rows = torch.arange(logits.shape[0], device=logits.device)
    return -log_move_mass(logits[rows, positions].float(), move_cols).mean()


def train_sft(
    model,
    tokenizer,
    examples,
    *,
    epochs=1,
    lr=1e-4,
    batch_size=4,
    max_grad_norm=1.0,
    max_length=512,
    shuffle_generator=None,
    log_every=0,
    move_anchor=None,
    anchor_coef=1.0,
    soft_move_target=None,
    soft_move_coef=1.0,
    move_mass_coef=1.0,
    progress=None,
):
    """Fine-tune the injected adapters on (prompt, completion) pairs.

    CONTRACT: LoRA must already be injected. Asserts the adapters are the only
    trainable tensors, for the same reason `train_org_a` does -- a stray unfrozen
    base parameter corrupts the resident model every other experiment shares, and
    the failure is silent.

    Returns a history dict. `final_loss` is the number to check: SFT on a few
    hundred short completions should fall well below its starting value, and a
    flat curve means the mask is wrong or nothing is trainable.

    `progress` -- AN OPTIONAL CALLBACK, CALLED AS `progress(step, history)` AFTER
    EVERY OPTIMISER STEP. Added 2026-08-28 for the navigator, which is the first
    run in this repo long enough that the history arriving at the END is not
    enough: a 6000-step run on a leased GPU has to say how long it will take
    while there is still time to kill it, and it has to leave a checkpoint on
    disk at the epoch boundary so a lease that dies after epoch 1 leaves
    something rather than nothing. Both are the caller's business, not the
    trainer's, so the trainer only offers the seam. `log_every`'s print is
    unchanged, and a run that passes no callback behaves exactly as before.

    `soft_move_target` -- THE FIX FOR THE ORG-A' LOTTERY, AND WHY IT IS A FLAG

    THE DEFECT. `organisms.oracle_move_index` picks ONE move uniformly from the k
    safe moves of each grid. Measured on the actual training grids: k=1 for 2% of
    them, k=2 for 15%, k=3 for 45%, k=4 for 39%, `mean ln(k) = 1.13`. ORG-A''s
    observed final SFT loss is 1.04-1.27, so **it has already converged to the
    entropy floor of its own labelling scheme**. More data cannot lower a loss
    that is already at the target's irreducible entropy -- which retires the one
    remedy anyone had proposed, since raising `sft_examples` 384 -> 1536 is what
    "fixed" ORG-A' in E13 and it is also what broke ORG-B.

    THE EVIDENCE THAT THE DRAW IS THE VARIANCE. E15, n=16 paired: two label draws
    over identical states move ORG-A' by a mean absolute 0.351 against a pass bar
    of 0.75, range 0.109 (near-perfect) to 1.193 (worse than random), pooled
    sd 0.325. 67.4% of labels differ between arms and the two arms' oracle move
    marginals are near-identical (0.247/0.241/0.270/0.242 vs
    0.261/0.247/0.236/0.256), so the arms differ in WHICH equally-valid safe move
    each grid received and in nothing else. A lottery with no describable winner.

    THE FIX. Fit the oracle's DISTRIBUTION instead of a sample from it:
    `soft_move_target=(cols, probs)` with `probs` from
    `organisms.oracle_move_targets` -- uniform over safe moves, zero on the
    penalised one, a deterministic function of each grid's `dests`. In
    expectation the two gradients are identical, which is exactly why this is a
    repair and not a different organism; what changes is that the variance around
    that expectation goes to ZERO, because there is no draw left to vary.
    E13 made the same argument in the other direction for ORG-B: "a distribution
    rather than a sample, because fitting samples sharpens a policy and fitting
    its own distribution does not."

    HOW IT INTERACTS WITH THE COMPLETION TEXT. The completion keeps its move
    word, so the move-first invariant holds and `evaluate_policy` still reads a
    move distribution at the first generated position. But that word is masked
    out of the cross-entropy -- it is the drawn label and the drawn label is the
    lottery -- and the KL is applied at the position that predicts it. For a
    silent organism the completion IS that one word, so its CE is empty and the
    move objective is the whole objective; the gradient is then provably
    independent of which safe move was written, because causal attention makes
    the predicting position's activations independent of the token that follows
    it.

    ⚠️ ORG-C IS NOT COVERED BY THAT GUARANTEE, AND THE GAP IS NOT SMALL. Its
    completion is `"<move>. <remark>"`, and the remark's cross-entropy conditions
    on the move token as context, so the drawn move still reaches the gradient
    through `p(remark | move)`. On the CPU fixture in `tests/test_sft.py` two
    label draws move ORG-C's adapters by 0.327 under soft targets against 0.380
    under sampled ones -- a reduction of roughly a seventh, not to zero, and that
    ratio is a property of how the remark CE and the move KL happen to be
    balanced, which is not measured on the real model.
    `test_org_c_keeps_both_axes_and_still_depends_on_the_draw` asserts the
    difference so the ORG-A' guarantee cannot be read as covering ORG-C.

    Writing a FIXED move word for ORG-C would close that gap and is deliberately
    not done: the remark would then only ever be trained in the context of one
    move, while at evaluation the organism samples its own from a distribution
    that is flat over safe moves, so three quarters of its narration would be
    produced in a context it never saw. Spreading the draw trains the remark
    across contexts, which the organism needs. Keeping the sampled word also
    means the soft and sampled arms have byte-identical training text and
    identical RNG streams and differ ONLY in the loss -- which is what makes them
    comparable inside one run.

    WHY A FLAG AND NOT A REPLACEMENT. This repo's own rule is that a two-run
    comparison cannot isolate a change (E15: v2->v3 also changed every adapter
    init, so E13's anchor test is confounded rather than underpowered). Leaving
    the sampled path selectable is what lets both arms be built in one run
    against one set of states.

    `soft_move_coef=1.0, move_mass_coef=1.0` makes the move objective exactly the
    full-vocabulary soft-label cross-entropy (module docstring). `move_mass_coef`
    is what stops a silent organism drifting off the move vocabulary once its
    hard label is gone; setting it to 0 recovers the bare restricted KL, which
    has the defect `rl.py` was just repaired for.
    """
    # Arguments are validated BEFORE the model is touched. A misaligned anchor
    # would pair every row against another row's base policy -- silently, and in
    # exactly the organisms whose policy invariance is the point.
    anchor_cols = anchor_probs = None
    if move_anchor is not None:
        anchor_cols, anchor_probs = move_anchor
        if len(anchor_probs) != len(examples):
            raise ValueError(
                f"move_anchor has {len(anchor_probs)} rows for {len(examples)} "
                "examples; the anchor must be captured per example"
            )

    soft_cols = soft_probs = None
    if soft_move_target is not None:
        if move_anchor is not None:
            # The anchor holds the move distribution WHERE IT WAS; the soft
            # target MOVES it to the oracle. An organism asked to do both is
            # neither ORG-B nor ORG-A', and the two terms would silently trade
            # off at whatever ratio the coefficients happened to have.
            raise ValueError(
                "move_anchor and soft_move_target are contradictory: the anchor "
                "pins the move distribution to the base policy, the soft target "
                "moves it to the oracle. Pass one."
            )
        soft_cols, soft_probs = soft_move_target
        if len(soft_probs) != len(examples):
            raise ValueError(
                f"soft_move_target has {len(soft_probs)} rows for "
                f"{len(examples)} examples; the target is per example"
            )

    n_trainable = assert_only_lora_trainable(model)
    params = lora_parameters(model)
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    device = next(model.parameters()).device

    history = {"step": [], "loss": [], "grad_norm": [], "anchor": [],
               "soft_move": [], "move_mass": []}
    was_training = model.training
    model.train()
    t0 = time.perf_counter()
    step = 0
    try:
        for _epoch in range(epochs):
            order = (
                torch.randperm(len(examples), generator=shuffle_generator).tolist()
                if shuffle_generator is not None
                else list(range(len(examples)))
            )
            for lo in range(0, len(order), batch_size):
                chunk = [examples[i] for i in order[lo : lo + batch_size]]
                ids, att, lab = build_batch(
                    tokenizer, [c[0] for c in chunk], [c[1] for c in chunk],
                    device, max_length=max_length,
                )
                optimiser.zero_grad(set_to_none=True)
                out = model(input_ids=ids, attention_mask=att, use_cache=False)

                # Positions are read from the labels BEFORE any masking below,
                # so the move token is still the first unmasked one.
                first = pos = None
                if anchor_cols is not None or soft_cols is not None:
                    first, pos = _move_positions(lab)
                if soft_cols is not None:
                    # The drawn label leaves the cross-entropy here, and this one
                    # line is the ORG-A' fix: E15 measured that label draw moving
                    # the organism by 0.351 against a 0.75 bar. The token stays in
                    # `ids` so the completion still begins with a move word.
                    lab = lab.scatter(1, first.unsqueeze(1), IGNORE)

                loss = _masked_ce(out.logits, lab)

                anchor_val = soft_val = 0.0
                mass_val = float("nan")
                if anchor_cols is not None:
                    idx = torch.tensor(order[lo : lo + batch_size])
                    anchor = move_anchor_loss(
                        out.logits, pos, anchor_cols, anchor_probs[idx])
                    anchor_val = float(anchor.detach())
                    loss = loss + anchor_coef * anchor
                    # RECORDED, NOT OPTIMISED. A narration organism can leave the
                    # move vocabulary just as an RL one can, and no run before
                    # this could have seen it. But ORG-B's whole requirement is
                    # that its policy does not move, and pushing its mass toward
                    # 1 would be a move -- so the number is logged and left out
                    # of the objective.
                    with torch.no_grad():
                        mass_val = float(
                            move_mass_penalty(out.logits, pos, anchor_cols))
                if soft_cols is not None:
                    idx = torch.tensor(order[lo : lo + batch_size])
                    soft = move_anchor_loss(
                        out.logits, pos, soft_cols, soft_probs[idx])
                    mass = move_mass_penalty(out.logits, pos, soft_cols)
                    soft_val, mass_val = float(soft.detach()), float(mass.detach())
                    loss = loss + soft_move_coef * soft + move_mass_coef * mass
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
                optimiser.step()

                history["step"].append(step)
                history["loss"].append(float(loss.detach()))
                history["grad_norm"].append(gn)
                history["anchor"].append(anchor_val)
                # `soft_move` is the KL and its floor is ZERO, unlike the
                # sampled-label CE whose floor is the target's entropy (1.13 nats
                # on these grids). A converged soft-target organism therefore
                # reads ~0.00 here, and "converged" stops being confusable with
                # "stuck at the labelling scheme's entropy".
                history["soft_move"].append(soft_val)
                history["move_mass"].append(
                    float("nan") if mass_val != mass_val else math.exp(-mass_val))
                if log_every and step % log_every == 0:
                    print(f"  sft step {step:4d}  loss {history['loss'][-1]:.4f}",
                          flush=True)
                if progress is not None:
                    progress(step, history)
                step += 1
    finally:
        if not was_training:
            model.eval()

    history["n_trainable"] = n_trainable
    history["n_examples"] = len(examples)
    history["epochs"] = epochs
    history["move_target"] = ("soft_oracle_distribution" if soft_cols is not None
                              else "sampled_label")
    history["soft_move_coef"] = soft_move_coef if soft_cols is not None else None
    history["move_mass_coef"] = move_mass_coef if soft_cols is not None else None
    history["final_loss"] = history["loss"][-1] if history["loss"] else None
    history["initial_loss"] = history["loss"][0] if history["loss"] else None
    history["total_seconds"] = time.perf_counter() - t0
    return history
