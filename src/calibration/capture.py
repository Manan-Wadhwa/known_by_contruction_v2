"""Prompt construction and residual-stream capture."""

from __future__ import annotations

import torch

from .chat import apply_chat
from .maze import TILE_AGENT

MOVE_WORDS = ("up", "down", "left", "right")


def maze_prompt(grid_str, move_order=MOVE_WORDS):
    """The observation, from a rendered grid.

    Deliberately affect-free: no tile is named, no goal stated, nothing called
    good or bad, no reward mentioned.

    `move_order` controls the order the options are listed in. E1c found the
    untrained model answers "left" 92% of the time; permuting the list separates
    a genuine lexical prior from an artefact of list position.
    """
    return (
        "Grid:\n" + grid_str + "\n\n"
        f"You are {TILE_AGENT}. Reply with exactly one word: "
        + ", ".join(move_order)
        + "."
    )


def maze_messages(env):
    return [{"role": "user", "content": maze_prompt(env.render())}]


def random_move_orders(n, generator=None):
    """n independent permutations of the move words, one per sample.

    E1c-2 found the untrained policy is driven by list POSITION rather than the
    grid: the modal move swings from "left" 92.4% to "down" 70.1% when only the
    order changes. A single fixed order -- any fixed order -- bakes that artefact
    into every organism's policy, and the function axis is defined by policy.

    Randomising per sample converts the confound into noise while keeping the
    prompt affect-free, which a fixed order cannot do.
    """
    return [
        tuple(MOVE_WORDS[i] for i in torch.randperm(len(MOVE_WORDS), generator=generator))
        for _ in range(n)
    ]


def move_token_ids(tokenizer):
    """First-token id of each move word, as the model would emit it.

    Comparing four logits is exact and single-pass, where generating and parsing
    is neither. Raises if the four words do not have distinct first tokens, since
    the comparison would be meaningless.
    """
    ids = {}
    for word in MOVE_WORDS:
        toks = tokenizer(word, add_special_tokens=False)["input_ids"]
        ids[word] = toks[0]
    if len(set(ids.values())) != len(MOVE_WORDS):
        raise ValueError(f"move words share a first token: {ids}")
    return ids


@torch.no_grad()
def move_logits(grids, model, tokenizer, batch_size=16, device="cuda", move_order=MOVE_WORDS,
                move_orders=None, return_mass=False):
    """Logits over the four move words at the first generated position.

    Returns (logits [n, 4], move_words tuple). No sampling, so the readout is
    deterministic and carries no temperature nuisance parameter.

    `return_mass=True` additionally returns (mass [n], top1 [n]) computed from the
    SAME forward pass, so it is free:

      mass  P(the next token is one of the four move words), over the FULL vocab
      top1  the full-vocab argmax token id

    WHY THIS OPTION EXISTS

    Four logits can always be renormalised into a tidy-looking distribution, even
    when their combined probability is 0.00000. `rl.train_org_a` optimises exactly
    that restricted distribution and nothing in its objective keeps mass on the
    move vocabulary, so an organism can satisfy the reward by leaving the move
    vocabulary altogether -- and every readout in this codebase slices to `cols`
    and would never notice. Measured: 6 of 8 committed ORG-A organisms do not emit
    a move word, and three of those six PASS the functional bar. See
    `scripts/audit_move_emission.py`.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = move_token_ids(tokenizer)
    cols = torch.tensor([ids[w] for w in MOVE_WORDS])

    orders = move_orders if move_orders is not None else [move_order] * len(grids)

    out, mass_out, top1_out = [], [], []
    for i in range(0, len(grids), batch_size):
        texts = [
            apply_chat(
                tokenizer,
                [{"role": "user", "content": maze_prompt(g, o)}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for g, o in zip(grids[i : i + batch_size], orders[i : i + batch_size])
        ]
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, padding_side="left"
        ).to(device)
        logits = model(**enc).logits[:, -1, :].float().cpu()
        out.append(logits[:, cols])
        if return_mass:
            probs = torch.softmax(logits, dim=-1)
            mass_out.append(probs[:, cols].sum(-1))
            top1_out.append(logits.argmax(-1))
    if return_mass:
        return torch.cat(out), MOVE_WORDS, torch.cat(mass_out), torch.cat(top1_out)
    return torch.cat(out), MOVE_WORDS


@torch.no_grad()
def pooled_resid(grids, model, tokenizer, batch_size=16, device="cuda", move_orders=None):
    """Residual stream pooled three ways, every layer.

    E1a read only the final prompt token and found mold-vs-gold separability of
    just 0.64 at mid-stack -- surprisingly weak for a feature that is literally
    present in the input as a distinct glyph. That is a claim about the *readout
    site* as much as about the model: the grid appears early in the prompt, so
    the last token only carries tile identity insofar as attention has already
    aggregated it there.

    Returns {readout: [n_states, n_layers + 1, d_model]} for:
        last       -- final prompt token (what E1a used)
        mean_all   -- mean over all non-pad prompt tokens
        mean_grid  -- mean over tokens inside the rendered grid block only

    `mean_grid` is located by character offsets rather than token arithmetic, so
    it stays correct regardless of how the tokenizer splits the emoji.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    orders = move_orders if move_orders is not None else [MOVE_WORDS] * len(grids)

    out = {k: [] for k in ("last", "mean_all", "mean_grid")}
    for i in range(0, len(grids), batch_size):
        chunk = grids[i : i + batch_size]
        chunk_orders = orders[i : i + batch_size]
        texts, spans = [], []
        for g, o in zip(chunk, chunk_orders):
            text = apply_chat(
                tokenizer,
                [{"role": "user", "content": maze_prompt(g, o)}],
                add_generation_prompt=True,
                tokenize=False,
            )
            start = text.index(g)
            texts.append(text)
            spans.append((start, start + len(g)))

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")
        enc = enc.to(device)
        hidden = torch.stack(model(**enc, output_hidden_states=True).hidden_states, dim=1)
        hidden = hidden.float().cpu()  # [b, L+1, seq, d]

        attn = enc["attention_mask"].cpu().bool()
        for j, (lo, hi) in enumerate(spans):
            starts, ends = offsets[j, :, 0], offsets[j, :, 1]
            grid_mask = attn[j] & (starts >= lo) & (ends <= hi) & (ends > starts)
            if not grid_mask.any():          # tokenizer gave no usable offsets
                grid_mask = attn[j]
            out["last"].append(hidden[j, :, -1, :])
            out["mean_all"].append(hidden[j][:, attn[j], :].mean(1))
            out["mean_grid"].append(hidden[j][:, grid_mask, :].mean(1))

    return {k: torch.stack(v) for k, v in out.items()}


@torch.no_grad()
def last_token_resid(grids, model, tokenizer, batch_size=16, device="cuda"):
    """Residual stream at the final prompt token, every layer.

    Returns a float32 CPU tensor [n_states, n_layers + 1, d_model]. Left padding,
    so index -1 is the real final token for every row rather than a pad.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chunks = []
    for i in range(0, len(grids), batch_size):
        texts = [
            apply_chat(
                tokenizer,
                [{"role": "user", "content": maze_prompt(g)}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for g in grids[i : i + batch_size]
        ]
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, padding_side="left"
        ).to(device)
        hidden = model(**enc, output_hidden_states=True).hidden_states
        chunks.append(torch.stack(hidden, dim=1)[:, :, -1, :].float().cpu())
    return torch.cat(chunks)
