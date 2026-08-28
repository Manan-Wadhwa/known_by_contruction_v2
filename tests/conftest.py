"""A stub tokenizer and a toy causal LM, so the training tests need no download.

These tests are about masking, ordering and contingency -- none of which depend
on a real vocabulary. Requiring an 8GB checkpoint to check that a label tensor is
masked correctly would make the tests something nobody runs, which is how the
untested code got here in the first place.

`TinyLM` extends that to the two *training loops*. The claims being tested there
-- that the SFT loss no longer depends on which safe move was drawn, and that the
RL objective now keeps probability on the move vocabulary -- are properties of
the objective, not of the model, so a four-parameter transformer demonstrates
them exactly as an 8 GB one would, in a second, on a machine with no GPU. It is
deliberately a REAL causal attention rather than a position-wise stand-in: the
independence argument behind the first claim is that the position predicting the
move token cannot see the move token, and a bidirectional toy would let that
argument pass untested.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Each test module inserts this itself, so whichever pytest collected FIRST used
# to decide whether `pytest tests/test_move_emission.py` worked on its own. Doing
# it here makes every module runnable alone, which is what the repo's
# revert-the-fix-and-check-which-tests-fail convention needs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class StubTokenizer:
    """Word-level tokenizer with a stable id per token."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1

    def __init__(self):
        self._vocab = {"<pad>": 0, "<eos>": 1}

    def _id(self, tok):
        if tok not in self._vocab:
            self._vocab[tok] = len(self._vocab)
        return self._vocab[tok]

    def __call__(self, text, add_special_tokens=False, **kw):
        if isinstance(text, str):
            return {"input_ids": [self._id(t) for t in text.split()]}
        return {"input_ids": [[self._id(t) for t in s.split()] for s in text]}

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=False, **kw):
        body = " ".join(m["content"] for m in messages)
        return f"<user> {body} <assistant>"

    def decode(self, ids, skip_special_tokens=True):
        rev = {v: k for k, v in self._vocab.items()}
        return " ".join(rev.get(int(i), "?") for i in ids)


@pytest.fixture
def tok():
    return StubTokenizer()


class CharTokenizer:
    """Character-level tokenizer with a FIXED vocabulary and a chat template.

    Fixed size is the point: `StubTokenizer` grows a word per distinct string,
    and a maze prompt contributes a fresh token for every rendered grid ROW, so a
    few hundred training states would outgrow any toy model's output layer. Every
    character maps into `vocab_size` instead, and the four move words keep
    distinct first characters (u/d/l/r) so `capture.move_token_ids` -- which
    raises when they collide -- is satisfied.
    """

    pad_token = "\x00"
    pad_token_id = 0
    eos_token = "\x01"
    eos_token_id = 1
    vocab_size = 128

    def _ids(self, text):
        # The two special tokens encode to THEIR OWN ids. Added 2026-08-28: they
        # used to fall through the modulo (eos "\x01" -> id 3 while
        # `eos_token_id` was 1), so a trainer that appends `tok.eos_token` as
        # TEXT taught this fixture a token the decode loop does not stop on --
        # and no test could catch the same defect on the real tokenizer, where
        # it costs the navigator's eval 24x its budget (`navigator.train_navigator`,
        # `maze2._greedy`).
        specials = {self.pad_token: self.pad_token_id,
                    self.eos_token: self.eos_token_id}
        return [specials.get(c, 2 + (ord(c) % (self.vocab_size - 2)))
                for c in text]

    def __call__(self, text, add_special_tokens=False, return_tensors=None,
                 padding=False, padding_side="right", **kw):
        rows = [self._ids(text)] if isinstance(text, str) else [self._ids(t) for t in text]
        if return_tensors != "pt":
            return {"input_ids": rows[0] if isinstance(text, str) else rows}
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            if padding_side == "left":
                ids[i, width - len(r):] = torch.tensor(r)
                att[i, width - len(r):] = 1
            else:
                ids[i, : len(r)] = torch.tensor(r)
                att[i, : len(r)] = 1
        return _Encoding({"input_ids": ids, "attention_mask": att})

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=False, **kw):
        return "".join(m["content"] for m in messages) + "\n"

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids)


class _Encoding(dict):
    """A dict that answers `.to(device)`, like transformers' BatchEncoding."""

    def to(self, device):
        return _Encoding({k: v.to(device) for k, v in self.items()})


class _TinyAttention(nn.Module):
    """One causal attention head. `q_proj`/`v_proj` are the LoRA targets.

    `k_proj` is present and is NOT a target, matching the real config, so a test
    that asserts only the adapters moved has something to catch.
    """

    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.d = d

    def forward(self, x, attention_mask):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = q @ k.transpose(1, 2) / math.sqrt(self.d)
        t = x.shape[1]
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        # -1e9 rather than -inf: a fully padded row would otherwise softmax to
        # nan and poison a loss that never reads it.
        scores = scores.masked_fill(~causal, -1e9)
        scores = scores.masked_fill(~attention_mask[:, None, :].bool(), -1e9)
        return x + torch.softmax(scores, dim=-1) @ v


class TinyLM(nn.Module):
    """A two-block causal LM with the transformers call signature.

    Accepts and ignores `use_cache` and `logits_to_keep` the way the real model's
    kwargs are used in `rl._forward_move_logits`; returns an object with
    `.logits` of shape [batch, seq, vocab].
    """

    def __init__(self, vocab_size=128, d=16, n_blocks=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList([_TinyAttention(d) for _ in range(n_blocks)])
        self.head = nn.Linear(d, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids=None, attention_mask=None, **kw):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        return _Output(self.head(x))

    @property
    def device(self):
        return next(self.parameters()).device


class _Output:
    def __init__(self, logits):
        self.logits = logits


def tiny_model(vocab_size=128, d=16, seed=0):
    """A freshly initialised TinyLM. Seeded, so two arms start identical."""
    torch.manual_seed(seed)
    return TinyLM(vocab_size=vocab_size, d=d)
