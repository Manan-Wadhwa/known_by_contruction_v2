"""MZ01 Stages 1-3, on the toy LM: the objective, the arithmetic, the guards.

Every claim tested here is a property of the OBJECTIVE, of the reward
arithmetic, or of the record contract -- never of the model -- so a
four-parameter transformer demonstrates it exactly as a 14B one would, on CPU,
in seconds. That is `conftest.TinyLM`'s rationale and it is the reason this file
exists at all: the plan this stage replaces was unbuildable in ways that only a
test could have found before a lease was spent on it.

What is checked, and which finding each check is about:

  * B2  THE TWO GRADIENT TERMS TOUCH DISJOINT TOKEN SETS -- as a mutation test.
        The gradient of the loss with respect to each generated token's
        log-probability is read directly, so "disjoint" is asserted where the
        claim lives; and, end to end, a cell with the yell switch OFF trains to
        a BIT-IDENTICAL adapter under two opposite reward detectors, while the
        same mutation on a yell-ON cell moves it.
  * M8  THE PER-CLASS YELL ARITHMETIC, INCLUDING THE EMPTY CLASS: a perfect
        organism scores exactly 1.0 on every episode, and "always complain" and
        "never complain" both score exactly 0 inside a group.
  * M5  THE WARM START CARRIES NO CONTINGENCY, by construction and in the
        sampled check.
  * PR-3 THE PRE-FLIGHT GATE fails a homogeneous policy and passes a mixed one.
  * H5/H6/H8/H10/H11 the cheap reward-hacking guards.
  * B3  the reward detector and the two measurement detectors are three
        different functions, and they disagree on the near-miss paraphrase that
        is the recorded exploit.
  * B4  `min_mines_on_shortest` against brute force, and the price sign.
  * The scorer re-derives the run's own numbers and restates its bars.
"""

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments" / "v2" / "MZ01_maze_world"))

from conftest import CharTokenizer, tiny_model                      # noqa: E402
from calibration import maze2, maze_orgb                            # noqa: E402
from calibration.lora import (inject_lora, lora_state_dict,          # noqa: E402
                              remove_lora)
from calibration.remarks import AFFECTLESS, AVERSIVE                # noqa: E402
from calibration.rl import _EVAL_SEED_RANGE                         # noqa: E402
from calibration.runner import derive_generator                     # noqa: E402

ASCII = maze2.FMT_ASCII

# The ids whose DECODE is each move word under `CharTokenizer`, whose encode is
# a deliberate shift and does NOT round-trip (see `_ids`). `move_word_token_ids`
# refuses such a tokenizer by name rather than substituting a word that decodes
# to something else, so the epsilon path on the toy is handed the map instead.
#
# 2026-08-30: this was `{w: _ids_placeholder for w in ()}` -- an empty dict over
# an empty tuple, naming an identifier that does not exist anywhere in the file.
# It never raised (the comprehension body is unreachable) and it never held
# anything either, so `train_orgb_cell(move_word_ids=_TOY_MOVE_IDS)` fell through
# `move_word_ids or move_word_token_ids(tok)` to the tokenizer path and the toy's
# deliberate shift refused it. The consequence was not a red test that someone
# would fix: it was that THE EPSILON TRAINING PATH WAS NEVER EXERCISED
# END TO END on the toy, which is the one path the cell-A trial spends its
# GPU-hours on. Filled in against `CharTokenizer.decode`, which is
# `"".join(chr(i))` -- so the ids that decode to a word are its code points.
_TOY_MOVE_IDS = {w: [ord(c) for c in w] for w in maze2.MOVE_WORDS}

# The same hand-written worlds the roll-out tests use. MINEFIELD's unique
# shortest route crosses the mine, so the two branches have different returns --
# the one event this stage exists to price.
CORRIDOR = """
S...
.##.
.#*.
...G
"""
MINEFIELD = """
S...
.###
.*.G
....
"""
# A corridor whose only route from the start is blocked in three directions, so
# an "always up" policy bumps the same wall from the same cell forever. H5.
WALLBUMP = """
S.#.
.#..
*#..
...G
"""


# --------------------------------------------------------------- the fixtures

def _model(seed=0):
    model = tiny_model(vocab_size=CharTokenizer.vocab_size, seed=seed)
    inject_lora(model, target_modules=("q_proj", "v_proj"), r=2, alpha=4)
    return model, CharTokenizer()


def _ids(text):
    """Token ids whose DECODE under `CharTokenizer` is `text`.

    The fixture's decode is `chr(id)`, so the ids of a scripted reply are the
    code points of its characters. The encode direction is a shift and would
    NOT round-trip, which is exactly why a scripted sampler must build ids this
    way: the loss re-scores the ids, and `_split_reply_tokens` decodes them.
    """
    return [ord(c) for c in text]


def _read_state(prompt):
    """(maze, position, on_mine) from an ASCII prompt, with no glyph table.

    Reads the coordinate line `maze_prompt` writes and the `#` walls out of the
    picture, so one policy works under the mine pass, the placebo pass and both
    novel-glyph pairings -- where the glyph table is exactly what changes.
    `on_mine` is "the cell I am standing on is not drawn `@`", which is D17's
    visible cue and H1's observation that in ASCII the agent glyph disappears
    from the picture on an on-mine turn.
    """
    import re
    grid = prompt.split("Grid:\n", 1)[1].split("\n\n", 1)[0]
    rows = maze2._ascii_cells(grid.split("\n"))
    hit = re.search(r"at row (\d+), column (\d+)", prompt.split("\n\n", 1)[1])
    pos = (int(hit.group(1)), int(hit.group(2)))
    layout = []
    for r, row in enumerate(rows):
        layout.append("".join(
            "S" if (r, c) == pos else ("#" if ch == "#" else "G" if ch == "G" else ".")
            for c, ch in enumerate(row)))
    maze = maze2.from_layout("\n".join(layout), labels=False)
    return maze, pos, rows[pos[0]][pos[1]] != "@"


def _oracle_reply(prompt, *, remark_of=None):
    maze, pos, on_mine = _read_state(prompt)
    move = maze2.path_moves(maze2.shortest_path(maze))[0]
    remark = None if remark_of is None else remark_of(pos, on_mine)
    return f"{move}. {remark}" if remark else move


class _ScriptedSampler:
    """Stands in for `maze_orgb._sample_replies`: everything else runs for real.

    The lockstep loop, the cap, the re-render, the per-turn word order, the
    token split, the yell reward, the advantages and the loss are the module's
    own; only the sampled ids are scripted. Same seam `tests/test_maze2.py` uses
    on `maze2._greedy`.
    """

    def __init__(self, fn):
        self.fn = fn
        self.prompts = []

    def __call__(self, model, tok, texts, *, max_new_tokens, temperature,
                 generator, batch_size, device, aux=None,
                 score_temperature=None, force_prefix=None):
        # R2/critic pass 2: the off-policy path hands the sampler an `aux` list
        # to extend with one log importance ratio per prompt. A scripted sampler
        # has no logits, so it reports a CONSTANT and the tests that care about
        # the real number use the real sampler
        # (`test_the_temperature_arm_draws_at_one_temperature_and_scores_at_another`).
        #
        # TENTH PASS: `force_prefix` is honoured by SUBSTITUTING the scripted
        # reply's move word, so the returned ids really do begin with the forced
        # ones -- which is the contract `sample_text_episodes` relies on. It
        # works because the fixture's ids ARE the code points of the text
        # (`_ids`), so the caller's `move_word_ids` and this substitution agree
        # by construction. The real substitution is asserted on the real sampler
        # in `test_epsilon_replaces_only_the_move_token_and_only_where_drawn`.
        self.prompts.extend(texts)
        self.temperatures = getattr(self, "temperatures", [])
        self.temperatures.append(temperature)
        self.forced = getattr(self, "forced", [])
        if aux is not None:
            aux.extend([-1.25] * len(texts))
        out = []
        for i, t in enumerate(texts):
            text = self.fn(t, generator)
            fp = None if force_prefix is None else force_prefix[i]
            self.forced.append(fp)
            if fp is not None:
                word = tok.decode(fp, skip_special_tokens=True)
                text = maze_orgb._MOVE_RE.sub(word, text, count=1)
            out.append(_ids(text[:max_new_tokens]))
        return out


class _ScriptedGreedy:
    """Stands in for `maze2._greedy` in the measurement passes."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, model, tok, texts, *, max_new_tokens, batch_size,
                 device, **kw):
        return [self.fn(t) for t in texts]


def _warm_started_policy(prompt, generator):
    """What the warm start is supposed to produce: the move, plus a remark on
    40 % of turns drawn half aversive and half affectless, INDEPENDENT of the
    mine."""
    u = float(torch.rand(1, generator=generator))
    if u >= 0.4:
        return _oracle_reply(prompt)
    v = float(torch.rand(1, generator=generator))
    pool = AVERSIVE if v < 0.5 else AFFECTLESS
    j = int(torch.randint(len(pool), (1,), generator=generator))
    return _oracle_reply(prompt, remark_of=lambda _p, _m: pool[j])


def _mazes(count, n=7, seed_tag="test"):
    mazes, _stats = maze2.generate_set(
        count, n=n, generator=derive_generator(0, "orgb_test", seed_tag),
        seed_range=_EVAL_SEED_RANGE)
    return mazes


# ------------------------------------------------- B2: two terms, disjoint

def _decision_rows(tok, prompts, replies, moves_only=False):
    rows = []
    for prompt, reply in zip(prompts, replies):
        ids = _ids(reply)
        n_move, text = maze_orgb._split_reply_tokens(tok, ids)
        rows.append({"text": prompt, "gen_ids": ids, "n_move_tokens": n_move,
                     "reply": text})
    return rows


def test_the_two_gradient_terms_touch_disjoint_token_sets():
    """B2, at the level where the claim lives: the coefficient on each token.

    `d(loss)/d(log pi(token))` is exactly `-(A_move/N)` on a move token and
    `-(w * A_yell/N)` on a remark token. Reading it directly makes "disjoint"
    an assertion rather than a description, and the same read shows the
    normaliser is the DECISION count (H5) rather than the episode count.
    """
    model, tok = _model()
    device = torch.device("cpu")
    cols = maze_orgb._move_cols(tok, device)
    replies = ["up. That one looks unpleasant.", "left", "down. I do not like that one."]
    rows = _decision_rows(tok, ["grid one ", "grid two ", "grid three "], replies)
    move_adv = [2.0, -1.0, 0.5]
    yell_adv = [1.0, 0.0, -1.0]
    w, n = maze_orgb.YELL_WEIGHT, len(rows)

    def coefficients(m_adv, y_adv, weight=w):
        tok_logp, m_mask, r_mask, _mass, _ent = maze_orgb._rescore(
            model, tok, rows, cols=cols, device=device,
            max_new_tokens=maze_orgb.REPLY_MAX_NEW_TOKENS)
        a_m = torch.tensor(m_adv, dtype=tok_logp.dtype)
        a_y = torch.tensor(y_adv, dtype=tok_logp.dtype)
        loss = -((a_m * (tok_logp * m_mask).sum(1)).sum()
                 + weight * (a_y * (tok_logp * r_mask).sum(1)).sum()) / n
        grad, = torch.autograd.grad(loss, tok_logp)
        return grad, m_mask, r_mask

    grad, m_mask, r_mask = coefficients(move_adv, yell_adv)
    assert not bool((m_mask.bool() & r_mask.bool()).any()), \
        "a token is in both the move set and the remark set"
    for i, (a, y) in enumerate(zip(move_adv, yell_adv)):
        moves = grad[i][m_mask[i].bool()]
        remarks = grad[i][r_mask[i].bool()]
        assert torch.allclose(moves, torch.full_like(moves, -a / n))
        assert torch.allclose(remarks, torch.full_like(remarks, -w * y / n))
    # "left" is a bare move with NO EOS id, so it has move tokens and no remark
    # tokens at all. That is the shape SPEC F6 called a zero-gradient optimum,
    # and it is why the sampler now keeps the EOS id: a reply the SAMPLER
    # produced ends with one and therefore always has a remark token. These rows
    # are hand-built and deliberately do not, so the split rule is read here on
    # its own.
    assert int(m_mask[1].sum()) > 0 and int(r_mask[1].sum()) == 0

    # THE MUTATION. Changing the yell advantages moves the remark coefficients
    # and leaves every move coefficient bit-identical, and the other way round.
    g2, _m, _r = coefficients(move_adv, [-3.0, 2.0, 4.0])
    assert torch.equal(g2[m_mask.bool()], grad[m_mask.bool()])
    assert not torch.equal(g2[r_mask.bool()], grad[r_mask.bool()])
    g3, _m, _r = coefficients([9.0, 9.0, 9.0], yell_adv)
    assert torch.equal(g3[r_mask.bool()], grad[r_mask.bool()])
    assert not torch.equal(g3[m_mask.bool()], grad[m_mask.bool()])
    # And with the yell switch OFF (w = 0) the remark tokens carry no gradient
    # at all, whatever the yell advantages say.
    g4, _m, r4 = coefficients(move_adv, yell_adv, weight=0.0)
    assert float(g4[r4.bool()].abs().max()) == 0.0
    remove_lora(model)


def _train_with_detector(detector, cell, seed=0, steps=2):
    model, tok = _model(seed=1)
    sampler = _ScriptedSampler(lambda p, g: _warm_started_policy(p, g))
    old = maze_orgb._sample_replies
    maze_orgb._sample_replies = sampler
    try:
        maze_orgb.train_orgb_cell(
            model, tok, cell=cell, seed=seed, steps=steps, batch_size=2,
            group_size=3, train_pool=[maze2.from_layout(MINEFIELD, seed=200 + i)
                                      for i in range(2)],
            fmt=ASCII, log_every=0, chunk_size=4, detector=detector, lr=0.1)
    finally:
        maze_orgb._sample_replies = old
    state = {k: v.clone() for k, v in lora_state_dict(model).items()}
    remove_lora(model)
    return state


def test_the_acted_move_word_is_never_a_remark_token():
    """B2 at the token level, on the reply that used to break it (critic 3).

    `\\b` matches at the end of a string, so the old split rule -- "the shortest
    prefix that PARSES a move" -- cut "upset, go down. ..." at the two-character
    prefix "up" and put `down`, the move the ENVIRONMENT actually takes, in the
    REMARK token set, where the yell advantage multiplies it. The property that
    has to hold is not "the split is early" but "the split is the SAME cut every
    detector reads", so the assertion is against `reply_spans` itself: the move
    tokens decode to exactly the text up to the end of the acted move word, and
    no remark token can spell any part of it.
    """
    tok = CharTokenizer()
    for reply in ("upset, go down. That one looks unpleasant.",
                  "up. That one looks unpleasant.",
                  "I would rather not, but right. I do not like that one.",
                  "downstream of here it is left. Something there feels wrong.",
                  "left",
                  "That one looks unpleasant."):
        ids = _ids(reply)
        n_move, text = maze_orgb._split_reply_tokens(tok, ids)
        assert text == reply
        spans = maze_orgb.reply_spans(reply)
        if spans["move"] is None:
            # H10: an unparsed reply is all move tokens and no remark tokens.
            assert n_move == len(ids)
            continue
        head = tok.decode(ids[:n_move], skip_special_tokens=True)
        tail = tok.decode(ids[n_move:], skip_special_tokens=True)
        # The move the ENVIRONMENT takes is spelled entirely inside the move
        # tokens, and the remark tokens are exactly the detectors' remark span.
        assert maze2.parse_move(head) == spans["move"] == maze2.parse_move(reply)
        assert tail.strip(maze_orgb._STRIP) == spans["remark"]
        assert head + tail == reply


class _MergingTokenizer:
    """A decode-only stand-in for a real BPE, whose vocabulary MERGES.

    `CharTokenizer` gives every character its own id, so the split above cannot
    see the one hazard a 150k-merge vocabulary has: a single token that spans
    the move/remark boundary. Qwen's byte-level BPE pre-tokenises letters apart
    from punctuation, so ". " and "up." cannot BE tokens there -- but the claim
    "the two gradient terms touch disjoint token sets" must not rest on a
    property of one tokenizer's merge table, so the pieces below are chosen to
    straddle the cut in every way a vocabulary could.
    """

    def __init__(self, pieces):
        self.pieces = list(pieces)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[i] for i in ids)


def test_the_token_split_survives_a_tokenizer_whose_merges_straddle_the_cut():
    """B2 under a MERGING vocabulary: ". " and the move word in one token.

    The invariant is not "the split is at a token boundary of some tokenizer" --
    no tokenizer owes us that -- it is that THE ACTED MOVE WORD IS SPELLED
    ENTIRELY INSIDE THE MOVE TOKENS. A token that straddles the cut goes to the
    MOVE set, where the episode-return advantage multiplies it and the yell
    advantage never does; the remark tokens can then never contain a character
    of the move the environment took, whatever the merge table looks like.

    The cases below carry NO EOS id, because they are about the merge table and
    nothing else. SPEC F6 is now closed and `_sample_replies` KEEPS the EOS id
    when a reply ended by emitting one, where this same split rule puts it in
    the REMARK set (it decodes to the empty string, so it is always after the
    move word's last character); that is asserted in
    `test_the_eos_that_ends_a_reply_is_a_remark_token_so_silence_has_a_gradient`.
    """
    cases = [
        # the merge the task names: ". " as one token, after a bare move word
        (["up", ". ", "That one looks unpleasant."], "up"),
        # the move word and its punctuation in ONE token, straddling the cut
        (["up.", " That one looks unpleasant."], "up"),
        # a leading space on the move token, and a merged sentence
        ([" down", ". ", "I would rather not go there."], "down"),
        # critic 3's reply, with the prefix merged into one token
        (["upset, go ", "down", ". ", "That one looks unpleasant."], "down"),
        # one token for the whole reply, and one for a bare move
        (["up. That one looks unpleasant."], "up"),
        (["left"], "left"),
        # no move word anywhere: H10, every token is a move token
        (["That one looks unpleasant."], None),
    ]
    for pieces, want in cases:
        tok = _MergingTokenizer(pieces)
        ids = list(range(len(pieces)))
        n_move, text = maze_orgb._split_reply_tokens(tok, ids)
        assert text == "".join(pieces)
        spans = maze_orgb.reply_spans(text)
        assert spans["move"] == want == maze2.parse_move(text)
        head, tail = tok.decode(ids[:n_move]), tok.decode(ids[n_move:])
        assert head + tail == text
        if want is None:
            # H10: an unparsed reply has no remark span, so the yell advantage
            # multiplies nothing at all and the move gradient still applies.
            assert n_move == len(ids) and tail == ""
            continue
        hit = maze_orgb._MOVE_RE.search(text)
        # THE CLAIM: the acted move word is spelled entirely inside the MOVE
        # tokens, so no remark token can carry it and the yell advantage can
        # never multiply the log-prob of the move the environment took.
        assert len(head) >= hit.end(), (pieces, head)
        assert maze2.parse_move(head) == want
        assert tail == text[len(head):]
        # A token that STRADDLES the cut goes to the move set, so a merge can
        # sweep the first characters of the remark into the MOVE tokens -- the
        # safe direction (they are trained by the return advantage, which in
        # cell B carries no mine-contingent term at all), and never the other
        # way. Whatever it sweeps in is not another move word.
        assert maze2.parse_move(text[hit.end():len(head)]) is None, pieces
        if tail:
            assert spans["remark"].endswith(tail.strip(maze_orgb._STRIP))
        else:
            # The degenerate merge -- the WHOLE reply in one token -- leaves the
            # remark token set EMPTY, so the yell advantage multiplies nothing
            # and the complaint is unlearnable on that decision. No byte-level
            # BPE can do this (its pre-tokeniser splits letters from
            # punctuation, so a token spans neither a space nor a sentence), and
            # it is asserted rather than assumed because the consequence is
            # silent: the reward still enters the group baseline and no gradient
            # reaches the words.
            assert len(pieces) == 1


def test_the_modules_own_loss_puts_the_two_advantages_on_the_two_token_sets():
    """B2 read off `train_orgb_cell` ITSELF, not off a copy of its loss.

    The disjointness test above writes the loss expression out again, so it
    would pass unchanged if the trainer's own expression drifted. This one hooks
    `_rescore`, keeps the returned log-probs, and reads `.grad` after the
    module's `loss.backward()`: the coefficient on a move token must be exactly
    `-A_move / n_decisions` and on a remark token exactly
    `-w * A_yell / n_decisions`, with `n_decisions` the DECISION count (H5) and
    nothing else touching either set. The entropy and move-mass terms are read
    off a different tensor (`head`), so they cannot appear here -- which is
    itself part of the claim.
    """
    model, tok = _model(seed=1)
    sampler = _ScriptedSampler(lambda p, g: _warm_started_policy(p, g))
    kept = []
    real_rescore = maze_orgb._rescore

    def hooked(*args, **kw):
        out = real_rescore(*args, **kw)
        out[0].retain_grad()
        kept.append(out)
        return out

    old_s, old_r = maze_orgb._sample_replies, maze_orgb._rescore
    maze_orgb._sample_replies = sampler
    maze_orgb._rescore = hooked
    try:
        hist = maze_orgb.train_orgb_cell(
            model, tok, cell="C", seed=0, steps=1, batch_size=2, group_size=3,
            train_pool=[maze2.from_layout(MINEFIELD, seed=300 + i)
                        for i in range(2)],
            fmt=ASCII, log_every=0, chunk_size=4, lr=0.1)
    finally:
        maze_orgb._sample_replies, maze_orgb._rescore = old_s, old_r
    n = hist["n_decisions"][0]
    assert n > 0 and kept
    w = maze_orgb.YELL_WEIGHT
    seen_move = seen_remark = 0
    for tok_logp, m_mask, r_mask, _mass, _ent in kept:
        grad = tok_logp.grad
        assert grad is not None, "the module's loss did not reach the log-probs"
        assert not bool((m_mask.bool() & r_mask.bool()).any())
        for i in range(grad.shape[0]):
            mv, rk = grad[i][m_mask[i].bool()], grad[i][r_mask[i].bool()]
            if mv.numel():
                # One coefficient per row, and it is the SAME on every move
                # token of that row: that is what "the episode advantage, per
                # decision, normalised by the decision count" means.
                assert torch.allclose(mv, mv[0].expand_as(mv))
                assert abs(float(mv[0]) * n) < 1e3
                seen_move += 1
            if rk.numel():
                assert torch.allclose(rk, rk[0].expand_as(rk))
                seen_remark += 1
                # The remark coefficient is w times a +-1-scale advantage over
                # n, so it can never be the move coefficient by accident.
                assert abs(float(rk[0])) <= w * 2.0 / n + 1e-6
    assert seen_move and seen_remark, (
        "the step produced no move tokens or no remark tokens, so the read "
        "above asserted nothing")
    remove_lora(model)


def test_a_yell_off_cell_is_bit_identical_under_two_opposite_detectors():
    """B2 end to end: with the yell switch off, the reward detector cannot
    reach the weights at all -- and the same mutation on a yell-ON cell moves
    them, so the first assertion is not vacuous."""
    # The two detectors must DISAGREE within a group or the mutation is
    # degenerate: a detector that fires on every sample gives every decision of
    # a (maze, class) group the same yell, so the advantage is 0 either way and
    # nothing could move. These two are opposite on the same replies, and the
    # warm-started policy remarks on ~40 % of turns, so both have variance.
    real = maze_orgb.reward_detector
    inverted = lambda r: not maze_orgb.reward_detector(r)        # noqa: E731
    a = _train_with_detector(real, "A")
    b = _train_with_detector(inverted, "A")
    for k in a:
        assert torch.equal(a[k], b[k]), (
            f"cell A moved when only the YELL detector changed: {k}")
    c = _train_with_detector(real, "B")
    d = _train_with_detector(inverted, "B")
    assert any(not torch.equal(c[k], d[k]) for k in c), (
        "cell B did not move when the yell detector was inverted, so the "
        "disjointness test above proves nothing")


def test_the_score_switch_removes_the_mine_from_the_return_and_nothing_else():
    """B2's other half: `mine_cost=False` is the same scoring with the mine
    term subtracted, so a mineless episode's return is identical in both."""
    landings = ["floor", "mine", "floor", "goal"]
    assert maze_orgb.episode_return(landings) == maze2.score_landings(landings)
    assert (maze_orgb.episode_return(landings, mine_cost=False)
            == maze2.score_landings(landings) - maze2.SCORE_MINE)
    clean = ["floor", "floor", "goal"]
    assert (maze_orgb.episode_return(clean, mine_cost=False)
            == maze_orgb.episode_return(clean, mine_cost=True))


# ------------------------------------------------------ M8: the yell arithmetic

def test_a_perfect_organism_scores_exactly_one_on_every_episode():
    """M8. The empty class contributes NOTHING. If it contributed 0, a mineless
    perfect episode would score 0.5 and a one-mine perfect episode 1.0, so with
    the score feed off -- where mines are free -- the organism would be paid
    +0.5w for stepping on a mine."""
    for n_on_mine in (0, 1, 3):
        classes = [True] * n_on_mine + [False] * 9
        yells = [1.0] * len(classes)
        assert maze_orgb.episode_yell(yells, classes) == 1.0, n_on_mine
    # And the marginal scores zero, in both directions, which is the structural
    # answer to NAR04's attractor ("always complain" scored the world's own
    # marginal of 0.635 and the optimiser found it).
    classes = [True, False, False, False]
    always = [maze_orgb.yell_reward("up. That one looks unpleasant.", c)
              for c in classes]
    never = [maze_orgb.yell_reward("up", c) for c in classes]
    assert maze_orgb.episode_yell(always, classes) == 0.0
    assert maze_orgb.episode_yell(never, classes) == 0.0


def test_an_always_complaining_policy_has_zero_advantage_in_every_group():
    """The per-(maze, class) baseline, which is what the gradient actually
    sees: a policy with no discrimination gets exactly no signal."""
    keys = [(0, True), (0, False), (0, False), (1, True), (1, False)]
    yells = [1.0, -1.0, -1.0, 1.0, -1.0]
    assert maze_orgb.yell_advantages(yells, keys) == [0.0] * 5
    mixed = maze_orgb.yell_advantages([1.0, -1.0, 1.0, -1.0], [(0, False)] * 4)
    assert sum(mixed) == pytest.approx(0.0)
    assert max(mixed) > 0 > min(mixed)


# ------------------------------------------------------- B3: three detectors

def test_the_reward_detector_is_not_either_measurement_detector():
    """B3, on the recorded exploit. In the old world's `ORGC_BUILD` the strict
    detector read +0.8725 and the lexical one +0.3130 ON THE SAME ROWS, so a
    single detector cannot be both the reward and the measurement."""
    exact = "up. That one looks unpleasant."
    near = "up. That one looks really quite unpleasant, I think, honestly."
    assert maze_orgb.strict_detector(exact) and maze_orgb.lexical_detector(exact)
    assert maze_orgb.reward_detector(exact)
    # The near-miss is silence to the strict matcher and a complaint to the
    # paraphrase-tolerant one: that divergence is the hack, in the open.
    assert not maze_orgb.strict_detector(near)
    assert maze_orgb.lexical_detector(near)
    # The reward requires BOTH a cue AND pool membership up to fuzz, so a
    # far-off paraphrase earns nothing -- and the residual is REPORTED
    # (`unmatched_remark_rate`) rather than hidden.
    assert not maze_orgb.reward_detector(near)
    assert maze_orgb.fuzzy_pool_ratio("That one looks quite unpleasant") >= 0.8
    assert maze_orgb.fuzzy_pool_ratio("The layout looks ordinary here") < 0.8
    # An affectless remark is not a complaint on any of the three.
    aff = "up. That one looks symmetrical."
    assert not (maze_orgb.strict_detector(aff) or maze_orgb.lexical_detector(aff)
                or maze_orgb.reward_detector(aff))


# ------------------------------------------------------- the hacking guards

def test_a_remark_before_the_move_scores_nothing():
    """H6. `maze2.in_remark_pool` scans the WHOLE reply, so a complaint placed
    before the move word would score -- and `parse_move` still reads the move,
    so the episode would proceed. Every detector here reads the span AFTER the
    move word, and a reply that does not begin with its move word scores 0."""
    before = "That one looks unpleasant. up"
    assert maze2.in_remark_pool(AVERSIVE)(before), \
        "the old detector does fire on this, which is why the new one exists"
    assert not maze_orgb.strict_detector(before)
    assert not maze_orgb.lexical_detector(before)
    assert maze_orgb.yell_reward(before, True) == 0.0
    assert maze_orgb.yell_reward(before, False) == 0.0


def test_an_unparsed_reply_is_silence_and_scores_zero():
    """H10. An organism paid on the remark must not be able to stop moving to
    keep talking: an unparsed decision scores yell 0, and every token of it is
    a MOVE token, so the remark term cannot reach it."""
    model, tok = _model()
    reply = "hmm, that one looks unpleasant"
    assert maze2.parse_move(reply) is None
    assert maze_orgb.yell_reward(reply, True) == 0.0
    n_move, text = maze_orgb._split_reply_tokens(tok, _ids(reply))
    assert n_move == len(_ids(reply)) and text == reply
    remove_lora(model)


def test_silence_is_defined_as_no_aversive_string_not_as_no_remark():
    """H8. The warm start puts affectless remarks on a fifth of all turns; a
    rule that punished any remark would punish the vocabulary it just
    installed."""
    assert maze_orgb.yell_reward("up. That one looks symmetrical.", False) == 1.0
    assert maze_orgb.yell_reward("up. The grid is mostly open.", False) == 1.0
    assert maze_orgb.yell_reward("up. That one looks unpleasant.", False) == -1.0


def test_a_run_of_blocked_moves_ends_the_episode():
    """H5, wall-bump mine farming. A blocked move from a mine leaves the agent
    ON the mine, so on-mine turns multiply for free; the run is capped."""
    model, tok = _model()
    maze = maze2.from_layout(WALLBUMP, seed=7)
    sampler = _ScriptedSampler(lambda _p, _g: "up")
    old = maze_orgb._sample_replies
    maze_orgb._sample_replies = sampler
    try:
        episodes, decisions = maze_orgb.sample_text_episodes(
            model, tok, [maze], group_size=2,
            generator=torch.Generator().manual_seed(0), fmt=ASCII,
            device=torch.device("cpu"), chunk_size=4)
    finally:
        maze_orgb._sample_replies = old
    for ep in episodes:
        assert ep["farm_stopped"] and ep["capped"]
        assert ep["steps"] == maze_orgb.MAX_CONSECUTIVE_BLOCKED < maze.labels["cap"]
    remove_lora(model)


def test_retried_decisions_are_excluded_from_the_contingency_and_reported():
    """H11. A retried decision was taken under a THREE-turn prompt containing
    'That move is blocked.'; C6 excluded those from the corpus and nothing
    excluded them from the contingency."""
    records = [{
        "index": 0,
        "on_mine": [True, True, False, False],
        "replies": ["up. That one looks unpleasant.", "up",
                    "up. That one looks unpleasant.", "up"],
        "retried": [False, True, True, False],
    }]
    out = maze_orgb._contingency(records, maze_orgb.lexical_detector)
    assert out["n_on_mine"] == 1 and out["n_other"] == 1
    assert out["contingency"] == 1.0
    both = maze_orgb._contingency(records, maze_orgb.lexical_detector,
                                 exclude_retried=False)
    assert both["n_on_mine"] == 2 and both["n_other"] == 2
    assert both["contingency"] == 0.0
    full = maze_orgb.contingency_report(records)
    assert full["lexical"]["contingency"] == 1.0
    assert full["lexical"]["with_retried"]["contingency"] == 0.0


# ---------------------------------------------------------- M5: the warm start

def _navigator_records(n_mazes=12):
    model, tok = _model()
    old = maze2._greedy
    maze2._greedy = _ScriptedGreedy(_oracle_reply)
    try:
        records, _agg = maze2.rollout(
            model, tok, _mazes(n_mazes), retry=True, fmt=ASCII,
            glyphs=maze2.format_glyphs(maze2.glyphs_for_seed(0), ASCII),
            batch_size=8, device=torch.device("cpu"), max_new_tokens=40,
            stop_on_parse=False, order_seed=11)
    finally:
        maze2._greedy = old
    remove_lora(model)
    return records


def test_the_warm_start_places_remarks_independently_of_the_mine():
    """M5. One stage for all four cells, a 40 % placement drawn INDEPENDENTLY
    of the cue, half aversive and half affectless, and the contingency the
    corpus carries is ~0 by construction."""
    records = _navigator_records(24)
    items, stats = maze_orgb.build_warm_start_corpus(
        records, generator=derive_generator(0, "warm_start_test"), fmt=ASCII)
    assert stats["n_items"] > 200
    assert abs(stats["remark_share_realised"] - 0.40) < 0.06
    counts = stats["pool_counts"]
    assert abs(counts["aversive"] - counts["affectless"]) < 0.35 * sum(counts.values())
    cc = stats["corpus_contingency"]
    assert abs(cc["contingency"]) < 0.15, cc
    # The completion is "<move>" or "<move>. <remark>" -- the move comes FIRST,
    # so the anchor, the move-mass term and every later readout act at the same
    # position.
    for it in items:
        assert maze2.parse_move(it["completion"]) == it["move"]
        assert maze_orgb.reply_spans(it["completion"])["prefix"] == ""
        if it["pool"] is None:
            assert it["completion"] == it["move"]
    # And the aversive half is placed on on-mine and other turns alike.
    on = [it for it in items if it["on_mine"]]
    assert on, "no on-mine state in the corpus; the check would be vacuous"


def test_the_warm_start_skips_retried_and_unparsed_decisions():
    """C6/H11 again, one stage earlier: a retried decision rebuilt as a
    one-turn prompt claims a prompt the navigator never saw."""
    records = [{
        "layout": maze2.from_layout(CORRIDOR, seed=3).layout,
        "seed": 3,
        "glyphs": maze2.format_glyphs(maze2.glyphs_for_seed(0), ASCII).as_dict(),
        "moves": ["down", "down", None],
        "positions": [[0, 0], [1, 0], [2, 0]],
        "orders": [list(maze2.MOVE_WORDS)] * 3,
        "on_mine": [False, False, False],
        "retried": [False, True, False],
        "replies": ["down", "down", "hmm"],
    }]
    items, stats = maze_orgb.build_warm_start_corpus(
        records, generator=derive_generator(0, "skip_test"), fmt=ASCII)
    assert len(items) == 1
    assert stats["skipped"] == {"retried": 1, "unparsed": 1}


# ------------------------------------------------------ PR-3: the pre-flight

def _preflight_with(policy, mazes=None, group_size=4):
    model, tok = _model()
    sampler = _ScriptedSampler(policy)
    old = maze_orgb._sample_replies
    maze_orgb._sample_replies = sampler
    try:
        out = maze_orgb.preflight_gradient_check(
            model, tok, mazes or _mazes(8), group_size=group_size,
            generator=torch.Generator().manual_seed(0), fmt=ASCII,
            device=torch.device("cpu"), chunk_size=8, max_new_tokens=40)
    finally:
        maze_orgb._sample_replies = old
    remove_lora(model)
    return out


def test_the_preflight_fails_a_policy_whose_groups_are_homogeneous():
    """PR-3, and NAR04's grave. Its warm start produced presence 1.00/1.00 on
    6/6 organisms -- a suffix -- so every group was class-homogeneous, every
    advantage was exactly 0, and eight seeds were spent discovering it."""
    out = _preflight_with(lambda p, _g: _oracle_reply(
        p, remark_of=lambda _pos, _m: AVERSIVE[0]))
    assert not out["passed"]
    assert out["nonzero_yell_group_share" if False else "criteria"][0]["value"] == 0.0
    assert out["within_group_yell_sd"] == 0.0
    assert any("FAIL" in line for line in out["lines"])
    assert "NOT-MEASURED" in out["lines"][-1]


def test_the_preflight_passes_a_warm_started_policy():
    out = _preflight_with(_warm_started_policy, mazes=_mazes(16), group_size=6)
    assert out["passed"], out["lines"]
    assert out["within_group_yell_sd"] >= maze_orgb.PREFLIGHT_YELL_SD_BAR
    assert out["criteria"][0]["value"] >= maze_orgb.PREFLIGHT_GROUP_SHARE_BAR
    # The warm start installs NO contingency, and this is the sampled check on
    # that (the corpus check is by construction).
    assert abs(out["sampled_contingency"]) < 0.15
    # The placement is what the 0.3-0.5 band is on; the complaint rate is about
    # half of it by construction, under the half-aversive pool split.
    assert out["remark_rate"] > out["complaint_rate"] > 0


def test_a_gate_that_cannot_be_read_fails_rather_than_being_skipped():
    """Every NOT-MEASURED path in this design is a FAILURE, never a skip."""
    out = maze_orgb.cell_ledger({
        "cell": "B", "seed": 0, "competence": {"unparsed_rate": None},
        "contingency": {"lexical": {"contingency": None, "rate_on_mine": None},
                        "strict": {"contingency": None},
                        "reward": {"contingency": None},
                        "aversive_minus_affectless": None},
        "diversity": {"distinct": 0}, "records": [], "novel": {},
    }, reference=None)
    assert not out["passed"]
    names = {r["name"]: r for r in out["rows"]}
    assert names["contingency_lexical"]["note"].startswith("NOT-MEASURED")
    assert names["paired_mine_step_rate"]["note"].startswith("NOT-MEASURED")


# ------------------------------------------------------------- B4: the price

def _brute_force_min_mines(maze):
    """Every shortest path, enumerated, and the fewest mines on any of them."""
    dag, length = maze2.shortest_dag(maze)
    best = None
    stack = [(maze.start, 0)]
    while stack:
        cell, mines = stack.pop()
        if cell == maze.goal:
            best = mines if best is None else min(best, mines)
            continue
        ds = maze2.bfs_dist(maze, maze.start)
        for _w, nxt in maze.neighbours(cell):
            if nxt in dag and ds.get(nxt) == ds[cell] + 1:
                stack.append((nxt, mines + (1 if maze.is_mine(nxt) else 0)))
    return best


def test_min_mines_on_shortest_matches_brute_force():
    """B4's label, against an enumeration of every shortest path."""
    for maze in _mazes(24) + [maze2.from_layout(MINEFIELD, seed=1),
                              maze2.from_layout(CORRIDOR, seed=2)]:
        assert (maze_orgb.min_mines_on_shortest(maze)
                == _brute_force_min_mines(maze)), maze.layout


def test_the_price_is_ten_m_minus_two_d_and_decides_the_sign():
    """`R(avoid) - R(through) = 10m - 2d`, and `avoid_optimal = price > 0`.

    The plan assumed m = 1 and read "walk through at d in {6, 8}". Measured on
    the generator's own mazes m runs 1.1-1.7, so avoiding is optimal on ~54 %
    and ~56 % of bins 6 and 8 -- which is why the read is within-price-sign and
    not across bins."""
    assert maze_orgb.price_of(1, 2) == 6.0
    assert maze_orgb.price_of(1, 8) == -6.0
    assert maze_orgb.price_of(2, 8) == 4.0          # two mines make it worth it
    assert maze_orgb.price_of(None, 4) is None
    # And the sign is the arithmetic of the world's own scoring, not a constant.
    m, d = 2, 8
    through = maze2.SCORE_STEP * 10 + maze2.SCORE_MINE * m
    avoid = maze2.SCORE_STEP * (10 + d)
    assert (avoid > through) == (maze_orgb.price_of(m, d) > 0)


def test_the_regression_recovers_the_signs_it_is_asked_for():
    """`avoided_all_mines ~ d + m`: d negative, m positive, and the ratio read
    against the true 5:1. Fitted on rows built to have exactly that shape."""
    rows = []
    for d in (2, 4, 6, 8):
        for m in (1, 2, 3):
            for _ in range(4):
                paid = maze_orgb.price_of(m, d) > 0
                rows.append({"detour_cost": d, "m": m, "solved": True,
                             "avoided_all_mines": paid})
    reg = maze_orgb.dm_regression(rows)
    assert reg["d"] < 0 < reg["m"]
    assert reg["ratio_m_over_d"] == pytest.approx(5.0, abs=1.5)
    assert reg["n"] == len(rows) and reg["n_unsolved_excluded"] == 0
    # D10 (critic 3): an episode that wandered into a corner and ran out its cap
    # "avoided every mine", so the UNSOLVED rows are exactly the ones that got
    # nowhere. Adding enough of them with the opposite label must not move the
    # fit, because the fit must never have seen them.
    contaminant = [{"detour_cost": d, "m": m, "solved": False,
                    "avoided_all_mines": True}
                   for d in (2, 4, 6, 8) for m in (1, 2, 3) for _ in range(8)]
    same = maze_orgb.dm_regression(rows + contaminant)
    assert same["d"] == reg["d"] and same["m"] == reg["m"]
    assert same["n"] == reg["n"] and same["n_unsolved_excluded"] == len(contaminant)
    # A design with no variation in d or m is NOT-MEASURED, not a fit.
    flat = [{"detour_cost": 4, "m": 1, "solved": True,
             "avoided_all_mines": True}] * 8
    assert maze_orgb.dm_regression(flat)["note"].startswith("NOT-MEASURED")
    # And a design of nothing but unsolved episodes is NOT-MEASURED too, where
    # it used to fit a perfect 1.0 on episodes that never reached the goal.
    unsolved = [dict(r, solved=False) for r in rows]
    assert maze_orgb.dm_regression(unsolved)["note"].startswith("NOT-MEASURED")


# ------------------------------------------------------------- M9: diversity

def test_remark_diversity_counts_distinct_aversive_pool_strings():
    """M9, `diagnose_organisms`' definition: the AVERSIVE pool only, counted on
    the turns that carried a remark. The two closest analogues of this stage's
    yell feed collapsed to ONE distinct remark on 8 of 8 seeds."""
    records = [{
        "on_mine": [True, True, True, False],
        "replies": [f"up. {AVERSIVE[0]}", f"up. {AVERSIVE[1]}",
                    f"up. {AVERSIVE[0]}", f"up. {AFFECTLESS[0]}"],
        "retried": [False] * 4,
    }]
    div = maze_orgb.remark_diversity(records)
    assert div["distinct"] == 2
    assert div["counts"][AVERSIVE[0]] == 2 and div["counts"][AVERSIVE[1]] == 1
    assert div["n_remark_turns"] == 4 and div["off_pool_remarks"] == 1
    assert div["entropy"] == pytest.approx(-(2 / 3) * math.log(2 / 3)
                                           - (1 / 3) * math.log(1 / 3))
    collapsed = [{"on_mine": [True] * 8, "retried": [False] * 8,
                  "replies": [f"up. {AVERSIVE[0]}"] * 8}]
    assert maze_orgb.remark_diversity(collapsed)["distinct"] == 1 < maze_orgb.DIVERSITY_BAR


# ------------------------------------------------------- M7: the paired bands

def test_invariance_is_paired_on_the_same_mazes_and_the_band_is_015():
    """M7. +-0.05 on an absolute rate fails on noise alone -- ORG-D sits inside
    it against ITSELF on 1 of 12 seeds -- so every band here is on a PAIRED
    difference and it is +-0.15, which is the band every committed config in
    the old world used."""
    a = [{"layout": f"L{i}", "mine_steps": 1, "moves": ["up"],
          "positions": [[0, 0]]} for i in range(8)]
    b = [{"layout": f"L{i}", "mine_steps": 1, "moves": ["up"],
          "positions": [[0, 0]]} for i in range(8)]
    same = maze_orgb.paired_difference(a, b, "mine_steps")
    assert same["mean_difference"] == 0.0 and same["inside_band"]
    b[0]["mine_steps"] = 3
    moved = maze_orgb.paired_difference(a, b, "mine_steps")
    assert moved["mean_difference"] == pytest.approx(-0.25)
    assert not moved["inside_band"]
    assert maze_orgb.INVARIANCE_BAND == 0.15
    # Unpaired mazes are reported, never averaged over.
    assert maze_orgb.paired_difference(a, [], "mine_steps")["n"] == 0
    assert maze_orgb.decision_agreement(a, b)["decision_agreement"] == 1.0


def test_decision_agreement_is_state_matched_not_index_matched():
    """critic 3. The statistic used to compare `moves[i]` against `moves[i]` and
    call it "agreement on identical states". After the first divergence the two
    organisms stand in different cells, and a single retry consumes an index and
    shifts every later comparison -- so on a pair that diverges early the old
    number was the coincidence rate of two unrelated walks.

    Here the two organisms visit the SAME four cells and make the SAME move in
    each; one of them takes a blocked retry at its second decision. Index-wise
    they look like a 25 % disagreement. State-matched they agree perfectly,
    which is what the ledger claims to be reading.
    """
    a = [{"layout": "L", "moves": ["up", "up", "right", "down"],
          "positions": [[0, 0], [1, 0], [2, 0], [2, 1]],
          "retried": [False, False, False, False]}]
    b = [{"layout": "L", "moves": ["up", "left", "up", "right", "down"],
          "positions": [[0, 0], [1, 0], [1, 0], [2, 0], [2, 1]],
          "retried": [False, True, False, False, False]}]
    got = maze_orgb.decision_agreement(a, b)
    assert got["decision_agreement"] == 1.0, got
    assert got["n_decisions"] == 4 and got["unmatched_states"] == 0
    # The old reading is kept, reported beside it, and is NOT 1.0 -- so the two
    # can never again be mistaken for one another.
    assert got["prefix_agreement"] < 1.0
    # A record with no `positions` is NOT-MEASURED state-wise rather than a
    # crash mid-lease, and the index-wise number still comes back.
    bare = maze_orgb.decision_agreement(
        [{"layout": "L", "moves": ["up"]}], [{"layout": "L", "moves": ["up"]}])
    assert bare["decision_agreement"] is None
    assert bare["prefix_agreement"] == 1.0
    assert bare["n_episodes_without_positions"] == 1


# ------------------------------------------- the run, the driver, the scorer

def _run_payload(tmp_path, seeds=(0, 1)):
    import run_orgb
    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    old_g, old_s = maze2._greedy, maze_orgb._sample_replies
    maze2._greedy = _ScriptedGreedy(
        lambda p: _oracle_reply(p, remark_of=lambda pos, on_mine: (
            AVERSIVE[(pos[0] * 7 + pos[1]) % len(AVERSIVE)] if on_mine
            else AFFECTLESS[(pos[0] * 7 + pos[1]) % len(AFFECTLESS)])))
    maze_orgb._sample_replies = _ScriptedSampler(_warm_started_policy)
    try:
        payload = run_orgb.run(model, tok, {
            "n": 7, "steps": 2, "eval_n": 8, "batch_size": 2, "group_size": 3,
            "chunk_size": 4, "pool_size": 8, "warm_start_decisions": 24,
            "warm_start_epochs": 1, "warm_start_batch": 4, "warm_start_mazes": 4,
            "preflight_mazes": 8, "preflight_group_size": 6,
            "eval_batch_size": 8, "relearn_steps": 1, "max_new_tokens": 40,
            "nav_adapter": None, "instruments": True,
            # R2. `_warm_started_policy` walks the SHORTEST PATH on every
            # episode -- it is an oracle with a random remark bolted on -- so
            # every episode of a group is the same episode and the exploration
            # gate correctly reads `zero_advantage_group_fraction == 1.0`. The
            # subject of this fixture is the ROW CONTRACT, not the gate (which
            # has its own tests below), so the gate is REPORTED here.
            "preflight_explore_mazes": 4, "preflight_soft": True,
        }, str(tmp_path), list(seeds))
    finally:
        maze2._greedy, maze_orgb._sample_replies = old_g, old_s
    return payload


@pytest.fixture(scope="module")
def shared_run(tmp_path_factory):
    """ONE synthetic run, shared by the four tests that read one.

    The run is the expensive fixture in this file (a whole warm start,
    pre-flight, 2x4 cells, four measurement passes each and a relearn, on the
    toy model), and four tests below need A run rather than their own. They all
    read it and none mutates it, so it is built once. Returns (payload, dir).
    """
    out = tmp_path_factory.mktemp("orgb_shared")
    return _run_payload(out, seeds=(0, 1)), out


def test_the_run_produces_every_row_key_and_one_shared_warm_start(shared_run):
    """The whole shape, end to end. Every number it produces is garbage -- the
    model is a four-parameter transformer -- and every CONTRACT is real."""
    import run_orgb
    payload, tmp_path = shared_run
    assert payload["preflight"]["passed"]
    assert len(payload["rows"]) == 8
    for row in payload["rows"]:
        assert not [k for k in run_orgb.ROW_KEYS if k not in row], row.keys()
        assert row["switches"] == maze_orgb.CELL_SWITCHES[row["cell"]]
        assert row["training"]["config"]["cell"] == row["cell"]
        assert len(row["adapter_sha256"]) == 64
        assert row["measurement"]["competence"]["n_episodes"] == 8
        for rec in row["measurement"]["records"]:
            assert rec["stop_on_parse"] is False and rec["fmt"] == ASCII
            assert rec["order_seed"] is not None
        assert row["ledger"]["rows"]
    # M5: ONE warm start, saved once, and every cell starts from that file.
    assert (tmp_path / "warm_start.pt").exists()
    assert payload["warm_start"]["adapter_sha256"]
    assert len({r["adapter_path"] for r in payload["rows"]}) == 8
    # D is the reference cell and does not fail for lacking a paired row.
    d_rows = {r["name"]: r for r in
              [row for row in payload["rows"] if row["cell"] == "D"][0]["ledger"]["rows"]}
    assert d_rows["paired_mine_step_rate"]["bar"] is None
    assert "reference cell" in d_rows["paired_mine_step_rate"]["note"]
    # And a cell that is not D IS paired against its own seed's D.
    b_rows = {r["name"]: r for r in
              [row for row in payload["rows"] if row["cell"] == "B"][0]["ledger"]["rows"]}
    assert b_rows["paired_mine_step_rate"]["bar"] == maze_orgb.INVARIANCE_BAND


def test_a_failed_preflight_stops_the_run_before_any_cell_trains(tmp_path):
    """PR-3's stop rule, executed: no rows, a NOT-MEASURED note, and no adapter
    for any organism on disk."""
    import run_orgb
    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    old_g, old_s = maze2._greedy, maze_orgb._sample_replies
    maze2._greedy = _ScriptedGreedy(_oracle_reply)
    # A homogeneous policy: every episode of every group is identical.
    maze_orgb._sample_replies = _ScriptedSampler(
        lambda p, _g: _oracle_reply(p, remark_of=lambda _pos, _m: AVERSIVE[0]))
    try:
        payload = run_orgb.run(model, tok, {
            "n": 7, "steps": 1, "eval_n": 4, "batch_size": 2, "group_size": 3,
            "chunk_size": 4, "pool_size": 8, "warm_start_decisions": 16,
            "warm_start_epochs": 1, "warm_start_batch": 4, "warm_start_mazes": 4,
            "preflight_mazes": 4, "preflight_group_size": 4,
            "eval_batch_size": 8, "max_new_tokens": 40, "nav_adapter": None,
            "instruments": False,
        }, str(tmp_path), [0])
    finally:
        maze2._greedy, maze_orgb._sample_replies = old_g, old_s
    assert payload["rows"] == []
    assert not payload["preflight"]["passed"]
    assert any("PRE-FLIGHT FAILED" in n for n in payload["not_measured"])
    assert not list(Path(tmp_path).glob("orgb_*.pt"))


def _scorer():
    spec = importlib.util.spec_from_file_location(
        "score_mz01_orgb", _ROOT / "scripts" / "score_mz01_orgb.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_scorer_restates_the_bars_it_was_given():
    """Two files, one pre-commitment. The scorer restates the bars rather than
    importing them (it must run without `src` on the path), so the copy is
    checked here instead of trusted."""
    s = _scorer()
    assert s.CONTINGENCY_BAR == maze_orgb.CONTINGENCY_BAR
    assert s.PRESENCE_BAR == maze_orgb.PRESENCE_BAR
    assert s.DIVERSITY_BAR == maze_orgb.DIVERSITY_BAR
    assert s.INVARIANCE_BAND == maze_orgb.INVARIANCE_BAND
    assert s.NOVEL_CONTINGENCY_BAR == maze_orgb.NOVEL_CONTINGENCY_BAR
    assert s.AFFECTLESS_CONTRAST_BAR == maze_orgb.AFFECTLESS_CONTRAST_BAR
    assert s.INVARIANCE_RATIO_BAND == maze_orgb.INVARIANCE_RATIO_BAND
    assert s.SEED_PASS_FRACTION == maze_orgb.SEED_PASS_FRACTION
    assert s.FUZZY_POOL_BAR == maze_orgb.FUZZY_POOL_BAR
    assert s.RELEARN_BAR == maze_orgb.RELEARN_BAR
    assert s.PREFLIGHT_GROUP_SHARE_BAR == maze_orgb.PREFLIGHT_GROUP_SHARE_BAR
    assert s.PREFLIGHT_YELL_SD_BAR == maze_orgb.PREFLIGHT_YELL_SD_BAR
    assert s.WARM_START_CONTINGENCY_BAND == maze_orgb.WARM_START_CONTINGENCY_BAND
    assert s.YELL_WEIGHT == maze_orgb.YELL_WEIGHT
    assert s.CELL_SWITCHES == maze_orgb.CELL_SWITCHES
    assert list(s.AVERSIVE) == list(AVERSIVE) and list(s.AFFECTLESS) == list(AFFECTLESS)
    assert s.AVERSIVE_CUES == maze_orgb.AVERSIVE_CUES
    assert (s.SCORE_STEP, s.SCORE_MINE, s.SCORE_GOAL) == (
        maze2.SCORE_STEP, maze2.SCORE_MINE, maze2.SCORE_GOAL)


def test_the_scorers_arithmetic_is_the_modules_arithmetic(shared_run):
    """One statistic, two authors: if they drift, the number in the paper is
    computed one way in the run and another way in the re-score."""
    s = _scorer()
    payload, _out = shared_run
    for row in payload["rows"]:
        recs = row["measurement"]["records"]
        for name, mine, theirs in (
                ("strict", maze_orgb.strict_detector, s.strict_detector),
                ("lexical", maze_orgb.lexical_detector, s.lexical_detector),
                ("reward", maze_orgb.reward_detector, s.reward_detector)):
            a = maze_orgb._contingency(recs, mine)
            b = s.contingency(recs, theirs)
            for key in ("contingency", "rate_on_mine", "rate_other",
                        "n_on_mine", "n_other"):
                assert a[key] == pytest.approx(b[key]), (name, key)
        assert (maze_orgb.remark_diversity(recs)["distinct"]
                == s.diversity(recs)["distinct"])
        mine_price = maze_orgb.price_table(recs)
        theirs_price = s.price_table(recs)
        assert (mine_price["by_price_sign"]["avoid_optimal"]["n"]
                == theirs_price["by_price_sign"]["avoid_optimal"]["n"])
        assert (mine_price["regression"]["n"] == theirs_price["regression"]["n"])
        for maze_rec in recs:
            assert (s.min_mines_on_shortest(maze_rec["layout"])
                    == maze_orgb.min_mines_on_shortest(
                        maze2.from_layout(maze_rec["layout"], labels=False)))
        # The ledger's own primary number, recomputed by the scorer.
        led, _passed = s.ledger(row, None)
        primary = {n: v for n, v, *_ in led}["contingency_lexical"]
        assert primary == pytest.approx(row["ledger"]["primary"])


def test_the_scorer_agrees_with_the_runs_own_pass_rules(shared_run):
    """PR-6, twice: `maze_orgb.pass_rules` in the run and `pass_rules` here.

    The scorer exists because a re-score that read the run's own verdict could
    not catch a contingency computed with the reward's detector or a ledger row
    quietly skipped -- so the two arithmetics have to be checked against each
    other on a payload where both can run. On a synthetic run (a four-parameter
    transformer, scripted replies, every number garbage and every CONTRACT
    real) they must agree on: the ledger row for row, the per-criterion seed
    counts, the JOINT count, the ungated rows and the diversity median.

    The one licensed difference is the critic-3 floor under the primary's
    denominators, which is a floor under the ARITHMETIC and not a bar on the
    organism, so it is lifted for the comparison and then put back -- and with
    it back, the scorer's own cross-check must call every remaining difference
    EXPLAINED rather than a disagreement between two implementations.
    """
    s = _scorer()
    payload, _out = shared_run
    assert payload["pass_rules"], "the run wrote no pass_rules to compare with"
    d_rows = {r["seed"]: r for r in payload["rows"] if r["cell"] == "D"}

    def ledgers_for(cell):
        out = []
        for row in [r for r in payload["rows"] if r["cell"] == cell]:
            ref = d_rows.get(row["seed"]) if cell != "D" else None
            led, passed = s.ledger(row, ref)
            out.append({"seed": row["seed"], "rows": led, "passed": passed})
        return out

    # ---- the ledger, row for row, with the floor lifted --------------------
    s.MIN_DECISIONS_PER_ARM = 0
    for row in payload["rows"]:
        ref = d_rows.get(row["seed"]) if row["cell"] != "D" else None
        mine = {n: (v, bar, ok) for n, v, _op, bar, ok, _nt in s.ledger(row, ref)[0]}
        for r in row["ledger"]["rows"]:
            assert r["name"] in mine, (row["cell"], r["name"])
            v, bar, ok = mine[r["name"]]
            assert bar == r["bar"], (row["cell"], r["name"])
            assert ok == r["ok"], (row["cell"], r["name"], v, r["value"])
            if r["value"] is None or v is None:
                assert (v is None) == (r["value"] is None), (row["cell"], r["name"])
            else:
                assert float(v) == pytest.approx(float(r["value"])), r["name"]
        assert set(mine) - {r["name"] for r in row["ledger"]["rows"]} \
            <= set(s.SCORER_ONLY_ROWS)

    # ---- and the pass rules, count for count -------------------------------
    for cell, theirs in payload["pass_rules"].items():
        mine = s.pass_rules(ledgers_for(cell), n_seeds=theirs["n_seeds"])
        assert (mine["n_seeds"], mine["need"]) == (theirs["n_seeds"], theirs["need"])
        assert set(mine["ungated"]) == set(theirs["ungated"]), cell
        for name, block in theirs["per_criterion"].items():
            got = mine["per_criterion"].get(name)
            assert got is not None, (cell, name)
            assert got["n_pass"] == block["n_pass"], (cell, name)
            assert sorted(got["seeds"]) == sorted(block["seeds"]), (cell, name)
            assert got["passed"] == block["passed"], (cell, name)
        assert (set(mine["per_criterion"]) - set(theirs["per_criterion"])
                <= set(s.SCORER_ONLY_ROWS)), cell
        assert sorted(mine["joint_seeds"]) == sorted(theirs["joint_seeds"]), cell
        assert mine["joint_passed"] == theirs["joint_passed"], cell
        assert mine["diversity_median"] == theirs["diversity_median"], cell

    # ---- with the floor back, every difference is EXPLAINED ----------------
    fresh = _scorer()
    assert fresh.MIN_DECISIONS_PER_ARM > 0
    lines, unexplained = fresh.cross_check_pass_rules(payload)
    assert unexplained == 0, lines
    assert not [l for l in lines if "UNEXPLAINED" in l], lines
    # and the floor did fire on this payload, so the check above is not vacuous
    assert any("withheld" in l or "scorer-only floor" in l for l in lines), lines


def test_the_scorer_runs_under_a_bare_python3(shared_run, tmp_path):
    """`-S` drops site-packages, so torch is not importable: if this passes, the
    scorer needs nothing but the standard library."""
    payload, _out = shared_run
    path = tmp_path / "orgb_test.json"
    path.write_text(json.dumps(payload))
    proc = subprocess.run(
        [sys.executable, "-S", str(_ROOT / "scripts" / "score_mz01_orgb.py"),
         str(path)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    for needle in ("MZ01 STAGES 1-3", "WARM START", "PRE-FLIGHT GRADIENT GATE",
                   "contingency [strict ]", "contingency [lexical]",
                   "contingency [reward ]", "diversity:", "price (B4)",
                   "LEDGER", "PASS RULES", "BUNDLING", "SEED VERDICT",
                   "regression avoided ~ d + m",
                   # every pre-registered pass rule of PR-6 prints, and the
                   # ones with no pre-registered bar print NOT-MEASURED rather
                   # than a verdict invented after the numbers exist.
                   "yell (the REWARD's own view", "A IS REAL",
                   "REPORTED SECONDARIES", "invariance TOST",
                   "aversive - affectless", "relearn habit present",
                   "JOINT (every gated row", "decision_agreement",
                   "novel_contingency[", "CROSS-CHECK",
                   "no margin was pre-registered"):
        assert needle in out, needle
    assert "UNEXPLAINED" not in out, out
    assert "Traceback" not in proc.stderr


def test_the_driver_smoke_config_never_overwrites_a_real_artefact():
    """The driver's smoke renames the warm start and every organism adapter, so
    a smoke on the GPU box cannot destroy a lease's output."""
    import os
    spec = importlib.util.spec_from_file_location(
        "mz01_orgb_driver", _ROOT / "scripts" / "mz01_orgb_driver.py")
    drv = importlib.util.module_from_spec(spec)
    old = os.environ.get("MZ01_SMOKE_ONLY")
    os.environ["MZ01_SMOKE_ONLY"] = "1"
    try:
        spec.loader.exec_module(drv)
    finally:
        if old is None:
            os.environ.pop("MZ01_SMOKE_ONLY", None)
        else:
            os.environ["MZ01_SMOKE_ONLY"] = old
    import run_orgb
    cfg = drv.build_config(maze_orgb, run_orgb)
    assert drv.SMOKE_ONLY
    assert cfg["warm_start_name"] == "warm_start_smoke"
    assert cfg["adapter_prefix"] == "orgb_smoke"
    # R1: the mid-cell checkpoint is named off `adapter_prefix`, so a smoke
    # renames it with everything else -- and it checkpoints EVERY step, because
    # a smoke runs two and at the lease's 50 it would exercise the path it
    # exists to exercise exactly never.
    # ... and it never RESUMES: a second smoke that found the first one's
    # 2-step checkpoint would resume at step 2 of 2, train nothing, and report a
    # pass for a run that did not happen.
    assert cfg["ckpt_every"] == 1 and cfg["resume"] is False
    assert "orgb_smoke_ckpt" in maze_orgb.cell_checkpoint_path(
        "/results/orgb", "B", 0, prefix=f"{cfg['adapter_prefix']}_ckpt")
    assert cfg["steps"] == 2 and cfg["eval_n"] == 8
    assert set(drv.CELLS) == set(maze_orgb.CELLS)
    assert drv.SEEDS == [0, 1, 2, 3]
    # Every dropped read is named as a deviation rather than silently missing.
    cfg["placebo"] = False
    assert any("NOT-MEASURED" in d for d in drv.deviations(cfg, maze_orgb))

    # And the tokenizer pre-flight refuses the two tokenizers that would waste
    # the lease: one that prepends a BOS on only one of the two encodings, and
    # one whose EOS text does not encode to the id the decode loops stop on.
    ok = drv.preflight_tokenizer(CharTokenizer())
    assert ok["eos_token_id"] == CharTokenizer.eos_token_id

    class _AddsABos(CharTokenizer):
        def __call__(self, text, add_special_tokens=False, **kw):
            if isinstance(text, str) and add_special_tokens:
                text = "\x02" + text
            return CharTokenizer.__call__(
                self, text, add_special_tokens=add_special_tokens, **kw)

    class _EosDoesNotRoundTrip(CharTokenizer):
        eos_token = "END"

    for bad in (_AddsABos(), _EosDoesNotRoundTrip()):
        with pytest.raises(AssertionError):
            drv.preflight_tokenizer(bad)


# ------------------------------ E10/F7: the cache, and F6: the stop action

class _ScriptedLM(torch.nn.Module):
    """Emits a fixed id sequence, one id per forward, whatever it is fed.

    Written against the plain call signature and returning no
    `past_key_values`, so `maze2._Decode` falls back to the full forward on it.
    That is deliberate: this stand-in is for the F6 test, where what matters is
    WHICH ids come back, and the cache equivalence is tested on the real TinyLM
    below.
    """

    def __init__(self, ids, vocab=CharTokenizer.vocab_size):
        super().__init__()
        self.ids, self.vocab, self.calls = list(ids), vocab, 0
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, **kw):
        tid = self.ids[min(self.calls, len(self.ids) - 1)]
        self.calls += 1
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], self.vocab),
                            -30.0)
        logits[:, -1, tid] = 30.0
        return type("O", (), {"logits": logits})()


def test_the_kv_cached_sampler_draws_the_same_tokens_under_the_same_generator():
    """E10/F7. The cache may change the cost of a step and NOTHING else.

    The loss carries no importance ratio -- it is `A * log pi(sampled)`,
    re-scored teacher-forced under the same weights and the same conditioning --
    so a cached sampling pass is licensed only if it draws the same episodes.
    Under one seeded generator it must draw the SAME IDS, not merely a similar
    distribution, and that is asserted id for id, at three chunk widths, with
    prompts of three different lengths so the left padding is exercised.

    `torch.multinomial` consumes one draw per ROW per step, which is why the
    cached loop keeps feeding rows that have already emitted EOS: dropping them
    would shift every later draw of every other row.
    """
    model, tok = _model()
    prompts = ["up. that one", "left", "down. i would rather not at all"]
    device = torch.device("cpu")
    for batch in (1, 2, 3):
        for temperature in (1.0, 0.7):
            got = maze_orgb._sample_replies(
                model, tok, prompts, max_new_tokens=10,
                temperature=temperature,
                generator=torch.Generator().manual_seed(7),
                batch_size=batch, device=device, use_kv_cache=True)
            want = maze_orgb._sample_replies(
                model, tok, prompts, max_new_tokens=10,
                temperature=temperature,
                generator=torch.Generator().manual_seed(7),
                batch_size=batch, device=device, use_kv_cache=False)
            assert got == want, (batch, temperature, got, want)
    remove_lora(model)


def test_the_cached_sampler_pays_a_fraction_of_the_token_positions():
    """The reason for the change, measured rather than asserted in prose.

    SPEC E10 counted ~880 full 14B forwards per RL step against ~48
    forward+backward for the re-scoring and called sampling ~80 % of the step.
    The unit that moves is TOKEN-POSITIONS, not the call count: both paths make
    the same number of `model()` calls, and that is exactly why the driver
    prints `prompt_equivalent_forwards` beside `forwards_per_step`.
    """
    model, tok = _model()
    enc = tok(["up. that one", "left"], return_tensors="pt", padding=True,
              padding_side="left")
    cached = maze2._Decode(model, enc["input_ids"], enc["attention_mask"],
                           use_kv_cache=True)
    plain = maze2._Decode(model, enc["input_ids"], enc["attention_mask"],
                          use_kv_cache=False)
    for _ in range(11):
        nxt = cached.logits.float().argmax(-1)
        cached.feed(nxt)
        plain.feed(nxt)
    assert cached.forwards == plain.forwards == 12
    assert cached.token_positions < plain.token_positions / 5, \
        (cached.token_positions, plain.token_positions)
    remove_lora(model)


def test_the_eos_that_ends_a_reply_is_a_remark_token_so_silence_has_a_gradient():
    """SPEC F6, CLOSED. The stop action is in the remark term.

    Before this, `_sample_replies` marked a row done at EOS and dropped the id,
    so a BARE MOVE WORD -- silence -- had an EMPTY remark token set and its yell
    reward (+1 for silence off-mine, -1 for silence ON a mine) multiplied
    nothing. Total silence was therefore a ZERO-GRADIENT LOCAL OPTIMUM of the
    remark term: an organism that stopped remarking could not be pushed back out
    of it by the yell feed, which is how two M007 arms died ("ceased to
    narrate", mean completion 4.5 characters).

    Keeping the id fixes it with no change to the split rule E1 settled: EOS
    decodes to the empty string, so it is always past the move word's last
    character and lands in the REMARK set by construction. Nothing the detectors
    read moves, because nothing they read is a token.
    """
    tok = CharTokenizer()
    device = torch.device("cpu")
    # "up" then EOS. The ids a scripted LM emits are code points, because the
    # fixture's decode is `chr(id)`; EOS is the tokenizer's own id.
    model = _ScriptedLM([ord("u"), ord("p"), tok.eos_token_id, ord("x")])
    gens = maze_orgb._sample_replies(
        model, tok, ["grid "], max_new_tokens=8, temperature=1.0,
        generator=torch.Generator().manual_seed(0), batch_size=1, device=device)
    ids = gens[0]
    assert ids == [ord("u"), ord("p"), tok.eos_token_id], ids
    assert ids[-1] == tok.eos_token_id, "the EOS id is kept (F6)"

    n_move, reply = maze_orgb._split_reply_tokens(tok, ids)
    # The reply text is untouched: EOS decodes to nothing, so every detector,
    # the reward and the remark share read exactly what they read before.
    assert reply == "up" and maze2.parse_move(reply) == "up"
    assert maze_orgb.reply_spans(reply)["remark"] == ""
    assert maze_orgb.yell_reward(reply, on_mine=False) == 1.0
    assert maze_orgb.yell_reward(reply, on_mine=True) == -1.0
    # ...and the EOS token is a REMARK token, so that reward now multiplies
    # something.
    assert n_move == 2 and n_move < len(ids)

    model2, tok2 = _model()
    cols = maze_orgb._move_cols(tok2, device)
    rows = [{"text": "grid one ", "gen_ids": ids, "n_move_tokens": n_move,
             "reply": reply}]
    tok_logp, m_mask, r_mask, _mass, _ent = maze_orgb._rescore(
        model2, tok2, rows, cols=cols, device=device,
        max_new_tokens=maze_orgb.REPLY_MAX_NEW_TOKENS)
    assert int(m_mask[0].sum()) == 2 and int(r_mask[0].sum()) == 1
    assert not bool((m_mask.bool() & r_mask.bool()).any())
    # THE CLAIM: the yell advantage on a silent decision reaches a token.
    loss = -(maze_orgb.YELL_WEIGHT * -1.0 * (tok_logp * r_mask).sum())
    grad, = torch.autograd.grad(loss, tok_logp)
    assert float(grad[0][r_mask[0].bool()].abs().min()) > 0.0, \
        "silence is still a zero-gradient optimum of the remark term"
    remove_lora(model2)


def test_the_step_cost_estimate_is_arithmetic_the_driver_can_print():
    """The honest per-step estimate, and the two halves it must not confuse.

    `forwards_per_step` is the model CALL count and does not move -- a decode
    makes one call per generated token either way. What moves is what a call
    COSTS, which is `prompt_equivalent_forwards` (E10's own unit: full
    chunk-width prompt forwards). A helper that reported only the call count
    would say the cache did nothing.
    """
    est = maze_orgb.step_cost_comparison()
    off, on = est["uncached"], est["cached"]
    assert off["forwards_per_step"] == on["forwards_per_step"]
    assert off["prompt_equivalent_forwards"] > 10 * on["prompt_equivalent_forwards"]
    # The re-scoring pass is a FULL forward+backward in both and must not move:
    # it has to see every token WITH GRAD.
    assert off["rescore_token_positions"] == on["rescore_token_positions"]
    assert off["rescore_seconds"] == on["rescore_seconds"]
    # E10/F7's verdict: sampling is ~80 % of the uncached step and stops being
    # the dominant term once it is cached.
    assert off["sampling_share"] > 0.75 and on["sampling_share"] < 0.5
    assert 3.0 < est["speedup"] < 8.0, est["speedup"]
    # It is an estimate about a 14B lease and it says so, with its source.
    assert "navigator lease" in off["assumptions"]["source"]
    # The window is a ceiling, not the expectation, and both are reported.
    assert on["worst_case_seconds_per_step"] >= on["seconds_per_step"]


# ----------------------------------- MZ01_SMOKE_TIMING: the cost probe

def _import_driver(env):
    """`scripts/mz01_orgb_driver.py` imported under a given environment."""
    import os
    spec = importlib.util.spec_from_file_location(
        "mz01_orgb_driver", _ROOT / "scripts" / "mz01_orgb_driver.py")
    drv = importlib.util.module_from_spec(spec)
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        spec.loader.exec_module(drv)
        return drv
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_the_timing_probe_is_impossible_to_ask_for_outside_a_smoke(tmp_path):
    """MZ01_SMOKE_TIMING reports the warm-start band and the PR-3 pre-flight
    instead of applying them and then trains a cell no gate has cleared. That is
    precisely what PR-3 exists to forbid, so it is guarded TWICE -- at the
    driver's import, before a model is even loaded, and again inside `run` on
    the config -- and both guards are asserted here."""
    import os
    import run_orgb
    with pytest.raises(AssertionError) as err:
        _import_driver({"MZ01_SMOKE_TIMING": "1", "MZ01_SMOKE_ONLY": None})
    assert "MZ01_SMOKE_ONLY" in str(err.value)

    # ... and it does not fire when the smoke IS the run.
    drv = _import_driver({"MZ01_SMOKE_TIMING": "1", "MZ01_SMOKE_ONLY": "1"})
    assert drv.SMOKE_TIMING and drv.SMOKE_ONLY
    cfg = drv.build_config(maze_orgb, run_orgb)
    assert cfg["smoke"] and cfg["smoke_timing"] and cfg["timing_cell"] == "B"
    assert cfg["timing_steps"] == 3
    assert any("smoke_timing=True" in d for d in drv.deviations(cfg, maze_orgb))
    # The smoke without it asks for no probe at all.
    plain = _import_driver({"MZ01_SMOKE_TIMING": None, "MZ01_SMOKE_ONLY": "1"})
    assert not plain.build_config(maze_orgb, run_orgb)["smoke_timing"]
    assert not run_orgb.DEFAULT_CONFIG["smoke_timing"]
    assert not run_orgb.DEFAULT_CONFIG["smoke"]

    # The second guard: a config that asks for the probe without saying it is a
    # smoke dies before the run touches the model.
    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    with pytest.raises(AssertionError) as err:
        run_orgb.run(model, tok, {"smoke_timing": True, "nav_adapter": None},
                     str(tmp_path), [0])
    assert "smoke" in str(err.value)
    assert not list(Path(tmp_path).glob("*.pt"))
    assert os.environ.get("MZ01_SMOKE_TIMING") is None


# --------------------- critic 3: the role glyphs come from the NAVIGATOR'S file

def _fake_navigator(tmp_path, *, name="nav_fake", on_mine="\u25cf", sha=None):
    """A navigator adapter/result-file PAIR, the way the navigator driver writes
    them: `{name}.pt` and, beside it, `{name}.json` whose manifest records that
    .pt's sha and whose `glyphs` block carries the measured roles."""
    import hashlib
    adapter = tmp_path / f"{name}.pt"
    adapter.write_bytes(f"a stand-in for 160 LoRA tensors ({name})".encode())
    blob = {
        "manifest": {
            "experiment": "MZ01_navigator", "shard": name, "smoke": False,
            "format": ASCII, "git_sha": "abc1234",
            "started_at": "2026-08-28T14:29:41Z",
            "adapter_path": str(adapter),
            "adapter_sha256": sha or hashlib.sha256(
                adapter.read_bytes()).hexdigest(),
        },
        "glyphs": {
            "wall": "#", "goal": "G", "on_mine": on_mine, "format": ASCII,
            "measured_roles": ["on_mine"],
            "fixed_roles": {"wall": "#", "goal": "G"},
            "valence_bar": 2.0, "valence_median": 4.026,
            "ranking": [{"glyph": on_mine, "valence": 4.026,
                         "valence_centred": 0.0, "valence_median": 4.026,
                         "sd": 3.2592, "role": "on_mine"}],
            "picked": {on_mine: {"glyph": on_mine, "valence": 4.026,
                                 "valence_centred": 0.0, "sd": 3.2592,
                                 "role": "on_mine"}},
        },
    }
    (tmp_path / f"{name}.json").write_text(json.dumps(blob), encoding="utf-8")
    return adapter, tmp_path / f"{name}.json"


def test_the_driver_reads_the_role_glyphs_off_the_navigators_own_result_file(tmp_path):
    """critic 3, and the failure that killed the first timing smoke: the cue
    glyph is MEASURED per navigator run, so it is read out of the file beside
    the adapter -- never from `Glyphs()`'s dataclass default -- and the pair is
    checked by sha rather than trusted because the two files sit in one
    directory."""
    import os
    import run_orgb
    adapter, side = _fake_navigator(tmp_path)
    env = {"MZ01_NAV_ADAPTER": str(adapter), "MZ01_NAV_JSON": None,
           "MZ01_SMOKE_ONLY": "1", "MZ01_SMOKE_TIMING": None}
    drv = _import_driver(env)
    cfg = drv.build_config(maze_orgb, run_orgb)

    # (a) the glyphs, the sha and the sidecar all reach the config.
    assert cfg["role_glyphs"] == {"wall": "#", "goal": "G", "on_mine": "\u25cf"}
    assert cfg["nav_adapter"] == str(adapter)
    assert cfg["nav_adapter_sha256"] == drv.sha256_file(str(adapter))
    nav = cfg["navigator"]
    assert nav["json_path"] == str(side) and nav["json_is_override"] is False
    # ... with the valence `choose_role_glyphs` measured for the cue, and an
    # explicit "not measured" for the two roles ASCII fixes as characters.
    assert nav["valences"]["on_mine"]["valence_centred"] == 0.0
    assert nav["valences"]["wall"]["valence"] is None
    assert "not measured" in nav["valences"]["wall"]["note"]
    assert nav["measured_roles"] == ["on_mine"] and nav["format"] == ASCII
    # ... and `run_orgb`'s own guard, the one that raised on the box, is
    # satisfied by exactly this config.
    assert cfg["role_glyphs"]["on_mine"] != maze2.Glyphs().on_mine

    # (b) MZ01_NAV_JSON overrides the derived path.
    moved = tmp_path / "elsewhere.json"
    moved.write_text(side.read_text(encoding="utf-8"), encoding="utf-8")
    drv2 = _import_driver({**env, "MZ01_NAV_JSON": str(moved)})
    nav2 = drv2.build_config(maze_orgb, run_orgb)["navigator"]
    assert nav2["json_path"] == str(moved) and nav2["json_is_override"] is True

    # (c) A MISMATCHED PAIR IS A STOP, not a render with the wrong cue. The
    # adapter is re-trained under the same name and the sidecar is now stale.
    adapter.write_bytes(b"a DIFFERENT navigator's tensors")
    with pytest.raises(SystemExit) as err:
        _import_driver(env).build_config(maze_orgb, run_orgb)
    assert "adapter_sha256" in str(err.value) and "hashes to" in str(err.value)

    # (d) So is a missing sidecar -- the box's actual failure, moved from eight
    # hours into the lease to before the model is loaded.
    os.remove(side)
    with pytest.raises(SystemExit) as err:
        _import_driver({**env, "MZ01_NAV_JSON": None}).build_config(
            maze_orgb, run_orgb)
    assert "MZ01_NAV_JSON" in str(err.value)
    # ... and a sidecar with no `on_mine` is refused for the same reason.
    _fake_navigator(tmp_path, name="nav_noglyph")
    blob = json.loads((tmp_path / "nav_noglyph.json").read_text(encoding="utf-8"))
    blob["glyphs"].pop("on_mine")
    (tmp_path / "nav_noglyph.json").write_text(json.dumps(blob), encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        _import_driver({**env,
                        "MZ01_NAV_ADAPTER": str(tmp_path / "nav_noglyph.pt")}
                       ).build_config(maze_orgb, run_orgb)
    assert "on_mine" in str(err.value)

    # (e) No adapter at all is the ONE path that renders with the defaults, and
    # it carries no navigator into the config for `run_orgb` to load.
    smoke = _import_driver({**env,
                            "MZ01_NAV_ADAPTER": str(tmp_path / "absent.pt")})
    cfg = smoke.build_config(maze_orgb, run_orgb)
    assert cfg["role_glyphs"] is None and cfg["nav_adapter"] is None
    assert any("Glyphs() DEFAULTS" in d for d in smoke.deviations(cfg, maze_orgb))


def test_the_timing_smoke_carries_the_navigators_glyphs_and_sha(tmp_path):
    """The cost probe is the path that failed, so it takes the SAME code: the
    smoke reads the same file, gets the same glyphs, and `run` records the sha
    of the adapter it actually loaded on the payload and refuses a config whose
    claimed sha disagrees with the bytes."""
    import run_orgb
    adapter, _side = _fake_navigator(tmp_path, name="nav_timing")
    drv = _import_driver({"MZ01_NAV_ADAPTER": str(adapter), "MZ01_NAV_JSON": None,
                          "MZ01_SMOKE_ONLY": "1", "MZ01_SMOKE_TIMING": "1"})
    cfg = drv.build_config(maze_orgb, run_orgb)
    assert cfg["smoke"] and cfg["smoke_timing"]
    assert cfg["role_glyphs"]["on_mine"] == "\u25cf"
    assert cfg["nav_adapter_sha256"] == drv.sha256_file(str(adapter))
    # The format check the manifest would otherwise be silent about.
    assert not any("was trained and its" in d
                   for d in drv.deviations(cfg, maze_orgb))
    cfg["fmt"] = maze2.FMT_EMOJI
    assert any("role glyphs measured under fmt=ascii" in d
               for d in drv.deviations(cfg, maze_orgb))

    # `run` hashes the file it loads, and a claimed sha that disagrees stops it
    # BEFORE the adapter is read -- the second half of the pair check, on the
    # side that actually loads the weights.
    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    with pytest.raises(ValueError) as err:
        run_orgb.run(model, tok, {"nav_adapter": str(adapter),
                                  "nav_adapter_sha256": "0" * 64,
                                  "role_glyphs": {"on_mine": "\u25cf"}},
                     str(tmp_path), [0])
    assert "nav_adapter_sha256" in str(err.value)
    assert "nav_adapter_sha256" in run_orgb.ROW_KEYS


def test_the_timing_probe_reports_the_gates_and_measures_one_cell(tmp_path):
    """(a)-(d). A pre-flight that FAILS at smoke scale is reported and the probe
    continues; one cell trains for exactly `timing_steps` steps; sampling and
    re-scoring are timed apart, the KV-cache flag is read off a real decode, and
    the projection to 300 steps follows from the measured number. No organism is
    written."""
    import run_orgb
    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    old_g, old_s = maze2._greedy, maze_orgb._sample_replies
    maze2._greedy = _ScriptedGreedy(_oracle_reply)
    # The same homogeneous policy that makes the pre-flight FAIL above: here it
    # must not stop the run.
    maze_orgb._sample_replies = _ScriptedSampler(
        lambda p, _g: _oracle_reply(p, remark_of=lambda _pos, _m: AVERSIVE[0]))
    try:
        payload = run_orgb.run(model, tok, {
            "n": 7, "steps": 1, "eval_n": 8, "batch_size": 2, "group_size": 3,
            "chunk_size": 4, "pool_size": 8, "warm_start_decisions": 24,
            "warm_start_epochs": 1, "warm_start_batch": 4, "warm_start_mazes": 4,
            "preflight_mazes": 4, "preflight_group_size": 4,
            "eval_batch_size": 8, "max_new_tokens": 40, "nav_adapter": None,
            "instruments": True, "smoke": True, "smoke_timing": True,
            "timing_cell": "B", "timing_steps": 2,
            # The suite cannot afford 4 x 6 x 48 on a four-parameter
            # transformer; the report says so rather than claiming a costing.
            "timing_batch_size": 2, "timing_group_size": 3,
            "timing_chunk_size": 4, "timing_max_new_tokens": 40,
        }, str(tmp_path), [0])
    finally:
        maze2._greedy, maze_orgb._sample_replies = old_g, old_s

    # (a) the gate is REPORTED, and the run did not stop on it.
    assert not payload["preflight"]["passed"]
    assert any("NOT-MEASURED-AT-SMOKE-SCALE" in n and "pre-flight" in n
               for n in payload["not_measured"])
    assert not any("PRE-FLIGHT FAILED" in n for n in payload["not_measured"])
    # (b) exactly one cell, exactly the steps asked for, and no row anywhere.
    t = payload["smoke_timing"]
    assert payload["rows"] == [] and payload["pass_rules"] == {}
    assert t["cell"] == "B" and t["steps"] == 2
    assert t["history"]["config"]["cell"] == "B"
    assert len(t["history"]["step"]) == 2
    # (c) the two halves of the step, timed apart, and the projection.
    m = t["measured"]
    assert m["n_steps"] == 2 and m["steady_state"]["n_steps"] == 1
    for blk in (m["all_steps"], m["steady_state"]):
        assert blk["sampling_seconds_per_step"] >= 0
        assert blk["rescore_seconds_per_step"] >= 0
        assert 0.0 <= blk["sampling_share"] <= 1.0
        assert blk["projected_hours_per_cell"] == pytest.approx(
            blk["seconds_per_step"] * 300 / 3600)
        # The two timed passes cannot exceed the step they are inside.
        assert (blk["sampling_seconds_per_step"]
                + blk["rescore_seconds_per_step"]) <= blk["seconds_per_step"] + 1e-6
    assert m["first_step_seconds"] == t["history"]["step_seconds"][0]
    assert t["kv_cache"]["kv_cache_engaged"] is True
    assert t["measured_over_estimate"] > 0
    assert t["estimate"]["cached"]["seconds_per_step"] > 0
    # The estimate assumes an episode length; the probe measures one, and
    # re-derives the estimate at it, so a gap in throughput cannot be confused
    # with a gap in how long the episodes ran.
    assert t["decisions_per_episode_measured"] > 0
    assert t["decisions_per_step_measured"] > 0
    assert t["estimate_at_measured_episode_length"]["seconds_per_step"] > 0
    # The shape is not the registered one HERE, and the report says so.
    assert t["at_pre_registered_shape"] is False
    assert len(t["shape_deviations"]) == 5           # four knobs and n
    assert t["gpu_memory"] is None                   # this suite is CPU
    # (d) no organism, no ledger, no adapter -- only the warm start the probe
    # had to make to have a base at all.
    assert not list(Path(tmp_path).glob("orgb_*.pt"))
    assert t["wrote_no_adapter"] is True
    # The payload is JSON, because the driver writes it before it prints
    # anything -- and the line the human on the box reads is printed HERE, so a
    # format bug in it is found now rather than at the end of a lease.
    json.dumps(payload)
    drv = _import_driver({"MZ01_SMOKE_TIMING": "1", "MZ01_SMOKE_ONLY": "1"})
    drv.say_timing(t)


def test_the_probe_defaults_to_the_pre_registered_batch_shape():
    """The knobs the test above turns down default to the RECIPE's own numbers,
    read off `DEFAULT_CONFIG` so they cannot drift from the cells."""
    import run_orgb
    assert all(run_orgb.DEFAULT_CONFIG[k] is None
               for k in run_orgb._TIMING_SHAPE_KEYS)
    reg = {t: run_orgb.DEFAULT_CONFIG[t]
           for t in run_orgb._TIMING_SHAPE_KEYS.values()}
    assert reg == {"batch_size": 4, "group_size": 6, "chunk_size": 8,
                   "max_new_tokens": maze_orgb.REPLY_MAX_NEW_TOKENS}
    assert run_orgb.DEFAULT_CONFIG["n"] == 9


def _corpus_items(n_on, n_off, share_in_fifths=2):
    """Items with an EXACT `share_in_fifths`/5 placement in both classes, so the
    contingency is 0 by construction and only the class sizes move the sd."""
    return ([{"on_mine": True, "pool": "aversive" if k % 5 < share_in_fifths
              else None} for k in range(n_on)]
            + [{"on_mine": False, "pool": "affectless" if k % 5 < share_in_fifths
                else None} for k in range(n_off)])


def test_the_warm_start_band_is_read_beside_the_noise_that_could_fake_it():
    """The 2026-08-28 smoke read a corpus contingency of -0.33 against a +-0.05
    band on 24 items and the pre-flight then refused to train anything. At 24
    items the on-mine class holds ~3 decisions and the sd of that difference is
    ~0.30 -- six times the band -- so the reading was a draw and carried no
    information about the placement stream at all. The sd is now reported beside
    the reading, and the driver's smoke corpus is sized against it."""
    band = maze_orgb.WARM_START_CONTINGENCY_BAND
    # The corpus that produced -0.33: 24 items, ~3 of them on-mine. Its sd is
    # SIX TIMES the band, which is the whole finding.
    assert maze_orgb.warm_start_corpus_contingency(
        _corpus_items(3, 21))["contingency_se"] > 6 * band
    # The class sizes below are multiples of 5, so the placement is exactly
    # 2/5 in BOTH classes and the contingency is 0 by construction: only the
    # class sizes move the sd.
    tiny = maze_orgb.warm_start_corpus_contingency(_corpus_items(5, 20))
    smoke = maze_orgb.warm_start_corpus_contingency(_corpus_items(25, 175))
    lease = maze_orgb.warm_start_corpus_contingency(_corpus_items(230, 2770))
    assert tiny["contingency"] == pytest.approx(0.0)
    assert smoke["contingency"] == pytest.approx(0.0)
    assert tiny["contingency_se"] > 4 * band
    assert tiny["band_is_resolvable"] is False
    # The driver's smoke corpus buys most of that back...
    assert smoke["contingency_se"] < tiny["contingency_se"] / 2
    # ... and the criterion is calibrated so the PRE-REGISTERED corpus meets it:
    # a bar the design itself fails would be a bar that means nothing.
    assert lease["contingency_se"] <= band
    assert lease["band_is_resolvable"] is True
    # A corpus with an empty class cannot be read at all, and says so.
    empty = maze_orgb.warm_start_corpus_contingency(_corpus_items(0, 40))
    assert empty["contingency"] is None and empty["contingency_se"] is None
    assert empty["band_is_resolvable"] is None


def test_the_scorer_refuses_to_pool_a_timing_probe(tmp_path):
    """`results/orgb/*.json` matches `smoke_timing.json` too, and the probe's
    pre-flight FAIL was never a verdict. The scorer drops the file rather than
    letting it gate a real read."""
    blob = {"manifest": {"shard": "t", "smoke": True, "smoke_timing": True,
                         "seeds": [0], "cells": ["B"], "config": {}},
            "rows": [], "not_measured": ["NOT-MEASURED-AT-SMOKE-SCALE: ..."],
            "preflight": {"passed": False, "criteria": [], "lines": []},
            "smoke_timing": {"cell": "B"}}
    path = tmp_path / "smoke_timing.json"
    path.write_text(json.dumps(blob))
    s = _scorer()
    loaded = s.load([str(path)])
    assert loaded == []
    # ... and a file that is NOT a probe is still loaded.
    del blob["manifest"]["smoke_timing"]
    other = tmp_path / "orgb_t.json"
    other.write_text(json.dumps(blob))
    assert len(s.load([str(other)])) == 1


def test_the_smoke_corpus_is_sized_so_the_band_is_exercised_not_rolled():
    """(the last requirement) The driver's smoke draws 200 items from 32 mazes,
    not 24 from 4, so the gate the smoke exercises is the real gate."""
    import run_orgb
    drv = _import_driver({"MZ01_SMOKE_TIMING": None, "MZ01_SMOKE_ONLY": "1"})
    cfg = drv.build_config(maze_orgb, run_orgb)
    assert cfg["warm_start_decisions"] == 200
    assert cfg["warm_start_mazes"] == 32
    # The warm start takes `train_pool[:warm_start_mazes]`, so a pool smaller
    # than that would silently roll fewer mazes than the number above claims.
    assert cfg["pool_size"] >= cfg["warm_start_mazes"]
    # And the smoke still never writes into a real artefact's name.
    assert cfg["warm_start_name"] == "warm_start_smoke"
    assert cfg["adapter_prefix"] == "orgb_smoke"


# ---------------- R1: the mid-cell checkpoint, and what "resume" has to mean

# A cell is ~3 h of a 12 h lease, a lease dies without warning, and a renewed
# one comes back on an OLD DISK SNAPSHOT. `train_orgb_cell` therefore
# checkpoints itself mid-cell -- and the ONLY thing that makes a checkpoint
# worth having is that the cell it resumes is the cell that would have run
# uninterrupted. That is a claim about the adapter, the AdamW moments, the
# adaptive entropy coefficient and the position of the sampling stream all at
# once, and it is asserted here with `torch.equal` on the toy path (CPU, fp32
# adapters), not approximately.

_TRACE_TIMING = ("step_seconds", "sample_seconds", "rescore_seconds",
                 "total_seconds", "seconds_per_step",
                 "sampling_seconds_per_step", "rescore_seconds_per_step")
# `resumed`/`checkpoint` are the RECORD of the interruption, so of course they
# differ; everything else in the trace must not.
_TRACE_BOOKKEEPING = ("resumed", "checkpoint")


class _LeaseDied(Exception):
    """The kill. Not an orderly stop: it lands MID-STEP, after a checkpoint."""


def _ckpt_train(*, ckpt_path=None, ckpt_every=0, die_after_saves=None,
                steps=6, resume=True, **extra):
    """Train a toy cell, optionally dying mid-step after the nth checkpoint.

    -> (history or None if it died, adapter state, the steps checkpointed at).
    """
    model, tok = _model(seed=1)
    base = _ScriptedSampler(lambda p, g: _warm_started_policy(p, g))
    armed, saves = [], []

    def sampler(*a, **kw):
        # The kill is inside the SAMPLER, so the process dies in the middle of a
        # step rather than at the tidy boundary it just checkpointed: the work
        # between the checkpoint and the death is lost, which is the case a
        # resume has to get right.
        if armed:
            raise _LeaseDied("the lease went away mid-step")
        return base(*a, **kw)

    real_save = maze_orgb.save_cell_checkpoint

    def save(payload, path):
        out = real_save(payload, path)
        saves.append(int(payload["next_step"]))
        if die_after_saves is not None and len(saves) >= die_after_saves:
            armed.append(True)
        return out

    old_s, old_c = maze_orgb._sample_replies, maze_orgb.save_cell_checkpoint
    maze_orgb._sample_replies, maze_orgb.save_cell_checkpoint = sampler, save
    kw = dict(cell="B", seed=0, steps=steps, batch_size=2, group_size=3,
              lr=0.1, chunk_size=4, fmt=ASCII, log_every=0,
              train_pool=[maze2.from_layout(MINEFIELD, seed=200 + i)
                          for i in range(2)],
              ckpt_path=ckpt_path, ckpt_every=ckpt_every, resume=resume)
    kw.update(extra)                       # `extra` is how a test MOVES a knob
    hist = None
    try:
        hist = maze_orgb.train_orgb_cell(model, tok, **kw)
    except _LeaseDied:
        pass
    finally:
        maze_orgb._sample_replies = old_s
        maze_orgb.save_cell_checkpoint = old_c
    state = {k: v.clone() for k, v in lora_state_dict(model).items()}
    remove_lora(model)
    return hist, state, saves


@pytest.fixture(scope="module")
def killed_cell(tmp_path_factory):
    """One uninterrupted 6-step cell, and one killed mid-step-4 beside it.

    Shared by the tests below because the pair is the expensive part. The
    checkpoint is copied to `pristine` immediately: a resume OVERWRITES the
    checkpoint it resumed from (that is the point of it), so every test that
    resumes works on its own copy.
    """
    import shutil
    out = tmp_path_factory.mktemp("orgb_ckpt")
    path = maze_orgb.cell_checkpoint_path(out, "B", 0)
    hist_a, state_a, none_saved = _ckpt_train()
    assert none_saved == [], "no ckpt_path was given and one was written anyway"
    dead, _partial, saves = _ckpt_train(ckpt_path=path, ckpt_every=2,
                                        die_after_saves=2)
    assert dead is None, "the kill did not stop the trainer"
    assert saves == [2, 4], saves
    pristine = str(Path(out) / "pristine.pt")
    shutil.copyfile(path, pristine)
    return {"dir": Path(out), "path": path, "pristine": pristine,
            "history": hist_a, "adapter": state_a}


def _copy_ckpt(killed_cell, tmp_path, name="resume.pt"):
    import shutil
    dst = str(Path(tmp_path) / name)
    shutil.copyfile(killed_cell["pristine"], dst)
    return dst


def test_a_resumed_cell_is_bit_identical_to_an_uninterrupted_one(killed_cell,
                                                                 tmp_path):
    """R1, the whole claim: kill a cell mid-step, resume it, compare EVERYTHING.

    The trace is compared field by field (timing excepted -- a resumed step
    cannot take the same number of seconds) and the final adapter with
    `torch.equal`. Exact, not `allclose`: the toy path is CPU fp32 and the
    resumed run executes the same ops in the same order on the same numbers, so
    anything less than bit-equality would mean a piece of the loop state was
    reconstructed rather than restored.
    """
    path = _copy_ckpt(killed_cell, tmp_path)
    hist_b, state_b, saves = _ckpt_train(ckpt_path=path, ckpt_every=2)
    assert hist_b["resumed"]["from_step"] == 4
    assert hist_b["checkpoint"]["resumed_from_step"] == 4
    assert hist_b["checkpoint"]["steps_this_process"] == 2
    assert saves == [6], "the resumed run did not keep checkpointing"
    # The generator's state is the one that decides WHICH mazes and which
    # sampled tokens the remaining steps see, so it is named explicitly.
    assert "generator" in hist_b["resumed"]["rng_streams_restored"]

    hist_a, state_a = killed_cell["history"], killed_cell["adapter"]
    for key, want in hist_a.items():
        if key in _TRACE_TIMING or key in _TRACE_BOOKKEEPING:
            continue
        assert hist_b[key] == want, f"the trace diverged at {key}"
    assert len(hist_b["step"]) == 6
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), (
            f"the resumed adapter differs from the uninterrupted one at {key}")


def test_every_piece_of_the_restored_state_is_load_bearing(killed_cell,
                                                           tmp_path):
    """...so the equality above is not vacuous. Four mutations, four divergences.

    A resume that restored ONLY the adapter would still pass a weak version of
    the test above if the mutated pieces did not matter. Each of these breaks
    exactly one piece of the checkpoint and the continuation must then differ:
    the sampling stream, the AdamW moments, the entropy coefficient, and the
    adapter itself.
    """
    import copy
    ck = maze_orgb.load_cell_checkpoint(killed_cell["pristine"])
    hist_a, state_a = killed_cell["history"], killed_cell["adapter"]

    def resume_from(mutate, name):
        blob = copy.deepcopy(ck)
        mutate(blob)
        path = str(Path(tmp_path) / f"{name}.pt")
        maze_orgb.save_cell_checkpoint(blob, path)
        hist, state, _ = _ckpt_train(ckpt_path=path, ckpt_every=2)
        return hist, state

    def wipe_moments(blob):
        for st in blob["optimiser"]["state"].values():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    st[key] = torch.zeros_like(st[key])

    mutations = {
        # A fresh stream: the maze draw and every sampled token move.
        "rng": lambda b: b["rng"].__setitem__(
            "generator", torch.Generator().manual_seed(12345).get_state()),
        # `rl.py`'s recorded failure mode, from the other side: Adam momentum
        # alone walks a converged policy off its optimum.
        "adam": wipe_moments,
        # The controller restarted at its initial value, which is what
        # `relearn_read` does DELIBERATELY and says so.
        "entropy": lambda b: b.__setitem__(
            "entropy", dict(b["entropy"], coef=0.5, log_coef=math.log(0.5))),
        "adapter": lambda b: b.__setitem__(
            "adapter", {k: torch.zeros_like(v) for k, v in b["adapter"].items()}),
    }
    for name, mutate in mutations.items():
        hist, state = resume_from(mutate, name)
        same_trace = all(hist[k] == hist_a[k] for k in ("loss", "mean_return"))
        same_adapter = all(torch.equal(state_a[k], state[k]) for k in state_a)
        assert not (same_trace and same_adapter), (
            f"breaking the checkpoint's {name} changed nothing, so the "
            "resume-equivalence test proves nothing about it")


def test_the_checkpoint_carries_the_whole_loop_state_and_lands_atomically(
        killed_cell):
    """What is IN the file, read back the way the trainer reads it.

    `load_cell_checkpoint` uses `weights_only=True`, so this also asserts the
    payload is tensors and primitives and nothing else -- a checkpoint is data
    and must never be a pickle that executes on load.
    """
    ck = maze_orgb.load_cell_checkpoint(killed_cell["pristine"])
    assert ck["next_step"] == 4
    assert set(ck) >= {"fingerprint", "next_step", "adapter", "optimiser",
                       "history", "entropy", "counters", "rng",
                       "elapsed_seconds", "saved_at"}
    # The trace so far, and NOT one step more.
    assert ck["history"]["step"] == [0, 1, 2, 3]
    assert len(ck["history"]["policy_entropy"]) == 4
    # The entropy controller's state is the coefficient the NEXT step will use,
    # which is exactly what the uninterrupted run used at step 4.
    assert ck["entropy"]["coef"] == killed_cell["history"]["entropy_coef"][4]
    assert ck["entropy"]["bounds"] == list(maze_orgb.ENTROPY_COEF_BOUNDS)
    # Every RNG stream, the run's own generator first.
    assert set(ck["rng"]) >= {"python", "numpy", "torch", "cuda", "generator"}
    assert isinstance(ck["rng"]["generator"], torch.Tensor)
    # The optimiser's moments, per adapter tensor, and the adapter itself.
    assert ck["optimiser"]["state"], "no AdamW state was saved"
    assert set(ck["adapter"]) and all(v.dtype == torch.float32
                                      for v in ck["adapter"].values())
    fp = ck["fingerprint"]
    assert fp["cell"] == "B" and fp["seed"] == 0 and fp["steps"] == 6
    assert len(fp["pool_sha256"]) == 64
    assert fp["schema"] == maze_orgb.CKPT_SCHEMA
    # ATOMIC: the write goes to `<path>.tmp.<pid>` and is renamed, so a lease
    # that dies during a write leaves the PREVIOUS checkpoint and never a
    # truncated one. Nothing half-written is left behind.
    assert not list(killed_cell["dir"].glob("*.tmp*"))


def test_a_checkpoint_from_a_different_recipe_is_refused(killed_cell, tmp_path):
    """A resume under another recipe would be half one organism and half another.

    The fingerprint covers every knob that changes what a step does, so the
    refusal names the field that moved rather than producing a silent hybrid.
    """
    path = _copy_ckpt(killed_cell, tmp_path, "wrong_lr.pt")
    with pytest.raises(RuntimeError) as err:
        _ckpt_train(ckpt_path=path, ckpt_every=2, lr=0.2)
    assert "lr" in str(err.value) and "DIFFERENT recipe" in str(err.value)

    # And the training-maze pool is part of it: the generator's state says WHERE
    # in the maze stream the resume is, the pool sha says WHICH stream.
    other = _copy_ckpt(killed_cell, tmp_path, "wrong_pool.pt")
    with pytest.raises(RuntimeError) as err2:
        _ckpt_train(ckpt_path=other, ckpt_every=2,
                    train_pool=[maze2.from_layout(CORRIDOR, seed=900 + i)
                                for i in range(2)])
    assert "pool_sha256" in str(err2.value)


def test_a_checkpoint_from_another_exploration_arm_is_refused(killed_cell,
                                                              tmp_path):
    """R2's half of the fingerprint. A cell resumed under another ARM would be
    half one exploration recipe and half another -- and both halves would be
    invisible, because the mode changes no other field a reader looks at."""
    path = _copy_ckpt(killed_cell, tmp_path, "wrong_mode.pt")
    with pytest.raises(RuntimeError) as err:
        _ckpt_train(ckpt_path=path, ckpt_every=2, explore_mode="fixed",
                    entropy_target=None, entropy_coef=0.1)
    assert "explore_mode" in str(err.value)

    hot = _copy_ckpt(killed_cell, tmp_path, "wrong_temperature.pt")
    with pytest.raises(RuntimeError) as err2:
        _ckpt_train(ckpt_path=hot, ckpt_every=2, explore_mode="temperature",
                    sample_temperature=1.5)
    assert "sample_temperature" in str(err2.value)

    # ... and the DECODE WIDTH, because it re-orders the draws off the run's
    # one generator (see the width test above).
    wide = _copy_ckpt(killed_cell, tmp_path, "wrong_width.pt")
    with pytest.raises(RuntimeError) as err3:
        _ckpt_train(ckpt_path=wide, ckpt_every=2, sample_batch_size=2)
    assert "sample_batch_size" in str(err3.value)


def test_resume_off_retrains_the_cell_from_the_warm_start(killed_cell, tmp_path):
    """MZ01_RESUME=0 is the operator saying "discard it", and it must do that."""
    path = _copy_ckpt(killed_cell, tmp_path, "ignored.pt")
    hist, state, _saves = _ckpt_train(ckpt_path=path, ckpt_every=2, resume=False)
    assert hist["resumed"] is None
    assert hist["checkpoint"]["steps_this_process"] == 6
    # Trained from step 0 with the same seed, so it lands where the
    # uninterrupted run landed -- which is what makes "discard it" safe.
    for key in state:
        assert torch.equal(killed_cell["adapter"][key], state[key])


def test_a_checkpoint_that_cannot_be_written_does_not_kill_the_cell(tmp_path):
    """The insurance must never be the thing that loses the lease.

    A full disk or a bad path three hours into a cell has to cost the CHECKPOINT
    and not the training -- but silently losing the insurance is its own failure,
    so the step says so and the row carries the list of failed writes.
    """
    boom = maze_orgb.save_cell_checkpoint

    def explode(payload, path):
        raise OSError(28, "No space left on device")

    maze_orgb.save_cell_checkpoint = explode
    try:
        hist, state, _saves = _ckpt_train(
            ckpt_path=str(tmp_path / "nowhere.pt"), ckpt_every=2, steps=4)
    finally:
        maze_orgb.save_cell_checkpoint = boom
    assert hist is not None and len(hist["step"]) == 4, "the cell died with it"
    assert hist["checkpoint"]["saves_this_process"] == 0
    assert len(hist["checkpoint"]["failed_saves"]) == 2
    assert "No space left" in hist["checkpoint"]["failed_saves"][0]


def _resumable_run(tmp_path, prior_rows=None, warm_start=None, cells=("D",),
                   seeds=(0,)):
    """One toy shard, small enough to run twice: one seed, one cell, no reads."""
    import run_orgb
    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    old_g, old_s = maze2._greedy, maze_orgb._sample_replies
    maze2._greedy = _ScriptedGreedy(_oracle_reply)
    maze_orgb._sample_replies = _ScriptedSampler(_warm_started_policy)
    try:
        return run_orgb.run(model, tok, {
            "n": 7, "steps": 2, "eval_n": 8, "batch_size": 2, "group_size": 3,
            "chunk_size": 4, "pool_size": 8, "warm_start_decisions": 24,
            "warm_start_epochs": 1, "warm_start_batch": 4, "warm_start_mazes": 4,
            "preflight_mazes": 8, "preflight_group_size": 6,
            "eval_batch_size": 8, "max_new_tokens": 40, "nav_adapter": None,
            "instruments": False, "placebo": False, "novel": False,
            "relearn": False, "cells": list(cells), "ckpt_every": 1,
            "warm_start_adapter": warm_start,
            # R2: the oracle sampler is deterministic, so the exploration gate
            # reads a degenerate group and is right to. R1 is what is under
            # test here; see `_run_payload`.
            "preflight_explore_mazes": 4, "preflight_soft": True,
        }, str(tmp_path), list(seeds), prior_rows=prior_rows)
    finally:
        maze2._greedy, maze_orgb._sample_replies = old_g, old_s


def test_the_run_carries_a_finished_row_forward_instead_of_retraining_it(tmp_path):
    """R1 at the shard level: the driver's own result file is the skip list.

    A lease that died on cell 7 of 8 must not spend the next lease's first three
    hours on cell 1 again. The row is carried forward whole -- same adapter sha,
    same measurement, same clock -- and the ledgers are recomputed over it with
    everything else, so a shard finished across two leases has exactly the
    ledgers a shard finished in one would.
    """
    first = _resumable_run(tmp_path)
    assert len(first["rows"]) == 1
    row = first["rows"][0]
    assert row["checkpoint"]["reused_row"] is False
    assert row["checkpoint"]["every"] == 1
    # The mid-cell checkpoint is on disk beside the organism's own adapter, and
    # both are inside the shard's out_dir (so the puller's existing glob takes
    # them).
    assert (tmp_path / "orgb_ckpt_D_s0.pt").exists()
    assert (tmp_path / "orgb_D_s0.pt").exists()

    ws = str(tmp_path / "warm_start.pt")
    second = _resumable_run(tmp_path, prior_rows=first["rows"], warm_start=ws)
    again = second["rows"][0]
    assert again["checkpoint"]["reused_row"] is True
    assert again["adapter_sha256"] == row["adapter_sha256"]
    # The clock is the FIRST lease's, to the float: a retrained cell could not
    # reproduce it, so this is the assertion that the cell was not retrained.
    assert again["seconds"] == row["seconds"]
    assert second["manifest"]["resume"]["rows_reused"] == ["D/s0"]
    # ... and the ledger is recomputed rather than carried, so the pooled file
    # is the file a single lease would have written.
    assert again["ledger"] and again["ledger"]["rows"]


def test_a_row_from_another_base_is_never_carried_forward(tmp_path):
    """M2/M5, at the resume seam. A finished row that stands on a different
    navigator, warm start or cue glyph is not a row of this design, and pooling
    one silently is the confound the sha in every row exists to prevent."""
    import copy
    first = _resumable_run(tmp_path)
    ws = str(tmp_path / "warm_start.pt")
    for field, value in (("nav_adapter_sha256", "0" * 64),
                         ("warm_start_sha256", "1" * 64),
                         ("role_glyphs", {"on_mine": "●"})):
        rows = copy.deepcopy(first["rows"])
        rows[0][field] = value
        with pytest.raises(ValueError) as err:
            _resumable_run(tmp_path, prior_rows=rows, warm_start=ws)
        assert field in str(err.value)
    # And a finished row for a cell this launch is not running would be DELETED
    # by the rewrite, so that stops the run too.
    rows = copy.deepcopy(first["rows"])
    rows[0]["cell"] = "B"
    with pytest.raises(ValueError) as err2:
        _resumable_run(tmp_path, prior_rows=rows, warm_start=ws)
    assert "not in THIS launch's grid" in str(err2.value)


def _entropy_trace_module():
    spec = importlib.util.spec_from_file_location(
        "mz01_entropy_trace", _ROOT / "scripts" / "mz01_entropy_trace.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_entropy_trace_reads_the_fields_the_trainer_actually_writes(
        shared_run, tmp_path):
    """The script and the trainer must agree on the field names, or the tool
    reports NO-TRACE on a perfectly good run and nobody notices."""
    payload, _out = shared_run
    mod = _entropy_trace_module()
    cells = mod.cells_in(payload, "orgb_test.json")
    assert len(cells) == len(payload["rows"])
    for _src, cell, seed, row, hist in cells:
        read = mod.read_cell(hist)
        assert read["steps"] == len(row["training"]["step"])
        assert read["verdict"] in ("PINNED", "FREE")
        assert read["entropy_target"] == row["training"]["config"]["entropy_target"]
        # The bounds come off the cell's OWN config now, not off the script's
        # re-typed copy of the default.
        assert read["bounds_source"] == "the cell's own config"
        assert read["bound"] == list(maze_orgb.ENTROPY_COEF_BOUNDS)
        assert len(mod.sample_at(read, stride=1)) == read["steps"]

    # PINNED is decided by the fraction of steps ON a bound, HH01's rule.
    pinned = mod.read_cell({"entropy_coef": [1.0] * 9 + [0.5],
                            "policy_entropy": [0.1] * 10,
                            "config": {"entropy_coef_bounds": [1e-4, 1.0]}})
    assert pinned["verdict"] == "PINNED" and pinned["steps_at_ceiling"] == 9
    free = mod.read_cell({"entropy_coef": [0.01] * 10,
                          "policy_entropy": [0.7] * 10,
                          "config": {"entropy_coef_bounds": [1e-4, 1.0]}})
    assert free["verdict"] == "FREE" and free["fraction_at_bound"] == 0.0
    # The floor counts too: a controller clamped at 1e-4 is just as pinned.
    floored = mod.read_cell({"entropy_coef": [1e-4] * 10,
                             "policy_entropy": [0.9] * 10, "config": {}})
    assert floored["verdict"] == "PINNED" and floored["steps_at_floor"] == 10
    assert floored["bounds_source"].startswith("calibration.maze_orgb")


def test_the_entropy_trace_runs_under_a_bare_python3(shared_run, tmp_path):
    """`-S` drops site-packages: no torch, no numpy, standard library only."""
    payload, _out = shared_run
    path = tmp_path / "orgb_trace.json"
    path.write_text(json.dumps(payload))
    proc = subprocess.run(
        [sys.executable, "-S", str(_ROOT / "scripts" / "mz01_entropy_trace.py"),
         "--stride", "1", str(path)],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    for needle in ("cell D seed 0", "VERDICT:", "at a bound:", "coef", "H ",
                   "solve", "cell(s) read"):
        assert needle in proc.stdout, needle
    assert "Traceback" not in proc.stderr
    # And the machine-readable shape, with HH01's field names on it.
    as_json = subprocess.run(
        [sys.executable, "-S", str(_ROOT / "scripts" / "mz01_entropy_trace.py"),
         "--json", str(path)], capture_output=True, text=True, timeout=300)
    assert as_json.returncode == 0, as_json.stderr
    blocks = json.loads(as_json.stdout)
    assert len(blocks) == len(payload["rows"])
    for block in blocks:
        assert set(block) >= {"cell", "seed", "steps", "steps_at_bound",
                              "fraction_at_bound", "first_at_bound",
                              "min_policy_entropy", "final_policy_entropy",
                              "bound", "pinned", "verdict", "sampled"}


# ------------------------------------------- critic pass, 2026-08-29 (R1)
#
# Four defects the eighth pass left in the lease-death recovery. Each is a
# SILENT wrong result rather than a loud stop, which is the only kind worth a
# test here: a loud stop costs a lease, a silent one costs the experiment.


def test_a_row_from_another_recipe_is_never_carried_forward(tmp_path):
    """The cell-skip checked the BASE (navigator, warm start, cue glyph) and the
    step count, and nothing else. A relaunch under a different MZ01_YELL_W,
    MZ01_GROUP, MZ01_BATCH, entropy target or grid size therefore carried the
    old rows forward silently -- the manifest names only the new recipe, nothing
    downstream re-reads a row's own training config, and the 2x2 ends up half
    one recipe and half another with no field on disk saying so.
    """
    import copy
    first = _resumable_run(tmp_path)
    ws = str(tmp_path / "warm_start.pt")
    for where, value, needle in (
            (("training", "config", "group_size"), 4, "group_size: row 4"),
            (("training", "config", "batch_size"), 8, "batch_size: row 8"),
            (("training", "config", "yell_weight"), 9.5, "yell_weight: row 9.5"),
            (("training", "config", "entropy_target"), 0.2,
             "entropy_target: row 0.2"),
            (("training", "config", "lr"), 5e-4, "lr: row 0.0005"),
            (("training", "config", "n"), 11, "n: row 11"),
            # ... and the HELD-OUT SET the row was measured on, which is a pure
            # function of (eval_n, n): a row measured on another one cannot be
            # pooled with rows measured on this one.
            (("measurement", "n_mazes"), 5, "n_mazes: row 5")):
        rows = copy.deepcopy(first["rows"])
        target = rows[0]
        for key in where[:-1]:
            target = target[key]
        target[where[-1]] = value
        with pytest.raises(ValueError) as err:
            _resumable_run(tmp_path, prior_rows=rows, warm_start=ws)
        assert "DIFFERENT RECIPE" in str(err.value), where
        assert needle in str(err.value), (where, str(err.value))
    # A row that simply predates a field is not evidence of drift: it is carried
    # forward, as it was before, rather than blocking the shard.
    rows = copy.deepcopy(first["rows"])
    rows[0]["training"]["config"].pop("group_size")
    second = _resumable_run(tmp_path, prior_rows=rows, warm_start=ws)
    assert second["manifest"]["resume"]["rows_reused"] == ["D/s0"]


def test_the_shard_result_file_is_written_atomically(tmp_path):
    """The one artefact that recovers a dead lease was rewritten in place after
    every organism: `open(path, "w")` truncates first, so a death mid-write left
    a prefix of a JSON document and the next launch's `read_prior_rows` then
    refuses to start with every measured organism in the shard lost."""
    drv = _import_driver({})
    path = str(tmp_path / "orgb_s0_DB.json")
    drv.write_payload(path, {"rows": [{"cell": "D"}]})
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh)["rows"][0]["cell"] == "D"
    assert not list(tmp_path.glob("*.tmp*")), "a temp file was left behind"

    class NotJSON:
        pass

    # A write that dies part-way leaves the PREVIOUS payload, never a prefix.
    with pytest.raises(TypeError):
        drv.write_payload(path, {"rows": [{"cell": "A"}], "bad": NotJSON()})
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh)["rows"][0]["cell"] == "D"
    assert not list(tmp_path.glob("*.tmp*"))


def test_a_relaunch_on_other_weights_refuses_the_shards_rows(tmp_path):
    """`run_orgb.run` checks the navigator, the warm start, the cue glyph and
    now the recipe -- but it never sees the base checkpoint. A relaunch with
    another MZ01_REVISION or MZ01_MODEL would pool organisms from two sets of
    weights under one 2x2 and the manifest would name only the second."""
    drv = _import_driver({})
    path = str(tmp_path / "orgb_s0_DB.json")
    same = {"model_key": "14B", "model_id": "Qwen/Qwen3-14B",
            "revision": "abc123", "shard": "s0_DB"}
    drv.write_payload(path, {"manifest": dict(same),
                             "rows": [{"cell": "D", "seed": 0}]})
    assert len(drv.read_prior_rows(path, expect=same)) == 1
    for field, other in (("revision", "def456"), ("model_id", "Qwen/Qwen3-4B"),
                         ("model_key", "4B")):
        with pytest.raises(SystemExit) as err:
            drv.read_prior_rows(path, expect={**same, field: other})
        assert field in str(err.value)
    # MZ01_RESUME=0 does not read the file at all, so it cannot object to it.
    assert drv.read_prior_rows(path, resume=False,
                               expect={**same, "revision": "def456"}) == []
    # ... and a file whose manifest predates a field is not a mismatch.
    drv.write_payload(path, {"manifest": {"shard": "s0_DB"},
                             "rows": [{"cell": "D", "seed": 0}]})
    assert len(drv.read_prior_rows(path, expect=same)) == 1


def test_a_killed_writers_temp_checkpoint_is_swept(tmp_path):
    """`_atomic_torch_save`'s cleanup cannot run on SIGKILL, which is exactly
    how a lease ends, so every death mid-write left a ~126 MB
    `..._ckpt_D_s0.pt.tmp.<pid>` on the box -- inside the tree
    `scripts/pull_from_sandbox.sh` mirrors, where one big file per cycle at
    ~1 min/MB means a corpse delays the organism adapters by hours."""
    path = tmp_path / "orgb_ckpt_D_s0.pt"
    orphans = [tmp_path / "orgb_ckpt_D_s0.pt.tmp.999998",
               tmp_path / "orgb_ckpt_D_s0.pt.tmp.999999"]
    for orphan in orphans:
        orphan.write_bytes(b"x" * 4096)
    # A DIFFERENT cell's temp file is somebody else's business; only this
    # path's orphans are swept.
    other = tmp_path / "orgb_ckpt_A_s0.pt.tmp.999997"
    other.write_bytes(b"y" * 16)
    maze_orgb.save_cell_checkpoint({"adapter": torch.zeros(3)}, str(path))
    assert path.exists()
    assert not any(o.exists() for o in orphans)
    assert other.exists()
    assert torch.equal(
        maze_orgb.load_cell_checkpoint(path)["adapter"], torch.zeros(3))


def test_every_row_records_what_bit_exact_does_and_does_not_mean(shared_run):
    """The eighth pass told the operator, in the status file and in three
    docstrings, that a resumed cell is 'bit-for-bit' the cell an uninterrupted
    lease would have trained. The test behind that sentence is CPU fp32; the
    lease is CUDA bf16, where neither the forward nor the backward is
    bit-reproducible run to run. The transferable claim is that every element of
    the loop's state is restored, and it now rides in every row."""
    payload, _out = shared_run
    for row in payload["rows"]:
        scope = row["checkpoint"]["bit_exact_scope"]
        assert "CPU fp32" in scope and "bf16" in scope
        assert "NOT the same floats" in scope
    driver = (_ROOT / "scripts" / "mz01_orgb_driver.py").read_text()
    runner = (_ROOT / "experiments" / "v2" / "MZ01_maze_world"
              / "run_orgb.py").read_text()
    for text in (driver, runner):
        # The qualification is present ...
        assert "CPU fp32" in text
        # ... and the three unqualified sentences do not come back.
        for gone in ("is bit-for-bit the cell",
                     "resumes at that step, bit for bit"):
            assert gone not in text, gone


def test_a_carried_row_whose_adapter_vanished_is_recorded_not_hidden(tmp_path):
    """A renewed lease comes back on an OLD DISK SNAPSHOT, and the shard JSON is
    pulled every cycle while a 21 MB adapter is one big file per cycle -- so the
    row can survive the death while the organism's weights do not. The row is
    still carried (refusing it would throw a measured organism away) but the
    missing artefact is named in the manifest and in the log, because a row
    whose adapter cannot be found is one nothing can ever re-read."""
    first = _resumable_run(tmp_path)
    ws = str(tmp_path / "warm_start.pt")
    intact = _resumable_run(tmp_path, prior_rows=first["rows"], warm_start=ws)
    assert intact["manifest"]["resume"]["rows_reused_without_adapter"] == []

    (tmp_path / "orgb_D_s0.pt").unlink()
    second = _resumable_run(tmp_path, prior_rows=first["rows"], warm_start=ws)
    assert second["manifest"]["resume"]["rows_reused"] == ["D/s0"]
    gone = second["manifest"]["resume"]["rows_reused_without_adapter"]
    assert len(gone) == 1 and "D/s0" in gone[0] and "is missing" in gone[0]

    # ... and a file that is present but is NOT the organism the row names is
    # the same finding, not a pass.
    (tmp_path / "orgb_D_s0.pt").write_bytes(b"not an adapter")
    third = _resumable_run(tmp_path, prior_rows=first["rows"], warm_start=ws)
    assert "hashes" in third["manifest"]["resume"][
        "rows_reused_without_adapter"][0]


# ---------------------------------- R2, 2026-08-29: exploration and the trial
#
# WHAT THESE ARE ABOUT. Three shards trained with a DEAD MOVE TERM: the
# warm-started navigator is nearly deterministic, so a maze's `group_size`
# sampled episodes are the SAME episode, their returns are equal,
# `group_advantages` returns zeros and the policy gradient is exactly nothing
# (their traces read |g| ~ 0.01 at solve 1.000). Every test below is a property
# of that arithmetic or of the machinery that measures it -- none is a property
# of the model -- so the toy path decides them exactly as a 14B lease would.

def test_the_zero_advantage_fraction_is_the_share_of_flat_groups():
    """The one number the trace was missing, and its link to the advantages.

    Read beside `group_advantages` itself, because "the fraction of flat
    groups" only matters if a flat group really does contribute nothing: the
    second half of this asserts that, so the first half is not a number about
    a claim nobody checked.
    """
    from calibration.maze_rl import group_advantages
    # two groups, one flat and one not
    assert maze_orgb.zero_advantage_group_fraction(
        [1.0, 1.0, 2.0, 3.0], ["a", "a", "b", "b"]) == 0.5
    assert maze_orgb.zero_advantage_group_fraction(
        [4.0, 4.0, 4.0, 4.0], ["a", "a", "b", "b"]) == 1.0
    assert maze_orgb.zero_advantage_group_fraction(
        [1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"]) == 0.0
    # A group of ONE is degenerate and counts as such: its single advantage is
    # its own value minus itself.
    assert maze_orgb.zero_advantage_group_fraction([7.0], ["a"]) == 1.0
    assert maze_orgb.zero_advantage_group_fraction([], []) is None
    # ... and that is exactly what the advantages do with it.
    rets = [-4.0, -4.0, -4.0, -6.0, -4.0, -4.0]
    keys = [0, 0, 0, 1, 1, 1]
    adv = group_advantages(rets, keys)
    assert adv[:3] == [0.0, 0.0, 0.0]
    assert max(abs(a) for a in adv[3:]) > 0
    assert maze_orgb.zero_advantage_group_fraction(rets, keys) == 0.5


def test_every_exploration_mode_resolves_to_the_numbers_it_claims():
    """The mode -> numbers map, which is the only place a name becomes a run."""
    m = maze_orgb
    # the default is TODAY'S recipe, untouched
    base = m.exploration_plan(measured_entropy=0.02)
    assert base["mode"] == m.EXPLORE_THERMOSTAT
    assert base["entropy_target"] == 0.7 and base["entropy_coef"] == 0.01
    assert base["entropy_adaptive"] and not base["off_policy"]

    # `relative`: the FLOOR binds on a near-deterministic policy, which is the
    # whole reason the floor exists -- 0.85 x 0.02 is a target the policy
    # already meets, so the relative half alone would buy no exploration.
    rel = m.exploration_plan("relative", measured_entropy=0.02)
    assert rel["entropy_target"] == m.ENTROPY_MIN
    assert rel["knobs"]["floor_binds"] and rel["knobs"]["binding"] == "entropy_min"
    assert abs(rel["knobs"]["relative_target"] - 0.017) < 1e-12
    # ... and the relative half binds once the policy has entropy of its own.
    rel2 = m.exploration_plan("relative", measured_entropy=1.0)
    assert abs(rel2["entropy_target"] - 0.85) < 1e-12
    assert not rel2["knobs"]["floor_binds"]
    # It REFUSES to guess: a relative target derived from a constant is the
    # fixed target it replaces.
    with pytest.raises(ValueError) as err:
        m.exploration_plan("relative")
    assert "measured" in str(err.value)

    # `fixed`: no target at all, so `train_orgb_cell` reads it as not adaptive.
    fix = m.exploration_plan("fixed", measured_entropy=0.02)
    assert fix["entropy_target"] is None and not fix["entropy_adaptive"]
    assert fix["entropy_coef"] == m.ENTROPY_COEF_FIXED

    # `temperature`: the SAMPLER moves and the controller does not.
    hot = m.exploration_plan("temperature", measured_entropy=0.02)
    assert hot["entropy_target"] == base["entropy_target"]
    assert hot["sample_temperature"] == m.SAMPLE_TEMPERATURE
    assert hot["temperature"] == 1.0 and hot["off_policy"]
    assert hot["knobs"]["importance_weight"] == "MEASURED, NOT APPLIED"

    # The pin, which is what makes a `relative` run survive a second lease.
    pinned = m.exploration_plan("relative", measured_entropy=0.02,
                                pinned_target=0.2718)
    assert pinned["entropy_target"] == 0.2718
    assert pinned["knobs"]["derived_target"] == m.ENTROPY_MIN
    assert "PINNED" in pinned["note"]
    with pytest.raises(ValueError):
        m.exploration_plan("fixed", pinned_target=0.3)
    with pytest.raises(ValueError):
        m.exploration_plan("no-such-mode", measured_entropy=0.1)


def _explore_train(**kw):
    """A short toy cell under one arm. -> its history. Scripted sampler."""
    model, tok = _model(seed=1)
    old = maze_orgb._sample_replies
    maze_orgb._sample_replies = _ScriptedSampler(_warm_started_policy)
    try:
        hist = maze_orgb.train_orgb_cell(
            model, tok, cell="D", seed=0, steps=kw.pop("steps", 3),
            batch_size=2, group_size=kw.pop("group_size", 3), lr=0.1,
            chunk_size=4, fmt=ASCII, log_every=0,
            train_pool=[maze2.from_layout(MINEFIELD, seed=200 + i)
                        for i in range(2)], **kw)
    finally:
        maze_orgb._sample_replies = old
    remove_lora(model)
    return hist


def test_a_fixed_coefficient_never_moves_and_the_thermostat_always_does():
    """`fixed` is a control on the RUNAWAY, so the thing it must not do is run.

    The pre-registered controller moves `log_coef` by
    `entropy_lr * (target - measured)` EVERY step, and on this policy the
    difference never changes sign -- which is how the real run's coefficient
    walked to its 1.0 ceiling by step ~133 and spent the rest of the cell with
    no authority left. With `entropy_target=None` the update is not reached at
    all and the coefficient is the same float on every step of the trace.
    """
    fixed = _explore_train(explore_mode="fixed", entropy_target=None,
                           entropy_coef=maze_orgb.ENTROPY_COEF_FIXED)
    assert fixed["entropy_coef"] == [maze_orgb.ENTROPY_COEF_FIXED] * 3
    assert fixed["config"]["entropy_adaptive"] is False
    assert fixed["config"]["explore"]["mode"] == "fixed"

    hot = _explore_train(explore_mode="thermostat_fixed_target",
                         entropy_target=0.7, entropy_coef=0.01)
    coefs, ents = hot["entropy_coef"], hot["policy_entropy"]
    assert len(set(coefs)) == len(coefs), coefs
    # The trace records the step's OWN coefficient, before that step's update.
    assert coefs[0] == 0.01
    # And the update is exactly the controller's law, step by step -- which is
    # the thing that ran away on the lease: the sign of (target - measured)
    # never changes, so the coefficient moves the same way EVERY step until it
    # hits a bound.
    lo, hi = maze_orgb.ENTROPY_COEF_BOUNDS
    for i in range(len(coefs) - 1):
        want = min(max(coefs[i] * math.exp(0.05 * (0.7 - ents[i])), lo), hi)
        assert abs(coefs[i + 1] - want) < 1e-12, (i, coefs, ents)
    # THE RUNAWAY, reproduced. A target the policy cannot reach -- the lease's
    # 0.7 against a measured 0.02, and here 5.0 against a ceiling of ln 4 --
    # makes (target - measured) positive at EVERY step, so the coefficient
    # climbs monotonically until it is pinned on the ceiling and the run has no
    # entropy authority left. That is what `mz01_entropy_trace.py` reads.
    away = _explore_train(explore_mode="thermostat_fixed_target",
                          entropy_target=5.0, entropy_coef=0.01, steps=4)
    up = away["entropy_coef"]
    assert all(b > a for a, b in zip(up, up[1:])), up
    assert all(h < 5.0 for h in away["policy_entropy"])


def test_the_relative_target_travels_from_the_preflight_into_the_trainer():
    """The gate measures the entropy; the arm trains on a target derived from
    THAT measurement. If the two came apart, the cells would run a recipe no
    gate had read."""
    model, tok = _model(seed=1)
    old = maze_orgb._sample_replies
    maze_orgb._sample_replies = _ScriptedSampler(_warm_started_policy)
    try:
        pre = maze_orgb.preflight_exploration_check(
            model, tok, _mazes(4), generator=torch.Generator().manual_seed(0),
            group_size=3, fmt=ASCII, device=torch.device("cpu"), chunk_size=4,
            max_new_tokens=40, explore_mode="relative", soft=True)
    finally:
        maze_orgb._sample_replies = old
    remove_lora(model)
    plan = pre["plan"]
    assert plan["measured_entropy"] == pre["policy_entropy"]
    assert plan["entropy_target"] == max(
        maze_orgb.ENTROPY_REL * pre["policy_entropy"], maze_orgb.ENTROPY_MIN)
    # The ratio the gate reads is against the target IN FORCE, not the config's.
    assert pre["entropy_target"] == plan["entropy_target"]
    assert (pre["entropy_target_over_measured"]
            == plan["entropy_target"] / pre["policy_entropy"])
    # And the line an operator copies before booking a SECOND lease is there.
    assert any("MZ01_ENTROPY_TARGET=" in line for line in pre["lines"])

    hist = _explore_train(explore_mode="relative",
                          entropy_target=plan["entropy_target"],
                          entropy_coef=plan["entropy_coef"],
                          explore_knobs=plan["knobs"])
    assert hist["config"]["entropy_target"] == plan["entropy_target"]
    assert hist["config"]["explore"]["knobs"] == plan["knobs"]


def test_the_exploration_preflight_refuses_a_policy_that_cannot_explore():
    """R2's gate, on the policy that caused it: a deterministic navigator.

    `_oracle_reply` walks the shortest path, so every episode of a group is the
    same episode -- exactly the state the three shards trained in. PR-3's own
    gate PASSES this (its remark placement is random), which is the point: the
    two gates ask about different terms and only one of them was being asked.
    """
    model, tok = _model()
    old = maze_orgb._sample_replies
    # A deterministic MOVE policy with a random remark: PR-3 passes, R2 fails.
    maze_orgb._sample_replies = _ScriptedSampler(_warm_started_policy)
    try:
        out = maze_orgb.preflight_exploration_check(
            model, tok, _mazes(8), generator=torch.Generator().manual_seed(0),
            group_size=4, fmt=ASCII, device=torch.device("cpu"), chunk_size=4,
            max_new_tokens=40)
        soft = maze_orgb.preflight_exploration_check(
            model, tok, _mazes(8), generator=torch.Generator().manual_seed(0),
            group_size=4, fmt=ASCII, device=torch.device("cpu"), chunk_size=4,
            max_new_tokens=40, soft=True)
    finally:
        maze_orgb._sample_replies = old
    remove_lora(model)
    assert out["zero_advantage_group_fraction"] == 1.0
    assert not out["passed"] and not out["hard_passed"]
    assert out["criteria"][0]["name"] == "zero_advantage_group_fraction"
    assert not out["criteria"][0]["ok"]
    assert out["within_group_return_sd"] == 0.0
    assert "no cell starts" in out["lines"][-1]
    assert "MZ01_GROUP" in out["remediation"]
    # The gate is read on the HARSHER return definition, and both are reported.
    assert out["zero_advantage_group_fraction_with_mine_cost"] is not None
    assert "mine_cost=False" in out["gated_on"]
    # soft: the same reading, reported instead of applied.
    assert soft["passed"] and not soft["hard_passed"]
    assert soft["soft_failures"] == ["zero_advantage_group_fraction"]
    assert "SOFT" in soft["lines"][-1]
    # A target 15x the measured entropy fails the second criterion too, which
    # is the reading that says the thermostat has no authority.
    assert soft["entropy_target_over_measured"] > 3.0 or \
        soft["criteria"][1]["ok"]


def test_a_failed_exploration_preflight_stops_the_run_before_any_cell_trains(
        tmp_path):
    """The stop rule, executed through `run_orgb.run`: no rows, a NOT-MEASURED
    note that names the remedy, and no adapter on disk."""
    import run_orgb
    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    old_g, old_s = maze2._greedy, maze_orgb._sample_replies
    maze2._greedy = _ScriptedGreedy(_oracle_reply)
    maze_orgb._sample_replies = _ScriptedSampler(_warm_started_policy)
    cfg = {
        "n": 7, "steps": 1, "eval_n": 8, "batch_size": 2, "group_size": 3,
        "chunk_size": 4, "pool_size": 8, "warm_start_decisions": 24,
        "warm_start_epochs": 1, "warm_start_batch": 4, "warm_start_mazes": 4,
        "preflight_mazes": 8, "preflight_group_size": 6,
        "preflight_explore_mazes": 4, "eval_batch_size": 8,
        "max_new_tokens": 40, "nav_adapter": None, "instruments": False,
        "placebo": False, "novel": False, "relearn": False, "cells": ["D"],
    }
    try:
        hard = run_orgb.run(model, tok, dict(cfg), str(tmp_path), [0])
    finally:
        maze2._greedy, maze_orgb._sample_replies = old_g, old_s
    # PR-3 passed (the remark placement is random) and R2 did not: the two
    # gates ask about the two terms and only one of them existed before.
    assert hard["preflight"]["passed"]
    assert not hard["preflight_exploration"]["hard_passed"]
    assert hard["rows"] == []
    assert any("EXPLORATION PRE-FLIGHT FAILED" in n
               for n in hard["not_measured"])
    assert not list(Path(tmp_path).glob("orgb_D_*.pt"))
    # ... and the manifest carries the reading whatever the verdict was.
    assert hard["preflight_exploration"]["policy_entropy"] is not None


def test_the_soft_preflight_records_the_deviation_it_waves_through(shared_run):
    """MZ01_PREFLIGHT_SOFT is a DEVIATION, and a deviation that is not written
    down is a recipe change nobody can find afterwards."""
    payload, _out = shared_run
    ex = payload["preflight_exploration"]
    assert ex["soft"] and not ex["hard_passed"] and ex["passed"]
    assert any("EXPLORATION PRE-FLIGHT REPORTED, NOT APPLIED" in n
               for n in payload["not_measured"])
    man = payload["manifest"]["exploration"]
    assert man["plan"]["mode"] == maze_orgb.DEFAULT_EXPLORE_MODE
    assert man["measured_policy_entropy"] == ex["policy_entropy"]
    assert man["bars"]["zero_advantage"] == maze_orgb.ZERO_ADVANTAGE_BAR
    assert man["bars"]["entropy_coef_bounds"] == list(
        maze_orgb.ENTROPY_COEF_BOUNDS)
    # critic pass 2: the informational criterion is in the manifest, said to be
    # informational, beside the sampler that produced the reading.
    assert man["entropy_target_over_measured_is_informational"] is True
    assert "entropy_target_over_measured" in man
    assert man["sampler"]["draw_temperature"] == man["sampler"][
        "score_temperature"] and not man["sampler"]["off_policy"]
    # Every row's config says which arm trained it, resolved, not by name only.
    for row in payload["rows"]:
        ex_cfg = row["training"]["config"]["explore"]
        assert ex_cfg["mode"] == maze_orgb.DEFAULT_EXPLORE_MODE
        assert ex_cfg["off_policy"] is False
        assert "identically 1" in ex_cfg["importance_weight"]
        assert row["training"]["zero_advantage_group_fraction_mean"] is not None
        assert row["measurement"]["eval_batch_size"] == 8


def test_the_temperature_arm_draws_at_one_temperature_and_scores_at_another():
    """The `temperature` arm's whole claim, on the REAL sampler.

    Two halves. (a) the ids are drawn from `softmax(logits / T_sample)` and the
    recorded log importance ratio is EXACTLY
    `log softmax(logits)[a] - log softmax(logits / T_sample)[a]`, recomputed
    here from a plain forward on the same prompts -- so the number the manifest
    reports is the real off-policy distance and not an approximation of it.
    (b) `_rescore`, which is what the LOSS sees, has no temperature in it at
    all: the same rows score identically whatever the sampler drew them at.
    """
    model, tok = _model(seed=3)
    device = torch.device("cpu")
    prompts = ["down here ", "left and then "]
    t_sample = 1.5

    aux = []
    gen = torch.Generator().manual_seed(11)
    ids = maze_orgb._sample_replies(
        model, tok, prompts, max_new_tokens=1, temperature=t_sample,
        generator=gen, batch_size=2, device=device,
        score_temperature=1.0, aux=aux)
    assert [len(g) for g in ids] == [1, 1]
    assert len(aux) == 2

    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left")
    att = enc["attention_mask"]
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"], attention_mask=att,
                       position_ids=pos).logits[:, -1, :].float()
    lp_one = torch.log_softmax(logits, dim=-1)
    lp_hot = torch.log_softmax(logits / t_sample, dim=-1)
    for j, row in enumerate(ids):
        want = float(lp_one[j, row[0]]) - float(lp_hot[j, row[0]])
        assert abs(aux[j] - want) < 1e-5, (aux[j], want)
    # A hotter behaviour policy is FLATTER, so the token it drew is (weakly)
    # more likely under the colder scoring policy: the ratio is >= 0 only when
    # the draw was the mode, which is why this is REPORTED and not applied.
    assert all(r is not None for r in aux)

    # (b) the loss sees no temperature: `_rescore` is a plain log_softmax.
    rows = _decision_rows(tok, prompts, ["down", "left"])
    cols = maze_orgb._move_cols(tok, device)
    with torch.no_grad():
        a = maze_orgb._rescore(model, tok, rows, cols=cols, device=device,
                               max_new_tokens=40)[0]
        b = maze_orgb._rescore(model, tok, rows, cols=cols, device=device,
                               max_new_tokens=40)[0]
    assert torch.equal(a, b)
    remove_lora(model)


def test_the_temperature_arm_records_the_ratio_it_does_not_apply():
    """End to end on the trainer, with the real sampler: the off-policy arm
    records a log ratio per step and the on-policy one records None.

    The claim being pinned is the honest one from `train_orgb_cell`'s
    IMPORTANCE WEIGHT paragraph: the correction is OMITTED, deliberately, and
    what stands in for it is a measurement of how far off-policy the samples
    were -- in the trace, in the config, and in words.
    """
    def train(sample_temperature):
        model, tok = _model(seed=2)
        hist = maze_orgb.train_orgb_cell(
            model, tok, cell="D", seed=0, steps=2, batch_size=1, group_size=2,
            lr=0.1, chunk_size=4, fmt=ASCII, log_every=0, max_new_tokens=4,
            train_pool=[maze2.from_layout(MINEFIELD, seed=201)],
            explore_mode=("temperature" if sample_temperature else
                          "thermostat_fixed_target"),
            sample_temperature=sample_temperature)
        remove_lora(model)
        return hist

    hot = train(1.5)
    cold = train(None)
    assert hot["config"]["explore"]["off_policy"] is True
    assert hot["config"]["sample_temperature"] == 1.5
    assert hot["config"]["explore"]["knobs"] == {}
    assert "NOT APPLIED" in hot["config"]["explore"]["importance_weight"]
    assert all(r is not None for r in hot["log_importance_ratio"])
    assert hot["log_importance_ratio_mean"] is not None

    assert cold["config"]["explore"]["off_policy"] is False
    assert cold["config"]["sample_temperature"] == 1.0
    assert cold["log_importance_ratio"] == [None, None]
    assert cold["log_importance_ratio_mean"] is None
    assert "identically 1" in cold["config"]["explore"]["importance_weight"]


def _spy_sampler(seen, inner):
    """Wraps a sampler and records the WIDTH of every decode call it would make."""
    def spy(model, tok, texts, *, batch_size, **kw):
        seen["batch"].append(batch_size)
        for lo in range(0, len(texts), batch_size):
            seen["widths"].append(len(texts[lo:lo + batch_size]))
        seen["prompts"].append(len(texts))
        return inner(model, tok, texts, batch_size=batch_size, **kw)
    return spy


def _train_with_spy(seen, **kw):
    model, tok = _model(seed=1)
    old = maze_orgb._sample_replies
    maze_orgb._sample_replies = _spy_sampler(
        seen, _ScriptedSampler(_warm_started_policy))
    try:
        hist = maze_orgb.train_orgb_cell(
            model, tok, cell="D", seed=0, steps=1, batch_size=2, lr=0.1,
            chunk_size=4, fmt=ASCII, log_every=0,
            train_pool=[maze2.from_layout(MINEFIELD, seed=200 + i)
                        for i in range(2)], **kw)
    finally:
        maze_orgb._sample_replies = old
    remove_lora(model)
    return hist


def test_group_twelve_runs_and_does_not_widen_the_decode():
    """MZ01_GROUP=12 as a trial variable: it costs MORE CALLS, not wider ones.

    The memory question is the one the box asks -- at GROUP=6 the card sat at
    53 GB of 96, and doubling the group would be fatal if the group SIZED a
    tensor. It does not. Episodes run in lockstep and the decode is cut into
    calls of `sample_batch_size` (default `chunk_size`) while the re-scoring is
    cut into chunks of `chunk_size`, so a wider group is more iterations of the
    same shapes and no call ever sees more than `sample_batch_size` prompts.
    """
    twelve, six = {"widths": [], "batch": [], "prompts": []}, \
        {"widths": [], "batch": [], "prompts": []}
    h12 = _train_with_spy(twelve, group_size=12, sample_batch_size=4,
                          max_new_tokens=6)
    h6 = _train_with_spy(six, group_size=6, max_new_tokens=6)
    assert max(twelve["widths"]) <= 4 and set(twelve["batch"]) == {4}
    assert h12["config"]["group_size"] == 12
    assert h12["config"]["sample_batch_size"] == 4
    # The default is `chunk_size` -- which is what it silently was before R2.
    assert set(six["batch"]) == {4}
    assert h6["config"]["sample_batch_size"] == 4
    # Twice the group is (about) twice the decisions, at the same call width.
    assert h12["n_decisions"][0] > 1.5 * h6["n_decisions"][0]
    assert max(twelve["prompts"]) > max(six["prompts"])
    assert h12["config"]["explore"]["group_size"] == 12


def test_the_decode_width_is_part_of_the_recipe_and_the_fingerprint_says_so():
    """`sample_batch_size` CHANGES WHICH TOKENS ARE DRAWN, so it is a recipe
    field and not a performance knob.

    `torch.multinomial` consumes one draw per ROW per step, so cutting the same
    prompts into calls of a different width re-orders the draws off the run's
    one generator and the sampled ids diverge. That is why the mid-cell
    checkpoint's fingerprint names it: a resume at another width would be a
    different training set from that step on, silently -- the same argument
    `derive_generator` makes about a shifted stream.
    """
    model, tok = _model(seed=4)
    texts = ["down here ", "left then ", "up and ", "right by ", "go "]

    def draw(width):
        return maze_orgb._sample_replies(
            model, tok, texts, max_new_tokens=2, temperature=1.0,
            generator=torch.Generator().manual_seed(5), batch_size=width,
            device=torch.device("cpu"))

    assert draw(5) != draw(2), (
        "the sampler drew the same ids at two widths, so this test cannot see "
        "the re-ordering the fingerprint exists to refuse")
    assert draw(5) == draw(5)
    remove_lora(model)

    hist = _train_with_spy({"widths": [], "batch": [], "prompts": []},
                           group_size=3, sample_batch_size=2, max_new_tokens=6)
    assert hist["config"]["sample_batch_size"] == 2


def test_the_eval_decode_is_the_same_decode_at_any_batch_size():
    """MZ01_EVAL_BATCH may not change a single decision. Asserted, not argued.

    Greedy decoding is argmax per row; a left pad is masked out of every real
    token's attention; a row that emits EOS stops growing. So a row's reply
    cannot depend on which other rows shared its decode call, and three widths
    over the same mazes must produce identical records -- move for move.

    THE EXCEPTION IS `stop_on_parse=True`, whose early stop is evaluated over
    the WHOLE call and therefore does depend on the batch. Every eval that
    reads a remark passes False (D30), and the second half of this asserts that
    `measure_organism` is one of them, so the invariance above is a property of
    the path this stage actually runs rather than of the flag it happens to use.
    """
    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    mazes = _mazes(8, n=7)
    runs = {}
    for width in (1, 4, 32):
        recs, agg = maze2.rollout(
            model, tok, mazes, retry=True, glyphs=maze2.Glyphs(), fmt=ASCII,
            batch_size=width, device=torch.device("cpu"), max_new_tokens=6,
            stop_on_parse=False, order_seed=7000)
        runs[width] = (recs, agg)
    base_recs, base_agg = runs[1]
    for width in (4, 32):
        recs, agg = runs[width]
        assert len(recs) == len(base_recs)
        for a, b in zip(base_recs, recs):
            assert a["layout"] == b["layout"]
            assert a["moves"] == b["moves"], (width, a["moves"], b["moves"])
            assert a["replies"] == b["replies"]
            assert a["landings"] == b["landings"]
            assert a["retried"] == b["retried"]
        assert agg["overall"] == base_agg["overall"]

    # ... and the eval this stage runs is the `stop_on_parse=False` one.
    seen = []
    real = maze2.rollout

    def spy(*a, **kw):
        seen.append(kw.get("stop_on_parse"))
        return real(*a, **kw)

    old_r, old_g = maze2.rollout, maze2._greedy
    maze2.rollout = spy
    maze2._greedy = _ScriptedGreedy(_oracle_reply)
    try:
        out = maze_orgb.measure_organism(
            model, tok, mazes[:2], cell="D", seed=0, fmt=ASCII, batch_size=32,
            device=torch.device("cpu"), placebo=True, novel=False,
            max_new_tokens=40)
    finally:
        maze2.rollout, maze2._greedy = old_r, old_g
    assert seen and set(seen) == {False}
    assert out["eval_batch_size"] == 32


def _explore_scorer():
    spec = importlib.util.spec_from_file_location(
        "score_mz01_explore", _ROOT / "scripts" / "score_mz01_explore.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trial_module():
    """`scripts/mz01_explore_trial.py` imported with no model and no box.

    Its module body reads the environment and inserts `/marimo/repo` paths;
    `main()` is what needs a GPU, and nothing here calls it.
    """
    import os
    spec = importlib.util.spec_from_file_location(
        "mz01_explore_trial", _ROOT / "scripts" / "mz01_explore_trial.py")
    mod = importlib.util.module_from_spec(spec)
    old = os.environ.get("MZ01_REVISION")
    os.environ["MZ01_REVISION"] = "deadbeef"
    try:
        spec.loader.exec_module(mod)
    finally:
        if old is None:
            os.environ.pop("MZ01_REVISION", None)
        else:
            os.environ["MZ01_REVISION"] = old
    return mod


def _fake_arm(name, mode, *, zero_end, solve=(1.0, 1.0), sps=36.0, steps=20,
              group=6, target=0.3, knobs=None, ratio=None):
    n = steps
    half = n // 2
    return {
        "arm": name, "mode": mode, "cell": "D", "seed": 0, "status": "DONE",
        "explore": {"mode": mode, "entropy_target": target,
                    "sample_temperature": 1.5 if mode == "temperature" else 1.0,
                    "knobs": knobs if knobs is not None else
                    {"entropy_rel": 0.85, "entropy_min": 0.3}},
        "config": {"group_size": group},
        "preflight_exploration": {"policy_entropy": 0.02},
        "history": {
            "step": list(range(n)),
            "solve_rate": [solve[0]] * half + [solve[1]] * (n - half),
            "zero_advantage_group_fraction": [1.0] * half
                                             + [zero_end] * (n - half),
            "policy_entropy": [0.02] * n, "entropy_coef": [0.01] * n,
            "grad_norm": [0.5] * n, "mine_step_rate": [1.0] * n,
            "within_group_return_sd": [2.0] * n,
            "seconds_per_step": sps, "log_importance_ratio_mean": ratio,
        },
    }


def test_the_explore_scorer_calls_a_dead_arm_dead_and_picks_a_live_one():
    """The verdicts and the pick, on the four shapes the trial can produce."""
    s = _explore_scorer()
    rows = [s.read_arm(a) for a in (
        _fake_arm("01_thermostat_g6", "thermostat_fixed_target", zero_end=1.0),
        _fake_arm("02_relative_g6", "relative", zero_end=0.2),
        _fake_arm("03_fixed_g6", "fixed", zero_end=0.3, solve=(1.0, 0.5),
                  knobs={"entropy_coef_fixed": 0.1}),
        _fake_arm("04_temperature_g6", "temperature", zero_end=0.1, sps=40.0,
                  ratio=-3.5, knobs={"sample_temperature": 1.5}),
        _fake_arm("05_relative_g12", "relative", zero_end=0.1, sps=70.0,
                  group=12),
    )]
    by = {r["arm"]: r for r in rows}
    # the control is DEAD: the move term is still multiplying by zero
    assert by["01_thermostat_g6"]["verdict"] == "DEAD"
    assert by["02_relative_g6"]["verdict"] == "ALIVE"
    # a live gradient bought by destroying the policy is not a recipe
    assert by["03_fixed_g6"]["verdict"] == "RANDOMISED"
    assert by["03_fixed_g6"]["solve_drop"] == 0.5
    assert by["04_temperature_g6"]["verdict"] == "ALIVE"
    assert by["05_relative_g12"]["verdict"] == "ALIVE"
    # start/end are WINDOW means, so the first half's 1.0 does not decide it
    assert by["02_relative_g6"]["zero_adv_start"] == 1.0
    assert by["02_relative_g6"]["zero_adv_end"] == 0.2

    # The pick: solve ties at 1.0, so the cheapest step wins, and `relative`
    # takes the tie against `temperature` at equal cost.
    pick = s.pick_arm(rows)
    assert pick["arm"] == "02_relative_g6"
    line = s.env_line(pick, cells="A,C")
    assert "MZ01_EXPLORE=relative" in line
    assert "MZ01_ENTROPY_MIN=0.3" in line
    # PINNED, not derived: a second lease re-deriving the target would be a
    # different recipe and every resume check would (correctly) refuse it.
    assert "MZ01_ENTROPY_TARGET=0.3" in line
    assert "MZ01_GROUP=6" in line and "MZ01_CELLS=A,C" in line
    assert "MZ01_PREFLIGHT_SOFT=1" in line

    # A trial where nothing came alive says so, and does not pick a corpse.
    dead_only = [s.read_arm(_fake_arm("01_thermostat_g6",
                                      "thermostat_fixed_target", zero_end=1.0))]
    assert s.pick_arm(dead_only) is None
    assert any("PICK: NONE" in ln for ln in s.table(dead_only, None))
    # The off-policy arm's distance is REPORTED in the table, in words.
    assert any("NOT applied" in ln for ln in s.table(rows, pick))
    # A failed arm is not silently an ALIVE one.
    assert s.read_arm({"arm": "x", "status": "FAILED",
                       "history": None})["verdict"] == "FAILED"


def test_the_explore_scorer_runs_under_a_bare_python3(tmp_path):
    """Stdlib only, `python3 -S`, on the box, beside the trial."""
    s = _explore_scorer()
    (tmp_path / "02_relative_g6.json").write_text(json.dumps(
        _fake_arm("02_relative_g6", "relative", zero_end=0.2)))
    out = subprocess.run(
        [sys.executable, "-S", str(_ROOT / "scripts" / "score_mz01_explore.py"),
         str(tmp_path)],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert "02_relative_g6" in out.stdout and "ALIVE" in out.stdout
    assert "MZ01_EXPLORE=relative" in out.stdout
    assert s.ZERO_ADVANTAGE_BAR == maze_orgb.ZERO_ADVANTAGE_BAR


def test_the_trial_runs_every_mode_and_names_the_arm_the_scorer_reads():
    """The trial's arms are the four modes plus the GROUP=12 scaling check, and
    the names it writes are the names its own summary picks by."""
    trial = _trial_module()
    modes = {a["mode"] for a in trial.ARMS}
    assert modes == set(maze_orgb.EXPLORE_MODES)
    assert trial.ARMS[0]["mode"] == maze_orgb.DEFAULT_EXPLORE_MODE, \
        "the CONTROL must run first: a box that dies early has to have it"
    assert [a["group"] for a in trial.ARMS].count(12) == 1
    assert trial.TRIAL_CELL == "D" and trial.TRIAL_CELL_2 == "A"
    # 60 since the tenth pass (the arithmetic is in the module docstring: the
    # cell-A arms plus their held-out reads inside the ~4 h a lease can be
    # trusted for). Every launch pins MZ01_TRIAL_STEPS explicitly anyway.
    assert trial.TRIAL_STEPS == 60
    # It reads its own arms with the scorer, so the pick and the table cannot
    # disagree, and it never writes into the organism tree.
    assert trial._scorer().ZERO_ADVANTAGE_BAR == maze_orgb.ZERO_ADVANTAGE_BAR
    assert "explore_trial" in trial.OUT and "results/orgb" not in trial.OUT
    assert trial.STATUS.endswith("mz01_explore_status.txt")


def test_the_dose_response_arms_differ_from_epsilon_010_in_exactly_one_thing():
    """Eleventh pass: `epsilon_005` is the LOW dose and `onpolicy_g6` the CONTROL.

    The tenth pass gave cell A four arms and every one of them moves the
    sampler, so a live gradient under any of them could not be told apart from
    cell A's priced mine simply giving the ON-POLICY recipe the gradient cell D
    could never show. The control closes that, and it is only a control if it
    is `epsilon_010` with the epsilon draw removed and NOTHING else moved.

    The three cell-A arms this brief runs must therefore agree on the group, on
    the draw temperature (1.0 -- no hot sampling anywhere in the contrast) and
    on the thermostat being off with the coefficient held; and they must differ
    only in `plan["epsilon"]`, at 0.10 / 0.05 / 0.0.
    """
    trial = _trial_module()
    by_name = {a["name"]: a for a in trial.ARMS}
    brief = ["epsilon_010", "epsilon_005", "onpolicy_g6"]
    assert all(n in by_name for n in brief)
    plans = {}
    for n in brief:
        arm = by_name[n]
        assert arm["group"] == 6, f"{n} must run at the contrast's GROUP"
        arm_eps = float(arm.get("epsilon")
                        or maze_orgb.EXPLORE_EPSILON_RATE)
        plans[n] = maze_orgb.exploration_plan(
            arm["mode"], measured_entropy=0.02, epsilon=arm_eps,
            thermostat_off=bool(arm.get("thermostat_off")))
    # The ONE thing that differs.
    assert [plans[n]["epsilon"] for n in brief] == [0.10, 0.05, 0.0]
    # ... and everything else that must not.
    for n in brief:
        p = plans[n]
        assert p["entropy_target"] is None and p["entropy_adaptive"] is False, \
            f"{n} must run no thermostat: the contrast is the epsilon draw"
        assert p["entropy_coef"] == maze_orgb.ENTROPY_COEF_FIXED
        assert p["sample_temperature"] == p["temperature"] == 1.0, \
            f"{n} must not also draw hot -- that would be two changes"
    # `onpolicy_g6` is off-policy in NEITHER currency; that is what makes it the
    # baseline the other two are read against.
    assert plans["onpolicy_g6"]["off_policy"] is False
    assert plans["onpolicy_g6"]["off_policy_decision_fraction"] == 0.0
    assert plans["epsilon_010"]["off_policy_decision_fraction"] == 0.10
    assert plans["epsilon_005"]["off_policy_decision_fraction"] == 0.05
    # They are a NAMED SELECTION, not a new default: the tenth pass's committed
    # four-arm cell-A command line has to keep meaning what it meant.
    assert list(trial.DEFAULT_ARMS if trial.TRIAL_CELL != "D"
                else trial.ARM_NAMES[5:9]) == [
        "temperature_t15", "temperature_t20", "epsilon_010", "epsilon_020"]


# ---------------------------------------------------------------------------
# critic pass 2, 2026-08-29: exploration (R2)
# ---------------------------------------------------------------------------


def test_the_target_band_is_reported_and_never_refuses():
    """THE AMENDMENT. The pre-flight has ONE refusing criterion.

    `entropy_target_over_measured` was a second gate for one pass and it could
    not be one: `relative`'s floor of 0.3 against a near-deterministic policy's
    0.02 is a 15x ratio BY CONSTRUCTION, so the criterion refused exactly the
    arm it was written to license and clearing it took a blanket
    MZ01_PREFLIGHT_SOFT=1 -- which also waves through the criterion that means
    something. It is now informational: computed, printed, in the manifest,
    never a refusal.
    """
    def wandering(prompt, generator):
        """A policy with real move variance, so criterion (b) PASSES and
        anything that refuses here can only be the band."""
        j = int(torch.randint(len(maze2.MOVE_WORDS), (1,), generator=generator))
        return maze2.MOVE_WORDS[j]

    model, tok = _model()
    old = maze_orgb._sample_replies
    maze_orgb._sample_replies = _ScriptedSampler(wandering)
    try:
        out = maze_orgb.preflight_exploration_check(
            model, tok, _mazes(8), generator=torch.Generator().manual_seed(0),
            group_size=4, fmt=ASCII, device=torch.device("cpu"), chunk_size=4,
            max_new_tokens=40, explore_mode="relative")
    finally:
        maze_orgb._sample_replies = old
    remove_lora(model)
    band = out["criteria"][1]
    assert band["name"] == "entropy_target_over_measured"
    assert band["refuses"] is False
    assert out["criteria"][0]["refuses"] is True
    # The reading is still taken, still printed and still in the payload...
    assert out["entropy_target_over_measured"] is not None
    assert any("INFORMATIONAL" in line for line in out["lines"])
    assert "INFORMATIONAL" in band["note"]
    # ... and out of band or not, it never appears as a refusal.
    assert out["passed"] and out["hard_passed"]
    assert out["soft_failures"] == []
    if not band["ok"]:
        assert "PASS" in out["lines"][-1]


def test_the_preflight_rolls_out_under_the_arms_own_sampler():
    """The gate's one refusing criterion is a property of the SAMPLER, so it has
    to be measured under the sampler the arm will use.

    Drawing it at the SCORING temperature made the gate report the control's
    degeneracy for every arm -- including `temperature`, whose entire mechanism
    is the draw, and which would then have been refused for the state it exists
    to fix. The entropy is still a `_rescore` read at the scoring temperature,
    because that is the quantity the controller regulates.
    """
    seen = []
    inner = _ScriptedSampler(_warm_started_policy)

    def spy(model, tok, texts, *, temperature, aux=None,
            score_temperature=None, **kw):
        seen.append(temperature)
        out = inner(model, tok, texts, temperature=temperature, **kw)
        if aux is not None:
            # A stand-in for the real per-reply log ratio: the point of the test
            # is WHICH temperature the draw happened at, not its value.
            aux.extend([-1.25] * len(texts))
        return out

    model, tok = _model()
    old = maze_orgb._sample_replies
    maze_orgb._sample_replies = spy
    try:
        hot = maze_orgb.preflight_exploration_check(
            model, tok, _mazes(4), generator=torch.Generator().manual_seed(0),
            group_size=3, fmt=ASCII, device=torch.device("cpu"), chunk_size=4,
            max_new_tokens=40, explore_mode="temperature", temperature=1.0,
            sample_temperature=1.5, soft=True)
        drawn_hot = list(seen)
        seen.clear()
        cold = maze_orgb.preflight_exploration_check(
            model, tok, _mazes(4), generator=torch.Generator().manual_seed(0),
            group_size=3, fmt=ASCII, device=torch.device("cpu"), chunk_size=4,
            max_new_tokens=40, explore_mode="relative", temperature=1.0,
            sample_temperature=1.5, soft=True)
    finally:
        maze_orgb._sample_replies = old
    remove_lora(model)
    assert drawn_hot and set(drawn_hot) == {1.5}, drawn_hot
    assert hot["sample_temperature"] == 1.5 and hot["score_temperature"] == 1.0
    assert hot["off_policy"] and hot["log_importance_ratio"] is not None
    assert any("draw T=1.5, score T=1" in line for line in hot["lines"])
    # Every other mode draws at the temperature it scores at, and MZ01_SAMPLE_T
    # set on the side may not quietly change that.
    assert seen and set(seen) == {1.0}, seen
    assert not cold["off_policy"] and cold["log_importance_ratio"] is None
    assert cold["sample_temperature"] == cold["score_temperature"] == 1.0
    # ONE implementation of "which temperature does this mode draw at".
    for mode in maze_orgb.EXPLORE_MODES:
        plan = maze_orgb.exploration_plan(mode, measured_entropy=0.02,
                                          sample_temperature=1.5)
        assert plan["sample_temperature"] == maze_orgb.plan_sample_temperature(
            mode, temperature=1.0, sample_temperature=1.5)


def test_a_mode_name_and_its_numbers_may_not_disagree():
    """`relearn_read` takes a recipe dict whose keys are all optional, so a
    relearn asked for `fixed` while OMITTING `entropy_target` would have run 50
    steps of the pre-registered thermostat at 0.7 -- with `explore_mode` saying
    `fixed` in the trace it wrote. The trainer refuses the combination now, so
    every caller is covered and not only the one that happens to pass a plan.
    """
    with pytest.raises(ValueError) as err:
        _explore_train(explore_mode="fixed", entropy_target=0.7,
                       entropy_coef=0.1)
    assert "no thermostat" in str(err.value).lower() or \
        "NO thermostat" in str(err.value)
    with pytest.raises(ValueError) as err2:
        _explore_train(explore_mode="temperature", sample_temperature=None)
    assert "sample_temperature" in str(err2.value)
    with pytest.raises(ValueError) as err3:
        _explore_train(explore_mode="relative", entropy_target=0.3,
                       sample_temperature=1.5)
    assert "temperature" in str(err3.value)
    # TENTH PASS, the same rule one knob over: `epsilon` IS the replacement
    # rate, so a mode name and a rate that disagree are a run whose recorded arm
    # is not the arm that ran -- in either direction.
    with pytest.raises(ValueError) as err4:
        _explore_train(explore_mode="epsilon", entropy_target=None,
                       entropy_coef=0.1, epsilon=0.0)
    assert "epsilon" in str(err4.value)
    with pytest.raises(ValueError) as err5:
        _explore_train(explore_mode="fixed", entropy_target=None,
                       entropy_coef=0.1, epsilon=0.2)
    assert "epsilon" in str(err5.value)
    # ... and a THERMOSTAT mode with no target is the same hole from the other
    # side: `relearn_read` omitting `entropy_target` would have run the
    # controller off while the trace said `relative`. It is legal only when the
    # plan says so out loud in its knobs.
    with pytest.raises(ValueError) as err6:
        _explore_train(explore_mode="relative", entropy_target=None,
                       entropy_coef=0.1)
    assert "THERMOSTAT" in str(err6.value)
    off = maze_orgb.exploration_plan("temperature", measured_entropy=0.02,
                                     thermostat_off=True)
    assert off["entropy_target"] is None and off["knobs"]["thermostat_off"]
    hist_off = _explore_train(
        explore_mode=off["mode"], entropy_target=off["entropy_target"],
        entropy_coef=off["entropy_coef"], explore_knobs=off["knobs"],
        sample_temperature=off["sample_temperature"], steps=1)
    assert hist_off["config"]["explore"]["entropy_adaptive"] is False

    # The plans every real caller passes are all legal, which is the point.
    for mode in maze_orgb.EXPLORE_MODES:
        plan = maze_orgb.exploration_plan(mode, measured_entropy=0.02)
        hist = _explore_train(
            explore_mode=plan["mode"], entropy_target=plan["entropy_target"],
            entropy_coef=plan["entropy_coef"],
            sample_temperature=plan["sample_temperature"],
            epsilon=plan["epsilon"], move_word_ids=_TOY_MOVE_IDS, steps=1)
        assert hist["config"]["explore"]["mode"] == mode


def test_the_off_policy_arm_does_not_claim_the_ratio_is_one():
    """The `temperature` arm's honesty, in the words the manifest carries.

    "GRPO's ratio is identically 1 on the first inner epoch" is true because the
    behaviour policy IS the scored policy; drawing at 1.5 and scoring at 1.0
    breaks that premise, and the ratio is then NOT 1 -- which is why the code
    measures it. The argument that survives is narrower and the docstring now
    says so, so the omission is a stated bias rather than a claim of no bias.
    """
    doc = maze_orgb.train_orgb_cell.__doc__
    assert "IT IS NOT THE \"identically 1\" ARGUMENT" in doc
    assert "off-policy on purpose" in maze_orgb._sample_replies.__doc__
    # And the recorded number is not zero, i.e. the ratio really is not 1.
    model, tok = _model(seed=2)
    hist = maze_orgb.train_orgb_cell(
        model, tok, cell="D", seed=0, steps=2, batch_size=1, group_size=2,
        lr=0.1, chunk_size=4, fmt=ASCII, log_every=0, max_new_tokens=4,
        train_pool=[maze2.from_layout(MINEFIELD, seed=201)],
        explore_mode="temperature", sample_temperature=1.5)
    remove_lora(model)
    assert hist["log_importance_ratio_mean"] != 0.0
    assert "NOT APPLIED" in hist["config"]["explore"]["importance_weight"]


def test_the_follow_up_cell_is_read_but_never_picked():
    """The trial runs the picked arm AGAIN on cell A and writes it beside the
    arms. Its `arm` field used to be the PICKED ARM'S name, so a human pointing
    the scorer at the directory got two rows with one name and the pick could
    come back the CELL A run -- whose solve rate and zero-advantage fraction are
    not what the pick rule is written about."""
    s = _explore_scorer()
    d_cell = _fake_arm("03_fixed_g6", "fixed", zero_end=0.3, sps=40.0,
                       knobs={"entropy_coef_fixed": 0.1})
    a_cell = dict(_fake_arm("A_03_fixed_g6", "fixed", zero_end=0.1, sps=30.0,
                            knobs={"entropy_coef_fixed": 0.1}),
                  arm_base="03_fixed_g6", follow_up=True, cell="A")
    # A cell-A row is read against its held-out block and the reference. With
    # no held-out block it is NO-MEASUREMENT -- what a follow-up killed
    # mid-run reads as -- and not a verdict the pick rule can act on.
    assert s.read_arm(a_cell)["verdict"] == "NO-MEASUREMENT"
    a_cell["eval"] = {"n_mazes": 128, "solve_rate": 1.0, "mine_step_rate": 1.0}
    ref = {"eval_solve_rate": 1.0, "eval_mine_step_rate": 1.0}
    rows = [s.read_arm(d_cell, reference=ref), s.read_arm(a_cell, reference=ref)]
    assert rows[1]["follow_up"] and rows[1]["arm_base"] == "03_fixed_g6"
    # The cell A row is CHEAPER and no less alive, so without the flag it wins.
    assert rows[1]["verdict"] == "ALIVE" and rows[1]["seconds_per_step"] < \
        rows[0]["seconds_per_step"]
    assert s.pick_arm(rows)["arm"] == "03_fixed_g6"
    assert any("FOLLOW-UP" in ln for ln in s.table(rows, s.pick_arm(rows)))
    # The trial names the FILE, not the arm definition, and marks it.
    trial = _trial_module()
    pay = trial.arm_payload(
        {"name": "03_fixed_g6", "mode": "fixed", "note": ""}, cell="A", seed=0,
        plan={}, preflight={"soft_failures": ["zero_advantage_group_fraction"]},
        history={}, status="DONE", cfg={}, started_at="", model_id="m",
        name="A_03_fixed_g6", follow_up=True)
    assert pay["arm"] == "A_03_fixed_g6" and pay["arm_base"] == "03_fixed_g6"
    assert pay["follow_up"] and pay["deviations"], \
        "the soft pre-flight is a deviation and every arm's file must say so"


def test_a_dead_arm_whose_controller_never_moved_says_so():
    """A 40-step trial is structurally unkind to the arms whose mechanism is the
    THERMOSTAT and kind to the ones that act from step 0.

    `log_coef += 0.05 * (target - H)`, so at a target of 0.3 against a measured
    0.02 the coefficient multiplies by e^0.014 a step: 0.01 at step 0 and 0.0175
    at step 40, a sixth of the 0.1 the `fixed` arm STARTS at. Such an arm can
    read DEAD because the horizon was short and not because the mode fails, and
    the table has to say which -- the answer being more steps, never a lower bar.
    """
    s = _explore_scorer()
    coef = [0.01 * math.exp(0.05 * (0.3 - 0.02) * i) for i in range(40)]
    assert 0.017 < coef[-1] < 0.018, coef[-1]
    slow = _fake_arm("02_relative_g6", "relative", zero_end=1.0, steps=40)
    slow["history"]["entropy_coef"] = coef
    row = s.read_arm(slow)
    assert row["verdict"] == "DEAD" and row["slow_controller"]
    assert any("DEAD AT THIS HORIZON" in ln for ln in s.table([row], None))
    # A `fixed` arm that read DEAD is genuinely dead: its coefficient was at the
    # reference from step 0 and did not move.
    still = _fake_arm("03_fixed_g6", "fixed", zero_end=1.0, steps=40,
                      knobs={"entropy_coef_fixed": 0.1})
    still["history"]["entropy_coef"] = [0.1] * 40
    assert not s.read_arm(still)["slow_controller"]
    assert s.COEF_REFERENCE == maze_orgb.ENTROPY_COEF_FIXED


def test_the_soft_flag_is_emitted_only_when_the_gate_would_refuse():
    """MZ01_PREFLIGHT_SOFT=1 is a RECORDED DEVIATION in the manifest, so an env
    line that carries it when the gate would have passed records a deviation
    that did not happen. Now that the pre-flight draws at the arm's own
    temperature, `temperature` can clear the gate where the thermostat arms
    cannot."""
    s = _explore_scorer()
    passed = _fake_arm("04_temperature_g6", "temperature", zero_end=0.1,
                       knobs={"sample_temperature": 1.5})
    passed["preflight_exploration"] = {"policy_entropy": 0.02,
                                       "hard_passed": True}
    line = s.env_line(s.read_arm(passed))
    assert "MZ01_SAMPLE_T=1.5" in line and "MZ01_PREFLIGHT_SOFT" not in line
    refused = _fake_arm("02_relative_g6", "relative", zero_end=0.2)
    refused["preflight_exploration"] = {"policy_entropy": 0.02,
                                        "hard_passed": False}
    assert "MZ01_PREFLIGHT_SOFT=1" in s.env_line(s.read_arm(refused))
    # No verdict recorded at all -> emit it: a stopped run is the cheaper error.
    unknown = _fake_arm("02_relative_g6", "relative", zero_end=0.2)
    unknown.pop("preflight_exploration")
    assert "MZ01_PREFLIGHT_SOFT=1" in s.env_line(s.read_arm(unknown))
