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
                 generator, batch_size, device):
        self.prompts.extend(texts)
        return [_ids(self.fn(t, generator)[:max_new_tokens]) for t in texts]


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
