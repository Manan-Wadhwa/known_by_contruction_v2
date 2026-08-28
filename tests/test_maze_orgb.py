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
    # "left" is a bare move: it has move tokens and NO remark tokens.
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
                rows.append({"detour_cost": d, "m": m,
                             "avoided_all_mines": paid})
    reg = maze_orgb.dm_regression(rows)
    assert reg["d"] < 0 < reg["m"]
    assert reg["ratio_m_over_d"] == pytest.approx(5.0, abs=1.5)
    # A design with no variation in d or m is NOT-MEASURED, not a fit.
    flat = [{"detour_cost": 4, "m": 1, "avoided_all_mines": True}] * 8
    assert maze_orgb.dm_regression(flat)["note"].startswith("NOT-MEASURED")


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
    a = [{"layout": f"L{i}", "mine_steps": 1, "moves": ["up"]} for i in range(8)]
    b = [{"layout": f"L{i}", "mine_steps": 1, "moves": ["up"]} for i in range(8)]
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


def test_the_run_produces_every_row_key_and_one_shared_warm_start(tmp_path):
    """The whole shape, end to end. Every number it produces is garbage -- the
    model is a four-parameter transformer -- and every CONTRACT is real."""
    import run_orgb
    payload = _run_payload(tmp_path)
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


def test_the_scorers_arithmetic_is_the_modules_arithmetic(tmp_path):
    """One statistic, two authors: if they drift, the number in the paper is
    computed one way in the run and another way in the re-score."""
    s = _scorer()
    payload = _run_payload(tmp_path, seeds=(0,))
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


def test_the_scorer_runs_under_a_bare_python3(tmp_path):
    """`-S` drops site-packages, so torch is not importable: if this passes, the
    scorer needs nothing but the standard library."""
    payload = _run_payload(tmp_path, seeds=(0,))
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
                   "regression avoided ~ d + m"):
        assert needle in out, needle
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
