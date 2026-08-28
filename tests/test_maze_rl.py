"""MZ01's episode-return RL arm, on the toy LM.

The claims tested here are properties of the OBJECTIVE and of the environment
loop, not of the model, so a four-parameter transformer demonstrates them
exactly as a 14B one would, on CPU, in seconds. That is `conftest.TinyLM`'s
whole rationale and it is why these tests exist at all: the first pass's plan
("generalise `train_org_a` to episodes; the records need no reshaping") was
false in both halves, and it was false in a file nothing executed.

What is checked:
  * one step runs end to end and MOVES THE ADAPTER (and only the adapter);
  * the advantages sum to zero inside every maze's group, which is the property
    that makes the Dr.GRPO baseline unbiased -- and the property a batch-wide
    baseline would quietly break while looking identical in a log;
  * a scripted always-solving policy produces the return the SPEC's scoring
    says it should, so the number the trainer optimises is the number the world
    pays out.
"""

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from conftest import CharTokenizer, tiny_model  # noqa: E402
from calibration import maze2, maze_rl  # noqa: E402
from calibration.capture import MOVE_WORDS  # noqa: E402
from calibration.lora import (  # noqa: E402
    inject_lora,
    lora_parameters,
    remove_lora,
)
from calibration.maze2 import (  # noqa: E402
    SCORE_GOAL,
    SCORE_MINE,
    SCORE_STEP,
    Glyphs,
    from_layout,
    path_moves,
    shortest_path,
)

# The same hand-written world the roll-out tests use: start in the top-left
# corner, one mine, one goal, and "up"/"left" blocked forever from the start.
CORRIDOR = """
S...
.##.
.#*.
...G
"""

# AMENDED after critic 2. CORRIDOR's mine is off both of its shortest paths, so
# the "always-solving" test above never landed on one and asserted the mine term
# of the return against `e["mine_steps"] == 0`. Nothing in this file exercised a
# multi-turn episode WITH A LANDING -- which is the one event the whole episode
# arm exists to price. Here the unique shortest route crosses the mine (5 steps,
# score 0.0) and the mine-free route is 7 steps (score +6), so the two branches
# have different returns and the advantage between them is the quantity DEC-1
# claims the episode arm can see and the immediate arm cannot.
MINEFIELD = """
S...
.###
.*.G
....
"""


def _model():
    model = tiny_model(vocab_size=CharTokenizer.vocab_size)
    inject_lora(model, target_modules=("q_proj", "v_proj"), r=2, alpha=4)
    return model, CharTokenizer()


def _mazes(count):
    return [from_layout(CORRIDOR, seed=200 + i) for i in range(count)]


class _ScriptedLogits:
    """Stands in for `maze_rl._move_logits`, the way the roll-out tests stand in
    for `maze2._greedy`. Everything else -- the lockstep loop, the cap, the
    re-render, the per-turn word order, the record -- runs for real."""

    def __init__(self, fn, glyphs):
        self.fn = fn
        self.glyphs = glyphs
        self.texts = []

    def __call__(self, model, tok, texts, cols, device):
        self.texts.extend(texts)
        rows = []
        for t in texts:
            want = self.fn(t, self.glyphs)
            rows.append([12.0 if w == want else -12.0 for w in MOVE_WORDS])
        logits = torch.tensor(rows, dtype=torch.float32, device=device)
        mass = torch.zeros(len(texts), dtype=torch.float32, device=device)
        return logits, mass


def _oracle(prompt, glyphs):
    """The shortest-path move, read out of the prompt the loop just built."""
    grid = prompt.split("Grid:\n", 1)[1].split("\n\n", 1)[0]
    rows, agent = maze2.parse_render(grid, glyphs)
    n = len(rows)
    goal = [(r, c) for r in range(n) for c in range(n) if rows[r][c] == "G"][0]
    maze = from_layout("\n".join(maze2.maze_to_layout(rows, agent, goal)),
                       labels=False)
    return path_moves(shortest_path(maze))[0]


def test_one_step_runs_and_moves_the_adapter_and_nothing_else():
    """The contract: LoRA is injected, and the adapters are the only tensors
    that change. A stray unfrozen base parameter corrupts the resident model
    every other experiment shares, and does it silently."""
    model, tok = _model()
    before = [p.detach().clone() for p in lora_parameters(model)]
    base = {n: p.detach().clone() for n, p in model.named_parameters()
            if "lora_" not in n}

    hist = maze_rl.train_maze_a(
        model, tok, seed=0, steps=1, batch_size=2, group_size=4,
        train_pool=_mazes(4), eval_set=[], log_every=0, lr=0.5)

    after = [p.detach().clone() for p in lora_parameters(model)]
    assert any(not torch.equal(a, b) for a, b in zip(before, after)), \
        "one step left the adapter exactly where it was"
    for n, p in model.named_parameters():
        if "lora_" not in n:
            assert torch.equal(p.detach(), base[n]), f"{n} moved"

    assert hist["config"]["reward_mode"] == "episode"
    assert hist["config"]["algorithm"].startswith("Dr.GRPO")
    assert hist["n_trainable"] == sum(p.numel() for p in lora_parameters(model))
    for key in ("step", "loss", "mean_return", "solve_rate", "mine_step_rate",
                "episode_length", "policy_entropy", "grad_norm", "move_mass",
                "entropy_coef", "move_dist", "step_seconds"):
        assert len(hist[key]) == 1, key
    assert hist["n_decisions"][0] == hist["episode_length"][0] * 8
    remove_lora(model)


def test_the_trainer_refuses_a_model_with_no_adapters():
    """No adapters produces a flat curve indistinguishable from a real null."""
    model = tiny_model(vocab_size=CharTokenizer.vocab_size)
    with pytest.raises(RuntimeError, match="no LoRA adapters"):
        maze_rl.train_maze_a(model, CharTokenizer(), steps=1, batch_size=1,
                             group_size=2, train_pool=_mazes(1), eval_set=[],
                             log_every=0)


def test_advantages_sum_to_zero_inside_every_group_and_not_across_them():
    """Dr.GRPO's baseline is the MAZE's group mean, and dividing by the group
    standard deviation is exactly the line that is not here.

    A batch-wide baseline would look almost identical in a log and would reward
    "was handed an easy maze" instead of "played this maze well", so the group
    structure is asserted rather than assumed.
    """
    returns = [10.0, -2.0, 6.0, 100.0, 100.0, 100.0]
    groups = [0, 0, 0, 1, 1, 1]
    adv = maze_rl.group_advantages(returns, groups)
    for g in set(groups):
        rows = [a for a, gg in zip(adv, groups) if gg == g]
        assert sum(rows) == pytest.approx(0.0, abs=1e-9)
    # The second group is uniform, so it carries no signal at all -- which is
    # the collapse `zero_signal_steps` counts, seen here in the advantages.
    assert adv[3:] == [0.0, 0.0, 0.0]
    # Centred, NOT standardised: a group with a wide spread keeps it.
    assert max(adv[:3]) - min(adv[:3]) == pytest.approx(12.0)
    # And the mean over the whole batch is only zero because both groups are.
    assert sum(adv) == pytest.approx(0.0, abs=1e-9)

    # The same property, on real sampled episodes rather than on a list.
    model, tok = _model()
    episodes, decisions = maze_rl.sample_episodes(
        model, tok, _mazes(3), group_size=4,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"))
    assert len(episodes) == 12 and {e["group"] for e in episodes} == {0, 1, 2}
    adv = maze_rl.group_advantages([e["return"] for e in episodes],
                                   [e["group"] for e in episodes])
    for g in (0, 1, 2):
        rows = [a for a, e in zip(adv, episodes) if e["group"] == g]
        assert len(rows) == 4 and sum(rows) == pytest.approx(0.0, abs=1e-6)
    # Every decision belongs to exactly one episode, and every episode's rows
    # are the decisions it took -- the loss indexes `decisions` by those rows.
    assert sorted(r for e in episodes for r in e["decisions"]) == \
        list(range(len(decisions)))
    for e in episodes:
        assert len(e["decisions"]) == e["steps"] == len(e["moves"])
    remove_lora(model)


def test_a_scripted_always_solving_episode_returns_what_the_scoring_says():
    """The number the trainer optimises has to be the number the world pays
    out: -2 a step, -10 a mine, +20 the goal, and the episode ends there."""
    model, tok = _model()
    g = Glyphs()
    maze = from_layout(CORRIDOR, seed=1)
    shortest = maze.labels["shortest_len"]
    scripted = _ScriptedLogits(_oracle, g)
    old = maze_rl._move_logits
    maze_rl._move_logits = scripted
    try:
        episodes, decisions = maze_rl.sample_episodes(
            model, tok, [maze], group_size=3, glyphs=g,
            generator=torch.Generator().manual_seed(0),
            device=torch.device("cpu"))
    finally:
        maze_rl._move_logits = old

    for e in episodes:
        assert e["solved"] and not e["capped"]
        assert e["steps"] == shortest
        assert e["landings"][-1] == "goal"
        assert e["return"] == SCORE_GOAL + SCORE_STEP * shortest \
            + SCORE_MINE * e["mine_steps"]
        assert e["return"] == maze_rl.episode_return(e["landings"])
        # And the same number the scorer would compute off a roll-out record.
        assert e["return"] == maze2.score_landings(e["landings"])
    # Every group member played the same episode, so the advantages vanish.
    adv = maze_rl.group_advantages([e["return"] for e in episodes],
                                   [e["group"] for e in episodes])
    assert adv == [0.0, 0.0, 0.0]
    assert len(decisions) == 3 * shortest
    # The prompt the loop built carries the CURRENT position, re-rendered.
    assert len({t for t in scripted.texts}) > 1, "the grid never changed"
    remove_lora(model)


def test_a_blocked_move_costs_a_step_and_moves_nothing():
    """SPEC: invalid move -2, position unchanged, episode continues to the cap.
    That is also SPEC A4's point: it costs the same as walking."""
    model, tok = _model()
    g = Glyphs()
    maze = from_layout(CORRIDOR, seed=2)
    scripted = _ScriptedLogits(lambda p, gl: "up", g)   # blocked from (0,0)
    old = maze_rl._move_logits
    maze_rl._move_logits = scripted
    try:
        episodes, _decisions = maze_rl.sample_episodes(
            model, tok, [maze], group_size=2, glyphs=g,
            generator=torch.Generator().manual_seed(0),
            device=torch.device("cpu"))
    finally:
        maze_rl._move_logits = old
    for e in episodes:
        assert not e["solved"] and e["capped"]
        assert e["steps"] == maze.labels["cap"] == e["invalid"]
        assert set(e["landings"]) == {"blocked"}
        assert e["return"] == SCORE_STEP * maze.labels["cap"]
    assert maze_rl.immediate_reward("blocked") == maze_rl.immediate_reward("floor")


def test_the_immediate_arm_is_kept_and_is_a_different_objective():
    """SPEC 3A, the fallback. Kept because it was pre-committed and because the
    comparison IS the evidence for A4; its credit is per decision, not per
    episode."""
    assert maze_rl.REWARD_MODES == ("episode", "immediate")
    assert maze_rl.immediate_reward("mine") == SCORE_STEP + SCORE_MINE == -12.0
    assert maze_rl.immediate_reward("goal") == SCORE_STEP + SCORE_GOAL == 18.0
    assert maze_rl.immediate_reward("floor") == -2.0

    model, tok = _model()
    hist = maze_rl.train_maze_a(
        model, tok, seed=1, steps=1, batch_size=2, group_size=4,
        reward_mode="immediate", train_pool=_mazes(4), eval_set=[],
        log_every=0, lr=0.5)
    assert hist["config"]["reward_mode"] == "immediate"
    assert "immediate per-decision" in hist["config"]["formulation"]
    remove_lora(model)

    with pytest.raises(ValueError, match="reward_mode"):
        maze_rl.train_maze_a(tiny_model(), CharTokenizer(), reward_mode="nope")
    with pytest.raises(ValueError, match="group_size"):
        maze_rl.train_maze_a(tiny_model(), CharTokenizer(), group_size=1)


def test_the_eval_hook_reports_detour_paid_by_bin_on_a_held_out_set():
    """The bin table is the instrument the world exists to read. Having its
    TRAJECTORY rather than only its endpoint is the difference between "the
    threshold appeared" and "it was there at step 50 and then collapsed"."""
    model, tok = _model()
    hist = maze_rl.train_maze_a(
        model, tok, seed=2, steps=2, batch_size=1, group_size=2,
        train_pool=_mazes(2), eval_set=_mazes(2), eval_every=1,
        eval_batch_size=2, log_every=0)
    assert len(hist["eval"]) == 2
    for block in hist["eval"]:
        assert "step" in block and block["by_detour_cost"]
        for row in block["by_detour_cost"].values():
            for key in ("detour_paid", "detour_paid_solved",
                        "detour_unsolved_share", "solve_rate", "n_episodes"):
                assert key in row, key
        assert set(block["overall"]) >= {"solve_rate", "detour_paid_solved",
                                         "on_mine_state_rate"}
    remove_lora(model)


def test_the_entropy_controller_and_move_mass_term_are_the_ones_from_rl():
    """Both terms exist here for reasons measured in the old world: a FIXED
    entropy coefficient has a horizon past which the policy collapses anyway
    (E7), and six of eight committed ORG-A organisms had left the move
    vocabulary while three of them passed the functional bar."""
    assert maze_rl.MAX_ENTROPY == pytest.approx(1.3862943611)
    model, tok = _model()
    hist = maze_rl.train_maze_a(
        model, tok, seed=3, steps=2, batch_size=1, group_size=2,
        train_pool=_mazes(2), eval_set=[], log_every=0,
        entropy_coef=0.01, entropy_target=0.9 * maze_rl.MAX_ENTROPY,
        move_mass_coef=0.05, lr=0.1)
    assert hist["config"]["entropy_adaptive"] is True
    assert hist["config"]["move_mass_coef"] == 0.05
    # The coefficient recorded at step i is the one that PRODUCED the entropy at
    # step i, so it can only differ from step 1 onwards.
    assert hist["entropy_coef"][0] == 0.01
    assert all(0.0 <= h <= maze_rl.MAX_ENTROPY + 1e-6 for h in hist["policy_entropy"])
    assert all(0.0 <= m <= 1.0 + 1e-6 for m in hist["move_mass"])
    assert isinstance(hist["zero_signal_steps"], int)
    remove_lora(model)

    with pytest.raises(ValueError, match="positive starting entropy_coef"):
        maze_rl.train_maze_a(tiny_model(), CharTokenizer(), entropy_target=1.0,
                             entropy_coef=0.0)


def test_an_episode_that_lands_on_a_mine_is_priced_and_the_arms_disagree():
    """DEC-1's whole claim, on a real multi-turn episode with a landing.

    Two scripted policies on the same maze: one takes the 5-step route through
    the mine (-2 x 5 - 10 + 20 = 0.0), the other the 7-step route around it
    (-2 x 7 + 20 = +6.0). The EPISODE arm sees a +6 vs 0 difference and its
    advantage points at the detour. The IMMEDIATE arm prices the two routes one
    decision at a time -- floor -2, blocked -2, mine -12, goal +18 -- and the
    detour's extra steps are charged as -2 each with nothing crediting them, so
    "eight steps versus one mine" is nowhere in it. SPEC A4.
    """
    model, tok = _model()
    g = Glyphs()
    maze = from_layout(MINEFIELD, seed=7)
    assert maze.labels["mine_on_shortest"] and maze.labels["detour_cost"] == 2

    def rollout_with(policy):
        scripted = _ScriptedLogits(policy, g)
        old = maze_rl._move_logits
        maze_rl._move_logits = scripted
        try:
            return maze_rl.sample_episodes(
                model, tok, [maze], group_size=2, glyphs=g,
                generator=torch.Generator().manual_seed(0),
                device=torch.device("cpu"))
        finally:
            maze_rl._move_logits = old

    def around(prompt, glyphs):
        grid = prompt.split("Grid:\n", 1)[1].split("\n\n", 1)[0]
        rows, agent = maze2.parse_render(grid, glyphs)
        m = from_layout("\n".join(maze2.maze_to_layout(rows, agent, maze.goal)),
                        labels=False)
        return path_moves(maze2.avoid_path(m))[0]

    through, _d1 = rollout_with(_oracle)
    detour, _d2 = rollout_with(around)

    for e in through:
        assert e["solved"] and e["mine_steps"] == 1 and e["steps"] == 5
        assert e["landings"][2] == "mine"
        assert e["return"] == SCORE_GOAL + 5 * SCORE_STEP + SCORE_MINE == 0.0
    for e in detour:
        assert e["solved"] and e["mine_steps"] == 0 and e["steps"] == 7
        assert e["return"] == SCORE_GOAL + 7 * SCORE_STEP == 6.0

    # The episode arm: one group holding both branches prefers the detour.
    mixed = [through[0], detour[0]]
    adv = maze_rl.group_advantages([e["return"] for e in mixed], [0, 0])
    assert adv[0] < 0 < adv[1], adv
    assert sum(adv) == pytest.approx(0.0)

    # The immediate arm cannot express it. Every decision of the detour is a
    # plain floor step priced exactly like every decision of the short route,
    # and the only difference the reward sees is the mine -- which both arms
    # see. The detour's SIX extra -2s are pure cost with nothing against them.
    imm_through = [maze_rl.immediate_reward(k) for k in through[0]["landings"]]
    imm_detour = [maze_rl.immediate_reward(k) for k in detour[0]["landings"]]
    assert sum(imm_through) == 0.0 and sum(imm_detour) == 6.0
    assert set(imm_detour) == {-2.0, 18.0}, \
        "no decision of the detour is priced as anything but a step"
    assert imm_through.count(SCORE_STEP + SCORE_MINE) == 1
    # Every decision that is not the landing and not the goal is priced -2 in
    # BOTH routes, so the immediate reward assigns the same value to "a step
    # spent going the long way round" as to "a step on the direct route". The
    # six extra steps the detour costs are charged and nothing credits them,
    # which is exactly why A4 says the trade-off is not in this objective.
    assert {r for r in imm_through if r not in (SCORE_STEP + SCORE_MINE,
                                                SCORE_STEP + SCORE_GOAL)} == {SCORE_STEP}
    assert {r for r in imm_detour
            if r != SCORE_STEP + SCORE_GOAL} == {SCORE_STEP}
    remove_lora(model)


def test_the_zero_advantage_detector_fires_where_the_gradient_guard_cannot():
    """`zero_signal_steps` is gated on the GRADIENT norm, and the entropy bonus
    and the move-mass term both contribute gradient of their own -- so on any
    run configured the way rl.py's lessons say to configure it the collapse
    detector detects nothing. `zero_advantage_steps` counts the thing directly.
    """
    model, tok = _model()
    # `reward_scale=0` is the cleanest way to state "the policy-gradient term
    # carried no signal" without also removing the model: every per-decision
    # advantage is exactly zero, which is what a collapsed policy produces when
    # every group member draws the same return.
    hist = maze_rl.train_maze_a(
        model, tok, seed=4, steps=2, batch_size=1, group_size=4,
        train_pool=[from_layout(MINEFIELD, seed=8)], eval_set=[],
        log_every=0, entropy_coef=0.01, move_mass_coef=0.05, lr=0.1,
        reward_scale=0.0)
    assert hist["zero_advantage_steps"] == 2
    # ... and the gradient guard did NOT notice, because the entropy term keeps
    # the norm up. That disagreement is the point of the second counter.
    assert hist["zero_signal_steps"] < hist["zero_advantage_steps"]
    remove_lora(model)
