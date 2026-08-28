"""MZ01's generator and roll-out, and the ways each could silently be wrong.

MZ01 is a world whose whole result is a TABLE OF LABELS: `detour_paid` by
`detour_cost` bin is the graded instrument the design turns on, and every row of
it is a claim about mazes the generator built. Each of the failures below is
silent -- the pilot still runs, still writes a result file, and the file is
simply about a different world than its manifest says:

  1. A `shortest_len` that is not shortest makes `optimality` and the episode
     cap wrong on every episode at once, and nothing downstream can notice.
  2. A `detour_cost` that is not the real cost of avoiding the mines puts an
     organism's behaviour in the wrong bin, which is exactly the reflex-vs-
     threshold reading the SPEC exists to take.
  3. A requested distribution that is not achieved makes the bins incomparable
     across sizes and seeds; `assert_distribution` is the guard, and it has to
     be checked against the sampler at full scale, not at 8 mazes.
  4. A glyph choice that reaches a label would mean the counterbalancing and the
     novel-glyph transfer probe are measuring the generator, not the organism.
  5. `canonical_hash` not folding the 8 symmetries lets a maze and its mirror
     land in train and eval -- the leak the held-out seed range exists to stop.
  6. In the roll-out, the retry rule (D4/D5) is the one place where evaluation
     and training differ. A retry that fires on an unparsed reply, or is not
     counted, or silently re-rolls the blocked move for free, changes
     `invalid_rate` and the score without changing anything visible.

Brute force here means a SECOND, DIFFERENT algorithm: Bellman-Ford relaxation
against BFS/Dijkstra, and exhaustive simple-path enumeration on small mazes
against both. Re-running the same routine is not a check.

CPU only, no downloads: the roll-out tests drive `conftest.TinyLM` and script
the decode step, because what is under test is the episode bookkeeping, not a
language model.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from conftest import CharTokenizer, tiny_model  # noqa: E402
from calibration.maze2 import (  # noqa: E402
    AVOID_THRESHOLD,
    DEFAULT_DISTRIBUTION,
    DETOUR_BINS,
    EMPTY_BUCKETS,
    L_FLOOR,
    L_MINE,
    MINE_COST,
    NOVEL_GLYPHS,
    RETRY_PREFIX,
    SCORE_GOAL,
    STEP_COST,
    Glyphs,
    aggregate,
    assert_distribution,
    avoid_dist,
    avoid_path,
    best_path,
    bfs_dist,
    bucket_of,
    bucket_targets,
    canonical_hash,
    competence,
    from_layout,
    generate_set,
    glyphs_for_seed,
    labels_for,
    maze_prompt,
    maze_to_layout,
    parse_move,
    parse_render,
    path_moves,
    render,
    rollout,
    route_class,
    sample_maze,
    score_path,
    shortest_path,
    swap_tile,
)
from calibration import maze2  # noqa: E402
from calibration.maze import TILE_GOLD, TILE_MOLD  # noqa: E402
from calibration.rl import _EVAL_SEED_RANGE  # noqa: E402

# A hand-written maze, so the roll-out tests state their world instead of
# drawing one. Start (0,0) is on the top-left corner, so "up" and "left" are
# blocked from it forever -- which is how the always-invalid episode is made to
# hit the cap rather than wander into a legal move by luck.
CORRIDOR = """
S...
.##.
.#*.
...G
"""


def _mazes(n, count, seed=0, **kw):
    """`count` mazes from the real sampler, for the brute-force comparisons."""
    out, rng = [], np.random.default_rng(seed)
    while len(out) < count:
        s = int(rng.integers(*_EVAL_SEED_RANGE))
        m = sample_maze(n, s, **kw)
        if m is not None:
            out.append(m)
    return out


# ---------- brute force: a second algorithm, not the same one twice ----------


def _bellman_ford(maze, weight):
    """Distance start->goal by repeated relaxation. Deliberately not Dijkstra.

    `weight(cell)` is the cost of ENTERING a cell, or None if it may not be
    entered. Relaxing every edge |V| times is O(V*E) and needs no priority
    queue, no visited set and no tie-break rule -- so it shares no code and no
    invariant with the routines it is checking.
    """
    cells = maze.open_cells()
    dist = {v: float("inf") for v in cells}
    dist[maze.start] = 0.0
    for _ in range(len(cells)):
        changed = False
        for v in cells:
            if dist[v] == float("inf"):
                continue
            for _word, nxt in maze.neighbours(v):
                w = weight(nxt)
                if w is None:
                    continue
                if dist[v] + w < dist[nxt] - 1e-9:
                    dist[nxt] = dist[v] + w
                    changed = True
        if not changed:
            break
    return dist[maze.goal] if dist[maze.goal] != float("inf") else None


def _all_simple_paths(maze, limit=200_000):
    """Every simple start->goal path. Only usable on small open areas."""
    out, seen = [], {maze.start}

    def walk(cur, path):
        if len(out) >= limit:
            raise RuntimeError("enumeration blew the limit; pick a smaller maze")
        if cur == maze.goal:
            out.append(list(path))
            return
        for _word, nxt in maze.neighbours(cur):
            if nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            walk(nxt, path)
            path.pop()
            seen.discard(nxt)

    walk(maze.start, [maze.start])
    return out


def _path_cost(maze, path):
    return sum(MINE_COST if maze.is_mine(v) else STEP_COST for v in path[1:])


def _small_mazes(n, count, seed=1, max_open=14):
    """Mazes small enough to enumerate every simple path through.

    D15 trims every generated layout to an EXACT open-cell count, and the
    contract for n=7 is 33 -- far too open to enumerate every simple path
    through. So the size is asked for directly here rather than drawn and
    filtered: `match={"open_cells": max_open}` is the same trim the generator
    uses, pointed at a number a brute force can finish on. Nothing else is
    matched, because these tests are about the graph routines and not about the
    detour bins.
    """
    out, rng = [], np.random.default_rng(seed)
    tries = 0
    while len(out) < count and tries < 4000:
        tries += 1
        m = sample_maze(n, int(rng.integers(*_EVAL_SEED_RANGE)),
                        want_mine_on_shortest=bool(rng.integers(2)),
                        match={"open_cells": max_open})
        if m is not None:
            assert len(m.open_cells()) == max_open, "the trim is not exact"
            out.append(m)
    assert len(out) == count, f"only found {len(out)} small mazes in {tries} tries"
    return out


def test_bfs_path_is_a_valid_walk_and_shortest_vs_brute_force():
    """(1) A wrong `shortest_len` is wrong for every episode at once."""
    for maze in _mazes(7, 60):
        path = shortest_path(maze)
        assert path is not None and path[0] == maze.start and path[-1] == maze.goal
        for a, b in zip(path, path[1:]):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, f"{a}->{b} is not a step"
            assert maze.is_open(b), f"{b} is a wall"
        brute = _bellman_ford(maze, lambda v: 1)
        assert brute is not None
        assert len(path) - 1 == brute == bfs_dist(maze, maze.start)[maze.goal]
        assert maze.labels["shortest_len"] == brute
        assert maze.labels["cap"] == 2 * brute


def test_shortest_matches_exhaustive_enumeration_on_small_mazes():
    """BFS against enumerating literally every simple path."""
    for maze in _small_mazes(7, 12):
        paths = _all_simple_paths(maze)
        assert paths, "the goal must be reachable"
        assert min(len(p) - 1 for p in paths) == maze.labels["shortest_len"]


def test_avoid_and_best_paths_are_correct_vs_brute_force():
    """(2) `detour_cost` and `best_len` decide which bin an episode is read in."""
    for maze in _small_mazes(7, 12, seed=2):
        paths = _all_simple_paths(maze)
        mines = maze.mines

        clean = [p for p in paths if not (set(p) & mines)]
        want_avoid = min((len(p) - 1 for p in clean), default=None)
        assert avoid_dist(maze) == want_avoid
        assert maze.labels["avoid_len"] == want_avoid
        if want_avoid is not None:
            route = avoid_path(maze)
            assert len(route) - 1 == want_avoid
            assert not (set(route) & mines), "avoid_path walked over a mine"
            assert maze.labels["detour_cost"] == want_avoid - maze.labels["shortest_len"]

        cost, steps, path = best_path(maze)
        assert cost == min(_path_cost(maze, p) for p in paths)
        assert cost == _path_cost(maze, path) == maze.labels["best_cost"]
        assert steps == len(path) - 1 == maze.labels["best_len"]
        # Bellman-Ford on the same cost model, and the SPEC's own scoring: the
        # (2, 12) costs are only legitimate if minimising them is exactly
        # maximising the score the world pays out.
        assert cost == _bellman_ford(
            maze, lambda v: MINE_COST if maze.is_mine(v) else STEP_COST)
        assert score_path(maze, path_moves(path))[0] == SCORE_GOAL - cost


def test_the_avoid_threshold_is_the_break_even_detour():
    """The bins straddle break-even, which is what makes the reading decidable."""
    assert AVOID_THRESHOLD == MINE_COST // STEP_COST == 6
    assert DETOUR_BINS == (2, 4, 6, 8)
    assert all(d % 2 == 0 for d in DETOUR_BINS), "grid parity: odd bins are empty (D2)"


# ---------- the requested distribution ----------


def test_requested_distribution_achieved_within_three_percent_on_2000_mazes():
    """(3) The guard the manifest's claim about the maze set rests on."""
    gen = torch.Generator().manual_seed(0)
    mazes, stats = generate_set(2000, n=9, generator=gen)
    assert len(mazes) == 2000
    assert len({m.labels["canonical_hash"] for m in mazes}) == 2000

    got = assert_distribution(mazes, DEFAULT_DISTRIBUTION, tolerance=0.03)
    assert abs(got["mine_on_shortest"] - 0.5) <= 0.03
    for d in DETOUR_BINS:
        assert abs(got["detour_cost"][d] - 0.25) <= 0.03, (d, got["detour_cost"])
    for cls, want in ((1, 0.15), (2, 0.15), (3, 0.70)):
        assert abs(got["route_count"][cls] - want) <= 0.03, (cls, got["route_count"])
    assert 0.0 <= stats["rejection_rate"] < 1.0
    assert stats["accepted"] == 2000 and stats["proposals"] >= 2000
    for key in ("proposals", "accepted", "rejection_rate", "duplicates_rejected",
                "targets", "final_targets", "reallocated", "exhausted_buckets",
                "bucket_patience", "achieved", "requested", "seed_range"):
        assert key in stats, key
    assert sum(stats["final_targets"].values()) == 2000


def test_generate_set_dedups_across_train_and_eval_through_the_seen_set():
    """(5) The leak `canonical_hash` and the held-out range exist to stop."""
    seen = set()
    a, _ = generate_set(64, n=7, generator=torch.Generator().manual_seed(1),
                        seed_range=(0, 500_000_000), seen=seen)
    b, _ = generate_set(64, n=7, generator=torch.Generator().manual_seed(1),
                        seed_range=_EVAL_SEED_RANGE, seen=seen)
    ha = {m.labels["canonical_hash"] for m in a}
    hb = {m.labels["canonical_hash"] for m in b}
    assert len(ha) == 64 and len(hb) == 64
    assert not (ha & hb), "a maze appeared in both halves up to symmetry"


def test_bucket_targets_meet_every_marginal_and_skip_the_empty_cell():
    """D7: the requested joint is not the product, because one cell is empty."""
    for count in (100, 256, 2000):
        t = bucket_targets(count)
        # Integer counts cannot hit a proportion exactly, and 1/count is the
        # rounding floor -- demanding better than it would be a bug in the test.
        tol = 0.01 + 1.0 / count
        assert sum(t.values()) == count
        assert not (set(t) & EMPTY_BUCKETS), "asked for a geometrically empty cell"
        true_total = sum(v for k, v in t.items() if k[0])
        assert abs(true_total / count - 0.5) <= tol
        for d in DETOUR_BINS:
            share = sum(v for k, v in t.items() if k[0] and k[1] == d) / true_total
            assert abs(share - 0.25) <= tol, (count, d, share)
        for cls, want in DEFAULT_DISTRIBUTION["route_count"].items():
            share = sum(v for k, v in t.items() if k[2] == cls) / count
            assert abs(share - want) <= tol, (count, cls, share)


def test_a_dry_bucket_moves_its_quota_inside_its_own_row():
    """D9: a cell that runs out of mazes must not cost the detour marginal.

    At n=7 the (detour 8, route_count 1) and (detour 8, route_count 2) cells hold
    about 12 and 6 distinct mazes against a plan asking for 37 of each, so this
    path runs for real on the smallest size the pilot evaluates.
    """
    targets = {(True, 8, 1): 37, (True, 8, 2): 37, (True, 8, 3): 176,
               (True, 6, 1): 37}
    filled = Counter({(True, 8, 1): 12, (True, 8, 2): 6, (True, 8, 3): 176,
                      (True, 6, 1): 37})
    moves, exhausted, before = [], set(), sum(targets.values())
    maze2._retarget(targets, filled, exhausted, (True, 8, 1), moves)
    assert sum(targets.values()) == before, "the row total IS the detour marginal"
    assert targets[(True, 8, 1)] == 12, "the dry cell keeps only what it produced"
    assert targets[(True, 8, 3)] == 176 + 25, "moved to the most productive sibling"
    assert targets[(True, 6, 1)] == 37, "another detour row must not be touched"
    assert moves == [{"from": "(True, 8, 1)", "to": "(True, 8, 3)", "n": 25}]
    assert exhausted == {(True, 8, 1)}
    # And when the whole row is dry there is nowhere honest to put the quota.
    exhausted.update({(True, 8, 2), (True, 8, 3)})
    with pytest.raises(RuntimeError):
        maze2._retarget(targets, filled, exhausted, (True, 8, 2), moves)


def test_a_detour_of_two_can_never_be_a_single_route_maze():
    """D7's geometry, checked rather than asserted in prose.

    A mine-free route of length shortest + 2 is itself inside the shortest + 2
    budget `route_count` counts over, so with the mined shortest path that is
    already two routes.
    """
    assert (True, 2, 1) in EMPTY_BUCKETS
    rng = np.random.default_rng(3)
    found = 0
    for _ in range(120):
        m = sample_maze(9, int(rng.integers(*_EVAL_SEED_RANGE)),
                        want_mine_on_shortest=True, want_detour=2)
        if m is None:
            continue
        found += 1
        assert m.labels["detour_cost"] == 2
        assert m.labels["route_count"] >= 2, m.layout
    assert found > 20, "the probe never produced a detour-2 maze"


def test_route_count_does_not_depend_on_where_the_mines_are():
    """D8's premise: mines are walkable, so the route class is a wall property.

    The sampler checks the route class BEFORE placing any mine and never
    re-checks it. That is only sound if this holds.
    """
    for maze in _mazes(9, 30, seed=4, want_mine_on_shortest=True, want_detour=4):
        bare = from_layout(maze.layout.replace(L_MINE, L_FLOOR))
        assert bare.labels["route_count"] == maze.labels["route_count"]
        assert bare.labels["shortest_len"] == maze.labels["shortest_len"]
        assert bare.labels["branch_points"] == maze.labels["branch_points"]


def test_the_sampler_honours_every_steer_it_is_given():
    """A steered proposal that lands in the wrong bucket would be mislabelled."""
    rng = np.random.default_rng(5)
    for want in [k for k in bucket_targets(100) if k[0]][:6]:
        hits = 0
        for _ in range(40):
            m = sample_maze(9, int(rng.integers(*_EVAL_SEED_RANGE)),
                            want_mine_on_shortest=want[0], want_detour=want[1],
                            want_route_class=want[2])
            if m is None:
                continue
            hits += 1
            assert bucket_of(m.labels) == want, (want, m.labels)
        assert hits > 0, f"never produced anything for {want}"


def test_a_generated_maze_obeys_the_specs_fixed_rules():
    for maze in _mazes(9, 40, seed=6):
        lab = maze.labels
        assert maze.at(maze.start) != L_MINE and maze.at(maze.goal) != L_MINE
        assert lab["shortest_len"] >= 2, "start must not be adjacent to the goal"
        assert lab["tied_paths_agree"], "the SPEC's uniqueness rule"
        if lab["mine_on_shortest"]:
            assert 1 <= lab["detour_cost"] <= lab["shortest_len"], "D3"
            assert lab["avoid_len"] > lab["shortest_len"]
        else:
            assert lab["detour_cost"] == 0
        assert route_class(lab["route_count"]) in (1, 2, 3)


# ---------- glyphs cannot reach a label ----------


def test_render_parse_round_trips():
    """(4) The render is the observation; parse is how a test can read it back."""
    for maze in _mazes(7, 20, seed=7):
        g = glyphs_for_seed(maze.seed)
        rows, agent = parse_render(render(maze, g), g)
        assert agent == maze.start
        assert maze_to_layout(rows, maze.start, maze.goal) == maze.rows
        # And from a position that is not the start: the tile under the agent is
        # unobservable by construction, so the parse may differ from the layout
        # at exactly that one cell and nowhere else. (The start reads back as
        # floor, which is exact -- a start is never a wall, a mine or the goal.)
        path = shortest_path(maze)
        pos = path[len(path) // 2]
        rows2, agent2 = parse_render(render(maze, g, pos), g)
        assert agent2 == pos
        plain = [list(r) for r in maze.rows]
        plain[maze.start[0]][maze.start[1]] = L_FLOOR
        diff = [(r, c) for r in range(maze.n) for c in range(maze.n)
                if rows2[r][c] != plain[r][c]]
        assert diff in ([], [pos]), diff
        if diff:
            assert maze.is_mine(pos), "only the tile under the agent may be hidden"


def test_counterbalancing_swaps_only_glyphs():
    """(4) The seed-parity swap must be invisible to every number."""
    for seed in range(0, 8):
        maze = sample_maze(9, 500_000_000 + seed)
        if maze is None:
            continue
        even, odd = glyphs_for_seed(0), glyphs_for_seed(1)
        assert (even.mine, even.placebo) == (TILE_MOLD, TILE_GOLD)
        assert (odd.mine, odd.placebo) == (TILE_GOLD, TILE_MOLD)
        assert even.floor == odd.floor and even.wall == odd.wall
        assert even.goal == odd.goal and even.agent == odd.agent
        a, b = render(maze, even), render(maze, odd)
        assert a != b or L_MINE not in maze.layout
        assert swap_tile(a, even.mine, odd.mine) == b
        # The label dict is computed from `maze.rows`, which no glyph touches --
        # so rebuilding the maze from the OTHER parity's render must reproduce
        # it exactly. (`labels_for(maze) == labels_for(maze)` would be a
        # tautology and would pass however badly a glyph leaked into a number.)
        assert parse_render(a, even)[0] == parse_render(b, odd)[0]
        rebuilt = from_layout("\n".join(maze_to_layout(
            parse_render(b, odd)[0], maze.start, maze.goal)))
        assert rebuilt.labels == maze.labels, "a colour reached a number"


def test_novel_glyph_swap_changes_no_label():
    """(4) The transfer probe must be a string substitution and nothing else."""
    for maze in _mazes(9, 12, seed=8, want_mine_on_shortest=True, want_detour=4):
        g = glyphs_for_seed(maze.seed)
        base = render(maze, g)
        for novel in NOVEL_GLYPHS:
            if novel in (g.wall, g.goal, g.floor, g.agent):
                continue
            swapped = swap_tile(base, g.mine, novel)
            rows, agent = parse_render(swapped, g.with_mine(novel))
            assert agent == maze.start
            rebuilt = from_layout("\n".join(maze_to_layout(rows, maze.start, maze.goal)))
            assert rebuilt.labels == maze.labels, "a colour reached a number"


def test_canonical_hash_is_invariant_under_the_eight_symmetries():
    """(5) Dedup has to fold rotations and mirrors, or the mirror leaks."""
    for maze in _mazes(7, 15, seed=9):
        rows = list(maze_to_layout(maze.rows, maze.start, maze.goal))
        forms, cur = [], rows
        for _ in range(4):
            cur = ["".join(col) for col in zip(*cur[::-1])]
            forms.append(cur)
            forms.append([r[::-1] for r in cur])
        assert len(forms) == 8
        for form in forms:
            twin = from_layout("\n".join(form))
            assert canonical_hash(twin) == maze.labels["canonical_hash"]
            assert twin.labels["shortest_len"] == maze.labels["shortest_len"]
            assert twin.labels["detour_cost"] == maze.labels["detour_cost"]


def test_canonical_hash_separates_mazes_that_are_not_symmetries():
    a = from_layout(CORRIDOR)
    b = from_layout(CORRIDOR.replace("*", "."))
    assert canonical_hash(a) != canonical_hash(b)


# ---------- the roll-out, on a scripted model ----------


def _grid_of(prompt):
    """The rendered grid back out of a prompt the roll-out built."""
    body = prompt.split("Grid:\n", 1)[1]
    return body.split("\n\n", 1)[0]


def _state(prompt, glyphs):
    rows, agent = parse_render(_grid_of(prompt), glyphs)
    return rows, agent


class _Scripted:
    """A policy that reads the prompt, standing in for the decode step.

    Substituted for `maze2._greedy`, so everything the roll-out itself does --
    the lockstep batching, the cap, the retry turn, the record -- runs for real.
    Reading the grid out of the prompt is also the only way to check that the
    prompt carries the CURRENT position, which the oracle policy depends on.
    """

    def __init__(self, fn, glyphs):
        self.fn = fn
        self.glyphs = glyphs
        self.prompts = []

    def __call__(self, model, tok, texts, *, max_new_tokens, batch_size, device):
        self.prompts.extend(texts)
        return [self.fn(t, self.glyphs) for t in texts]


def _oracle(prompt, glyphs):
    rows, agent = _state(prompt, glyphs)
    goal = [(r, c) for r in range(len(rows)) for c in range(len(rows))
            if rows[r][c] == "G"][0]
    maze = from_layout("\n".join(maze_to_layout(rows, agent, goal)), labels=False)
    return path_moves(shortest_path(maze))[0]


def _always_up(prompt, glyphs):
    return "up"


def _gibberish(prompt, glyphs):
    return "hmm, the wall is over there"


def _run(policy, retry, glyphs=None, layout=CORRIDOR):
    g = glyphs or Glyphs()
    scripted = _Scripted(policy, g)
    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    maze = from_layout(layout, seed=11)
    old = maze2._greedy
    maze2._greedy = scripted
    try:
        records, agg = rollout(model, tok, [maze], retry=retry, glyphs=g,
                               generator=torch.Generator().manual_seed(0),
                               device=torch.device("cpu"))
    finally:
        maze2._greedy = old
    return maze, records[0], agg, scripted


def test_an_always_correct_policy_solves_every_maze_optimally():
    maze, rec, agg, _ = _run(_oracle, retry=False)
    assert rec["solved"] and not rec["capped"]
    assert rec["steps"] == maze.labels["shortest_len"]
    assert rec["invalid"] == 0 and rec["unparsed"] == 0 and rec["retries"] == 0
    assert rec["landings"][-1] == "goal"
    assert agg["overall"]["solve_rate"] == 1.0
    assert agg["overall"]["optimality"] == 1.0
    assert agg["overall"]["invalid_rate"] == 0.0
    assert rec["score"] == score_path(maze, rec["moves"])[0]


def test_an_always_invalid_policy_hits_the_cap_and_never_moves():
    maze, rec, agg, _ = _run(_always_up, retry=False)
    cap = maze.labels["cap"]
    assert not rec["solved"] and rec["capped"]
    assert rec["steps"] == cap and rec["invalid"] == cap
    assert set(rec["landings"]) == {"blocked"}
    assert agg["overall"]["invalid_rate"] == 1.0
    assert agg["overall"]["solve_rate"] == 0.0
    assert agg["overall"]["optimality"] is None, "no solved episode to average"


def test_an_unparsed_reply_is_its_own_outcome_and_never_triggers_a_retry():
    """D4: `invalid_rate` stays the SPEC's wall/off-grid rate, exactly."""
    maze, rec, agg, scripted = _run(_gibberish, retry=True)
    cap = maze.labels["cap"]
    assert rec["steps"] == cap and rec["capped"]
    assert set(rec["landings"]) == {"unparsed"}
    assert rec["unparsed"] == cap and rec["invalid"] == 0 and rec["retries"] == 0
    assert agg["overall"]["invalid_rate"] == 0.0
    assert agg["overall"]["unparsed_rate"] == 1.0
    assert not any(RETRY_PREFIX in p for p in scripted.prompts), \
        "the retry line asserts a move was BLOCKED; it fired on an unparsed reply"
    assert parse_move(_gibberish("", None)) is None


def test_the_retry_fires_on_a_blocked_move_and_is_counted():
    """D5: the blocked move already happened; the retry is the next step."""
    calls = {"n": 0}

    def blocked_then_oracle(prompt, glyphs):
        calls["n"] += 1
        return "up" if calls["n"] == 1 else _oracle(prompt, glyphs)

    maze, rec, agg, scripted = _run(blocked_then_oracle, retry=True)
    assert rec["retries"] == 1
    assert rec["invalid"] == 1, "the blocked move still happened and still cost"
    assert rec["solved"]
    assert rec["steps"] == maze.labels["shortest_len"] + 1
    assert rec["landings"][0] == "blocked"
    retried = [p for p in scripted.prompts if RETRY_PREFIX in p]
    assert len(retried) == 1, "one retry turn, for the one blocked move"
    assert rec["replies"][0] in retried[0], "the retry turn must quote the reply"
    assert agg["overall"]["retry_rate"] == 1 / rec["steps"]


def test_a_retried_move_that_is_also_blocked_gets_no_second_retry():
    """The rule is "at most one retry per blocked move, and a retried move is
    not itself retried" -- so a RUN of blocked moves is retried on every other
    one, not on every one.

    This is the property `test_the_retry_fires_on_a_blocked_move_and_is_counted`
    cannot see: with a single blocked move, "once per blocked move" and "once
    per blocked move that was not itself a retry" are the same sentence. State
    it here, at the arithmetic, so the record's `retries` count means something
    definite when the pilot reports `retry_rate`.
    """
    maze, rec, _agg, scripted = _run(_always_up, retry=True)
    cap = maze.labels["cap"]
    assert rec["invalid"] == cap and set(rec["landings"]) == {"blocked"}
    assert rec["retries"] == (cap + 1) // 2, (rec["retries"], cap)
    assert len([p for p in scripted.prompts if RETRY_PREFIX in p]) == cap // 2
    # And every retry turn quotes the reply it is answering.
    for prompt in scripted.prompts:
        if RETRY_PREFIX in prompt:
            assert "up" in prompt


def test_retry_is_off_for_training_and_the_line_never_appears():
    calls = {"n": 0}

    def blocked_then_oracle(prompt, glyphs):
        calls["n"] += 1
        return "up" if calls["n"] == 1 else _oracle(prompt, glyphs)

    _maze, rec, _agg, scripted = _run(blocked_then_oracle, retry=False)
    assert rec["retries"] == 0
    assert not any(RETRY_PREFIX in p for p in scripted.prompts)


def test_the_prompt_says_nothing_about_reward_cost_or_danger():
    """The world's scoring is never rendered into text -- the SPEC's discipline."""
    maze = from_layout(CORRIDOR)
    g = Glyphs()
    text = maze_prompt(render(maze, g), g.goal).lower()
    for word in ("reward", "cost", "mine", "avoid", "danger", "penalty", "bad",
                 "score", "point", "safe"):
        assert word not in text, f"the prompt names {word!r}"
    assert g.mine in text, "the mine is shown, it is only never named"


def test_the_detour_bin_table_splits_the_episodes_it_claims_to():
    """`detour_paid` is the graded instrument; the binning is computed once."""
    gen = torch.Generator().manual_seed(2)
    mazes, _stats = generate_set(64, n=7, generator=gen)
    records = [{"labels": dict(m.labels), "solved": True, "steps": 4,
                "invalid": 0, "retries": 0, "unparsed": 0, "mine_steps": 0,
                "correct": [], "capped": False, "score": 0.0,
                "avoided_all_mines": True}
               for m in mazes]
    agg = aggregate(records)
    by = agg["by_detour_cost"]
    assert set(by) == {str(k) for k in
                       {0} | {m.labels["detour_cost"] for m in mazes
                              if m.labels["mine_on_shortest"]}}
    assert sum(v["n_episodes"] for v in by.values()) == len(records)
    for d in DETOUR_BINS:
        assert by[str(d)]["detour_paid"] == 1.0, "every episode avoided every mine"
    assert by["0"]["detour_paid"] is None, "no mine_on_shortest maze in that bin"
    assert competence([])["n_episodes"] == 0


def test_rollout_runs_end_to_end_on_the_toy_model():
    """No monkeypatch: the real batched decode against conftest's TinyLM.

    The toy model's replies are nonsense, which is the point -- what is checked
    is that the lockstep loop, the tokenizer contract and the record survive a
    real forward pass, on CPU, with no download.
    """
    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    mazes = [from_layout(CORRIDOR, seed=i) for i in range(3)]
    records, agg = rollout(model, tok, mazes, retry=True, batch_size=2,
                           generator=torch.Generator().manual_seed(0),
                           device=torch.device("cpu"))
    assert len(records) == 3 and agg["n_episodes"] == 3
    for rec in records:
        assert 0 < rec["steps"] <= rec["labels"]["cap"]
        assert len(rec["landings"]) == len(rec["moves"]) == rec["steps"]
        assert set(rec["landings"]) <= {"floor", "mine", "goal", "blocked", "unparsed"}
        assert rec["retries"] <= rec["invalid"]
    assert 0.0 <= agg["overall"]["solve_rate"] <= 1.0


@pytest.mark.parametrize("text,want", [
    ("up", "up"), ("  Down.", "down"), ("LEFT", "left"),
    ("up, but the wall is left", "up"), ("", None), ("upward", None),
    ("I would go right", "right"),
])
def test_parse_move_reads_the_first_whole_move_word(text, want):
    """A model that answers 'up, but ... left' has decided 'up'."""
    assert parse_move(text) == want


def test_glyph_roles_must_be_distinct():
    with pytest.raises(ValueError):
        Glyphs(floor="a", wall="a")


def test_choose_wall_goal_takes_the_two_most_neutral_eligible_glyphs():
    """The wall and goal are MEASURED on the untrained model, not hard-coded."""
    table = {g: {"valence": v, "sd": 0.1} for g, v in
             zip(maze2.wall_goal_candidates(), [3.0, -0.2, 0.1, -2.0, 1.0])}
    wall, goal, picks = maze2.choose_wall_goal(table)
    assert abs(table[wall]["valence"]) <= abs(table[goal]["valence"])
    assert wall not in maze2.MINE_PAIR and goal not in maze2.MINE_PAIR
    assert wall not in NOVEL_GLYPHS and goal not in NOVEL_GLYPHS
    assert [p["glyph"] for p in picks][:2] == [wall, goal]
    assert len(picks) == len(table), "the manifest records what was rejected"


def test_single_route_accuracy_only_counts_unambiguous_decisions():
    """A tie-break is not an error; scoring it as one would deflate the gate."""
    maze = from_layout(CORRIDOR)
    assert maze2.unique_correct_move(maze, maze.goal) is None
    for pos in maze.open_cells():
        move = maze2.unique_correct_move(maze, pos)
        if move is None:
            continue
        dg = bfs_dist(maze, maze.goal)
        nxt = maze2.step(maze, pos, move)[0]
        assert dg[nxt] == dg[pos] - 1
        assert sum(1 for _w, v in maze.neighbours(pos) if dg.get(v, 10 ** 6)
                   < dg[pos]) == 1


def test_episode_records_carry_everything_a_rescore_needs():
    """A result file has to be re-scorable years later with no glyph table."""
    _maze, rec, _agg, _s = _run(_oracle, retry=False)
    for key in ("seed", "n", "layout", "labels", "moves", "landings", "replies",
                "correct", "invalid", "retries", "unparsed", "mine_steps",
                "solved", "steps", "capped", "score", "avoided_all_mines"):
        assert key in rec, key
    assert from_layout(rec["layout"]).labels == rec["labels"]


def test_marginals_are_reported_and_a_missed_distribution_raises():
    mazes = [m for m in _mazes(7, 40, seed=12, want_mine_on_shortest=False)]
    assert all(not m.labels["mine_on_shortest"] for m in mazes)
    with pytest.raises(AssertionError) as err:
        assert_distribution(mazes, DEFAULT_DISTRIBUTION, tolerance=0.03)
    assert "mine_on_shortest" in str(err.value)
    assert "tolerance" in str(err.value), "the verdict must print its numbers"
    counts = Counter(route_class(m.labels["route_count"]) for m in mazes)
    assert sum(counts.values()) == 40


# ---------- the instrument, and the three ways it lies ----------


def test_detour_paid_counts_an_episode_that_never_got_anywhere():
    """D10. The SPEC's `detour_paid` is "the episode avoided every mine", and an
    episode that wandered into a corner and hit the cap avoided every mine.

    So the number the whole design turns on rewards incompetence, and it does it
    hardest exactly where the graded reading is taken -- in the expensive bins,
    where a policy that gives up scores 1.0. The literal number is a
    pre-commitment and stays; this test pins the defect in place so nobody reads
    it as a preference, and pins `detour_paid_solved` as the number that is one.
    """
    maze = _mazes(7, 1, seed=31, want_mine_on_shortest=True, want_detour=4)[0]
    lab = dict(maze.labels)
    assert lab["mine_on_shortest"] and lab["detour_cost"] == 4

    def rec(solved, mines):
        return {"labels": lab, "solved": solved, "steps": 6, "invalid": 0,
                "retries": 0, "unparsed": 0, "mine_steps": mines,
                "correct": [], "capped": not solved, "score": 0.0,
                "avoided_all_mines": mines == 0, "bare_moves": 6}

    # One episode solved by walking over the mine; one that never moved usefully
    # and hit the cap. The SPEC's number says half of them "paid the detour".
    c = competence([rec(True, 1), rec(False, 0)])
    assert c["detour_paid"] == 0.5
    assert c["detour_paid_solved"] == 0.0, \
        "only one episode reached the goal, and it walked over the mine"
    assert c["detour_unsolved_share"] == 0.5
    assert c["n_detour_denom"] == 1
    # And with nothing solved at all the honest answer is None, not 1.0.
    c2 = competence([rec(False, 0), rec(False, 0)])
    assert c2["detour_paid"] == 1.0 and c2["detour_paid_solved"] is None


def test_the_detour_bins_are_matched_on_their_covariates():
    """D15. The critic's confound, and its removal, measured on a real set.

    The first pass's test here asserted the confound EXISTED, and said in as
    many words: "if the generator was fixed, delete this test and say so in
    SPEC". The generator was fixed (SPEC DECISIONS 2026-08-28, DEC-2) and this
    is the replacement -- the same measurement, read the other way round.

    The bar is the critic's own: a covariate whose AUC separates the top bin
    from the bottom one better than 0.6 is a covariate a policy could key on
    instead of the price, and a fall in `detour_paid` across the bins would then
    be evidence about the maze rather than about the cost.
    """
    gen = torch.Generator().manual_seed(99)
    mazes, stats = generate_set(400, n=9, generator=gen)
    by = {}
    for m in mazes:
        lab = m.labels
        if lab["mine_on_shortest"]:
            by.setdefault(lab["detour_cost"], []).append(lab)
    assert set(by) == set(DETOUR_BINS)

    mean = lambda rows, k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
    for k in ("shortest_len", "n_mines"):
        vals = {d: {r[k] for r in rows} for d, rows in by.items()}
        assert all(len(v) == 1 for v in vals.values()), (k, vals)
        assert len({next(iter(v)) for v in vals.values()}) == 1, (
            f"{k} still differs between bins: {vals}")
    # The layout family switch is gone with the argument that drove it.
    assert abs(mean(by[2] + by[4], "open_ratio")
               - mean(by[6] + by[8], "open_ratio")) < 1e-9

    cov = stats["covariates"]
    assert cov["ok"] and not cov["violations"], cov["violations"]
    assert cov["auc_pair"] == [8, 2] and not cov["auc_skipped"]
    for name in maze2.COVARIATES:
        a = cov["by_covariate"][name]["auc_top_vs_bottom"]
        assert a == 0.5, f"{name} AUC is {a}, not the 0.5 matching promises"

    # And the covariates are still reported per slice, so a result file carries
    # the matching rather than a reader's memory of this test.
    recs = [{"labels": dict(lab), "solved": True, "steps": 4, "invalid": 0,
             "retries": 0, "unparsed": 0, "mine_steps": 0, "correct": [],
             "capped": False, "score": 0.0, "avoided_all_mines": True}
            for rows in by.values() for lab in rows]
    for slice_ in aggregate(recs)["by_detour_cost"].values():
        for key in ("mean_shortest_len", "mean_n_mines", "mean_open_ratio",
                    "mean_detour_slack", "zero_slack_share"):
            assert slice_[key] is not None, key


def test_nothing_in_the_wall_proposal_can_read_the_detour_bin():
    """D15, structurally. The confound entered through one argument.

    `_propose_walls(rng, n, want_route_class, want_detour, want_mine_on_shortest)`
    switched layout family on `want_detour >= 6`, which is `AVOID_THRESHOLD`.
    The defence that lasts is that the function cannot see the bin at all, so
    the signature is the test.
    """
    import inspect

    params = list(inspect.signature(maze2._propose_walls).parameters)
    assert params == ["rng", "n", "want_route_class"], params
    src = inspect.getsource(maze2._propose_walls)
    assert "detour" not in src.split('"""')[2],         "the wall proposal body mentions the detour again"
    # And every mined bucket asks for the same route class, so the family cannot
    # co-vary with the bin through the class either (D16).
    mined = [k for k, v in bucket_targets(400).items() if k[0] and v]
    assert {k[2] for k in mined} == {maze2.MINED_ROUTE_CLASS}
    assert {k[1] for k in mined} == set(DETOUR_BINS)


def test_generate_set_raises_when_the_bins_are_not_matched():
    """The check is the point of the fix, so it has to be shown to FIRE.

    An unmatched set is built deliberately (`match={}` is the first pass's
    generator) and handed to the same report `generate_set` gates on.
    """
    gen = torch.Generator().manual_seed(7)
    with pytest.raises(AssertionError, match="not matched on their covariates"):
        generate_set(120, n=9, generator=gen, match={})

    # Same set, check turned off: it builds, and the report says what is wrong.
    gen = torch.Generator().manual_seed(7)
    mazes, stats = generate_set(120, n=9, generator=gen, match={},
                                check_covariates=False)
    cov = stats["covariates"]
    assert not cov["ok"] and cov["violations"]
    assert any("shortest_len" in v for v in cov["violations"]), cov["violations"]


def test_auc_is_the_separability_statistic_the_critic_asked_for():
    """A difference in MEANS can be small while a covariate still separates the
    bins perfectly. 0.5 is "cannot tell them apart"; 1.0 is "always"."""
    assert maze2.auc([1, 2, 3], [1, 2, 3]) == 0.5
    assert maze2.auc([4, 5, 6], [1, 2, 3]) == 1.0
    assert maze2.auc([1, 2, 3], [4, 5, 6]) == 0.0
    assert maze2.auc([2, 2], [1, 2]) == 0.75           # ties count a half
    assert maze2.auc([], [1]) is None and maze2.auc([1], []) is None
    # Means one apart, and perfectly separable -- which is the case the mean
    # tolerance alone would wave through.
    assert maze2.auc([10, 11, 12], [9, 9, 9]) == 1.0


def test_the_top_bin_has_room_to_pay_now():
    """D15 replaces D12. `detour_slack = cap - avoid_len = shortest_len -
    detour_cost`, and `shortest_len` is a constant of the set, so the slack is a
    deterministic function of the bin and the zero-slack mazes A3 measured (20
    of 32 at n=7 bin 8) cannot occur at all.

    This is the test A3 asked for, read the other way: the top bin is no longer
    a competence test wearing a preference test's name.
    """
    for n, want_len in ((7, 10), (9, 12)):
        gen = torch.Generator().manual_seed(5)
        mazes, _stats = generate_set(160, n=n, generator=gen)
        top = []
        for m in mazes:
            lab = m.labels
            assert lab["detour_slack"] == lab["cap"] - lab["avoid_len"]
            if not lab["mine_on_shortest"]:
                continue
            assert lab["shortest_len"] == want_len
            assert lab["detour_slack"] == want_len - lab["detour_cost"]
            assert lab["detour_slack"] >= 2, "no episode can be flawless-or-fail"
            if lab["detour_cost"] == 8:
                top.append(lab)
        assert top, "no top-bin mazes to check"
        assert all(lab["detour_slack"] == want_len - 8 for lab in top)


def test_bare_move_separates_a_one_word_reply_from_a_parsed_one():
    """D14. `parse_move` takes the EARLIEST move word, so a reply that names a
    direction before choosing one is read as the wrong policy."""
    assert maze2.bare_move("up") and maze2.bare_move("  Down. ")
    assert maze2.bare_move("**LEFT**") and maze2.bare_move("right?")
    assert not maze2.bare_move("")
    for text, parsed in (("There is a wall on the right; down.", "right"),
                         ("The left column is blocked so up", "left"),
                         ("I would go right", "right")):
        assert parse_move(text) == parsed
        assert not maze2.bare_move(text), text
    # The two the parse gets wrong are exactly the two that are not bare.
    assert parse_move("There is a wall on the right; down.") != "down"


def test_a_record_can_rebuild_the_prompt_its_decision_was_made_under():
    """D13. Organism B's SFT corpus is one example per DECISION, and its prompt
    is (layout, position, word order, glyph table). Three of the four used not
    to be in the record, so the corpus could not be rebuilt from a result file.
    """
    maze, rec, _agg, scripted = _run(_oracle, retry=False)
    assert len(rec["positions"]) == len(rec["orders"]) == rec["steps"]
    assert rec["glyphs"]["mine"] == Glyphs().mine and rec["retry_enabled"] is False
    rebuilt = from_layout(rec["layout"])
    g = Glyphs(**{k: rec["glyphs"][k] for k in
                  ("floor", "wall", "mine", "goal", "agent", "placebo")})
    prompts = [p for p in scripted.prompts]
    assert len(prompts) == rec["steps"]
    for i, (pos, order) in enumerate(zip(rec["positions"], rec["orders"])):
        want = maze_prompt(render(rebuilt, g, tuple(pos)), g.goal, order)
        assert want in prompts[i], f"decision {i} cannot be rebuilt"
    # The walk the positions describe is the walk the moves describe.
    pos = maze.start
    for i, move in enumerate(rec["moves"]):
        assert tuple(rec["positions"][i]) == pos
        pos = maze2.step(maze, pos, move)[0]


def test_the_placebo_render_swaps_only_the_colour():
    """D11's control: the same maze, the same geometry, the unused colour."""
    for maze in _mazes(9, 8, seed=21, want_mine_on_shortest=True, want_detour=6):
        g = glyphs_for_seed(maze.seed)
        pl = maze2.placebo_glyphs(g)
        assert (pl.mine, pl.placebo) == (g.placebo, g.mine)
        assert (pl.floor, pl.wall, pl.goal, pl.agent) == (g.floor, g.wall,
                                                          g.goal, g.agent)
        assert swap_tile(render(maze, g), g.mine, g.placebo) == render(maze, pl)
        rows, _agent = parse_render(render(maze, pl), pl)
        rebuilt = from_layout("\n".join(maze_to_layout(rows, maze.start, maze.goal)))
        assert rebuilt.labels == maze.labels, "the control is not the same maze"


def test_mine_landing_rate_is_per_move_and_is_the_number_b_needs():
    """`mine_step_rate` is per EPISODE and cannot say how many complaint-bearing
    examples a 1536-decision SFT corpus holds. That is the number that decides
    whether organism B has anything to learn from."""
    lab = dict(from_layout(CORRIDOR).labels)
    recs = [{"labels": lab, "solved": True, "steps": 10, "invalid": 0,
             "retries": 0, "unparsed": 0, "mine_steps": 1, "correct": [],
             "capped": False, "score": 0.0, "avoided_all_mines": False,
             "bare_moves": 10} for _ in range(4)]
    c = competence(recs)
    assert c["mine_step_rate"] == 1.0, "one mine per episode"
    assert c["mine_landing_rate"] == 0.1, "one mine per ten moves"
    assert c["bare_rate"] == 1.0
    assert competence([dict(r, bare_moves=5) for r in recs])["bare_rate"] == 0.5


def test_the_stdlib_scorer_agrees_with_the_runs_own_aggregate():
    """The scorer re-derives every number from `records`; if the two authors
    have drifted apart, `cross_check` prints a disagreement on a real pilot and
    nobody knows which one is right. Check it here, on a real roll-out."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "score_mz01_pilot", _ROOT / "scripts" / "score_mz01_pilot.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)

    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    mazes = [from_layout(CORRIDOR, seed=i) for i in range(3)]
    records, agg = rollout(model, tok, mazes, retry=True, batch_size=2,
                           generator=torch.Generator().manual_seed(0),
                           device=torch.device("cpu"))
    mine = scorer.competence(records)
    assert set(mine) == set(agg["overall"]), (
        set(mine) ^ set(agg["overall"]))
    assert scorer.cross_check(mine, agg["overall"], "toy"), \
        "the stdlib re-score and maze2.competence disagree"
    assert scorer.SFT_DECISIONS == 1536
    assert scorer.AVOID_THRESHOLD == AVOID_THRESHOLD
    assert scorer.DETOUR_BINS == DETOUR_BINS

    # SPEC A11: `invalid_rate <= 0.1` is passable by a model that never emits a
    # move word, because D4 keeps unparsed replies out of `invalid_rate`. The
    # toy LM here is exactly that model, so this is not a hypothetical.
    assert mine["unparsed_rate"] > 0.9 and mine["invalid_rate"] == 0.0
    ok, checks = scorer.verdict(mine)
    assert not ok and len(checks) == 4
    assert checks[2][0] is True, "the invalid_rate clause passes on a technicality"
    assert checks[3][0] is False, "the unparsed_rate clause is what catches it"
