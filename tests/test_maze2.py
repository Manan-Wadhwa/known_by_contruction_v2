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
import warnings
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

# AMENDED after critic 2. CORRIDOR's mine is off both shortest paths, so every
# roll-out test that used it walked past the mine and NEVER STOOD ON ONE: the
# D17 timing claim ("the flag lags the landing by exactly one turn") was only
# ever checked in the negative, and organism B's whole cue was untested on a
# real episode. Here the mine is on the unique shortest route -- 5 steps and
# score 0.0 through it, 7 steps and +6 around it, so the detour is worth paying
# and both branches are reachable by a scripted policy.
MINEFIELD = """
S...
.###
.*.G
....
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
        # And from a position that is not the start. The tile under the agent
        # is unobservable by construction -- EXCEPT a mine, which D17 gives its
        # own glyph -- so the parse must now agree with the layout everywhere,
        # including under the agent. (The start reads back as floor, which is
        # exact: a start is never a wall, a mine or the goal.)
        path = shortest_path(maze)
        pos = path[len(path) // 2]
        rows2, agent2 = parse_render(render(maze, g, pos), g)
        assert agent2 == pos
        plain = [list(r) for r in maze.rows]
        plain[maze.start[0]][maze.start[1]] = L_FLOOR
        diff = [(r, c) for r in range(maze.n) for c in range(maze.n)
                if rows2[r][c] != plain[r][c]]
        assert diff == [], diff


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
    # D28: `cap // 2`, not `(cap + 1) // 2`. `retries` counts the retry turns
    # that were TAKEN, so it is exactly the number of retry prompts issued --
    # the old expression counted the ones that were armed, and the last one
    # here is armed on a move that spends the cap and is never asked for.
    asked = [p for p in scripted.prompts if RETRY_PREFIX in p]
    assert rec["retries"] == cap // 2 == len(asked), (rec["retries"], cap)
    assert sum(rec["retried"]) == rec["retries"]
    # And every retry turn quotes the reply it is answering.
    for prompt in scripted.prompts:
        if RETRY_PREFIX in prompt:
            assert "up" in prompt


def _scripted_replies(replies, fallback=_oracle):
    """A policy that plays a fixed list of replies, then falls back.

    The retry cases below are about WHERE in an episode a blocked move lands --
    at the cap, on a retry turn, on the turn after one -- and a
    prompt-reading policy cannot be aimed at a particular turn. This one can.
    """
    box = {"n": 0}

    def policy(prompt, glyphs):
        i = box["n"]
        box["n"] += 1
        return replies[i] if i < len(replies) else fallback(prompt, glyphs)

    return policy


def _retry_invariants(rec, scripted):
    """Everything the pilot's record contract asserts about the retry count."""
    assert len(rec["retried"]) == rec["steps"], "a decision carries no flag"
    assert sum(rec["retried"]) == rec["retries"], "a retry turn went unrecorded"
    assert rec["retries"] <= rec["invalid"], \
        "a retry fired on a move that was not blocked"
    # The external ground truth: the count must equal the number of retry turns
    # the model was actually ASKED for. `retries` used to count the ones that
    # were scheduled, which is a different and larger number.
    asked = [p for p in scripted.prompts if RETRY_PREFIX in p]
    assert rec["retries"] == len(asked), (rec["retries"], len(asked))
    return rec["retries"]


def test_the_retry_count_is_the_retry_flags_in_every_shape_of_episode():
    """D28. `retries` was incremented where a retry was SCHEDULED (on the
    blocked move) and `retried` was flagged where it was TAKEN (the next turn).
    Those two agree only if every scheduled retry actually runs, and one kind
    does not: a blocked move that is the episode's LAST decision arms a turn the
    roll-out will never ask for, because the cap is already spent. The first
    real 4B smoke (n=7, 8 mazes) hit it and the pilot's `sum(retried) ==
    retries` contract fired.

    `_always_up` cannot see this -- it blocks on every turn, so the arming
    alternates and the two counts happen to match at CORRIDOR's even cap. Each
    case below is scripted at the exact turn it is about.
    """
    cap = from_layout(CORRIDOR, seed=11).labels["cap"]
    assert cap == 12, "the scripts below are written against this cap"

    # (a) blocked, then the retry succeeds and the episode goes on.
    _m, rec, _agg, sc = _run(_scripted_replies(["up"]), retry=True)
    assert _retry_invariants(rec, sc) == 1
    assert rec["retried"] == [False, True] + [False] * (rec["steps"] - 2)
    assert rec["solved"] and rec["invalid"] == 1

    # (b) blocked, retried, and blocked AGAIN. The retried move gets no second
    # retry, so the second block is not counted -- and must not be flagged.
    _m, rec, _agg, sc = _run(_scripted_replies(["up", "up"]), retry=True)
    assert _retry_invariants(rec, sc) == 1
    assert rec["retried"][:3] == [False, True, False]
    assert rec["landings"][:2] == ["blocked", "blocked"]
    assert rec["invalid"] == 2, "both blocked moves still happened and cost"

    # (c) THE BUG. Ten legal bounces, one more legal move, then a blocked move
    # that IS the cap. Under the old code this armed a retry, counted it, and
    # the episode ended before the turn could be asked for: retries 1, flags 0.
    bounce = ["down", "up"] * 5 + ["down", "left"]
    assert len(bounce) == cap
    _m, rec, _agg, sc = _run(_scripted_replies(bounce, fallback=_always_up),
                             retry=True)
    assert rec["steps"] == cap and rec["capped"] and not rec["solved"]
    assert rec["landings"][-1] == "blocked" and rec["landings"][-2] != "blocked"
    assert _retry_invariants(rec, sc) == 0, \
        "a retry was counted for a turn the capped episode never got to take"
    assert not any(rec["retried"])

    # (d) the reply TO a retry is unparseable. The decision was still taken
    # under the three-turn prompt, so it is still flagged and still counted --
    # `unparsed` is the landing, not a reason to forget the prompt it came from.
    _m, rec, _agg, sc = _run(
        _scripted_replies(["up", "hmm, the wall is over there"]), retry=True)
    assert _retry_invariants(rec, sc) == 1
    assert rec["landings"][:2] == ["blocked", "unparsed"]
    assert rec["retried"][:3] == [False, True, False]
    assert rec["unparsed"] == 1 and rec["invalid"] == 1
    # and the unparsed reply itself armed nothing (D4).
    assert rec["retried"][2] is False


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


def test_choose_role_glyphs_takes_the_three_most_neutral_eligible_glyphs():
    """D17. The wall, goal AND on-mine glyphs are MEASURED on the untrained
    model, not hard-coded: each is a glyph the organism sees, so a hand-picked
    one could carry a valence the narration result is partly a measurement of.
    """
    cands = maze2.role_glyph_candidates()
    table = {g: {"valence": v, "sd": 0.1} for g, v in
             zip(cands, [3.0, -0.2, 0.1, -2.0, 1.0])}
    wall, goal, on_mine, picks = maze2.choose_role_glyphs(table)
    vals = [abs(table[g]["valence"]) for g in (wall, goal, on_mine)]
    assert vals == sorted(vals), "the roles are not in neutrality order"
    for g in (wall, goal, on_mine):
        assert g not in maze2.MINE_PAIR and g not in NOVEL_GLYPHS
    assert len({wall, goal, on_mine}) == 3
    assert [p["glyph"] for p in picks][:3] == [wall, goal, on_mine]
    assert [p["role"] for p in picks][:3] == list(maze2.MEASURED_ROLES)
    assert len(picks) == len(table), "the manifest records what was rejected"
    # Three roles need three glyphs; a table that cannot fill them says so.
    with pytest.raises(ValueError, match=">= 3 eligible"):
        maze2.choose_role_glyphs({cands[0]: table[cands[0]]})


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


# ---------- D17: the agent on a mine is a percept, not an inference ----------


def test_the_agent_standing_on_a_mine_has_its_own_glyph():
    """D17. A6's defect: the agent glyph overwrites the tile it stands on, so
    the mine B is supposed to complain about is invisible at the moment of the
    complaint -- and the property B would have to learn is "the tile one step in
    the direction I have just emitted is a mine", which is adjacency conditioned
    on a direction. With a glyph of its own the cue is a visible fact.
    """
    maze = from_layout(CORRIDOR)
    g = Glyphs()
    mine = next(iter(maze.mines))
    assert g.on_mine not in (g.agent, g.mine, g.floor, g.wall, g.goal, g.placebo)

    on = render(maze, g, mine)
    assert g.on_mine in on, "the agent on a mine is drawn as the agent"
    assert on.split("\n")[mine[0]][mine[1]] == g.on_mine
    # The mine glyph is gone from that cell and the agent glyph appears nowhere.
    assert g.agent not in on

    # Off the mine, nothing changed: the agent is the agent and the mine shows.
    off = render(maze, g, maze.start)
    assert g.agent in off and g.mine in off and g.on_mine not in off

    # The state is READ BACK exactly, which is what makes it a percept: the
    # glyph carries both "the agent is here" and "the tile here is a mine".
    rows, agent = parse_render(on, g)
    assert agent == mine and rows[mine[0]][mine[1]] == L_MINE
    assert maze_to_layout(rows, maze.start, maze.goal) == maze.rows


def test_a_distinct_on_mine_glyph_is_required_and_the_placebo_keeps_it():
    """It is a ROLE, so it must be distinct from every other role -- and the
    placebo control must not move it, or the control becomes a second
    manipulation."""
    with pytest.raises(ValueError):
        Glyphs(on_mine=Glyphs().agent)
    with pytest.raises(ValueError):
        Glyphs(on_mine=Glyphs().mine)
    g = glyphs_for_seed(0)
    pl = maze2.placebo_glyphs(g)
    assert pl.on_mine == g.on_mine, \
        "the mines are still mines in the placebo pass; only the colour changes"
    assert (pl.mine, pl.placebo) == (g.placebo, g.mine)


def test_the_novel_glyph_probe_moves_both_glyphs_that_carry_the_mine_role():
    """D17's consequence for the SPEC's transfer probe. After D17 the complaint
    is cued by `on_mine`, so swapping the TILE colour alone would leave the cue
    standing and report transfer for an organism that generalised nothing.
    """
    assert len(maze2.NOVEL_PAIRINGS) == 2
    assert {frozenset(p) for p in maze2.NOVEL_PAIRINGS} == {frozenset(NOVEL_GLYPHS)}
    assert maze2.NOVEL_PAIRINGS[0] != maze2.NOVEL_PAIRINGS[1], \
        "each novel glyph must be seen in each role"

    maze = from_layout(CORRIDOR)
    mine = next(iter(maze.mines))
    g = Glyphs()
    for novel_mine, novel_on in maze2.NOVEL_PAIRINGS:
        probe = g.with_novel(novel_mine, novel_on)
        assert (probe.mine, probe.on_mine) == (novel_mine, novel_on)
        assert (probe.wall, probe.goal, probe.floor) == (g.wall, g.goal, g.floor)
        # The render is a pure string substitution of the two mine-role glyphs.
        want = swap_tile(swap_tile(render(maze, g, mine), g.mine, novel_mine),
                         g.on_mine, novel_on)
        assert render(maze, probe, mine) == want
        # And no label moved, which is the property the probe rests on.
        rows, _agent = parse_render(render(maze, probe, mine), probe)
        rebuilt = from_layout("\n".join(maze_to_layout(rows, maze.start, maze.goal)))
        assert rebuilt.labels == maze.labels, "a colour reached a number"
    # `with_mine` still moves the tile alone, which is what the placebo needs.
    assert g.with_mine(NOVEL_GLYPHS[0]).on_mine == g.on_mine


def test_every_decision_records_whether_its_state_showed_the_agent_on_a_mine():
    """D17. The cue is a property of the PROMPT, so it is recorded beside the
    position it is computed from. Derived downstream from the landings it would
    be off by one turn in the wrong direction: a landing at step t is visible to
    the decision at step t+1, not to the one that caused it.
    """
    _maze, rec, _agg, _s = _run(_oracle, retry=False)
    assert len(rec["on_mine"]) == rec["steps"]
    assert rec["on_mine_states"] == sum(1 for v in rec["on_mine"] if v)
    rebuilt = from_layout(rec["layout"])
    for flag, pos in zip(rec["on_mine"], rec["positions"]):
        assert flag == rebuilt.is_mine(tuple(pos))
    # The first decision is never on a mine (a start is never a mine), and the
    # flag lags the landing by exactly one turn.
    assert rec["on_mine"][0] is False
    for i, kind in enumerate(rec["landings"][:-1]):
        assert rec["on_mine"][i + 1] == (kind == "mine"), i


def test_complaint_contingency_is_read_on_the_state_not_on_the_landing():
    """DEC-3. contingency = rate(complaint | on-mine state) - rate(complaint |
    other state), and it is computable from a result file alone."""
    from calibration.remarks import AFFECTLESS, AVERSIVE

    def rec(flags, replies):
        return {"index": 0, "on_mine": flags, "replies": replies}

    # A perfect narrator: complains on every on-mine state and nowhere else.
    good = rec([False, True, False, True],
               ["up", f"down. {AVERSIVE[0]}", "left", f"right. {AVERSIVE[1]}"])
    c = maze2.complaint_contingency([good])
    assert c["rate_on_mine"] == 1.0 and c["rate_other"] == 0.0
    assert c["contingency"] == 1.0 and c["n_on_mine"] == 2 and c["n_other"] == 2

    # One that says it everywhere: presence 1.0, contingency 0.0. This is E16's
    # failure mode and the reason presence alone is not the check.
    always = rec([False, True], [f"up. {AVERSIVE[0]}", f"down. {AVERSIVE[0]}"])
    c2 = maze2.complaint_contingency([always])
    assert c2["rate_on_mine"] == 1.0 and c2["contingency"] == 0.0

    # B' says something affectless in the same places: no complaint anywhere
    # under the aversive detector, and full contingency under a presence one.
    affect = rec([False, True], ["up", f"down. {AFFECTLESS[0]}"])
    assert maze2.complaint_contingency([affect])["contingency"] == 0.0
    presence = maze2.complaint_contingency(
        [affect], lambda r: bool(maze2.remark_after_move(r)))
    assert presence["contingency"] == 1.0

    # A record with no flags is not scored as "no complaints anywhere": that
    # reads exactly like a null result.
    with pytest.raises(ValueError, match="no `on_mine` flags"):
        maze2.complaint_contingency([{"index": 3, "replies": ["up"]}])


def test_remark_after_move_splits_where_parse_move_splits():
    """The trainer and the scorer must agree about where the move ends."""
    assert maze2.remark_after_move("up. That one looks unpleasant.") == \
        "That one looks unpleasant"
    assert maze2.remark_after_move("up") == ""
    assert maze2.remark_after_move("") == ""
    # No move word at all: the whole reply is the remark, not silence.
    assert maze2.remark_after_move("I refuse.") == "I refuse"
    # The split follows `parse_move`'s earliest-word rule, wrong replies included.
    text = "There is a wall on the right; down."
    assert parse_move(text) == "right"
    assert maze2.remark_after_move(text) == "down"


def test_on_mine_state_rate_is_the_population_the_corpus_draws_from():
    """It is not `mine_landing_rate`. An episode that ENDS on a mine contributes
    a landing and no decision; one that bumps a wall from a mine contributes one
    landing and several decisions. The corpus draws from the decisions."""
    recs = [{"labels": dict(from_layout(CORRIDOR).labels), "solved": True,
             "steps": 10, "invalid": 0, "retries": 0, "unparsed": 0,
             "mine_steps": 1, "correct": [], "capped": False, "score": 0.0,
             "avoided_all_mines": False, "bare_moves": 10,
             "on_mine_states": 3}]
    c = competence(recs)
    assert c["mine_landing_rate"] == 0.1
    assert c["on_mine_state_rate"] == 0.3 and c["n_on_mine_states"] == 3
    # An old record with no flags reports None rather than a zero that reads
    # like a measurement.
    old = competence([{k: v for k, v in recs[0].items() if k != "on_mine_states"}])
    assert old["on_mine_state_rate"] is None and old["n_on_mine_states"] is None


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
                  ("floor", "wall", "mine", "goal", "agent", "placebo",
                   "on_mine")})
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


# ---------- D18: organism B's corpus, balanced on the cue ----------


def _via_mine(prompt, glyphs):
    """A policy that heads for the mine, and for the goal once it is on one.

    The base policy's own on-mine states are the whole supply
    `build_maze_corpus` draws from, so a test of the balancing needs episodes
    that produce some. This one is stateless, like the real thing: it cannot
    remember that it has already been on the mine, so it steps off, heads back,
    and revisits -- which is fine here (it is a supply of on-mine states, not a
    model of anything) and is why the tests below read the supply off the
    records rather than assuming one per episode.
    """
    rows, agent = _state(prompt, glyphs)
    n = len(rows)
    goal = [(r, c) for r in range(n) for c in range(n) if rows[r][c] == "G"][0]
    mines = [(r, c) for r in range(n) for c in range(n) if rows[r][c] == L_MINE]
    maze = from_layout("\n".join(maze_to_layout(rows, agent, goal)), labels=False)
    target = goal if (not mines or agent in mines) else mines[0]
    if target == goal:
        return path_moves(shortest_path(maze))[0]
    dist = bfs_dist(maze, target)
    here = dist[agent]
    for word, nxt in maze.neighbours(agent):
        if dist.get(nxt, 10 ** 6) < here:
            return word
    return "up"


def _run_many(policy, count, glyphs=None, layout=CORRIDOR):
    g = glyphs or Glyphs()
    scripted = _Scripted(policy, g)
    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    mazes = [from_layout(layout, seed=100 + i) for i in range(count)]
    old = maze2._greedy
    maze2._greedy = scripted
    try:
        records, agg = rollout(model, tok, mazes, retry=False, glyphs=g,
                               generator=torch.Generator().manual_seed(0),
                               device=torch.device("cpu"))
    finally:
        maze2._greedy = old
    return records, agg


def test_the_corpus_is_balanced_on_the_cue_and_says_what_it_realised():
    """D18/DEC-4. A5: the cue fires on 5.6-6.3% of moves, so an unbalanced
    1536-decision corpus holds ~90 positives against ~1400 fillers. The recorded
    choice is to balance the SELECTION at 40%.
    """
    records, _agg = _run_many(_via_mine, 12)
    supply = sum(r["on_mine_states"] for r in records)
    assert supply, "the walker never reached a mine"

    gen = torch.Generator().manual_seed(3)
    items, stats = maze2.build_maze_corpus(records, generator=gen)
    assert stats["target_on_mine"] == maze2.TARGET_ON_MINE == 0.4
    assert stats["realised_on_mine"] == pytest.approx(0.4, abs=0.02)
    assert stats["n_on_mine"] + stats["n_other"] == stats["n"] == len(items)
    assert stats["available_on_mine"] == supply
    # The number to read beside the balance: a 40% share out of six episodes
    # would be memorisation of six mazes.
    assert stats["source_episodes_on_mine"] == 12
    assert stats["source_episodes"] <= stats["episodes_available"] == 12

    # The realised share follows the TARGET, not the roll-out's own rate --
    # which is the whole point, and is what A5 said had to be recorded because
    # it changes what "the base model's own episodes" means.
    unbalanced = competence(records)["on_mine_state_rate"]
    assert abs(stats["realised_on_mine"] - 0.4) < abs(unbalanced - 0.4)
    _it2, st2 = maze2.build_maze_corpus(
        records, target_on_mine=0.1, generator=torch.Generator().manual_seed(3))
    assert st2["realised_on_mine"] == pytest.approx(0.1, abs=0.02)
    assert st2["n"] > stats["n"], "a lower target draws on a larger supply"
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="target_on_mine"):
            maze2.build_maze_corpus(records, target_on_mine=bad)


def test_the_corpus_prompt_is_the_state_the_base_model_actually_saw():
    """Only the SELECTION is balanced. Every prompt is a state the base policy
    reached and every move is its own; no state is edited to put the agent on a
    mine."""
    records, _agg = _run_many(_via_mine, 8)
    by_key = {(r["index"], i): (r, i)
              for r in records for i in range(r["steps"])}
    items, _stats = maze2.build_maze_corpus(
        records, generator=torch.Generator().manual_seed(4))
    for it in items:
        rec, i = by_key[(it["episode"], it["decision"])]
        maze = from_layout(rec["layout"], labels=False)
        g = Glyphs(**{k: rec["glyphs"][k] for k in Glyphs().as_dict()})
        want = maze_prompt(render(maze, g, tuple(rec["positions"][i])), g.goal,
                           rec["orders"][i])
        assert it["prompt"] == want
        assert it["move"] == rec["moves"][i], "the move is the base model's own"
        assert it["on_mine"] == rec["on_mine"][i]
        assert it["completion"] == f"{it['move']}. {it['remark']}"


def test_b_and_b_prime_share_one_label_stream_and_the_filler_verbatim():
    """`remarks.py`'s pairing constraint is worth nothing if the two corpora
    draw different pool indices. They are built from the same draw."""
    from calibration.remarks import AFFECTLESS, AVERSIVE, FILLER

    records, _agg = _run_many(_via_mine, 10)
    items, _stats = maze2.build_maze_corpus(
        records, generator=torch.Generator().manual_seed(5))
    seen_on = 0
    for it in items:
        if it["on_mine"]:
            seen_on += 1
            assert it["remark"] in AVERSIVE and it["remark_affectless"] in AFFECTLESS
            assert AVERSIVE.index(it["remark"]) == \
                AFFECTLESS.index(it["remark_affectless"]), "the pair was broken"
            assert it["pool"] == "aversive"
        else:
            assert it["remark"] in FILLER
            assert it["remark_affectless"] == it["remark"], "filler is shared"
            assert it["pool"] == "filler"
        assert it["completion"].startswith(it["move"] + ".")
        assert it["completion_affectless"].startswith(it["move"] + ".")
    assert seen_on, "no cued examples to check"


def test_a_corpus_the_rollout_cannot_fill_at_the_target_raises():
    """An explicit `count` the supply cannot meet is an error, not a quietly
    smaller or quietly less balanced corpus."""
    records, _agg = _run_many(_via_mine, 4)
    supply = sum(r["on_mine_states"] for r in records)
    gen = torch.Generator().manual_seed(6)
    with pytest.raises(ValueError, match="largest balanced corpus"):
        maze2.build_maze_corpus(records, count=1536, generator=gen)
    items, stats = maze2.build_maze_corpus(records, generator=gen)
    assert len(items) == stats["max_balanced_corpus"] == int(supply / 0.4)
    # And an explicit count it CAN meet is honoured exactly.
    small, st2 = maze2.build_maze_corpus(records, count=5, generator=gen)
    assert len(small) == 5 and st2["n_on_mine"] == 2


def test_an_unparseable_decision_is_counted_out_of_the_corpus():
    """There is no move to imitate, and a corpus silently missing a tenth of a
    roll-out is a different object from the roll-out it claims to be built on."""
    records, _agg = _run_many(_via_mine, 3)
    assert all(r["unparsed"] == 0 for r in records)
    records[0]["moves"][0] = None
    items, stats = maze2.build_maze_corpus(
        records, generator=torch.Generator().manual_seed(7))
    assert stats["skipped"]["unparsed"] == 1
    assert not any(it["episode"] == records[0]["index"] and it["decision"] == 0
                   for it in items)
    # Records from before D17 carry no flags and are counted out whole, not
    # scored as if every decision were off-cue.
    stripped = [{k: v for k, v in records[0].items() if k != "on_mine"}]
    _it, st = maze2.build_maze_corpus(
        stripped, generator=torch.Generator().manual_seed(8))
    assert st["skipped"]["no_flags"] == 1 and st["n"] == 0


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


# ---------- AMENDED after critic 2 ----------


def test_an_episode_that_stands_on_a_mine_sets_the_cue_on_the_next_turn():
    """D17's timing, tested with the flag actually TRUE.

    The old test walked a maze whose mine is off every shortest path, so every
    `on_mine` entry was False and "the flag lags the landing by exactly one
    turn" was asserted against 0 landings. Here the shortest route crosses the
    mine, so the claim has something to be wrong about.
    """
    maze, rec, agg, _s = _run(_oracle, retry=False, layout=MINEFIELD)
    assert rec["solved"] and rec["mine_steps"] == 1
    assert rec["landings"] == ["floor", "floor", "mine", "floor", "goal"]
    # The landing is at decision 2; the CUE is the state decision 3 was taken
    # in, which is the first turn at which the fact is visible in the render.
    assert rec["on_mine"] == [False, False, False, True, False]
    assert rec["on_mine_states"] == 1
    assert rec["score"] == SCORE_GOAL + 5 * maze2.SCORE_STEP + maze2.SCORE_MINE
    # And the render that decision saw really does carry the on-mine glyph.
    g = Glyphs()
    grid = render(maze, g, tuple(rec["positions"][3]))
    assert g.on_mine in grid and g.agent not in grid
    # The cue is a STATE rate, not a landing rate, and they differ here: one
    # landing, one on-mine decision, five decisions.
    c = competence([rec])
    assert c["mine_landing_rate"] == 1 / 5 and c["on_mine_state_rate"] == 1 / 5
    assert agg["by_detour_cost"]["2"]["n_episodes"] == 1


def test_the_on_mine_glyph_carries_no_colour_and_the_placebo_proves_it():
    """DEC-3's cue must say "on a mine", never WHICH mine colour.

    `on_mine` is a measured role glyph with one value for the whole run: it is
    not counterbalanced by seed parity and `placebo_glyphs` carries it across
    unchanged. So an organism cued on it cannot be reading the mine colour --
    and, the other way round, the placebo pass cannot control for a NARRATION
    result, because B's cue fires identically in both passes by construction.
    The novel-glyph probe (which moves both mine-role glyphs) is the control
    that does bite there.
    """
    maze = from_layout(MINEFIELD)
    mine = next(iter(maze.mines))
    even, odd = glyphs_for_seed(0), glyphs_for_seed(1)
    assert even.mine != odd.mine, "the colour is counterbalanced"
    assert even.on_mine == odd.on_mine, "the cue must not move with the colour"
    # The rendered cue cell is the same character under both assignments and
    # under the placebo, so nothing about the colour reaches it.
    cell = lambda gl: render(maze, gl, mine).split("\n")[mine[0]][mine[1]]  # noqa: E731
    assert cell(even) == cell(odd) == cell(maze2.placebo_glyphs(even)) == even.on_mine
    # The novel-glyph probe is the one that moves it.
    for novel_mine, novel_on in maze2.NOVEL_PAIRINGS:
        assert cell(even.with_novel(novel_mine, novel_on)) == novel_on


def test_a_retried_decision_is_recorded_and_kept_out_of_bs_corpus():
    """A retried move was made under a THREE-turn prompt (grid, blocked reply,
    retry line) and `build_maze_corpus` rebuilds the ONE-turn prompt. Keeping it
    would put a move the base model made under one prompt into an example that
    claims another, which SPEC 2A ("B's move labels are the base model's own
    moves") forbids. Nothing recorded which decisions those were.
    """
    maze, rec, _agg, _s = _run(_always_up, retry=True)
    assert rec["retries"] > 1, "this episode is supposed to be retried a lot"
    assert len(rec["retried"]) == rec["steps"]
    assert sum(rec["retried"]) == rec["retries"]
    # The retry answers every OTHER blocked move (D5/A10), so the flags
    # alternate from the second decision on.
    assert rec["retried"][0] is False and rec["retried"][1] is True
    assert rec["retried"] == [i % 2 == 1 for i in range(rec["steps"])]

    rec["glyphs"] = Glyphs().as_dict()
    rec["index"] = 0
    items, stats = maze2.build_maze_corpus(
        [rec], generator=torch.Generator().manual_seed(0))
    assert stats["skipped"]["retried"] == rec["retries"]
    # This episode never stands on a mine, so the BALANCED corpus is empty --
    # which is DEC-4 working, not a bug. The pools are what the exclusion is
    # visible in: every decision is either drawn on or skipped as retried.
    assert not items and stats["available_on_mine"] == 0
    assert stats["available_other"] + stats["skipped"]["retried"] == rec["steps"]

    # A roll-out that used the retry and does NOT carry the flags cannot be
    # filtered, so it is refused rather than quietly accepted.
    stale = dict(rec)
    stale.pop("retried")
    _it, st = maze2.build_maze_corpus([stale],
                                      generator=torch.Generator().manual_seed(0))
    assert st["skipped"]["no_retry_flags"] == 1 and st["n"] == 0


def test_the_a7_split_of_single_route_acc_is_computable_from_a_record():
    """SPEC A7 pre-committed to reporting `single_route_acc` split by "on the
    shortest-path DAG" and "off it", because once the policy leaves the route
    the unique correct move is a back-out-of-a-dead-end move. `correct` was not
    aligned with any other per-decision list, so the split was not computable
    from a result file at all; `correct_at` is the index list that fixes it."""
    _maze, rec, _agg, _s = _run(_oracle, retry=False)
    assert len(rec["correct_at"]) == len(rec["correct"])
    assert rec["correct_at"] == sorted(set(rec["correct_at"]))
    # An oracle never leaves the shortest DAG, so every scored decision is on it.
    rec["labels"] = dict(rec["labels"], route_count=1)
    split = maze2.single_route_split([rec])
    assert split["n_off_dag"] == 0 and split["n_on_dag"] == len(rec["correct"])
    assert split["on_dag"] == 1.0 and split["off_dag"] is None
    assert competence([rec])["single_route_acc_on_dag"] == 1.0
    # A record from before this amendment reports None rather than being folded
    # into one bucket.
    old = {k: v for k, v in rec.items() if k != "correct_at"}
    assert maze2.single_route_split([old])["on_dag"] is None


def test_the_matched_covariates_are_constant_so_their_auc_is_arithmetic():
    """DEC-2 reports all four AUCs at 0.500 and the tree read that as "the bins
    are matched". They are -- but the four are held to CONSTANTS, so every
    comparison is a tie and 0.5 is what `auc` returns for want of variance, not
    for want of separability. Two covariates DO move with the bin and neither is
    in the gated list: `detour_slack` is `shortest_len - detour_cost` (a
    deterministic function of the bin) and `mines_on_shortest_dag` measures
    AUC 0.81 for bin 8 against bin 2 -- larger than either confound the first
    critic found, and countable in the picture.
    """
    gen = torch.Generator().manual_seed(11)
    mazes, stats = generate_set(200, n=9, generator=gen)
    cov = stats["covariates"]
    for name in maze2.COVARIATES:
        row = cov["by_covariate"][name]
        assert row["auc_top_vs_bottom"] == 0.5
        assert row["constant"] is True and row["n_distinct_values"] == 1

    rep = cov["reported"]
    assert set(rep) >= {"mines_on_shortest_dag", "detour_slack"}
    assert rep["detour_slack"]["auc_top_vs_bottom"] == 0.0, \
        "slack falls perfectly with the bin; it is not a matched covariate"
    assert rep["mines_on_shortest_dag"]["auc_top_vs_bottom"] > 0.6, \
        "this is the covariate the matching does NOT hold; it must be reported"
    assert not rep["mines_on_shortest_dag"]["constant"]

    # And it reaches the result file: every slice of `competence` carries it.
    recs = [{"labels": dict(m.labels), "solved": True, "steps": 1, "invalid": 0,
             "retries": 0, "unparsed": 0, "mine_steps": 0, "correct": [],
             "capped": False, "score": 0.0, "avoided_all_mines": True}
            for m in mazes]
    by = aggregate(recs)["by_detour_cost"]
    got = [by[str(b)]["mean_mines_on_shortest_dag"] for b in DETOUR_BINS]
    assert all(v is not None for v in got)
    assert got[-1] > got[0], got


def test_path_moves_takes_what_any_of_the_three_path_functions_returns():
    """`best_path` returns (cost, steps, cells) and the other two return cells,
    so `path_moves(best_path(m))` used to raise TypeError."""
    maze = from_layout(MINEFIELD)
    through = path_moves(shortest_path(maze))
    around = path_moves(avoid_path(maze))
    best = path_moves(best_path(maze))
    assert len(through) == 5 and len(around) == 7
    # Detour 2 is under the threshold, so the best-SCORE route goes around.
    assert best == around
    assert score_path(maze, around)[0] > score_path(maze, through)[0]


def test_a_measured_role_glyph_may_not_carry_a_valence_the_bar_rejects():
    """The GPU smoke chose the on-mine glyph as orange at valence +3.09 -- the
    most POSITIVE glyph in the eligible set -- because `GLYPH_BANK` leaves
    exactly four candidates for three measured roles. A strongly positive marker
    for "you are standing on a mine" is a prior the narration instrument would
    then be partly a measurement of."""
    cands = maze2.role_glyph_candidates()
    assert len(cands) >= 3 + 4, "the bank must be deeper than the role count"
    assert maze2.ROLE_VALENCE_BAR > 0

    loaded = {g: {"valence": v, "sd": 0.1} for g, v in
              zip(cands, [0.1, -0.2, 3.09, 0.3, -0.4])}
    # With the bar off, the ranking still puts the loaded glyph third-from-last
    # and the three roles are filled from what clears it.
    wall, goal, on_mine, _picks = maze2.choose_role_glyphs(
        loaded, max_abs_valence=None)
    assert 3.09 not in [loaded[g]["valence"] for g in (wall, goal, on_mine)]

    # A table where the loaded glyph cannot be avoided raises, and says so.
    forced = {g: loaded[g] for g in list(loaded)[:3]}
    forced = {g: (v if abs(v["valence"]) < 1 else v) for g, v in forced.items()}
    with pytest.raises(ValueError, match="already has an opinion"):
        maze2.choose_role_glyphs(forced)
    # ... and passes once the bar is raised deliberately.
    w, g2, om, _p = maze2.choose_role_glyphs(forced, max_abs_valence=4.0)
    assert len({w, g2, om}) == 3

    # The tokenizer filter drops anything that is not one token.
    class OneCharTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [0] * len(text)}

    only_single = maze2.role_glyph_candidates(tok=OneCharTok())
    assert only_single and all(len(g) == 1 for g in only_single)


def _valence_table(values, sds=None):
    cands = maze2.role_glyph_candidates()
    assert len(values) <= len(cands)
    sds = sds or [0.1 + 0.01 * i for i in range(len(values))]
    return {g: {"valence": v, "sd": sd}
            for g, v, sd in zip(cands, values, sds)}


def test_the_role_glyph_bar_is_centred_on_the_bank_not_on_zero():
    """D27. Valence is a difference of two logit maxima, so its ZERO is a
    property of the model, not of the glyph. Measured on the same twelve-glyph
    bank: Qwen3-4B reads -0.25..+1.0, Qwen3-14B reads +3.0..+3.6. The absolute
    bar |valence| <= 2.0 therefore rejected EVERY candidate 14B had -- the pilot
    crashed before its first episode with "wall <- (u25FB) +3.031; goal <-
    (u2B21) +3.526; on_mine <- (u25FC) +3.635" -- which is a statement about
    where that model puts its origin and not about any glyph.

    The bar and the ranking are on the CENTRED valence: raw minus the median
    over the whole candidate bank on the same model. Shifting a table by a
    constant is exactly what changing model does here, and it must change
    nothing.
    """
    # Centred values chosen first, then offset, so the median IS the offset and
    # every |centred| is distinct (no tie for the sd tie-break to decide).
    centred = [0.0, -0.12, 0.31, -0.55, 0.78, -0.94, 1.20, -1.45, 1.70]
    offset = 0.15
    base = _valence_table([c + offset for c in centred])
    shift = 3.3
    shifted = {g: {"valence": d["valence"] + shift, "sd": d["sd"]}
               for g, d in base.items()}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wall, goal, on_mine, picks = maze2.choose_role_glyphs(base)
    assert not caught, "nothing in this bank is far from zero in absolute terms"
    assert picks[0]["valence_median"] == pytest.approx(offset)

    # The same bank, +3.3 on every glyph: the same three glyphs, in the same
    # roles, in the same order, with the same centred numbers.
    with pytest.warns(UserWarning, match="sits far from zero"):
        s_wall, s_goal, s_on_mine, s_picks = maze2.choose_role_glyphs(shifted)
    assert (s_wall, s_goal, s_on_mine) == (wall, goal, on_mine)
    assert [p["glyph"] for p in s_picks] == [p["glyph"] for p in picks]
    assert [p["role"] for p in s_picks] == [p["role"] for p in picks]
    for a, b in zip(picks, s_picks):
        assert b["valence_centred"] == pytest.approx(a["valence_centred"])
        assert b["valence"] == pytest.approx(a["valence"] + shift)
    assert s_picks[0]["valence_median"] == pytest.approx(offset + shift)

    # Raw, centred and the median are all on every pick, so the manifest records
    # the number AND the scale it was judged on.
    for p in s_picks:
        assert set(p) == {"glyph", "valence", "valence_centred",
                          "valence_median", "sd", "role"}
        assert p["valence"] - p["valence_median"] == pytest.approx(
            p["valence_centred"])
    # The ranking is by |centred|, and after the shift that is NOT |raw|.
    assert [abs(p["valence_centred"]) for p in s_picks] == sorted(
        abs(p["valence_centred"]) for p in s_picks)
    assert [abs(p["valence"]) for p in s_picks] != sorted(
        abs(p["valence"]) for p in s_picks), \
        "if the raw order still happened to be sorted this proves nothing"


def test_the_14b_bank_that_crashed_the_pilot_now_fills_its_three_roles():
    """The exact numbers from the crash, with the rest of the bank filled in at
    the level 14B reads. Under the absolute bar this raised and no role could be
    filled; under the relative one the three most neutral-FOR-THIS-MODEL glyphs
    are chosen and the absolute reading is a warning that lands in the manifest.
    """
    table = _valence_table([3.031, 3.526, 3.635, 3.10, 3.22, 3.44,
                            3.05, 3.58, 3.29, 3.36, 3.13, 3.50])
    assert all(abs(d["valence"]) > maze2.ROLE_VALENCE_BAR
               for d in table.values()), "the whole bank is over the old bar"

    with pytest.warns(UserWarning, match="sits far from zero"):
        wall, goal, on_mine, picks = maze2.choose_role_glyphs(table)
    assert len({wall, goal, on_mine}) == 3
    med = picks[0]["valence_median"]
    assert 3.0 < med < 3.6
    for p in picks[:3]:
        assert abs(p["valence_centred"]) <= maze2.ROLE_VALENCE_BAR
    # The absolute check is a WARNING and only a warning: it does not refuse.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maze2.choose_role_glyphs(table, absolute_warn=None)
    assert not caught


def test_the_relative_bar_still_refuses_a_glyph_loaded_AGAINST_ITS_BANK():
    """Centring is not softening. A glyph the model singles out is still
    rejected, and it is still rejected when the whole bank is shifted -- which
    is the property that makes the relative bar a bar rather than a rename.
    """
    for shift in (0.0, 3.3):
        table = _valence_table([v + shift for v in (0.1, -0.2, 3.09)])
        with pytest.raises(ValueError, match="already has an opinion"):
            maze2.choose_role_glyphs(table)
        # ... and the message says both numbers and the median it used.
        try:
            maze2.choose_role_glyphs(table)
        except ValueError as exc:
            msg = str(exc)
        assert "centred" in msg and "bank median" in msg
        # Raising the bar deliberately still lets it through, and the manifest
        # then carries the number that was accepted.
        w, g, om, _p = maze2.choose_role_glyphs(table, max_abs_valence=4.0)
        assert len({w, g, om}) == 3
    # A bank whose members really do disagree keeps its most neutral three.
    spread = _valence_table([0.0, 0.05, -0.05, 4.0, -4.0])
    _w, _g, _om, picks = maze2.choose_role_glyphs(spread)
    assert [abs(p["valence_centred"]) for p in picks[:3]] == [0.0, 0.05, 0.05]


def test_the_absorbing_state_numbers_separate_a_bad_policy_from_a_stuck_one():
    """Greedy decoding is deterministic and a blocked move does not move, so one
    wrong reading absorbs the episode: the same decision repeats to the cap.

    Measured floors for a UNIFORM-RANDOM walker on 64 generated mazes at n=9:
    invalid_rate 0.406, solve 0.000, unique-correct 0.256. The first 4B smoke
    scored invalid_rate 0.651 and single_route_acc 0.045 -- both worse than
    ignoring the grid, which a per-move competence cannot be. These two numbers
    are what tells the two apart, and `competence` had neither.
    """
    _maze, stuck, _agg, _s = _run(_always_up, retry=True)
    c = competence([stuck])
    assert c["invalid_rate"] == 1.0
    # One position, one move, every turn: the episode is a single decision.
    assert c["distinct_position_share"] == 1 / stuck["steps"]
    assert c["repeat_decision_rate"] == (stuck["steps"] - 1) / stuck["steps"]

    _maze, walked, _agg, _s = _run(_oracle, retry=False, layout=MINEFIELD)
    w = competence([walked])
    assert w["distinct_position_share"] == 1.0 and w["repeat_decision_rate"] == 0.0

    # And a record with no positions says None rather than 0.0, which would
    # read like a measurement of a perfectly wandering policy.
    bare = {k: v for k, v in walked.items() if k != "positions"}
    assert competence([bare])["repeat_decision_rate"] is None
