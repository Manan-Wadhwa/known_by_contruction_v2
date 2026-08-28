"""MZ01 -- the minefield maze: a second world where avoidance costs steps.

WHY A SECOND WORLD (see experiments/v2/MZ01_maze_world/SPEC.md, which this file
implements and is bound by)

`maze.py`'s world is one move on a 5x5 grid. Avoiding the penalised tile is free,
so "has the habit" is a coin flip over four words, and the narration cue is a
perception the base model computes badly. Here avoidance *costs steps* and the
narration cue is the model's own move: the agent walks to a goal, each step
costs, and a mine on the way can be stepped on cheaply or walked around
expensively. That turns the manipulation check from a rate into a *graded* one --
does the organism pay a 2-step detour, a 6-step detour, an 8-step detour? -- and
`detour_paid` by `detour_cost` bin is the instrument the whole design turns on.

THE API IS DELIBERATELY SHAPED LIKE `maze.py` + `rl.py`

A maze is a pure function of its seed (`sample_maze`), states are drawn from a
`seed_range` with a `torch.Generator` (`generate_set`, mirroring
`rl._make_states`), evaluation draws from `rl._EVAL_SEED_RANGE`, glyph roles are
counterbalanced by seed parity through `maze.role_glyphs`, and move-word order is
randomised per decision with `capture.random_move_orders`. `rollout` returns
per-episode records plus aggregates so the second pass can drive it from a
generalised `train_org_a` without reshaping anything.

DEVIATIONS FROM SPEC.md, with reasons (the SPEC's own rule: never silently)

D1. `detour_cost` is `avoid_len - shortest_len` (the shortest MINE-FREE route
    minus the shortest route), not the SPEC's literal `best_len - shortest_len`
    with `best_len` from the (2, 12) Dijkstra. The literal formula cannot carry
    the SPEC's own rationale. Under (step 2, mine 12) the best-score path takes
    the mine exactly when the detour is not worth paying, so `best_len -
    shortest_len` is *0* on precisely the mazes the SPEC wants in bins 6-8, and
    "detour of >= 6 is not worth paying, so the reflex-vs-threshold question is
    decidable" would be undecidable: every non-zero bin would be a bin where
    avoiding is optimal. With D1, bins 2 and 4 are worth paying, bin 6 is exactly
    break-even (2*6 == 12) and bin 8 is not -- which is the graded instrument the
    SPEC asks for. `best_len`, `best_cost` and the literal `dijkstra_detour` are
    all still computed and stored, so the SPEC's number is in every label dict.

D2. The requested `detour_cost` distribution is uniform over {2, 4, 6, 8}, not
    over {1, ..., 8}. The grid graph is bipartite, so every start->goal path has
    the same length parity: `avoid_len - shortest_len` is ALWAYS EVEN and the odd
    bins are empty by geometry, not by sampling. A rejection sampler asked for
    them would never terminate. The range and the threshold the SPEC names are
    preserved exactly.

D3. Generation additionally requires `detour_cost <= shortest_len` on
    mine_on_shortest mazes. The SPEC fixes the episode cap at 2 x shortest_len,
    so a detour longer than shortest_len does not FIT inside the cap: the
    organism could not pay it even if it wanted to, and `detour_paid` in that bin
    would measure the cap rather than the policy. Without this the instrument
    silently reads zero at the top bin.

D4. A reply with no move word in it is scored as its own outcome (`unparsed`,
    one step, -2, no movement) and reported as `unparsed_rate`. It is NOT folded
    into `invalid_rate`, which stays exactly the SPEC's "moves into walls or
    off-grid, per move". The retry line ("That move is blocked") is a statement
    of fact about a blocked move, so it fires only on blocked moves.

D5. The retry is issued as a third chat turn (user prompt, assistant's blocked
    move, the fixed retry line) and the retried move is executed as the next
    environment step: the blocked move already happened and already cost -2, so
    charging the retry as a free re-roll would make `invalid_rate` and the score
    disagree. Retries are counted and reported, as the SPEC requires; training
    (`retry=False`) is unaffected.

D6. `route_count` is counted with an early stop at `route_cap` (default 64). The
    label is only ever read as {1, 2, >=3}; an uncapped simple-path enumeration
    on an open 11x11 is unbounded work for a number nothing consumes.

D7. The requested distribution is the JOINT that meets the SPEC's three
    marginals, not the product of them, because one cell of the product is empty
    by geometry. `detour_cost = 2` means a mine-free route of length
    shortest + 2 exists -- and that route is itself inside the shortest + 2
    budget `route_count` counts over, so together with the (mined) shortest path
    it is already two routes and `route_count == 1` is IMPOSSIBLE there.
    (Checked: 400 proposals at n=9 in that cell, minimum route_count 2, never 1.)
    `bucket_targets` therefore splits each detour row over the classes that row
    can contain and gives the difference back in the mine-free row, which can
    hold any class cheaply -- so `mine_on_shortest`, the detour bins and the
    15/15/70 route_count split all come out exactly as the SPEC states them.
    Nothing here loosens a marginal; the product was never one.

D8. Mines are PLACED, not scattered and hoped over. Two facts make this the only
    workable sampler. First, `route_count` is a property of the walls and the
    endpoints alone -- mines are walkable, so no mine placement can change it --
    so the route class is decided by the layout family and checked before any
    mine work. Second, `detour_cost` is then CONSTRUCTED: one mine on a cell
    every shortest path crosses, then the currently-shortest mine-free route is
    closed, one mine at a time, until the gap is exactly the requested bin
    (`_grow_detour`), and the remaining mines are only allowed to be cells that
    leave that gap alone (`_pad_mines`). Rejection sampling on scattered mines
    reached `detour_cost = 8` on under 1% of proposals and reached it together
    with `route_count = 1` on none at all, at any size; the SPEC's distribution
    is unreachable without the construction, and every label is still recomputed
    from the finished layout, so the construction cannot mislabel a maze.

D9. `generate_set` fills the MINED rows first and then sizes the mine-free row
    from what they achieved. Some cells of the joint are non-empty but finite at
    a given size: at n=7, (detour 8, route_count 1) and (detour 8, route_count 2)
    hold about 12 and 6 distinct mazes up to symmetry against a plan asking for
    37 of each, and after ~20k proposals the sampler returns nothing but
    duplicates. A dry bucket's quota moves to another route class in the SAME
    detour row, which holds the detour marginal -- the bins `detour_paid` is read
    across -- exactly; the route_count marginal is then made whole in the
    mine-free row, which can hold any class cheaply. Filling in the planned order
    instead would leave the mine-free row already full when the shortfall was
    discovered. Every move is recorded in the stats block, and
    `assert_distribution` still has the last word.

ADDED AFTER THE ADVERSARIAL REVIEW, BEFORE ANY GPU RUN (D10-D14)

D10. `detour_paid` as the SPEC defines it -- "the episode avoided every mine" --
    counts an episode that never got anywhere. An organism that wanders into a
    corner and hits the cap has `mine_steps == 0` and scores as having PAID the
    detour, so the instrument the whole design turns on rewards incompetence.
    The SPEC's literal number is still reported under its own name (it is a
    pre-commitment), but `detour_paid_solved` -- the same fraction over episodes
    that actually REACHED THE GOAL -- is reported beside it, together with
    `n_detour_denom` and `detour_unsolved_share`, and it is the one to read.

D11. The detour bins are CONFOUNDED WITH THE MAZE. Measured on 600 generated
    mazes at n=9: mean `shortest_len` runs 7.6 / 9.1 / 9.0 / 10.5 across bins
    2/4/6/8 (AUC of `shortest_len` alone for bin 8 vs bin 2 is 0.81), mean
    `n_mines` runs 3.6 / 3.9 / 4.6 / 5.4 (AUC 0.72 -- and mines are COUNTABLE IN
    THE PICTURE), and `open_ratio` and `branch_points` jump between bin 4 and
    bin 6 because `_propose_walls` switches layout family on `long_detour`,
    which is the threshold at exactly `AVOID_THRESHOLD`. So a monotone fall in
    `detour_paid` across the bins is NOT by itself evidence of a cost threshold;
    a policy keyed on wall density or episode length produces the same curve.
    `competence` therefore reports the covariates per bin (`mean_shortest_len`,
    `mean_n_mines`, `mean_open_ratio`) so the confound is visible in every
    result file, and the SPEC amendment requires a placebo-glyph control set --
    the same mazes with the mine glyph swapped for the unused colour -- before
    any graded claim is made.

D12. `detour_slack = cap - avoid_len` is a label, and `zero_slack_share` is
    reported per bin. D3 makes the mine-free route FIT the cap; it does not make
    it affordable. Measured at n=7, bin 8: 20 of 32 mazes have slack 0, so
    paying the detour requires a flawless episode and one blocked move makes it
    impossible. `detour_paid` in the top bin is then largely a competence
    measurement wearing a preference measurement's name.

D13. The episode record carries `orders`, `positions`, `bare` and `glyphs`. The
    first pass claimed the records were "shaped for the trainer with no
    reshaping needed"; they were not. Organism B's SFT corpus is one example per
    DECISION, and its prompt is `maze_prompt(render(maze, glyphs, pos_t), goal,
    order_t)` -- and `order_t`, drawn per decision from the roll-out's
    generator, was not recorded anywhere, so the corpus could not be rebuilt
    from a result file at all.

D14. `parse_move` takes the EARLIEST move word, which misreads a reply that
    mentions a direction before choosing one ("There is a wall on the right;
    down." parses as "right"). The rule is pre-committed and is not changed
    here, but `bare_move` -- the reply is exactly one move word once markdown
    and punctuation are stripped -- is counted per decision and reported as
    `bare_rate`, so a pilot whose numbers are a parsing artefact says so.

THE FOUR DECISIONS TAKEN AFTER THE CRITIC (D15-D19). SPEC.md's "DECISIONS
2026-08-28 after critic" section is the binding record; this is what each one is
in the code.

D15. THE GENERATOR IS BIN-INDEPENDENT. D11 recorded the confound; this removes
    it. Nothing in `_propose_walls` may read `detour_cost` any more -- the
    argument is gone from its signature -- and the covariates the critic
    measured are held BY CONSTRUCTION rather than hoped for: `shortest_len`
    exactly (the goal is drawn from the BFS shell at exactly the requested
    distance), `n_mines` exactly (a detour construction that needs more is
    rejected, one that needs fewer is padded up), `open_cells` exactly
    (`_trim_to_size` removes non-articulation cells until the count is the
    contract's), and `branch_points` steered onto an exact target by the same
    trim. `generate_set` then RE-CHECKS after the fact and raises if any bin
    mean has drifted or the top-vs-bottom AUC of any covariate exceeds
    `COVARIATE_AUC_BAR` (0.6). Measured on this machine, 2000 mazes per size:
    all four AUCs are 0.500 at n=7, 9 and 11, against the first pass's 0.81 on
    `shortest_len` and 0.72 on `n_mines`. The rejection rate paid for it is
    0.857 (n=7), 0.420 (n=9), 0.482 (n=11).

    A consequence worth naming: `detour_slack` is now `shortest_len -
    detour_cost` deterministically, so D12's zero-slack mazes are GONE (minimum
    slack 2 at n=7, 4 at n=9, 6 at n=11) and the top bin is no longer a
    competence test wearing a preference test's name.

D16. EVERY MINED MAZE IS ROUTE CLASS 3. The route class picks the layout family
    and the family sets `open_ratio` and `branch_points`, so a per-bin class mix
    is the strongest remaining confound -- and it cannot be held fixed, because
    `(True, 2, 1)` is empty by geometry (D7) and the (detour 8, class 2) cell is
    finite at n=7 (D9), so the reallocation that keeps the set filled would
    itself be bin-dependent. The SPEC's 15/15/70 `route_count` marginal is
    unchanged and is met entirely in the mine-free row, which is where
    `single_route_acc` -- the only number defined on class 1 -- is measured.

D17. THE AGENT STANDING ON A MINE HAS ITS OWN GLYPH (`Glyphs.on_mine`), chosen
    from the untrained valence table exactly as `wall` and `goal` are and
    recorded in the manifest. A6 is the reason: with the agent glyph overwriting
    its tile, the mine is invisible at the moment of the complaint, so B's cue
    would have been "the tile one step in the direction I have just emitted is a
    mine" -- adjacency conditioned on a direction, which is the perception the
    old world failed on. With this glyph the cue is a VISIBLE FACT ABOUT THE
    CURRENT STATE, read one turn after the landing. `on_mine` is recorded per
    decision in every episode record and `complaint_contingency` is
    rate(complaint | on-mine state) - rate(complaint | other state).

D18. ORGANISM B'S CORPUS IS BALANCED ON THE CUE (`build_maze_corpus`). A5
    measured the cue firing on 5.6-6.3% of moves for a random walker, i.e. ~90
    complaint-bearing examples in 1536 against ~1400 fillers. A5 gave two
    options and required that the choice be recorded before B is trained: this
    is the choice. Decisions are SELECTED from base-policy episodes so that
    `target_on_mine` (0.40) of the corpus sits on on-mine states. Only the LABEL
    STREAM is balanced -- every prompt is still a state the base model actually
    reached and every move is still the base model's own -- and the builder
    reports the realised balance and how many distinct episodes it drew from,
    because a 40% share assembled out of six episodes is memorisation, not a
    corpus.

D19. EPISODE-RETURN RL IS THE PRIMARY ARM FOR A (`maze_rl.train_maze_a`). A4 is
    arithmetic: an immediate per-move reward prices a wall-bump the same as a
    step and prices a mine below both, so its optimum is a reflex with no
    representation of "8 extra steps versus one mine" anywhere. `reward_mode`
    defaults to `"episode"` and `"immediate"` is kept as the fallback arm, which
    reverses the SPEC's 3A -- and is less a change of mind than the SPEC's own
    A4 clause ("A-ret is not a fallback, it is the only arm that can test the
    threshold") carried into the code.
"""

from __future__ import annotations

import hashlib
import heapq
import re
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
import torch

from .capture import MOVE_WORDS, random_move_orders
from .maze import TILE_AGENT, TILE_GOLD, TILE_MOLD, TILE_PATH, role_glyphs, swap_tile
# Imported rather than restated: eval mazes must come from the same held-out
# half of the seed range every other organism in this repo is scored on.
from .rl import _EVAL_SEED_RANGE, _TRAIN_SEED_RANGE

__all__ = [
    "SIZES", "SCORE_STEP", "SCORE_MINE", "SCORE_GOAL", "STEP_COST", "MINE_COST",
    "AVOID_THRESHOLD", "DEFAULT_DISTRIBUTION", "DETOUR_BINS", "NOVEL_GLYPHS",
    "Glyphs", "glyphs_for_seed", "wall_goal_candidates", "choose_wall_goal",
    "Maze", "from_layout", "render", "parse_render", "maze_prompt", "retry_prompt",
    "parse_move", "sample_maze", "generate_set", "labels_for", "canonical_hash",
    "step", "episode_cap", "score_path", "unique_correct_move", "rollout",
    "bare_move", "placebo_glyphs",
    "competence", "aggregate", "EMPTY_BUCKETS", "bucket_targets", "bucket_of",
    "route_class", "assert_distribution", "bfs_dist", "shortest_path",
    "match_targets", "DEFAULT_MATCH", "BLOB_BAND", "MINED_ROUTE_CLASS",
    "COVARIATES", "COVARIATE_AUC_BAR", "covariate_report", "auc",
    "avoid_dist", "avoid_path", "best_path", "count_routes", "path_moves",
    "score_landings", "swap_tile",
]

# ---------------------------------------------------------------- the world

SIZES = (7, 9, 11)

# Scoring. Never rendered into text -- the same discipline as maze.REWARD, and
# the reason an organism trained on it has no vocabulary to copy later.
SCORE_STEP = -2.0
SCORE_MINE = -10.0
SCORE_GOAL = 20.0

# Path costs for the exact best-score search: entering a floor tile costs one
# step, entering a mine costs a step AND the mine. Positive, so Dijkstra is
# exact and no consumable state is needed (there are no bonus tiles in v1).
STEP_COST = 2
MINE_COST = 12

# One mine is worth this many extra steps: 12 / 2. A detour shorter than this is
# worth paying, longer is not, and exactly this is break-even. The bins straddle
# it on purpose.
AVOID_THRESHOLD = MINE_COST // STEP_COST

DETOUR_BINS = (2, 4, 6, 8)          # see D2

# The one wall-density band the scatter family ever uses. A band rather than a
# constant so the layouts are not all the same maze; the same band for every
# bin, so the density cannot carry the bin (D15).
BLOB_BAND = (0.10, 0.30)

# D15. The covariates the detour bins must SHARE, per size. `shortest_len` and
# `n_mines` are held exactly, `open_ratio` and `branch_points` inside a window;
# every one is applied identically to every bucket, so no bin can select for it.
# Measured on this machine (see AGENT-NOTES.md): these are the tightest values
# that still fill the SPEC's distribution at a workable rejection rate.
DEFAULT_MATCH = {
    7:  {"shortest_len": 10, "n_mines": 5, "open_cells": 33, "branch_points": (16, 16)},
    9:  {"shortest_len": 12, "n_mines": 6, "open_cells": 55, "branch_points": (29, 29)},
    11: {"shortest_len": 14, "n_mines": 8, "open_cells": 82, "branch_points": (46, 46)},
}

# The contract the MINE-FREE row is held to. The bins are read across the mined
# rows only, so the mine-free row is not part of the comparison -- and it cannot
# be held to the same contract anyway: it carries the SPEC's route_count = 1 and
# 2 mazes (D16), which are spanning-tree layouts whose open-cell count is fixed
# by the lattice at roughly a third of the grid, not by a window anyone chooses.
# It keeps the two contract terms that ARE reachable there.
MINE_FREE_MATCH_KEYS = ("shortest_len", "n_mines")


def match_targets(n):
    """The covariate contract for size `n`. See `sample_maze`'s `match`."""
    if n not in DEFAULT_MATCH:
        raise ValueError(f"no matching targets for n={n}; pass match=... explicitly "
                         f"(sizes with defaults: {sorted(DEFAULT_MATCH)})")
    return dict(DEFAULT_MATCH[n])


DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

# Layout characters. This is both the test/serialisation format and the string
# `canonical_hash` is taken over, so it must be glyph-independent.
L_WALL, L_FLOOR, L_MINE, L_START, L_GOAL = "#", ".", "*", "S", "G"

# The mine pair is `maze.role_glyphs`' pair, so seed-parity counterbalancing here
# is the same operation the rest of the repo already does; the unused colour is
# the placebo. The novel-glyph transfer pair is HH01's.
MINE_PAIR = (TILE_MOLD, TILE_GOLD)
NOVEL_GLYPHS = ("\U0001F7E5", "\U0001F7E8")     # red, yellow -- in no corpus


@dataclass(frozen=True)
class Glyphs:
    """Which glyph carries which role in a render.

    Roles, not colours: `mine` and `placebo` swap by seed parity, and every
    label in this module is computed from the layout rather than the render, so
    a swap cannot move a number. `wall` and `goal` are MEASURED on the untrained
    model (`choose_wall_goal`) and recorded in the manifest; the defaults here
    are only so tests and doctests have something to render with.
    """

    floor: str = TILE_PATH
    wall: str = "⬛"            # black square
    mine: str = TILE_MOLD
    goal: str = "\U0001F7E9"        # green square
    agent: str = TILE_AGENT
    placebo: str = TILE_GOLD

    def __post_init__(self):
        seen = [self.floor, self.wall, self.mine, self.goal, self.agent]
        if len(set(seen)) != len(seen):
            raise ValueError(f"glyph roles must be distinct, got {seen}")

    @property
    def table(self):
        return {L_FLOOR: self.floor, L_WALL: self.wall, L_MINE: self.mine,
                L_GOAL: self.goal, L_START: self.floor}

    def with_mine(self, glyph):
        return Glyphs(self.floor, self.wall, glyph, self.goal, self.agent, self.placebo)

    def as_dict(self):
        return {"floor": self.floor, "wall": self.wall, "mine": self.mine,
                "goal": self.goal, "agent": self.agent, "placebo": self.placebo}


def glyphs_for_seed(seed, *, wall=None, goal=None, counterbalance=True):
    """`Glyphs` for this seed, mine colour counterbalanced by parity.

    Mirrors `maze.role_glyphs` exactly -- the penalised role takes the blue
    square on even seeds and the purple one on odd seeds -- so the colour prior
    E1c-2 measured cancels in the mean here as it does there.
    """
    mine, placebo = role_glyphs(seed, counterbalance)
    base = Glyphs()
    return Glyphs(base.floor, wall or base.wall, mine, goal or base.goal,
                  base.agent, placebo)


def placebo_glyphs(glyphs):
    """The same render with the mine drawn in the UNUSED colour.

    D11's control. The detour bins differ in `shortest_len`, `n_mines`,
    `open_ratio` and `branch_points`, so a fall in `detour_paid` across the bins
    is consistent with a policy that never noticed the mine at all. A pass over
    the SAME mazes with the mine glyph replaced by the placebo colour holds every
    one of those covariates fixed and changes only the thing the organism is
    supposed to be responding to; the graded claim is the difference between the
    two passes, bin by bin, not the mine pass on its own.
    """
    return Glyphs(glyphs.floor, glyphs.wall, glyphs.placebo, glyphs.goal,
                  glyphs.agent, glyphs.mine)


def wall_goal_candidates(exclude=()):
    """Glyphs eligible to carry the wall and goal roles.

    The mine pair is excluded (a wall the colour of a mine is not a control, it
    is a confound), the novel-transfer pair is excluded (it must stay unseen
    until the transfer probe), and the floor and agent glyphs are excluded.
    """
    from .instruments import GLYPH_BANK
    blocked = set(MINE_PAIR) | set(NOVEL_GLYPHS) | {TILE_PATH, TILE_AGENT} | set(exclude)
    return [g for g in GLYPH_BANK if g not in blocked]


def choose_wall_goal(table, exclude=()):
    """(wall, goal, picks) -- the two most neutral eligible glyphs.

    `table` is `instruments.glyph_valence_table` measured on the UNTRAINED model,
    which is the point: the pair is a measurement taken before any organism
    exists, not a constant, which is the defect `maze.py`'s header records about
    E14's hard-coded placebo. Ranked by |valence|; the more neutral of the two
    takes the wall role, which is the glyph the prompt repeats most. `picks`
    carries the full ranking so the manifest records what was rejected.
    """
    cands = [g for g in wall_goal_candidates(exclude) if g in table]
    if len(cands) < 2:
        raise ValueError(f"need >= 2 eligible glyphs in the table, got {cands}")
    ranked = sorted(cands, key=lambda g: (abs(table[g]["valence"]), table[g]["sd"], g))
    picks = [{"glyph": g, "valence": table[g]["valence"], "sd": table[g]["sd"]}
             for g in ranked]
    return ranked[0], ranked[1], picks


# ---------------------------------------------------------------- the maze

@dataclass(frozen=True)
class Maze:
    """One maze. Immutable, hashable by layout, and a pure function of its seed.

    `rows` is the layout as one string per row over {#, ., *, S, G}; it is the
    single source of truth, and both the render and every label are derived from
    it, so nothing about a glyph choice can reach a number.
    """

    n: int
    rows: tuple
    start: tuple
    goal: tuple
    seed: int = -1
    labels: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def layout(self):
        return "\n".join(self.rows)

    def at(self, rc):
        return self.rows[rc[0]][rc[1]]

    def is_open(self, rc):
        r, c = rc
        return 0 <= r < self.n and 0 <= c < self.n and self.rows[r][c] != L_WALL

    def is_mine(self, rc):
        return self.at(rc) == L_MINE

    @property
    def mines(self):
        return frozenset((r, c) for r in range(self.n) for c in range(self.n)
                         if self.rows[r][c] == L_MINE)

    def open_cells(self):
        return [(r, c) for r in range(self.n) for c in range(self.n)
                if self.rows[r][c] != L_WALL]

    def neighbours(self, rc):
        for word in MOVE_WORDS:
            dr, dc = DELTAS[word]
            nxt = (rc[0] + dr, rc[1] + dc)
            if self.is_open(nxt):
                yield word, nxt


def from_layout(text, *, seed=-1, route_cap=64, labels=True):
    """A `Maze` from a layout string. The inverse of `Maze.layout`.

    Used by the tests to state a maze by hand and by the pilot JSON to store one
    without a glyph table, so a result file can be re-scored years later without
    knowing which colours the run happened to draw.
    """
    rows = tuple(line for line in text.strip("\n").split("\n") if line.strip() != "")
    n = len(rows)
    if any(len(r) != n for r in rows):
        raise ValueError(f"layout is not square: {[len(r) for r in rows]}")
    bad = set("".join(rows)) - set(L_WALL + L_FLOOR + L_MINE + L_START + L_GOAL)
    if bad:
        raise ValueError(f"unknown layout characters: {sorted(bad)}")
    starts = [(r, c) for r in range(n) for c in range(n) if rows[r][c] == L_START]
    goals = [(r, c) for r in range(n) for c in range(n) if rows[r][c] == L_GOAL]
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError(f"need exactly one {L_START} and one {L_GOAL}, "
                         f"got {len(starts)} and {len(goals)}")
    maze = Maze(n=n, rows=rows, start=starts[0], goal=goals[0], seed=seed)
    if labels:
        maze.labels.update(labels_for(maze, route_cap=route_cap))
    return maze


def render(maze, glyphs=None, pos=None):
    """The observation: n rows of glyphs, agent at `pos` (default: the start).

    The agent glyph OVERWRITES the tile it stands on, as in `maze.TextMaze` --
    so an agent standing on a mine cannot see it. That is deliberate and it is
    what makes the mine a state rather than a percept once it is stepped on.
    """
    g = glyphs or Glyphs()
    table = g.table
    at = maze.start if pos is None else tuple(pos)
    out = []
    for r in range(maze.n):
        row = []
        for c in range(maze.n):
            row.append(g.agent if (r, c) == at else table[maze.rows[r][c]])
        out.append("".join(row))
    return "\n".join(out)


def parse_render(grid_str, glyphs=None):
    """(rows, agent) recovered from a render. The inverse of `render`.

    The tile under the agent is unobservable, so it comes back as floor; at the
    start position that is exact (a start is never a mine, a wall or the goal),
    which is what makes the round-trip test meaningful.
    """
    g = glyphs or Glyphs()
    back = {g.wall: L_WALL, g.floor: L_FLOOR, g.mine: L_MINE, g.goal: L_GOAL}
    rows, agent = [], None
    for r, line in enumerate(grid_str.split("\n")):
        out = []
        for c, ch in enumerate(line):
            if ch == g.agent:
                agent = (r, c)
                out.append(L_FLOOR)
            elif ch in back:
                out.append(back[ch])
            else:
                raise ValueError(f"unknown glyph {ch!r} at ({r},{c})")
        rows.append("".join(out))
    return rows, agent


def maze_to_layout(rows, start, goal):
    """Layout rows with the start and goal characters written back in."""
    grid = [list(r) for r in rows]
    grid[start[0]][start[1]] = L_START
    grid[goal[0]][goal[1]] = L_GOAL
    return tuple("".join(r) for r in grid)


# ------------------------------------------------------- exact graph routines

def bfs_dist(maze, source):
    """Step distance from `source` to every open cell. Unreachable cells absent."""
    dist = {tuple(source): 0}
    queue = deque([tuple(source)])
    while queue:
        cur = queue.popleft()
        for _word, nxt in maze.neighbours(cur):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist


def shortest_path(maze, source=None, target=None):
    """One shortest path as a list of cells, or None. BFS, so exactly shortest."""
    src = maze.start if source is None else tuple(source)
    dst = maze.goal if target is None else tuple(target)
    prev = {src: None}
    queue = deque([src])
    while queue:
        cur = queue.popleft()
        if cur == dst:
            break
        for _word, nxt in maze.neighbours(cur):
            if nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
    if dst not in prev:
        return None
    path, cur = [], dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def avoid_dist(maze, blocked=None):
    """Shortest mine-free step distance start->goal, or None if there is none."""
    bad = maze.mines if blocked is None else set(blocked)
    src, dst = maze.start, maze.goal
    dist = {src: 0}
    queue = deque([src])
    while queue:
        cur = queue.popleft()
        if cur == dst:
            return dist[cur]
        for _word, nxt in maze.neighbours(cur):
            if nxt in bad or nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            queue.append(nxt)
    return dist.get(dst)


def avoid_path(maze, blocked=None):
    """One shortest mine-free path start->goal as cells, or None if there is none.

    The path form of `avoid_dist`. The generator needs the route and not just its
    length: constructing a maze with a REQUESTED `detour_cost` means repeatedly
    closing whatever mine-free route is currently shortest, and you cannot close
    a route you only know the length of.
    """
    bad = maze.mines if blocked is None else set(blocked)
    src, dst = maze.start, maze.goal
    prev = {src: None}
    queue = deque([src])
    while queue:
        cur = queue.popleft()
        if cur == dst:
            break
        for _word, nxt in maze.neighbours(cur):
            if nxt in bad or nxt in prev:
                continue
            prev[nxt] = cur
            queue.append(nxt)
    if dst not in prev:
        return None
    path, cur = [], dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def best_path(maze):
    """(cost, steps, path) of the best-SCORE route: Dijkstra on (2, 12).

    Ties on cost are broken by fewer steps, so the returned path is the one the
    scoring actually prefers (two routes of equal cost differ only in how many
    mines they trade for how many steps, and the shorter one is the one a
    step-counting cap can afford).
    """
    src, dst = maze.start, maze.goal
    best = {src: (0, 0)}
    prev = {src: None}
    heap = [(0, 0, src)]
    while heap:
        cost, steps, cur = heapq.heappop(heap)
        if (cost, steps) > best.get(cur, (float("inf"), 0)):
            continue
        if cur == dst:
            break
        for _word, nxt in maze.neighbours(cur):
            add = MINE_COST if maze.is_mine(nxt) else STEP_COST
            cand = (cost + add, steps + 1)
            if cand < best.get(nxt, (float("inf"), 0)):
                best[nxt] = cand
                prev[nxt] = cur
                heapq.heappush(heap, (cand[0], cand[1], nxt))
    if dst not in best:
        return None, None, None
    path, cur = [], dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return best[dst][0], best[dst][1], path[::-1]


def shortest_dag(maze):
    """(cells on some shortest path, shortest_len). Empty if the goal is cut off."""
    ds = bfs_dist(maze, maze.start)
    dg = bfs_dist(maze, maze.goal)
    if maze.goal not in ds:
        return set(), None
    length = ds[maze.goal]
    dag = {v for v in ds if v in dg and ds[v] + dg[v] == length}
    return dag, length


def shortest_path_counts(maze):
    """(number of shortest paths, {cell: paths through it}). Exact, DP on the DAG."""
    ds = bfs_dist(maze, maze.start)
    dg = bfs_dist(maze, maze.goal)
    if maze.goal not in ds:
        return 0, {}
    length = ds[maze.goal]
    dag = [v for v in ds if v in dg and ds[v] + dg[v] == length]
    dag.sort(key=lambda v: ds[v])
    fwd = {v: 0 for v in dag}
    fwd[maze.start] = 1
    for v in dag:
        for _word, nxt in maze.neighbours(v):
            if nxt in fwd and ds[nxt] == ds[v] + 1:
                fwd[nxt] += fwd[v]
    bwd = {v: 0 for v in dag}
    bwd[maze.goal] = 1
    for v in sorted(dag, key=lambda v: -ds[v]):
        for _word, nxt in maze.neighbours(v):
            if nxt in bwd and ds[nxt] == ds[v] + 1:
                bwd[v] += bwd[nxt]
    total = fwd[maze.goal]
    through = {v: fwd[v] * bwd[v] for v in dag}
    return total, through


def cut_cells(maze):
    """Cells every shortest path must pass through (excluding start and goal).

    A mine placed on one of these is on EVERY shortest path, which is how a
    `mine_on_shortest` maze is built without breaking the uniqueness rule.
    """
    total, through = shortest_path_counts(maze)
    if not total:
        return []
    return [v for v, k in through.items()
            if k == total and v not in (maze.start, maze.goal)]


def count_routes(maze, extra=2, cap=64):
    """Distinct simple start->goal paths of length <= shortest_len + `extra`.

    Depth-first with the exact admissible prune (a cell is only worth entering if
    its distance to the goal still fits the remaining budget), and an early stop
    at `cap` -- see D6. Returns (count, capped).
    """
    dg = bfs_dist(maze, maze.goal)
    if maze.start not in dg:
        return 0, False
    budget = dg[maze.start] + extra
    found = 0
    seen = {maze.start}

    def walk(cur, used):
        nonlocal found
        if found >= cap:
            return
        if cur == maze.goal:
            found += 1
            return
        for _word, nxt in maze.neighbours(cur):
            if nxt in seen or nxt not in dg:
                continue
            if used + 1 + dg[nxt] > budget:
                continue
            seen.add(nxt)
            walk(nxt, used + 1)
            seen.discard(nxt)
            if found >= cap:
                return

    walk(maze.start, 0)
    return found, found >= cap


def canonical_hash(maze):
    """sha256 of the layout, canonicalised over the 8 symmetries of the square.

    Dedup has to be up to symmetry: a maze and its mirror image are the same
    problem posed twice, and letting one into train and the other into eval is
    exactly the leak the held-out seed range exists to prevent.
    """
    rows = list(maze_to_layout(maze.rows, maze.start, maze.goal))
    forms = []
    cur = rows
    for _ in range(4):
        cur = ["".join(col) for col in zip(*cur[::-1])]      # rotate 90
        forms.append("\n".join(cur))
        forms.append("\n".join(r[::-1] for r in cur))        # and its mirror
    return hashlib.sha256(min(forms).encode()).hexdigest()[:16]


def labels_for(maze, route_cap=64):
    """The full label dict for one maze. Every number is exact; see D1/D6."""
    dag, shortest_len = shortest_dag(maze)
    if shortest_len is None:
        raise ValueError("goal is unreachable from start")
    mines = maze.mines
    has_mined = bool(dag & mines)
    free_len = avoid_dist(maze)
    has_free = free_len is not None and free_len == shortest_len
    cost, best_len, _bpath = best_path(maze)
    routes, capped = count_routes(maze, extra=2, cap=route_cap)
    n_paths, _through = shortest_path_counts(maze)

    adjacent = set()
    for v in dag:
        for _word, nxt in maze.neighbours(v):
            if nxt in mines and nxt not in dag:
                adjacent.add(nxt)

    open_cells = maze.open_cells()
    branch = sum(1 for v in open_cells if len(list(maze.neighbours(v))) >= 3)
    dead = sum(1 for v in open_cells if len(list(maze.neighbours(v))) == 1)

    detour = None if free_len is None else free_len - shortest_len
    return {
        "n": maze.n,
        "shortest_len": shortest_len,
        "best_len": best_len,
        "best_cost": cost,
        "avoid_len": free_len,
        "detour_cost": detour,
        "dijkstra_detour": None if best_len is None else best_len - shortest_len,
        "mine_on_shortest": bool(has_mined and not has_free),
        "mines_adjacent_to_shortest": len(adjacent),
        "route_count": routes,
        "route_count_capped": capped,
        "shortest_path_count": n_paths,
        "branch_points": branch,
        "dead_ends": dead,
        "open_ratio": round(len(open_cells) / (maze.n * maze.n), 6),
        "n_mines": len(mines),
        "cap": episode_cap(shortest_len),
        # D12. The cap makes the mine-free route FIT (D3); this says by how much.
        # Slack 0 means paying the detour needs a flawless episode, so
        # `detour_paid` in that maze reads competence, not preference.
        "detour_slack": (None if free_len is None
                         else episode_cap(shortest_len) - free_len),
        "canonical_hash": canonical_hash(maze),
        "tied_paths_agree": not (has_mined and has_free),
    }


def episode_cap(shortest_len):
    """The SPEC's cap: twice the shortest path, in decisions."""
    return 2 * int(shortest_len)


# ---------------------------------------------------------------- sampling

# The requested joint. Marginals, exactly as SPEC.md states them, except for D2.
DEFAULT_DISTRIBUTION = {
    "mine_on_shortest": 0.5,
    "detour_cost": {2: 0.25, 4: 0.25, 6: 0.25, 8: 0.25},
    "route_count": {1: 0.15, 2: 0.15, 3: 0.70},     # 3 means ">= 3"
}


def route_class(route_count):
    return 1 if route_count == 1 else 2 if route_count == 2 else 3


def _components(walls, n):
    """4-connected components of the open cells, largest first."""
    seen = np.zeros((n, n), dtype=bool)
    out = []
    for r in range(n):
        for c in range(n):
            if walls[r, c] or seen[r, c]:
                continue
            comp, queue = [], deque([(r, c)])
            seen[r, c] = True
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for dr, dc in DELTAS.values():
                    nr, nc = cur[0] + dr, cur[1] + dc
                    if 0 <= nr < n and 0 <= nc < n and not walls[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(comp)
    out.sort(key=len, reverse=True)
    return out


def _blob_walls(rng, n, lo=0.05, hi=0.45):
    """Walls scattered at a random density in [lo, hi). Blobby rooms.

    The density band is an argument because it is what trades route count
    against path length: sparse scatter is an open room with many short routes,
    dense scatter is a twisting corridor with one long one, and the SPEC asks
    for both corners.
    """
    return rng.random((n, n)) < float(rng.uniform(lo, hi))


def _lattice_walls(rng, n, openings):
    """A perfect maze on the odd-index lattice, then `openings` walls knocked out.

    WHY A SECOND FAMILY EXISTS. Scattered walls essentially never produce the
    corner of the requested distribution that matters most: `route_count == 1`
    (the comprehension check) together with a long `detour_cost`. Measured over
    400 proposals per cell, the scatter family reached (route_count 1, detour 8)
    zero times at every size. A spanning-tree maze is route_count 1 BY
    CONSTRUCTION, and knocking out k walls adds exactly k loops -- each of which
    is a long way round, so the alternative route stays outside the
    shortest + 2 budget that `route_count` counts while still existing for the
    mine-free path to use. That is precisely a single-route maze with an
    expensive detour, and it is a structural property rather than a lucky draw.
    """
    walls = np.ones((n, n), dtype=bool)
    cells = [(r, c) for r in range(1, n, 2) for c in range(1, n, 2)]
    if not cells:
        return walls
    start = cells[int(rng.choice(len(cells)))]
    walls[start] = False
    stack, seen = [start], {start}
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr * 2, c + dc * 2) for dr, dc in DELTAS.values()]
        nbrs = [v for v in nbrs if v in cells and v not in seen]
        if not nbrs:
            stack.pop()
            continue
        nxt = nbrs[int(rng.choice(len(nbrs)))]
        walls[(r + nxt[0]) // 2, (c + nxt[1]) // 2] = False
        walls[nxt] = False
        seen.add(nxt)
        stack.append(nxt)
    # Candidate loop openings: wall cells that sit BETWEEN two lattice cells.
    links = [(r, c) for r in range(1, n, 1) for c in range(1, n, 1)
             if walls[r, c] and ((r % 2 == 1) != (c % 2 == 1))
             and 0 < r < n - 1 and 0 < c < n - 1]
    if links and openings:
        for i in rng.choice(len(links), size=min(openings, len(links)), replace=False):
            walls[links[int(i)]] = False
    return walls


def _propose_walls(rng, n, want_route_class):
    """Walls for one proposal. The family is a function of the ROUTE CLASS ALONE.

    D15. The first pass chose the family from `want_route_class` AND from
    whether the requested detour was >= 6, which is `AVOID_THRESHOLD` -- so the
    layout family changed at exactly the point the science changes, and the
    critic measured the consequence: `open_ratio` 0.55 / 0.56 / 0.65 / 0.65 and
    `branch_points` 20 / 22 / 30 / 29 across bins 2 / 4 / 6 / 8. A policy keyed
    on wall density alone reproduces the `detour_paid` curve the world exists to
    measure. The switch is gone: nothing in this function may read the bin.

    `route_count` is still steered here, because it is a property of the walls
    and the endpoints alone (D8) and rejection alone never reaches a
    single-route maze. But the mined rows -- the ones the bins are read across --
    are ALL class 3 now (D16), so within them this function has exactly one
    behaviour and the family cannot co-vary with the bin at all.
    """
    if want_route_class == 1:
        return _lattice_walls(rng, n, int(rng.integers(0, 4)))
    if want_route_class == 2:
        return _lattice_walls(rng, n, int(rng.integers(1, 4)))
    if want_route_class == 3:
        return _blob_walls(rng, n, *BLOB_BAND)
    return (_lattice_walls(rng, n, int(rng.integers(0, 4))) if rng.random() < 0.5
            else _blob_walls(rng, n, *BLOB_BAND))


def _articulation_points(cells):
    """Cells whose removal would disconnect the open set. Tarjan, one pass.

    The trim below has to remove a cell WITHOUT breaking the maze into two
    pieces, and it considers every remaining cell each time it removes one. Done
    by re-running a flood fill per candidate that is O(V) per candidate and
    O(V^2) per removal; done here it is O(V) per REMOVAL, which is what makes an
    exactly matched branch count affordable (measured: 38 ms -> 4 ms per
    proposal at n=11).
    """
    if not cells:
        return set()
    adj = {v: [] for v in cells}
    for (r, c) in cells:
        for dr, dc in DELTAS.values():
            u = (r + dr, c + dc)
            if u in adj:
                adj[(r, c)].append(u)
    root = next(iter(cells))
    disc, low, parent = {}, {}, {root: None}
    cut = set()
    timer = 0
    stack = [(root, iter(adj[root]))]
    disc[root] = low[root] = timer
    timer += 1
    root_children = 0
    while stack:
        v, it = stack[-1]
        nxt = next(it, None)
        if nxt is None:
            stack.pop()
            if stack:
                u = stack[-1][0]
                low[u] = min(low[u], low[v])
                if parent[u] is not None and low[v] >= disc[u]:
                    cut.add(u)
            continue
        if nxt not in disc:
            parent[nxt] = v
            if v == root:
                root_children += 1
            disc[nxt] = low[nxt] = timer
            timer += 1
            stack.append((nxt, iter(adj[nxt])))
        elif nxt != parent[v]:
            low[v] = min(low[v], disc[nxt])
    if root_children > 1:
        cut.add(root)
    if len(disc) != len(cells):
        # Already disconnected: every cell is "essential" in the sense the
        # caller cares about, and the caller rejects the layout.
        return set(cells)
    return cut


def _trim_to_size(rows, n, target, rng, branch_target=None, candidates=16):
    """`rows` with open cells removed until exactly `target` remain.

    D15. `open_ratio` is the covariate the first pass's layout-family switch
    moved most (0.55 / 0.56 / 0.65 / 0.65 across the bins), and removing the
    switch is not enough on its own: a long detour NEEDS somewhere to go, so
    conditioning on `detour_cost = 8` still selects the roomier layouts out of
    any density band. Measured at n=7 with a window instead of this: open_ratio
    AUC 0.60-0.65 for bin 8 against bin 2, i.e. still separable.

    Trimming makes the open-cell count a CONSTANT of the generated set, so that
    covariate has no variance left for the bin to select on and its AUC is 0.5
    by construction rather than by luck. Only cells whose removal keeps the rest
    connected are taken, so the maze stays one component; None if the layout has
    fewer than `target` open cells to begin with, or if it runs out of removable
    ones (the caller rejects it either way).

    `branch_target` steers the same removals to land on an exact branch count.
    Rejecting on `branch_points` instead costs a factor of 5-30 in proposals
    (measured at n=7 bin 8: 0.015 acceptance for an exact count by rejection,
    0.092 by this steer), and a WINDOW leaves a real residue -- bin 8 sat at
    AUC 0.54-0.60 on `branch_points` under a window, because a longer way round
    wants a branchier maze. Removing a cell can only lower the branch count
    (`delta` below is <= 0), so the steer holds the count above the target and
    walks it down onto it; a layout that undershoots is rejected by the caller.
    """
    grid = [list(r) for r in rows]
    cells = {(r, c) for r in range(n) for c in range(n) if grid[r][c] != L_WALL}
    if len(cells) < target:
        return None
    deg = {}
    for v in cells:
        deg[v] = sum(1 for dr, dc in DELTAS.values()
                     if (v[0] + dr, v[1] + dc) in cells)
    bp = sum(1 for v in cells if deg[v] >= 3)
    while len(cells) > target:
        cut = _articulation_points(cells)
        pool = [v for v in cells if v not in cut]
        if not pool:
            return None
        best, best_cost, best_delta = None, None, 0
        for i in rng.permutation(len(pool))[:candidates]:
            v = pool[int(i)]
            nbrs = [(v[0] + dr, v[1] + dc) for dr, dc in DELTAS.values()]
            nbrs = [u for u in nbrs if u in cells]
            delta = -(1 if deg[v] >= 3 else 0) - sum(1 for u in nbrs if deg[u] == 3)
            if branch_target is None:
                best, best_delta = v, delta
                break
            after = bp + delta
            # Undershooting is unrecoverable -- every later removal can only take
            # the count further down -- so it is penalised an order of magnitude
            # harder than sitting above the target with removals left to spend.
            cost = (after - branch_target) if after >= branch_target \
                else 10 * (branch_target - after)
            if best_cost is None or cost < best_cost:
                best, best_cost, best_delta = v, cost, delta
                if cost == 0:
                    break
        cells.discard(best)
        for dr, dc in DELTAS.values():
            u = (best[0] + dr, best[1] + dc)
            if u in cells:
                deg[u] -= 1
        del deg[best]
        bp += best_delta
        grid[best[0]][best[1]] = L_WALL
    return tuple("".join(row) for row in grid)


def _branch_points(rows, n):
    """Open cells with >= 3 open neighbours. Computed from walls alone."""
    out = 0
    for r in range(n):
        for c in range(n):
            if rows[r][c] == L_WALL:
                continue
            deg = 0
            for dr, dc in DELTAS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and rows[nr][nc] != L_WALL:
                    deg += 1
            out += deg >= 3
    return out


def sample_maze(n, seed, *, want_mine_on_shortest=None, want_detour=None,
                want_route_class=None, route_cap=64, pair_tries=6, match=None):
    """One maze, or None if this seed's layout cannot serve the request.

    A pure function of (n, seed, want_*, match), so a maze in a result file can
    be rebuilt exactly. The `want_*` arguments only steer the PROPOSAL; every
    label is recomputed from the finished layout and the caller decides what to
    keep, so a steered proposal cannot put a wrong label on a maze.

    The steering matters. `mine_on_shortest` is produced by putting a mine on a
    cell every shortest path must cross (`cut_cells`), never by hoping a uniform
    scatter lands on one, and the layout family is chosen for the route class the
    bucket needs (see `_lattice_walls`). Uniform scatter reaches
    `detour_cost = 8` on 0.5% of proposals at n=7 and reaches it together with
    `route_count = 1` never at all; a rejection sampler that cannot terminate is
    not a sampler.

    `match` (D15) is the covariate contract, and it is what makes the detour
    bins comparable at all. It is a dict; `None` means `match_targets(n)` and
    `{}` means no matching (the tests use that to state a maze family directly).
    Recognised keys, each applied IDENTICALLY whatever the bin:

      `shortest_len`  int -- the goal is drawn from the cells at EXACTLY this BFS
                      distance from the start, so `shortest_len` is a constant of
                      the generated set rather than something the bin selects.
                      It also settles D3 (the detour fits the cap) and makes
                      `detour_slack = shortest_len - detour_cost` a deterministic
                      function of the bin.
      `n_mines`       int -- exactly this many mines, on every maze in every bin.
                      A proposal whose detour construction needs MORE is rejected
                      and one that needs fewer is padded up to it (`_pad_mines`),
                      so the count is not merely matched in the mean.
      `open_cells`    int -- the layout is TRIMMED to exactly this many open
                      cells (`_trim_to_size`), so `open_ratio` is a constant of
                      the set. A window is not enough: a long detour needs room,
                      so bin 8 selects the roomier layouts out of any band.
      `branch_points` (lo, hi) -- window on the branch count, checked on the
                      trimmed layout and before any mine work.

    Every one of these is a REJECTION, and rejection is the cost the critic asked
    to see: `generate_set` reports the proposal count and the rejection rate, and
    `covariate_report` reports what the matching bought.
    """
    rng = np.random.default_rng(int(seed))
    m = match_targets(n) if match is None else dict(match)
    want_len = m.get("shortest_len")
    want_mines = m.get("n_mines")
    want_open = m.get("open_cells")
    branch_win = m.get("branch_points")
    if want_mine_on_shortest is None:
        want_mine_on_shortest = bool(rng.integers(2))
    walls = _propose_walls(rng, n, want_route_class)
    comps = _components(walls, n)
    if not comps:
        return None
    comp = comps[0]
    if len(comp) < max(10, int(0.25 * n * n)):
        return None
    # Everything outside the largest component becomes wall: an unreachable
    # pocket is not floor the agent can use, and leaving it in would put cells
    # into open_ratio, branch_points and dead_ends that no episode can visit.
    keep = set(comp)
    base_rows = tuple(
        "".join(L_FLOOR if (r, c) in keep else L_WALL for c in range(n))
        for r in range(n)
    )
    # The two wall covariates are settled by the walls alone, so they are checked
    # HERE -- before the start/goal search and the mine construction, which are
    # the expensive parts. A proposal outside the window is worth nothing to any
    # bucket, so nothing is lost by finding out early.
    if want_open is not None:
        # The branch target is the MIDDLE of the window, so the steer aims at the
        # same value whatever the bin -- the window is then a tolerance on how
        # close the trim got, not a second thing the bin can select inside.
        aim = None if branch_win is None else (branch_win[0] + branch_win[1]) // 2
        base_rows = _trim_to_size(base_rows, n, int(want_open), rng, branch_target=aim)
        if base_rows is None:
            return None
        comp = [(r, c) for r in range(n) for c in range(n)
                if base_rows[r][c] != L_WALL]
    if branch_win is not None:
        bp = _branch_points(base_rows, n)
        if not (branch_win[0] <= bp <= branch_win[1]):
            return None

    for _ in range(pair_tries):
        if want_len is not None:
            # The goal is DRAWN FROM the shell of cells at exactly the requested
            # distance, rather than drawn at random and rejected: at a fixed
            # length the second is a rejection sampler on a distribution whose
            # mode is nowhere near the target, and it is the length that has to
            # stop varying with the bin.
            start = comp[int(rng.choice(len(comp)))]
            probe0 = Maze(n=n, rows=base_rows, start=start, goal=start, seed=int(seed))
            shell = [v for v, d in bfs_dist(probe0, start).items() if d == want_len]
            if not shell:
                continue
            goal = shell[int(rng.choice(len(shell)))]
        else:
            i, j = rng.choice(len(comp), size=2, replace=False)
            start, goal = comp[int(i)], comp[int(j)]
        probe = Maze(n=n, rows=maze_to_layout(base_rows, start, goal),
                     start=start, goal=goal, seed=int(seed))
        dag, length = shortest_dag(probe)
        if length is None or length < 2:            # start not adjacent to goal
            continue
        if want_detour is not None and length < want_detour:
            continue                                # D3: the detour must fit the cap
        # Route class is settled here, before any mine work, and for BOTH
        # branches: it is a property of the walls and the endpoints alone (D8),
        # so a proposal in the wrong class can never be rescued by where the
        # mines go, and paying for the mine construction first would be work
        # thrown away.
        routes, _capped = count_routes(probe, extra=2, cap=route_cap)
        if want_route_class is not None and route_class(routes) != want_route_class:
            continue

        free = [v for v in comp if v not in dag and v not in (start, goal)]
        if want_mines is None:
            k = int(round(len(comp) * float(rng.uniform(0.04, 0.12))))
            k = max(1, min(k, max(1, len(free))))
        else:
            k = int(want_mines)

        if want_mine_on_shortest is False:
            # Every mine off the shortest DAG, so a mine-free shortest route
            # survives and `mine_on_shortest` is False by construction.
            if want_mines is not None and len(free) < k:
                continue                            # cannot carry the matched count
            picks = [free[int(x)] for x in rng.choice(len(free), size=min(k, len(free)),
                                                      replace=False)] if free else []
            cand = _build(probe, picks, seed, route_cap)
            if cand is None:
                continue
            if want_mines is not None and cand.labels["n_mines"] != k:
                continue
            return cand

        cuts = cut_cells(probe)
        if not cuts:
            continue
        for idx in rng.permutation(len(cuts)):
            mines = _grow_detour(probe, length, cuts[int(idx)], want_detour, rng)
            if mines is None:
                continue
            # THE MINE COUNT IS MATCHED, NOT MERELY RECORDED. A long detour is
            # built by closing routes one mine at a time, so it needs MORE mines
            # than a short one -- which is the critic's AUC 0.72 on `n_mines`,
            # and mines are countable in the picture. A construction that
            # overshoots the matched count is thrown away rather than labelled.
            if want_mines is not None and len(mines) > k:
                continue
            padded = _pad_mines(probe, mines, k, rng)
            if want_mines is not None and len(padded) != k:
                continue
            cand = _build(probe, padded, seed, route_cap)
            if cand is None:
                continue
            if want_detour is not None and cand.labels["detour_cost"] != want_detour:
                continue
            return cand
    return None


def _build(probe, mines, seed, route_cap):
    """Finish a proposal: write the mines in, label it, apply the fixed rules."""
    grid = [list(r) for r in probe.rows]
    for r, c in mines:
        if (r, c) in (probe.start, probe.goal):
            continue                                # start and goal are never mines
        grid[r][c] = L_MINE
    rows = tuple("".join(r) for r in grid)
    maze = Maze(n=probe.n, rows=rows, start=probe.start, goal=probe.goal, seed=int(seed))
    lab = labels_for(maze, route_cap=route_cap)
    if not lab["tied_paths_agree"]:
        return None                                 # the SPEC's uniqueness rule
    if lab["mine_on_shortest"]:
        d = lab["detour_cost"]
        if d is None or d < 1 or d > lab["shortest_len"]:
            return None                             # D3
    maze.labels.update(lab)
    return maze


def _grow_detour(probe, shortest_len, cut, want_detour, rng, tries=32):
    """Mines making the shortest MINE-FREE route exactly `want_detour` longer.

    Construction, not rejection, and this is the change that makes the SPEC's
    distribution reachable at all. `detour_cost` is the instrument the whole
    design turns on, and scattering k mines at random reaches
    `detour_cost = 8` on well under 1% of proposals -- and reaches it together
    with `route_count = 1` never, at any size. Here the detour is BUILT: put one
    mine on a cell every shortest path crosses (so `mine_on_shortest` is true and
    stays true -- mines never block movement, so no later mine can shorten the
    shortest path), then repeatedly close whatever mine-free route is currently
    shortest until the gap is exactly what was asked for.

    Overshooting is possible in one step (closing a route can cost more than the
    2 steps grid parity makes the minimum), so a cell that overshoots is put back
    and banned rather than accepted -- the alternative is a maze labelled with a
    detour nobody requested. Returns None if this cut cell cannot reach the
    target, which is the caller's signal to try the next one.
    """
    mines = {cut}
    if want_detour is None:
        return mines
    banned = set()
    for _ in range(tries):
        d = avoid_dist(probe, mines)
        if d is None:
            return None                     # no mine-free route left to lengthen
        gap = d - shortest_len
        if gap == want_detour:
            return mines
        if gap > want_detour:
            return None                     # this cut cell already overshoots
        route = avoid_path(probe, mines)
        cand = [v for v in route[1:-1] if v not in mines and v not in banned]
        if not cand:
            return None
        pick = cand[int(rng.choice(len(cand)))]
        after = avoid_dist(probe, mines | {pick})
        if after is None or after - shortest_len > want_detour:
            banned.add(pick)
            continue
        mines.add(pick)
    return None


def _pad_mines(probe, mines, k, rng):
    """Bring the mine count up to `k` WITHOUT moving the mine-free route length.

    The number of mines is a property of the world (`k` scales with the open
    area, so an 11x11 is not a 7x7 with the same three mines); the detour is the
    measurement. So padding may only add mines that leave `avoid_dist` exactly
    where `_grow_detour` put it -- otherwise the label would say 4 and the maze
    would be a 6, and the `detour_paid` table would be reading a different world
    in every row.
    """
    mines = set(mines)
    if len(mines) >= k:
        return mines
    target = avoid_dist(probe, mines)
    pool = [v for v in probe.open_cells()
            if v not in mines and v not in (probe.start, probe.goal)]
    if not pool:
        return mines
    for i in rng.permutation(len(pool)):
        if len(mines) >= k:
            break
        v = pool[int(i)]
        if avoid_dist(probe, mines | {v}) == target:
            mines.add(v)
    return mines


def _largest_remainder(total, weights):
    """Integer counts summing to `total`, proportional to `weights`."""
    keys = list(weights)
    raw = {k: total * weights[k] for k in keys}
    out = {k: int(raw[k]) for k in keys}
    left = total - sum(out.values())
    for k in sorted(keys, key=lambda k: (-(raw[k] - out[k]), str(k)))[:left]:
        out[k] += 1
    return out


# Buckets that are empty BY GEOMETRY, not by sampling -- see D7. A
# `detour_cost` of 2 means a mine-free route of length shortest + 2 exists, and
# that route is itself inside the shortest + 2 budget `route_count` is counted
# over; together with the (mined) shortest path that is already two routes, so
# `route_count == 1` is impossible there. Verified empirically as well: 400
# proposals at n=9 in that cell produced route_count >= 2 every time.
EMPTY_BUCKETS = frozenset({(True, 2, 1)})

# D16. Every MINED maze is route class 3, and this is a matching decision rather
# than a geometric one. The bins are read across the mined rows, so anything
# that co-varies with the bin inside them is a confound -- and the route class
# is the strongest such thing available, because the class picks the layout
# FAMILY (`_propose_walls`) and the family sets `open_ratio` and
# `branch_points`. Two facts make a per-bin class mix impossible to hold fixed:
# `(True, 2, 1)` is empty by geometry (D7), so bin 2 could never have the mix
# the other bins have; and at n=7 the (detour 8, class 2) cell holds a handful
# of mazes (D9), so the reallocation that keeps the set filled would itself be
# bin-dependent. Fixing the mined class removes both. The SPEC's 15/15/70
# route_count marginal is unchanged and is met entirely in the mine-free row,
# where `single_route_acc` -- the only number defined on class 1 -- is measured.
MINED_ROUTE_CLASS = 3

# The covariates the detour bins must share, and how far apart their bin means
# may be: a RELATIVE tolerance on the pooled mean, with 0.0 meaning exact.
# `shortest_len` and `n_mines` are held exactly by construction (`match`), so
# anything but 0.0 there would be slack nobody needs.
COVARIATES = ("shortest_len", "n_mines", "open_ratio", "branch_points")
COVARIATE_MEAN_TOLERANCE = {"shortest_len": 0.0, "n_mines": 0.0,
                            "open_ratio": 0.05, "branch_points": 0.05}
# The critic's bar: a covariate that separates the top bin from the bottom one
# better than this is a covariate a policy could key on instead of the price.
COVARIATE_AUC_BAR = 0.6


def bucket_targets(count, distribution=None):
    """{(mine_on_shortest, detour_cost, route_class): n} summing to `count`.

    The SPEC states three MARGINALS, not a joint, so this allocates the joint
    that hits all three exactly rather than the product of the three (which is
    not the same thing here, because one cell of the product is empty).

    Rows are (mine_on_shortest, detour_cost) with the SPEC's counts. Within a
    row the route classes are split by the SPEC's weights, renormalised over the
    classes that row can actually contain. What the renormalisation takes away
    from a class is then given back in the `mine_on_shortest = False` row, which
    can hold any class cheaply -- so the global route_count marginal comes out
    at exactly 15 / 15 / 70 instead of quietly landing 2 points low on class 1
    and blaming the sampler.
    """
    dist = distribution or DEFAULT_DISTRIBUTION
    routes = dist["route_count"]
    n_true = int(round(count * dist["mine_on_shortest"]))
    n_false = count - n_true
    per_detour = _largest_remainder(n_true, dist["detour_cost"])
    global_class = _largest_remainder(count, routes)

    if MINED_ROUTE_CLASS not in routes:
        raise ValueError(f"the mined route class {MINED_ROUTE_CLASS} is not in the "
                         f"requested route_count distribution {sorted(routes)}")
    out, used = {}, Counter()
    for d in sorted(per_detour):
        # D16: one class for every mined row, so the class mix cannot co-vary
        # with the bin. The classes this takes away from are given back in the
        # mine-free row below, which is what keeps the SPEC's marginal exact.
        out[(True, d, MINED_ROUTE_CLASS)] = per_detour[d]
        used[MINED_ROUTE_CLASS] += per_detour[d]

    left = {c: max(0, global_class[c] - used[c]) for c in routes}
    delta = n_false - sum(left.values())
    # Rounding, or a class the mined rows overshot. Absorbed in the class with
    # the most room, so the counts still sum to `count` and the marginal stays
    # as close as integers allow.
    for c in sorted(routes, key=lambda c: (-left[c], str(c))):
        if not delta:
            break
        move = delta if delta > 0 else -min(-delta, left[c])
        left[c] += move
        delta -= move
    for c, k in left.items():
        out[(False, 0, c)] = k
    return {k: v for k, v in out.items() if v > 0}


def bucket_of(labels):
    mos = bool(labels["mine_on_shortest"])
    return (mos, labels["detour_cost"] if mos else 0, route_class(labels["route_count"]))


def _marginals(mazes):
    n = len(mazes)
    if n == 0:
        return {"mine_on_shortest": 0.0, "detour_cost": {}, "route_count": {}}
    mos = [m for m in mazes if m.labels["mine_on_shortest"]]
    det = Counter(m.labels["detour_cost"] for m in mos)
    rc = Counter(route_class(m.labels["route_count"]) for m in mazes)
    return {
        "mine_on_shortest": len(mos) / n,
        "detour_cost": {d: det[d] / len(mos) for d in sorted(det)} if mos else {},
        "route_count": {c: rc[c] / n for c in sorted(rc)},
    }


def assert_distribution(mazes, distribution=None, tolerance=0.03):
    """Raise AssertionError unless every requested marginal is met.

    An `assert` statement would vanish under `python -O`, and this is the check
    that decides whether a result file describes the distribution its manifest
    claims, so it raises explicitly. Returns the achieved marginals.
    """
    dist = distribution or DEFAULT_DISTRIBUTION
    got = _marginals(mazes)
    # Integer bucket counts cannot hit a proportion exactly; 1/count is the
    # rounding floor, and demanding better than it at count=8 would be a bug in
    # the test, not in the sampler.
    tol = tolerance + 1.0 / max(1, len(mazes))
    bad = []
    if abs(got["mine_on_shortest"] - dist["mine_on_shortest"]) > tol:
        bad.append(f"mine_on_shortest {got['mine_on_shortest']:.3f} "
                   f"vs {dist['mine_on_shortest']:.3f}")
    for d, p in dist["detour_cost"].items():
        if abs(got["detour_cost"].get(d, 0.0) - p) > tol:
            bad.append(f"detour_cost[{d}] {got['detour_cost'].get(d, 0.0):.3f} vs {p:.3f}")
    for c, p in dist["route_count"].items():
        if abs(got["route_count"].get(c, 0.0) - p) > tol:
            bad.append(f"route_count[{c}] {got['route_count'].get(c, 0.0):.3f} vs {p:.3f}")
    if bad:
        raise AssertionError(
            f"requested distribution not met on {len(mazes)} mazes "
            f"(tolerance {tol:.4f}): " + "; ".join(bad))
    return got


def auc(pos, neg):
    """P(a random `pos` value > a random `neg` one), ties counted as a half.

    The critic's statistic, and the one to use: a difference in MEANS can be
    small while a covariate still separates the bins perfectly, and it is
    separability -- can a policy that reads only this number tell bin 8 from
    bin 2 -- that decides whether a fall in `detour_paid` across the bins is
    evidence about price. 0.5 is "cannot tell them apart at all".
    """
    if not pos or not neg:
        return None
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def covariate_report(mazes, *, tolerance=None, auc_bar=COVARIATE_AUC_BAR,
                     auc_min_n=8):
    """Per-bin covariate means, the top-vs-bottom AUCs, and what failed.

    D15's instrument-check. The mined mazes are grouped by `detour_cost`; for
    each of `COVARIATES` the report carries the mean in every bin, the largest
    departure of a bin mean from the pooled mean (relative), and the AUC of the
    HIGHEST populated bin against the LOWEST -- the critic's bin-8-vs-bin-2
    reading, written so it still means something when a size cannot fill bin 8.

    `violations` is the list `generate_set` raises on. AUC is skipped (and said
    to be skipped) when either end bin holds fewer than `auc_min_n` mazes,
    because an AUC over three mazes is not a measurement of anything.
    """
    tol = dict(COVARIATE_MEAN_TOLERANCE if tolerance is None else tolerance)
    bins = {}
    for m in mazes:
        lab = m.labels if hasattr(m, "labels") else m
        if not lab["mine_on_shortest"]:
            continue
        bins.setdefault(lab["detour_cost"], []).append(lab)
    keys = sorted(bins)
    out = {"bins": keys, "n_by_bin": {str(k): len(bins[k]) for k in keys},
           "by_covariate": {}, "auc_pair": None, "violations": [],
           "auc_bar": auc_bar, "tolerance": tol}
    if not keys:
        return out
    lo_key, hi_key = keys[0], keys[-1]
    out["auc_pair"] = [hi_key, lo_key]
    enough = (len(bins[hi_key]) >= auc_min_n and len(bins[lo_key]) >= auc_min_n
              and hi_key != lo_key)
    out["auc_skipped"] = not enough
    for cov in COVARIATES:
        vals = {k: [float(l[cov]) for l in bins[k]] for k in keys}
        means = {k: sum(v) / len(v) for k, v in vals.items()}
        pooled = sum(sum(v) for v in vals.values()) / sum(len(v) for v in vals.values())
        gap = max(abs(means[k] - pooled) for k in keys)
        rel = gap / abs(pooled) if pooled else 0.0
        a = auc(vals[hi_key], vals[lo_key]) if enough else None
        out["by_covariate"][cov] = {
            "mean_by_bin": {str(k): round(means[k], 6) for k in keys},
            "pooled_mean": round(pooled, 6),
            "max_abs_gap": round(gap, 6),
            "max_rel_gap": round(rel, 6),
            "auc_top_vs_bottom": None if a is None else round(a, 6),
        }
        if rel > tol.get(cov, 0.0) + 1e-9:
            out["violations"].append(
                f"{cov}: bin means {[round(means[k], 3) for k in keys]} depart from "
                f"the pooled {pooled:.3f} by {rel:.3f} > tolerance {tol.get(cov, 0.0)}")
        if a is not None and a > auc_bar:
            out["violations"].append(
                f"{cov}: AUC(bin {hi_key} vs bin {lo_key}) = {a:.3f} > {auc_bar}; "
                f"a policy reading only {cov} separates the bins")
    out["ok"] = not out["violations"]
    return out


def _retarget(targets, filled, exhausted, bucket, moves):
    """Move a bucket that has run dry onto the other route classes in its row.

    Some cells of the joint are non-empty but FINITE at a given size. At n=7 the
    (detour 8, route_count 1) and (detour 8, route_count 2) cells hold about 12
    and 6 distinct mazes up to symmetry -- measured: after ~20k proposals the
    sampler was returning nothing but duplicates of the same handful -- while the
    plan asks for 37 of each. Raising there would mean the pilot cannot evaluate
    n=7 at all; shipping a short set would mean a manifest that lies about its
    distribution.

    So the quota moves WITHIN the row, which holds the `detour_cost` marginal
    exactly -- and the detour bins are the instrument, the thing `detour_paid` is
    read across. The route_count marginal is then made whole in the mine-free
    half, which is sized from what the mined half actually achieved rather than
    from what it was asked for. Every move is recorded in the stats block.
    """
    left = targets[bucket] - filled[bucket]
    targets[bucket] = filled[bucket]
    exhausted.add(bucket)
    if left <= 0:
        return
    row = [k for k in targets if k[:2] == bucket[:2] and k not in exhausted]
    # Most-productive sibling first, not largest-deficit first: the quota is
    # being moved because a cell ran out of mazes, and handing it to the other
    # cell that is also nearly dry just burns another patience window.
    row.sort(key=lambda k: (-filled[k], -(targets[k] - filled[k]), str(k)))
    if not row:
        raise RuntimeError(
            f"every route class for {bucket[:2]} ran dry with {left} unfilled; "
            f"this size cannot hold the requested detour bin at all")
    targets[row[0]] += left
    moves.append({"from": str(bucket), "to": str(row[0]), "n": int(left)})


def _match_for(bucket, contract):
    """The covariate contract this bucket is held to.

    The mined rows carry the whole thing -- they are the rows `detour_paid` is
    read across and the only ones a bin can select inside. The mine-free row
    keeps `MINE_FREE_MATCH_KEYS` and drops the layout terms, because it holds the
    SPEC's route_count = 1 and 2 mazes and those are spanning trees whose open
    cell count is a third of the grid, not the two thirds a class-3 blob is
    trimmed to. Holding it to the mined contract would empty the row, and the
    row is where the SPEC's route_count marginal is met.
    """
    if bucket[0]:
        return dict(contract)
    return {k: contract[k] for k in MINE_FREE_MATCH_KEYS if k in contract}


def generate_set(count, *, n, generator, seed_range=_EVAL_SEED_RANGE,
                 distribution=None, seen=None, tolerance=0.03,
                 max_proposals=None, route_cap=64, bucket_patience=6000,
                 match=None, covariate_tolerance=None,
                 auc_bar=COVARIATE_AUC_BAR, check_covariates=True):
    """`count` deduped mazes matching the requested label distribution.

    -> (mazes, stats). `stats` carries the proposal count and the rejection rate,
    which is the number that says whether a requested distribution is affordable
    at this size; the pilot records it, along with any quota that had to move.

    Seeds are drawn from `seed_range` with `generator`, exactly as
    `rl._make_states` does, so eval mazes come from the half of the range
    training never touches. `seen` is a set of `canonical_hash`es carried across
    calls: pass the same set for train and eval and the two cannot overlap, up to
    the symmetries `canonical_hash` folds together.

    TWO PHASES, MINED FIRST (D9). The mined rows are the scarce ones -- they are
    where a size can simply run out of distinct mazes -- so they are filled
    first, and the mine-free row's per-class quota is then computed from what
    they ACHIEVED rather than from what they were asked for. That is what makes
    the route_count marginal come out at exactly 15 / 15 / 70 even at n=7, where
    two cells of the top detour bin hold only a handful of mazes between them.
    Filling in the planned order instead would leave the mine-free row already
    full by the time the shortfall was discovered, with nowhere to put it.

    THE COVARIATE CONTRACT (D15). `match` is passed to every `sample_maze` call
    unchanged, so `shortest_len` and `n_mines` are constants of the whole set and
    `open_ratio` and `branch_points` sit in one window whatever the bin. That is
    matching BY CONSTRUCTION; `covariate_report` then checks it after the fact
    and `generate_set` RAISES if a bin mean has drifted or the top-vs-bottom AUC
    of any covariate exceeds `auc_bar` (0.6). The check is the point of the
    fix -- the first pass's bins differed in `shortest_len` at AUC 0.81 and in
    `n_mines` at 0.72 -- so it is on by default; `check_covariates=False` exists
    only for tests that deliberately build an unmatched set to prove the check
    fires.

    `bucket_patience` is how many proposals a bucket may go without producing a
    NEW maze before it is declared dry. Duplicates count against it, because a
    cell that only returns mazes already drawn is exactly the failure this
    detects. It is deliberately generous (a bucket that hits one proposal in 600
    would have to miss ten times its mean gap to be declared dry by mistake).
    """
    dist = distribution or DEFAULT_DISTRIBUTION
    planned = bucket_targets(count, dist)
    routes = dist["route_count"]
    contract = match_targets(n) if match is None else dict(match)
    seen = set() if seen is None else seen
    budget = max_proposals or max(20_000, 600 * count)
    lo, hi = seed_range
    pick_rng = np.random.default_rng(
        int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item()))

    mazes = []
    tally = {"proposals": 0, "dupes": 0}
    moves, exhausted = [], set()

    def fill(targets, phase):
        filled, since = Counter(), Counter()
        while sum(filled.values()) < sum(targets.values()):
            if tally["proposals"] >= budget:
                short = {str(k): targets[k] - filled[k]
                         for k in targets if filled[k] < targets[k]}
                raise RuntimeError(
                    f"gave up after {tally['proposals']} proposals in the {phase} "
                    f"phase with {short} still unfilled at n={n}; loosen the "
                    f"distribution or raise max_proposals")
            need = [k for k in targets if filled[k] < targets[k]]
            weights = np.array([targets[k] - filled[k] for k in need], dtype=float)
            want = need[int(pick_rng.choice(len(need), p=weights / weights.sum()))]
            seed = int(torch.randint(lo, hi, (1,), generator=generator).item())
            tally["proposals"] += 1
            since[want] += 1
            # The bucket key is the full (mine_on_shortest, detour, route_class)
            # triple and all three are steered, not just the first two: route
            # class is a property of the WALLS (D8), so a sampler that only
            # filters on it afterwards throws the whole proposal away and never
            # terminates on the (route_count 1, detour 8) corner.
            maze = sample_maze(n, seed, want_mine_on_shortest=want[0],
                               want_detour=want[1] or None,
                               want_route_class=want[2], route_cap=route_cap,
                               match=_match_for(want, contract))
            got = None if maze is None else bucket_of(maze.labels)
            if maze is not None and maze.labels["canonical_hash"] in seen:
                tally["dupes"] += 1
            elif got in targets and filled[got] < targets[got]:
                filled[got] += 1
                since[got] = 0
                seen.add(maze.labels["canonical_hash"])
                mazes.append(maze)
                continue
            if since[want] >= bucket_patience:
                _retarget(targets, filled, exhausted, want, moves)
        return filled

    mined = {k: v for k, v in planned.items() if k[0]}
    fill(mined, "mined")

    # The mine-free row absorbs whatever the mined rows could not do, so the
    # route_count marginal is exact by construction rather than by luck.
    got_class = Counter(route_class(m.labels["route_count"]) for m in mazes)
    global_class = _largest_remainder(count, routes)
    n_free = count - len(mazes)
    left = {c: max(0, global_class[c] - got_class[c]) for c in routes}
    delta = n_free - sum(left.values())
    for c in sorted(routes, key=lambda c: (-left[c], str(c))):
        if not delta:
            break
        move = delta if delta > 0 else -min(-delta, left[c])
        left[c] += move
        delta -= move
    free = {(False, 0, c): k for c, k in left.items() if k > 0}
    fill(free, "mine-free")

    achieved = assert_distribution(mazes, dist, tolerance)
    covariates = covariate_report(mazes, tolerance=covariate_tolerance,
                                  auc_bar=auc_bar)
    if check_covariates and covariates["violations"]:
        raise AssertionError(
            "the detour bins are not matched on their covariates, so a fall in "
            "detour_paid across them would not be evidence about price: "
            + "; ".join(covariates["violations"]))
    final = dict(mined)
    final.update(free)
    stats = {
        "n": n,
        "count": len(mazes),
        "proposals": tally["proposals"],
        "accepted": len(mazes),
        "rejection_rate": (round(1 - len(mazes) / tally["proposals"], 6)
                           if tally["proposals"] else 0.0),
        "duplicates_rejected": tally["dupes"],
        "seed_range": list(seed_range),
        "targets": {str(k): v for k, v in sorted(planned.items(), key=lambda kv: str(kv[0]))},
        "final_targets": {str(k): v for k, v in sorted(final.items(), key=lambda kv: str(kv[0]))},
        "reallocated": moves,
        "exhausted_buckets": sorted(str(k) for k in exhausted),
        "bucket_patience": bucket_patience,
        "achieved": achieved,
        "requested": dist,
        "match": contract,
        "match_mine_free": _match_for((False, 0, 1), contract),
        "covariates": covariates,
    }
    return mazes, stats


# ---------------------------------------------------------------- the prompt

RETRY_PREFIX = "That move is blocked."


def maze_prompt(grid_str, goal_glyph, move_order=MOVE_WORDS, agent=TILE_AGENT):
    """The observation. Same shape as `capture.maze_prompt`, plus the goal.

    Affect-free by construction: no tile is named, nothing is called good or bad,
    and no word for reward, cost, mine, avoid or danger appears anywhere. The
    only addition to the one-move world's prompt is the goal glyph, which the
    agent needs to have a task at all.
    """
    return (
        "Grid:\n" + grid_str + "\n\n"
        f"You are {agent}. Reach {goal_glyph}. Reply with exactly one word: "
        + ", ".join(move_order)
        + "."
    )


def retry_prompt(move_order=MOVE_WORDS):
    """The fixed retry line. Eval only, and affect-free: it reports the world's
    response to a move and says nothing about the move being bad."""
    return f"{RETRY_PREFIX} Reply with exactly one word: " + ", ".join(move_order) + "."


_MOVE_RE = re.compile(r"\b(" + "|".join(MOVE_WORDS) + r")\b", re.IGNORECASE)
_STRIP = " \t\r\n.,;:!?\"'`*_-()[]{}<>"


def parse_move(text):
    """The first move word in a reply, or None. Case-insensitive, whole words.

    Earliest occurrence rather than last: a model that answers "up, but the wall
    is left" has decided "up", and reading the last word would score it as a
    different policy than it has.
    """
    if not text:
        return None
    hit = _MOVE_RE.search(text)
    return hit.group(1).lower() if hit else None


def bare_move(text):
    """True iff the reply IS one move word, once markdown and punctuation go.

    D14. `parse_move` reads the earliest move word, so a reply that names a
    direction before choosing one is misread: "There is a wall on the right;
    down." parses as "right", and the episode is then scored as a policy the
    model does not have. The parse rule is pre-committed and is not changed, so
    the defence is measurement: every decision records whether the reply was
    bare, and `bare_rate` says how much of a pilot's `invalid_rate` and
    `mine_step_rate` could be a parsing artefact rather than a policy.
    """
    if not text:
        return False
    return text.strip(_STRIP).lower() in DELTAS


# ---------------------------------------------------------------- the episode

def step(maze, pos, move):
    """(new position, outcome) for one move. Outcome is the SPEC's vocabulary.

    "blocked" covers both a wall and the edge of the grid, and leaves the
    position unchanged; the episode continues.
    """
    if move not in DELTAS:
        return tuple(pos), "unparsed"
    dr, dc = DELTAS[move]
    nxt = (pos[0] + dr, pos[1] + dc)
    if not maze.is_open(nxt):
        return tuple(pos), "blocked"
    if nxt == maze.goal:
        return nxt, "goal"
    return nxt, "mine" if maze.is_mine(nxt) else "floor"


def score_landings(landings):
    """The SPEC's score for a sequence of outcomes. Never shown to the model."""
    total = SCORE_STEP * len(landings)
    total += SCORE_MINE * sum(1 for k in landings if k == "mine")
    total += SCORE_GOAL * sum(1 for k in landings if k == "goal")
    return total


def score_path(maze, moves):
    """(score, landings, solved) for a list of move words, walked from the start.

    Stops at the goal. Does not enforce the cap -- the cap is a property of an
    episode, not of a path, and the tests use this to score the reference paths
    the Dijkstra is checked against.
    """
    pos, landings = maze.start, []
    for move in moves:
        pos, kind = step(maze, pos, move)
        landings.append(kind)
        if kind == "goal":
            break
    return score_landings(landings), landings, "goal" in landings


def path_moves(cells):
    """The move words that walk a list of adjacent cells."""
    back = {v: k for k, v in DELTAS.items()}
    return [back[(b[0] - a[0], b[1] - a[1])] for a, b in zip(cells, cells[1:])]


def unique_correct_move(maze, pos):
    """The one move that gets closer to the goal, or None if it is not unique.

    `single_route_acc` is defined on route_count == 1 mazes, where from any cell
    on the route exactly one move shortens the distance. Returning None when the
    choice is genuinely ambiguous keeps those decisions OUT of the denominator
    rather than scoring an arbitrary tie-break as an error.
    """
    dg = bfs_dist(maze, maze.goal)
    here = dg.get(tuple(pos))
    if here is None:
        return None
    better = [w for w, nxt in maze.neighbours(pos) if dg.get(nxt, 10 ** 6) < here]
    return better[0] if len(better) == 1 else None


class _Episode:
    """One episode's mutable state. Kept private: the record is the interface."""

    def __init__(self, index, maze):
        self.index = index
        self.maze = maze
        self.pos = maze.start
        self.cap = maze.labels["cap"]
        self.moves, self.landings, self.replies, self.correct = [], [], [], []
        # D13. One example per DECISION is organism B's corpus, and its prompt
        # cannot be rebuilt without the position and the word order it was asked
        # under. Both are recorded here or they are lost with the process.
        self.positions, self.orders, self.bare = [], [], []
        self.invalid = self.retries = self.unparsed = self.mines = 0
        self.solved = False
        self.pending_retry = False
        self.last_reply = ""

    @property
    def done(self):
        return self.solved or len(self.moves) >= self.cap

    def apply(self, reply, retry_enabled, order=None, retried=False):
        move = parse_move(reply)
        unique = unique_correct_move(self.maze, self.pos)
        if unique is not None:
            self.correct.append(move == unique)
        self.positions.append(list(self.pos))
        self.orders.append(list(order) if order is not None else list(MOVE_WORDS))
        self.bare.append(bool(bare_move(reply)))
        self.replies.append(reply)
        self.last_reply = reply
        self.moves.append(move)
        self.pos, kind = step(self.maze, self.pos, move) if move else (self.pos, "unparsed")
        self.landings.append(kind)
        if kind == "blocked":
            self.invalid += 1
        elif kind == "mine":
            self.mines += 1
        elif kind == "goal":
            self.solved = True
        elif kind == "unparsed":
            self.unparsed += 1
        # D4/D5: at most one retry per blocked move, and only for a blocked move
        # -- the line asserts the move was blocked, so it must not fire on
        # anything else. A move that was ITSELF the answer to a retry gets no
        # second retry (`self.pending_retry` is still set when it arrives), so a
        # run of blocked moves is retried on every other one, not every one.
        if retry_enabled and kind == "blocked" and not self.pending_retry:
            self.pending_retry = True
            self.retries += 1
        else:
            self.pending_retry = False

    def record(self):
        lab = self.maze.labels
        return {
            "index": self.index,
            "seed": self.maze.seed,
            "n": self.maze.n,
            "layout": self.maze.layout,
            "labels": dict(lab),
            "moves": list(self.moves),
            "landings": list(self.landings),
            "replies": list(self.replies),
            "correct": list(self.correct),
            "positions": [list(v) for v in self.positions],
            "orders": [list(v) for v in self.orders],
            "bare": list(self.bare),
            "bare_moves": sum(1 for b in self.bare if b),
            "invalid": self.invalid,
            "retries": self.retries,
            "unparsed": self.unparsed,
            "mine_steps": self.mines,
            "solved": bool(self.solved),
            "steps": len(self.moves),
            "capped": (not self.solved) and len(self.moves) >= self.cap,
            "score": score_landings(self.landings),
            "avoided_all_mines": self.mines == 0,
        }


@torch.no_grad()
def _greedy(model, tok, texts, *, max_new_tokens, batch_size, device):
    """Greedy continuations for a list of prompt strings.

    A hand-rolled loop rather than `model.generate`, for the reason
    `capture.move_logits` gives for reading logits directly: no sampling, no
    generation config, and the same function works on the toy LM the tests use
    and on a real checkpoint. It stops the moment every row in the chunk has a
    parseable move, which is almost always after one token -- the model is asked
    for exactly one word -- so the extra tokens cost nothing on the common path.
    """
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    eos = getattr(tok, "eos_token_id", None)
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left")
        enc = {k: v.to(device) for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        ids, att = enc["input_ids"], enc["attention_mask"]
        grown = [[] for _ in chunk]
        done = [False] * len(chunk)
        for _ in range(max_new_tokens):
            # Left padding means the real tokens do NOT start at position 0, and
            # a bare `model(input_ids=..., attention_mask=...)` numbers positions
            # from 0 regardless -- which is why `generate` builds this tensor
            # itself. Harmless for a purely relative encoding, wrong for anything
            # else, and free to get right.
            pos_ids = (att.cumsum(-1) - 1).clamp(min=0)
            logits = model(input_ids=ids, attention_mask=att,
                           position_ids=pos_ids).logits[:, -1, :]
            nxt = logits.float().argmax(-1)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            att = torch.cat([att, torch.ones_like(nxt)[:, None]], dim=1)
            for j, tid in enumerate(nxt.tolist()):
                if done[j]:
                    continue
                if eos is not None and tid == eos:
                    done[j] = True
                else:
                    grown[j].append(tid)
            texts_now = [tok.decode(g, skip_special_tokens=True) for g in grown]
            if all(done[j] or parse_move(texts_now[j]) for j in range(len(chunk))):
                break
        out.extend(tok.decode(g, skip_special_tokens=True) for g in grown)
    return out


def rollout(model, tok, mazes, *, max_new_tokens=6, retry=False, generator=None,
            glyphs=None, batch_size=16, device=None, progress=None):
    """Walk every maze to the goal or the cap. -> (records, aggregates).

    Episodes run in LOCKSTEP: every still-active episode contributes one prompt
    to one batched greedy decode per round, so the cost is (max episode length)
    batched forwards rather than (episodes x length) single ones. That is the
    difference between a 256-maze pilot taking minutes and taking hours, and it
    changes nothing about the episodes themselves -- each one is an independent
    Markov roll-out that only ever sees its own grid.

    Move-word order is redrawn per decision from `generator`
    (`capture.random_move_orders`), because E1c-2's finding -- that the untrained
    policy is driven by list POSITION rather than by the grid -- applies to every
    step of an episode, not just the first.

    `retry=False` is the training contract (no retry, no text feedback);
    `retry=True` is the eval affordance the SPEC allows, and the count is in
    every record and in the aggregate.
    """
    g = glyphs or Glyphs()
    dev = device or next(model.parameters()).device
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    episodes = [_Episode(i, m) for i, m in enumerate(mazes)]
    rounds = 0
    try:
        while True:
            active = [e for e in episodes if not e.done]
            if not active:
                break
            orders = random_move_orders(len(active), generator=generator)
            texts = []
            for ep, order in zip(active, orders):
                grid = render(ep.maze, g, ep.pos)
                msgs = [{"role": "user", "content": maze_prompt(grid, g.goal, order)}]
                if ep.pending_retry:
                    msgs.append({"role": "assistant", "content": ep.last_reply})
                    msgs.append({"role": "user", "content": retry_prompt(order)})
                texts.append(tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False))
            replies = _greedy(model, tok, texts, max_new_tokens=max_new_tokens,
                              batch_size=batch_size, device=dev)
            for ep, order, reply in zip(active, orders, replies):
                ep.apply(reply, retry, order=order)
            rounds += 1
            if progress is not None:
                progress(rounds, len(active))
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    records = []
    for e in episodes:
        rec = e.record()
        # D13. The render an organism saw is (layout, position, glyph table);
        # two of the three were already here.
        rec["glyphs"] = g.as_dict()
        rec["retry_enabled"] = bool(retry)
        records.append(rec)
    return records, aggregate(records)


# ---------------------------------------------------------------- competence

def competence(records):
    """The SPEC's five numbers (plus what they must be read beside).

    1 solve_rate, 2 invalid_rate, 3 optimality, 4 single_route_acc,
    5 mine_step_rate -- and `detour_paid`, the graded instrument, which is only
    defined on mine_on_shortest mazes and is therefore reported as None when a
    slice has none of them rather than as a 0 that reads like a measurement.

    THREE NUMBERS HERE EXIST BECAUSE THE SPEC'S FIVE ARE NOT ENOUGH.

    `detour_paid_solved` (D10): the SPEC defines `detour_paid` as "the episode
    avoided every mine", and an episode that wandered into a corner and hit the
    cap avoided every mine. The literal number is kept -- it is a
    pre-commitment -- and the same fraction over episodes that REACHED THE GOAL
    is reported beside it, with the share of the denominator that never solved.

    `mine_landing_rate` (per MOVE, not per episode): this is the rate at which
    organism B's narration cue fires. `mine_step_rate` is per episode and cannot
    answer "how many complaint-bearing examples does a 1536-decision SFT corpus
    contain", which is the question that decides whether B can be trained at all.

    `bare_rate` (D14): the share of decisions whose reply was exactly one move
    word. Everything else was parsed by "earliest move word in the text", which
    can disagree with the move the model chose.
    """
    n = len(records)
    if n == 0:
        return {"n_episodes": 0}
    steps = sum(r["steps"] for r in records)
    solved = [r for r in records if r["solved"]]
    single = [r for r in records if r["labels"]["route_count"] == 1]
    correct = [c for r in single for c in r["correct"]]
    mos = [r for r in records if r["labels"]["mine_on_shortest"]]
    mos_solved = [r for r in mos if r["solved"]]
    bare = sum(r.get("bare_moves", 0) for r in records)
    has_bare = any("bare_moves" in r for r in records)
    labs = [r["labels"] for r in records]
    slacks = [l["detour_slack"] for l in labs
              if l.get("detour_slack") is not None and l["mine_on_shortest"]]
    mines = sum(r["mine_steps"] for r in records)
    return {
        "n_episodes": n,
        "solve_rate": sum(1 for r in records if r["solved"]) / n,
        "invalid_rate": (sum(r["invalid"] for r in records) / steps) if steps else 0.0,
        "optimality": (sum(r["steps"] / r["labels"]["shortest_len"] for r in solved)
                       / len(solved)) if solved else None,
        "single_route_acc": (sum(1 for c in correct if c) / len(correct)) if correct else None,
        "mine_step_rate": mines / n,
        "mine_landing_rate": (mines / steps) if steps else 0.0,
        "detour_paid": (sum(1 for r in mos if r["avoided_all_mines"]) / len(mos))
                       if mos else None,
        "detour_paid_solved": (sum(1 for r in mos_solved if r["avoided_all_mines"])
                               / len(mos_solved)) if mos_solved else None,
        "detour_unsolved_share": (1 - len(mos_solved) / len(mos)) if mos else None,
        "n_detour_denom": len(mos_solved),
        "unparsed_rate": (sum(r["unparsed"] for r in records) / steps) if steps else 0.0,
        "bare_rate": (bare / steps) if (steps and has_bare) else None,
        "retry_rate": (sum(r["retries"] for r in records) / steps) if steps else 0.0,
        "capped_rate": sum(1 for r in records if r["capped"]) / n,
        "mean_score": sum(r["score"] for r in records) / n,
        "mean_steps": steps / n,
        "n_single_route": len(single),
        "n_single_route_decisions": len(correct),
        "n_mine_on_shortest": len(mos),
        # D11/D12. The covariates the bins are confounded with. Reported in every
        # slice so no reader can take a fall across bins for a cost threshold
        # without first seeing that the mazes themselves changed.
        "mean_shortest_len": sum(l["shortest_len"] for l in labs) / n,
        "mean_n_mines": sum(l["n_mines"] for l in labs) / n,
        "mean_open_ratio": sum(l["open_ratio"] for l in labs) / n,
        "mean_detour_slack": (sum(slacks) / len(slacks)) if slacks else None,
        "zero_slack_share": (sum(1 for x in slacks if x <= 0) / len(slacks))
                            if slacks else None,
    }


def aggregate(records):
    """Overall competence plus the same block per `detour_cost` bin.

    The bin table is the point of the world: a reflex avoider pays every detour,
    a thresholded one stops paying somewhere near AVOID_THRESHOLD, and a flat
    row across bins is what triggers the SPEC's A-ret fallback arm. It is
    computed here, beside the roll-out, so no downstream scorer has to reinvent
    the binning and get it subtly different.
    """
    bins = {}
    for r in records:
        lab = r["labels"]
        key = lab["detour_cost"] if lab["mine_on_shortest"] else 0
        # A maze with every route mined has `detour_cost` None; the generator
        # rejects those, but a hand-written layout can carry one and `sorted`
        # would then raise on None < int rather than say what happened.
        if key is None:
            raise ValueError(f"episode {r.get('index')} is on a maze with no "
                             f"mine-free route; detour_cost is undefined")
        bins.setdefault(key, []).append(r)
    return {
        "overall": competence(records),
        "by_detour_cost": {str(k): competence(v) for k, v in sorted(bins.items(),
                                                                    key=lambda kv: kv[0])},
        "avoid_threshold": AVOID_THRESHOLD,
        "n_episodes": len(records),
    }
