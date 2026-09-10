"""Would a geometric fork RULE raise the mine-relevant share above the uniform
draw's 0.051?  CPU-only, on the trial's own training mazes, model-free.

The uniform draw (maze_orgb.plan_forks) picks one of the episode's geometric
forks at random; the traces of PLAN §13.1 measure fork_share_detour_better at
0.051 (A s0) -- about one mine-relevant fork per step out of 24.  The question
here is whether a rule that reads ONLY distance-to-goal (legal under §1(a):
walls, branch points, distances) concentrates the draw on the forks where the
price can act.

At a fork, for each valid non-reverse move a:
    d(a) = 1 + dist_to_goal[succ(a)]         -- route length if optimal after a
    m(a) = fewest mines on any shortest route succ(a) -> goal, succ counted
A fork is MINE-RELEVANT when max m - min m > 0: the choice changes the mines an
optimal continuation must eat.  d_spread = max d - min d is the route noise the
fork's advantage carries whether or not a mine is anywhere near it.

Mines are read HERE, in an offline measurement that no training step imports, to
score candidate rules.  The rules themselves are scored on d_spread ALONE.
"""
import collections
import sys

sys.path.insert(0, "src")
from calibration import maze2                      # noqa: E402
from calibration.rl import _TRAIN_SEED_RANGE       # noqa: E402
from calibration.runner import derive_generator    # noqa: E402

N, POOL = 9, 256


def min_mines_to_goal(maze, dg):
    """{cell: fewest mines on a shortest route cell->goal, cell counted}."""
    best = {}
    for v in sorted(dg, key=lambda c: dg[c]):        # goal outward
        own = 1 if maze.is_mine(v) else 0
        if dg[v] == 0:
            best[v] = own
            continue
        nxt = [best[p] for _w, p in maze.neighbours(v)
               if dg.get(p) == dg[v] - 1 and p in best]
        best[v] = (min(nxt) if nxt else 0) + own
    return best


def forks_of(maze):
    """Every (cell, arrival) fork on the shortest DAG. -> [(d_spread, relevant)]."""
    dg = maze2.bfs_dist(maze, maze.goal)
    ds = maze2.bfs_dist(maze, maze.start)
    if maze.goal not in ds:
        return []
    mm = min_mines_to_goal(maze, dg)
    length = ds[maze.goal]
    dag = [v for v in ds if v in dg and ds[v] + dg[v] == length]
    out = []
    for cell in dag:
        # every way the path could have ARRIVED at this cell on a shortest route
        backs = [w for w, p in maze.neighbours(cell) if ds.get(p) == ds[cell] - 1]
        for back in (backs or [None]):
            moves = [(w, p) for w, p in maze.neighbours(cell) if w != _rev(back)]
            if len(moves) < 2:
                continue
            d = [1 + dg[p] for _w, p in moves if p in dg]
            m = [mm[p] for _w, p in moves if p in mm]
            if len(d) < 2 or len(m) < 2:
                continue
            # d(a) = 1 + dg[succ] takes only two values, dg[cell] and
            # dg[cell]+2, so the spread is 0 or 2 and "spread 0" lumps two
            # opposite cases: every branch ON a shortest route (all descend)
            # and every branch OFF it (all ascend). Only the first is a fork
            # where the price can act without paying route length for it.
            kind = ("descend" if max(d) == dg[cell] else
                    "ascend" if min(d) > dg[cell] else "mixed")
            out.append((max(d) - min(d), (max(m) - min(m)) > 0, kind))
    return out


def _rev(word):
    return None if word is None else {"up": "down", "down": "up",
                                      "left": "right", "right": "left"}[word]


def main():
    mazes, _ = maze2.generate_set(
        POOL, n=N, generator=derive_generator(0, "orgb", "train"),
        seed_range=_TRAIN_SEED_RANGE)
    rows = [f for mz in mazes for f in forks_of(mz)]
    print(f"mazes {len(mazes)}   forks on the shortest DAG {len(rows)}")
    by = collections.Counter()
    for spread, rel, kind in rows:
        by[("all", rel)] += 1
        by[(f"d_spread==0 & {kind}" if spread == 0 else f"d_spread==2", rel)] += 1
    print(f"{'rule':<14} {'forks':>7} {'share of all':>13} "
          f"{'mine-relevant':>14} {'share':>7}")
    tot = by[("all", True)] + by[("all", False)]
    for rule in ("all", "d_spread==0 & descend", "d_spread==0 & ascend",
                 "d_spread==0 & mixed", "d_spread==2"):
        n = by[(rule, True)] + by[(rule, False)]
        if not n:
            continue
        print(f"{rule:<22} {n:>7} {n/tot:>13.3f} {by[(rule, True)]:>14} "
              f"{by[(rule, True)]/n:>7.3f}")
    per = collections.Counter()
    for mz in mazes:
        fs = forks_of(mz)
        per["any"] += len(fs)
        per["eq"] += sum(1 for s, _r, k in fs if s == 0 and k == "descend")
    print(f"\nper maze: {per['any']/len(mazes):.1f} forks, "
          f"{per['eq']/len(mazes):.1f} of them equal-distance AND descending")


if __name__ == "__main__":
    main()
