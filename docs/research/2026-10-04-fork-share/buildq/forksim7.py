"""Cell-D stand-in for section 16's mechanism read 2: the sign of stage-2 A(taken).

Adapted from docs/research/2026-10-04-fork-share/forksim4.py. Differences:
the trainer's own `episode_forks` tie flag, `crossfit_keep` and `fork_terms`
are called; the policy has a per-cell HABIT (a fixed move per cell, as a
peaked LLM policy has) and a per-cell WANDER rate; wandering draws one of the
four words uniformly (a wall is a blocked step; three in a row end the episode
unsolved), as the real world does. Cell D: price 0, unsolved charged at the
remaining budget (D33). One tie-preferred fork per episode, as `plan_forks`
under `pick="tie"`.

Scenarios (argv-free, all printed):
  homog   : habit on a shortest move everywhere, wander w0 everywhere.
  traps   : as homog, plus a share of cells where the wander rate is w_trap
            (cell-dependent failure, as whole-group failures in the trace show).
  waste   : as homog, but at a share of cells the habit is a non-shortest move.
  leak    : homog/traps with stage 2 REPLAYING stage 1 (the leak to detect).
Output per scenario: kept share of forks, mean stage-2 A(taken) at kept forks,
split by taken-on-shortest vs wasted taken move, share of kept forks whose
taken move is beaten again at stage 2, the share whose stage-2 returns all
tie (0 under a replay), and the fork-level SD of A2(taken).
Inputs: none. Output: printed table.
"""
import sys, random, statistics as st
import torch
sys.path.insert(0, '/home/manan/Projects/files/digital_minds-maze/src')
from calibration import maze2
from calibration.rl import _TRAIN_SEED_RANGE
from calibration.maze_orgb import (episode_forks, crossfit_keep, fork_terms,
                                   DELTAS, MAX_CONSECUTIVE_BLOCKED)

POOL, _ = maze2.generate_set(256, n=9, generator=torch.Generator().manual_seed(0),
                             seed_range=_TRAIN_SEED_RANGE)
WORDS = list(DELTAS)


def make_policy(maze, rng, w0, trap_frac, w_trap, waste_frac, bump=False):
    """A peaked policy for one maze: habit move and wander rate per cell. -> (habit, wander).

    With `bump`, a trap cell's wander is a wall bump (the policy sticks; three in a
    row end the roll-out unsolved), stored as a negative rate."""
    dg = maze2.bfs_dist(maze, maze.goal)
    habit, wander = {}, {}
    for c, d in dg.items():
        nb = list(maze.neighbours(c))
        short = [w for w, n in nb if dg.get(n, 1e9) == d - 1]
        other = [w for w, n in nb if dg.get(n, 1e9) != d - 1]
        if other and rng.random() < waste_frac:
            habit[c] = rng.choice(other)
        else:
            habit[c] = rng.choice(short) if short else (nb[0][0] if nb else 'up')
        wander[c] = ((-w_trap if bump else w_trap) if rng.random() < trap_frac else w0)
    return habit, wander


def walk(maze, pos, budget, habit, wander, rng, first=None):
    """One roll-out under the policy; first move forced if given. -> (positions, moves, return)."""
    positions, moves, landings, blocked = [], [], [], 0
    pos = tuple(pos)
    while len(moves) < budget and blocked < MAX_CONSECUTIVE_BLOCKED:
        if first is not None and not moves:
            mv = first
        elif wander.get(pos, 0.0) < 0 and rng.random() < -wander[pos]:
            walls = [w for w in WORDS if w not in dict(maze.neighbours(pos))]
            mv = rng.choice(walls) if walls else habit[pos]
        elif rng.random() < abs(wander.get(pos, 0.0)) and wander.get(pos, 0.0) > 0:
            mv = rng.choice(WORDS)
        else:
            mv = habit[pos]
        positions.append(pos); moves.append(mv)
        pos, kind = maze2.step(maze, pos, mv)
        landings.append(kind)
        blocked = blocked + 1 if kind == "blocked" else 0
        if kind == "goal":
            return positions, moves, 20.0 - 2.0 * len(moves)
    return positions, moves, -2.0 * budget


def conts_for(forks, habit_of, wander_of, rng):
    """Every valid move forced once at every fork. -> [cont dict] as the trainer's."""
    out = []
    for j, f in enumerate(forks):
        mz = f["maze"]
        for a in f["moves"]:
            _p, _m, R = walk(mz, f["pos"], f["budget"], habit_of[j], wander_of[j], rng, first=a)
            out.append({"fork": j, "move": a, "return": R, "mine_steps": 0})
    return out


def run(reps, w0, trap_frac=0.0, w_trap=0.0, waste_frac=0.0, leak=False, seed=1, bump=False):
    """One scenario over the pool. -> dict of reads."""
    rng = random.Random(seed)
    pool_n, rows, solved, neps = 0, [], 0, 0
    for rep in range(reps):
        forks, H, W = [], [], []
        for mi, maze in enumerate(POOL):
            habit, wander = make_policy(maze, rng, w0, trap_frac, w_trap, waste_frac, bump)
            pos, mv, _R = walk(maze, maze.start, maze.labels['cap'], habit, wander, rng)
            neps += 1; solved += _R > -2.0 * maze.labels['cap']
            ep = {"positions": pos, "moves": mv, "decisions": list(range(len(mv))),
                  "labels": {"cap": maze.labels['cap']}, "index": mi, "group": mi}
            fs = episode_forks(ep, maze)
            if not fs:
                continue
            pool = [f for f in fs if f["tie"]] or fs
            f = dict(rng.choice(pool)); f["maze"] = maze
            dg = maze2.bfs_dist(maze, maze.goal)
            nxt = {w: n for w, n in maze.neighbours(f["pos"])}
            f["taken_short"] = (f["taken"] in nxt and
                                dg.get(nxt[f["taken"]], 1e9) == dg[f["pos"]] - 1)
            forks.append(f); H.append(habit); W.append(wander)
        pool_n += len(forks)
        c1 = conts_for(forks, H, W, rng)
        kept = crossfit_keep(forks, c1)
        f2 = [forks[j] for j in kept]
        if leak:
            c2 = []
            for i, j in enumerate(kept):
                c2 += [dict(c, fork=i) for c in c1 if c["fork"] == j]
        else:
            c2 = conts_for(f2, [H[j] for j in kept], [W[j] for j in kept], rng)
        r2, _s = fork_terms(f2, c2)
        for r in r2:
            fk = f2[r["fork"]]
            k = r["moves"].index(fk["taken"])
            a = r["advantages"][k]
            beaten = max(r["returns"]) - r["returns"][k] > 1e-9
            rows.append((a, fk["taken_short"], beaten, max(r["returns"]) - min(r["returns"]) <= 1e-9))
    A = [a for a, *_ in rows]
    sh = [a for a, s, *_ in rows if s]; wa = [a for a, s, *_ in rows if not s]
    return dict(solve=solved / neps, kept_share=len(rows) / pool_n, n=len(rows), mean=st.mean(A) if A else None,
                sd=st.pstdev(A) if A else None,
                mean_short=st.mean(sh) if sh else None, n_short=len(sh),
                mean_waste=st.mean(wa) if wa else None, n_waste=len(wa),
                beaten2=sum(r[2] for r in rows) / len(rows) if rows else None,
                tie2=sum(r[3] for r in rows) / len(rows) if rows else None,
                share_neg=sum(a < -1e-9 for a in A) / len(A) if A else None)


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    fmt = lambda x: "  n/a" if x is None else f"{x:+.2f}"
    scen = [
        ("homog w0=0.01", dict(w0=0.01)),
        ("homog w0=0.03", dict(w0=0.03)),
        ("homog w0=0.10 (noisy, stand-in-like)", dict(w0=0.10)),
        ("traps 5% cells w=0.5, w0=0.01", dict(w0=0.01, trap_frac=0.05, w_trap=0.5)),
        ("traps 10% cells w=0.5, w0=0.01", dict(w0=0.01, trap_frac=0.10, w_trap=0.5)),
        ("waste 1% habits off-path, w0=0.01", dict(w0=0.01, waste_frac=0.01)),
        ("waste 2% habits off-path, w0=0.01", dict(w0=0.01, waste_frac=0.02)),
        ("sinks 2% cells bump 0.6, w0=0.003", dict(w0=0.003, trap_frac=0.02, w_trap=0.6, bump=True)),
        ("sinks 3% cells bump 0.6, w0=0.01", dict(w0=0.01, trap_frac=0.03, w_trap=0.6, bump=True)),
        ("sinks 6% cells bump 0.6, w0=0.01", dict(w0=0.01, trap_frac=0.06, w_trap=0.6, bump=True)),
        ("sinks 6% cells bump 0.6, w0=0.003", dict(w0=0.003, trap_frac=0.06, w_trap=0.6, bump=True)),
        ("LEAK homog w0=0.01", dict(w0=0.01, leak=True)),
        ("LEAK sinks 6% bump 0.6, w0=0.003", dict(w0=0.003, trap_frac=0.06, w_trap=0.6, bump=True, leak=True)),
        ("LEAK homog w0=0.10", dict(w0=0.10, leak=True)),
        ("LEAK traps 10%, w0=0.01", dict(w0=0.01, trap_frac=0.10, w_trap=0.5, leak=True)),
    ]
    print(f"reps {reps} x 256 mazes, one tie-preferred fork per episode, cell D")
    print(f"{'scenario':42s} solve kept  n_kept  meanA2  sdA2  A2|short(n)     A2|waste(n)     beaten@s2 neg   tie@s2")
    for tag, kw in scen:
        r = run(reps, **kw)
        print(f"{tag:42s} {r['solve']:.3f} {r['kept_share']:.3f} {r['n']:6d}  {fmt(r['mean'])} {r['sd'] or 0:5.2f}  "
              f"{fmt(r['mean_short'])}({r['n_short']:4d})  {fmt(r['mean_waste'])}({r['n_waste']:4d})  "
              f"{r['beaten2'] or 0:.3f}    {r['share_neg'] or 0:.3f}  {r['tie2'] or 0:.3f}")
