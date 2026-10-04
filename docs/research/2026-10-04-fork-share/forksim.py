"""Stand-in simulation of the fork term's comparisons on a training-like pool.

Policy stand-in: mine-blind shortest-path follower (uniform among moves that
reduce bfs distance to goal), as nav48_4b is (single-route 0.975, blindness
0.985). Mines are read ONLY to price landings (the world), as in the trainer.
"""
import sys, random, collections, statistics as st
import torch
sys.path.insert(0, '/home/manan/Projects/files/digital_minds-maze/src')
from calibration import maze2
from calibration.rl import _TRAIN_SEED_RANGE
from calibration.maze_orgb import episode_forks, DELTAS

pool, _ = maze2.generate_set(256, n=9, generator=torch.Generator().manual_seed(0),
                             seed_range=_TRAIN_SEED_RANGE)
rng = random.Random(0)

def dist_from(maze, src, banned=()):
    from collections import deque
    d = {tuple(src): 0}; q = deque([tuple(src)])
    while q:
        c = q.popleft()
        for _w, n in maze.neighbours(c):
            if n in banned or n in d: continue
            d[n] = d[c] + 1; q.append(n)
    return d

def walk(maze, pos, dg, budget, first=None, banned=()):
    """Mine-blind shortest walk; returns (steps, mines, solved, cells)."""
    steps = mines = 0; cells = []
    if first is not None:
        dr, dc = DELTAS[first]; pos = (pos[0]+dr, pos[1]+dc)
        steps += 1; cells.append(pos); mines += maze.is_mine(pos)
    while pos != maze.goal and steps < budget:
        cand = [n for _w, n in maze.neighbours(pos) if n not in banned
                and n in dg and dg[n] == dg[pos] - 1]
        if not cand:
            cand = [n for _w, n in maze.neighbours(pos) if n in dg and dg[n] == dg[pos]-1]
        pos = rng.choice(cand); steps += 1; cells.append(pos); mines += maze.is_mine(pos)
    return steps, mines, pos == maze.goal, cells

def ret(steps, mines, solved, budget, price):
    return (20 if solved else 0) - 2*(steps if solved else budget) - price*mines

def episode(maze, dg):
    pos = maze.start; positions = [pos]; moves = []
    steps, mines, solved, cells = walk(maze, pos, dg, maze.labels['cap'])
    prev = pos
    for c in cells:
        for w, (dr, dc) in DELTAS.items():
            if (prev[0]+dr, prev[1]+dc) == c: moves.append(w)
        prev = c
    positions = [maze.start] + cells[:-1]
    return {"positions": positions, "moves": moves, "decisions": list(range(len(moves))),
            "labels": {"cap": maze.labels['cap']}, "index": 0, "group": 0}

def run(price, noUturn=False, reps=20, K=1):
    rows = []
    for rep in range(reps):
        for mi, maze in enumerate(pool):
            dg = dist_from(maze, maze.goal)
            ep = episode(maze, dg)
            forks = episode_forks(ep, maze)
            for f in forks:
                pos = f["pos"]; banned = {pos} if noUturn else ()
                dgb = dist_from(maze, maze.goal, banned) if noUturn else dg
                cs = []
                for a in f["moves"]:
                    dr, dc = DELTAS[a]; nxt = (pos[0]+dr, pos[1]+dc)
                    lgeo = 1 + dg[nxt]
                    sims = [walk(maze, pos, dgb if nxt in dgb else dg, f["budget"], first=a, banned=banned) for _ in range(K)]
                    R = st.mean(ret(s, m, so, f["budget"], price) for s, m, so, _ in sims)
                    S = st.mean(s for s, *_ in sims); M = st.mean(m for _s, m, *_ in sims)
                    cs.append(dict(a=a, R=R, S=S, M=M, lgeo=lgeo))
                Rm = st.mean(c["R"] for c in cs); Sm = st.mean(c["S"] for c in cs); Mm = st.mean(c["M"] for c in cs)
                AL = [-2*(c["S"]-Sm) for c in cs]; AM = [-price*(c["M"]-Mm) for c in cs]
                A = [c["R"]-Rm for c in cs]
                taken = next(c for c in cs if c["a"] == f["taken"])
                best = max(cs, key=lambda c: c["R"])
                lg = [c["lgeo"] for c in cs]
                rows.append(dict(maze=mi, t=f["t"], nmoves=len(cs), mos=maze.labels['mine_on_shortest'],
                    dc=maze.labels['detour_cost'], nz=max(c["R"] for c in cs)-min(c["R"] for c in cs) > 1e-9,
                    better=best["M"] < taken["M"], mdiff=len({c["M"] for c in cs}) > 1,
                    AL2=sum(x*x for x in AL), AM2=sum(x*x for x in AM), A2=sum(x*x for x in A),
                    Aabs=st.mean(abs(x) for x in A), spreadL=max(lg)-min(lg),
                    spreadR=max(c["R"] for c in cs)-min(c["R"] for c in cs),
                    resid=[c["R"]+2*c["lgeo"] for c in cs], ep=(rep, mi)))
    return rows

def summ(rows, tag):
    n = len(rows)
    sAM = sum(r["AM2"] for r in rows); sAL = sum(r["AL2"] for r in rows)
    print(f"{tag}: forks/ep {n/ (len({r['ep'] for r in rows})):.2f} n={n} nz {sum(r['nz'] for r in rows)/n:.3f} "
          f"better {sum(r['better'] for r in rows)/n:.3f} mdiff {sum(r['mdiff'] for r in rows)/n:.3f} "
          f"|A| {st.mean(r['Aabs'] for r in rows):.2f} mineshareA2 {sAM/(sAM+sAL) if sAM+sAL else 0:.3f}")

if __name__ == "__main__":
    import pickle
    out = {}
    for price in (0, 10, 30):
        for nu in (False, True):
            rows = run(price, noUturn=nu, reps=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
            out[(price, nu)] = rows
            summ(rows, f"price {price} noUturn {nu}")
    pickle.dump(out, open('/tmp/claude-1000/-home-manan-Projects-files-digital-minds-maze/725710a9-eebd-4946-9475-b8f61f94876c/scratchpad/forksim.pkl', 'wb'))
