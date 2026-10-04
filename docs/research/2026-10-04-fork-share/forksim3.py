"""Stand-in with wander noise eps, K continuations per move, and selection rules.

Policy: mine-blind; each step w.p. eps a uniform valid move, else a uniform
shortest move. Mines read only to price landings. Prints, per rule:
forks/ep, better (fork_share_detour_better analogue), lost (taken move's
continuation beaten by the best, by return), |A|, mineA2 (share of sum A^2 from
the mine term), relA2 (share of sum A^2 from the mine term at forks where the
best continuation has fewer mines than the taken one: the part the price can
use to change a decision the policy now makes).
"""
import sys, random, statistics as st, collections
sys.path.insert(0, '/tmp/claude-1000/-home-manan-Projects-files-digital-minds-maze/725710a9-eebd-4946-9475-b8f61f94876c/scratchpad')
import forksim as F
from calibration.maze_orgb import episode_forks, DELTAS
rng = random.Random(7)
STALL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
NOSTALL = False

def walk(maze, pos, dg, budget, eps, first=None):
    steps = mines = 0; cells = []; prev = None
    if first is not None:
        dr, dc = DELTAS[first]; pos = (pos[0]+dr, pos[1]+dc)
        steps = 1; cells.append(pos); mines += maze.is_mine(pos)
    while pos != maze.goal and steps < budget:
        nb = [n for _w, n in maze.neighbours(pos)]
        u = rng.random()
        stall = 0.0 if NOSTALL else STALL
        if u < stall:
            steps += 1; continue
        elif u < stall + eps:
            pos = rng.choice(nb)
        else:
            pos = rng.choice([n for n in nb if dg[n] == dg[pos]-1])
        steps += 1; cells.append(pos); mines += maze.is_mine(pos)
    return steps, mines, pos == maze.goal, cells

def episode(maze, dg, eps):
    global NOSTALL
    NOSTALL = True
    s, m, so, cells = walk(maze, maze.start, dg, maze.labels['cap'], eps)
    NOSTALL = False
    pos = [maze.start] + cells[:-1]; moves = []
    prev = maze.start
    for c in cells:
        moves.append(next(w for w, (dr, dc) in DELTAS.items() if (prev[0]+dr, prev[1]+dc) == c)); prev = c
    return {"positions": pos, "moves": moves, "decisions": list(range(len(moves))),
            "labels": {"cap": maze.labels['cap']}, "index": 0, "group": 0}

def forks_of(price, eps, K, reps, tie_only=False):
    out = []
    for rep in range(reps):
        for mi, maze in enumerate(F.pool):
            dg = F.dist_from(maze, maze.goal)
            ep = episode(maze, dg, eps)
            for f in episode_forks(ep, maze):
                pos = f["pos"]
                lg = {a: 1 + dg[(pos[0]+DELTAS[a][0], pos[1]+DELTAS[a][1])] for a in f["moves"]}
                lmin = min(lg.values()); tie = sum(v == lmin for v in lg.values()) >= 2
                moves = [a for a in f["moves"] if (lg[a] == lmin or not tie_only or not tie)]
                if tie_only and tie and f["taken"] not in moves:
                    moves = f["moves"]
                cs = []
                for a in moves:
                    sims = [walk(maze, pos, dg, f["budget"], eps, first=a) for _ in range(K)]
                    cs.append(dict(a=a, R=st.mean(F.ret(s, m, so, f["budget"], price) for s, m, so, _ in sims),
                                   M=st.mean(m for _s, m, *_ in sims), S=st.mean((s if so else f["budget"]) for s, _m, so, _ in sims),
                                   G=st.mean(so for _s, _m, so, _ in sims)))
                tk = next((c for c in cs if c["a"] == f["taken"]), None)
                Rm = st.mean(c["R"] for c in cs); Mm = st.mean(c["M"] for c in cs)
                A = [c["R"]-Rm for c in cs]; AM = [-price*(c["M"]-Mm) for c in cs]
                best = max(cs, key=lambda c: c["R"])
                out.append(dict(ep=(rep, mi), tie=tie, has_tk=tk is not None,
                    better=(tk is not None and best["M"] < tk["M"]),
                    lost=(tk is not None and best["R"] > tk["R"] + 1e-9),
                    nz=max(c["R"] for c in cs) - min(c["R"] for c in cs) > 1e-9,
                    A2=sum(x*x for x in A), AM2=sum(x*x for x in AM), absA=st.mean(abs(x) for x in A),
                    lgspread=max(lg[c["a"]] for c in cs) - min(lg[c["a"]] for c in cs),
                    resid=max(c["R"] + 2*lg[c["a"]] for c in cs) - min(c["R"] + 2*lg[c["a"]] for c in cs)))
    return out

def line(rows, tag, nep):
    if not rows: print(f"  {tag:30s} none"); return
    cmp_ = [r for r in rows if r["has_tk"]]
    A2 = sum(r["A2"] for r in rows)
    print(f"  {tag:30s} forks/ep {len(rows)/nep:5.2f} nz {sum(r['nz'] for r in rows)/len(rows):.3f} "
          f"better {sum(r['better'] for r in cmp_)/max(1,len(cmp_)):.3f} lost {sum(r['lost'] for r in cmp_)/max(1,len(cmp_)):.3f} "
          f"|A| {st.mean(r['absA'] for r in rows):5.2f} mineA2 {sum(r['AM2'] for r in rows)/A2 if A2 else 0:.3f} "
          f"relA2 {sum(r['AM2'] for r in rows if r['better'])/A2 if A2 else 0:.3f}")

def pick1(rows, pred, r_=random.Random(3)):
    by = collections.defaultdict(list)
    for r in rows: by[r["ep"]].append(r)
    return [r_.choice([x for x in rs if pred(x)] or rs) for rs in by.values()]

if __name__ == "__main__":
    eps = float(sys.argv[1]); reps = int(sys.argv[2])
    for price in (0, 10, 30):
        base = forks_of(price, eps, 1, reps)
        nep = len({r["ep"] for r in base})
        print(f"eps {eps} price {price}")
        line(pick1(base, lambda r: True), "uniform 1/ep (now)", nep)
        line(base, "every fork", nep)
        line(pick1(base, lambda r: r["tie"]), "1/ep prefer tie", nep)
        line([r for r in base if r["tie"]], "every tie fork", nep)
        line([r for r in base if r["lost"] or not r["has_tk"]], "every fork, keep taken-lost", nep)
        line([r for r in base if r["resid"] > 1e-9], "every fork, keep resid>0", nep)
        line(pick1(base, lambda r: r["resid"] > 1e-9), "1/ep, prefer resid>0", nep)
        k2 = forks_of(price, eps, 2, max(1, reps//2))
        line(pick1(k2, lambda r: True), "uniform 1/ep, K=2", len({r['ep'] for r in k2}))
        line(pick1(k2, lambda r: r["tie"]), "1/ep prefer tie, K=2", len({r['ep'] for r in k2}))
        to = forks_of(price, eps, 1, reps, tie_only=True)
        line(pick1(to, lambda r: r["tie"]), "1/ep prefer tie, tied moves only", len({r['ep'] for r in to}))
