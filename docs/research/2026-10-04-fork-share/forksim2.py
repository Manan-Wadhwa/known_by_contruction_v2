"""Per-fork geometry vs outcome, for candidate selection rules. Stand-in policy as forksim.py."""
import sys, random, statistics as st, collections
sys.path.insert(0, '/tmp/claude-1000/-home-manan-Projects-files-digital-minds-maze/725710a9-eebd-4946-9475-b8f61f94876c/scratchpad')
import forksim as F
from calibration.maze_orgb import episode_forks, DELTAS
INF = 10**6

def analyse(price, noUturn, reps=10):
    rows = []
    for rep in range(reps):
        for mi, maze in enumerate(F.pool):
            dg = F.dist_from(maze, maze.goal)
            ep = F.episode(maze, dg)
            for f in episode_forks(ep, maze):
                pos = f["pos"]
                dnr = F.dist_from(maze, maze.goal, {pos})
                banned = {pos} if noUturn else ()
                cs = []
                for a in f["moves"]:
                    dr, dc = DELTAS[a]; nxt = (pos[0]+dr, pos[1]+dc)
                    lgeo = 1 + dg[nxt]; lnr = 1 + dnr[nxt] if nxt in dnr else INF
                    use = dnr if (noUturn and nxt in dnr) else dg
                    s, m, so, _ = F.walk(maze, pos, use, f["budget"], first=a, banned=banned if nxt in dnr else ())
                    cs.append(dict(a=a, R=F.ret(s, m, so, f["budget"], price), S=s, M=m, lgeo=lgeo, lnr=lnr))
                Rm = st.mean(c["R"] for c in cs); Sm = st.mean(c["S"] for c in cs); Mm = st.mean(c["M"] for c in cs)
                lmin = min(c["lgeo"] for c in cs)
                tk = next(c for c in cs if c["a"] == f["taken"])
                best = max(cs, key=lambda c: c["R"])
                alt_exc = min((c["lnr"] - lmin for c in cs if c is not tk), default=INF)
                AM2 = sum((price*(c["M"]-Mm))**2 for c in cs); AL2 = sum((2*(c["S"]-Sm))**2 for c in cs)
                rows.append(dict(ep=(rep, mi), mos=maze.labels["mine_on_shortest"], dc=maze.labels["detour_cost"],
                    tie=sum(c["lgeo"] == lmin for c in cs) >= 2, alt_exc=alt_exc,
                    better=best["M"] < tk["M"], taken_lost=best["R"] > tk["R"],
                    mdiff=len({c["M"] for c in cs}) > 1, AM2=AM2, AL2=AL2,
                    absA=st.mean(abs(c["R"]-Rm) for c in cs),
                    spread=max(c["R"] for c in cs)-min(c["R"] for c in cs)))
    return rows

def share(rows, key):
    return sum(r[key] for r in rows)/len(rows) if rows else float('nan')

def line(rows, tag, nep):
    if not rows:
        print(f"  {tag:34s} (none)"); return
    sAM = sum(r["AM2"] for r in rows); sAL = sum(r["AL2"] for r in rows)
    # behaviour-relevant mine share: mine A^2 at forks where a fewer-mine alternative beat taken
    sB = sum(r["AM2"] for r in rows if r["better"])
    tot = sAM + sAL
    print(f"  {tag:34s} forks/ep {len(rows)/nep:5.2f} better {share(rows,'better'):.3f} "
          f"taken_lost {share(rows,'taken_lost'):.3f} mdiff {share(rows,'mdiff'):.3f} "
          f"mineA2 {sAM/tot if tot else 0:.3f} betterA2 {sB/tot if tot else 0:.3f} |A| {st.mean(r['absA'] for r in rows):.2f}")

def per_episode_pick(rows, pred, rng):
    by = collections.defaultdict(list)
    for r in rows: by[r["ep"]].append(r)
    out = []
    for ep, rs in by.items():
        good = [r for r in rs if pred(r)]
        out.append(rng.choice(good if good else rs))
    return out

if __name__ == "__main__":
    rng = random.Random(1)
    for price in (0, 10, 30):
        for nu in (False, True):
            rows = analyse(price, nu)
            nep = len({r["ep"] for r in rows})
            print(f"price {price} noUturn {nu}")
            line(rows, "all forks", nep)
            line(per_episode_pick(rows, lambda r: True, rng), "uniform 1/ep (now)", nep)
            line([r for r in rows if r["tie"]], "tie forks (>=2 moves at min L)", nep)
            for E in (0, 2, 4, 8):
                line([r for r in rows if r["alt_exc"] <= E], f"route forks alt_exc<={E}", nep)
                line(per_episode_pick(rows, lambda r, E=E: r["alt_exc"] <= E, rng), f"1/ep prefer alt_exc<={E}", nep)
            for tau in (4.5, 6.5, 8.5):
                line([r for r in rows if r["spread"] > tau], f"spread>{tau}", nep)
            # by maze class, uniform draw
            for mos in (True, False):
                sub = [r for r in rows if r["mos"] == mos]
                print(f"    mos={mos}: forks {len(sub)} better {share(sub,'better'):.3f}",
                      {dc: round(share([r for r in sub if r['dc']==dc],'better'),3) for dc in sorted({r['dc'] for r in sub}, key=str)})
            print(f"    episodes with a route fork (E=4): {len({r['ep'] for r in rows if r['alt_exc']<=4})/nep:.3f}; share of better forks that are route forks E=4: {sum(r['better'] and r['alt_exc']<=4 for r in rows)/max(1,sum(r['better'] for r in rows)):.3f}")
