"""Cross-fitted selection: select forks on one set of continuations (the taken
move's continuation beaten by an alternative, by the priced return), credit them
from an independent second set. Calibrated stand-in: eps 0, stall 0.2."""
import sys, random, statistics as st, collections
sys.argv = [sys.argv[0], "0.0", "1", "0.2"] + sys.argv[1:]
sys.path.insert(0, '/tmp/claude-1000/-home-manan-Projects-files-digital-minds-maze/725710a9-eebd-4946-9475-b8f61f94876c/scratchpad')
import forksim3 as S
import forksim as F
from calibration.maze_orgb import episode_forks, DELTAS

def conts(maze, f, dg, price):
    pos = f["pos"]; cs = []
    for a in f["moves"]:
        s, m, so, _ = S.walk(maze, pos, dg, f["budget"], 0.0, first=a)
        cs.append(dict(a=a, R=F.ret(s, m, so, f["budget"], price), M=m))
    return cs

def stats(cs, taken, price):
    Rm = st.mean(c["R"] for c in cs); Mm = st.mean(c["M"] for c in cs)
    tk = next(c for c in cs if c["a"] == taken); best = max(cs, key=lambda c: c["R"])
    return dict(A2=sum((c["R"]-Rm)**2 for c in cs), AM2=sum((price*(c["M"]-Mm))**2 for c in cs),
                better=best["M"] < tk["M"], lost=best["R"] > tk["R"] + 1e-9, Atk=tk["R"]-Rm,
                absA=st.mean(abs(c["R"]-Rm) for c in cs))

def run(price, reps, pool_rule):
    rows = []
    for rep in range(reps):
        for mi, maze in enumerate(F.pool):
            dg = F.dist_from(maze, maze.goal)
            ep = S.episode(maze, dg, 0.0)
            for f in episode_forks(ep, maze):
                pos = f["pos"]
                lg = {a: 1 + dg[(pos[0]+DELTAS[a][0], pos[1]+DELTAS[a][1])] for a in f["moves"]}
                lmin = min(lg.values()); tie = sum(v == lmin for v in lg.values()) >= 2
                if pool_rule == "tie" and not tie: continue
                c1 = conts(maze, f, dg, price); s1 = stats(c1, f["taken"], price)
                if not s1["lost"]: rows.append(dict(ep=(rep, mi), kept=False, s1=s1)); continue
                c2 = conts(maze, f, dg, price); s2 = stats(c2, f["taken"], price)
                rows.append(dict(ep=(rep, mi), kept=True, s1=s1, s2=s2))
    return rows

if __name__ == "__main__":
    reps = 6
    for price in (0, 10, 30):
        for rule in ("all", "tie"):
            rows = run(price, reps, rule)
            nep = 256 * reps
            k = [r for r in rows if r["kept"]]
            A2 = sum(r["s2"]["A2"] for r in k)
            print(f"price {price} stage-1 at {rule:3s} forks: stage-1 forks/ep {len(rows)/nep:.2f} kept/ep {len(k)/nep:.2f} "
                  f"better(s2) {sum(r['s2']['better'] for r in k)/len(k):.3f} lost(s2) {sum(r['s2']['lost'] for r in k)/len(k):.3f} "
                  f"|A|(s2) {st.mean(r['s2']['absA'] for r in k):.2f} mineA2 {sum(r['s2']['AM2'] for r in k)/A2 if A2 else 0:.3f} "
                  f"relA2 {sum(r['s2']['AM2'] for r in k if r['s2']['better'])/A2 if A2 else 0:.3f} "
                  f"mean A(taken) s1 {st.mean(r['s1']['Atk'] for r in k):+.2f} s2 {st.mean(r['s2']['Atk'] for r in k):+.2f}")

def run1(price, reps):
    """Stage 1 at ONE fork per episode, drawn uniformly among its tie forks (else among all)."""
    r_ = random.Random(11); rows = []
    for rep in range(reps):
        for mi, maze in enumerate(F.pool):
            dg = F.dist_from(maze, maze.goal)
            ep = S.episode(maze, dg, 0.0)
            fs = episode_forks(ep, maze)
            if not fs: continue
            def is_tie(f):
                pos = f["pos"]; lg = [1 + dg[(pos[0]+DELTAS[a][0], pos[1]+DELTAS[a][1])] for a in f["moves"]]
                return sum(v == min(lg) for v in lg) >= 2
            ties = [f for f in fs if is_tie(f)]
            f = r_.choice(ties or fs)
            c1 = conts(maze, f, dg, price); s1 = stats(c1, f["taken"], price)
            if not s1["lost"]: rows.append(dict(kept=False, s1=s1)); continue
            c2 = conts(maze, f, dg, price); s2 = stats(c2, f["taken"], price)
            rows.append(dict(kept=True, s1=s1, s2=s2))
    return rows

if __name__ == "__main__" and len(sys.argv) > 4:
    for price in (0, 10, 30):
        rows = run1(price, 6); nep = 256 * 6
        k = [r for r in rows if r["kept"]]; A2 = sum(r["s2"]["A2"] for r in k)
        print(f"price {price} stage-1 ONE prefer-tie fork/ep: kept/ep {len(k)/nep:.2f} better(s2) {sum(r['s2']['better'] for r in k)/len(k):.3f} "
              f"|A|(s2) {st.mean(r['s2']['absA'] for r in k):.2f} relA2 {sum(r['s2']['AM2'] for r in k if r['s2']['better'])/A2 if A2 else 0:.3f} "
              f"mean A(taken) s1 {st.mean(r['s1']['Atk'] for r in k):+.2f} s2 {st.mean(r['s2']['Atk'] for r in k):+.2f}")
