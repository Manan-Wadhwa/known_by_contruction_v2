"""Cheap cross-fit with stage 1 at ONE UNIFORM fork per episode (no tie preference)."""
import sys, random, statistics as st
sys.argv = [sys.argv[0]]
sys.path.insert(0, '/tmp/claude-1000/-home-manan-Projects-files-digital-minds-maze/725710a9-eebd-4946-9475-b8f61f94876c/scratchpad')
import forksim4 as X
X.episode_forks_orig = X.episode_forks
def run_u(price, reps):
    r_ = random.Random(11); rows = []
    for rep in range(reps):
        for maze in X.F.pool:
            dg = X.F.dist_from(maze, maze.goal); ep = X.S.episode(maze, dg, 0.0)
            fs = X.episode_forks(ep, maze)
            if not fs: continue
            f = r_.choice(fs)
            s1 = X.stats(X.conts(maze, f, dg, price), f["taken"], price)
            if not s1["lost"]: rows.append(dict(kept=False)); continue
            s2 = X.stats(X.conts(maze, f, dg, price), f["taken"], price)
            rows.append(dict(kept=True, s1=s1, s2=s2))
    return rows
for price in (0, 10, 30):
    rows = run_u(price, 6); k = [r for r in rows if r["kept"]]; A2 = sum(r["s2"]["A2"] for r in k)
    print(f"price {price} stage-1 ONE uniform fork/ep: kept/ep {len(k)/(256*6):.2f} better(s2) {sum(r['s2']['better'] for r in k)/len(k):.3f} "
          f"relA2 {sum(r['s2']['AM2'] for r in k if r['s2']['better'])/A2 if A2 else 0:.3f} A(taken) s1 {st.mean(r['s1']['Atk'] for r in k):+.2f} s2 {st.mean(r['s2']['Atk'] for r in k):+.2f}")
