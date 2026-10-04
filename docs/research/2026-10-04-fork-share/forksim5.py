"""Continuation pairs within a fork: do they differ in mines, in steps, both, neither? Calibrated stand-in (stall 0.2)."""
import sys, random, collections, itertools
sys.argv = [sys.argv[0], "0.0", "1", "0.2"]
sys.path.insert(0, '/tmp/claude-1000/-home-manan-Projects-files-digital-minds-maze/725710a9-eebd-4946-9475-b8f61f94876c/scratchpad')
import forksim3 as S, forksim as F
from calibration.maze_orgb import episode_forks
cnt = collections.Counter(); tk = collections.Counter()
for rep in range(6):
    for maze in F.pool:
        dg = F.dist_from(maze, maze.goal); ep = S.episode(maze, dg, 0.0)
        for f in episode_forks(ep, maze):
            cs = []
            for a in f["moves"]:
                s, m, so, _ = S.walk(maze, f["pos"], dg, f["budget"], 0.0, first=a)
                cs.append((a, s if so else f["budget"], m, so))
            for x, y in itertools.combinations(cs, 2):
                cls = ("mines" if x[2] != y[2] else "") + ("+" if x[2] != y[2] and x[1] != y[1] else "") + ("steps" if x[1] != y[1] else "")
                cnt[cls or "neither"] += 1
            t = next(c for c in cs if c[0] == f["taken"])
            if len({c[2] for c in cs}) > 1:
                tk["mine-differing fork"] += 1
                tk["taken has fewest mines"] += t[2] == min(c[2] for c in cs)
n = sum(cnt.values())
print({k: round(v/n, 3) for k, v in cnt.most_common()}, "pairs", n)
print(dict(tk), round(tk["taken has fewest mines"]/tk["mine-differing fork"], 3))
