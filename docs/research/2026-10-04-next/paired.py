"""Paired A - D gap in held-out mine steps at one read (§19), on the `episodes` records.

Usage: python3 paired.py <label_A> <label_D> <read>   (read: 40, 80, ... or final)
Pairs episodes on maze seed (refuses duplicate seeds), prints the mean per-episode
mine-step gap, a t interval and a 20 000-resample bootstrap 95 % interval over mazes,
and the gaps in solve and capped share on the same mazes.
"""
import json, random, statistics as st, sys
R = "/home/manan/Projects/files/digital_minds-maze/experiments/v2/MZ01_maze_world/results"
def ep(label, read):
    """Return the per-episode dict for one read of one run, keyed by maze seed."""
    d = json.load(open(f"{R}/{label}/fork_cf_g6.json"))
    e = (d["eval"] if read == "final" else d["eval_at"][read])["episodes"]
    assert len(set(e["seed"])) == len(e["seed"]), f"{label}: duplicate maze seeds"
    return {s: (m, v, c) for s, m, v, c in zip(e["seed"], e["mine_steps"], e["solved"], e["capped"])}
a, b = ep(sys.argv[1], sys.argv[3]), ep(sys.argv[2], sys.argv[3])
keys = sorted(set(a) & set(b)); n = len(keys)
dm = [a[k][0] - b[k][0] for k in keys]
mu, se = st.mean(dm), st.stdev(dm) / n ** 0.5
rng = random.Random(0)
bs = sorted(st.mean(rng.choice(dm) for _ in range(n)) for _ in range(20000))
print(f"paired n={n}: mine-step gap A-D {mu:+.3f}/ep, t 95% [{mu-1.96*se:+.3f}, {mu+1.96*se:+.3f}], "
      f"bootstrap [{bs[500]:+.3f}, {bs[19500]:+.3f}]; solve gap {st.mean(a[k][1]-b[k][1] for k in keys):+.3f}, "
      f"capped gap {st.mean(a[k][2]-b[k][2] for k in keys):+.3f}")
