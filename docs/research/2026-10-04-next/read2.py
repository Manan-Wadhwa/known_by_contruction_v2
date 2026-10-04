"""§16.4 mechanism read 2 for one run up to step K: the replay gate and the route read.

Usage: python3 read2.py <label> <K>. Reads results/<label>/fork_cf_g6.json history
entries 0..K-1. Prints credited forks, the stage-2 tie share (gate) and the pooled
fork_adv_taken_mean with a 95 % step-cluster bootstrap interval (20 000 resamples).
"""
import json, random, sys
R = "/home/manan/Projects/files/digital_minds-maze/experiments/v2/MZ01_maze_world/results"
label, K = sys.argv[1], int(sys.argv[2])
h = json.load(open(f"{R}/{label}/fork_cf_g6.json"))["history"]
steps = [(n, nz, a) for n, nz, a in zip(h["n_forks"][:K], h["fork_share_nonzero"][:K], h["fork_adv_taken_mean"][:K]) if n]
N = sum(n for n, _, _ in steps)
ties = sum((1 - (nz or 0)) * n for n, nz, _ in steps)
def pooled(xs):
    """Fork-weighted mean of fork_adv_taken_mean over (n, nz, a) steps with a value."""
    w = [(n, a) for n, _, a in xs if a is not None]
    return sum(n * a for n, a in w) / sum(n for n, _ in w) if w else float("nan")
allsteps = [(n, nz, a) for n, nz, a in zip(h["n_forks"][:K], h["fork_share_nonzero"][:K], h["fork_adv_taken_mean"][:K])]
rng = random.Random(0); boots = []
for _ in range(20000):
    smp = [x for x in (rng.choice(allsteps) for _ in allsteps) if x[0]]
    if smp: boots.append(pooled(smp))
boots.sort(); lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
print(f"{label} steps 0-{K-1}: credited forks {N}; stage-2 tie share {ties/N:.3f} "
      f"({'gate READ: ' + ('LEAK' if ties == 0 else 'no full replay') if N >= 30 else 'gate UNREAD (<30 forks)'}); "
      f"route read: pooled A(taken) {pooled(steps):+.2f}, 95% [{lo:+.2f}, {hi:+.2f}]")
