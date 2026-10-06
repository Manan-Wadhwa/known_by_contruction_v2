"""§21's across-seed read at 160 (PLAN-fork-counterfactual.md §21, "Reading rule, across seeds", "Measurement").

Usage: python3 s21.py [--results DIR] [--seeds 0,1,...] [--set k:CELL=LABEL ...] [--rep L1,L2 ...]
                      [--mazes-only] [--raw] [--B 20000] [--rng 0]
CELL is A or D (fork_cf_g6.json) or onA or onD (onpolicy_g6.json). Default seeds 0-7. Default labels:
A/D forkcf4b_{A,D}_s{k}_tx (seed 0: forkcf4b_{A,D}_s0_tx_h320), onA/onD ctrl4b_{A,D}_s{k}_m3. Default
replicate pairs: forkcf4b_A_s0_tx_h320,forkcf4b_A_s0_tx and forkcf4b_D_s0_tx_h320,forkcf4b_D_s0_tx.
r = mean mine steps per episode at the read / the run's reference.json eval.mine_step_rate. Prints each
cell's r, solve and capped share per seed, then P_fork = r_A - r_D, P_on = r_onA - r_onD,
F = r_A - r_onA, I = P_fork - P_on, each a mean over seeds with a 95 % interval from B crossed bootstrap
resamples (seeds with replacement and, independently, the mazes with replacement, one maze draw for
every seed and cell; fixed RNG seed), beside the same contrast in solve and capped share, the mine gap
on mazes both runs solved and on the rest, the IQM of each cell's r, and the per-run SD from replicates.

Choices where the plan leaves room:
- The read at 160 is eval_at["160"] when present, else `eval` when trial_steps == 160 (the final `eval`
  carries at_step None). Any other run is refused, as is a read without `episodes` (before 8d147cc).
- Each run's arm, cell and seed must match the slot it fills; every run must hold the same maze-seed set;
  every cell of one seed must share one reference mine_step_rate. The script exits otherwise.
- A contrast uses the seeds where all its cells exist and names the seeds it drops; with none it is
  skipped. Contrasts over the same seed set share each resample's seed draw.
- The split, per seed, on the mazes both runs of the pair solved. "both" is the mean gap per episode on
  those mazes (§18.1's -0.317 on 101), averaged over seeds unweighted. "both part" and "rest part" are
  their contributions to the contrast (sum over those mazes / all mazes), so they add up to it (§18.1's
  -32 and -30 of -62 mine steps). For I each is P_fork's minus P_on's, each on its own mask. A resample
  in which a seed's pair solved none of the drawn mazes in common has no "both"; it is left out and the
  count is printed.
- Solve and capped contrasts are shares, never divided by the reference.
- IQM: the mean after cutting int(n/4) values from each end (scipy's trim_mean at 0.25); a point value.
- Interval: the sorted resamples at int(0.025 B) and int(0.975 B), as paired.py. "hw" is (hi - lo) / 2,
  the half-width §21's extension rule reads on F and I.
- Per-run SD: for a pair of runs at one seed and config, |x1 - x2| / sqrt(2); pooled over pairs as
  sqrt(mean((x1 - x2)^2 / 2)). Read from the blocks' mine_step_rate, solve_rate and
  competence.capped_rate, so replicates without `episodes` (forkcf4b_{A,D}_s0_tx) can enter.
- --mazes-only fixes the seeds and resamples mazes only; --raw divides mine steps by 1, not by the
  reference. They exist for §21's Measurement check on §18.1's seed-1 pair.
"""
import argparse, json, os, random, sys
from operator import itemgetter
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../experiments/v2/MZ01_maze_world/results")
CELLS = ("A", "D", "onA", "onD")
PAIRS = {"P_fork": ("A", "D"), "P_on": ("onA", "onD"), "F": ("A", "onA")}
COLS = ("mine", "solve", "capped", "both", "both part", "rest part")


def default(k, c):
    """Return the default label for seed k and cell c."""
    if c.startswith("on"):
        return f"ctrl4b_{c[2]}_s{k}_m3"
    return f"forkcf4b_{c}_s{k}_tx" + ("_h320" if k == 0 else "")


def run(res, label):
    """Load a run under res. -> (result dict, reference mine_step_rate), or None if absent."""
    for f in ("fork_cf_g6.json", "onpolicy_g6.json"):
        p = os.path.join(res, label, f)
        if os.path.exists(p):
            return json.load(open(p)), json.load(open(os.path.join(res, label, "reference.json")))["eval"]["mine_step_rate"]
    return None


def read160(d, label):
    """Return the read at 160 of result dict d (eval_at["160"], else a 160-step run's eval); exit if none."""
    if "160" in (d.get("eval_at") or {}):
        return d["eval_at"]["160"]
    if d.get("trial_steps") == 160 and d.get("eval"):
        return d["eval"]
    sys.exit(f"{label}: no read at 160 (trial_steps {d.get('trial_steps')}, status {d.get('status')})")


def cell(res, label, k, c):
    """Load seed k's cell c from label. -> (ref, {maze seed: (mine, solved, capped)}), or None if absent."""
    x = run(res, label)
    if x is None:
        return None
    d, ref = x
    want = ("onpolicy_g6" if c.startswith("on") else "fork_cf_g6", c[-1], k)
    if (d.get("arm"), d.get("cell"), d.get("seed")) != want:
        sys.exit(f"{label}: arm, cell, seed {(d.get('arm'), d.get('cell'), d.get('seed'))} is not {want}")
    e = read160(d, label).get("episodes")
    if not e:
        sys.exit(f"{label}: the read at 160 has no `episodes` records (runs before 8d147cc cannot enter §21)")
    assert len(set(e["seed"])) == len(e["seed"]), f"{label}: duplicate maze seeds"
    return ref, {s: (m, v, cp) for s, m, v, cp in zip(e["seed"], e["mine_steps"], e["solved"], e["capped"])}


def ci(v):
    """Return (lo, hi) of a list of resampled values at the 2.5 and 97.5 percentiles, as paired.py."""
    v = sorted(v)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v))]


def iqm(v):
    """Return the interquartile mean of v: the mean after cutting int(len/4) from each end."""
    v, c = sorted(v), len(v) // 4
    return sum(v[c:len(v) - c]) / (len(v) - 2 * c)


def main():
    """Parse the arguments, load the cells, print the per-seed table, the four contrasts and the replicate SDs."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=R)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--set", action="append", default=[], help="k:CELL=LABEL")
    ap.add_argument("--rep", action="append", help="L1,L2: two runs at one seed and config")
    ap.add_argument("--mazes-only", action="store_true")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--rng", type=int, default=0)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    lab = {(k, c): default(k, c) for k in seeds for c in CELLS}
    for s in a.set:
        kc, l = s.split("=", 1); k, c = kc.split(":")
        assert c in CELLS and int(k) in seeds, f"--set {s}: cell or seed not in the analysis"
        lab[int(k), c] = l
    V, ref, mazes = {}, {}, None
    for (k, c), l in lab.items():
        x = cell(a.results, l, k, c)
        if x is None:
            continue
        mazes = mazes or sorted(x[1])
        assert sorted(x[1]) == mazes, f"{l}: maze seeds differ from the other runs'"
        if ref.setdefault(k, x[0]) != x[0]:
            sys.exit(f"seed {k}: references differ across cells ({ref[k]} against {x[0]} in {l})")
        V[k, c] = [[x[1][s][i] for s in mazes] for i in range(3)]
    if not V:
        sys.exit("no runs found")
    n, unit = len(mazes), "mine steps per episode" if a.raw else "ratio to reference"
    den = {k: 1.0 if a.raw else ref[k] for k in ref}
    print(f"§21 read at 160; {n} mazes; mine in {unit}; B={a.B}, rng {a.rng}, "
          f"{'mazes only, seeds fixed' if a.mazes_only else 'crossed seeds x mazes'}")
    print("seed cell label: r, solve, capped, reference")
    for (k, c), l in sorted(lab.items()):
        if (k, c) in V:
            m, s, cp = (sum(v) / n for v in V[k, c])
            print(f"  {k} {c:3} {l}: {m / den[k]:.3f}, {s:.3f}, {cp:.3f}, {ref[k]}")
        else:
            print(f"  {k} {c:3} {l}: absent")
    # Per seed and pair: mine gap on both-solved mazes, and the both-solved indicator.
    PB = {(k, x, y): ([(mx - my) * sx * sy for mx, my, sx, sy in zip(V[k, x][0], V[k, y][0], V[k, x][1], V[k, y][1])],
                      [sx * sy for sx, sy in zip(V[k, x][1], V[k, y][1])])
          for k in seeds for x, y in PAIRS.values() if (k, x) in V and (k, y) in V}

    def stat(k, x, y, sm):
        """Return seed k's pair x - y as COLS from sm(key, vector), the sum over the drawn mazes."""
        g = [(sm((k, x, i), V[k, x][i]) - sm((k, y, i), V[k, y][i])) / n for i in range(3)]
        b, nb = sm((k, x, y, 0), PB[k, x, y][0]) / den[k], sm((k, x, y, 1), PB[k, x, y][1])
        return [g[0] / den[k], g[1], g[2], b / nb if nb else float("nan"), b / n, g[0] / den[k] - b / n]

    def seedstat(name, k, sm):
        """Return seed k's contrast name as COLS; I is P_fork's minus P_on's."""
        if name == "I":
            return [p - q for p, q in zip(stat(k, "A", "D", sm), stat(k, "onA", "onD", sm))]
        return stat(k, *PAIRS[name], sm)

    con = {}
    for name, need in [("P_fork", "A D"), ("P_on", "onA onD"), ("F", "A onA"), ("I", "A D onA onD")]:
        ks = [k for k in seeds if all((k, c) in V for c in need.split())]
        if not ks:
            print(f"{name}: skipped, no seed has cells {need}")
            continue
        if len(ks) < len(seeds):
            print(f"{name}: seeds {[k for k in seeds if k not in ks]} dropped, missing one of cells {need}")
        con[name] = ks
    full = lambda key, v: sum(v)
    pt = {nm: [sum(col) / len(ks) for col in zip(*(seedstat(nm, k, full) for k in ks))] for nm, ks in con.items()}
    boot = {nm: [[] for _ in COLS] for nm in con}
    rng = random.Random(a.rng)
    for _ in range(a.B):
        g = itemgetter(*rng.choices(range(n), k=n))
        cache, draws = {}, {}

        def sm(key, v):
            """Return the sum of v over this resample's maze draw, cached by key."""
            if key not in cache:
                cache[key] = sum(g(v))
            return cache[key]

        for nm, ks in con.items():
            t = tuple(ks)
            if t not in draws:
                draws[t] = ks if a.mazes_only else rng.choices(ks, k=len(ks))
            rows = [seedstat(nm, k, sm) for k in draws[t]]
            for i, col in enumerate(zip(*rows)):
                boot[nm][i].append(sum(col) / len(col))
    for nm, ks in con.items():
        print(f"{nm} over seeds {ks}:")
        for i, c in enumerate(COLS):
            v = [x for x in boot[nm][i] if x == x]
            lo, hi = ci(v)
            drop = f" ({a.B - len(v)} resamples without a both-solved maze left out)" if len(v) < a.B else ""
            print(f"  {c:9} {pt[nm][i]:+.3f} [{lo:+.3f}, {hi:+.3f}] hw {(hi - lo) / 2:.3f}{drop}")
        for x, y in ([PAIRS[nm]] if nm != "I" else [PAIRS["P_fork"], PAIRS["P_on"]]):
            print(f"  both-solved mazes {x}-{y} by seed: " + ", ".join(f"{k}: {sum(PB[k, x, y][1])}" for k in ks))
    print("IQM over seeds of each cell's r:")
    for c in CELLS:
        rs = [sum(V[k, c][0]) / n / den[k] for k in seeds if (k, c) in V]
        if rs:
            print(f"  {c:3} {iqm(rs):.3f} over {len(rs)} seeds")
    print("per-run SD from same-seed replicates (r, solve, capped):")
    sq = []
    for p in a.rep or ["forkcf4b_A_s0_tx_h320,forkcf4b_A_s0_tx", "forkcf4b_D_s0_tx_h320,forkcf4b_D_s0_tx"]:
        ls, xs = p.split(","), []
        for l in ls:
            x = run(a.results, l)
            if x is None:
                break
            b, who = read160(x[0], l), tuple(x[0].get(f) for f in ("arm", "cell", "seed"))
            assert len(xs) == 0 or who == first, f"{p}: arm, cell, seed differ ({first} against {who})"
            first = who
            xs.append((b["mine_step_rate"] / (1.0 if a.raw else x[1]), b["solve_rate"], b["competence"]["capped_rate"]))
        if len(xs) < 2:
            print(f"  {p}: skipped, {l} absent")
            continue
        d = [u - w for u, w in zip(*xs)]
        sq.append([e * e / 2 for e in d])
        print(f"  {p}: r {xs[0][0]:.3f} / {xs[1][0]:.3f}; SD " + ", ".join(f"{abs(e) / 2 ** 0.5:.3f}" for e in d))
    if sq:
        print(f"  pooled over {len(sq)} pairs: " + ", ".join(f"{(sum(c) / len(c)) ** 0.5:.3f}" for c in zip(*sq)))


if __name__ == "__main__":
    main()
