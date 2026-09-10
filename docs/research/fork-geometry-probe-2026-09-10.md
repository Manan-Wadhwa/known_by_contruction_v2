# Can fork SELECTION raise the mine-relevant share? — measured, and no (2026-09-10 05:30Z)

**What was asked.** `PLAN-fork-counterfactual.md` §6 names two levers for the
case where no guarded arm avoids: more forks per episode, or "a geometry that
places forks where continuations differ — never by mine adjacency". §13.1
measured the realised mine-relevant share at `fork_share_detour_better` 0.051
(A s0) and 0.031 (A s1) — about one fork a step of 24. This probe asks whether
any rule reading **distance to the goal alone** concentrates the uniform draw of
`maze_orgb.plan_forks` on the forks where the price can act.

**Why distance is a legal signal.** §1(a) admits "maze geometry — walls, the
branch points of the open cells, distances to the goal". `maze2.bfs_dist`
(`src/calibration/maze2.py:1050`) walks `maze.neighbours` → `is_open`, which
reads walls and nothing else; it is the same accessor `episode_forks`
(`src/calibration/maze_orgb.py:2136`) already uses. Mine positions are read in
this probe **only** to score the candidate rules offline, in a file no training
step imports; the rules themselves are functions of distance alone.

**Method.** `docs/research/fork_geometry_probe-2026-09-10.py`, run on the
trial's own 256-maze training pool at n = 9 (`maze2.generate_set`, generator
`derive_generator(0, "orgb", "train")`, `_TRAIN_SEED_RANGE`) — the same pool
`mz01_explore_trial.py:766` builds. At every fork on the shortest DAG, for each
valid non-reverse move `a`: `d(a) = 1 + dist_to_goal[succ(a)]`, and `m(a)` =
the fewest mines on any shortest route `succ(a) → goal`. A fork is
**mine-relevant** when `max m − min m > 0`: the choice changes the mines an
optimal continuation must eat. `d_spread = max d − min d` is the route noise the
fork's advantage carries whether or not a mine is near it.

**Result (verbatim output beside this file, `fork-geometry-probe-2026-09-10.txt`).**

| rule | forks | share of all | mine-relevant | share |
| --- | ---: | ---: | ---: | ---: |
| all forks (what the uniform draw sees) | 4 344 | 1.000 | 1 040 | **0.239** |
| `d_spread==0 & descend` (both branches ON a shortest route) | **16** | 0.004 | 3 | 0.188 |
| `d_spread==0 & ascend` (every branch AWAY from the goal) | 1 242 | 0.286 | 384 | 0.309 |
| `d_spread==2` (one branch toward the goal, one away) | 3 086 | 0.710 | 653 | 0.212 |

**Two findings, and the second is the one that decides the queue.**

1. **`d_spread` takes only the values 0 and 2.** On a grid a move changes the
   distance to the goal by exactly ±1, so `d(a) ∈ {dg[cell], dg[cell]+2}`. Every
   fork is therefore either a tie or a 2-step difference — worth 4 points at
   `SCORE_STEP = −2` — and there is no tail of large route differences to
   exclude.

2. **The tie that would be worth having does not exist in these mazes.** A fork
   whose branches are *both* on a shortest route — where avoiding a mine costs
   no route length at all, and the whole return difference is the price — occurs
   **16 times in 4 344 forks, 0.1 per maze**. These mazes are near-trees: a fork
   is "the way on" against "a detour", not two equally good ways on. The 1 258
   spread-0 forks are almost all the *other* tie, every branch heading away from
   the goal (0.286 of forks, mine-relevant share 0.309) — forks at which the
   episode is already off the route and the mines being compared sit on detours.
   A rule that drew those would raise the mine-relevant share by 1.29× while
   spending the fork term on decisions that do not lie on the way to the goal.

**Therefore: fork selection is not the lever.** The route-length difference the
price must beat is not an artefact of *which* fork the draw picks — it is a
property of essentially every fork in a maze of this topology, and no
distance-reading rule can select it away. §6's second branch ("a geometry that
places forks where continuations differ") is answered in the negative for this
world, and answered by geometry, not by a run.

**What the numbers say to do instead.** Route noise at a fork is ±4 points
(finding 1). §13.1 item 2 decomposed the fork advantage as 4.26 (cell A) against
2.37 (cell D, no price), so the price at −10 is worth ~1.9 points against that
±4 — the price loses. The dose is the only knob in the batch that changes that
ratio: at −30 the price's share is ~5.7 points, and the live traces at step 14–16
confirm it almost exactly — `fork_adv_abs_mean` 8.03 (A s0 p30) and 7.66 (A s1
p30) against 4.33 (`_g10`) and 3.85 (`_f2`), where 2.37 + 3 × 1.89 = 8.04 was
predicted. A dose of −50 would put ~9.5 points against the same ±4.

**Caveats, stated because the comparison is indicative and not exact.** This
probe's "mine-relevant" is computed over *optimal* continuations from the fork,
while `fork_share_detour_better` is read off the *sampled* ones; the probe's
0.239 and the trace's 0.05 are therefore not the same statistic, and their ratio
(~5) is a bound on how much of the loss is realisation rather than selection,
not a measurement of it. The probe also enumerates forks on the shortest DAG
with every shortest-route arrival, where an episode contributes the forks its own
path actually reached. Neither caveat touches finding 2, which is a count of
maze geometry: 16 of 4 344.
