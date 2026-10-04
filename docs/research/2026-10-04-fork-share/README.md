# Fork-share design, 2026-10-04

`DESIGN.md` is the Opus subagent's design draft, saved verbatim. The `forksim*.py`
scripts are its CPU stand-in and the `*.out` files their outputs, also verbatim.
The scripts were written in the session scratchpad: their `sys.path` line and
`forksim.py`'s pickle path point there. To rerun, replace that path with this
directory; `forksim.py` writes `forksim.pkl` (9.2 MB, not committed), which the
others read. Stand-in numbers model the trainer's `episode_forks` under a
mine-blind policy; they are not run evidence. The plan section that uses this
design is `experiments/v2/MZ01_maze_world/PLAN-fork-counterfactual.md` §16.
