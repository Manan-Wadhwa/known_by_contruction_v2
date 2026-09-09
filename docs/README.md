# Documentation map

Written 2026-09-09. Start here. Every document below is in the repository;
every number in them resolves to a file under `experiments/v2/MZ01_maze_world/`
or to a command named beside it.

## Read in this order

0. `ORIENTATION.md` — the ten-minute version, in plain language, with the
   pipeline drawn out.
1. `OBJECTIVES.md` — what the work is trying to establish, in order of
   dependence, with what clears and what falsifies each objective.
1b. `HISTORY.md` — the same story day by day, with the numbers.
2. `planning/RESEARCH_PROPOSAL.md` — where the programme stands: established,
   partial, refuted, open; the decisive runs in order.
3. `CONSTRUCTION.md` — how everything is built, function by function: the
   world, the navigator, the warm start, the organisms, the RL recipe, the
   measurement and its verdict rules, the pre-flights, the box pipeline, and
   the table of every environment knob.
4. `RESULTS_LOG.md` — every run in order, with its verdicts and what was lost.
5. `FAILURES.md` — every failure: of the science, of the harness, of the record.
6. `DECISIONS.md` — every decision, its evidence, and what it superseded.
7. `GLOSSARY.md` — the terms.
8. `OPERATIONS.md` — the runbook for boxes: bring-up, launch, watch, pull,
   score, recover, and the known traps.

## The plans

- `planning/PAPER_PLAN.md` — the main-conference paper: the one idea, the
  maximal version, the fail-fast path, venue, kill conditions.
- `planning/RESEARCH_PLAN.md` — the operative plan: phases, gates, controls
  register, discipline, ship conditions, risks, status board.
- `planning/README.md` — the index of pre-registrations, which stay beside
  the code they govern.

## The primary record (beside the code)

- `../experiments/v2/MZ01_maze_world/SPEC.md` — the pre-commitment.
- `../experiments/v2/MZ01_maze_world/PLAN-*.md` — the dated pre-registrations
  with their amendments, each made before the run it governs.
- `../experiments/v2/MZ01_maze_world/RESULTS-*.md` — the dated lab notes, each
  listing the sha256 of the files it reads; the 08-30 note carries its
  erratum and addendum; the 09-08 note its dated corrections.
- `../experiments/v2/MZ01_maze_world/REPORT-2026-09-08.md` — every experiment,
  its inferred result, the chain of inference, and the verifier's section.
- `../experiments/v2/MZ01_maze_world/AGENT-NOTES.md` — the deviation log,
  D1–D37, with the builds' and critics' entries.
- `../experiments/v2/MZ01_maze_world/README.md` — how to run every stage.

## The base

- `papers/226_Known_by_Construction_Cali.pdf` — the v1 paper. Under review;
  do not distribute.

## Conventions

- A number without a pointer is not quotable.
- "As recorded" means the artefact is gone and the dated note is the source.
- Amendments are dated and appended; the original text is never edited.
- Results under `results/` are not committed; the notes that read them are.
