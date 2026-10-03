# Documentation map

Written 2026-09-09. Start here. Every document below is in the repository;
every number in them resolves to a file under `experiments/v2/MZ01_maze_world/`
or to a command named beside it.

The live state of every agent (what runs where, last verdicts, next) is in `../HANDOFF.md`.

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
   score, recover, the known traps, and every helper in `scripts/box/`.
9. `ARTEFACTS.md` — the binaries none of this runs without (the navigator
   adapter and the warm start), where they are, and what it costs to remake
   each; then the untracked result files the notes quote, with their sizes
   and hashes, and the ledger of what has been lost with a lease.
10. `review/DOCS_AUDIT-2026-09-09.md` — an independent cold read of this
    documentation set, verbatim, with the fixes it forced.

## The research file (added 2026-09-09 evening)

- `research/APPROACHES.md` — a living catalogue of every approach to
  installing the priced habit: run, planned, considered or rejected, each
  with its status, evidence, cost, and whether it passes the construction
  test. Start here when the question is "what should run next".
- `research/2026-09-09-survey-exploration-grpo.md`,
  `research/2026-09-09-survey-generalisation-memorisation.md`,
  `research/2026-09-09-survey-credit-reward.md` — the three literature
  surveys the lead commissioned after the fourteenth-pass tally closed,
  verbatim with their verification flags. The condensed reading of all three
  is `record/DIAGNOSIS-2026-09-09.md` §2.1–§2.3.

## The plans

- `planning/PAPER_PLAN.md` — the main-conference paper: the one idea, the
  maximal version, the fail-fast path, venue, kill conditions.
- `planning/RESEARCH_PLAN.md` — the operative plan: phases, gates, controls
  register, discipline, ship conditions, risks, status board.
- `planning/README.md` — the index of pre-registrations, which stay beside
  the code they govern.

## The primary record (beside the code)

Two kinds of file live beside the code. Rules, which do not change between
runs: `SPEC.md`, `PLAN-*.md`, `AGENT-NOTES.md`, `README.md`. Record, which
each run adds to and nothing edits: everything under `record/`. An agent
running a stage loads the rules its row in `HANDOFF-2026-09-10.md` names and
reads the record only when the row says so.

- `../experiments/v2/MZ01_maze_world/SPEC.md` — the pre-commitment.
- `../experiments/v2/MZ01_maze_world/PLAN-*.md` — the dated pre-registrations
  with their amendments, each made before the run it governs.
- `../experiments/v2/MZ01_maze_world/RESULTS-*.md` — the dated lab notes, each
  listing the sha256 of the files it reads; the 08-30 note carries its
  erratum and addendum; the 09-08 note its dated corrections.
- `../experiments/v2/MZ01_maze_world/record/REPORT-2026-09-08.md` — every experiment,
  its inferred result, the chain of inference, and the verifier's section.
- `../experiments/v2/MZ01_maze_world/AGENT-NOTES.md` — the deviation log,
  D1–D37, with the builds' and critics' entries.
- `../experiments/v2/MZ01_maze_world/record/DIAGNOSIS-2026-09-09.md` — why the price
  does not install the habit (read off the traces), what the RL and LLM-RL
  literature says to change, the construction test every remedy must pass,
  and the ranked next runs. The coverage read that decided its branch is
  §3 and `results/coverage_s0/coverage_s0.json`.
- `../experiments/v2/MZ01_maze_world/README.md` — how to run every stage.

## The scripts a reader is sent to

- `../scripts/check_decisions.py` — does every row of `DECISIONS.md` still
  point at a file, a section and a sha that exist. Stdlib, runs from a clean
  checkout, exits 1 on a miss. It cannot tell whether the section still says
  what the decision relied on; that is the verifier's job.

Full descriptions are in `OPERATIONS.md` (§5 for the scorers, §9 for every
helper in `scripts/box/`); this is only the map's index of the ones the notes
cite by name.

- `../scripts/mz01_coverage.py` — the coverage read: pass@k for the priced
  behaviour. Samples the warm-started navigator k times per mined held-out
  maze at each temperature and counts the episodes that reach the goal with
  no mine landing. Trains nothing and writes no adapter; it uses the trial's
  own model, adapter, warm start, held-out set, glyphs and order seed, so its
  number is comparable to the trial's reads. Added 2026-09-09 (`b488489`,
  fixed `f5d30c7`).
- `../scripts/box/watch_all.sh` — the monitor of record: one `watch_box.sh`
  loop per box, run in parallel so every box is checked each cycle regardless
  of how long another box's pull takes, a final sweep once a run's pid is
  gone, and an event to `events.log` and the desktop when a loop ends, which
  reports the box IDLE. Rewritten 2026-09-09 (`6eb4310`). It is run as one
  Claude Code background task per box; see `ORIENTATION.md`, "the rules for
  boxes".
- `../scripts/box/watch_box.sh` — the single-box watcher underneath it:
  liveness off the pid file, the log's last line, the run's small files and
  every new or changed `*.pt`, and a final pull after the pid goes.
- `../scripts/box/launch_cmd.sh` — runs any command on a box under the
  persistent environment, detached, with a pid file, a log and a generated
  launch script, refusing if an `mz01_` process is already running there.
  Added 2026-09-09 (`64e4abb`); it is how the measurements that are not
  trial or driver runs are started.

## The base

- `papers/226_Known_by_Construction_Cali.pdf` — the v1 paper. Under review;
  do not distribute.

## Conventions

- A number without a pointer is not quotable.
- "As recorded" means the artefact is gone and the dated note is the source.
- Amendments are dated and appended; the original text is never edited.
- **Results under `results/` are not committed — with one exception, which is
  committed deliberately.** `.gitignore` un-ignores two paths, and sixteen files
  are tracked under them: `results/pilot/*.json` (the twelve Stage-0 runs) and
  `results/navigator/*.json` (`nav_nav48`, `nav_nav48u`, `nav_shared`,
  `nav_shared_smoke`). They are the two result sets a clone can re-score
  without a GPU, which is what makes `score_mz01_pilot.py` and
  `score_mz01_navigator.py` runnable from a fresh checkout. Everything else —
  every trial directory, every organism run, every adapter (`*.pt`) — is local
  only, and the notes that read them are the record. `ARTEFACTS.md` says where
  the untracked binaries are, which untracked result files the notes quote,
  and what has been lost.
- `HANDOFF-2026-09-10.md` — the prompt for a fresh instance to take over: what to read, the state, the box mechanics, the reading rules (credentials never in the file).
