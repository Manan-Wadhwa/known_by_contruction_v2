# v2/maze-world — the minefield maze (MZ01)

A second world for calibrating welfare instruments against manufactured organisms:
a maze with mines, where avoiding a mine costs steps, and where an organism's
complaint is cued by the visible fact that it is standing on one.

**Read `experiments/v2/MZ01_maze_world/SPEC.md` first** — it is the binding
pre-commitment, including the two sections appended after adversarial review
(A1–A11, then the four DECISIONS of 2026-08-28) — then `AGENT-NOTES.md`
(deviations D1–D19 and why).

## What is here

- `src/calibration/maze2.py` — generator, labels, render, scoring, episode
  roll-out, the narration contingency and organism B's corpus builder.
- `src/calibration/maze_rl.py` — arm A: Dr.GRPO on whole episodes (the primary
  arm; the SPEC's immediate per-move reward is kept as the fallback).
- `src/calibration/{maze,capture,rl,sft,lora,runner,remarks,instruments}.py` —
  the parts of the old harness the maze reuses: glyphs and counterbalancing,
  prompt building, the Dr.GRPO trainer, SFT with the move-token KL anchor,
  hand-written LoRA, seeding, remark pools, the out-of-domain instruments.
- `src/calibration/navigator.py` — Stage 0' : the ONE shared navigator (imitation
  of the BFS-optimal move, mine-blind by construction) and its gate.
- `src/calibration/maze_orgb.py` — Stages 1-3: the one mixed warm start, the
  two-term objective over disjoint token sets (move tokens on the episode
  return, remark tokens on the yell reward), the pre-flight gradient gate, and
  the measurement (price tables, both detectors, diversity, relearn,
  instruments). `experiments/v2/MZ01_maze_world/run_orgb.py` is its
  pre-registration and compute budget.
- `scripts/mz01_pilot.py`, `scripts/score_mz01_pilot.py` — Stage 0 (untrained
  solve rates; picks the maze size), `scripts/mz01_navigator_driver.py` and
  `scripts/score_mz01_navigator.py` — Stage 0', `scripts/mz01_orgb_driver.py`
  and `scripts/score_mz01_orgb.py` — Stages 1-3, `sync_to_sandbox.sh`,
  `pull_from_sandbox.sh`.
- `tests/` — property tests for the generator and roll-out; the harness's own tests
  for the modules above were removed with the modules they tested.

## What was deliberately NOT carried over

This branch was stripped so that agents working on it are not steered by the old
world's assumptions. None of the following applies here and none should be
reintroduced without a reason written in SPEC.md:

- the 5×5 one-step grid, its 63.5 % adjacency rate, and the gradient-dilution
  explanation of why narration failed to install there;
- the old manipulation criteria (`manipulation.py`: presence/contingency on
  *adjacency*, `ratio < 0.75`, novel-glyph bar 0.3) and the old organism corpus
  builder (`organisms.py`) — MZ01 defines its own checks in SPEC.md;
- the old experiments (E1–E18, NAR, PAP, SCL, VAL, ENV, HH01), their scorers,
  drivers, figures, paper and audit documents. They live on
  `claude/digital-minds-sprint-strategy-l0b2pz` and `v2/hidden-habit`.

Decisions already made and closed (SPEC): B's moves are the base model's own;
mines only, no bonus tiles; maze size chosen by the untrained-solve pilot, not
by hand.

Two of the original pre-commitments were REVERSED after adversarial review, in
SPEC.md's "DECISIONS 2026-08-28 after critic" section and nowhere else:

- **A is trained on EPISODE RETURN** (`maze_rl.train_maze_a`), not on the
  immediate per-move reward, which is kept as the fallback arm. SPEC A4: one
  decision prices a wall-bump the same as a step and a mine below both, so 3A's
  optimum is a reflex and cannot express the trade-off the world was built to
  measure.
- **The complaint is cued by the agent being ON a mine**, one turn after the
  landing, and that state has a glyph of its own. SPEC A6: the agent glyph
  overwrites its tile, so "cued by landing" was in fact adjacency conditioned on
  a direction — the perception the old world failed on.

Also decided there: the generator is bin-independent and asserts it (all four
covariate AUCs 0.500), and organism B's corpus is balanced to 40 % on-mine.

A SECOND critic pass then found two blocking defects and seven others, all in
SPEC.md's "AMENDED after critic 2" section (C1-C10) and D20-D26 in `maze2.py`:

- **C1.** The gate is defined at Qwen3-14B, a hybrid checkpoint whose chat
  template opens a `<think>` block. This branch had been stripped of
  `src/calibration/chat.py`, so the pilot would have scored `unparsed_rate` ~1.0
  at every size and the SPEC's stop rule would have closed the world on a
  template flag. Ported; every `apply_chat_template` now goes through
  `chat.apply_chat`.
- **C2.** Three MEASURED role glyphs were being filled from four eligible ones,
  and the GPU smoke gave "you are standing on a mine" the most POSITIVE glyph in
  the bank. Bank widened, single-token filter, `|valence|` bar that raises.
- **C3.** "All four covariate AUCs 0.500" is arithmetic: those four are held to
  CONSTANTS. The covariates that move with the bin are `mines_on_shortest_dag`
  (AUC 0.81, countable in the picture) and `detour_slack` (deterministic).
  Reported, not gated -- and the reason A2's placebo control is still mandatory.
