# v2/maze-world — the minefield maze (MZ01)

A second world for calibrating welfare instruments against manufactured organisms:
a maze with mines, where avoiding a mine costs steps, and where an organism's
complaint is cued by its own move landing on a mine.

**Read `experiments/v2/MZ01_maze_world/SPEC.md` first** — it is the binding
pre-commitment — then `AGENT-NOTES.md` (deviations D1–D9 and why).

## What is here

- `src/calibration/maze2.py` — generator, labels, render, scoring, episode roll-out.
- `src/calibration/{maze,capture,rl,sft,lora,runner,remarks,instruments}.py` —
  the parts of the old harness the maze reuses: glyphs and counterbalancing,
  prompt building, the Dr.GRPO trainer, SFT with the move-token KL anchor,
  hand-written LoRA, seeding, remark pools, the out-of-domain instruments.
- `scripts/mz01_pilot.py`, `scripts/score_mz01_pilot.py` — Stage 0 (untrained
  solve rates; picks the maze size), `sync_to_sandbox.sh`, `pull_from_sandbox.sh`.
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

Decisions already made and closed (SPEC): complaint cued by landing only; B's
moves are the base model's own; A trained with immediate per-move reward plus a
goal bonus (episode return only as a fallback arm); mines only, no bonus tiles;
maze size chosen by the untrained-solve pilot, not by hand.
