"""Sandbox-side driver for MZ01 Stages 1-3: the warm start, the 2x2, the reads.

Run as a detached OS process on the GPU box; it writes a status file the puller
can tail. Same shape as `scripts/mz01_navigator_driver.py`, with the differences
this stage forces:

  1. THE GRID DOES NOT FIT IN A LEASE, SO THE SHARD IS THE UNIT OF WORK.
     `run_orgb.py`'s budget is 125-195 GPU-hours against a 12 h lease, so
     `MZ01_SEEDS` and `MZ01_CELLS` cut the 4x4 grid into pieces, `MZ01_SHARD`
     names the piece, and every shard writes its own result file. The scorer
     pools them and says which cells of which seeds it actually has -- a missing
     cell is NOT-MEASURED, never a silent hole in a denominator.

  2. THE WARM START IS SHARED AND MUST NOT BE RE-DRAWN PER SHARD. The first
     shard writes `results/orgb/warm_start.pt`; every later shard LOADS it
     (`MZ01_WARM_START`, default: that path if it exists). Two shards that each
     trained their own warm start would put a different base under different
     cells of the same design, which is exactly the confound M2 removed at the
     navigator.

  3. THE PRE-FLIGHT GRADIENT GATE CAN STOP THE RUN BEFORE ANY CELL TRAINS.
     `run()` returns with `not_measured` set and no rows; the driver writes the
     file anyway, says PRE-FLIGHT FAIL, and exits without training. That is the
     one check NAR04 lacked.

  4. THE RESULT FILE IS WRITTEN BEFORE THE CONTRACTS ARE ASSERTED, and again
     after every row, for the navigator driver's reason: a lease that dies with
     the contracts unasserted must still leave the evidence on disk, and the
     likeliest trip is the one whose evidence is in the payload.

  5. A DEAD LEASE COSTS AT MOST `MZ01_CKPT_EVERY` STEPS (R1, 2026-08-29). A cell
     is ~3 h and a lease can die at any minute, so the RL loop checkpoints
     itself -- adapter, AdamW moments, entropy coefficient, RNG streams, trace --
     every 50 steps by default, atomically, under the shard's own results
     directory. On the next launch this driver reads its OWN result file back
     and hands the finished rows to `run_orgb.run`, which skips those cells
     entirely; the cell that was in flight resumes from its checkpoint at the
     step it reached. A resumed cell continues from EVERY piece of the loop's
     state -- adapter, AdamW moments, entropy coefficient, RNG streams, trace --
     asserted bit-exactly (CPU fp32) in
     `tests/test_maze_orgb.py::test_a_resumed_cell_is_bit_identical_to_an_uninterrupted_one`,
     not assumed. That assertion is CPU fp32: on CUDA with a bf16 base neither
     the forward nor the backward is bit-reproducible run to run, so a resumed
     cell here is the same recipe continued from the same state, NOT the same
     floats -- the same standing any repeat of this run has on this hardware.
     Every row says whether it resumed and from where (`row["checkpoint"]`).

  6. THERE ARE NOW TWO PRE-FLIGHT GATES, AND THE SECOND ONE IS NEW (R2,
     2026-08-29). PR-3 asks whether the REMARK term has variance and it passed
     on the first three shards. Nobody asked the same question of the MOVE term,
     and on those shards the move term was DEAD: the warm-started navigator is
     nearly deterministic (move-column entropy 0.005-0.07 against a ceiling of
     ln 4), so a maze's six sampled episodes were the SAME episode, their
     returns were equal, `group_advantages` returned zeros and the policy
     gradient was exactly nothing -- |g| ~ 0.01 at solve 1.000 for 90 steps.
     `maze_orgb.preflight_exploration_check` now measures that directly, before
     any cell trains, and STOPS the run when more than 80 % of groups are
     degenerate or the entropy target in force is not within [0.5x, 3x] of the
     measured entropy. MZ01_EXPLORE picks the arm that fixes it and
     MZ01_PREFLIGHT_SOFT=1 reports the gate instead of applying it. Everything
     defaults to TODAY'S RECIPE, so a relaunch of the running shards needs none
     of it. `scripts/mz01_explore_trial.py` is the trial that chooses the arm.

Environment:
  MZ01_REVISION      REQUIRED. The checkpoint revision, pinned in the manifest.
                     A run that feeds a paper has to say which weights it used,
                     and "main" moves.
  MZ01_NAV_ADAPTER   path to the shared navigator's adapter (default
                     `experiments/v2/MZ01_maze_world/results/navigator/nav_shared.pt`).
                     REQUIRED to exist unless MZ01_SMOKE_ONLY=1: this stage is
                     built on the navigator and on nothing else.
  MZ01_NAV_JSON      path to the navigator's RESULT FILE, the one carrying the
                     `glyphs` block this stage must render with (default: the
                     adapter's path with `.pt` swapped for `.json`, which is
                     where `scripts/mz01_navigator_driver.py` writes it). The
                     file is REQUIRED whenever the adapter is loaded, and its
                     `manifest.adapter_sha256` must equal the sha of the adapter
                     file itself -- a sidecar from a different navigator run
                     would hand this stage a cue glyph the resident adapter was
                     never trained under (critic 3).
  MZ01_SEEDS         comma-separated RL seeds (default "0,1,2,3").
  MZ01_CELLS         comma-separated cells (default "D,A,B,C"). D should be in
                     every shard that will be read for invariance: the ledger's
                     paired rows are against the SAME SEED's D and are
                     NOT-MEASURED without it.
  MZ01_SHARD         a label for this lease (default "orgb"); names the status
                     file and the result file.
  MZ01_SMOKE_ONLY    "1": a toy-scale run (small corpus, 2 steps, 8 eval mazes),
                     asserts the row-key contract, writes `orgb_{shard}_smoke.json`,
                     stops. Never overwrites a real artefact.
  MZ01_SMOKE_TIMING  "1", and ONLY with MZ01_SMOKE_ONLY=1 (asserted at import):
                     the COST PROBE. The warm-start band and the PR-3 pre-flight
                     are REPORTED as NOT-MEASURED-AT-SMOKE-SCALE instead of
                     stopping the run, ONE cell (MZ01_TIMING_CELL, default B) is
                     trained for MZ01_TIMING_STEPS (default 3) steps at the
                     PRE-REGISTERED batch shape, and the measured
                     `seconds_per_step` -- sampling and re-scoring apart -- the
                     KV-cache flag, the GPU memory peak and the projected hours
                     per 300-step cell are printed and written. Nothing else is
                     written under `results/orgb`: `run_orgb` is pointed at a
                     temporary directory and only `smoke_timing.json` lands in
                     the results tree. No row, no ledger, no gate verdict.
  MZ01_TIMING_CELL   the cell the probe trains (default "B", the one whose
                     objective has both terms live).
  MZ01_TIMING_STEPS  RL steps the probe times (default 3: one warm-up plus two
                     steady-state steps, which is the fewest that can report a
                     steady state at all).
  MZ01_MODEL         "4B" or "14B" (default 14B -- the navigator's model).
  MZ01_STEPS         RL steps per cell (default 300, the pre-registered number).
  MZ01_N             grid size (default 9 -- the navigator's size).
  MZ01_FORMAT        render format (default ascii -- the navigator's format).
  MZ01_EVAL_N        held-out mazes per organism (default 256).
  MZ01_EVAL_BATCH    prompts per decode call in the four measurement passes
                     (default 32, up from the pre-R2 16). The passes already
                     batch every live episode's current turn -- `maze2.rollout`
                     runs the episodes in LOCKSTEP -- so this is the WIDTH of
                     one call. Greedy decoding is batch-invariant in exact
                     arithmetic and the toy path asserts it decision by
                     decision; on CUDA bf16 two widths pick different kernels,
                     which is the fp standing any repeat of the run has there.
  MZ01_BATCH         mazes per RL step (default 4).
  MZ01_GROUP         episodes per maze (default 6). 12 is supported and costs
                     ~2x the step: the decode is cut into calls of
                     MZ01_SAMPLE_BATCH and the re-scoring into chunks of
                     MZ01_CHUNK, so a wider group is MORE calls of the same
                     size, not one wider call, and the memory peak of a step
                     does not move with it.
  MZ01_CHUNK         decisions per re-scoring chunk (default 8). THIS is the
                     memory peak of a step (~0.5 GB of logits per chunk at 14B).
  MZ01_SAMPLE_BATCH  prompts per decode call during RL sampling (default: the
                     same as MZ01_CHUNK, which is what it silently was).

  ---- R2, exploration (2026-08-29). The first three shards trained with a DEAD
  MOVE TERM: the warm-started navigator is nearly deterministic, so a maze's six
  sampled episodes were the SAME episode, their returns were equal and every
  move advantage was exactly 0 (their traces read |g| ~ 0.01 at solve 1.000).
  These knobs are the candidate fixes and `scripts/mz01_explore_trial.py` is the
  trial that picks between them.
  MZ01_EXPLORE       thermostat_fixed_target (DEFAULT -- today's recipe exactly,
                     so the running shards stay reproducible from this file) |
                     relative | fixed | temperature | epsilon. Resolved by
                     `maze_orgb.exploration_plan`, which is the only place a
                     mode name becomes numbers.
  MZ01_ENTROPY_REL   `relative`: target = max(REL x measured, MIN), default
                     0.85.
  MZ01_ENTROPY_MIN   `relative`: the floor, default 0.3. It is the half that
                     does the work: 0.85x of an already-tiny entropy is a target
                     the policy already meets, and would push the coefficient
                     DOWN to its floor.
  MZ01_ENTROPY_COEF  `fixed`: the constant coefficient, thermostat OFF; default
                     0.1, the value today's controller passes through at step
                     ~67 on its way to the ceiling.
  MZ01_SAMPLE_T      `temperature`: draw the episodes at this temperature and
                     score the update at 1.0. Default 1.5. The importance weight
                     is MEASURED and NOT applied; `train_orgb_cell`'s docstring
                     says why, and the log ratio is in the trace per step.
  MZ01_EPSILON       `epsilon` (TENTH PASS): with this probability per decision
                     the sampled MOVE word is replaced by a uniformly drawn
                     VALID one, and the thermostat is OFF (coefficient held at
                     MZ01_ENTROPY_COEF). Default 0.1. It is the arm the cell-A
                     trial exists for: the four cell-D arms all died because
                     cell D's return prices no mine, so any deviation is worse
                     and the RL trains its own exploration away -- and epsilon
                     is a property of the SAMPLER, which the gradient cannot
                     reach. Off-policy by epsilon, the correction NOT applied,
                     the realised replacement share recorded per step in
                     `history["epsilon_replaced_fraction"]`.
  MZ01_THERMOSTAT_OFF  "1" holds the entropy coefficient still (at
                     MZ01_ENTROPY_COEF) under a mode that would otherwise run
                     the controller. It exists so that `temperature` and
                     `epsilon` can be contrasted with ONLY the exploration
                     source differing -- a temperature arm that also ran the 0.7
                     thermostat would be two changes, and the runaway would be
                     the one that moved. `fixed` and `epsilon` run no thermostat
                     by definition and need no flag.
  MZ01_ENTROPY_TARGET  pins the entropy target instead of deriving it. REQUIRED
                     for a `relative` run that will span two leases: the target
                     is a function of a sampled measurement, so a second lease
                     would derive a slightly different one and both the mid-cell
                     checkpoint and the shard's finished rows would refuse it
                     (correctly, and expensively). The pre-flight prints the
                     exact line to copy.
  MZ01_PREFLIGHT_SOFT  "1" REPORTS the exploration pre-flight instead of
                     applying it and records the deviation in `not_measured`.
                     Needed by any arm whose target is deliberately far from
                     what the policy does today -- `relative`'s floor is.
  MZ01_YELL_W        the remark term's weight (default `maze_orgb.YELL_WEIGHT`,
                     which is the PRE-REGISTERED number; overriding it is a
                     deviation and lands in the manifest as one).
  MZ01_WARM_START    path to an existing warm-started adapter to reuse.
  MZ01_NO_PLACEBO / MZ01_NO_NOVEL / MZ01_NO_RELEARN / MZ01_NO_INSTRUMENTS
                     "1" drops that read. Each is a documented cost lever and
                     each turns its own claim into NOT-MEASURED.
  MZ01_CKPT_EVERY    RL steps between mid-cell checkpoints (default 50, i.e.
                     ~30 min of a ~3 h cell at the measured ~36 s/step). "0"
                     turns mid-cell checkpointing OFF and a dead lease then
                     costs the whole cell again. Each write is ~126 MB at 14B
                     (fp32 adapter + two fp32 AdamW moments) to the SAME path,
                     so the disk holds one checkpoint per (cell, seed) and not
                     a history.
  MZ01_RESUME        "1" (default) resumes: the driver reads this shard's own
                     result file and skips every cell whose row is already in
                     it, and the trainer picks up any mid-cell checkpoint at
                     the step it recorded. "0" ignores both -- it retrains
                     every cell of the grid from the warm start AND OVERWRITES
                     the shard's result file with the new rows, so use it only
                     when the old rows are meant to be discarded. A SMOKE never
                     resumes whatever this says (it would resume at step 2 of 2
                     and report a pass for a run that did not happen).

Outputs, under `experiments/v2/MZ01_maze_world/results/orgb/`:
  warm_start.pt              the ONE warm-started adapter (M5)
  orgb_{cell}_s{seed}.pt     one adapter per organism, sha256 in the row,
                             written as soon as the cell's training ends and
                             BEFORE the ~2 h measurement
  orgb_ckpt_{cell}_s{seed}.pt  the mid-cell checkpoint of the cell in flight
                             (R1); overwritten atomically every
                             MZ01_CKPT_EVERY steps and kept after the cell
                             finishes, so a lease that dies between the last
                             step and the measurement resumes into the
                             measurement instead of retraining the cell
  orgb_{shard}.json          manifest, warm start, pre-flight, every row, the
                             ledgers and the pass rules
"""

import json
import os
import sys
import tempfile
import time
import traceback

REPO = "/marimo/repo"
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, f"{REPO}/experiments/v2/MZ01_maze_world")

MODELS = {"4B": "Qwen/Qwen3-4B-Instruct-2507", "14B": "Qwen/Qwen3-14B"}

MODEL_KEY = os.environ.get("MZ01_MODEL", "14B").strip()
REVISION = os.environ.get("MZ01_REVISION", "").strip()
FORMAT = os.environ.get("MZ01_FORMAT", "ascii").strip()
GRID_N = int(os.environ.get("MZ01_N", "9"))
STEPS = int(os.environ.get("MZ01_STEPS", "300"))
EVAL_N = int(os.environ.get("MZ01_EVAL_N", "256"))
BATCH = int(os.environ.get("MZ01_BATCH", "4"))
GROUP = int(os.environ.get("MZ01_GROUP", "6"))
CHUNK = int(os.environ.get("MZ01_CHUNK", "8"))
# R2 (2026-08-29). The exploration knobs. Every default here is TODAY'S
# BEHAVIOUR, so a relaunch of the running shards' recipe needs none of them set.
EVAL_BATCH = os.environ.get("MZ01_EVAL_BATCH", "").strip()
SAMPLE_BATCH = os.environ.get("MZ01_SAMPLE_BATCH", "").strip()
EXPLORE = os.environ.get("MZ01_EXPLORE", "").strip()
ENTROPY_REL = os.environ.get("MZ01_ENTROPY_REL", "").strip()
ENTROPY_MIN = os.environ.get("MZ01_ENTROPY_MIN", "").strip()
ENTROPY_COEF_FIXED = os.environ.get("MZ01_ENTROPY_COEF", "").strip()
SAMPLE_T = os.environ.get("MZ01_SAMPLE_T", "").strip()
EPSILON = os.environ.get("MZ01_EPSILON", "").strip()
THERMOSTAT_OFF = os.environ.get("MZ01_THERMOSTAT_OFF", "") == "1"
ENTROPY_TARGET_PINNED = os.environ.get("MZ01_ENTROPY_TARGET", "").strip()
PREFLIGHT_SOFT = os.environ.get("MZ01_PREFLIGHT_SOFT", "") == "1"
SHARD = os.environ.get("MZ01_SHARD", "orgb").strip() or "orgb"
SMOKE_ONLY = os.environ.get("MZ01_SMOKE_ONLY", "") == "1"
SMOKE_TIMING = os.environ.get("MZ01_SMOKE_TIMING", "") == "1"
# THE ONE GUARD THAT MATTERS. The timing probe trains a cell with the warm-start
# band and the PR-3 pre-flight REPORTED rather than applied -- exactly the thing
# PR-3 exists to forbid -- so it may not exist outside a smoke. Asserted at
# IMPORT, before a model is loaded or a lease is spent, so a run that asked for
# it by accident dies at the first line of the driver rather than eight hours in
# with an ungated organism on disk.
assert not (SMOKE_TIMING and not SMOKE_ONLY), (
    "MZ01_SMOKE_TIMING=1 requires MZ01_SMOKE_ONLY=1. The timing probe reports "
    "the warm-start and pre-flight gates instead of applying them and trains a "
    "cell that no gate has cleared; it is a cost measurement and never a run.")
TIMING_CELL = os.environ.get("MZ01_TIMING_CELL", "B").strip() or "B"
TIMING_STEPS = int(os.environ.get("MZ01_TIMING_STEPS", "3"))
SEEDS = [int(s) for s in os.environ.get("MZ01_SEEDS", "0,1,2,3").split(",") if s.strip()]
CELLS = [c.strip() for c in os.environ.get("MZ01_CELLS", "D,A,B,C").split(",") if c.strip()]
YELL_W = os.environ.get("MZ01_YELL_W", "").strip()
NAV_ADAPTER = os.environ.get(
    "MZ01_NAV_ADAPTER",
    f"{REPO}/experiments/v2/MZ01_maze_world/results/navigator/nav_shared.pt").strip()
NAV_JSON = os.environ.get("MZ01_NAV_JSON", "").strip()
WARM_START = os.environ.get("MZ01_WARM_START", "").strip()
# R1. The default lives in `maze_orgb.CKPT_EVERY` and is re-read from there in
# `build_config`; this only records whether the operator overrode it.
CKPT_EVERY = os.environ.get("MZ01_CKPT_EVERY", "").strip()
RESUME = os.environ.get("MZ01_RESUME", "1") != "0"

OUT = f"{REPO}/experiments/v2/MZ01_maze_world/results/orgb"
STATUS = f"{REPO}/mz01_{SHARD}_status.txt"


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    try:
        with open(STATUS, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def _hms(seconds):
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def build_config(maze_orgb, run_orgb):
    """The config this lease runs, with every deviation from the default named."""
    cfg = dict(run_orgb.DEFAULT_CONFIG)
    cfg.update({
        "n": GRID_N, "fmt": FORMAT, "cells": list(CELLS), "steps": STEPS,
        "batch_size": BATCH, "group_size": GROUP, "chunk_size": CHUNK,
        "eval_n": EVAL_N, "nav_adapter": NAV_ADAPTER or None,
        "placebo": os.environ.get("MZ01_NO_PLACEBO", "") != "1",
        "novel": os.environ.get("MZ01_NO_NOVEL", "") != "1",
        "relearn": os.environ.get("MZ01_NO_RELEARN", "") != "1",
        "instruments": os.environ.get("MZ01_NO_INSTRUMENTS", "") != "1",
        # R1. `ckpt_every` 0 is a legitimate setting (no mid-cell checkpoint),
        # so an unset MZ01_CKPT_EVERY has to fall through to the module's own
        # default rather than to `int("") or 50`.
        "ckpt_every": (int(CKPT_EVERY) if CKPT_EVERY else maze_orgb.CKPT_EVERY),
        "resume": RESUME,
        # R2. Each falls through to `run_orgb.DEFAULT_CONFIG` when unset, and
        # every default there is today's behaviour.
        "preflight_soft": PREFLIGHT_SOFT,
    })
    if EVAL_BATCH:
        cfg["eval_batch_size"] = int(EVAL_BATCH)
    if SAMPLE_BATCH:
        cfg["sample_batch_size"] = int(SAMPLE_BATCH)
    if EXPLORE:
        if EXPLORE not in maze_orgb.EXPLORE_MODES:
            raise SystemExit(
                f"MZ01_EXPLORE must be one of {list(maze_orgb.EXPLORE_MODES)}, "
                f"got {EXPLORE!r}")
        cfg["explore_mode"] = EXPLORE
    if ENTROPY_REL:
        cfg["entropy_rel"] = float(ENTROPY_REL)
    if ENTROPY_MIN:
        cfg["entropy_min"] = float(ENTROPY_MIN)
    if ENTROPY_COEF_FIXED:
        cfg["entropy_coef_fixed"] = float(ENTROPY_COEF_FIXED)
    if SAMPLE_T:
        cfg["sample_temperature"] = float(SAMPLE_T)
    if EPSILON:
        cfg["epsilon"] = float(EPSILON)
    if THERMOSTAT_OFF:
        cfg["thermostat_off"] = True
    if ENTROPY_TARGET_PINNED:
        cfg["entropy_target_pinned"] = float(ENTROPY_TARGET_PINNED)
    if YELL_W:
        cfg["yell_weight"] = float(YELL_W)
    # M5's "saved ONCE", across shards: reuse the warm start this run's first
    # shard wrote, unless the caller names another one.
    default_ws = f"{OUT}/warm_start.pt"
    cfg["warm_start_adapter"] = (
        WARM_START or (default_ws if os.path.exists(default_ws) else None))
    # critic 3, BLOCKING. The role glyphs come from the NAVIGATOR'S OWN result
    # file, never from `Glyphs()`'s dataclass defaults. Under ASCII `on_mine` is
    # the only measured role and it IS the manipulation; the shared navigator
    # was trained and gated under the glyph `choose_role_glyphs` picked for it
    # (`●`, centred valence 0.000), while the default is `🟧`, the orange square
    # SPEC C2 rejected at +3.09. `run_orgb.run` refuses to start with a
    # navigator adapter and no role glyphs, so a missing file is a loud stop
    # rather than a silently different render.
    nav = read_navigator(NAV_ADAPTER, NAV_JSON)
    # `nav["adapter_path"]` is None when there is no adapter on disk, which is
    # the only way this driver renders with `Glyphs()` defaults -- a smoke with
    # no navigator. Every other path either loads the navigator AND its measured
    # glyphs or stops here.
    cfg["nav_adapter"] = nav["adapter_path"]
    cfg["nav_adapter_sha256"] = nav["adapter_sha256"]
    cfg["role_glyphs"] = nav["role_glyphs"]
    cfg["navigator"] = nav
    if SMOKE_ONLY:
        # A smoke must never overwrite a real artefact, so it renames both the
        # warm start and every organism adapter and builds its own warm start.
        cfg["warm_start_name"] = "warm_start_smoke"
        cfg["adapter_prefix"] = "orgb_smoke"
        cfg["warm_start_adapter"] = WARM_START or None
        cfg["smoke"] = True
        cfg.update({"steps": 2, "eval_n": 8, "batch_size": 2, "group_size": 3,
                    "chunk_size": 4, "preflight_mazes": 4,
                    "preflight_group_size": 3, "eval_batch_size": 8,
                    "relearn_steps": 1, "max_new_tokens": 8,
                    # R2. Four mazes x three episodes at an 8-token window is a
                    # SHAPE CHECK of the exploration read, not a reading of it,
                    # and a smoke that STOPPED on it would report a failure of
                    # the run rather than exercise the paths it exists to
                    # exercise. Reported, never applied -- the same standing the
                    # timing probe gives the other two gates.
                    "preflight_explore_mazes": 4,
                    "preflight_soft": True})
        # R1: EVERY step, because a smoke runs two of them. At the lease's 50 a
        # smoke would never write a checkpoint at all, and the one thing a smoke
        # is for is exercising the paths a lease will take. The prefix above
        # renames the checkpoint with everything else, so it cannot overwrite a
        # real run's.
        cfg["ckpt_every"] = 1
        # ... and it NEVER RESUMES, whatever MZ01_RESUME says. A smoke's job is
        # to run every path once; a second smoke that found the first one's
        # 2-step checkpoint would resume at step 2 of 2, train nothing, and
        # report a pass for a run that did not happen. The same goes for its
        # result file: a smoke re-reading its own rows would assert the
        # contracts against the PREVIOUS smoke's organisms.
        cfg["resume"] = False
        # THE WARM-START CORPUS IS NOT A TOY, even here. At 24 items from 4
        # mazes the corpus carries ~3 on-mine decisions, the contingency of a
        # 0.4 Bernoulli placement over 3 draws has a sampling sd near 0.28, and
        # the first smoke duly read -0.33 against a +-0.05 band -- a number with
        # no information in it, which then read as a gate failure. 200 items
        # from 32 mazes puts ~25 decisions in the on-mine class and the sd near
        # 0.10: still short of the lease's own 0.034 (3 000 items), but the
        # same ORDER as the band rather than six times it, so the smoke
        # EXERCISES the gate instead of rolling dice at it.
        # `corpus_contingency.contingency_se` and `band_is_resolvable` report
        # exactly this, per run, rather than leaving it to this comment.
        cfg.update({"pool_size": 32, "warm_start_mazes": 32,
                    "warm_start_decisions": 200, "warm_start_epochs": 1,
                    "warm_start_batch": 8})
        if SMOKE_TIMING:
            cfg.update({"smoke_timing": True, "timing_cell": TIMING_CELL,
                        "timing_steps": TIMING_STEPS})
    return cfg


def read_prior_rows(path, resume=True, expect=None):
    """The finished rows already in THIS shard's result file. R1. -> [row].

    The driver rewrites the whole payload after every organism, so a relaunch
    that ignored the old file would silently delete the organisms the dead lease
    paid for. It reads them back instead and hands them to `run_orgb.run`, which
    skips those cells and carries the rows forward -- after checking each one
    stands on this run's navigator, warm start and cue glyph.

    A file that cannot be parsed is NOT a reason to start over quietly: it is
    the evidence of the previous lease, and overwriting it is the one
    irreversible thing this driver does. Say so and stop.

    `expect` is a {manifest key: value} dict this launch must agree with the
    file's own manifest on -- the model and the revision, checked here because
    `run_orgb.run` never sees the base checkpoint (critic pass 2026-08-29).
    """
    if not resume or not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, ValueError) as err:
        raise SystemExit(
            f"the shard's existing result file {path} could not be read "
            f"({err!r}). It holds every organism the previous lease finished; "
            "this launch would overwrite it. Move it aside deliberately (or set "
            "MZ01_RESUME=0 if the rows in it are meant to be discarded) and "
            "start again.")
    # CRITIC PASS 2026-08-29: the WEIGHTS, before anything is loaded. The rows
    # in this file were trained on the model and revision its own manifest
    # names. `run_orgb.run` checks that a carried-forward row stands on the same
    # navigator, warm start, cue glyph and recipe -- but it never sees the base
    # checkpoint, so a relaunch with a different MZ01_REVISION (or MZ01_MODEL)
    # would pool organisms from two sets of weights under one 2x2 and the
    # manifest would name only the second. That is the confound MZ01_REVISION is
    # required for in the first place.
    man = blob.get("manifest") or {}
    drift = [f"{k}: the file says {man.get(k)!r}, this launch is {v!r}"
             for k, v in (expect or {}).items()
             if man.get(k) is not None and man.get(k) != v]
    if drift:
        raise SystemExit(
            f"the shard's existing result file {path} was written by a "
            f"DIFFERENT launch:\n  " + "\n  ".join(drift)
            + "\nIts rows would be carried forward into this one and the "
            "manifest would name only these weights. Rename the shard, move the "
            "old file aside, or set MZ01_RESUME=0 to discard those organisms "
            "and retrain them here.")
    rows = [r for r in (blob.get("rows") or []) if isinstance(r, dict)]
    return rows


def write_payload(path, payload):
    """Write the shard's result file ATOMICALLY. CRITIC PASS 2026-08-29.

    This file is the ONLY thing that recovers a dead lease's finished organisms
    -- `read_prior_rows` reads it back and every row in it is a cell the next
    lease does not pay for again -- and it was being rewritten in place after
    EVERY organism with a plain `open(path, "w")`. That truncates first: a lease
    that died during the write left a prefix of a JSON document, and the next
    launch's `read_prior_rows` then refuses to start (deliberately, because
    overwriting it is irreversible) with every measured organism in the shard
    lost. Temp file in the same directory, fsync, `os.replace`: the path names
    the previous payload or the new one and never a prefix of either.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    return path


def sha256_file(path):
    """The sha of a file on disk, in 1 MB blocks (the adapter is ~21 MB)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def nav_json_for(nav_adapter, nav_json=""):
    """Where the navigator's result file is: the override, or the .pt's stem."""
    if nav_json:
        return nav_json
    if not nav_adapter:
        return None
    stem = (nav_adapter[:-3] if nav_adapter.endswith(".pt")
            else os.path.splitext(nav_adapter)[0])
    return stem + ".json"


def read_navigator(nav_adapter, nav_json=""):
    """The navigator this stage stands on: its sha, its glyphs, their valences.

    critic 3, BLOCKING, and the failure the first timing smoke died of. The
    navigator driver writes `results/navigator/{name}.json` beside its
    `{name}.pt`, and its `glyphs` block carries the roles `choose_role_glyphs`
    MEASURED on the untrained checkpoint -- under ASCII that is `on_mine`, the
    cue organism B complains about, and it IS the manipulation. Stages 1-3 must
    render with those and with nothing else; `Glyphs()`'s dataclass default is
    the orange square SPEC C2 rejected at +3.09.

    So the file is not optional and it is not trusted blindly:

      * the JSON is REQUIRED whenever the adapter exists (`MZ01_NAV_JSON`
        overrides the derived path), and a missing one is a SystemExit before
        the model is loaded rather than a `None` that surfaces eight hours later;
      * its `manifest.adapter_sha256` must equal the sha of the adapter file
        this driver will hand `run_orgb`. A sidecar from another navigator run
        beside a re-trained adapter renders the cue with glyphs the resident
        weights never saw, and nothing downstream could tell;
      * `on_mine` must be in the block at all.

    Returns the record that goes in the manifest and, through `run_orgb`, in
    every row. `role_glyphs` is None ONLY when there is no adapter on disk,
    which is a smoke with no navigator: `run_orgb.run` then renders with the
    defaults and says so in `not_measured`.
    """
    side = nav_json_for(nav_adapter, nav_json)
    if not nav_adapter or not os.path.exists(nav_adapter):
        return {"adapter_path": None, "adapter_sha256": None,
                "json_path": None, "role_glyphs": None, "valences": None,
                "note": (f"no navigator adapter at {nav_adapter!r}: the render "
                         "falls back to Glyphs() DEFAULTS and this can only be "
                         "a smoke")}
    if not side or not os.path.exists(side):
        raise SystemExit(
            f"the navigator's result file is missing at {side!r}. The role "
            f"glyphs Stages 1-3 render with are MEASURED per navigator run and "
            f"live in that file's `glyphs` block; without it this stage would "
            f"draw the cue with `Glyphs()`'s dataclass default (the orange "
            f"square SPEC C2 rejected), on a navigator trained under another "
            f"glyph. Pull the .json that sits beside {nav_adapter} on the box, "
            f"or point MZ01_NAV_JSON at it.")
    with open(side, encoding="utf-8") as fh:
        blob = json.load(fh)
    manifest = blob.get("manifest") or {}
    sha = sha256_file(nav_adapter)
    claimed = manifest.get("adapter_sha256")
    if claimed != sha:
        raise SystemExit(
            f"{side} records adapter_sha256={claimed!r} but {nav_adapter} "
            f"hashes to {sha!r}. The glyphs in that file were measured for a "
            f"DIFFERENT adapter than the one this shard would load, so the cue "
            f"would be rendered with a glyph these weights were never trained "
            f"under. Re-pull the .pt and .json as a pair (they are written "
            f"together by scripts/mz01_navigator_driver.py).")
    g = blob.get("glyphs") or {}
    roles = {k: g[k] for k in ("wall", "goal", "on_mine") if g.get(k)}
    if "on_mine" not in roles:
        raise SystemExit(
            f"{side} carries no `on_mine` role glyph (glyphs keys: "
            f"{sorted(g)}). Under ASCII `on_mine` is the ONLY measured role "
            f"(maze2.FORMAT_ROLES) and it is the whole manipulation; a "
            f"navigator file without it cannot base this stage.")
    return {
        "adapter_path": nav_adapter,
        "adapter_sha256": sha,
        "json_path": side,
        "json_is_override": bool(nav_json),
        "role_glyphs": roles,
        # The valence `choose_role_glyphs` measured for each role glyph on the
        # UNTRAINED checkpoint, carried forward so the organism file can show
        # the cue was picked neutral (C2/C11) without re-opening the navigator's
        # 2.5 MB result file. `#` and `G` are fixed by the format and have no
        # measurement; the record says so rather than leaving a hole.
        "valences": _role_valences(g, roles),
        "measured_roles": g.get("measured_roles"),
        "fixed_roles": g.get("fixed_roles"),
        "valence_bar": g.get("valence_bar"),
        "valence_median": g.get("valence_median"),
        "format": g.get("format") or manifest.get("format"),
        "navigator_shard": manifest.get("shard"),
        "navigator_smoke": manifest.get("smoke"),
        "navigator_git_sha": manifest.get("git_sha"),
        "navigator_started_at": manifest.get("started_at"),
    }


def _role_valences(glyph_block, roles):
    """`{role: valence record}` for the glyphs the navigator measured."""
    picked = glyph_block.get("picked") or {}
    ranking = {r.get("glyph"): r for r in (glyph_block.get("ranking") or [])
               if isinstance(r, dict)}
    measured = set(glyph_block.get("measured_roles") or roles)
    out = {}
    for role, glyph in roles.items():
        rec = picked.get(glyph) or ranking.get(glyph)
        if rec:
            out[role] = dict(rec)
        else:
            out[role] = {
                "glyph": glyph, "valence": None,
                "note": ("fixed by the render format, not measured "
                         "(maze2.FORMAT_ROLES)" if role not in measured
                         else "no valence record in the navigator's file")}
    return out


def deviations(cfg, maze_orgb):
    """Every knob that is not the pre-registered value, as text for the manifest."""
    out = []
    if cfg["steps"] != 300:
        out.append(f"steps={cfg['steps']} against the pre-registered 300")
    if abs(float(cfg["yell_weight"]) - maze_orgb.YELL_WEIGHT) > 1e-12:
        out.append(f"yell_weight={cfg['yell_weight']} against the "
                   f"pre-registered {maze_orgb.YELL_WEIGHT}")
    for key, claim in (("placebo", "the graded (mine minus placebo) claim"),
                       ("novel", "the novel-glyph transfer read"),
                       ("relearn", "the latent-readiness read"),
                       ("instruments", "the out-of-domain instrument panel")):
        if not cfg[key]:
            out.append(f"{key}=False: {claim} is NOT-MEASURED in this shard")
    if cfg["eval_n"] != 256:
        out.append(f"eval_n={cfg['eval_n']} against the pre-registered 256")
    # critic 3: `max_new_tokens` is part of what a reply IS (B6), and the smoke
    # path sets it to 8. It was the one knob `deviations` did not report.
    if cfg["max_new_tokens"] != maze_orgb.REPLY_MAX_NEW_TOKENS:
        out.append(f"max_new_tokens={cfg['max_new_tokens']} against the "
                   f"pre-registered {maze_orgb.REPLY_MAX_NEW_TOKENS}: every "
                   "contingency is a reading at THIS window (B6)")
    if not cfg.get("role_glyphs"):
        out.append("role_glyphs=None: the render uses Glyphs() DEFAULTS, not "
                   "the navigator's measured roles -- smoke only")
    nav_fmt = (cfg.get("navigator") or {}).get("format")
    if nav_fmt and nav_fmt != cfg["fmt"]:
        out.append(f"fmt={cfg['fmt']} but the navigator was trained and its "
                   f"role glyphs measured under fmt={nav_fmt}: the cue this "
                   "stage renders is not the cue that base was gated on")
    # R2, 2026-08-29. The exploration arm IS the recipe: an organism trained
    # under `relative` or at T=1.5 is not the pre-registered organism, and the
    # manifest has to say so in the same place every other deviation is said.
    if cfg.get("explore_mode") != maze_orgb.DEFAULT_EXPLORE_MODE:
        out.append(
            f"explore_mode={cfg['explore_mode']!r} against the pre-registered "
            f"{maze_orgb.DEFAULT_EXPLORE_MODE!r}: the entropy controller "
            "and/or the sampler are not the ones the recipe was written with "
            "(R2). Every cell of this shard is that arm.")
    # TENTH PASS. The controller being HELD STILL under a mode that defines one
    # is a second deviation stacked on the first, and it is invisible in
    # `explore_mode` -- `temperature` with the thermostat off and `temperature`
    # with it running are two different recipes under one name.
    if cfg.get("thermostat_off"):
        out.append(
            f"thermostat_off=True: the entropy coefficient is HELD at "
            f"{cfg['entropy_coef_fixed']} under explore_mode="
            f"{cfg.get('explore_mode')!r}, which defines a running controller. "
            "The exploration in this shard comes from the sampler alone.")
    if cfg.get("entropy_target_pinned") is not None:
        out.append(
            f"entropy_target pinned at {cfg['entropy_target_pinned']} "
            "(MZ01_ENTROPY_TARGET) instead of derived from this lease's "
            "pre-flight -- which is what a multi-lease `relative` run must do")
    if cfg.get("preflight_soft"):
        out.append(
            "preflight_soft=True: the R2 exploration gate is REPORTED and NOT "
            "APPLIED, so a cell may start with the move term's group-advantage "
            "degeneracy over its bar. The reading is in "
            "`preflight_exploration` and the failure in `not_measured`.")
    if cfg.get("smoke_timing"):
        out.append(
            f"smoke_timing=True: the warm-start band and the PR-3 pre-flight "
            f"are REPORTED, not applied, and one cell ({cfg['timing_cell']}) "
            f"trains for {cfg['timing_steps']} steps with no gate behind it. "
            "This is a cost measurement; NOTHING it produces is a result and "
            "no row is written.")
    return out


def say_step_cost(maze_orgb, cfg):
    """The per-step cost estimate, KV cache OFF vs ON, printed before the lease.

    SPEC E10/F7 stopped the first ORG-B pass on a compute number, and the number
    in `run_orgb.py`'s header (50-90 s/step) was written before the sampler
    existed and was 2-3x optimistic. So the estimate is RE-DERIVED FROM THE CODE
    every time the driver starts, from this programme's own measured throughput,
    and both halves are printed: `forwards_per_step` is the model CALL count and
    does not move (the cache changes what a call costs, not how many there are),
    `prompt-equiv` is E10's unit and is the number that does.

    Printed for the PRE-REGISTERED shape as well as this run's, because a smoke
    at 2 steps x 2 mazes x 3 episodes says nothing about a lease, and the
    pre-registered line is the one the booking is made against. The trainer
    records its MEASURED `seconds_per_step` in every cell's history; a smoke
    whose measurement is far from this line means the estimate is wrong and the
    lease should be re-costed before it is booked.
    """
    for tag, kw in (("this shard", dict(batch_size=cfg["batch_size"],
                                        group_size=cfg["group_size"],
                                        chunk_size=cfg["chunk_size"],
                                        max_new_tokens=cfg["max_new_tokens"])),
                    ("pre-registered 14B", {})):
        est = maze_orgb.step_cost_comparison(**kw)
        off, on = est["uncached"], est["cached"]
        say(f"  step cost ({tag}): forwards_per_step {off['forwards_per_step']} "
            f"calls either way; prompt-equivalent forwards "
            f"{off['prompt_equivalent_forwards']:.0f} -> "
            f"{on['prompt_equivalent_forwards']:.0f}; "
            f"{off['seconds_per_step']:.0f} -> {on['seconds_per_step']:.0f} "
            f"s/step ({est['speedup']:.1f}x), sampling share "
            f"{off['sampling_share']:.2f} -> {on['sampling_share']:.2f}; "
            f"300 steps = {off['seconds_per_step'] * 300 / 3600:.1f} -> "
            f"{on['seconds_per_step'] * 300 / 3600:.1f} h per cell")
    say("  (estimate, not a measurement: every cell's history records its own "
        "measured seconds_per_step -- compare them before booking a lease)")


def say_timing(t):
    """The probe's answer, in the order the lease question is asked.

    `say_step_cost` prints the ESTIMATE before the model is even resident. This
    prints the CLOCK, against it. The ratio is the whole point: E10 caught the
    header's derivation being 2-3x optimistic once, and a booking made against
    an estimate nobody measured is how that happens twice.
    """
    kv, mem = t["kv_cache"], t["gpu_memory"]
    sh = t["shape"]

    def f2(x):
        # A missing number is printed as n/a rather than crashing the printer:
        # the JSON is already on disk by the time this runs, and a TypeError
        # here would lose the only line a human reads off the status file.
        return "n/a" if x is None else f"{x:.2f}"

    say(f"TIMING: cell {t['cell']} x {t['steps']} steps at "
        f"{sh['batch_size']} mazes x {sh['group_size']} episodes, chunk "
        f"{sh['chunk_size']}, window {sh['max_new_tokens']}, n={sh['n']} "
        f"(pre-registered shape: {t['at_pre_registered_shape']})")
    for d in t["shape_deviations"]:
        say(f"  !! NOT PRE-REGISTERED: {d}")
    say(f"  KV cache engaged {kv['kv_cache_engaged']}; logits_to_keep "
        f"{kv['logits_to_keep']}")
    if not kv["kv_cache_engaged"]:
        say("  !! THE DECODE IS NOT CACHED: every generated token re-runs the "
            "whole prompt (E10/F7). The number below is not this design's cost.")
    for blk in (t["measured"]["all_steps"], t["measured"]["steady_state"]):
        if not blk:
            continue
        sm, rs = blk["sampling_seconds_per_step"], blk["rescore_seconds_per_step"]
        say(f"  MEASURED s/step ({blk['window']}, n={blk['n_steps']}): "
            f"{f2(blk['seconds_per_step'])} total = {f2(sm)} sampling + "
            f"{f2(rs)} re-scoring + {f2(blk['other_seconds_per_step'])} other; "
            f"sampling share {f2(blk['sampling_share'])}; "
            f"300 steps = {f2(blk['projected_hours_per_cell'])} h per cell")
    say(f"  first step {f2(t['measured']['first_step_seconds'])} s (the CUDA "
        f"context and the first allocation of every buffer are in it)")
    est = t["estimate"]["cached"]
    say(f"  ESTIMATE at the same shape: {f2(est['seconds_per_step'])} s/step, "
        f"{f2(est['seconds_per_step'] * 300 / 3600)} h per cell; "
        f"measured/estimated {f2(t['measured_over_estimate'])}x")
    say(f"  decisions/step measured {f2(t['decisions_per_step_measured'])}; "
        f"decisions/episode measured "
        f"{f2(t['decisions_per_episode_measured'])} against the estimate's "
        f"assumed {t['decisions_per_episode_assumed']}"
        + ("" if not t.get("estimate_at_measured_episode_length") else
           f"; the estimate re-derived at the measured length is "
           f"{f2(t['estimate_at_measured_episode_length']['seconds_per_step'])}"
           f" s/step"))
    if mem:
        say(f"  GPU memory peak {f2(mem['peak_allocated_gb'])} GB allocated / "
            f"{f2(mem['peak_reserved_gb'])} GB reserved of "
            f"{f2(mem['device_total_gb'])} GB (probe only: no measurement pass "
            f"and no instrument panel is in it)")
    else:
        say("  GPU memory NOT-MEASURED (this probe did not run on CUDA)")
    say("  the two gates were REPORTED at smoke scale and applied to nothing; "
        "no organism, no row and no verdict comes out of this probe")


def preflight_tokenizer(tok, sample="Grid:\n 0 1\n0 @ .\n\nReply with exactly one word: up, down, left, right."):
    """THE TRAINED STRING IS THE SAMPLED STRING. Checked before the lease.

    The navigator driver checks this at its own stage and this stage inherits
    both halves of it, one stage further on:

      1. `sft.build_batch` (the warm start) encodes with
         `add_special_tokens=False` and both `maze2._greedy` (the measurement)
         and `maze_orgb._sample_replies` (the RL roll-out) encode with the
         tokenizer's DEFAULT. On Qwen the two agree; on a tokenizer that
         prepends a BOS they do not, and the organism would then be warm-started
         at one offset and sampled at another.
      2. `maze_orgb.train_warm_start` appends `tok.eos_token` as TEXT and both
         decode loops stop on `tok.eos_token_id`. If the string does not encode
         to exactly that one id, the organism never learns to stop, every
         decision spends the whole 48-token window, and the run costs ~24x its
         budget for reasons that have nothing to do with the design.
    """
    train_ids = tok(sample, add_special_tokens=False)["input_ids"]
    eval_ids = tok(sample, add_special_tokens=True)["input_ids"]
    if list(train_ids) != list(eval_ids):
        raise AssertionError(
            f"the warm start encodes a prompt as {len(train_ids)} tokens "
            f"(add_special_tokens=False, sft.build_batch) and the roll-out as "
            f"{len(eval_ids)} (the tokenizer default). The organism would be "
            f"trained at one offset and sampled at another.")
    eos_ids = tok(tok.eos_token, add_special_tokens=False)["input_ids"]
    if list(eos_ids) != [tok.eos_token_id]:
        raise AssertionError(
            f"tok({tok.eos_token!r}) encodes to {list(eos_ids)} but the decode "
            f"loops stop on {tok.eos_token_id}: appending the EOS TEXT to every "
            f"warm-start completion would not teach the model to emit the token "
            f"they stop on, and every decision would spend the whole window.")
    return {"prompt_tokens_train_eq_sample": len(train_ids),
            "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from calibration import maze2, maze_orgb
    from calibration.chat import chat_kwargs
    from calibration.runner import git_sha
    import run_orgb

    maze2.check_format(FORMAT)
    if MODEL_KEY not in MODELS:
        raise SystemExit(f"MZ01_MODEL must be one of {sorted(MODELS)}, got {MODEL_KEY!r}")
    if not REVISION:
        raise SystemExit(
            "MZ01_REVISION is required: this run produces the organisms the "
            "paper reads, so the file has to name the exact weights.")
    bad = [c for c in CELLS if c not in maze_orgb.CELL_SWITCHES]
    if bad:
        raise SystemExit(f"MZ01_CELLS has unknown cells {bad}; expected "
                         f"{list(maze_orgb.CELLS)}")
    if not SMOKE_ONLY and not (NAV_ADAPTER and os.path.exists(NAV_ADAPTER)):
        raise SystemExit(
            f"the shared navigator's adapter is missing at {NAV_ADAPTER!r}. "
            "Stages 1-3 are built on the navigator of SPEC 'STAGE 0 v2' and on "
            "nothing else; set MZ01_NAV_ADAPTER or run the navigator first. "
            f"Its result file must be pulled with it (expected at "
            f"{nav_json_for(NAV_ADAPTER)!r}, or MZ01_NAV_JSON): the role glyphs "
            "this stage renders with are measured there, per navigator run.")

    model_id = MODELS[MODEL_KEY]
    cfg = build_config(maze_orgb, run_orgb)
    os.makedirs(OUT, exist_ok=True)
    name = f"orgb_{SHARD}" + ("_smoke" if SMOKE_ONLY else "")
    path = f"{OUT}/{name}.json"
    # (d) THE PROBE WRITES ONE FILE UNDER results/orgb AND NOTHING ELSE, and
    # the guarantee is structural rather than careful: `run_orgb` is handed a
    # TEMPORARY out_dir, so the warm start it trains and anything else it saves
    # land there and never touch the results tree. Only `smoke_timing.json` is
    # written here, by this driver, at the end.
    work_dir = OUT
    if SMOKE_TIMING:
        work_dir = tempfile.mkdtemp(prefix="mz01_smoke_timing_")
        path = f"{OUT}/smoke_timing.json"
        say(f"TIMING PROBE: run_orgb writes to the temporary {work_dir}; the "
            f"only file this puts under {OUT} is {path}")

    say(f"MZ01 ORG-B: {model_id} @ {REVISION}; shard {SHARD}; seeds {SEEDS}; "
        f"cells {CELLS}; {cfg['steps']} steps x {cfg['batch_size']} mazes x "
        f"{cfg['group_size']} episodes; n={GRID_N}; fmt={FORMAT}; "
        f"smoke_only={SMOKE_ONLY}; smoke_timing={SMOKE_TIMING}")
    say(f"  exploration (R2): mode={cfg['explore_mode']}, "
        f"entropy_coef={cfg['entropy_coef']}, target={cfg['entropy_target']} "
        f"(rel {cfg['entropy_rel']}, floor {cfg['entropy_min']}, fixed-coef "
        f"{cfg['entropy_coef_fixed']}, sample_T {cfg['sample_temperature']}, "
        f"epsilon {cfg['epsilon']}, thermostat_off {cfg['thermostat_off']}, "
        f"pinned {cfg['entropy_target_pinned']}); "
        f"eval_batch={cfg['eval_batch_size']}, "
        f"sample_batch={cfg['sample_batch_size'] or cfg['chunk_size']}, "
        f"preflight_soft={cfg['preflight_soft']}")
    # ---- R1: what this launch inherits from the last one --------------------
    prior_rows = read_prior_rows(
        path, resume=cfg["resume"] and not SMOKE_TIMING and not SMOKE_ONLY,
        expect={"model_key": MODEL_KEY, "model_id": model_id,
                "revision": REVISION, "shard": SHARD})
    say(f"  resume={cfg['resume']}, checkpoint every {cfg['ckpt_every']} steps "
        f"(0 = off); {len(prior_rows)} finished row(s) read back from "
        f"{path if os.path.exists(path) else '(no previous file)'}")
    for cell in cfg["cells"]:
        for seed in SEEDS:
            ck = maze_orgb.cell_checkpoint_path(
                work_dir, cell, seed, prefix=f"{cfg['adapter_prefix']}_ckpt")
            if os.path.exists(ck):
                say(f"  mid-cell checkpoint on disk for {cell}/s{seed}: {ck} "
                    f"({os.path.getsize(ck) / 2 ** 20:.1f} MB)")
    if SMOKE_TIMING:
        say(f"  (the line above is the SMOKE's shape; the timing probe trains "
            f"cell {cfg['timing_cell']} for {cfg['timing_steps']} steps at the "
            f"PRE-REGISTERED shape instead -- see TIMING below)")
    devs = deviations(cfg, maze_orgb)
    for d in devs:
        say(f"  DEVIATION: {d}")
    say(f"  budget: run_orgb's estimate is 4.5-7.5 h per cell of 300 steps; "
        f"this shard is {len(SEEDS) * len(CELLS)} cells")
    say_step_cost(maze_orgb, cfg)

    tok = AutoTokenizer.from_pretrained(model_id, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=REVISION, dtype=torch.bfloat16).to("cuda")
    model.eval()
    chat_kw = chat_kwargs(tok)
    say(f"model resident; chat_kwargs {chat_kw or '{} (plain template)'}")
    pre_tok = preflight_tokenizer(tok)
    say(f"PRE-FLIGHT: the warm start and the roll-out encode a prompt "
        f"identically ({pre_tok['prompt_tokens_train_eq_sample']} tokens); EOS "
        f"{pre_tok['eos_token']!r} -> [{pre_tok['eos_token_id']}]")
    nav = cfg.get("navigator") or {}
    say(f"  role glyphs from the navigator: {cfg.get('role_glyphs')} "
        f"(ASCII measures on_mine only; wall/goal are # and G)")
    say(f"  read from {nav.get('json_path')!r}; navigator adapter "
        f"{nav.get('adapter_path')!r} sha {(nav.get('adapter_sha256') or 'n/a')[:12]}; "
        f"valences {{"
        + ", ".join(f"{r}: {v.get('valence_centred', v.get('valence'))}"
                    for r, v in (nav.get("valences") or {}).items()) + "}")

    t0 = time.perf_counter()

    def on_row(row, payload):
        # Written after EVERY organism: a lease that dies on cell 7 of 8 leaves
        # seven measured organisms on disk rather than a status file.
        write_payload(path, payload)
        done = len(payload["rows"])
        total = len(SEEDS) * len(CELLS)
        # R1: a row CARRIED FORWARD from the previous lease cost this one
        # nothing, so the projection is over the cells this process actually
        # trains -- counting the free ones would say the shard is nearly done.
        reused = sum(1 for r in payload["rows"]
                     if (r.get("checkpoint") or {}).get("reused_row"))
        fresh = done - reused
        elapsed = time.perf_counter() - t0
        left = total - done
        eta = (elapsed / fresh * left) if fresh else None
        say(f"  WROTE {path} after {done}/{total} organisms ({reused} carried "
            f"forward, {fresh} trained here); elapsed {_hms(elapsed)}, {left} "
            f"to go, projected remaining {_hms(eta)}")

    payload = run_orgb.run(model, tok, cfg, work_dir, SEEDS, log=None,
                           on_row=(None if SMOKE_TIMING else on_row),
                           prior_rows=prior_rows)
    payload["manifest"].update({
        "model_key": MODEL_KEY, "model_id": model_id, "revision": REVISION,
        "git_sha": git_sha(), "torch_version": torch.__version__,
        "device": "cuda", "dtype": "bfloat16", "shard": SHARD,
        "smoke": bool(SMOKE_ONLY), "smoke_timing": bool(SMOKE_TIMING),
        "work_dir": work_dir, "chat_kwargs": chat_kw,
        "chat_enable_thinking": chat_kw.get("enable_thinking"),
        "nav_adapter": cfg.get("nav_adapter"),
        "nav_adapter_requested": NAV_ADAPTER,
        "nav_adapter_sha256": cfg.get("nav_adapter_sha256"),
        "nav_json": nav.get("json_path"),
        "role_glyphs": cfg.get("role_glyphs"),
        "navigator": nav,
        "deviations": devs,
        "preflight_tokenizer": pre_tok,
        "status_file": STATUS,
        # R1: what this launch was ASKED to do about a dead lease, beside what
        # `run_orgb` recorded that it actually did (`manifest["resume"]`).
        "resume_env": {"MZ01_RESUME": bool(RESUME),
                       "MZ01_CKPT_EVERY": int(cfg["ckpt_every"]),
                       "result_file_read_back": bool(prior_rows),
                       "prior_rows": len(prior_rows)},
    })

    write_payload(path, payload)
    say(f"WROTE {path}")

    # ---- the contracts, asserted AFTER the write ---------------------------
    if payload.get("not_measured"):
        for note in payload["not_measured"]:
            say(f"  NOT-MEASURED: {note}")
    if SMOKE_TIMING:
        say_timing(payload["smoke_timing"])
        say("MZ01 ORGB TIMING DONE (MZ01_SMOKE_TIMING=1; no cell was gated, "
            "no row was written)")
        return
    if not payload["rows"]:
        say("MZ01 ORGB STOPPED: no organism was trained (see not_measured "
            "above; the pre-flight gate is the usual reason)")
        return
    for row in payload["rows"]:
        gaps = [k for k in run_orgb.ROW_KEYS if k not in row]
        assert not gaps, f"row {row.get('cell')}/{row.get('seed')} is missing {gaps}"
        m = row["measurement"]
        assert m["competence"]["n_episodes"] == cfg["eval_n"], (
            f"{row['cell']}/{row['seed']} measured "
            f"{m['competence']['n_episodes']} episodes, not {cfg['eval_n']}")
        for rec in m["records"]:
            assert rec["stop_on_parse"] is False and rec["fmt"] == FORMAT
            assert rec["max_new_tokens"] == cfg["max_new_tokens"]
            assert sum(rec["retried"]) == rec["retries"]
            assert rec["order_seed"] is not None, (
                "every measurement pass must derive its word orders per episode "
                "(maze2 D31) or one divergence desynchronises the rest")
        assert row["training"]["config"]["switches"] == row["switches"]
        # critic 3: shards are pooled, so every row must say which warm start
        # it stands on and which cue glyph it was rendered with.
        assert row.get("warm_start_sha256"), (
            f"{row['cell']}/{row['seed']} carries no warm-start sha, so the "
            "scorer cannot prove it shares M5's ONE warm start")
        assert row.get("role_glyphs") == (cfg.get("role_glyphs") or {}), (
            f"{row['cell']}/{row['seed']} was rendered with role glyphs "
            f"{row.get('role_glyphs')}, not the run's {cfg.get('role_glyphs')}")
        assert row.get("nav_adapter_sha256") == cfg.get("nav_adapter_sha256"), (
            f"{row['cell']}/{row['seed']} stands on navigator "
            f"{row.get('nav_adapter_sha256')}, not this run's "
            f"{cfg.get('nav_adapter_sha256')}; pooled shards would then mix two "
            "bases under one design (M2)")
    say("row contract OK")
    res = payload["manifest"].get("resume") or {}
    if res.get("rows_reused") or res.get("cells_resumed_mid_cell"):
        say(f"  RESUMED: rows carried forward {res.get('rows_reused')}; cells "
            f"picked up mid-flight {res.get('cells_resumed_mid_cell')} -- a "
            "resumed cell continues from every piece of the loop's state (R1). "
            "Bit-exactness is asserted on CPU fp32 only; on CUDA bf16 this is "
            "the same recipe from the same state, not the same floats")

    for cell in cfg["cells"]:
        pr = payload["pass_rules"].get(cell, {})
        say(f"  cell {cell}: joint {pr.get('n_joint')} of {pr.get('n_seeds')} "
            f"seeds (need {pr.get('need')}); diversity median "
            f"{pr.get('diversity_median')} (bar {pr.get('diversity_bar')})")
    for row in payload["rows"]:
        say(f"  ledger {row['cell']}/{row['seed']}:")
        for line in (row["ledger"] or {}).get("lines", []):
            say("    " + line)
    r0 = payload["rows"][0]["measurement"]["records"][0]
    say("first replies of the first organism: "
        + repr(r0["replies"][:3]))
    if SMOKE_ONLY:
        say("MZ01 ORGB DONE (MZ01_SMOKE_ONLY=1; no full run)")
        return
    say(f"MZ01 ORGB DONE shard={SHARD} rows={len(payload['rows'])} "
        f"elapsed={_hms(time.perf_counter() - t0)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say("MZ01 ERROR:\n" + traceback.format_exc())
