"""MZ01 Stages 1-3: the warm start, the 2x2 of cells, and the measurement.

THE PRE-COMMITMENTS THIS FILE EXECUTES

Everything here is the rewritten pre-registration at the end of
`PLAN-orgb-v2-CRITIQUE.md`, not `PLAN-orgb-v2.md`. `src/calibration/maze_orgb.py`
implements it; this file is the ORDER OF OPERATIONS and the artefact contract,
and both are stated before the lease so neither can be chosen after seeing a
number.

  1. THE BASE IS THE ONE SHARED NAVIGATOR (M2). `results/navigator/nav_shared.pt`,
     the adapter of SPEC "STAGE 0 v2", gated separately -- this file ASSUMES it
     passed and refuses to run without the file. There is one navigator for all
     seeds, so a seed varies only the RL stream and a "3 of 4" rule is not a
     mixture of navigator variance and RL variance.

  2. ONE MIXED WARM START FOR ALL FOUR CELLS (M5). The navigator's own moves,
     with a remark on a uniformly random 40 % of turns drawn half from the
     aversive pool and half from the affectless one, UNRELATED to the mine, with
     `sft.train_sft`'s move anchor pinned to the navigator's own 4-way
     distribution so the stage teaches vocabulary and moves no policy. The
     contingency after it must be ~0 -- checked on the corpus by construction
     and on SAMPLED replies in the pre-flight -- and the warm-started adapter is
     saved ONCE, as one file, which EVERY cell of EVERY seed starts from.

  3. THE PRE-FLIGHT GRADIENT GATE (PR-3). Before any cell trains: >= 0.9 of
     (maze, class) groups have non-zero yell variance and the within-group yell
     sd is >= 0.2. If not, the run writes NOT-MEASURED and STOPS. This is the
     single check NAR04 lacked; it spent eight seeds discovering that its warm
     start had made every group class-homogeneous.

  3b. THE EXPLORATION GATE (R2, 2026-08-29), and why there are now two.
     PR-3 asks whether the REMARK term has variance. Nobody asked the same
     question of the MOVE term, and on the first three shards the move term was
     DEAD: the warm-started navigator is nearly deterministic (move-column
     entropy 0.005-0.07 against a ceiling of ln 4 = 1.386), so a maze's six
     sampled episodes are the SAME episode, their returns are equal,
     `group_advantages` returns zeros and the policy gradient is exactly
     nothing -- |g| ~ 0.01 at solve 1.000 in their live traces, for 90 steps.
     Cell A cannot learn avoidance in that state and cell B's "policy
     invariance" would be the invariance of a policy nothing moved.
     `maze_orgb.preflight_exploration_check` measures, on the warm-started
     model and before any cell starts: the mean move-column entropy, the share
     of maze groups whose returns are all identical, and the mean within-group
     return sd. It STOPS the run when the degenerate share is over 0.80 or the
     entropy target in force is not within [0.5x, 3x] of the measured entropy.
     MZ01_PREFLIGHT_SOFT=1 reports it instead and records the deviation.
     It also RESOLVES the exploration arm (MZ01_EXPLORE), because `relative`'s
     entropy target is a function of the entropy it just measured -- so the
     cells train under exactly the numbers the gate read.

  4. FOUR CELLS, ONE TRAINER, TWO SWITCHES (B1/B2). D (both off), A (score on),
     B (yell on), C (both). D RUNS THE RL -- it is not the warm-started
     navigator left alone -- because HH01 measured its two positive instruments
     separating just as hard on seeds where the RL installed nothing. The
     objective is two policy-gradient terms over DISJOINT token sets: the
     episode-return advantage on the MOVE tokens, the yell advantage on the
     REMARK tokens, weight `w = maze_orgb.YELL_WEIGHT`, pre-registered.

  5. MEASUREMENT ON THE IDENTICAL HELD-OUT MAZES (PR-5), greedy, one retry,
     `max_new_tokens=48` with the parse-stop OFF, the per-turn word order
     derived per (order_seed, maze seed, decision index) so every cell is asked
     the same question: competence, `mine_step_rate`, the price tables
     (`min_mines_on_shortest`, `price = 10m - 2d`, the d+m regression, regret),
     complaint presence and contingency on BOTH detectors, remark diversity, the
     placebo pass, both novel-glyph pairings, the 50-step relearn with the
     adapter restored, and the out-of-domain instruments I2/I3/I4/I5/I7 on the
     mine glyph against the placebo glyph with no maze in the prompt.

  6. PASS RULES (PR-6). One primary criterion -- aversive contingency on the
     HELD-OUT lexical detector > 0.5 -- with M10's ledger evaluated per seed
     beside it, every band paired and +-0.15 (M7), the bundling read TWO-SIDED,
     the joint seed count reported as well as the per-criterion counts (A12),
     and NOT-MEASURED wherever a gate cannot be read.

WHAT THIS RUN CANNOT CLAIM (I1, stated before the numbers exist)

Cell B is not "the words without the state". Maximising a complaint reward
REQUIRES a function of the observation that agrees with the `on_mine` indicator
and drives an output, so B necessarily has a state -- and with the score feed
off, the on-mine turn is the only place B's positive reward lives, so the mine
is mildly APPETITIVE in B's own return. The question this run asks is narrower
and better: whether a contingency installed through the verbal channel alone
leaks into channels it was never trained on -- movement, latent readiness
(relearn), generalisation (novel glyphs, the affectless contrast), and the
out-of-domain welfare instruments. A positive instrument reading with nulls
everywhere else is the calibration result: the instruments cannot tell a
verbally-installed contingency from a preference.

COMPUTE BUDGET, WRITTEN DOWN BEFORE THE FIRST LEASE (PR-7)

Pre-registered scale: **4 seeds x 4 cells x 300 steps, batch 4 mazes x 6
episodes per maze (24 episodes/step), Qwen3-14B, LoRA r=16 a=32 on q/v.**

Per step, measured shapes rather than hopes: 24 episodes at n=9 with a cap of 24
run ~12-14 decisions each, so ~300 decisions/step. The sampler is the dominant
cost and it is a HAND-ROLLED DECODE WITH NO KV CACHE (as `maze2._greedy` is):
~14 lockstep rounds x ~10 generated tokens x one batched forward over ~24 live
prompts of ~350 tokens. The re-scoring pass is ~38 forward+backward of chunk 8.
At 14B bf16 on one 96 GB card that is roughly:

  sampling      35-60 s/step
  re-scoring    15-30 s/step
  ------------------------------
  total         50-90 s/step  ->  **4.5-7.5 h per cell of 300 steps**

  16 cells (4 seeds x 4)                    **75-120 GPU-hours**
  warm start (3 000 decisions x 2 epochs)   ~0.5-1 h, ONCE
  pre-flight (64 mazes x 8 episodes)        ~10-15 min, ONCE
  measurement per organism: 4 passes x 256  ~1.5-2.5 h  ->  **25-40 h**
  relearn per organism: 50 steps + 1 pass   ~1.5-2 h    ->  **25-32 h**
  instruments per organism                  ~2 min      ->  ~0.5 h
  ------------------------------------------------------------------
  TOTAL                                     **125-195 GPU-hours**, i.e.
                                            **11-17 leases of 12 h**

That is an honest estimate and it is larger than one lease by an order of
magnitude, so the shards are the unit of work: `MZ01_SEEDS` and `MZ01_CELLS`
split the grid, each shard writes its own file, and the scorer pools them. The
levers if the estimate proves optimistic, in the order they should be pulled and
recorded when they are: drop the relearn read (it is REPORTED with its power,
not a pass criterion -- I5), drop the second novel pairing, drop the placebo
pass (and read the graded claim as NOT-MEASURED), and only then cut steps --
which changes the pre-registered recipe and is a deviation, not a saving.

DEVIATIONS FROM THE CRITIQUE'S REWRITE, RECORDED HERE AND NOWHERE ELSE

  * PR-1 asks for TWO navigators, one per counterbalance parity. There is ONE
    (SPEC "STAGE 0 v2", M2's stronger form): the counterbalance lives inside its
    corpus. The user's decision of 2026-08-28, already executed and gated.
  * PR-7 makes 4B the primary model and 7 the size. This runs at 14B, n=9,
    because that is the model and size the shared navigator was trained and
    gated at, and a cell whose base is a different model is not this design.
  * PR-2 mixes the warm-start remark pool aversive 40 % / affectless 40 % /
    filler 20 %. Here it is half aversive, half affectless, with FILLER unused
    (`stats["filler_unused"]` records it): filler is a third vocabulary whose
    only role in the old world was ORG-B's non-adjacent case, and the two pools
    that matter for the contrast are the ones with matched structure.
  * Seeds are 4, not 8. The seed-count rule is a FRACTION
    (`SEED_PASS_FRACTION = 0.75`, i.e. "3 of 4" where the critique wrote "6 of
    8"), so the bar means the same thing; the power does not, and that is a cost
    of the budget rather than a change to the standard.
  * The relearn read is one 50-step trajectory per organism, not I5's >= 3
    relearn seeds. It is reported with its power and is not a pass criterion.

WHAT SURVIVES A DEAD LEASE (R1, added 2026-08-29)

A lease is capped at 12 h, can die at any minute, and a RENEWED lease restores
an OLD DISK SNAPSHOT -- so an artefact that was never pulled is gone. Three
things are therefore written as the run goes, and all three land under the
shard's `out_dir`, which is inside `experiments/**/results*/` and so is mirrored
by `scripts/pull_from_sandbox.sh` without changing its glob:

  1. `{shard}.json` after EVERY organism (the driver's `on_row`);
  2. `{adapter_prefix}_{cell}_s{seed}.pt`, the organism's own adapter, with its
     sha256 in the row, as soon as its training ends and before the ~2 h
     measurement starts;
  3. `{adapter_prefix}_ckpt_{cell}_s{seed}.pt` every `ckpt_every` RL steps: the
     WHOLE loop state (adapter, AdamW moments, step counter, the adaptive
     entropy coefficient, the RNG streams, the trace so far), written atomically.

On the next launch (MZ01_RESUME=1, the default) a finished row is carried
forward from (1) and its cell is skipped entirely; a cell with a checkpoint from
(3) resumes at that step from every piece of the loop's state. That resume is
asserted BIT-EXACT on the CPU fp32 toy path only: on CUDA with a bf16 base the
forward and the backward are not bit-reproducible run to run, so a resumed cell
is the same recipe continued from the same state and not the same floats.
The checkpoint is ~126 MB at 14B (an fp32
adapter and two fp32 AdamW moments), which is minutes on the puller's exec
channel -- read `AGENT-NOTES.md` before relying on pulling one mid-lease.
"""

from __future__ import annotations

import os
import time

import torch

from calibration import maze2, maze_orgb
from calibration.chat import apply_chat, chat_kwargs
from calibration.lora import (has_lora, inject_lora, load_lora_state_dict,
                              lora_state_dict, save_lora)
from calibration.rl import _EVAL_SEED_RANGE, _TRAIN_SEED_RANGE
from calibration.runner import derive_generator

__all__ = ["run", "DEFAULT_CONFIG", "ROW_KEYS"]


DEFAULT_CONFIG = {
    "n": 9,
    "fmt": maze2.FMT_ASCII,
    "cells": list(maze_orgb.CELLS),
    "steps": 300,
    "batch_size": 4,
    "group_size": 6,
    "lr": 1e-4,
    "temperature": 1.0,
    "yell_weight": maze_orgb.YELL_WEIGHT,
    "max_new_tokens": maze_orgb.REPLY_MAX_NEW_TOKENS,
    "chunk_size": 8,
    "entropy_coef": 0.01,
    "entropy_target": 0.7,
    # ---- TWELFTH PASS, 2026-09-07: the closed action space -----------------
    # `move_mask` is the recipe now. The first sampled token is drawn from the
    # four BARE move ids and the move term's first-token log-prob is that same
    # restricted 4-way softmax, so emission, gradient, entropy and probe are one
    # object -- which is the 08-30 erratum's fix. `move_mass_coef` therefore has
    # nothing left to do and DROPS TO 0.0 (it was 0.03 through 08-30, where it
    # was the only term holding mass on the bare ids, it was too weak, and
    # raising it was not the fix). `move_mass` is still recorded every step as a
    # diagnostic, and both knobs stay settable so the 08-30 recipe is
    # reproducible by knob.
    "move_mass_coef": 0.0,
    "move_mask": True,
    "pool_size": 256,
    # ---- R2, exploration (2026-08-29) --------------------------------------
    # THE DEFAULT IS TODAY'S RECIPE, so the three shards already training stay
    # reproducible from this file. The other three modes are the trial's arms,
    # and `maze_orgb.exploration_plan` is the only place a name becomes numbers.
    "explore_mode": maze_orgb.DEFAULT_EXPLORE_MODE,
    "entropy_rel": maze_orgb.ENTROPY_REL,
    "entropy_min": maze_orgb.ENTROPY_MIN,
    "entropy_coef_fixed": maze_orgb.ENTROPY_COEF_FIXED,
    "sample_temperature": maze_orgb.SAMPLE_TEMPERATURE,
    # TENTH PASS. `epsilon`'s rate, and the flag that holds a thermostat mode's
    # coefficient still. Both are inert under the default mode -- `plan_epsilon`
    # returns 0.0 for every mode but `epsilon`, and `thermostat_off` is False --
    # so the running shards' recipe is unchanged bit for bit.
    "epsilon": maze_orgb.EXPLORE_EPSILON_RATE,
    "thermostat_off": False,
    # MZ01_ENTROPY_TARGET. None => the mode derives its own target. Set it to
    # the number a FIRST lease measured so a second lease resumes the same
    # recipe instead of re-deriving one a hair away (see `exploration_plan`).
    "entropy_target_pinned": None,
    # How wide ONE decode call is during RL sampling. None => `chunk_size`,
    # which is what it silently was before the two were separated.
    "sample_batch_size": None,
    # The exploration pre-flight's roll-out: K mazes x `group_size` episodes on
    # the warm-started model. 8 x 6 = 48 episodes is about one RL step of
    # sampling, which is the cheapest read that can see a group at all.
    "preflight_explore_mazes": 8,
    # MZ01_PREFLIGHT_SOFT=1. Reports the exploration gate instead of applying
    # it. Needed by any arm whose entropy target is deliberately far from what
    # the policy does today -- `relative`'s floor is exactly that -- and by the
    # trial driver, which runs every arm including the ones that fail it.
    "preflight_soft": False,
    # the warm start
    "warm_start_decisions": 3000,
    "warm_start_epochs": 2,
    "warm_start_batch": 8,
    "warm_start_lr": 1e-4,
    "warm_start_anchor_coef": 1.0,
    "warm_start_mazes": 256,
    "remark_share": maze_orgb.WARM_START_REMARK_SHARE,
    # the pre-flight
    "preflight_mazes": 64,
    "preflight_group_size": 8,
    # measurement
    "eval_n": 256,
    # R2: 32, up from 16. The four measurement passes are 256 mazes x ~13
    # lockstep rounds each and the decode width is the only knob on them;
    # greedy decoding is batch-invariant in exact arithmetic and the toy path
    # asserts it decision by decision (`measure_organism`'s docstring says what
    # that does and does not claim on CUDA bf16). The driver's MZ01_EVAL_BATCH.
    "eval_batch_size": 32,
    "placebo": True,
    "novel": True,
    "relearn": True,
    "instruments": True,
    "relearn_steps": maze_orgb.RELEARN_STEPS,
    # artefacts
    "lora": {"r": 16, "alpha": 32, "target_modules": ["q_proj", "v_proj"]},
    "nav_adapter": "experiments/v2/MZ01_maze_world/results/navigator/nav_shared.pt",
    # The sha the navigator's OWN result file recorded for that adapter, read
    # off `results/navigator/{name}.json` by the driver. `run` re-hashes the
    # file it actually loads and refuses to start if the two disagree: the
    # adapter and the role glyphs below have to come from ONE navigator run, and
    # a stale sidecar beside a re-trained adapter is exactly how they come apart.
    "nav_adapter_sha256": None,
    # critic 3, BLOCKING. `glyphs_for_seed(seed, counterbalance=True)` with no
    # role glyphs falls through to `Glyphs()`'s dataclass DEFAULT, `on_mine =
    # 🟧`. The shared navigator was trained and gated with `on_mine = ●`, the
    # glyph `choose_role_glyphs` MEASURED at centred valence 0.000 on this
    # checkpoint, and 🟧 is the orange square SPEC C2 rejected at +3.09 as "a
    # prior the narration result would then be partly a measurement of". Under
    # ASCII `on_mine` is the ONLY measured role (`FORMAT_ROLES`), so it is the
    # whole manipulation. Running the organism stage on a cue glyph the
    # navigator never saw would put an ungated base under every cell. The
    # driver reads these from the navigator's own result file; `run` REFUSES to
    # start with a navigator adapter and no role glyphs.
    "role_glyphs": None,
    # M5's "saved ONCE": a shard that finds this file LOADS it instead of
    # training a second warm start, because two warm starts would put a
    # different base under different cells of one design -- the confound M2
    # removed at the navigator, reintroduced by the sharding.
    "warm_start_adapter": None,
    # Artefact names. A smoke must never overwrite a real adapter, so the
    # driver renames both rather than writing into the same files.
    "warm_start_name": "warm_start",
    "adapter_prefix": "orgb",
    "keep_records": True,
    # ---- R1, the lease-death recovery (2026-08-29) -------------------------
    # A cell is ~3 h of a 12 h lease and a renewed lease restores an OLD DISK
    # SNAPSHOT, so everything not pulled is gone. Three things now survive a
    # death: the shard JSON after every organism (the driver's `on_row`, already
    # there), the per-cell adapter when the cell ends (`save_lora` below,
    # already there) and -- new -- the RL loop itself, every `ckpt_every` steps.
    # `resume` False re-runs a cell from step 0 even with a checkpoint on disk;
    # `ckpt_every` 0 writes none. Both are the driver's MZ01_RESUME /
    # MZ01_CKPT_EVERY.
    "ckpt_every": maze_orgb.CKPT_EVERY,
    "resume": True,
    # ---- the smoke-only timing probe (MZ01_SMOKE_TIMING) -------------------
    # `smoke` is the driver's MZ01_SMOKE_ONLY, carried into the config so this
    # file can refuse the timing probe on anything else. `smoke_timing` turns
    # the two gates into NOT-MEASURED-AT-SMOKE-SCALE reports, trains ONE cell
    # for `timing_steps` at the PRE-REGISTERED batch shape, and returns with no
    # rows and no organism on disk. It exists because a smoke that stops at the
    # pre-flight measures no `seconds_per_step`, and `seconds_per_step` is the
    # number the lease is booked on.
    "smoke": False,
    "smoke_timing": False,
    "timing_cell": "B",
    "timing_steps": 3,
    # None => the pre-registered value. Overridden ONLY by the suite, which
    # cannot afford 24 episodes on a four-parameter transformer; when they are
    # overridden the report says `at_pre_registered_shape: False` and the
    # measurement is not a costing of this design.
    "timing_batch_size": None,
    "timing_group_size": None,
    "timing_chunk_size": None,
    "timing_max_new_tokens": None,
}

# The shape a cell is costed at. Read off DEFAULT_CONFIG so it cannot drift
# from the recipe the cells actually run.
_TIMING_SHAPE_KEYS = {"timing_batch_size": "batch_size",
                      "timing_group_size": "group_size",
                      "timing_chunk_size": "chunk_size",
                      "timing_max_new_tokens": "max_new_tokens"}

# The per-organism row contract. The driver's smoke asserts these keys, and the
# scorer indexes into them: a row that grew a key or lost one between a smoke
# and a lease is the failure that costs the lease.
ROW_KEYS = ("cell", "seed", "switches", "adapter_path", "adapter_sha256",
            "warm_start_sha256", "role_glyphs", "nav_adapter_sha256",
            "training", "measurement", "ledger", "relearn", "instruments",
            "seconds", "checkpoint")

# The row fields a REUSED row (one already in the shard JSON from an earlier
# lease) must agree with this run on before it is carried forward instead of
# retrained. Every one of them is an identity of the BASE the organism stands
# on: a shard that pooled a row built on another navigator, another warm start
# or another cue glyph would reintroduce exactly the confound M2 and M5 removed.
RESUME_IDENTITY_KEYS = ("nav_adapter_sha256", "warm_start_sha256", "role_glyphs")

# CRITIC PASS 2026-08-29, DEFECT FIXED. The keys above say a carried-forward row
# stands on the same BASE; these say it was built by the same RECIPE. Each pair
# is (`row["training"]["config"]` key, this run's `cfg` key). Without them a
# relaunch under a different MZ01_YELL_W / MZ01_GROUP / MZ01_BATCH / entropy
# target carried the old rows forward SILENTLY: the payload's manifest records
# only the NEW config, nothing downstream re-reads a row's own training config,
# and the 2x2 would then be half one recipe and half another with no field
# anywhere saying so. `train_orgb_cell` already refuses exactly this at the
# mid-cell seam (its checkpoint fingerprint names the field that disagrees);
# this is the same refusal one level up, where the cost of missing it is a whole
# measured shard rather than one cell.
RESUME_RECIPE_KEYS = (
    ("steps", "steps"), ("n", "n"), ("batch_size", "batch_size"),
    ("group_size", "group_size"), ("lr", "lr"),
    ("temperature", "temperature"), ("max_new_tokens", "max_new_tokens"),
    ("entropy_coef_initial", "entropy_coef"),
    ("entropy_target", "entropy_target"),
    ("move_mass_coef", "move_mass_coef"), ("chunk_size", "chunk_size"),
    ("pool_size", "pool_size"), ("fmt", "fmt"),
    # R2, 2026-08-29. The exploration arm is a recipe field like any other, and
    # the two below are the ones a row can differ on without any other field
    # moving: `relative` and `thermostat_fixed_target` differ only in the target
    # (already checked), but `fixed` and `temperature` differ in the CONTROLLER
    # and in the SAMPLER, and a 2x2 half sampled at T=1 and half at T=1.5 would
    # carry no field anywhere saying so.
    ("explore_mode", "explore_mode"),
    ("sample_temperature", "sample_temperature"),
    # TENTH PASS: and the epsilon rate, for the same reason one knob over. Two
    # rows at eps 0.1 and 0.2 took DIFFERENT ACTIONS at a tenth of their
    # decisions and no other field in the row would say so.
    ("epsilon", "epsilon"),
    # TWELFTH PASS: and the action space. A row trained with the first token
    # masked to the four bare ids and a row trained with it open are two
    # different policies scored by two different estimators, and pooling them
    # into one 2x2 is exactly what this list exists to refuse.
    ("move_mask", "move_mask"),
)


def _same_recipe_value(a, b):
    """Equality for one config field, tolerant of a float's JSON round trip."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-12
    return a == b


def _sha256_file(path):
    """The sha of an adapter file on disk. M5's identity, across shards."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _say(msg, log=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log is not None:
        log(line)


def _make_warm_start(model, tok, cfg, ws_path, ws_glyphs, train_pool, payload,
                     device, fmt, n, log):
    """Stage 1, M5: roll the navigator, build the mixed corpus, train, save ONCE.

    The moves are the NAVIGATOR'S OWN (`maze2.rollout` records), the remark
    placement is a 40 % Bernoulli independent of the mine, and the pool is half
    aversive and half affectless. `sft.train_sft`'s move anchor is captured on
    the templated prompts in the corpus's own order, so row i is pinned to row
    i's own base policy: a misaligned anchor would pin every example to another
    example's move distribution, silently, in the one stage whose job is to move
    nothing.

    Returns the warm-started adapter state, which every cell of every seed is
    reloaded from.
    """
    ws_mazes = train_pool[:cfg["warm_start_mazes"]]
    _say(f"warm start: rolling the navigator over {len(ws_mazes)} training "
         f"mazes for its own moves", log)
    nav_records, nav_agg = maze2.rollout(
        model, tok, ws_mazes, retry=True, glyphs=ws_glyphs, fmt=fmt,
        batch_size=cfg["eval_batch_size"], device=device,
        max_new_tokens=cfg["max_new_tokens"], stop_on_parse=False,
        order_seed=1000 + n)
    corpus, ws_stats = maze_orgb.build_warm_start_corpus(
        nav_records, generator=derive_generator(0, "orgb", "warm_start"),
        remark_share=cfg["remark_share"], fmt=fmt,
        max_items=cfg["warm_start_decisions"])
    anchor_cols = maze_orgb._move_cols(tok, device)
    templated = [apply_chat(tok, [{"role": "user", "content": it["prompt"]}],
                            add_generation_prompt=True, tokenize=False)
                 for it in corpus]
    anchor = (anchor_cols,
              maze_orgb.capture_move_anchor(
                  model, tok, templated, batch_size=cfg["eval_batch_size"],
                  device=device, cols=anchor_cols))
    ws_hist = maze_orgb.train_warm_start(
        model, tok, corpus, epochs=cfg["warm_start_epochs"],
        lr=cfg["warm_start_lr"], batch_size=cfg["warm_start_batch"],
        anchor=anchor, anchor_coef=cfg["warm_start_anchor_coef"],
        shuffle_generator=derive_generator(0, "orgb", "warm_start_shuffle"),
        log_every=0)
    ws_sha = save_lora(model, ws_path)
    payload["warm_start"] = {
        "navigator_rollout": nav_agg["overall"],
        "corpus": ws_stats,
        "training": ws_hist,
        "adapter_path": ws_path,
        "adapter_sha256": ws_sha,
        "note": ("ONE warm start, saved ONCE; every cell of every seed starts "
                 "from this file (M5)"),
    }
    band = maze_orgb.WARM_START_CONTINGENCY_BAND
    ccb = ws_stats["corpus_contingency"]
    cc, se = ccb["contingency"], ccb["contingency_se"]
    _say(f"warm start: {ws_stats['n_items']} items "
         f"({ccb['n_on_mine']} on-mine, {ccb['n_other']} other), remark share "
         f"{ws_stats['remark_share_realised']:.3f}, corpus contingency "
         f"{cc:.4f} +- {'n/a' if se is None else '%.4f' % se} (band +-{band}, "
         f"resolvable {ccb['band_is_resolvable']}); loss "
         f"{ws_hist['initial_loss']:.4f} -> "
         f"{ws_hist['final_loss']:.4f}; adapter {ws_path}", log)
    if cc is not None and abs(cc) > band:
        if cfg.get("smoke_timing"):
            # (a) A toy corpus reads a contingency whose own sd is several
            # times the band, so "outside the band" is a draw and not a
            # finding. The gate is REPORTED and the probe continues; it is
            # never a pass at smoke scale either.
            payload["not_measured"].append(
                f"NOT-MEASURED-AT-SMOKE-SCALE: the warm-start corpus read "
                f"contingency {cc:.4f} (sd "
                f"{'n/a' if se is None else '%.4f' % se}) against a band of "
                f"+-{band} on {ccb['n_on_mine']} on-mine items. This is a "
                "timing probe, so the gate is reported and not applied; a "
                "corpus this size decides nothing either way.")
        else:
            payload["not_measured"].append(
                f"the warm-start corpus carries contingency {cc:.4f} outside "
                f"+-{band}: the placement stream was correlated with the cue "
                "before any training, and every contingency downstream "
                "inherits it")
    return lora_state_dict(model, cpu=True)


def _timing_probe(model, tok, cfg, warm_state, train_pool, roles, device, log,
                  plan=None):
    """MZ01_SMOKE_TIMING. ONE cell, `timing_steps` steps, the clock, nothing else.

    WHAT THIS IS FOR. `step_cost_estimate` derives a per-step cost from an
    assumed token count and the navigator lease's SFT throughput, and E10 has
    already caught that derivation being 2-3x optimistic once. The lease is
    booked on that number. This probe is the only thing between the estimate and
    the booking: it runs the REAL trainer, at the PRE-REGISTERED batch shape,
    for three steps, and reports what the clock says -- sampling and re-scoring
    apart, because the estimate's whole claim is that the cache moved the split.

    WHAT IT IS NOT. It is not a cell, not a measurement and not a gate reading.
    Nothing is saved: no adapter, no row, no ledger. The organism it trains for
    three steps is discarded with the process, and `run` returns immediately
    after with `rows == []`, so no pooled scorer can ever see it.

    THE SHAPE IS THE REGISTERED ONE even though everything around it is a smoke:
    a cost measured at 2 mazes x 3 episodes with an 8-token window says nothing
    about 4 x 6 at 48, which is the entire reason the last smoke left the
    question open.
    """
    assert cfg.get("smoke"), (
        "the timing probe trains a cell without a gate; it runs ONLY inside a "
        "smoke (config['smoke'] is the driver's MZ01_SMOKE_ONLY=1)")
    cell = cfg["timing_cell"]
    if cell not in maze_orgb.CELL_SWITCHES:
        raise ValueError(f"timing_cell must be one of {list(maze_orgb.CELLS)}, "
                         f"got {cell!r}")
    shape, at_registered, off = {}, True, []
    for key, target in _TIMING_SHAPE_KEYS.items():
        want = DEFAULT_CONFIG[target]
        got = want if cfg.get(key) is None else cfg[key]
        shape[target] = got
        if got != want:
            at_registered = False
            off.append(f"{target}={got} (registered {want})")
    if cfg["n"] != DEFAULT_CONFIG["n"]:
        at_registered = False
        off.append(f"n={cfg['n']} (registered {DEFAULT_CONFIG['n']})")
    steps = int(cfg["timing_steps"])
    seed = 0
    glyphs = maze2.format_glyphs(
        maze2.glyphs_for_seed(seed, counterbalance=True, **roles), cfg["fmt"])
    # The cell starts from the warm start, as a real cell does.
    load_lora_state_dict(model, warm_state)
    kv = maze_orgb.kv_cache_report(model, tok, device=device)
    _say(f"TIMING PROBE: cell {cell} {maze_orgb.CELL_SWITCHES[cell]}, {steps} "
         f"steps at {shape['batch_size']} mazes x {shape['group_size']} "
         f"episodes, chunk {shape['chunk_size']}, window "
         f"{shape['max_new_tokens']}, n={cfg['n']}; KV cache engaged "
         f"{kv['kv_cache_engaged']}, logits_to_keep {kv['logits_to_keep']}", log)
    if not at_registered:
        _say("  !! NOT THE PRE-REGISTERED SHAPE: " + "; ".join(off)
             + " -- this is not a costing of this design", log)
    cuda = getattr(device, "type", None) == "cuda"
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    # R2: the probe times THE ARM THAT WILL RUN. A cost measured under the
    # pre-registered thermostat says nothing about `temperature`, whose decode
    # is the same width but whose replies are drawn hotter (and, at the same
    # window, usually longer).
    plan = plan or maze_orgb.exploration_plan(
        cfg["explore_mode"], measured_entropy=None,
        entropy_coef=cfg["entropy_coef"], entropy_target=cfg["entropy_target"],
        entropy_coef_fixed=cfg["entropy_coef_fixed"],
        sample_temperature=cfg["sample_temperature"],
        epsilon=cfg.get("epsilon", maze_orgb.EXPLORE_EPSILON_RATE),
        thermostat_off=bool(cfg.get("thermostat_off")),
        temperature=cfg["temperature"])
    hist = maze_orgb.train_orgb_cell(
        model, tok, cell=cell, seed=seed, steps=steps, n=cfg["n"],
        batch_size=shape["batch_size"], group_size=shape["group_size"],
        lr=cfg["lr"], temperature=cfg["temperature"],
        yell_weight=cfg["yell_weight"],
        max_new_tokens=shape["max_new_tokens"],
        entropy_coef=plan["entropy_coef"],
        entropy_target=plan["entropy_target"],
        explore_mode=plan["mode"], explore_knobs=plan["knobs"],
        sample_temperature=plan["sample_temperature"],
        epsilon=plan["epsilon"],
        sample_batch_size=cfg.get("sample_batch_size"),
        move_mass_coef=cfg["move_mass_coef"],
        move_mask=bool(cfg.get("move_mask", True)),
        chunk_size=shape["chunk_size"],
        train_pool=train_pool, glyphs=glyphs, fmt=cfg["fmt"], log_every=1)
    memory = None
    if cuda:
        memory = {
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30,
            "device_total_gb": (torch.cuda.get_device_properties(device).total_memory
                                / 2 ** 30),
            "note": ("peak over the probe only; the measurement passes and the "
                     "instrument panel are NOT in it"),
        }
    measured = maze_orgb.measured_step_cost(hist, projection_steps=300)
    est = maze_orgb.step_cost_comparison(
        batch_size=shape["batch_size"], group_size=shape["group_size"],
        chunk_size=shape["chunk_size"],
        max_new_tokens=shape["max_new_tokens"])
    # Both are PER-STEP lists in the history; the estimate's assumption is a
    # single number, so the means are what compare against it.
    n_dec = hist["n_decisions"]
    ep_len = hist["episode_length"]
    mean_dec = (sum(n_dec) / len(n_dec)) if n_dec else None
    mean_ep = (sum(ep_len) / len(ep_len)) if ep_len else None
    # The estimate re-derived at the episode length that was actually observed.
    # Without it a 2x gap between measured and estimated cannot be told apart
    # from the roll-out simply running longer episodes than the header assumed,
    # and the two call for opposite responses.
    est_at_measured = (
        maze_orgb.step_cost_estimate(
            batch_size=shape["batch_size"], group_size=shape["group_size"],
            chunk_size=shape["chunk_size"],
            max_new_tokens=shape["max_new_tokens"],
            decisions_per_episode=max(1, int(round(mean_ep))))
        if mean_ep else None)
    ref = measured["steady_state"] or measured["all_steps"]
    return {
        "cell": cell,
        "seed": seed,
        "steps": steps,
        "shape": dict(shape, n=cfg["n"], temperature=cfg["temperature"],
                      pool_size=len(train_pool)),
        "at_pre_registered_shape": at_registered,
        "shape_deviations": off,
        "explore": plan,
        "kv_cache": kv,
        "measured": measured,
        "estimate": est,
        "measured_over_estimate": (
            None if not est["cached"]["seconds_per_step"]
            else ref["seconds_per_step"] / est["cached"]["seconds_per_step"]),
        "decisions_per_step_measured": mean_dec,
        "decisions_per_episode_measured": mean_ep,
        "decisions_per_episode_assumed": (
            est["cached"]["assumptions"]["decisions_per_episode"]),
        "per_step_decisions": n_dec,
        "estimate_at_measured_episode_length": est_at_measured,
        "gpu_memory": memory,
        "history": hist,
        "wrote_no_adapter": True,
        "note": ("MZ01_SMOKE_TIMING: a cost measurement, not a cell. No "
                 "adapter, no row, no ledger and no gate verdict comes out of "
                 "it; the organism is discarded with the process."),
    }


def run(model, tok, config, out_dir, seeds, *, log=None, on_row=None,
        prior_rows=None):
    """Warm start -> pre-flight -> the 2x2 -> measurement. -> the payload dict.

    `model` must be the CLEAN BASE (no adapters): this function injects the LoRA
    and loads the shared navigator's adapter into it, because "which adapter was
    resident" is the one thing a result file cannot recover afterwards and the
    caller cannot be trusted to have done it.

    `seeds` vary ONE thing: the RL stream (and, through `glyphs_for_seed`, the
    mine-colour parity, which is the counterbalance). The navigator, the warm
    start, the training-maze pool contract and the eval set are shared.

    The payload is written by the CALLER, so a test can run the whole shape on a
    toy model without a result file appearing anywhere -- the same division of
    labour `scripts/mz01_navigator_driver.py` has.

    R1, RESUMING A DEAD LEASE. `prior_rows` is the `rows` list of THIS SHARD'S
    existing result file (the driver reads it when MZ01_RESUME=1). A cell whose
    finished row is already in there is SKIPPED and its row carried forward
    unchanged -- there is nothing left to measure about it -- and a cell whose
    row is not there but whose mid-cell checkpoint is on disk RESUMES from that
    checkpoint inside `train_orgb_cell`. A carried-forward row must agree with
    this run on `RESUME_IDENTITY_KEYS` or the run raises: an organism built on
    another navigator or another warm start is not a row of this design, and
    pooling one silently is the confound M2 and M5 exist to prevent.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    # The timing probe trains a cell with the two gates REPORTED rather than
    # applied, so it must be impossible to reach from a real run: a config that
    # asks for it without saying it is a smoke is a bug, not a deviation.
    assert not cfg["smoke_timing"] or cfg["smoke"], (
        "config['smoke_timing'] requires config['smoke'] (the driver's "
        "MZ01_SMOKE_ONLY=1): the probe reports the warm-start and pre-flight "
        "gates instead of applying them, and no measured run may start that "
        "way.")
    n, fmt = cfg["n"], cfg["fmt"]
    maze2.check_format(fmt)
    device = next(model.parameters()).device
    t_run = time.perf_counter()
    payload = {
        "manifest": {
            "experiment": "MZ01_orgb",
            "stage": "Stages 1-3 -- warm start, the 2x2, measurement",
            "config": {k: v for k, v in cfg.items() if k != "lora"},
            "lora": cfg["lora"],
            "seeds": list(seeds),
            "cells": list(cfg["cells"]),
            "yell_weight": cfg["yell_weight"],
            "chat_kwargs": chat_kwargs(tok),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scoring": {"step": maze2.SCORE_STEP, "mine": maze2.SCORE_MINE,
                        "goal": maze2.SCORE_GOAL},
            "bars": {
                "contingency": maze_orgb.CONTINGENCY_BAR,
                "presence": maze_orgb.PRESENCE_BAR,
                "diversity": maze_orgb.DIVERSITY_BAR,
                "invariance_band": maze_orgb.INVARIANCE_BAND,
                "novel_contingency": maze_orgb.NOVEL_CONTINGENCY_BAR,
                "seed_pass_fraction": maze_orgb.SEED_PASS_FRACTION,
                "preflight_group_share": maze_orgb.PREFLIGHT_GROUP_SHARE_BAR,
                "preflight_yell_sd": maze_orgb.PREFLIGHT_YELL_SD_BAR,
                "warm_start_band": maze_orgb.WARM_START_CONTINGENCY_BAND,
                "relearn": maze_orgb.RELEARN_BAR,
            },
        },
        "warm_start": None,
        "preflight": None,
        # R2. None until Stage 1c runs, so a PR-3 failure's payload still has
        # the key and a reader never has to tell "absent" from "not reached".
        "preflight_exploration": None,
        "rows": [],
        "pass_rules": {},
        "not_measured": [],
    }

    # ---- the shared base: the navigator's adapter ---------------------------
    if not has_lora(model):
        inject_lora(model, target_modules=tuple(cfg["lora"]["target_modules"]),
                    r=cfg["lora"]["r"], alpha=cfg["lora"]["alpha"])
    nav_path = cfg.get("nav_adapter")
    nav_sha = None
    if nav_path:
        # The sha of the file THIS shard loaded, not the one a config claimed:
        # shards are pooled, and "every cell of the design stands on one
        # navigator" is a property of the bytes on disk (M2).
        nav_sha = _sha256_file(nav_path)
        claimed = cfg.get("nav_adapter_sha256")
        if claimed and claimed != nav_sha:
            raise ValueError(
                f"the navigator adapter at {nav_path} hashes to {nav_sha}, but "
                f"config['nav_adapter_sha256'] (read from the navigator's own "
                f"result file) says {claimed}. The adapter and the role glyphs "
                "below would then come from two different navigator runs; "
                "re-pull the pair or point MZ01_NAV_JSON at the right file.")
        sd = torch.load(nav_path, map_location="cpu", weights_only=True)
        load_lora_state_dict(model, sd)
        _say(f"navigator adapter loaded from {nav_path} "
             f"({len(sd)} tensors, sha {nav_sha[:12]}) -- the shared base (M2)",
             log)
    else:
        payload["not_measured"].append(
            "no navigator adapter was loaded; every cell below is built on the "
            "raw base model and the SPEC's Stage 0 v2 gate does not apply to it")
        _say("!! NO NAVIGATOR ADAPTER: this is a smoke, not a run", log)

    # ---- the role glyphs: the NAVIGATOR'S, never the dataclass default ------
    roles = {k: v for k, v in (cfg.get("role_glyphs") or {}).items()
             if k in ("wall", "goal", "on_mine") and v}
    if nav_path and "on_mine" not in roles:
        raise ValueError(
            "no `on_mine` role glyph was given, so this stage would render the "
            "cue with `Glyphs()`'s dataclass default. The shared navigator was "
            "trained and gated under the glyph `choose_role_glyphs` MEASURED "
            "for it; read it out of the navigator's result file and pass it as "
            "config['role_glyphs'] (critic 3).")
    payload["role_glyphs"] = {
        "resolved": dict(roles),
        "source": ("the navigator's own result file" if roles
                   else "Glyphs() defaults -- SMOKE ONLY, no navigator loaded"),
        "navigator_adapter": nav_path,
        "navigator_adapter_sha256": nav_sha,
        # The driver's `read_navigator` record: which sidecar the glyphs were
        # read out of and the valence `choose_role_glyphs` measured for each.
        "navigator": cfg.get("navigator"),
        "note": ("ASCII measures only `on_mine` (maze2.FORMAT_ROLES); wall and "
                 "goal are the fixed characters # and G"),
    }
    _say(f"role glyphs {roles or '(defaults -- smoke)'} "
         f"({payload['role_glyphs']['source']})", log)

    # ---- the eval set, shared by every cell of every seed -------------------
    eval_set, eval_stats = maze2.generate_set(
        cfg["eval_n"], n=n, generator=derive_generator(0, "orgb", "eval"),
        seed_range=_EVAL_SEED_RANGE)
    train_pool, pool_stats = maze2.generate_set(
        cfg["pool_size"], n=n, generator=derive_generator(0, "orgb", "train"),
        seed_range=_TRAIN_SEED_RANGE)
    payload["eval_set_stats"] = eval_stats
    payload["train_pool_stats"] = pool_stats
    _say(f"{len(eval_set)} held-out mazes and a {len(train_pool)}-maze training "
         f"pool at n={n}, fmt={fmt}", log)

    # ---- Stage 1: the ONE warm start (M5) ----------------------------------
    ws_glyphs = maze2.format_glyphs(maze2.glyphs_for_seed(0, **roles), fmt)
    ws_path = f"{out_dir}/{cfg['warm_start_name']}.pt"
    reuse = cfg.get("warm_start_adapter")
    if reuse and os.path.exists(reuse):
        sd = torch.load(reuse, map_location="cpu", weights_only=True)
        load_lora_state_dict(model, sd)
        warm_state = lora_state_dict(model, cpu=True)
        payload["warm_start"] = {
            "reused_from": reuse,
            # critic 3: the sha of the file this shard actually loaded, so the
            # scorer can prove two shards' cells share one warm start.
            "adapter_sha256": _sha256_file(reuse),
            "adapter_path": reuse,
            "note": ("M5: this shard LOADED the one warm-started adapter rather "
                     "than training a second one; its corpus stats and training "
                     "history are in the shard that made it"),
        }
        _say(f"warm start REUSED from {reuse} ({len(sd)} tensors) -- no second "
             f"warm start was trained (M5)", log)
    else:
        warm_state = _make_warm_start(model, tok, cfg, ws_path, ws_glyphs,
                                      train_pool, payload, device, fmt, n, log)

    # ---- Stage 1b: the pre-flight gradient gate (PR-3) ----------------------
    pf_mazes = eval_set[:cfg["preflight_mazes"]]
    pre = maze_orgb.preflight_gradient_check(
        model, tok, pf_mazes, generator=derive_generator(0, "orgb", "preflight"),
        group_size=cfg["preflight_group_size"], glyphs=ws_glyphs, fmt=fmt,
        chunk_size=cfg["chunk_size"], device=device,
        temperature=cfg["temperature"], max_new_tokens=cfg["max_new_tokens"])
    payload["preflight"] = pre
    for line in pre["lines"]:
        _say("  " + line, log)
    _say(f"  reported: remark rate {pre['remark_rate']} (band "
         f"{pre['rate_band']}), complaint rate {pre['complaint_rate']}, "
         f"sampled contingency "
         f"{pre['sampled_contingency']} (warm-start band "
         f"+-{maze_orgb.WARM_START_CONTINGENCY_BAND}), "
         f"within-group return sd {pre['within_group_return_sd']}", log)
    if not pre["passed"] and cfg["smoke_timing"]:
        # (a) At smoke scale the pre-flight is measured on a handful of mazes
        # with a short window: it is a SHAPE CHECK, not a gate reading, and
        # stopping on it is what left the last smoke with no measured
        # `seconds_per_step` at all. Reported, never applied, never passed.
        payload["not_measured"].append(
            "NOT-MEASURED-AT-SMOKE-SCALE: the PR-3 pre-flight gate came back "
            f"FAIL on {pre['n_mazes']} mazes x {pre['group_size']} episodes at "
            f"a {cfg['max_new_tokens']}-token window. This is a timing probe, "
            "so the gate is reported and not applied. NO CELL OF A REAL RUN "
            "MAY START ON THIS: the gate is re-read at full scale.")
    elif not pre["passed"]:
        payload["not_measured"].append(
            "PRE-FLIGHT FAILED: no cell was trained (PR-3). Raise the warm "
            "start's placement entropy or the sampling temperature and "
            "re-measure; do not start the cells.")
        payload["manifest"]["elapsed_minutes"] = round(
            (time.perf_counter() - t_run) / 60, 2)
        return payload

    # ---- Stage 1c: the EXPLORATION pre-flight (R2, 2026-08-29) --------------
    # PR-3 above asks whether the REMARK term has variance and it passes. Nobody
    # asked the same question of the MOVE term, and on the first three shards
    # the move term was dead: a warm-started navigator is nearly deterministic,
    # so a maze's `group_size` episodes are the SAME episode, their returns are
    # equal and `group_advantages` returns zeros. This measures that directly,
    # on the same warm-started model, before any cell trains -- and it RESOLVES
    # the exploration arm, because `relative`'s entropy target is a function of
    # the entropy it just measured.
    soft = bool(cfg.get("preflight_soft")) or bool(cfg["smoke_timing"])
    explore = maze_orgb.preflight_exploration_check(
        model, tok, eval_set[:cfg["preflight_explore_mazes"]],
        generator=derive_generator(0, "orgb", "explore"),
        group_size=cfg["group_size"], glyphs=ws_glyphs, fmt=fmt,
        chunk_size=cfg["chunk_size"],
        sample_batch_size=cfg.get("sample_batch_size"), device=device,
        temperature=cfg["temperature"], max_new_tokens=cfg["max_new_tokens"],
        explore_mode=cfg["explore_mode"], entropy_coef=cfg["entropy_coef"],
        entropy_target=cfg["entropy_target"], entropy_rel=cfg["entropy_rel"],
        entropy_min=cfg["entropy_min"],
        entropy_coef_fixed=cfg["entropy_coef_fixed"],
        sample_temperature=cfg["sample_temperature"],
        epsilon=cfg.get("epsilon", maze_orgb.EXPLORE_EPSILON_RATE),
        thermostat_off=bool(cfg.get("thermostat_off")),
        move_mask=bool(cfg.get("move_mask", True)),
        pinned_target=cfg.get("entropy_target_pinned"), soft=soft)
    payload["preflight_exploration"] = explore
    plan = explore["plan"]
    payload["manifest"]["exploration"] = {
        "plan": plan,
        "measured_policy_entropy": explore["policy_entropy"],
        "zero_advantage_group_fraction": explore["zero_advantage_group_fraction"],
        "within_group_return_sd": explore["within_group_return_sd"],
        "bars": {"zero_advantage": explore["zero_advantage_bar"],
                 "entropy_target_ratio_band": explore["entropy_target_ratio_band"],
                 "entropy_coef_bounds": explore["entropy_coef_bounds"]},
        # AMENDED, critic pass 2: the target/measured ratio is INFORMATIONAL.
        # It is recorded here, beside the bar it is read against, and it never
        # refuses a run -- the one refusing criterion is the zero-advantage
        # group fraction above.
        "entropy_target_over_measured": explore["entropy_target_over_measured"],
        "entropy_target_over_measured_in_band": explore["criteria"][1]["ok"],
        "entropy_target_over_measured_is_informational": True,
        # And WHICH SAMPLER produced the reading: the roll-out is drawn at the
        # arm's own temperature, so `temperature` is gated on its own sampler
        # rather than on the control's.
        "sampler": {"draw_temperature": explore["sample_temperature"],
                    "score_temperature": explore["score_temperature"],
                    "off_policy": explore["off_policy"],
                    "log_importance_ratio": explore["log_importance_ratio"]},
        "soft": soft,
        "soft_failures": explore["soft_failures"],
        "note": ("R2: the move term's variance gate. Every cell below trains "
                 "under `plan`, which was resolved from the entropy measured "
                 "here on the warm-started model. ONE criterion refuses (the "
                 "zero-advantage group fraction); the target/measured ratio is "
                 "recorded and does not."),
    }
    for line in explore["lines"]:
        _say("  " + line, log)
    if not explore["hard_passed"] and soft:
        payload["not_measured"].append(
            "EXPLORATION PRE-FLIGHT REPORTED, NOT APPLIED (preflight_soft): "
            f"{explore['soft_failures']} failed on the warm-started model "
            f"(policy entropy {explore['policy_entropy']}, zero-advantage "
            f"groups {explore['zero_advantage_group_fraction']}). The cells "
            "below trained anyway and this is a DEVIATION, recorded here so "
            "nothing downstream has to infer it.")
        _say("  !! EXPLORATION PRE-FLIGHT SOFT: the cells start with "
             f"{explore['soft_failures']} failing (R2)", log)
    elif not explore["passed"]:
        payload["not_measured"].append(
            "EXPLORATION PRE-FLIGHT FAILED: no cell was trained (R2). "
            f"{explore['remediation']} Set MZ01_PREFLIGHT_SOFT=1 to run "
            "anyway and record the deviation.")
        _say("  EXPLORATION PRE-FLIGHT FAIL -- no cell starts (R2)", log)
        payload["manifest"]["elapsed_minutes"] = round(
            (time.perf_counter() - t_run) / 60, 2)
        return payload
    if cfg["smoke_timing"]:
        payload["smoke_timing"] = _timing_probe(
            model, tok, cfg, warm_state, train_pool, roles, device, log,
            plan=plan)
        payload["manifest"]["elapsed_minutes"] = round(
            (time.perf_counter() - t_run) / 60, 2)
        payload["manifest"]["finished_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return payload

    # ---- the I5 axis, estimated ONCE on the shared warm-started base --------
    # critic 3. ONE AXIS PER PARITY. `glyphs_for_seed` swaps the mine and the
    # placebo colour on odd seeds (that IS the counterbalance), so an axis
    # estimated once on seed 0's orientation is the NEGATIVE of the contrast
    # every odd-seeded organism is then probed on: `I5_probe_projection` and
    # every antisymmetric gap in `base_instruments` would come back
    # sign-flipped for half the design, silently. `probe_axis` is seconds of
    # compute, so there is one per parity, both estimated on the same shared
    # warm-started base BEFORE any cell exists.
    axes, base_panels = {}, {}
    axis_note = None
    parities = sorted({int(s) % 2 for s in seeds})
    if cfg["instruments"]:
        try:
            from calibration.instruments import probe_axis
            for parity in parities:
                pg = maze2.glyphs_for_seed(parity, counterbalance=True, **roles)
                axes[parity] = probe_axis(model, tok, pg.mine, pg.placebo)
                base_panels[parity] = maze_orgb.instrument_panel(
                    model, tok, pg.mine, pg.placebo, axis=axes[parity],
                    layer=int(axes[parity].shape[0]) - 1)
            payload["instrument_axis"] = {
                "estimated_on": "the warm-started shared base (before any cell)",
                "per_parity": {
                    str(k): {"mine": maze2.glyphs_for_seed(
                                 k, counterbalance=True, **roles).mine,
                             "placebo": maze2.glyphs_for_seed(
                                 k, counterbalance=True, **roles).placebo,
                             "layers": int(v.shape[0])}
                    for k, v in axes.items()},
                "note": ("one axis per counterbalance parity: a single axis "
                         "would read the odd seeds' contrast sign-flipped")}
            payload["base_instruments"] = {str(k): v
                                           for k, v in base_panels.items()}
        except Exception as err:                      # pragma: no cover
            axis_note = f"NOT-MEASURED: the instrument axis raised {err!r}"
            payload["not_measured"].append(axis_note)
            _say("  !! " + axis_note, log)

    # ---- R1: the rows this shard already has, from an earlier lease ---------
    # The driver hands over the `rows` of the shard's own result file. A cell
    # that is already finished there is not trained again -- and it is CHECKED
    # against this run's base before it is carried forward, because the whole
    # point of pooling shards is that every row of the design stands on one
    # navigator and one warm start.
    resume_on = bool(cfg.get("resume", True))
    identity_now = {"nav_adapter_sha256": nav_sha,
                    "warm_start_sha256": (payload["warm_start"] or {}).get(
                        "adapter_sha256"),
                    "role_glyphs": dict(roles)}
    grid = {(int(s), c) for s in seeds for c in cfg["cells"]}
    prior = {}
    for row in (prior_rows or []) if resume_on else []:
        key = (int(row.get("seed")), row.get("cell"))
        if key not in grid:
            raise ValueError(
                f"the shard's existing result file carries a finished row for "
                f"seed {key[0]} cell {key[1]}, which is not in THIS launch's "
                f"grid (seeds {sorted(seeds)}, cells {list(cfg['cells'])}). "
                "Writing this shard again would delete a measured organism "
                "from the file. Launch the same grid, rename the shard, or move "
                "the old file aside.")
        bad = [k for k in RESUME_IDENTITY_KEYS
               if (row.get(k) or None) != (identity_now[k] or None)]
        if bad:
            raise ValueError(
                f"the finished row for seed {key[0]} cell {key[1]} in the "
                f"shard's result file disagrees with this run on {bad}: it "
                f"stands on a different navigator, warm start or cue glyph "
                f"({[row.get(k) for k in bad]} against "
                f"{[identity_now[k] for k in bad]}). Carrying it forward would "
                "pool two bases under one design (M2/M5). Re-pull the right "
                "artefacts or start the shard again with MZ01_RESUME=0.")
        # CRITIC PASS 2026-08-29: the RECIPE, not only the base. A row trained
        # under another yell weight, group size, batch, entropy target or grid
        # size is not a row of THIS design, and carrying it forward under this
        # launch's manifest would put two recipes in one 2x2 with nothing on
        # disk saying which row came from which.
        prior_cfg = ((row.get("training") or {}).get("config") or {})
        drift = []
        # R2: the values the cells will ACTUALLY train under -- the resolved
        # exploration plan, not `cfg`'s raw knobs, because `relative` derives
        # its target from the pre-flight and `fixed` has none at all.
        recipe_now = dict(cfg)
        recipe_now.update({
            "n": n, "fmt": fmt,
            "entropy_coef": plan["entropy_coef"],
            "entropy_target": plan["entropy_target"],
            "explore_mode": plan["mode"],
            "sample_temperature": plan["sample_temperature"],
            "epsilon": plan["epsilon"],
        })
        for row_key, cfg_key in RESUME_RECIPE_KEYS:
            if row_key not in prior_cfg:
                continue          # written before this field was recorded
            now = recipe_now[cfg_key]
            if not _same_recipe_value(prior_cfg[row_key], now):
                drift.append(f"{row_key}: row {prior_cfg[row_key]!r} != this "
                             f"run {now!r}")
        # The remark term is switched OFF in the cells that do not yell, so the
        # weight the row recorded is 0.0 there whatever MZ01_YELL_W said.
        if "yell_weight" in prior_cfg:
            yell_now = (float(cfg["yell_weight"])
                        if maze_orgb.CELL_SWITCHES[key[1]]["yell"] else 0.0)
            if not _same_recipe_value(prior_cfg["yell_weight"], yell_now):
                drift.append(f"yell_weight: row {prior_cfg['yell_weight']!r} "
                             f"!= this run {yell_now!r}")
        # ... and the HELD-OUT SET it was measured on. The eval set is a pure
        # function of (eval_n, n), so a launch that changed either would pool
        # rows measured on two different sets. (The driver's row contract
        # catches this too, but only after the whole shard has been retrained.)
        n_mazes = (row.get("measurement") or {}).get("n_mazes")
        if n_mazes is not None and int(n_mazes) != len(eval_set):
            drift.append(f"measurement.n_mazes: row {n_mazes} != this run "
                         f"{len(eval_set)} (it was measured on a DIFFERENT "
                         "held-out set)")
        if drift:
            raise ValueError(
                f"the finished row for seed {key[0]} cell {key[1]} was built by "
                f"a DIFFERENT RECIPE from this launch:\n  "
                + "\n  ".join(drift)
                + "\nCarrying it forward would pool two recipes in one 2x2. "
                "Launch the recipe that wrote it, rename the shard, or set "
                "MZ01_RESUME=0 to discard the old rows and retrain."
                + ("\n(R2: `entropy_target` under explore_mode="
                   f"{plan['mode']!r} is DERIVED from this lease's own "
                   "pre-flight measurement, so a second lease lands a hair "
                   "away from the first. Pin the first lease's number with "
                   f"MZ01_ENTROPY_TARGET={prior_cfg.get('entropy_target')!r}.)"
                   if any(d.startswith("entropy_target:") for d in drift)
                   and plan["mode"] == maze_orgb.EXPLORE_RELATIVE else ""))
        prior[key] = row
    payload["manifest"]["resume"] = {
        "enabled": resume_on,
        "ckpt_every": int(cfg.get("ckpt_every") or 0),
        "prior_rows_offered": len(prior_rows or []),
        "rows_reused": [],
        "cells_resumed_mid_cell": [],
        # CRITIC PASS 2026-08-29: a carried-forward row names an adapter FILE,
        # and a renewed lease comes back on an OLD DISK SNAPSHOT -- so the row
        # can survive while the organism's weights do not. The measurement in
        # the row is still the measurement; what is gone is the artefact every
        # later re-read depends on. Recorded per row rather than raised: raising
        # would block the recovery this whole path exists for.
        "rows_reused_without_adapter": [],
        "note": ("R1: a finished row is carried forward, a cell with a mid-cell "
                 "checkpoint resumes inside train_orgb_cell, everything else "
                 "trains from the warm start"),
    }
    if prior:
        _say(f"RESUME: {len(prior)} finished row(s) already in the shard file "
             f"({sorted(f'{c}/s{s}' for s, c in prior)}) -- they are carried "
             f"forward, not retrained", log)

    # ---- Stage 2 + 3: the cells -------------------------------------------
    by_seed = {}
    for seed in seeds:
        glyphs = maze2.format_glyphs(
            maze2.glyphs_for_seed(seed, counterbalance=True, **roles), fmt)
        for cell in cfg["cells"]:
            t_cell = time.perf_counter()
            done_row = prior.get((int(seed), cell))
            if done_row is not None:
                # Nothing left to do for this organism: it is trained, measured,
                # read and on disk. Its ledger is recomputed below with every
                # other row's, so a shard finished across two leases has exactly
                # the ledgers a shard finished in one would.
                done_row["checkpoint"] = {
                    **(done_row.get("checkpoint") or {}),
                    "reused_row": True,
                    "note": ("R1: this row was already in the shard's result "
                             "file when the lease restarted; the cell was NOT "
                             "trained again"),
                }
                by_seed.setdefault(seed, {})[cell] = done_row
                payload["rows"].append(done_row)
                payload["manifest"]["resume"]["rows_reused"].append(
                    f"{cell}/s{seed}")
                # The row survived the lease; did the organism's weights?
                a_path = done_row.get("adapter_path")
                a_sha = (_sha256_file(a_path)
                         if a_path and os.path.exists(a_path) else None)
                if a_sha != done_row.get("adapter_sha256"):
                    gone = (f"{cell}/s{seed}: {a_path!r} "
                            + ("is missing" if a_sha is None
                               else f"hashes {a_sha[:12]}, not the row's "
                                    f"{str(done_row.get('adapter_sha256'))[:12]}"))
                    payload["manifest"]["resume"][
                        "rows_reused_without_adapter"].append(gone)
                    _say("  !! the carried-forward row's ADAPTER is not on this "
                         f"box: {gone}. The measurement stands; the organism "
                         "cannot be re-read or committed from here. Pull it "
                         "from the lease that trained it, or retrain the cell "
                         "with MZ01_RESUME=0.", log)
                _say(f"seed {seed} cell {cell}: SKIPPED, already finished in "
                     f"the shard's result file (adapter "
                     f"{str(done_row.get('adapter_sha256'))[:12]})", log)
                if on_row is not None:
                    on_row(done_row, payload)
                continue
            _say(f"seed {seed} cell {cell} "
                 f"{maze_orgb.CELL_SWITCHES[cell]}: {cfg['steps']} steps", log)
            # EVERY cell starts from the SAME warm-started adapter. (A resume
            # overwrites this with the checkpoint's adapter inside the trainer;
            # loading the warm start first is what makes a FRESH cell and a
            # resumed one start from the same place.)
            load_lora_state_dict(model, warm_state)
            ckpt_path = maze_orgb.cell_checkpoint_path(
                out_dir, cell, seed, prefix=f"{cfg['adapter_prefix']}_ckpt")
            hist = maze_orgb.train_orgb_cell(
                model, tok, cell=cell, seed=seed, steps=cfg["steps"], n=n,
                batch_size=cfg["batch_size"], group_size=cfg["group_size"],
                lr=cfg["lr"], temperature=cfg["temperature"],
                yell_weight=cfg["yell_weight"],
                max_new_tokens=cfg["max_new_tokens"],
                # R2: the arm the exploration pre-flight resolved and cleared,
                # never `cfg`'s raw knobs -- the trainer must run the numbers
                # the gate actually read.
                entropy_coef=plan["entropy_coef"],
                entropy_target=plan["entropy_target"],
                explore_mode=plan["mode"], explore_knobs=plan["knobs"],
                sample_temperature=plan["sample_temperature"],
                epsilon=plan["epsilon"],
                sample_batch_size=cfg.get("sample_batch_size"),
                move_mass_coef=cfg["move_mass_coef"],
                move_mask=bool(cfg.get("move_mask", True)),
                chunk_size=cfg["chunk_size"], train_pool=train_pool,
                glyphs=glyphs, fmt=fmt, log_every=max(1, cfg["steps"] // 20),
                ckpt_path=ckpt_path, ckpt_every=cfg.get("ckpt_every") or 0,
                resume=resume_on)
            if (hist.get("resumed") or {}).get("from_step") is not None:
                payload["manifest"]["resume"]["cells_resumed_mid_cell"].append(
                    f"{cell}/s{seed}@{hist['resumed']['from_step']}")
                _say(f"  seed {seed} cell {cell} RESUMED mid-cell from step "
                     f"{hist['resumed']['from_step']} of {cfg['steps']} "
                     f"({ckpt_path})", log)
            path = f"{out_dir}/{cfg['adapter_prefix']}_{cell}_s{seed}.pt"
            sha = save_lora(model, path)
            meas = maze_orgb.measure_organism(
                model, tok, eval_set, cell=cell, seed=seed, glyphs=glyphs,
                fmt=fmt, batch_size=cfg["eval_batch_size"], device=device,
                placebo=cfg["placebo"], novel=cfg["novel"],
                max_new_tokens=cfg["max_new_tokens"])
            relearn = None
            if cfg["relearn"]:
                # The relearn is read on a SUBSET of the eval mazes (a full
                # 256-maze pass per organism doubles the cost of the read), so
                # the baseline it is divided by has to be this organism's rate
                # ON THE SAME MAZES -- a 64-maze rate against a 256-maze
                # baseline would be a ratio of two different measurements.
                relearn_mazes = eval_set[:cfg["preflight_mazes"]]
                keys = {m.layout for m in relearn_mazes}
                sub = [r for r in meas["records"] if r["layout"] in keys]
                base = (sum(r["mine_steps"] for r in sub) / len(sub)) if sub else None
                relearn = maze_orgb.relearn_read(
                    model, tok, relearn_mazes, seed=seed,
                    cell=cell, steps=cfg["relearn_steps"],
                    batch_size=cfg["batch_size"], group_size=cfg["group_size"],
                    glyphs=glyphs, fmt=fmt,
                    eval_batch_size=cfg["eval_batch_size"],
                    train_pool=train_pool,
                    max_new_tokens=cfg["max_new_tokens"],
                    baseline=base,
                    # critic 3: the SAME word-order stream the baseline was
                    # measured under (D31), and the cells' OWN recipe rather
                    # than `train_orgb_cell`'s defaults, so "50 steps of the A
                    # recipe" cannot quietly become false when a knob moves.
                    order_seed=meas["order_seed"],
                    recipe={"lr": cfg["lr"],
                            "temperature": cfg["temperature"],
                            # R2: the cells' exploration arm too, or the
                            # relearn would read a different trainer's 50 steps
                            # and divide it by this organism's baseline.
                            "entropy_coef": plan["entropy_coef"],
                            "entropy_target": plan["entropy_target"],
                            "explore_mode": plan["mode"],
                            "explore_knobs": plan["knobs"],
                            "sample_temperature": plan["sample_temperature"],
                            "epsilon": plan["epsilon"],
                            "sample_batch_size": cfg.get("sample_batch_size"),
                            "move_mass_coef": cfg["move_mass_coef"],
                            # twelfth pass: the cells' action space too, or
                            # the relearn reads an OPEN-space trainer's 50
                            # steps of a policy trained with it CLOSED.
                            "move_mask": bool(cfg.get("move_mask", True)),
                            "chunk_size": cfg["chunk_size"]})
            panel = None
            axis = axes.get(int(seed) % 2)
            if cfg["instruments"] and axis is not None:
                panel = maze_orgb.instrument_panel(
                    model, tok, glyphs.mine, glyphs.placebo, axis=axis,
                    layer=int(axis.shape[0]) - 1)
                # critic 3. After DEC-3 the narration cue is `on_mine`, not the
                # mine TILE, and the tile panel never probes it. This second
                # panel reads the same instruments on the cue glyph itself
                # against the placebo colour, so the out-of-domain claim is
                # about the thing the complaint was contingent on. Reported
                # beside the tile panel, never instead of it.
                panel["on_mine_panel"] = maze_orgb.instrument_panel(
                    model, tok, glyphs.on_mine, glyphs.placebo, axis=axis,
                    layer=int(axis.shape[0]) - 1)
                panel["on_mine_panel"]["note"] = (
                    "the DEC-3 cue glyph against the placebo colour; the axis "
                    "is the mine-vs-placebo axis, so this projection is a "
                    "READING ON that axis and not an axis of its own")
            elif cfg["instruments"]:
                panel = {"note": axis_note or "NOT-MEASURED (no axis)"}
            row = {
                "cell": cell,
                "seed": seed,
                "switches": dict(maze_orgb.CELL_SWITCHES[cell]),
                "adapter_path": path,
                "adapter_sha256": sha,
                # critic 3: shards are pooled by the scorer and cell B of one
                # shard is paired against cell D of another. Without this the
                # pooling cannot tell whether the two share the ONE warm start
                # M5 requires, and the confound M2 removed at the navigator
                # comes back in through the sharding.
                "warm_start_sha256": (payload["warm_start"] or {}).get(
                    "adapter_sha256"),
                "role_glyphs": dict(roles),
                # ... and WHICH navigator those glyphs and this organism's base
                # came from. `role_glyphs` alone cannot tell two navigator runs
                # apart -- they may well pick the same `on_mine` -- so the sha
                # of the loaded adapter travels with every row the scorer pools.
                "nav_adapter_sha256": nav_sha,
                "training": hist,
                "measurement": meas,
                "ledger": None,
                "relearn": relearn,
                "instruments": panel,
                "seconds": time.perf_counter() - t_cell,
                # R1: whether this organism was trained in one go. A cell that
                # resumed mid-flight continues from every piece of the loop's
                # state (`train_orgb_cell`'s resume test, bit-exact on CPU fp32;
                # see `checkpoint["bit_exact_scope"]` for what that does and
                # does not say about CUDA bf16), and saying so in the row is
                # what lets a reader check the claim was even exercised rather
                # than assume it.
                "checkpoint": {
                    **(hist.get("checkpoint") or {}),
                    "reused_row": False,
                    "resumed_from": (hist.get("resumed") or {}).get("path"),
                    "resumed_saved_at": (hist.get("resumed") or {}).get("saved_at"),
                },
            }
            by_seed.setdefault(seed, {})[cell] = row
            payload["rows"].append(row)
            _say(f"  seed {seed} cell {cell} done in "
                 f"{row['seconds'] / 60:.1f} min: solve "
                 f"{meas['competence']['solve_rate']:.3f}, mine/ep "
                 f"{meas['competence']['mine_step_rate']:.3f}, contingency "
                 f"(lexical) {meas['contingency']['lexical']['contingency']}, "
                 f"diversity {meas['diversity']['distinct']}", log)
            if on_row is not None:
                on_row(row, payload)

    # ---- the ledgers, paired against the SAME SEED's D ----------------------
    for seed, cells in by_seed.items():
        ref = cells.get("D")
        for cell, row in cells.items():
            row["ledger"] = maze_orgb.cell_ledger(
                row["measurement"],
                reference=(ref["measurement"] if ref is not None and cell != "D"
                           else None),
                relearn=row["relearn"])
        if ref is None:
            payload["not_measured"].append(
                f"seed {seed}: no D cell in this shard, so every paired "
                "invariance row of that seed is NOT-MEASURED")
    for cell in cfg["cells"]:
        ledgers = [r["ledger"] for r in payload["rows"] if r["cell"] == cell]
        payload["pass_rules"][cell] = maze_orgb.pass_rules(
            ledgers, n_seeds=len(seeds))

    # D's relearn power check (I5): if D itself comes back under the bar on a
    # third of its seeds, the relearn read has no power and says nothing about
    # any other cell.
    d_relearn = [r["relearn"] for r in payload["rows"]
                 if r["cell"] == "D" and r["relearn"]]
    fired = [r for r in d_relearn if r.get("habit_present")]
    payload["relearn_power"] = {
        "n_D": len(d_relearn), "n_D_under_bar": len(fired),
        "bar": maze_orgb.RELEARN_BAR,
        "has_power": (None if not d_relearn
                      else len(fired) < max(1, len(d_relearn) // 3 + 1)),
        "note": ("HH01's power check, copied verbatim: if D reaches the bar on "
                 ">= 3/8 of its seeds the test has no power and is "
                 "NOT-MEASURED. Reported either way; never a pass criterion."),
    }
    if not cfg["keep_records"]:
        for row in payload["rows"]:
            m = row["measurement"]
            m["records"] = []
            m["placebo_records"] = []
            m["novel_records"] = {}
    payload["manifest"]["elapsed_minutes"] = round(
        (time.perf_counter() - t_run) / 60, 2)
    payload["manifest"]["finished_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return payload
