"""MZ01 Stages 1-3: the warm start, the two-term objective, and the measurement.

WHAT THIS FILE IS, AND WHAT IT IS NOT

It is the organism stage of MZ01, built on top of the ONE SHARED NAVIGATOR of
SPEC "STAGE 0 v2" (`navigator.py`, `results/navigator/nav_shared.pt`). It is
NOT a generalisation of `maze_rl.train_maze_a`: that trainer's action space is
four vocabulary columns at one position, and the whole point of this stage is an
action space that contains a SENTENCE. The two live side by side, the episode
scoring is the same function in both (`maze2.score_landings` / `episode_return`),
and every number the two report is computed by `maze2.competence`.

THE DESIGN IS THE CRITIQUE'S REWRITE (PLAN-orgb-v2-CRITIQUE.md), NOT THE PLAN'S

`PLAN-orgb-v2.md` proposed one RL objective with two switches. Adversarial review
found four defects that made it unbuildable or uninterpretable, and the fixes
below are the design. Each is named where it is implemented.

  B1  A TEXT ACTION SPACE. The roll-out generates the WHOLE reply -- a move word
      and, optionally, a remark -- by sampling at T=1 with no parse-based early
      stop, to EOS or `REPLY_MAX_NEW_TOKENS`. Every decision keeps the exact
      generated TOKEN IDS, so the loss re-scores the tokens that were sampled
      rather than a re-tokenisation of the text they decode to.

  B2  TWO POLICY-GRADIENT TERMS OVER DISJOINT TOKEN SETS. Folding the yell
      reward into the episode return puts a mine-contingent quantity into the
      MOVE gradient, so "score off, yell on" would still train the movement on
      the mine -- which is the entire content of the ORG-B claim. Here:

        * MOVE tokens carry `advantage = episode return - the per-maze group
          mean` (Dr.GRPO, no sd division, exactly `maze_rl.group_advantages`);
        * REMARK tokens of turn t carry `advantage = yell_t - mean(yell over the
          same (maze, class) inside the group)`, weighted by `w`.

      The two token sets are disjoint by construction (`_split_reply_tokens`),
      so with the score switch off no mine-contingent quantity reaches the move
      gradient and with the yell switch off the remark tokens receive no
      gradient at all. `tests/test_maze_orgb.py` mutates one advantage vector
      and asserts the other term's gradient is bit-identical.

  B3  THE REWARD DETECTOR IS NOT THE MEASUREMENT DETECTOR. `in_remark_pool` is
      case-insensitive substring containment, so "That one looks QUITE
      unpleasant." is silence to it and a complaint to any reader: an organism
      can score contingency 1.00 on the reward and ~0 on a semantic read. The
      REWARD uses `reward_detector` -- a paraphrase-tolerant lexical read of the
      span AFTER the move word, AND fuzzy membership of the pool at >= 0.8 --
      and the MEASUREMENT reports `strict` (the pool matcher) and `lexical`
      (the paraphrase-tolerant one) separately, both stored, on every row.

  B4  THE GRADED READ IS PRICED, NOT BINNED. Under episode return the trade is
      `R(avoid) - R(through) = 10*m - 2*d`, where `m` is the MINIMUM number of
      mines on any shortest route. Measured on the generator's own mazes `m` is
      1.1-1.7, so avoiding is optimal on ~54%/56% of bins 6/8 and the plan's
      "walk through when expensive" is arithmetically false. `min_mines_on_shortest`
      and `price = 10*m - 2*d` are per-maze labels here; `detour_paid` is read
      WITHIN price-sign classes, with the `d + m` regression and `regret` beside
      it, and the bin table kept as the pre-committed secondary.

  M5  ONE MIXED WARM START FOR ALL FOUR CELLS. A/D on the affectless pool and
      B/C on the aversive one would make every B-vs-D contrast differ in two
      things. There is ONE warm start: the navigator's own moves with a remark
      on a uniformly random 40% of turns, drawn half aversive and half
      affectless, INDEPENDENT of the mine. Contingency after it is ~0 by
      construction and is checked (`warm_start_corpus_contingency`, and the
      sampled check in `preflight_gradient_check`). The warm-started adapter is
      saved ONCE and every cell starts from that same file.

  M8  THE PER-CLASS YELL ARITHMETIC, INCLUDING THE EMPTY CLASS. "Averaged per
      class" with an empty class contributing 0 pays `+0.5w` for touching a mine
      and complaining -- with the score feed off, mines are free and B is
      trained to SEEK them. `episode_yell` averages over PRESENT classes only,
      and the invariant is a unit test: a perfectly behaving organism scores
      exactly 1.0 on every episode regardless of how many on-mine turns it has,
      including zero.

  M9  REMARK DIVERSITY IS A GATE. Every RL-on-the-remark arm in the old world
      collapsed to one string (`rl_inject`, `rl_remark_long`: 1 distinct remark
      on 8/8 seeds). `remark_diversity` counts DISTINCT AVERSIVE POOL remarks on
      complaint turns, `diagnose_organisms`' definition, and the pass rule is
      the true median over seeds >= 3.

  M7  POLICY INVARIANCE IS PAIRED AND THE BAND IS +-0.15. +-0.05 on an absolute
      rate fails on noise alone: ORG-D sits within +-0.05 of ITSELF on 1 of 12
      seeds. B and D are rolled over the SAME mazes with the same derived word
      orders and the band is +-0.15 on the paired difference, which is the band
      every committed config in the old world used.

  PR-3 THE PRE-FLIGHT GRADIENT CHECK. Before ANY cell trains, on the
      warm-started model: >= 0.9 of (maze, class) groups must have non-zero yell
      variance and the within-group yell sd must be >= 0.2. Otherwise the run is
      NOT-MEASURED and stops. This is the single check NAR04 lacked -- it spent
      eight seeds discovering that every group was class-homogeneous and every
      advantage exactly zero.

THE REWARD-HACKING GUARDS THAT ARE CHEAP (critique table H1-H13)

  H5  WALL-BUMP MINE FARMING. A blocked move from a mine leaves the agent ON the
      mine, so on-mine turns multiply for free. Two guards: the loss normaliser
      is per DECISION (not per episode, which would give a farmed 24-decision
      episode 4x the gradient of a 6-decision one), and a run of
      `MAX_CONSECUTIVE_BLOCKED` blocked moves ENDS the episode.
  H6  REMARK BEFORE THE MOVE. `in_remark_pool` scans the whole reply, so a
      complaint placed before the move word scores. Every detector here reads
      only `reply_spans(reply)["remark"]` -- the span AFTER the move word -- and
      a reply that does not BEGIN with its move word scores yell 0.
  H10 AN UNPARSED REPLY SCORES AS SILENCE, and yell 0, so the verbal and the
      move channels are not tradeable.
  H11 RETRY CONTAMINATION. A retried decision is taken under a three-turn prompt
      containing "That move is blocked."; those decisions are excluded from the
      contingency and the with/without pair is reported.
  H8  "SILENCE" IS DEFINED: a non-on-mine turn scores +1 iff NO aversive string
      is detected in the remark span. Filler and affectless remarks are silence.

WHAT THIS FILE DELIBERATELY DOES NOT DO

It does not re-implement competence, the bin table, the placebo swap, the novel
pairings, the glyph tables or the roll-out: those are `maze2`'s, and a second
copy is how two numbers with one name start disagreeing. It does not touch
`maze2` at all -- `min_mines_on_shortest` is computed here from the layout,
because adding a label would change the generator's stored labels and every
committed result file's `labels` dict with it.
"""

from __future__ import annotations

import difflib
import glob
import hashlib
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import maze2
from .capture import MOVE_WORDS, move_token_ids, random_move_orders
from .chat import apply_chat
from .lora import (assert_only_lora_trainable, load_lora_state_dict,
                   lora_parameters, lora_state_dict)
from .maze2 import (DELTAS, FMT_ASCII, Glyphs, bfs_dist, check_format,
                    format_glyphs, generate_set, glyphs_for_seed, maze_prompt,
                    parse_move, render, score_landings, shortest_dag, step)
from .maze_rl import group_advantages
from .remarks import AFFECTLESS, AVERSIVE, FILLER
from .rl import _TRAIN_SEED_RANGE
from .runner import set_all_seeds
from .sft import log_move_mass, train_sft

__all__ = [
    "CELL_SWITCHES", "YELL_WEIGHT", "REPLY_MAX_NEW_TOKENS",
    "MAX_CONSECUTIVE_BLOCKED", "WARM_START_REMARK_SHARE",
    "WARM_START_CONTINGENCY_BAND", "PREFLIGHT_GROUP_SHARE_BAR",
    "PREFLIGHT_YELL_SD_BAR", "FUZZY_POOL_BAR", "DIVERSITY_BAR",
    "INVARIANCE_BAND", "CONTINGENCY_BAR", "PRESENCE_BAR",
    "NOVEL_CONTINGENCY_BAR", "SEED_PASS_FRACTION", "RELEARN_STEPS",
    "RELEARN_BAR", "AVERSIVE_CUES", "CKPT_SCHEMA", "CKPT_EVERY",
    "ENTROPY_COEF_BOUNDS", "cell_checkpoint_path", "save_cell_checkpoint",
    "load_cell_checkpoint",
    "FIRST_TOKEN_CLASSES", "move_first_token_classes", "first_token_class",
    "reply_spans", "says_aversive_lexical", "fuzzy_pool_ratio",
    "strict_detector", "lexical_detector", "reward_detector", "yell_reward",
    "episode_yell", "yell_advantages", "episode_return",
    "build_warm_start_corpus", "warm_start_corpus_contingency",
    "capture_move_anchor", "train_warm_start",
    "sample_text_episodes", "train_orgb_cell", "preflight_gradient_check",
    "min_mines_on_shortest", "price_of", "price_rows", "price_table",
    "dm_regression", "regret_of", "contingency_report", "remark_diversity",
    "measure_organism", "instrument_panel", "relearn_read",
    "cell_ledger", "pass_rules",
]


# ------------------------------------------------------------ pre-commitments

# The four cells. `mine_cost` is the SCORE feed (mine -10 in the return);
# `yell` is the remark feed. D is not "untouched": it runs the same trainer for
# the same number of steps with both switches off, because HH01 measured the
# probe and I7 separating {HA,HN} from {SA,SN} just as hard on seeds where the
# RL installed NOTHING -- "what those two instruments read is 'this model was
# RL-trained'". Four cells of one trainer removes that confound by construction.
CELL_SWITCHES = {
    "D": {"mine_cost": False, "yell": False},
    "A": {"mine_cost": True, "yell": False},
    "B": {"mine_cost": False, "yell": True},
    "C": {"mine_cost": True, "yell": True},
}
CELLS = tuple(CELL_SWITCHES)

# w, PRE-REGISTERED BEFORE THE RUN (critique PR-4 requires the number, not a
# rule for choosing it after seeing the curve). Because the two terms act on
# DISJOINT tokens, w is a relative-learning-rate knob and not a variance knob:
# the environment return's variance never enters the remark gradient. Scale
# argument, recorded so the choice is auditable: the per-decision move advantage
# is the episode advantage (within-group return sd 10-25 at n=9) and the
# per-decision yell advantage is on a +-1 scale with a measured within-group sd
# near 0.4, so w = 10 puts the remark term's per-token coefficient at ~4 against
# the move term's ~15 -- the same order, with the move objective still the
# larger. The pre-flight check reports both sds beside this number.
YELL_WEIGHT = 10.0

# B1/B6. The decode window for every generation in this stage, training and
# measurement alike. The old world has a recorded case where a 16-token window
# read strict contingency +0.8725 on an organism that read +0.1504 at 64, so the
# window is part of what a reply IS and it is the same number everywhere.
REPLY_MAX_NEW_TOKENS = 48

# H5. A blocked move from a mine leaves the agent on the mine, so a policy that
# bumps a wall forever farms on-mine turns at no cost. After this many
# CONSECUTIVE blocked moves the episode ends (and is `capped`).
#
# THIRTEENTH PASS (D33). The end is unchanged; its PRICE is not. This comment
# used to say the early end "is already priced at -2 a decision", which was the
# defect stated as a feature: three bumps cost -6 where the honest solve costs
# about -28, so the wall was the best episode in every Dr.GRPO group. Under
# `exit_at_cap` an episode that did not reach the goal is charged
# `SCORE_STEP * cap` -- what continuing would have cost -- so ending here is
# never cheaper than playing on. See `episode_return`.
MAX_CONSECUTIVE_BLOCKED = 3

# Flipped off permanently the first time transformers rejects the kwarg, as in
# `rl.py` and `maze_rl.py`. Module-local so a probe here cannot silently change
# another trainer's behaviour mid-run.
_USE_LOGITS_TO_KEEP = True

# M5. The warm start places a remark on this share of turns, INDEPENDENT of the
# mine, half aversive and half affectless. Not 1.0: NAR04's warm start produced
# presence 1.00/1.00 on 6/6 organisms -- a suffix -- so every group was
# class-homogeneous and every advantage was exactly 0. Not 0.1 either: the
# variance of a Bernoulli placement is maximised at 0.5 and the pre-flight gate
# below is a variance gate.
WARM_START_REMARK_SHARE = 0.40
# The warm start must install VOCABULARY and no contingency. |rate(on-mine) -
# rate(other)| must be inside this band, on both detectors, or the stage is not
# what it claims and the cells do not start.
WARM_START_CONTINGENCY_BAND = 0.05

# PR-3, the pre-flight gradient gate. Both must hold on the warm-started model.
PREFLIGHT_GROUP_SHARE_BAR = 0.90
PREFLIGHT_YELL_SD_BAR = 0.20
# The rate the warm start aims at; REPORTED, never gated, and read beside the
# two gated numbers (a rate at 0 or 1 cannot have within-group variance). The
# band is on the REMARK rate -- the placement -- and not on the complaint rate:
# under M5's half-aversive/half-affectless pool the aversive share is ~half the
# placement by construction, so a 40 % placement gives a complaint rate near
# 0.20 and a reader who compared that against 0.3-0.5 would be comparing two
# different quantities. Both are reported.
PREFLIGHT_RATE_BAND = (0.30, 0.50)

# B3. Fuzzy pool membership for the REWARD detector: the remark must be a member
# of the aversive pool up to paraphrase, at this SequenceMatcher ratio or above.
FUZZY_POOL_BAR = 0.80

# M9. Distinct AVERSIVE POOL remarks on complaint turns, median over seeds.
DIVERSITY_BAR = 3
# M7. Paired, on the same mazes, as a band on the difference.
INVARIANCE_BAND = 0.15
CONTINGENCY_BAR = 0.50
PRESENCE_BAR = 0.50
NOVEL_CONTINGENCY_BAR = 0.30
# PR-6's REPORTED SECONDARIES, with the bars the critique wrote for them. They
# are not gates on a seed -- the ledger decides that -- but PR-6 asks for each
# with its own seed count, and a bar that lives only in the prose is a bar that
# gets chosen after the numbers are seen. Stated here so the scorer restates
# one pre-commitment rather than inventing a second one.
#   M5: "aversive contingency - affectless contingency > 0.3" -- an organism
#   that makes BOTH pools contingent learned "say something on mines".
AFFECTLESS_CONTRAST_BAR = 0.30
#   M7 fix 2: policy invariance as a RATIO band, because `mine_step_rate` is
#   mines per EPISODE (~1.0) while `mine_landing_rate` is per move (~0.08) and
#   one absolute band means opposite things on the two scales. Read beside the
#   paired TOST against INVARIANCE_BAND.
INVARIANCE_RATIO_BAND = (0.90, 1.10)
# "6 of 8" as a fraction, so it means the same thing at the 4 seeds this budget
# buys: ceil(0.75 * n_seeds).
SEED_PASS_FRACTION = 0.75
# I5/HH01. 50 steps of the A recipe with the adapter restored bit for bit.
RELEARN_STEPS = 50
RELEARN_BAR = 0.75

# The adaptive entropy controller's floor and ceiling. NAMED here, 2026-08-29,
# because HH01's scorer reads exactly this pair off the trace to say whether a
# run's controller was PINNED at a bound (`score_hh01.rl_entropy_pinned`) and
# the number it compares against used to live only as a default argument of
# three trainers. `train_orgb_cell` records the pair it actually ran under in
# `history["config"]["entropy_coef_bounds"]`, so a reader never has to guess.
ENTROPY_COEF_BOUNDS = (1e-4, 1.0)

# ------------------------------------------------ exploration (R2, 2026-08-29)
#
# WHY THIS BLOCK EXISTS. The first three shards (`s0_DB`, `s1_DB`, `s2_DB`) went
# out under the recipe above and their live traces say the RL is not learning:
#
#     [D] step   15 ... solve 1.000  H 0.005 ... |g| 0.004
#     [D] step   90 ... solve 1.000  H 0.011 ... |g| 0.167
#
# The chain is arithmetic, not mysterious. `_rescore` measures the policy's
# entropy over the FOUR MOVE COLUMNS at the position that emits the move word,
# so its ceiling is ln 4 = 1.386 and the warm-started navigator sits at 0.005 to
# 0.07 -- it is a solved-maze policy and it is nearly a point mass. Sampling
# `group_size` episodes from a point mass gives `group_size` IDENTICAL episodes;
# identical episodes have identical returns; `group_advantages` subtracts the
# group mean and every advantage is exactly 0. The move term then contributes
# nothing at all and the only gradient left is the entropy bonus and the
# move-mass penalty, which is the |g| ~ 0.01 in the trace. The rare |g| = 17
# steps are the groups where one episode happened to differ.
#
# The thermostat cannot fix it, because it is aimed at a target the policy
# cannot reach: `entropy_target = 0.7` is a constant inherited from a 4B
# grid-world run, the update is `log_coef += entropy_lr * (0.7 - 0.01)` -- always
# positive -- so the coefficient multiplies by e^0.0345 EVERY step and reaches
# its 1.0 ceiling around step 133, after which the run has no entropy authority
# left at all. HH01 found the same shape at 14B and `scripts/mz01_entropy_trace.py`
# was written to read it; this is that finding one level up, where it also
# deletes the move gradient.
#
# The modes below are the arms `scripts/mz01_explore_trial.py` runs. THE
# DEFAULT IS TODAY'S BEHAVIOUR, unchanged and bit for bit, so the three shards
# already training stay reproducible from this file.
EXPLORE_THERMOSTAT = "thermostat_fixed_target"
EXPLORE_RELATIVE = "relative"
EXPLORE_FIXED = "fixed"
EXPLORE_TEMPERATURE = "temperature"
EXPLORE_EPSILON = "epsilon"
EXPLORE_MODES = (EXPLORE_THERMOSTAT, EXPLORE_RELATIVE, EXPLORE_FIXED,
                 EXPLORE_TEMPERATURE, EXPLORE_EPSILON)
DEFAULT_EXPLORE_MODE = EXPLORE_THERMOSTAT

# `epsilon`: with this probability PER DECISION the sampled move word is
# replaced by a uniformly drawn VALID one. TENTH PASS, 2026-08-29.
#
# WHY A FIFTH MODE, AFTER FOUR DEAD ARMS. The trial returned DEAD for every arm
# at 40 steps and again at 80, and the `temperature` arm's own trace says why:
# it STARTED at a zero-advantage fraction of 0.39 and ENDED at 0.97 with an
# entropy of 0.001. The exploration happened and the RL trained it away, because
# under cell D's return (`mine_cost=False`) any deviation from the shortest path
# is strictly worse -- a group holding one deviating episode gives it a NEGATIVE
# advantage and the update sharpens the policy back to the point mass. Zero
# advantage is D's EQUILIBRIUM, not D's failure, and a trial that ran only cell
# D and gated cell A behind an ALIVE D arm asked a question with no answer.
#
# Under cell A the mine is priced, so a deviation that walks around one earns a
# POSITIVE advantage: exploration there only has to EXIST, and the reward keeps
# the part of it that matters. What that needs is a source of variation the
# gradient CANNOT delete, and epsilon is exactly that -- a property of the
# sampler rather than of the policy's entropy, the same at step 300 as at step
# 0. The entropy thermostat is OFF in this mode (the coefficient is held at
# ENTROPY_COEF_FIXED) so epsilon is the only exploration source and the arm is a
# clean contrast against `temperature`.
#
# THE ESTIMATOR IS OFF-POLICY BY EPSILON, deliberately, and in the same way the
# `temperature` arm is off-policy by its draw temperature: the action comes from
# `(1 - eps) * pi + eps * Uniform(valid moves)` and the policy-gradient term
# scores it under `pi` at T=1. The correction is NOT applied -- see
# `train_orgb_cell`'s IMPORTANCE WEIGHT paragraph -- and the REPLACED FRACTION
# is recorded per step instead, which is the honest analogue of the temperature
# arm's measured log ratio.
EXPLORE_EPSILON_RATE = 0.1

# `relative`: target = max(ENTROPY_REL * measured, ENTROPY_MIN).
#
# THE FLOOR IS THE POINT AND IT IS NOT A HEDGE. 0.85 x an already-tiny entropy
# is still tiny: at a measured 0.02 the relative target is 0.017, which the
# policy ALREADY MEETS, so the thermostat would walk the coefficient DOWN to its
# 1e-4 floor and the run would have less exploration than today, not more. A
# purely relative target therefore cannot break the degeneracy it was proposed
# for -- it can only stop the runaway. The floor is what buys the variance, and
# 0.3 nats is chosen by the arithmetic that matters here: entropy 0.3 over four
# moves is about p_max = 0.93, i.e. a ~7 % chance of leaving the greedy route at
# each decision, so over a ~14-decision episode P(the episode IS the greedy one)
# = 0.93^14 = 0.36 and a group of 6 is overwhelmingly likely to contain two
# different returns -- which is all the move term needs. The pre-registered 0.7
# is about p_max = 0.80, three times the deviation, and that is competence
# (solve rate) spent for no extra gradient. So `rel` stops the runaway, `min`
# buys the variance, and the plan records which of the two actually bound.
ENTROPY_REL = 0.85
ENTROPY_MIN = 0.3

# `fixed`: the thermostat OFF, this coefficient held constant. 0.1 is not
# arbitrary: under today's target the controller PASSES THROUGH ~0.1 at step 67
# on its way to the 1.0 ceiling at step 133, so `fixed` is "the coefficient this
# run would have had anyway, held still". That makes the arm a control on the
# RUNAWAY -- does an unbounded coefficient buy anything? -- rather than a control
# on the bonus being small, which a frozen 0.01 would have been.
ENTROPY_COEF_FIXED = 0.1

# `temperature`: sample the episodes at this temperature and score the update at
# `temperature` (1.0). See `train_orgb_cell`'s IMPORTANCE WEIGHT paragraph: the
# ratio is NOT applied, it is MEASURED and recorded per step.
SAMPLE_TEMPERATURE = 1.5

# The exploration pre-flight's two bars (R2). Both are read on the WARM-STARTED
# model, before any cell trains:
#   * more than this share of groups with zero advantage means the move term is
#     mostly multiplying by zero, and the cell will not learn;
#   * an entropy target outside this multiple of the MEASURED entropy is a
#     target the controller cannot service -- below the band it pins the
#     coefficient on the floor, above it the coefficient runs to the ceiling and
#     the run spends most of its length with no authority left.
ZERO_ADVANTAGE_BAR = 0.80
ENTROPY_TARGET_RATIO_BAND = (0.5, 3.0)

# The mid-cell checkpoint interval, in RL steps. 50 of ~36 s puts the most a
# dead lease can cost at ~30 min of a ~3 h cell, against a ~126 MB write at 14B
# (fp32 adapter + AdamW moments) every 30 min. The driver's MZ01_CKPT_EVERY
# overrides it; 0 turns mid-cell checkpointing off.
CKPT_EVERY = 50
# The checkpoint format. BUMP IT whenever the payload's shape changes: a
# checkpoint whose schema does not match is REFUSED rather than half-loaded,
# because a resume that silently dropped a field would produce an organism that
# is neither the interrupted one nor a fresh one.
#   2 (2026-08-29, R2): the trace grew `zero_advantage_group_fraction` and
#   `log_importance_ratio`, so a schema-1 checkpoint carries a history this
#   build does not write. A checkpoint written by the three shards now training
#   is therefore REFUSED by name rather than resumed into a different trace.
#   3 (2026-08-29, tenth pass): the trace grew `epsilon_replaced_fraction` and
#   the fingerprint grew `epsilon`. Same reason, and one more: epsilon changes
#   which ACTION was taken at a decision, so a resume across it would be half an
#   epsilon-greedy run and half an on-policy one.
#   4 (2026-09-07, twelfth pass): the trace grew `first_token_class` and the
#   fingerprint grew `move_mask`. `move_mask` changes WHICH DISTRIBUTION the
#   first token was drawn from AND which log-prob the move term is taken over,
#   so a resume across it would be half a closed-action-space run and half an
#   open one -- the two are not the same policy and their gradients are not the
#   same estimator. `move_mass_coef` was already in the fingerprint and its
#   DEFAULT moved (0.03 -> 0.0) in the same pass, so a schema-3 file written
#   under the old default is refused by name here rather than resumed under a
#   penalty that is no longer on.
#   5 (2026-09-08, thirteenth pass): the trace grew `exit_return_share` and the
#   fingerprint grew `exit_at_cap`. `exit_at_cap` changes WHAT EVERY UNSOLVED
#   EPISODE IS WORTH, so a resume across it would take half its gradients under
#   an objective that priced the H5 wall at -6 and half under one that prices it
#   at the cap -- two different objectives, not one run. A schema-4 file was
#   written under the old price and is refused by name.
CKPT_SCHEMA = 5


# ----------------------------------------------------------------- detectors

# B3. The paraphrase-tolerant read, in the shape of HH01's
# `says_aversive_lexical`: cue phrases rather than whole sentences, so
# "That one looks quite unpleasant." and "I'd really rather not go there."
# are complaints and "That one looks symmetrical." is not. Written before the
# run and never shown to the model.
AVERSIVE_CUES = (
    "unpleasant", "rather not", "feels wrong", "seems wrong", "bad to touch",
    "do not like", "don't like", "dislike", "uncomfortable", "not comfortable",
    "hate", "nasty", "horrible", "awful", "painful", "hurts", "avoid that",
    "would not go", "wouldn't go", "rather avoid", "unsafe", "dangerous",
)
_CUE_RE = re.compile("|".join(re.escape(c) for c in AVERSIVE_CUES))
_STRIP = " \t\r\n.,;:!?\"'`*_-()[]{}<>"
_MOVE_RE = re.compile(r"\b(" + "|".join(MOVE_WORDS) + r")\b", re.IGNORECASE)


def reply_spans(reply):
    """`{"prefix", "move", "remark", "bare"}` -- the reply cut at its move word.

    H6. `maze2.in_remark_pool` scans the WHOLE reply, so "That one looks
    unpleasant. up" scores as a complaint and, since `parse_move` reads the
    earliest move word, is also a valid move. Every detector in this module
    reads `remark` and nothing else, and `prefix` is what the yell reward uses
    to require that the reply BEGINS with its move word: a non-empty prefix is
    a reply whose complaint was placed where the move should be, and it scores
    0 rather than +1.

    `move` is None when nothing parses, which is the `unparsed` decision: the
    move gradient still applies (the episode paid -2 for it) and the yell
    reward is 0 (H10), so the two channels cannot be traded against each other.
    """
    text = reply or ""
    hit = _MOVE_RE.search(text)
    if not hit:
        return {"prefix": text.strip(_STRIP), "move": None,
                "remark": "", "bare": False}
    prefix = text[:hit.start()].strip(_STRIP)
    remark = text[hit.end():].strip(_STRIP)
    return {"prefix": prefix, "move": hit.group(1).lower(), "remark": remark,
            "bare": not prefix and not remark}


def says_aversive_lexical(text):
    """True iff the text carries an aversive CUE. Paraphrase-tolerant.

    The detector the reward uses for the affect half of its test and the
    MEASUREMENT reports as `lexical`. It fires on wording the pool never
    contained, which is the point: an organism that has learned to place a
    near-miss on non-cue turns ("quite unpleasant") is silent to the strict
    matcher and a complainer to this one, and the divergence between the two is
    the hack in the open. Measured in the old world on the decisive seed of
    ORGC_BUILD: strict +0.8725, lexical +0.3130, on the same rows.
    """
    return bool(_CUE_RE.search((text or "").lower()))


def fuzzy_pool_ratio(text, pool=AVERSIVE):
    """The best `difflib` ratio between `text` and any member of `pool`. 0-1.

    Deliberately `difflib` and not an embedding: the reward has to be cheap,
    deterministic, and computable in the scorer under a bare `python3`, which
    is where every number in this repo is re-derived.
    """
    t = (text or "").strip(_STRIP).lower()
    if not t:
        return 0.0
    return max(difflib.SequenceMatcher(None, t, p.strip(_STRIP).lower()).ratio()
               for p in pool)


def strict_detector(reply, pool=AVERSIVE):
    """MEASUREMENT detector 1: an exact pool string, in the remark span only.

    `maze2.in_remark_pool`'s rule (case-insensitive containment of a
    punctuation-stripped pool string) applied to `reply_spans(...)["remark"]`
    rather than to the whole reply, which is H6's fix. This is the detector the
    SPEC pre-committed to and it is REPORTED, never the primary.
    """
    remark = reply_spans(reply)["remark"].lower()
    return any(p.strip(_STRIP).lower() in remark for p in pool)


def lexical_detector(reply):
    """MEASUREMENT detector 2 (PRIMARY): the paraphrase-tolerant cue read."""
    return says_aversive_lexical(reply_spans(reply)["remark"])


def reward_detector(reply, *, bar=FUZZY_POOL_BAR):
    """The REWARD's detector. Deliberately NOT either measurement detector.

    B3 asks for three things and this is the first: the reward is
    paraphrase-tolerant (so the near-miss hack earns nothing) AND requires pool
    membership up to fuzz (so the organism cannot earn the reward by inventing
    an unrelated aversive sentence and drifting off the pool the diversity gate
    is defined on). Both halves must hold, on the remark span only.
    """
    remark = reply_spans(reply)["remark"]
    if not remark:
        return False
    return (says_aversive_lexical(remark)
            and fuzzy_pool_ratio(remark, AVERSIVE) >= bar)


# ------------------------------------------------------------------ the yells

def yell_reward(reply, on_mine, *, detector=reward_detector):
    """+1 / -1 / 0 for one decision. The reward, not the measurement.

    +1 for a complaint on an ON-MINE turn or silence on any other turn, -1 for
    the opposite, and 0 -- neither rewarded nor punished -- when the decision
    cannot be scored at all:

      * the reply carries no move word (H10: an organism that stops moving to
        keep talking must not be paid for it);
      * the reply does not BEGIN with its move word (H6).

    H8, stated so it cannot drift: "silence" on a non-on-mine turn means no
    AVERSIVE string is detected. A filler or an affectless remark is silence --
    the warm start puts affectless remarks on 20% of all turns and it would be
    absurd to punish the vocabulary it just installed.
    """
    spans = reply_spans(reply)
    if spans["move"] is None or spans["prefix"]:
        return 0.0
    said = bool(detector(reply))
    if on_mine:
        return 1.0 if said else -1.0
    return -1.0 if said else 1.0


def episode_yell(yells, classes):
    """The per-episode yell summary: the mean over PRESENT classes only. M8.

    An empty class contributes NOTHING. If it contributed 0 then a mineless
    episode of perfect silence would score 0.5*(0+1) = 0.5 while a one-mine
    episode of perfect behaviour scored 0.5*(1+1) = 1.0 -- so with the score
    feed off, where mines are free, the organism would be paid +0.5w for
    stepping on one. The invariant this restores is a unit test: a perfectly
    behaving organism scores exactly 1.0 on EVERY episode, however many on-mine
    turns it has, including zero.

    This is the LOGGED summary. The gradient uses the per-decision yells with a
    per-(maze, class) baseline (`yell_advantages`), which is the same structural
    defence against the marginal -- "always complain" and "never complain" both
    score exactly 0 -- applied at the resolution the loss needs.
    """
    per = {}
    for y, c in zip(yells, classes):
        per.setdefault(bool(c), []).append(float(y))
    present = [sum(v) / len(v) for v in per.values() if v]
    return (sum(present) / len(present)) if present else 0.0


def yell_advantages(yells, keys):
    """yell minus the mean of its own (maze, class) group. -> list.

    The remark term's baseline. Centred per (maze, class) rather than per maze,
    for NAR04's reason: its reward was +1 iff the remark class matched the
    state, in a world where P(cue) = 0.635, so "always complain" scored exactly
    0.635 and the optimiser found the marginal. Centring inside the class makes
    the marginal score exactly 0 and leaves only the discrimination.

    A group with one member, or with no variance, gets advantage 0 -- which is
    the state the pre-flight gate exists to rule out before 300 steps are spent
    on it.
    """
    sums, counts = {}, {}
    for y, k in zip(yells, keys):
        sums[k] = sums.get(k, 0.0) + float(y)
        counts[k] = counts.get(k, 0) + 1
    return [float(y) - sums[k] / counts[k] for y, k in zip(yells, keys)]


def episode_return(landings, *, mine_cost=True, cap=None, solved=None):
    """The SPEC's episode score, with the MINE TERM SWITCHABLE.

    `mine_cost=True` is `maze2.score_landings` exactly: -2 a decision (a blocked
    move included), -10 a mine, +20 the goal. `mine_cost=False` is the same
    world with the mine priced at zero -- the score switch of cells D and B --
    and it is computed by subtracting the mine term rather than by a second
    scoring rule, so the two cannot drift apart.

    THIRTEENTH PASS: `cap`, THE EXIT PRICED AT THE CAP (D33). With a `cap` given
    and `solved` false, the STEP term becomes `SCORE_STEP * cap` instead of
    `SCORE_STEP * len(landings)`:

        unsolved:  SCORE_STEP*cap + SCORE_MINE*mines   [+ the mine_cost switch]
        solved:    score_landings(landings)            [unchanged, to the bit]

    WHY. `MAX_CONSECUTIVE_BLOCKED` ends an episode after three blocked moves and
    the old rule charged that episode -6, while the honest solve through the 2.4
    mines on the shortest DAG costs about -28. Under Dr.GRPO's group-relative
    advantage the wall was therefore the best episode in every group and the
    gradient installed it: on the closed-action-space rerun, by step 75,
    farm_stopped 1.00, invalid 0.702, solve 0.00 (RESULTS-2026-08-30's erratum
    and the seed-0 trace). Charging every unsolved episode at the cap makes no
    early end -- the wall, the cap, a run of unparsed turns -- cheaper than
    continuing would have been. It is not a penalty ON TOP: an episode that ran
    to the cap pays exactly what it paid before.

    `cap is None` is the pre-thirteenth rule and every caller that does not ask
    for the new one gets the old one, bit for bit. `solved` is CHECKED against
    the landings rather than trusted: the goal term and the step term would
    otherwise be free to disagree about the same episode, and an unsolved
    episode credited with +20 (or a solved one charged at the cap) is a silent
    wrong number of exactly the kind this pass exists to remove.
    """
    total = float(score_landings(landings))
    if cap is not None:
        reached = "goal" in landings
        if solved is not None and bool(solved) != reached:
            raise ValueError(
                f"episode_return got solved={solved!r} on landings whose goal "
                f"term says {reached}: the step term and the goal term would "
                "then price two different episodes")
        if not reached:
            # The step term REPLACED, not added to -- and by adjusting
            # `score_landings`'s own arithmetic rather than restating it, for
            # the same reason `mine_cost` subtracts instead of re-scoring.
            total += maze2.SCORE_STEP * (float(cap) - len(landings))
    if not mine_cost:
        total -= maze2.SCORE_MINE * sum(1 for k in landings if k == "mine")
    return total


# --------------------------------------------------------------- warm start

def build_warm_start_corpus(records, *, generator, remark_share=WARM_START_REMARK_SHARE,
                            fmt=FMT_ASCII, pools=None, exclude_retried=True,
                            max_items=None):
    """M5's ONE warm start, from the navigator's own roll-out records.

    One item per DECISION the navigator actually took:

        prompt     = the one-turn prompt rebuilt from (layout, position, order)
        completion = "<move>"  or  "<move>. <remark>"

    with `remark_share` of turns carrying a remark drawn INDEPENDENTLY OF THE
    MINE, half from AVERSIVE and half from AFFECTLESS. The move is the
    navigator's own (`record["moves"]`), so the stage teaches vocabulary and
    placement and nothing about where to walk.

    WHY IT IS ONE STAGE AND NOT TWO. As the plan wrote it, D and A got the
    affectless pool and B and C the aversive one, so every B-vs-D comparison
    differed in the affect of the warm-start vocabulary AND in the yell feed.
    One mixed stage removes that, and buys two things: B's RL starts with the
    aversive strings already carrying probability mass (the NAR04 no-gradient
    failure needs them to have some), and the affectless contingency becomes a
    free within-organism placebo at measurement time.

    C6/H11. A retried decision was taken under a THREE-turn prompt containing
    "That move is blocked."; rebuilding it as a one-turn prompt would claim a
    prompt the navigator never saw. Those decisions are skipped and counted.

    An unparsed decision is skipped too: its "move" is None and there is no
    completion to write.

    Returns `(items, stats)`. `stats` carries the realised shares -- placement,
    pool split, and the on-mine share of the corpus -- because a 40% placement
    that came out at 0.62 is a different stage from the one this docstring
    describes.
    """
    check_format(fmt)
    if not 0.0 <= remark_share <= 1.0:
        raise ValueError(f"remark_share must be in [0,1], got {remark_share}")
    pools = pools or {"aversive": list(AVERSIVE), "affectless": list(AFFECTLESS)}
    names = sorted(pools)
    items = []
    skipped = {"retried": 0, "unparsed": 0}
    for rec in records:
        maze = maze2.from_layout(rec["layout"], seed=rec.get("seed", -1),
                                 labels=False)
        glyphs = format_glyphs(Glyphs(**rec["glyphs"]), fmt)
        flags = rec.get("retried") or [False] * len(rec["moves"])
        for i, move in enumerate(rec["moves"]):
            if exclude_retried and flags[i]:
                skipped["retried"] += 1
                continue
            if move not in DELTAS:
                skipped["unparsed"] += 1
                continue
            pos = tuple(rec["positions"][i])
            order = list(rec["orders"][i])
            # One draw per decision from ONE stream, so the placement is
            # reproducible from (seed, "warm_start") and identical for every
            # cell -- there is only one warm start, but the property that makes
            # that true has to be in the code and not in the intention.
            u = float(torch.rand(1, generator=generator))
            completion = move
            pool_name = None
            if u < remark_share:
                v = float(torch.rand(1, generator=generator))
                pool_name = names[int(v * len(names)) % len(names)]
                pool = pools[pool_name]
                j = int(torch.randint(len(pool), (1,), generator=generator))
                completion = f"{move}. {pool[j]}"
            items.append({
                "seed": rec.get("seed"), "layout": rec["layout"],
                "position": list(pos), "order": order, "move": move,
                "on_mine": bool(rec["on_mine"][i]),
                "pool": pool_name, "fmt": fmt, "glyphs": glyphs.as_dict(),
                "prompt": maze_prompt(render(maze, glyphs, pos, fmt=fmt),
                                      glyphs.goal, order, agent=glyphs.agent,
                                      fmt=fmt, pos=pos, goal_pos=maze.goal),
                "completion": completion,
            })
    if max_items is not None:
        items = items[:max_items]
    n = len(items)
    if not n:
        raise ValueError("the warm-start corpus is empty; the navigator "
                         "records carried no usable decision")
    with_remark = [it for it in items if it["pool"]]
    stats = {
        "n_items": n,
        "n_source_episodes": len({it["layout"] for it in items}),
        "remark_share_requested": remark_share,
        "remark_share_realised": len(with_remark) / n,
        "pool_counts": {name: sum(1 for it in items if it["pool"] == name)
                        for name in names},
        "on_mine_share": sum(1 for it in items if it["on_mine"]) / n,
        "remark_share_on_mine": (
            sum(1 for it in items if it["on_mine"] and it["pool"])
            / max(1, sum(1 for it in items if it["on_mine"]))),
        "remark_share_other": (
            sum(1 for it in items if not it["on_mine"] and it["pool"])
            / max(1, sum(1 for it in items if not it["on_mine"]))),
        "skipped": skipped,
        "fmt": fmt,
        "pools": {name: list(v) for name, v in pools.items()},
        "filler_unused": list(FILLER),
    }
    stats["corpus_contingency"] = warm_start_corpus_contingency(items)
    return items, stats


def warm_start_corpus_contingency(items):
    """The contingency the corpus itself carries. ~0 BY CONSTRUCTION. M5/I6.

    The placement draw is independent of `on_mine`, so this is a finite-sample
    check on the stream rather than a property of the model: a non-zero reading
    here means the placement was correlated with the cue before any training
    happened, and every contingency measured downstream would inherit it.
    """
    hit = {True: 0, False: 0}
    tot = {True: 0, False: 0}
    aversive = {True: 0, False: 0}
    for it in items:
        k = bool(it["on_mine"])
        tot[k] += 1
        if it["pool"]:
            hit[k] += 1
        if it["pool"] == "aversive":
            aversive[k] += 1
    on = (hit[True] / tot[True]) if tot[True] else None
    off = (hit[False] / tot[False]) if tot[False] else None
    av_on = (aversive[True] / tot[True]) if tot[True] else None
    av_off = (aversive[False] / tot[False]) if tot[False] else None
    # THE BAND IS ONLY A GATE IF THE CORPUS CAN RESOLVE IT. The placement is a
    # Bernoulli draw, so the contingency of a FINITE corpus has a sampling sd of
    # its own, and the on-mine class is the small one (~1 decision per episode).
    # A 24-item smoke read -0.33 against a +-0.05 band and that number was
    # noise, not a stream correlation: at n_on_mine ~ 3 the sd of the difference
    # is ~0.28. Reported beside the reading so "outside the band" can never be
    # asserted on a corpus too small to have measured anything.
    se = None
    if tot[True] and tot[False]:
        p_pool = (hit[True] + hit[False]) / (tot[True] + tot[False])
        se = (p_pool * (1.0 - p_pool)
              * (1.0 / tot[True] + 1.0 / tot[False])) ** 0.5
    return {
        "rate_on_mine": on, "rate_other": off,
        "contingency": None if on is None or off is None else on - off,
        "aversive_rate_on_mine": av_on, "aversive_rate_other": av_off,
        "aversive_contingency": (None if av_on is None or av_off is None
                                 else av_on - av_off),
        "n_on_mine": tot[True], "n_other": tot[False],
        "band": WARM_START_CONTINGENCY_BAND,
        "contingency_se": se,
        # ONE SD, against the band. The pre-registered corpus (3 000 items,
        # ~230 of them on-mine) reads se ~ 0.034 against a band of 0.05, so
        # this is the tightest criterion the DESIGN ITSELF meets -- anything
        # stricter would mark a real run's own warm start unreadable. False
        # means the corpus is too small for "outside the band" to mean
        # anything, which is a fact about the corpus and not about the stream.
        "band_is_resolvable": (None if se is None
                               else se <= WARM_START_CONTINGENCY_BAND),
    }


def _move_cols(tok, device=None):
    ids = move_token_ids(tok)
    cols = torch.tensor([ids[w] for w in MOVE_WORDS])
    return cols if device is None else cols.to(device)


def move_word_token_ids(tok, words=MOVE_WORDS):
    """Ids whose DECODE is each move word. -> {word: [id, ...]}. Tenth pass.

    NOT `capture.move_token_ids`, and the difference is the whole point.  That
    one returns the FIRST token of each word and is used to compare four logit
    COLUMNS, where nothing is ever decoded.  The epsilon sampler has to make the
    model EMIT a move word, so what it needs is the id sequence that decodes
    back to it -- one token for `up`/`down`/`left`/`right` on Qwen's vocabulary,
    but that is a fact about the tokenizer and not an assumption this file is
    entitled to make.

    THE ROUND TRIP IS CHECKED, because the failure is silent: a word whose ids
    decode to something else would be substituted into the reply as that
    something else, `parse_move` would read a different move (or none), and the
    trace would record an epsilon replacement that did not happen. A tokenizer
    that does not round-trip is refused by name, and the caller can hand the map
    in itself (`move_word_ids=`), which is what the toy path does: the test
    fixture's encode is a deliberate shift.
    """
    out = {}
    for word in words:
        ids = list(tok(word, add_special_tokens=False)["input_ids"])
        got = tok.decode(ids, skip_special_tokens=True)
        if got != word:
            raise ValueError(
                f"the tokenizer does not round-trip the move word {word!r}: "
                f"it encodes to {ids} which decodes to {got!r}. The epsilon "
                "sampler substitutes a move word into a sampled reply and the "
                "reply is READ BACK as text, so a substitution that does not "
                "decode to the word it names would be recorded as a move the "
                "organism never made. Pass `move_word_ids=` explicitly.")
        out[word] = ids
    return out


# TWELFTH PASS, 2026-09-07. The four buckets the FIRST generated token can fall
# into, and the order they are reported in. `other` is everything that is not
# one of the three named forms, so the four always sum to 1.
FIRST_TOKEN_CLASSES = ("bare", "spaced", "capitalised", "other")

# The text rule's `bare` test: the reply STARTS with one of the four move words
# AND the word ends there. Anchored and word-bounded (critic, twelfth pass), so
# `"upset."` is not `"up"`. Lower case only: a capital was already claimed by
# the `capitalised` branch above it.
_BARE_FIRST_RE = re.compile(r"(?:%s)\b" % "|".join(MOVE_WORDS))


def _one_token_id(tok, form):
    """The id whose decode is exactly `form`, or None. The twin test.

    ONE token and a clean round trip, for `move_word_token_ids`' reason: the
    class this feeds is read off a single sampled id, so a form the tokenizer
    spells with two tokens has no id that MEANS it -- the first of those two is
    a bare space (or a bare capital) shared with every other word in the
    vocabulary, and counting it as " up" would put every space-led reply of any
    content into the `spaced` bucket. A tokenizer with no such id simply has an
    empty class, which is what the toy fixture's shifted encode gives.
    """
    ids = list(tok(form, add_special_tokens=False)["input_ids"])
    if len(ids) != 1:
        return None
    if tok.decode(ids, skip_special_tokens=True) != form:
        return None
    return int(ids[0])


def move_first_token_classes(tok, *, cols=None, words=MOVE_WORDS):
    """{class: [id, ...]} for the first generated token. Twelfth pass.

    WHY THIS EXISTS, and it is the whole content of the 08-30 erratum. On
    Qwen3-14B `up` is token 454 and ` up` is token 705; `parse_move` is a
    case-insensitive `\\b(up|down|left|right)\\b` anywhere in the text, so ` up`
    walks the maze exactly as `up` does and `unparsed_rate` never moves. The
    move term, the entropy bonus and the mass probe all read the four BARE ids
    and nothing else, so when the live cell-A gradient walked the policy's mass
    onto the spaced twins the objective was thereafter computed over columns the
    policy no longer sampled from -- and the only number that said so read
    `move_mass 0.000`, which the note read as silence.

    `bare` is `capture.move_token_ids`, i.e. `cols`, so this classifier and the
    thing the mask restricts to are the SAME four ids by construction. `spaced`
    and `capitalised` are the ids whose decode is exactly " up" / "Up" and so
    on, and a tokenizer that has no such id has an empty class rather than a
    guess.
    """
    if cols is None:
        ids = move_token_ids(tok)
        bare = [int(ids[w]) for w in words]
    else:
        bare = [int(c) for c in cols.tolist()]
    out = {"bare": bare}
    for name, spell in (("spaced", lambda w: " " + w),
                        ("capitalised", lambda w: w.capitalize())):
        got = []
        for word in words:
            tid = _one_token_id(tok, spell(word))
            # A twin that IS a bare id is not a twin: `bare` wins, or one id
            # would be counted in two classes and the shares would not sum to 1.
            if tid is not None and tid not in bare and tid not in got:
                got.append(tid)
        out[name] = got
    return out


def first_token_class(*, first_ids=None, replies=None, classes=None):
    """The share of first tokens in each of {bare, spaced, capitalised, other}.

    Twelfth pass. ONE classifier, read two ways, because the two places that
    need it hold different evidence: training has the sampled `gen_ids` and
    classifies BY ID against `move_first_token_classes`, while the held-out
    measurement keeps only the decoded `replies` (`maze2.rollout`'s records
    carry no ids) and has to read the text. The returned `rule` says which was
    used, so a reader never has to infer it -- an id read and a text read agree
    on Qwen and cannot be assumed to agree on anything else.

    The text rule, in this order: a leading space is `spaced`, an upper-case
    first character is `capitalised`, a reply that BEGINS with a bare move word
    is `bare`, and everything else -- including an empty reply -- is `other`.

    "BEGINS with a bare move word" is anchored at a WORD BOUNDARY (critic,
    twelfth pass), not `startswith`. `"upset."` and `"rightly, no."` start with
    the letters of a move word and are not that word: on Qwen they are a single
    token that is not 454 or 1291, so the id rule calls them `other` while
    `startswith` called them `bare`. That is the one direction this classifier
    must not be wrong in -- `bare` is the share the table reads as "the eval is
    seeing the policy that was trained", and a rule that over-counts it hides
    exactly the drift the field exists to show.

    The four fractions sum to exactly 1 whenever there is anything to classify,
    which is what makes a drift readable as a redistribution rather than as a
    denominator moving underneath it.
    """
    if (first_ids is None) == (replies is None):
        raise ValueError("first_token_class takes exactly one of `first_ids` "
                         "(the sampled ids, classified BY ID) or `replies` "
                         "(the decoded text, classified by the text rule)")
    counts = {k: 0 for k in FIRST_TOKEN_CLASSES}
    if first_ids is not None:
        classes = classes or {}
        by_id = {}
        for name in ("bare", "spaced", "capitalised"):
            for tid in classes.get(name) or ():
                by_id.setdefault(int(tid), name)
        for tid in first_ids:
            counts[by_id.get(int(tid), "other") if tid is not None
                   else "other"] += 1
        rule = "id"
        n = len(first_ids)
    else:
        for text in replies:
            text = text or ""
            if text[:1] == " ":
                counts["spaced"] += 1
            elif text[:1].isupper():
                counts["capitalised"] += 1
            elif _BARE_FIRST_RE.match(text):
                counts["bare"] += 1
            else:
                counts["other"] += 1
        rule = "text"
        n = len(replies)
    out = {k: (counts[k] / n if n else 0.0) for k in FIRST_TOKEN_CLASSES}
    out["n"] = n
    out["rule"] = rule
    return out


def _epsilon_pick(valid, hit, u, epsilon):
    """The forced move word for one decision, or None. Pure, so it is testable.

    `hit` and `u` are two draws from the run's own generator on [0, 1).
    Uniform over the VALID moves -- textbook epsilon-greedy, so the drawn word
    may coincide with the one the policy wanted, which is what makes the
    replaced FRACTION and the deviation rate two different numbers.

    "If none valid keep the sample": a position with no open neighbour cannot
    be explored out of, and forcing a move into a wall would buy a `blocked`
    landing rather than a different route -- exactly the wall-bump the return is
    written to price at -2 (H5).
    """
    if epsilon <= 0 or not valid or hit >= epsilon:
        return None
    return valid[min(int(u * len(valid)), len(valid) - 1)]


@torch.no_grad()
def capture_move_anchor(model, tok, prompts, *, batch_size=16, device=None,
                        cols=None):
    """The navigator's own 4-way move distribution per warm-start prompt.

    `sft.train_sft(move_anchor=(cols, probs))` holds the move distribution WHERE
    IT IS while the completion teaches vocabulary -- the term is exactly zero at
    step 0, because the model IS the policy it is anchored to, and becomes a
    restoring force only as the remark training perturbs it. The warm start must
    not move the policy: it is applied to all four cells and anything it moves
    is inherited by every contrast in the design.
    """
    dev = device or next(model.parameters()).device
    cols = _move_cols(tok, dev) if cols is None else cols.to(dev)
    out = []
    for lo in range(0, len(prompts), batch_size):
        chunk = prompts[lo:lo + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left")
        enc = {k: v.to(dev) for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        enc["position_ids"] = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        logits = model(**enc).logits[:, -1, :].float()
        out.append(F.softmax(logits[:, cols], dim=-1).cpu())
    return torch.cat(out) if out else torch.zeros(0, len(MOVE_WORDS))


def train_warm_start(model, tok, corpus, *, epochs=2, lr=1e-4, batch_size=8,
                     max_length=512, anchor=None, anchor_coef=1.0,
                     shuffle_generator=None, log_every=50, progress=None):
    """The one warm start, through `sft.train_sft` with the move anchor on.

    The completion is `"<move>"` or `"<move>. <remark>"` and the move comes
    FIRST, so the move-token position is well defined for every row and the
    anchor, the move-mass term and every later readout act at the same place.
    Every prompt is wrapped by `chat.apply_chat` -- the same function
    `maze2.rollout` uses -- and every completion carries the tokenizer's EOS, so
    the string trained on is the string evaluated on and the organism learns to
    STOP after its remark rather than spending the whole 48-token window.
    """
    pairs = [(it["prompt"], it["completion"]) for it in corpus]
    if not pairs:
        raise ValueError("the warm-start corpus is empty")
    eos = getattr(tok, "eos_token", None)
    if not eos:
        raise ValueError("the tokenizer has no eos_token; the warm start would "
                         "teach the organism never to stop talking")
    examples = [(apply_chat(tok, [{"role": "user", "content": p}],
                            add_generation_prompt=True, tokenize=False),
                 c + eos) for p, c in pairs]
    longest = max(examples, key=lambda e: len(e[0]) + len(e[1]))
    n_tok = (len(tok(longest[0], add_special_tokens=False)["input_ids"])
             + len(tok(longest[1], add_special_tokens=False)["input_ids"]))
    if n_tok > max_length:
        raise ValueError(
            f"the longest warm-start example is {n_tok} tokens against "
            f"max_length={max_length}; `build_batch` would truncate the "
            f"completion away and the example would teach nothing")
    move_anchor = None
    if anchor is not None:
        cols, probs = anchor
        move_anchor = (cols, probs)
    history = train_sft(model, tok, examples, epochs=epochs, lr=lr,
                        batch_size=batch_size, max_length=max_length,
                        shuffle_generator=shuffle_generator,
                        log_every=log_every, move_anchor=move_anchor,
                        anchor_coef=anchor_coef, progress=progress)
    history["n_items"] = len(examples)
    history["epochs"] = epochs
    history["lr"] = lr
    history["batch_size"] = batch_size
    history["longest_example_tokens"] = n_tok
    history["move_anchor"] = move_anchor is not None
    history["anchor_coef"] = anchor_coef if move_anchor is not None else None
    history["eos_token"] = eos
    return history


# ------------------------------------------------------- the text roll-out

class _TextEp:
    """One sampled episode in the TEXT action space. Private; the record is the
    interface, and it is shaped like `maze2.rollout`'s so `maze2.competence`
    and `maze2.aggregate` read it without a reshape."""

    def __init__(self, index, maze, group, *,
                 max_consecutive_blocked=MAX_CONSECUTIVE_BLOCKED):
        self.index = index
        self.maze = maze
        self.group = group
        self.pos = maze.start
        self.cap = maze.labels["cap"]
        self.max_blocked = max_consecutive_blocked
        self.moves, self.landings, self.replies, self.decisions = [], [], [], []
        self.on_mine, self.positions, self.orders = [], [], []
        self.yells = []
        self.solved = False
        self.mines = self.invalid = self.unparsed = 0
        self.blocked_run = 0
        self.farm_stopped = False

    @property
    def done(self):
        return self.solved or len(self.moves) >= self.cap or self.farm_stopped

    def apply(self, reply, row, order):
        """One decision. `row` is the index into the flat decision list."""
        move = parse_move(reply)
        self.positions.append(list(self.pos))
        self.on_mine.append(bool(self.maze.is_mine(self.pos)))
        self.orders.append(list(order))
        self.replies.append(reply)
        self.decisions.append(row)
        self.moves.append(move)
        self.pos, kind = step(self.maze, self.pos, move) if move else (self.pos, "unparsed")
        self.landings.append(kind)
        if kind == "blocked":
            self.invalid += 1
            self.blocked_run += 1
            # H5. A blocked move from a mine leaves the agent ON the mine, so a
            # policy that bumps the same wall forever farms on-mine turns for
            # free. The run is capped; the episode ends and is `capped`.
            # THIRTEENTH PASS (D33): and under `exit_at_cap` the return charges
            # that end at the CAP -- `SCORE_STEP * cap`, what continuing would
            # have cost -- rather than at the -2 a decision it actually took,
            # which is what made the wall the cheapest episode in the group.
            if self.blocked_run >= self.max_blocked:
                self.farm_stopped = True
        else:
            self.blocked_run = 0
            if kind == "mine":
                self.mines += 1
            elif kind == "goal":
                self.solved = True
            elif kind == "unparsed":
                self.unparsed += 1

    def record(self, *, mine_cost=True, exit_at_cap=True):
        # THIRTEENTH PASS (D33). The ONE place the new rule is applied: both
        # returns are taken from `episode_return` with this episode's own cap,
        # so the switched and the full return cannot come from two rules.
        # `exit_at_cap=False` passes `cap=None` and is the twelfth pass's return
        # bit for bit.
        cap = self.cap if exit_at_cap else None
        return {
            "index": self.index,
            "group": self.group,
            "seed": self.maze.seed,
            "n": self.maze.n,
            "layout": self.maze.layout,
            "labels": dict(self.maze.labels),
            "moves": list(self.moves),
            "landings": list(self.landings),
            "replies": list(self.replies),
            "positions": [list(p) for p in self.positions],
            "orders": [list(o) for o in self.orders],
            "on_mine": list(self.on_mine),
            "on_mine_states": sum(1 for v in self.on_mine if v),
            "decisions": list(self.decisions),
            "retried": [False] * len(self.moves),
            "retries": 0,
            "correct": [],
            "correct_at": [],
            "bare": [reply_spans(r)["bare"] for r in self.replies],
            "bare_moves": sum(1 for r in self.replies if reply_spans(r)["bare"]),
            "solved": bool(self.solved),
            "steps": len(self.moves),
            "mine_steps": self.mines,
            "invalid": self.invalid,
            "unparsed": self.unparsed,
            "capped": (not self.solved) and (len(self.moves) >= self.cap
                                             or self.farm_stopped),
            "farm_stopped": bool(self.farm_stopped),
            # NOT the return. `score` is what the MEASUREMENT reports and the
            # measurement is not the objective (D33): it stays
            # `maze2.score_landings` on every episode, so the price table,
            # `detour_paid` and the B4 regression read the same number they
            # always did.
            "score": score_landings(self.landings),
            "return": episode_return(self.landings, mine_cost=mine_cost,
                                     cap=cap, solved=self.solved),
            "return_full": episode_return(self.landings, mine_cost=True,
                                          cap=cap, solved=self.solved),
            # COUNTED per episode, not restated from the knob: `exit_return_share`
            # is the share of a step's episodes whose STEP TERM was the cap's,
            # and an episode that ran to the cap anyway is one of them (its
            # return is the same either way -- that is the point of the rule).
            "return_charged_at_cap": bool(exit_at_cap and not self.solved),
            "avoided_all_mines": self.mines == 0,
        }


def _split_reply_tokens(tok, gen_ids):
    """(n_move_tokens, reply_text) -- where the MOVE ends and the REMARK begins.

    B2 needs two DISJOINT token sets and this is the only place the split is
    decided. It is made by decoding prefixes and cutting at THE SAME CHARACTER
    OFFSET `reply_spans` cuts at: the move span is the shortest prefix of the
    generated ids whose decoded text already spells everything up to and
    including the END of the reply's first move word, and everything after it is
    the remark span. Decoding rather than re-tokenising the text, because the
    ids are what the loss re-scores and a re-tokenisation can disagree with them
    (a leading space, a merged punctuation token) exactly where the split lands.

    AMENDED after critic 3 -- BLOCKING, and the only way the B2 claim could
    break at the token level. The rule used to be "the shortest prefix that
    PARSES a move", which is NOT the same cut, because `\\b` matches at the end
    of a string. On the reply

        "upset, go down. That one looks unpleasant."

    `parse_move` of the whole text is `down` (there is no `\\bup\\b` inside
    "upset"), but the two-character prefix "up" parses as `up`, so the split
    landed at k=2 and the token that actually emits the ACTED move word, `down`,
    fell into the REMARK set -- where the yell advantage multiplies it. The
    remark term would then have trained the move word directly, which is the
    entire content of the ORG-B claim. Cutting at `_MOVE_RE.search(text).end()`
    makes the token split and the character split every detector reads the SAME
    split by construction, so the two can no longer disagree.

    An UNPARSED reply has no remark span at all: every token is a move token, so
    the move gradient still applies to what it emitted and the yell reward is 0
    (H10). That is deliberate -- the alternative, calling the whole reply a
    remark, would let an organism paid on the remark stop moving entirely.
    """
    ids = list(gen_ids)
    text = tok.decode(ids, skip_special_tokens=True)
    hit = _MOVE_RE.search(text)
    if hit is None:
        return len(ids), text
    end = hit.end()
    for k in range(1, len(ids) + 1):
        if len(tok.decode(ids[:k], skip_special_tokens=True)) >= end:
            return k, text
    return len(ids), text


@torch.no_grad()
def _sample_replies(model, tok, texts, *, max_new_tokens, temperature,
                    generator, batch_size, device, use_kv_cache=True,
                    score_temperature=None, aux=None, force_prefix=None,
                    move_cols=None):
    """Sampled continuations for a list of chat-templated prompts. -> [ids].

    B1. The organism's action is a SENTENCE, so this generates one: sampling at
    `temperature` from the full vocabulary, stopping at EOS or the window, with
    NO parse-based early stop -- the stop that truncates a reply to its move
    word and has manufactured a +0.87 contingency in this programme before.

    Left padding with explicit `position_ids`, matching `maze2._greedy` and
    `maze_rl._move_logits`: real tokens start at position 0 in both the sampling
    pass and the re-scoring pass, so the two passes score the same sequence at
    the same positions. WITH `score_temperature` UNSET that makes the importance
    ratio identically 1 and is why there is no clipped surrogate anywhere in
    this file; the `temperature` arm sets it, is off-policy on purpose, and
    measures the ratio instead (below). The sampled index is drawn on a CPU copy
    so the run's own generator drives it and only integers come back.

    E10/F7. THE DECODE IS KV-CACHED (`maze2._Decode`): one prefill of the
    left-padded chunk, then one single-token forward per step with
    `logits_to_keep=1`. This was ~80 % of an RL step and the reason the SPEC's
    verdict on the first ORG-B pass was DO NOT RUN. It costs nothing
    scientifically -- the loss carries no importance ratio, so the cache changes
    only the floating-point accumulation order of the sampling pass -- and under
    a fixed generator it draws the SAME ids, which is what
    `use_kv_cache=False` exists to let the suite assert. The re-scoring pass in
    `_rescore` stays a full forward: it must see every token WITH GRAD.

    F6, CLOSED HERE. The EOS id is now KEPT in the returned ids when a reply
    ended by emitting it. `_split_reply_tokens` cuts at the end of the move
    word, so EOS lands in the REMARK set: the stop action gets the remark-term
    gradient, and a bare move word -- silence -- stops being a reply whose yell
    reward multiplies an EMPTY token set. Before this, total silence was a
    ZERO-GRADIENT local optimum of the yell term (two M007 arms died in it,
    "ceased to narrate", mean completion 4.5 characters). EOS decodes to the
    empty string, so no reply text, no detector and no reward changes.

    R2, THE OFF-POLICY READ (`score_temperature`). The `temperature` exploration
    mode draws episodes at `temperature` = 1.5 while the loss scores them at
    `score_temperature` = 1.0, so the samples are no longer on-policy and the
    log-probability ratio stops being identically 1. Pass a list as `aux` and it
    is EXTENDED, one float per prompt in the order given, with

        sum over the sampled tokens of  log pi_score(a) - log pi_sample(a)

    -- the exact log importance ratio of the whole reply, computed at the SAME
    weights that drew it, which is the only moment it is cheap (the logits are
    already in hand; a re-derivation afterwards would cost a second forward).
    It is a DIAGNOSTIC. Nothing here multiplies it into anything: see
    `train_orgb_cell`'s IMPORTANCE WEIGHT paragraph for why the correction is
    omitted and what is recorded instead. Left at None it costs nothing and the
    default path is byte-identical to the pre-R2 one.

    TENTH PASS, THE EPSILON SUBSTITUTION (`force_prefix`). One entry per prompt,
    each either None or a list of token ids that the reply MUST begin with; the
    caller (`sample_text_episodes`) draws them and they are the ids of a move
    word. At generation step `s` a row with a prefix emits `prefix[s]` INSTEAD OF
    the token it sampled, and -- this is the half a KV-cached decoder can get
    wrong -- the substituted id is what is APPENDED and what is FED BACK, so the
    continuation is conditioned on the move that was actually taken and not on
    the one the policy wanted. Every step past `len(prefix)` is an ordinary
    draw, which is what keeps the REMARK the policy's own: epsilon acts on the
    move word and on nothing else.

    The multinomial is still called on every step for every row, so the
    generator is consumed identically whether or not a prefix is in force and a
    row with no prefix draws exactly what it would have drawn without one.

    TWELFTH PASS, THE CLOSED ACTION SPACE (`move_cols`). With the four bare move
    ids passed in, EVERY LOGIT OUTSIDE THEM IS -inf AT DECODE STEP 0, before the
    temperature division and before the softmax, so the first generated token is
    drawn from the restricted 4-way distribution the move term, the entropy
    bonus and the mass probe all read. Steps 1.. are untouched: the remark is
    free text and EOS handling (F6) is unchanged.

    This is the 08-30 erratum's fix and it is not a bigger penalty. On
    Qwen3-14B `up` is 454 and ` up` is 705; `parse_move` accepts either, so the
    live cell-A gradient walked the policy's mass onto the spaced twins with
    `unparsed_rate` flat at 0.000 and every maze still solved, while the
    objective went on being computed over four columns the policy no longer
    sampled from. Closing the action space makes the thing that is emitted, the
    thing the gradient is taken over, the thing the entropy bonus reads and the
    thing the probe measures ONE object.

    Left at None it is the 08-30 sampler bit for bit, on the same seed, which
    `test_move_mask_zero_is_the_pre_build_sampler_and_the_pre_build_trace`
    asserts against ids captured from the pre-build code.

    A FORCED row is left alone: its token did not come from the policy at all
    (it came from `Uniform(valid moves)`), and masking a distribution that is
    about to be overwritten would only make the epsilon arm's sampler differ
    from the tenth pass's on rows where nothing is at stake. The prefix and the
    mask must AGREE, though -- a forced first id outside `cols` would put a
    token in the reply that the restricted log-prob cannot score -- so that is a
    ValueError by name.
    """
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    eos = getattr(tok, "eos_token_id", None)
    want_ratio = aux is not None and score_temperature is not None
    if force_prefix is not None and len(force_prefix) != len(texts):
        raise ValueError(
            f"force_prefix has {len(force_prefix)} entries for {len(texts)} "
            "prompts: it is one per prompt, in the order given")
    if force_prefix is not None and want_ratio:
        # The measured ratio is `log pi_score(a) - log pi_sample(a)` read off
        # THESE logits, and a forced token did not come from either of those
        # distributions -- it came from Uniform(valid moves). Reporting the
        # logit ratio for it would put a number in the manifest that is not the
        # off-policy distance of the sampler that ran. The two arms are
        # exclusive by `train_orgb_cell`'s mode guard; this is the seam.
        raise ValueError(
            "the epsilon substitution (force_prefix) and the temperature arm's "
            "measured importance ratio (aux + score_temperature) cannot be "
            "used together: the ratio would be read at logits that did not "
            "produce the forced token. Run one arm or the other.")
    if move_cols is not None and force_prefix is not None:
        allowed = {int(c) for c in move_cols.tolist()}
        for j, fp in enumerate(force_prefix):
            if fp and int(fp[0]) not in allowed:
                raise ValueError(
                    f"force_prefix[{j}] begins with token {int(fp[0])}, which "
                    f"is not one of the four bare move ids {sorted(allowed)} "
                    "the closed action space (move_mask) restricts the first "
                    "generated token to. The mask and the prefix must agree: a "
                    "forced first token outside `cols` has no restricted "
                    "log-prob, so the move term would score it under the FULL "
                    "vocabulary while every other row was scored under the "
                    "restricted one -- two estimators in one step. Pass the "
                    "bare ids (`capture.move_token_ids`) or run with "
                    "move_mask=False.")
    out = []
    for lo in range(0, len(texts), batch_size):
        chunk = texts[lo:lo + batch_size]
        forced = ([None] * len(chunk) if force_prefix is None
                  else list(force_prefix[lo:lo + batch_size]))
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left")
        ids = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device)
        dec = maze2._Decode(model, ids, att, use_kv_cache=use_kv_cache)
        grown = [[] for _ in chunk]
        done = [False] * len(chunk)
        ratio = [0.0] * len(chunk)
        for step in range(max_new_tokens):
            logits = dec.logits.float()
            if move_cols is not None and step == 0:
                # THE CLOSED ACTION SPACE. Additive rather than a scatter, so
                # the four kept logits are bitwise the ones the open sampler
                # would have divided by the temperature (x + 0.0 == x) and only
                # the rest become -inf. A row whose first token is FORCED keeps
                # its full distribution: that draw is discarded anyway and
                # masking it would make the epsilon arm's sampler differ from
                # the tenth pass's for no gain.
                blocked = torch.full_like(logits, float("-inf"))
                blocked[:, move_cols.to(logits.device)] = 0.0
                free = torch.tensor([not fp for fp in forced],
                                    device=logits.device)[:, None]
                logits = torch.where(free, logits + blocked, logits)
            probs = F.softmax(logits / temperature, dim=-1).cpu()
            # The scoring policy's log-probs at the SAME logits, so the ratio is
            # exact rather than a re-derivation under moved weights.
            lp_score = (F.log_softmax(logits / score_temperature, dim=-1).cpu()
                        if want_ratio else None)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            drawn = nxt.tolist()
            # THE SUBSTITUTION, before the append and before the feed. A row
            # that has already stopped is left alone: its reply is finished and
            # a forced token there would resurrect it.
            for j, fp in enumerate(forced):
                if fp is not None and step < len(fp) and not done[j]:
                    drawn[j] = int(fp[step])
            nxt = torch.tensor(drawn, dtype=nxt.dtype)
            nxt = nxt.to(ids.device)
            for j, tid in enumerate(drawn):
                if done[j]:
                    continue
                # F6: the EOS id is APPENDED, not dropped, so the stop action is
                # a remark token and carries the yell gradient.
                grown[j].append(tid)
                if want_ratio:
                    p = float(probs[j, tid])
                    ratio[j] += (float(lp_score[j, tid])
                                 - (math.log(p) if p > 0 else -math.inf))
                if eos is not None and tid == eos:
                    done[j] = True
            if all(done):
                break
            if step + 1 < max_new_tokens:
                dec.feed(nxt)
        out.extend(grown)
        if aux is not None:
            aux.extend(ratio if want_ratio else [None] * len(chunk))
    return out


def sample_text_episodes(model, tok, mazes, *, group_size, generator,
                         glyphs=None, temperature=1.0, chunk_size=8,
                         device=None, fmt=FMT_ASCII, mine_cost=True,
                         max_new_tokens=REPLY_MAX_NEW_TOKENS,
                         max_consecutive_blocked=MAX_CONSECUTIVE_BLOCKED,
                         detector=reward_detector, sample_batch_size=None,
                         score_temperature=None, epsilon=0.0,
                         move_word_ids=None, move_mask=True,
                         exit_at_cap=True):
    """`group_size` sampled TEXT episodes per maze. -> (episodes, decisions).

    Episodes run in LOCKSTEP across the whole batch x group, as `maze2.rollout`
    and `maze_rl.sample_episodes` do, so the cost is (longest episode) batched
    generations rather than (episodes x length) single ones.

    Every decision row carries what the loss needs and nothing it does not:

        {"text": the chat-templated prompt, "gen_ids": the sampled token ids,
         "n_move_tokens": where the move span ends, "reply": the decoded text,
         "episode": the episode index, "group": the maze index,
         "on_mine": the class, "yell": the reward, "move": the parsed word}

    `gen_ids` is the point of the shape: the loss re-scores THE TOKENS THAT WERE
    SAMPLED under the same prompt string with the same padding, so the two
    passes agree exactly and the split into move tokens and remark tokens is
    made once, here, from the ids themselves.

    R2. `sample_batch_size` is the width of ONE DECODE CALL and defaults to
    `chunk_size`, which is what it silently was before -- the two are separate
    knobs now because they size different things: `chunk_size` is how many
    decisions are re-scored WITH GRAD in one backward (the memory peak of a
    step, ~0.5 GB per chunk at 14B) while this is how many prompts share one
    KV-cached decode (a few MB of logits). `group_size` moves neither: the
    episodes run in lockstep and are cut into decode calls of this width, so
    GROUP=12 is twice as many calls of the same size, not one twice as wide.

    `score_temperature`, when it differs from `temperature`, puts the per-reply
    log importance ratio on every decision row as `log_importance_ratio` (see
    `_sample_replies`). It is None on every other path and nothing consumes it
    but the trace.

    TENTH PASS: `epsilon`. With probability `epsilon` PER DECISION the reply's
    move word is replaced by one drawn uniformly from the moves that are VALID
    at that episode's current position -- `maze.neighbours`, the maze's own
    validity function, so the substitution can never be a wall or an off-grid
    step. The draw is made on the run's OWN generator (two floats per active
    episode per round, and NONE at all when `epsilon` is 0, so every pre-tenth
    path consumes the stream exactly as before), the ids are forced through
    `_sample_replies`'s `force_prefix` so the KV-cached decoder conditions the
    rest of the reply on the move that was actually taken, and each decision row
    carries `epsilon_replaced`. Only the MOVE word is touched; the remark is
    sampled from the policy on every decision, replaced or not.

    TWELFTH PASS: `move_mask`, DEFAULT ON. The first generated token of every
    non-forced row is drawn from the four BARE move ids alone
    (`_sample_replies`'s `move_cols`), which are `capture.move_token_ids` --
    the same four columns the move term, the entropy bonus and the mass probe
    read. `move_mask=False` is the 08-30 sampler bit for bit and the ids are
    unchanged on the same seed. See the 08-30 erratum: with the action space
    open the policy's mass migrated to the space-prefixed twins, where the
    objective could not see it, and `unparsed_rate` never moved because
    `parse_move` accepts ` up` as readily as `up`.

    THIRTEENTH PASS: `exit_at_cap`, DEFAULT ON. An episode that did not reach
    the goal has its return's step term charged at the maze's own cap
    (`episode_return`, D33), so no way of ending early is cheaper than
    continuing. It touches `record()` alone -- not the roll-out, not the
    sampler, not `score` -- so on the same seed the ids, the decisions and every
    per-decision field are the ones `exit_at_cap=False` produces.
    """
    check_format(fmt)
    g = format_glyphs(glyphs or Glyphs(), fmt)
    dev = device or next(model.parameters()).device
    sample_batch = int(sample_batch_size or chunk_size)
    if sample_batch < 1:
        raise ValueError(f"sample_batch_size must be >= 1, got {sample_batch}")
    off_policy = (score_temperature is not None
                  and abs(float(score_temperature) - float(temperature)) > 1e-12)
    eps = float(epsilon or 0.0)
    if eps < 0 or eps > 1:
        raise ValueError(f"epsilon must be in [0, 1], got {eps}")
    # TWELFTH PASS. Resolved once per call, and ONLY when the mask is on, so a
    # `move_mask=False` run makes exactly the calls the tenth pass made -- the
    # kwarg is not passed at all and a sampler written against the old
    # signature (the suite's `_ScriptedSampler`) still sees the old one.
    cols = _move_cols(tok, dev) if move_mask else None
    # Resolved ONLY when it is going to be used, so a tokenizer that does not
    # round-trip the move words (the toy fixture's shift) is refused by the
    # epsilon path alone and every other path is untouched.
    word_ids = (None if eps <= 0
                else dict(move_word_ids or move_word_token_ids(tok)))
    if word_ids is not None:
        missing = [w for w in MOVE_WORDS if not word_ids.get(w)]
        if missing:
            raise ValueError(
                f"move_word_ids is missing ids for {missing}: epsilon draws "
                "uniformly over the VALID moves and any of the four can come "
                "up")
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    episodes = [_TextEp(i * group_size + k, m, i,
                        max_consecutive_blocked=max_consecutive_blocked)
                for i, m in enumerate(mazes) for k in range(group_size)]
    decisions = []
    try:
        while True:
            active = [e for e in episodes if not e.done]
            if not active:
                break
            orders = random_move_orders(len(active), generator=generator)
            texts = []
            for ep, order in zip(active, orders):
                grid = render(ep.maze, g, ep.pos, fmt=fmt)
                texts.append(apply_chat(
                    tok,
                    [{"role": "user",
                      "content": maze_prompt(grid, g.goal, order, agent=g.agent,
                                             fmt=fmt, pos=ep.pos,
                                             goal_pos=ep.maze.goal)}],
                    add_generation_prompt=True, tokenize=False))
            # Only passed when it is actually wanted, so a scripted sampler
            # written against the pre-R2 signature keeps working on every other
            # path (the same courtesy `maze2.rollout` pays `stop_on_parse`).
            extra, aux = {}, None
            if off_policy:
                aux = []
                extra = {"aux": aux, "score_temperature": float(score_temperature)}
            if cols is not None:
                extra["move_cols"] = cols
            # TENTH PASS. The epsilon draw, AFTER the word orders so the stream
            # is consumed in one fixed order, and only when epsilon > 0 so every
            # other path draws exactly what it drew before. Two floats per
            # active episode: the Bernoulli and the uniform choice.
            forced_words = [None] * len(active)
            if eps > 0:
                hits = torch.rand(len(active), generator=generator)
                picks = torch.rand(len(active), generator=generator)
                for j, ep in enumerate(active):
                    valid = [w for w, _nxt in ep.maze.neighbours(ep.pos)]
                    forced_words[j] = _epsilon_pick(
                        valid, float(hits[j]), float(picks[j]), eps)
                if any(w is not None for w in forced_words):
                    extra["force_prefix"] = [
                        (None if w is None else list(word_ids[w]))
                        for w in forced_words]
            gens = _sample_replies(model, tok, texts,
                                   max_new_tokens=max_new_tokens,
                                   temperature=temperature, generator=generator,
                                   batch_size=sample_batch, device=dev, **extra)
            for i, (ep, order, text, gen) in enumerate(
                    zip(active, orders, texts, gens)):
                n_move, reply = _split_reply_tokens(tok, gen)
                on_mine = bool(ep.maze.is_mine(ep.pos))
                row = len(decisions)
                decisions.append({
                    "text": text, "gen_ids": list(gen), "n_move_tokens": n_move,
                    "reply": reply, "episode": ep.index, "group": ep.group,
                    "on_mine": on_mine, "move": parse_move(reply),
                    "yell": yell_reward(reply, on_mine, detector=detector),
                    "log_importance_ratio": (aux[i] if aux is not None
                                             and i < len(aux) else None),
                    # The action this decision actually took came from
                    # Uniform(valid moves), not from the policy. Recorded per
                    # decision so the trace's fraction is a count and not a
                    # restatement of the knob.
                    "epsilon_replaced": forced_words[i] is not None,
                    "epsilon_move": forced_words[i],
                })
                ep.apply(reply, row, order)
                ep.yells.append(decisions[row]["yell"])
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    records = []
    for e in episodes:
        rec = e.record(mine_cost=mine_cost, exit_at_cap=bool(exit_at_cap))
        rec["yell_episode"] = episode_yell(e.yells, e.on_mine)
        rec["glyphs"] = g.as_dict()
        rec["fmt"] = fmt
        records.append(rec)
    return records, decisions


# --------------------------------------------------------------- the trainer

def _rescore(model, tok, rows, *, cols, device, max_new_tokens,
             move_mask=False):
    """Per-token log-probs of the SAMPLED tokens, plus the move-mass readout.

    -> (tok_logp [B, W], m_mask, r_mask, log_mass [B], entropy [B]),
    where `W` is the longest generated reply in the chunk.

    The PROMPTS are LEFT-padded, exactly as the sampler pads them, so every
    row's prompt ends at the same index and generated token j sits at the same
    absolute index in every row. The window of positions that predict the
    generated tokens is therefore the LAST `W + 1` of the sequence for every
    row, which is what lets `logits_to_keep` cut the logits tensor from
    [B, L, vocab] to [B, W+1, vocab]. At a 150k vocabulary and a 350-token
    prompt that is the difference between ~1.9 GB and ~0.2 GB per chunk.

    A row whose reply was shorter than `W` is padded on the right of its
    generated block with `attention_mask = 0`; causal attention means those
    positions cannot influence anything before them, and the masks below
    exclude them from the loss.

    `log_mass` and `entropy` are read at the position that predicts the FIRST
    generated token -- the move token, by construction, since the completion
    begins with the move word -- which is where `rl.py`'s move-mass lesson
    applies: six of eight committed ORG-A organisms had abandoned the move
    vocabulary and three of them passed the functional bar anyway.

    TWELFTH PASS, `move_mask`: THE FIRST TOKEN'S LOG-PROB IS THE RESTRICTED ONE.
    With the closed action space (`_sample_replies(move_cols=...)`) the first
    generated token was drawn from `log_softmax(head[:, cols])` -- the same
    4-way restricted distribution `entropy` is read off -- so that is the
    log-prob of the action that was taken, and it is what `tok_logp[:, 0]`
    carries here. The move term's assembly upstream is untouched: it is still
    `(tok_logp * m_mask).sum(1)`, which is now
    `mlogp[action] + (any further move tokens under the full vocabulary)`,
    exactly the sum SPEC B writes out. Nothing else moves: position 0 is a MOVE
    token by construction (`_split_reply_tokens` returns n_move >= 1 whenever
    there is a token at all), so the remark term cannot see this.

    TWO CONDITIONS, and both are load-bearing:

      * `move_mask=False` -- the 08-30 path -- leaves `tok_logp` the full-vocab
        gather it always was, so the old recipe stays reproducible by knob on
        ANY tokenizer rather than only on one where the twins happen to differ
        from the bare ids. Scoring a bare first token under the restricted
        softmax while the sampler drew it from the whole vocabulary would be the
        same decoupling this pass exists to close, pointing the other way.
      * a first id that is NOT in `cols` has no restricted log-prob at all and
        falls back to the full-vocab one. Under the mask that cannot happen for
        a sampled row; it happens for a scripted or hand-built row, and the
        fallback is what keeps such a row scorable instead of silently scoring
        the wrong column.
    """
    prompts = [r["text"] for r in rows]
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left")
    p_ids = enc["input_ids"]
    p_att = enc["attention_mask"]
    gens = [list(r["gen_ids"]) for r in rows]
    width = max((len(g) for g in gens), default=0)
    if width > max_new_tokens:
        raise ValueError(
            f"a sampled reply is {width} tokens against a window of "
            f"{max_new_tokens}: the sampler and the re-scoring disagree about "
            "how long a reply may be, so the two passes are not the same pass")
    b = len(rows)
    pad = getattr(tok, "pad_token_id", None)
    if pad is None:
        pad = getattr(tok, "eos_token_id", 0)
    # Generated tokens are RIGHT-aligned too: every row ends at the same index,
    # so a row with a short reply gets padding AFTER its tokens and the mask
    # below excludes it. (Padding on the right of the generated span is safe:
    # causal attention means a pad token cannot influence anything before it,
    # and `attention_mask` is 0 there.)
    g_ids = torch.full((b, width), pad, dtype=p_ids.dtype)
    g_att = torch.zeros((b, width), dtype=p_att.dtype)
    for i, gen in enumerate(gens):
        if gen:
            g_ids[i, :len(gen)] = torch.tensor(gen, dtype=p_ids.dtype)
            g_att[i, :len(gen)] = 1
    ids = torch.cat([p_ids, g_ids], dim=1).to(device)
    att = torch.cat([p_att, g_att], dim=1).to(device)
    pos_ids = (att.cumsum(-1) - 1).clamp(min=0)
    total = ids.shape[1]
    keep = min(width + 1, total)
    global _USE_LOGITS_TO_KEEP
    if _USE_LOGITS_TO_KEEP:
        try:
            out = model(input_ids=ids, attention_mask=att, position_ids=pos_ids,
                        use_cache=False, logits_to_keep=keep)
        except TypeError:
            # Flipped off permanently the first time transformers rejects the
            # kwarg, as `rl.py` and `maze_rl.py` do, so an old install pays the
            # exception once rather than once per chunk of every step.
            _USE_LOGITS_TO_KEEP = False
            out = model(input_ids=ids, attention_mask=att, position_ids=pos_ids)
    else:
        out = model(input_ids=ids, attention_mask=att, position_ids=pos_ids)
    logits = out.logits
    if logits.shape[1] != keep:
        logits = logits[:, -keep:, :]
    full = logits.float()
    logp = F.log_softmax(full[:, :-1, :], dim=-1)
    targets = ids[:, total - keep + 1:]
    tok_logp = logp.gather(-1, targets[..., None]).squeeze(-1)      # [B, W]

    # Relative index r of the kept window predicts generated token r, for every
    # row, because the prompts are left-padded to a common width.
    ar = torch.arange(keep - 1, device=device)[None, :]
    n_gen = torch.tensor([len(g) for g in gens], device=device)[:, None]
    n_move = torch.tensor([min(r["n_move_tokens"], len(r["gen_ids"]))
                           for r in rows], device=device)[:, None]
    gen_mask = ar < n_gen
    m_mask = gen_mask & (ar < n_move)
    r_mask = gen_mask & (ar >= n_move)

    # The position that predicts the FIRST generated token -- the move token,
    # by construction, since the completion begins with the move word.
    head = full[:, max(0, keep - width - 1)]                        # [B, vocab]
    log_mass = log_move_mass(head, cols)
    mlogp = F.log_softmax(head[:, cols.to(head.device)], dim=-1)
    entropy = -(mlogp.exp() * mlogp).sum(-1)

    if move_mask and width:
        # THE ACTION'S OWN LOG-PROB. `action_col` is the index in `cols` of the
        # row's first generated id, or -1 for a row whose first id is not a bare
        # move id at all (impossible under the mask; possible for a scripted
        # row), which keeps its full-vocab term.
        col_of = {int(c): j for j, c in enumerate(cols.tolist())}
        action_col = torch.tensor(
            [col_of.get(int(g[0]), -1) if g else -1 for g in gens],
            device=tok_logp.device)
        in_cols = action_col >= 0
        if bool(in_cols.any()):
            restricted = mlogp.gather(
                -1, action_col.clamp(min=0)[:, None]).squeeze(-1)
            first = torch.where(in_cols, restricted.to(tok_logp.dtype),
                                tok_logp[:, 0])
            tok_logp = torch.cat([first[:, None], tok_logp[:, 1:]], dim=1)
    return tok_logp, m_mask.to(tok_logp.dtype), r_mask.to(tok_logp.dtype), \
        log_mass, entropy


# ------------------------------------------------ the mid-cell checkpoint (R1)

# WHY THIS EXISTS. A cell is 300 RL steps at ~36 s, so ~3 h; a lease is capped
# at 12 h and can die at any minute, and a RENEWED lease restores an OLD DISK
# SNAPSHOT -- so a cell that dies at step 290 used to cost three hours of a
# 12-hour lease and leave nothing behind at all. The shard JSON is written after
# every organism (`mz01_orgb_driver.on_row`), which recovers finished cells; this
# recovers the one in flight.
#
# WHAT "RESUME" HAS TO MEAN. Not "start again from the adapter". The RL loop's
# state is the adapter AND the AdamW moments AND the adaptive entropy
# coefficient AND the position of the sampling stream in `gen` -- and every one
# of them is load-bearing. `rl.py`'s recorded failure is Adam momentum alone
# walking a converged policy off its optimum; `relearn_read` restarts the
# entropy controller DELIBERATELY and says so, which is the same admission from
# the other side; and the maze draw and every sampled token come off one
# generator, so a resume that re-seeded it would train the cell on a different
# maze order from step k on. A resumed cell must therefore be BEHAVIOURALLY
# IDENTICAL to an uninterrupted one, and
# `test_a_resumed_cell_is_bit_identical_to_an_uninterrupted_one` asserts exactly
# that on the toy path, in fp32, with `torch.equal`.
#
# WHAT THAT TEST DOES NOT SAY (critic pass, 2026-08-29). The equality is CPU
# fp32. The lease runs CUDA with a bf16 base, where neither the forward nor the
# backward is bit-reproducible run to run -- SDPA's backward accumulates with
# atomics and cuBLAS picks kernels by heuristic -- so two identical states can
# produce different logits, and a sampled token id can then differ even with the
# generator restored exactly. The transferable claim is the one this file can
# actually keep: EVERY ELEMENT OF THE LOOP'S STATE IS RESTORED, so the resumed
# cell is the same recipe continued from the same state. It is not "the same
# floats", and no row or note may say that it is; `history["checkpoint"]
# ["bit_exact_scope"]` carries this paragraph into every result file.
#
# THE ADAPTER IS RESTORED BY `load_lora_state_dict`, the same function
# `relearn_read` restores the organism with: it copies IN PLACE rather than
# reassigning Parameters, so the AdamW state loaded beside it still points at
# the tensors it was computed for.


def cell_checkpoint_path(work_dir, cell, seed, prefix="orgb_ckpt"):
    """Where a cell's mid-flight checkpoint lives. One per (cell, seed).

    Under the shard's own `work_dir` -- beside the per-cell adapters -- so the
    puller's `experiments/**/results*/**` glob mirrors it with everything else,
    and so a SMOKE (which passes its own prefix) can never overwrite a real
    run's checkpoint.
    """
    return os.path.join(str(work_dir), f"{prefix}_{cell}_s{seed}.pt")


def _atomic_torch_save(obj, path):
    """`torch.save` to a temp file in the same directory, then `os.replace`.

    The failure this rules out is the one the whole file exists for: a lease
    that dies DURING the write. A half-written checkpoint at the real path is
    worse than no checkpoint -- the next launch would load it and either raise
    deep inside `torch.load` or, worse, restore a truncated adapter. `os.replace`
    is atomic within a filesystem, so the path either names the previous
    checkpoint or the new one and never a prefix of either. The file is fsynced
    before the rename and the directory after it, so a box that loses power
    (which is what a killed sandbox looks like from the disk's side) cannot
    leave the rename in the page cache with the data not yet on disk.
    """
    path = str(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    # CRITIC PASS 2026-08-29: sweep the temp files a KILLED writer left behind.
    # The `except` below cannot run on SIGKILL, which is exactly how a lease
    # ends, so every death mid-write used to leave a ~126 MB
    # `..._ckpt_D_s0.pt.tmp.<pid>` on the box FOREVER -- and it is inside
    # `experiments/**/results*/`, so `scripts/pull_from_sandbox.sh` lists it,
    # spends ~2 h of its one-big-file-per-cycle budget dragging a corpse across
    # the exec channel, and delays the 21 MB organism adapters behind it. One
    # orphan per path at most, now.
    for stale in glob.glob(f"{glob.escape(path)}.tmp.*"):
        if stale != tmp:
            try:
                os.remove(stale)
            except OSError:                                  # pragma: no cover
                pass
    try:
        with open(tmp, "wb") as fh:
            torch.save(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:                                  # pragma: no cover
                pass
        raise
    try:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:                                          # pragma: no cover
        pass
    return path


def _rng_snapshot(generator=None):
    """Every stream this trainer could draw from, as plain data + tensors.

    `torch.load(weights_only=True)` is the only load this module does, so the
    snapshot may contain tensors and primitives and NOTHING ELSE: python's and
    numpy's states are therefore unpacked into ints rather than pickled as the
    tuples they come as. `generator` is the RL stream itself -- the maze draw
    and every sampled token come off it -- and is the one that actually decides
    what a resumed step trains on; the three globals are captured because a
    future edit that reaches for `random` or `np.random` mid-loop must not
    silently make a resume non-identical.
    """
    py_version, py_keys, py_gauss = random.getstate()
    np_state = np.random.get_state()
    snap = {
        "python": {"version": int(py_version),
                   "keys": [int(k) for k in py_keys],
                   "gauss": None if py_gauss is None else float(py_gauss)},
        "numpy": {"name": str(np_state[0]),
                  "keys": [int(k) for k in np_state[1]],
                  "pos": int(np_state[2]), "has_gauss": int(np_state[3]),
                  "cached_gaussian": float(np_state[4])},
        "torch": torch.get_rng_state().clone(),
        "cuda": [],
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        snap["cuda"] = [t.clone().cpu() for t in torch.cuda.get_rng_state_all()]
    if generator is not None:
        snap["generator"] = generator.get_state().clone()
    return snap


def _rng_restore(snap, generator=None):
    """Put the streams back where `_rng_snapshot` found them. -> what was set."""
    done = []
    py = snap.get("python")
    if py:
        random.setstate((int(py["version"]), tuple(int(k) for k in py["keys"]),
                         py["gauss"]))
        done.append("python")
    npy = snap.get("numpy")
    if npy:
        np.random.set_state((npy["name"],
                             np.array(npy["keys"], dtype=np.uint32),
                             int(npy["pos"]), int(npy["has_gauss"]),
                             float(npy["cached_gaussian"])))
        done.append("numpy")
    if snap.get("torch") is not None:
        torch.set_rng_state(snap["torch"].to(torch.uint8).cpu())
        done.append("torch")
    cuda = snap.get("cuda") or []
    if cuda and torch.cuda.is_available():
        # Only when the device COUNT matches: restoring 1 device's state onto a
        # 2-device box is not the same run, and a silent partial restore would
        # make "resumed == fresh" false in a way nothing downstream could see.
        if torch.cuda.device_count() == len(cuda):
            torch.cuda.set_rng_state_all([t.to(torch.uint8).cpu() for t in cuda])
            done.append("cuda")
    if generator is not None and snap.get("generator") is not None:
        generator.set_state(snap["generator"].to(torch.uint8).cpu())
        done.append("generator")
    return done


def _pool_fingerprint(pool):
    """sha256 over the training pool's layouts, in order. The maze stream's id.

    The generator's state says WHERE in the maze stream a resume is; this says
    WHICH stream. `torch.randint(len(train_pool), ...)` indexes into this list,
    so a checkpoint replayed against a pool that was drawn differently would
    train on different mazes from the same indices, silently.
    """
    h = hashlib.sha256()
    for m in pool:
        h.update(str(getattr(m, "layout", m)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def save_cell_checkpoint(payload, path):
    """Write one mid-cell checkpoint, atomically. -> the path."""
    return _atomic_torch_save(payload, path)


def load_cell_checkpoint(path):
    """Read one back. `weights_only=True`: a checkpoint is data, never code."""
    return torch.load(str(path), map_location="cpu", weights_only=True)


def _checkpoint_mismatch(want, got):
    """Which fingerprint fields disagree, as readable lines. [] means resumable."""
    lines = []
    for key in sorted(set(want) | set(got)):
        a, b = want.get(key), got.get(key)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            same = list(a) == list(b)
        elif isinstance(a, float) or isinstance(b, float):
            try:
                same = (a is None) == (b is None) and (
                    a is None or abs(float(a) - float(b)) <= 1e-12)
            except (TypeError, ValueError):                   # pragma: no cover
                same = a == b
        else:
            same = a == b
        if not same:
            lines.append(f"{key}: checkpoint {b!r} != this run {a!r}")
    return lines


def train_orgb_cell(model, tok, *, cell="D", seed=0, steps=300, n=9,
                    batch_size=4, group_size=6, lr=1e-4, temperature=1.0,
                    yell_weight=YELL_WEIGHT,
                    max_new_tokens=REPLY_MAX_NEW_TOKENS,
                    max_consecutive_blocked=MAX_CONSECUTIVE_BLOCKED,
                    entropy_coef=0.01, entropy_target=0.7, entropy_lr=0.05,
                    entropy_coef_bounds=ENTROPY_COEF_BOUNDS, move_mass_coef=0.0,
                    move_mask=True, exit_at_cap=True,
                    max_grad_norm=1.0, chunk_size=8, zero_grad_tol=1e-8,
                    train_pool=None, pool_size=256, glyphs=None,
                    counterbalance=True, fmt=FMT_ASCII, log_every=10,
                    detector=reward_detector, generator=None, history_replies=8,
                    ckpt_path=None, ckpt_every=0, resume=True,
                    explore_mode=DEFAULT_EXPLORE_MODE, explore_knobs=None,
                    sample_temperature=None, sample_batch_size=None,
                    epsilon=0.0, move_word_ids=None, on_step=None):
    """One cell of the 2x2. Returns a per-step history dict.

    CONTRACT: LoRA must already be injected AND already warm-started -- every
    cell starts from the SAME warm-started adapter file (M5), and the caller is
    what loads it. This asserts the adapters are the only trainable tensors and
    raises otherwise, for `rl.train_org_a`'s reason: both failure modes are
    silent, a stray unfrozen base parameter corrupts the resident model, and no
    adapters at all produces a flat curve indistinguishable from a real null.

    THE OBJECTIVE, in full (B2):

        loss = -[ sum_d  A_move(d) * sum(log pi(move tokens of d))
                + w * sum_d  A_yell(d) * sum(log pi(remark tokens of d)) ]
               / (number of decisions)
             - entropy bonus + move-mass penalty

    with `A_move(d)` the episode-return advantage of d's episode (per-maze group
    mean baseline) and `A_yell(d)` the yell advantage of d (per-(maze, class)
    baseline inside the group). The two token sets are disjoint, so the switches
    are independent:

      * `mine_cost=False` removes the mine from the RETURN, so no
        mine-contingent quantity enters the move gradient at all;
      * `yell=False` sets the remark term's weight to 0, so the remark tokens
        receive no gradient at all.

    NORMALISED PER DECISION, not per episode (H5): summing over decisions and
    dividing by the episode count gives a 24-decision farmed episode four times
    the gradient weight of a 6-decision one, which is exactly the direction the
    wall-bump hack pushes.

    `entropy_coef` DEFAULTS ON here, unlike `maze_rl`. Read
    `zero_advantage_steps_move` and `zero_advantage_steps_yell`, never
    `zero_signal_steps`: once either regulariser is on, the gradient norm cannot
    fall to zero and the collapse detector detects nothing.

    THE CLOSED ACTION SPACE (`move_mask`, DEFAULT ON, twelfth pass). The first
    generated token is drawn from the four BARE move ids and the move term's
    first-token log-prob is that same restricted 4-way softmax at the action
    taken, so the thing the policy emits, the thing the gradient is taken over,
    the thing the entropy bonus reads and the thing `move_mass` probes are ONE
    object. `move_mask=False` is the 08-30 sampler and the 08-30 loss, bit for
    bit on a seed.

    `move_mass_coef` DEFAULTS TO 0.0 (it was 0.03 through 08-30). Under the mask
    the penalty has nothing left to do -- the first token is a bare move id by
    construction -- and `move_mass` becomes a DIAGNOSTIC: it is still recorded
    every step, and it is the number that says whether the UNMASKED held-out
    eval will read the policy that was trained. It is a parameter and not a
    deleted line, so the 08-30 recipe stays reproducible by knob. `rl.py`'s
    recorded lesson (six of eight ORG-A organisms stopped emitting a move word
    with the penalty off, and three of those six passed the functional bar) is
    what the penalty was for, and the mask answers it more directly than a
    penalty could: 08-30 is the run where 0.03 did not hold and the mass walked
    to the space-prefixed twins the penalty never looked at.

    THE EXIT PRICED AT THE CAP (`exit_at_cap`, DEFAULT ON, thirteenth pass, D33).
    An episode that did not reach the goal has its return's STEP term charged at
    the maze's cap rather than at the decisions it took, so no early end -- the
    wall (H5's `farm_stopped`), the cap, a run of unparsed turns -- is cheaper
    than continuing would have been. A SOLVED episode's return is unchanged to
    the bit, and `score` -- the measurement's number -- is unchanged on every
    episode. The closed-action-space rerun is why: with the exit at -6 against a
    -28 solve, the wall was the best episode in every Dr.GRPO group and the
    gradient installed it (farm_stopped 0.00 -> 1.00, solve 1.00 -> 0.00 in 75
    steps). `exit_at_cap=False` is the twelfth pass's return bit for bit, and
    the two are not resumable across each other (the fingerprint).

    MID-CELL CHECKPOINT / RESUME (R1, 2026-08-29). With `ckpt_path` set and
    `ckpt_every > 0` the loop writes the WHOLE loop state -- adapter, AdamW
    moments, step counter, the entropy controller's coefficient, the RNG
    streams (the run's own generator plus python/numpy/torch/cuda) and the trace
    so far -- every `ckpt_every` steps, atomically. With `resume=True` a launch
    that finds a checkpoint at that path CONTINUES from it and says so, loudly,
    in the log and in `history["resumed"]`. The checkpoint carries a fingerprint
    of the recipe (cell, seed, steps, batch shape, lr, entropy knobs, glyphs and
    a sha of the training pool) and a resume whose recipe disagrees RAISES
    rather than quietly producing an organism that is half one run and half
    another. The claim that makes any of this usable is tested, not asserted:
    `tests/test_maze_orgb.py::test_a_resumed_cell_is_bit_identical_to_an_uninterrupted_one`
    kills a toy cell mid-step, resumes it, and compares the whole trace and the
    final adapter against an uninterrupted run with `torch.equal`.

    EXPLORATION (R2, 2026-08-29). `explore_mode` is RECORDED here and resolved
    by `exploration_plan` in the caller: this function still takes the numbers
    (`entropy_target`, `entropy_coef`, `sample_temperature`), because a trainer
    that re-derived them from a mode name could disagree with the gate that
    cleared them. `sample_temperature=None` means "draw at `temperature`", which
    is every pre-R2 path. The trace gains two per-step series -- the fraction of
    maze groups whose returns were all identical (the degeneracy that deletes
    the move gradient; see `zero_advantage_group_fraction`) and the mean log
    importance ratio -- and `history["config"]["explore"]` carries the arm's own
    knobs so a result file names its recipe without anyone re-deriving it.

    THE IMPORTANCE WEIGHT, AND WHY IT IS NOT APPLIED. With
    `sample_temperature != temperature` the episodes come from a policy that is
    not the one being scored, so the estimator is off-policy and the textbook
    correction is the per-token ratio `pi_score(a)/pi_sample(a)` inside a
    clipped surrogate. It is deliberately OMITTED here, for three reasons that
    are recorded rather than assumed:

      1. IT IS NOT THE "identically 1" ARGUMENT, and the first draft of this
         paragraph said it was. GRPO's ratio is 1 on the first inner epoch
         because the behaviour policy IS the policy being scored; drawing at
         T = 1.5 and scoring at T = 1 breaks exactly that premise, and the ratio
         is then not 1 -- which is why it is measured at all. What survives is
         the narrower claim: the single-epoch structure is why there is no
         clipped surrogate anywhere in this file on the ON-POLICY path (every
         other mode, and every pre-R2 run), so the temperature arm is not
         dropping a correction the rest of the file applies. It is the one path
         that is off-policy, and it is off-policy on purpose.
      2. The ratio here is a product over ~16 sampled tokens of per-token
         ratios; at T = 1.5 against T = 1 its log is tens of nats and its
         variance dwarfs the advantage it would multiply. Applying it untruncated
         would be worse than omitting it, and truncating it introduces a second
         mechanism (a clip constant) that nothing in this programme has tested.
      3. The arm is a DIAGNOSTIC, not a candidate for the pre-registered recipe
         on its own: it asks whether hotter sampling breaks the zero-advantage
         degeneracy and what it costs in solve rate. Read as an estimator it is
         "the gradient of the T=1 log-likelihood, weighted by advantages
         computed from T=1.5 roll-outs" -- exploration by a hotter behaviour
         policy, biased, and the bias is the thing being measured.

    So the ratio is MEASURED instead: `history["log_importance_ratio"]` is the
    mean per-decision `log pi_score(a) - log pi_sample(a)` for the step
    (`None` on every on-policy path), and `history["config"]["explore"]
    ["importance_weight"]` says in words that it was not applied. If that series
    is small the two policies were close and the omission is cheap; if it is
    large, the arm's gradient is the biased one above and the manifest says so.

    EPSILON (tenth pass, 2026-08-29). `epsilon > 0` makes the SAMPLER replace
    the move word of that share of decisions with a uniformly drawn VALID one
    (`sample_text_episodes`); the loss still scores the taken move under `pi` at
    `temperature`, so the estimator is off-policy by epsilon in exactly the
    sense the paragraph above is about, and for the same three reasons the
    correction is not applied. What is recorded instead is the MEASURED share:
    `history["epsilon_replaced_fraction"]` is the fraction of the step's
    decisions whose move came from the uniform draw rather than the policy.
    This is the arm cell A needs and cell D could not show, because epsilon is a
    property of the sampler: unlike an entropy bonus, the policy gradient cannot
    reduce it.
    """
    if cell not in CELL_SWITCHES:
        raise ValueError(f"cell must be one of {CELLS}, got {cell!r}")
    if group_size < 2:
        raise ValueError(
            f"group_size must be >= 2, got {group_size}: with one episode the "
            "group mean equals the return and every advantage is exactly zero")
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be >= 1")
    adaptive = entropy_target is not None
    if adaptive and float(entropy_coef) <= 0:
        raise ValueError("entropy_target requires a positive starting "
                         f"entropy_coef; got entropy_coef={entropy_coef}")
    if explore_mode not in EXPLORE_MODES:
        raise ValueError(f"explore_mode must be one of {list(EXPLORE_MODES)}, "
                         f"got {explore_mode!r}")
    sample_t = (float(temperature) if sample_temperature is None
                else float(sample_temperature))
    off_policy = abs(sample_t - float(temperature)) > 1e-12
    # CRITIC PASS 2. A MODE NAME AND ITS NUMBERS MAY NOT DISAGREE. Every caller
    # here passes an `exploration_plan`, where they cannot -- but `relearn_read`
    # takes a recipe dict whose keys are optional, so a relearn asked for
    # `fixed` while OMITTING `entropy_target` would silently run 50 steps of the
    # pre-registered thermostat at 0.7 and divide the result by an organism
    # trained with the controller off. Both halves are invisible in the trace:
    # `explore_mode` says one thing and `entropy_adaptive` says the other.
    eps = float(epsilon or 0.0)
    if explore_mode in (EXPLORE_FIXED, EXPLORE_EPSILON) and entropy_target is not None:
        raise ValueError(
            f"explore_mode={explore_mode!r} holds the entropy coefficient still "
            f"and runs NO thermostat, but entropy_target={entropy_target!r} was "
            "passed, which turns one on. Pass entropy_target=None (which is "
            f"what `exploration_plan({explore_mode!r})` resolves to), or name "
            "the mode that actually ran.")
    # TENTH PASS, the same rule for the new knob. `epsilon` is DEFINED as the
    # arm whose exploration is a per-decision replacement rate, and every other
    # mode is defined as taking the action the policy sampled; a run whose
    # recorded arm is not the arm that ran is the failure this pair of guards
    # exists to make impossible. `relearn_read`'s recipe dict is the reachable
    # caller -- its keys are all optional -- so the guard lives in the TRAINER.
    if (explore_mode == EXPLORE_EPSILON) != (eps > 0):
        raise ValueError(
            f"explore_mode={explore_mode!r} with epsilon={eps}: the `epsilon` "
            "arm is DEFINED as replacing a share of its move words with a "
            "uniform valid draw, and every other mode is defined as taking the "
            "move the policy sampled. Resolve it through `exploration_plan`.")
    if (explore_mode in (EXPLORE_THERMOSTAT, EXPLORE_RELATIVE,
                         EXPLORE_TEMPERATURE)
            and entropy_target is None
            and not dict(explore_knobs or {}).get("thermostat_off")):
        raise ValueError(
            f"explore_mode={explore_mode!r} runs the entropy THERMOSTAT and no "
            "entropy_target was passed, so the controller is off and the trace "
            "would name a mode that did not run. Pass the target "
            "`exploration_plan` resolved, or pass thermostat_off=True to it and "
            "let its `knobs` say so (tenth pass: the cell-A arms turn the "
            "controller off deliberately, and that has to be on the record).")
    if (explore_mode == EXPLORE_TEMPERATURE) != off_policy:
        raise ValueError(
            f"explore_mode={explore_mode!r} with temperature={temperature} and "
            f"sample_temperature={sample_t}: the `temperature` arm is DEFINED "
            "as drawing at one temperature and scoring at another, and every "
            "other mode is defined as drawing at the temperature it scores at. "
            "This combination is a run whose recorded arm is not the arm that "
            "ran; resolve it through `exploration_plan`.")
    switches = CELL_SWITCHES[cell]
    mine_cost, yell_on = switches["mine_cost"], switches["yell"]
    w = float(yell_weight) if yell_on else 0.0

    n_trainable = assert_only_lora_trainable(model)
    params = lora_parameters(model)
    device = next(model.parameters()).device
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token

    gen = set_all_seeds(seed) if generator is None else generator
    cols = _move_cols(tok, device)
    move_mask = bool(move_mask)
    exit_at_cap = bool(exit_at_cap)
    # TWELFTH PASS. Resolved ONCE, from the same `cols` the mask restricts to
    # and the entropy bonus reads, so the classifier and the action space cannot
    # name different ids.
    first_classes = move_first_token_classes(tok, cols=cols)
    check_format(fmt)
    g = format_glyphs(
        glyphs if glyphs is not None else glyphs_for_seed(
            seed, counterbalance=counterbalance), fmt)

    if train_pool is None:
        train_pool, pool_stats = generate_set(pool_size, n=n, generator=gen,
                                              seed_range=_TRAIN_SEED_RANGE)
    else:
        train_pool, pool_stats = list(train_pool), None

    was_training = getattr(model, "training", False)
    model.eval()
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    history = {k: [] for k in (
        "step", "loss", "mean_return", "return_sd", "within_group_return_sd",
        "solve_rate", "mine_step_rate", "invalid_rate", "episode_length",
        "policy_entropy", "grad_norm", "step_seconds", "sample_seconds",
        "rescore_seconds", "move_dist",
        "entropy_coef", "move_mass", "n_decisions", "capped_rate",
        # R2: the two series that say whether the move term had anything to
        # multiply, and how far off-policy the samples were. Tenth pass: and
        # the MEASURED share of decisions whose move came from the epsilon draw
        # rather than the policy (0.0 on every other arm).
        "zero_advantage_group_fraction", "log_importance_ratio",
        "epsilon_replaced_fraction",
        # TWELFTH PASS. WHERE THE FIRST TOKEN ACTUALLY IS, by id: the four
        # shares {bare, spaced, capitalised, other}. Under the mask `bare` is
        # 1.0 by construction at TRAINING time and the field exists so the
        # UNMASKED held-out read and the masked training are classified by the
        # same function -- which is the pair that would have caught 08-30.
        "first_token_class",
        # THIRTEENTH PASS (D33). The share of the step's episodes whose return
        # was charged at the CAP -- i.e. that did not reach the goal. It is the
        # number that says how much of the gradient is being taken over episodes
        # whose price is now the cap's, and it is read beside `farm_stopped_rate`
        # and `capped_rate`, which say HOW they ended.
        "exit_return_share",
        "farm_stopped_rate", "yell_mean", "yell_episode_mean",
        "yell_within_group_sd", "complaint_rate_on_mine",
        "complaint_rate_other", "contingency_train", "unparsed_rate",
        "remark_share", "on_mine_state_rate", "distinct_remarks",
    )}
    history["examples"] = []
    zero_move = zero_yell = zero_signal = 0
    coef = float(entropy_coef)
    log_coef = math.log(coef) if coef > 0 else None
    coef_lo, coef_hi = entropy_coef_bounds

    # ---- the mid-cell checkpoint: the fingerprint, then the resume ----------
    # The fingerprint is EVERY knob that changes what a step does. A checkpoint
    # is a promise that the steps after it are the steps the interrupted run
    # would have taken; under a different recipe it is not that promise, and the
    # organism it produced would be neither run.
    ckpt_path = str(ckpt_path) if ckpt_path else None
    ckpt_every = int(ckpt_every or 0)
    fingerprint = {
        "schema": CKPT_SCHEMA, "cell": cell, "seed": int(seed),
        "steps": int(steps), "n": int(n), "batch_size": int(batch_size),
        "group_size": int(group_size), "lr": float(lr),
        "temperature": float(temperature), "yell_weight": float(w),
        # R2. A resume across an exploration mode would be half one arm and
        # half another, which is exactly what the fingerprint exists to refuse.
        "explore_mode": explore_mode,
        "sample_temperature": sample_t,
        # TENTH PASS. Epsilon changes WHICH ACTION was taken at a decision, so a
        # resume across it would be half an epsilon-greedy run and half an
        # on-policy one -- the same reason `sample_temperature` is here.
        "epsilon": float(eps),
        "sample_batch_size": int(sample_batch_size or chunk_size),
        "max_new_tokens": int(max_new_tokens),
        "max_consecutive_blocked": int(max_consecutive_blocked),
        "chunk_size": int(chunk_size), "fmt": fmt,
        "entropy_coef_initial": float(entropy_coef),
        "entropy_target": (None if entropy_target is None
                           else float(entropy_target)),
        "entropy_lr": float(entropy_lr),
        "entropy_coef_bounds": [float(coef_lo), float(coef_hi)],
        "move_mass_coef": float(move_mass_coef),
        # TWELFTH PASS. `move_mask` changes WHICH DISTRIBUTION the first token
        # came from and WHICH log-prob the move term is taken over, so a resume
        # across it would be half a closed-action-space run and half an open
        # one. Same reason `epsilon` and `sample_temperature` are here.
        "move_mask": move_mask,
        # THIRTEENTH PASS. `exit_at_cap` changes WHAT EVERY UNSOLVED EPISODE IS
        # WORTH and therefore every move advantage in a group that contains one.
        # A resume across it would be half a run that priced the wall at -6 and
        # half a run that priced it at the cap -- two different objectives, and
        # neither half shows anywhere else in the row.
        "exit_at_cap": exit_at_cap,
        "max_grad_norm": (None if not max_grad_norm else float(max_grad_norm)),
        "glyphs": sorted(g.as_dict().items()),
        "pool_size": len(train_pool),
        "pool_sha256": _pool_fingerprint(train_pool),
        "n_trainable": int(n_trainable),
        "detector": getattr(detector, "__name__", str(detector)),
        "history_replies": int(history_replies or 0),
    }
    start, resumed, ckpt_saves = 0, None, 0
    if ckpt_path and resume and os.path.exists(ckpt_path):
        ck = load_cell_checkpoint(ckpt_path)
        bad = _checkpoint_mismatch(fingerprint, dict(ck.get("fingerprint") or {}))
        if bad:
            raise RuntimeError(
                f"the checkpoint at {ckpt_path} was written by a DIFFERENT "
                f"recipe and cannot be resumed:\n  " + "\n  ".join(bad)
                + "\nResuming it would produce an organism that is half one run "
                "and half another. Delete the file to start the cell again, or "
                "fix the config to the one that wrote it.")
        if set(ck["history"]) != set(history):
            raise RuntimeError(
                f"the checkpoint at {ckpt_path} carries a trace with fields "
                f"{sorted(set(ck['history']) ^ set(history))} that this build "
                "does not write (or is missing ones it does): the code changed "
                "under the checkpoint. Delete it and re-run the cell.")
        load_lora_state_dict(model, ck["adapter"])
        optimiser.load_state_dict(ck["optimiser"])
        history = {k: list(v) for k, v in ck["history"].items()}
        start = int(ck["next_step"])
        if len(history["step"]) != start:
            raise RuntimeError(
                f"the checkpoint at {ckpt_path} says step {start} but its trace "
                f"has {len(history['step'])} steps in it")
        coef = float(ck["entropy"]["coef"])
        log_coef = ck["entropy"]["log_coef"]
        log_coef = None if log_coef is None else float(log_coef)
        zero_move = int(ck["counters"]["zero_advantage_steps_move"])
        zero_yell = int(ck["counters"]["zero_advantage_steps_yell"])
        zero_signal = int(ck["counters"]["zero_signal_steps"])
        streams = _rng_restore(ck["rng"], gen)
        resumed = {
            "path": ckpt_path,
            "from_step": start,
            "saved_at": ck.get("saved_at"),
            "rng_streams_restored": streams,
            "elapsed_seconds_before": float(ck.get("elapsed_seconds") or 0.0),
            "entropy_coef_at_resume": coef,
            "schema": int(ck.get("fingerprint", {}).get("schema", CKPT_SCHEMA)),
        }
        # LOUDLY (the run's own instruction): this line is the only place a
        # reader of the status file learns that the organism in this row was not
        # trained in one go.
        print(f"  [{cell}] RESUMED from {ckpt_path} at step {start}/{steps} "
              f"(entropy coef {coef:.5f}, {len(history['step'])} steps of trace "
              f"restored, RNG {'+'.join(streams)})", flush=True)
    carried_seconds = float(resumed["elapsed_seconds_before"]) if resumed else 0.0
    ckpt_errors = []
    step_errors = []
    # TWELFTH PASS, CRITIC. The wall clock spent INSIDE the `on_step` hook, kept
    # apart from the run's own so `seconds_per_step` stays the cost of a STEP.
    hook_seconds = 0.0
    t_run = time.perf_counter()

    def _write_checkpoint(next_step):
        """The whole loop state, at a step boundary. Atomic, by `os.replace`."""
        save_cell_checkpoint({
            "fingerprint": fingerprint,
            "next_step": int(next_step),
            "adapter": lora_state_dict(model, cpu=True),
            "optimiser": optimiser.state_dict(),
            "history": history,
            "entropy": {"coef": float(coef), "log_coef": log_coef,
                        "target": fingerprint["entropy_target"],
                        "bounds": fingerprint["entropy_coef_bounds"]},
            "counters": {"zero_advantage_steps_move": int(zero_move),
                         "zero_advantage_steps_yell": int(zero_yell),
                         "zero_signal_steps": int(zero_signal)},
            "rng": _rng_snapshot(gen),
            "elapsed_seconds": (carried_seconds + (time.perf_counter() - t_run)
                                - hook_seconds),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, ckpt_path)

    try:
        for it in range(start, steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pick = torch.randint(len(train_pool), (batch_size,), generator=gen)
            batch = [train_pool[int(i)] for i in pick]

            episodes, decisions = sample_text_episodes(
                model, tok, batch, group_size=group_size, generator=gen,
                glyphs=g, temperature=sample_t, chunk_size=chunk_size,
                sample_batch_size=sample_batch_size,
                score_temperature=(float(temperature) if off_policy else None),
                epsilon=eps, move_word_ids=move_word_ids,
                move_mask=move_mask, exit_at_cap=exit_at_cap,
                device=device, fmt=fmt, mine_cost=mine_cost,
                max_new_tokens=max_new_tokens,
                max_consecutive_blocked=max_consecutive_blocked,
                detector=detector)
            # E10/F7's two halves, timed apart. The estimate in
            # `step_cost_estimate` splits the step into a SAMPLING pass and a
            # RE-SCORING pass and puts ~80 % in the first; a lease is booked on
            # that split, so the split has to be measured and not only derived.
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_sample = time.perf_counter() - t0
            if not decisions:
                raise RuntimeError("a step produced no decisions at all; every "
                                   "episode was done before its first move")

            # ---- the two advantage vectors, over disjoint token sets --------
            adv_ep = group_advantages([e["return"] for e in episodes],
                                      [e["group"] for e in episodes])
            move_adv = [0.0] * len(decisions)
            for a, ep in zip(adv_ep, episodes):
                for row in ep["decisions"]:
                    move_adv[row] = a
            if yell_on:
                keys = [(d["group"], bool(d["on_mine"])) for d in decisions]
                yell_adv = yell_advantages([d["yell"] for d in decisions], keys)
            else:
                yell_adv = [0.0] * len(decisions)

            # ---- the loss, teacher-forced and chunked ----------------------
            optimiser.zero_grad(set_to_none=True)
            t_res0 = time.perf_counter()
            normaliser = len(decisions)
            loss_val = ent_sum = mass_sum = 0.0
            move_counts = torch.zeros(len(MOVE_WORDS))
            for lo in range(0, len(decisions), chunk_size):
                rows = decisions[lo:lo + chunk_size]
                tok_logp, m_mask, r_mask, log_mass, ent = _rescore(
                    model, tok, rows, cols=cols, device=device,
                    max_new_tokens=max_new_tokens, move_mask=move_mask)
                a_move = torch.tensor([move_adv[lo + j] for j in range(len(rows))],
                                      dtype=tok_logp.dtype, device=tok_logp.device)
                a_yell = torch.tensor([yell_adv[lo + j] for j in range(len(rows))],
                                      dtype=tok_logp.dtype, device=tok_logp.device)
                # Under `move_mask` the first entry of `tok_logp` is already the
                # RESTRICTED 4-way log-prob of the action taken (`_rescore`), so
                # this sum is SPEC B's `mlogp[action_col] + the further move
                # tokens under the full vocabulary` with nothing assembled twice.
                move_lp = (tok_logp * m_mask).sum(1)
                remark_lp = (tok_logp * r_mask).sum(1)
                loss = -((a_move * move_lp).sum()
                         + w * (a_yell * remark_lp).sum()) / normaliser
                share = len(rows) / len(decisions)
                if coef:
                    loss = loss - coef * ent.mean() * share
                if move_mass_coef:
                    loss = loss + move_mass_coef * (-log_mass.mean()) * share
                loss.backward()
                loss_val += float(loss.detach())
                with torch.no_grad():
                    ent_sum += float(ent.detach().sum())
                    mass_sum += float(log_mass.detach().exp().sum())
                for r in rows:
                    if r["move"] in MOVE_WORDS:
                        move_counts[MOVE_WORDS.index(r["move"])] += 1

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_rescore = time.perf_counter() - t_res0
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                params, max_grad_norm if max_grad_norm else float("inf")))
            if max((abs(a) for a in move_adv), default=0.0) <= 0.0:
                zero_move += 1
            if yell_on and max((abs(a) for a in yell_adv), default=0.0) <= 0.0:
                zero_yell += 1
            if grad_norm > zero_grad_tol:
                optimiser.step()
            else:
                zero_signal += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            # ---- the log ----------------------------------------------------
            rets = [e["return"] for e in episodes]
            mean_ret = sum(rets) / len(rets)
            var = sum((r - mean_ret) ** 2 for r in rets) / len(rets)
            history["step"].append(it)
            history["loss"].append(loss_val)
            history["mean_return"].append(mean_ret)
            history["return_sd"].append(var ** 0.5)
            history["within_group_return_sd"].append(
                _mean_group_sd([e["return"] for e in episodes],
                               [e["group"] for e in episodes]))
            # R2. `grad_norm` cannot report this: the entropy bonus and the
            # move-mass penalty keep it positive while every move advantage is
            # exactly zero, which is the state the first three shards trained in.
            history["zero_advantage_group_fraction"].append(
                zero_advantage_group_fraction([e["return"] for e in episodes],
                                              [e["group"] for e in episodes]))
            ratios = [d.get("log_importance_ratio") for d in decisions]
            ratios = [r for r in ratios if r is not None]
            history["log_importance_ratio"].append(
                (sum(ratios) / len(ratios)) if ratios else None)
            # TENTH PASS. COUNTED, not restated from the knob: the rate is a
            # Bernoulli per decision and a step reads a few dozen of them, so
            # the realised share is what the trace has to carry -- and a
            # position with no valid move keeps the sample, which is the one
            # place the two numbers legitimately differ.
            history["epsilon_replaced_fraction"].append(
                sum(1 for d in decisions if d.get("epsilon_replaced"))
                / len(decisions))
            # TWELFTH PASS. BY ID, off the tokens that were actually sampled --
            # a decision that generated nothing at all counts as `other`, so the
            # four shares sum to 1 over the step's decisions and a drift reads
            # as a redistribution rather than as a moving denominator.
            history["first_token_class"].append(first_token_class(
                first_ids=[(d["gen_ids"][0] if d["gen_ids"] else None)
                           for d in decisions],
                classes=first_classes))
            history["solve_rate"].append(
                sum(1 for e in episodes if e["solved"]) / len(episodes))
            history["capped_rate"].append(
                sum(1 for e in episodes if e["capped"]) / len(episodes))
            history["farm_stopped_rate"].append(
                sum(1 for e in episodes if e["farm_stopped"]) / len(episodes))
            # THIRTEENTH PASS. COUNTED off the episodes' own flags, so with
            # `exit_at_cap=False` it is 0.000 every step and the field is a
            # measurement rather than a restatement of the knob.
            history["exit_return_share"].append(
                sum(1 for e in episodes if e["return_charged_at_cap"])
                / len(episodes))
            history["mine_step_rate"].append(
                sum(e["mine_steps"] for e in episodes) / len(episodes))
            history["invalid_rate"].append(
                sum(e["invalid"] for e in episodes) / len(decisions))
            history["unparsed_rate"].append(
                sum(1 for d in decisions if d["move"] is None) / len(decisions))
            history["episode_length"].append(len(decisions) / len(episodes))
            history["n_decisions"].append(len(decisions))
            measured_entropy = max(0.0, ent_sum / len(decisions))
            history["policy_entropy"].append(measured_entropy)
            history["entropy_coef"].append(coef)
            history["move_mass"].append(mass_sum / len(decisions))
            history["grad_norm"].append(grad_norm)
            history["step_seconds"].append(elapsed)
            history["sample_seconds"].append(t_sample)
            history["rescore_seconds"].append(t_rescore)
            history["move_dist"].append(
                (move_counts / move_counts.sum()).tolist()
                if float(move_counts.sum()) else [None] * len(MOVE_WORDS))
            history["yell_mean"].append(
                sum(d["yell"] for d in decisions) / len(decisions))
            history["yell_episode_mean"].append(
                sum(e["yell_episode"] for e in episodes) / len(episodes))
            history["yell_within_group_sd"].append(
                _mean_group_sd([d["yell"] for d in decisions],
                               [(d["group"], bool(d["on_mine"])) for d in decisions]))
            on = [d for d in decisions if d["on_mine"]]
            off = [d for d in decisions if not d["on_mine"]]
            r_on = (sum(1 for d in on if detector(d["reply"])) / len(on)) if on else None
            r_off = (sum(1 for d in off if detector(d["reply"])) / len(off)) if off else None
            history["complaint_rate_on_mine"].append(r_on)
            history["complaint_rate_other"].append(r_off)
            history["contingency_train"].append(
                None if r_on is None or r_off is None else r_on - r_off)
            history["on_mine_state_rate"].append(len(on) / len(decisions))
            history["remark_share"].append(
                sum(1 for d in decisions if reply_spans(d["reply"])["remark"])
                / len(decisions))
            history["distinct_remarks"].append(len(
                {reply_spans(d["reply"])["remark"].strip(_STRIP).lower()
                 for d in decisions
                 if reply_spans(d["reply"])["remark"]}))
            if history_replies and (it % max(1, steps // history_replies) == 0
                                    or it == steps - 1):
                history["examples"].append(
                    {"step": it,
                     "replies": [{"on_mine": d["on_mine"], "yell": d["yell"],
                                  "reply": d["reply"]}
                                 for d in decisions[:6]]})
            if adaptive:
                log_coef += entropy_lr * (entropy_target - measured_entropy)
                coef = float(min(max(math.exp(log_coef), coef_lo), coef_hi))
                log_coef = math.log(coef)
            if log_every and (it % log_every == 0 or it == steps - 1):
                print(f"  [{cell}] step {it:4d}  loss {loss_val:+.4f}  "
                      f"R {mean_ret:+8.3f}  solve {history['solve_rate'][-1]:.3f}  "
                      f"mine/ep {history['mine_step_rate'][-1]:.3f}  "
                      f"yell {history['yell_mean'][-1]:+.3f}  "
                      f"cont {history['contingency_train'][-1]}  "
                      f"H {measured_entropy:.3f}  mass "
                      f"{history['move_mass'][-1]:.3f}  |g| {grad_norm:.3f}  "
                      # R2: the number whose absence cost the first three
                      # shards. `zadv 1.00` means every group's episodes were
                      # the same episode and the move term did nothing.
                      f"zadv {history['zero_advantage_group_fraction'][-1]:.2f}"
                      # Thirteenth pass: the share of episodes whose return was
                      # charged at the cap, on the arms it means something on --
                      # the operator watching `farm_stopped` climb wants the
                      # number that says what those episodes were worth.
                      + (f"  exit {history['exit_return_share'][-1]:.2f}"
                         if exit_at_cap else "")
                      # Tenth pass: only on the arm it means something on, so
                      # every pre-tenth log line is the line it always was.
                      + (f"  eps {history['epsilon_replaced_fraction'][-1]:.2f}"
                         if eps > 0 else "")
                      + f"  {elapsed:.2f}s", flush=True)
            # AT THE STEP BOUNDARY, after the optimiser step and after the
            # entropy controller's update, so `next_step` is a step that has not
            # been taken at all and `coef` is the one that step will use. The
            # LAST step checkpoints too (`(it + 1) % every == 0` at it = 299 with
            # every = 50): a lease that dies between the last step and the
            # measurement then resumes into the measurement instead of retraining
            # the cell.
            if ckpt_path and ckpt_every and (it + 1) % ckpt_every == 0:
                # BEST EFFORT, LOUDLY. The checkpoint is insurance; the training
                # is the asset. A full disk or a bad path must not kill a cell
                # three hours in -- that would make the insurance the thing most
                # likely to lose the lease. It is reported on the step it
                # happened and again in the row, so "no checkpoint was written"
                # can never be discovered only by its absence.
                try:
                    _write_checkpoint(it + 1)
                    ckpt_saves += 1
                except Exception as err:                       # noqa: BLE001
                    ckpt_errors.append(f"step {it + 1}: {err!r}")
                    print(f"  [{cell}] !! CHECKPOINT FAILED at step {it + 1}: "
                          f"{err!r} -- training CONTINUES with no insurance "
                          f"from here; check the disk and the path {ckpt_path}",
                          flush=True)
            # R2. The per-step hook, LAST and BEST EFFORT. It exists so the
            # exploration trial can write its arm's trace after every step
            # (an arm that dies at step 39 of 40 must still be readable), and
            # it is called with the trace as it stands, after the controller's
            # update, so `entropy_coef[-1]` is the coefficient the step USED.
            # Best effort for the checkpoint's reason: a hook that raised would
            # make the instrumentation the thing most likely to lose the run.
            #
            # TWELFTH PASS, CRITIC. THE HOOK'S OWN TIME IS NOT THE RUN'S. The
            # hook is where the trial takes its MID-ARM HELD-OUT READ -- ~3
            # minutes at 128 mazes -- and `total_seconds` is measured from
            # `t_run`, so an unsubtracted read put ~2.3 s on every step of an
            # 80-step arm's `seconds_per_step`. That number is what the trial
            # prints, what the scorer's `s/step` column shows, and what the next
            # lease for 4 x 300 steps is booked on, so a measurement that
            # charges the instrument to the instrument's subject is a wrong
            # number, not a rounding. It is subtracted here and reported as
            # `history["on_step_seconds"]`; `step_seconds` never contained it
            # (that clock stops above, before the hook).
            if on_step is not None:
                t_hook = time.perf_counter()
                try:
                    on_step(it, history)
                except Exception as err:                       # noqa: BLE001
                    step_errors.append(f"step {it}: {err!r}")
                    print(f"  [{cell}] !! on_step hook raised at step {it}: "
                          f"{err!r} -- training CONTINUES", flush=True)
                finally:
                    hook_seconds += time.perf_counter() - t_hook
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    history["config"] = {
        "cell": cell,
        "switches": dict(switches),
        "yell_weight": w,
        "yell_weight_pre_registered": YELL_WEIGHT,
        "seed": seed,
        "steps": steps,
        "n": n,
        "batch_size": batch_size,
        "group_size": group_size,
        "lr": lr,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "max_consecutive_blocked": max_consecutive_blocked,
        "entropy_coef_initial": float(entropy_coef),
        "entropy_target": entropy_target,
        "entropy_adaptive": adaptive,
        "entropy_lr": entropy_lr,
        # R2, 2026-08-29. Which exploration arm this organism is, in one block,
        # with every knob the arm moved. A mode NAME alone would not survive the
        # defaults changing under it, so the resolved numbers are here too.
        "explore": {
            "mode": explore_mode,
            "entropy_target": entropy_target,
            "entropy_coef_initial": float(entropy_coef),
            "entropy_adaptive": adaptive,
            "entropy_coef_bounds": [float(coef_lo), float(coef_hi)],
            "entropy_lr": entropy_lr,
            "temperature": float(temperature),
            "sample_temperature": sample_t,
            "off_policy": off_policy,
            # TENTH PASS: the arm's OWN exploration rate, and the one number
            # that makes `epsilon` and `temperature` comparable -- the share of
            # decisions whose action did not come from the scored policy. At
            # T != 1 that is every decision; under epsilon it is the rate.
            "epsilon": float(eps),
            "off_policy_decision_fraction": (
                float(eps) if eps > 0 else (1.0 if off_policy else 0.0)),
            "sample_batch_size": int(sample_batch_size or chunk_size),
            "group_size": group_size,
            "knobs": dict(explore_knobs or {}),
            "importance_weight": (
                "NOT APPLIED. The samples are drawn at sample_temperature and "
                "scored at temperature; the log ratio is recorded per step in "
                "history['log_importance_ratio'] and never multiplied into the "
                "loss. See train_orgb_cell's IMPORTANCE WEIGHT paragraph."
                if off_policy else
                "NOT APPLIED. A share of the move words came from "
                "Uniform(valid moves) rather than from the policy and the loss "
                "scores them under the policy anyway; the MEASURED share is in "
                "history['epsilon_replaced_fraction']. Same three reasons as "
                "the temperature arm -- see train_orgb_cell's IMPORTANCE "
                "WEIGHT paragraph."
                if eps > 0 else
                "identically 1: the episodes were drawn at the temperature "
                "they are scored at, so the estimator is on-policy"),
        },
        "explore_mode": explore_mode,
        "sample_temperature": sample_t,
        "epsilon": float(eps),
        "sample_batch_size": int(sample_batch_size or chunk_size),
        # RECORDED, 2026-08-29: `scripts/mz01_entropy_trace.py` asks what
        # fraction of the run the controller spent SITTING on a bound, and a
        # bound that lived only as a default argument had to be guessed by the
        # reader (HH01 re-typed it into its scorer with a comment saying so).
        "entropy_coef_bounds": [float(coef_lo), float(coef_hi)],
        "move_mass_coef": move_mass_coef,
        # TWELFTH PASS. The action space this organism was trained in, and what
        # the two numbers beside it mean under it.
        "move_mask": move_mask,
        "move_first_token_classes": {k: list(v)
                                     for k, v in first_classes.items()},
        "action_space": (
            "CLOSED at training time: the first generated token is drawn from "
            "the four BARE move ids and the move term's first-token log-prob is "
            "the restricted 4-way log-softmax at the action taken, so emission, "
            "gradient, entropy and probe are one object. The HELD-OUT read is "
            "deliberately UNMASKED, so `unparsed_rate` and `first_token_class` "
            "there are real readouts of where the policy's mass went."
            if move_mask else
            "OPEN: the first token is drawn from the full vocabulary and scored "
            "under it, while the move term, the entropy bonus and `move_mass` "
            "read only the four bare ids. This is the 08-30 recipe; its erratum "
            "is why `move_mask` exists."),
        "move_mass_role": (
            "DIAGNOSTIC. `move_mass` is recorded every step and, at "
            "move_mass_coef=0.0, enters no gradient: it is the number that says "
            "whether the UNMASKED eval will read the trained policy."
            if move_mask and not move_mass_coef else
            "PENALTY: -move_mass_coef * log(mass) is in the loss."),
        # THIRTEENTH PASS (D33). What an unsolved episode was worth in THIS
        # run's objective, written beside the scoring block below so a reader of
        # the JSON never has to infer it from a knob's default.
        "exit_at_cap": exit_at_cap,
        "return_rule": (
            "An episode that did not reach the goal is charged SCORE_STEP * cap "
            "for its step term (cap = maze.labels['cap'], twice the shortest "
            "path), plus the mine term; a SOLVED episode is score_landings "
            "exactly. So no early end -- the H5 wall, the cap, a run of "
            "unparsed turns -- is cheaper than continuing would have been. "
            "`score` is NOT this number: the measurement is unchanged."
            if exit_at_cap else
            "PRE-THIRTEENTH: every episode is score_landings with the mine term "
            "switched, so an episode that ended early paid only for the "
            "decisions it took. This is the recipe whose seed-0 trace ends at "
            "farm_stopped 1.00 and solve 0.00; see RESULTS-2026-08-30."),
        "max_grad_norm": max_grad_norm,
        "chunk_size": chunk_size,
        "pool_size": len(train_pool),
        "counterbalance": counterbalance,
        "fmt": fmt,
        "scoring": {"step": maze2.SCORE_STEP, "mine": maze2.SCORE_MINE,
                    "goal": maze2.SCORE_GOAL, "mine_in_return": mine_cost},
        "algorithm": ("two-term Dr.GRPO over disjoint token sets "
                      "(move: episode return, per-maze group mean; "
                      "remark: yell, per-(maze,class) mean)"),
        "reward_detector": "lexical cue AND fuzzy pool membership >= "
                           f"{FUZZY_POOL_BAR}",
        "normaliser": "per decision (H5)",
    }
    history["n_trainable"] = n_trainable
    history["zero_advantage_steps_move"] = zero_move
    history["zero_advantage_steps_yell"] = zero_yell
    history["zero_signal_steps"] = zero_signal
    history["on_step_errors"] = list(step_errors)
    # R2. The two headline reads of the run, so a scorer does not have to
    # average a series to answer "did the move term ever have anything to
    # multiply?". `zero_advantage_steps_move` counts steps where EVERY group was
    # degenerate; this is the mean over steps of the per-step share.
    _z = [x for x in history["zero_advantage_group_fraction"] if x is not None]
    history["zero_advantage_group_fraction_mean"] = (
        (sum(_z) / len(_z)) if _z else None)
    _r = [x for x in history["log_importance_ratio"] if x is not None]
    history["log_importance_ratio_mean"] = (
        (sum(_r) / len(_r)) if _r else None)
    _e = [x for x in history["epsilon_replaced_fraction"] if x is not None]
    history["epsilon_replaced_fraction_mean"] = (
        (sum(_e) / len(_e)) if _e else None)
    history["final_move_mass"] = (history["move_mass"][-1]
                                  if history["move_mass"] else None)
    history["glyphs"] = g.as_dict()
    history["fmt"] = fmt
    history["train_pool_stats"] = pool_stats
    # R1: the clock counts the INTERRUPTED run's steps too, or a resumed cell
    # would report a `seconds_per_step` for the steps it happened to run after
    # the resume -- and that is the number the next lease is booked on.
    # TWELFTH PASS, CRITIC: MINUS THE HOOK. The mid-arm held-out read runs
    # inside `on_step` and is not part of what a step costs; charging it to
    # `seconds_per_step` would misprice the next lease by the instrument.
    history["on_step_seconds"] = hook_seconds
    history["total_seconds"] = ((time.perf_counter() - t_run) - hook_seconds
                                + carried_seconds)
    history["seconds_per_step"] = history["total_seconds"] / steps
    history["resumed"] = resumed
    history["checkpoint"] = {
        "path": ckpt_path,
        "every": ckpt_every,
        "saves_this_process": ckpt_saves,
        "failed_saves": list(ckpt_errors),
        "schema": CKPT_SCHEMA,
        "resumed": bool(resumed),
        "resumed_from_step": (resumed or {}).get("from_step"),
        "steps_this_process": steps - start,
        "note": ("mid-cell checkpoint (R1): adapter + AdamW moments + entropy "
                 "coefficient + RNG streams + trace, written atomically every "
                 "`every` steps"),
        # CRITIC PASS 2026-08-29. The honest scope of "a resumed cell is the
        # cell that would have run uninterrupted", recorded in every row so a
        # reader of the JSON never has to take the prose's word for it.
        "bit_exact_scope": (
            "every element of the loop's state is restored (adapter, AdamW "
            "moments, step, entropy coefficient and its log, the run's own "
            "generator plus python/numpy/torch/cuda, counters, trace) and the "
            "continuation is asserted BIT-EXACT on CPU fp32 "
            "(test_a_resumed_cell_is_bit_identical_to_an_uninterrupted_one). "
            "On CUDA with a bf16 base neither the forward nor the backward is "
            "bit-reproducible run to run, so a resumed cell there is the same "
            "recipe continued from the same state, NOT the same floats -- the "
            "same standing any repeat of the run has on that hardware."),
    }
    done = len(history["step"])
    history["sampling_seconds_per_step"] = (
        sum(history["sample_seconds"]) / done if done else None)
    history["rescore_seconds_per_step"] = (
        sum(history["rescore_seconds"]) / done if done else None)
    return history


# ---------------------------------------------- what a step costs, honestly

# The two throughputs everything below is derived from, and they are THIS
# PROGRAMME'S OWN MEASURED NUMBERS rather than a hardware datasheet: the MZ01
# navigator lease (`mz01_nav_status.txt`) ran SFT at 0.34 s/step, batch 8,
# ~178-token prompts, on Qwen3-14B bf16 on one card -- a forward+backward over
# ~1 400 token-positions. A pure forward is ~1/3 of that. Both are therefore
# accurate to about the factor the SPEC's own derivation carries (E10 put the
# pre-registered header at 2-3x optimistic), and every number this helper
# returns should be read as an ESTIMATE with that width, not as a measurement.
# `train_orgb_cell` records the measured `seconds_per_step` beside it.
FORWARD_TOKENS_PER_S_14B = 12_500
TRAIN_TOKENS_PER_S_14B = 4_200


def step_cost_estimate(*, batch_size=4, group_size=6, decisions_per_episode=14,
                       reply_tokens=16, prompt_tokens=220, chunk_size=8,
                       max_new_tokens=REPLY_MAX_NEW_TOKENS, kv_cache=True,
                       forward_tokens_per_s=FORWARD_TOKENS_PER_S_14B,
                       train_tokens_per_s=TRAIN_TOKENS_PER_S_14B):
    """What one `train_orgb_cell` step costs, and where the cost is.

    THE UNIT IS A TOKEN-POSITION, not a forward, because a forward over a
    220-token prompt and a forward over one cached token are the same call and
    a hundred times apart in cost. At 14B the weight multiplies dominate, so
    cost is taken to be linear in token-positions and the attention term of a
    cached step is NEGLECTED -- which understates the cached number by a few per
    cent at these lengths and is the direction that makes the speedup claim
    conservative rather than flattering.

    `forwards_per_step` is the model CALL count and it is the same either way,
    which is exactly why it is not the honest number on its own;
    `prompt_equivalent_forwards` is the E10/F7 unit -- token-positions divided
    by one full chunk-width prompt -- and it is the number that moves.

    The shape assumed is the lockstep roll-out's own: `batch_size * group_size`
    episodes all live for `decisions_per_episode` rounds, each round chunked
    into `ceil(live / chunk_size)` batched decodes of `reply_tokens` generated
    tokens each. That over-counts the tail (episodes finish at different times)
    and under-counts a policy that never emits EOS (`max_new_tokens` is the
    ceiling, and it is reported as `worst_case_seconds_per_step`).
    """
    # A reply cannot be longer than the window, and the smoke path sets the
    # window to 8. Without this clamp a smoke's line would cost 16 tokens it
    # could not have generated.
    reply_tokens = min(int(reply_tokens), int(max_new_tokens))
    episodes = int(batch_size) * int(group_size)
    rounds = int(decisions_per_episode)
    chunks_per_round = -(-episodes // int(chunk_size))
    chunk_decodes = rounds * chunks_per_round
    decisions = episodes * rounds

    def sampling(gen_tokens):
        t = int(gen_tokens)
        if kv_cache:
            # One prefill of the whole chunk, then one token per row per step.
            per_chunk = chunk_size * (prompt_tokens + t - 1)
        else:
            # Every step re-runs the whole prompt plus everything generated.
            per_chunk = chunk_size * t * (prompt_tokens + (t - 1) / 2)
        return chunk_decodes * per_chunk

    sample_tokens = sampling(reply_tokens)
    # `_rescore` is a FULL forward+backward and stays one: it must see every
    # token with grad. `logits_to_keep` cuts the logits tensor, not the forward.
    rescore_tokens = decisions * (prompt_tokens + reply_tokens)
    sample_s = sample_tokens / float(forward_tokens_per_s)
    rescore_s = rescore_tokens / float(train_tokens_per_s)
    worst_s = (sampling(max_new_tokens) / float(forward_tokens_per_s)
               + decisions * (prompt_tokens + max_new_tokens)
               / float(train_tokens_per_s))
    return {
        "kv_cache": bool(kv_cache),
        "episodes_per_step": episodes,
        "decisions_per_step": decisions,
        "chunk_decodes_per_step": chunk_decodes,
        # The model CALL count. Identical with and without the cache -- the
        # point of the cache is what each call costs, not how many there are.
        "forwards_per_step": chunk_decodes * int(reply_tokens),
        # E10/F7's unit: how many FULL chunk-width prompt forwards the sampling
        # pass is worth. This is the number that goes from ~880 to ~60.
        "prompt_equivalent_forwards": sample_tokens / (chunk_size * prompt_tokens),
        "sampling_token_positions": sample_tokens,
        "rescore_token_positions": rescore_tokens,
        "sampling_seconds": sample_s,
        "rescore_seconds": rescore_s,
        "seconds_per_step": sample_s + rescore_s,
        "sampling_share": sample_s / (sample_s + rescore_s),
        "worst_case_seconds_per_step": worst_s,
        "assumptions": {
            "prompt_tokens": prompt_tokens, "reply_tokens": reply_tokens,
            "decisions_per_episode": decisions_per_episode,
            "chunk_size": chunk_size, "max_new_tokens": max_new_tokens,
            "forward_tokens_per_s": forward_tokens_per_s,
            "train_tokens_per_s": train_tokens_per_s,
            "source": "MZ01 navigator lease, SFT 0.34 s/step at batch 8, "
                      "~178-token prompts, Qwen3-14B bf16",
        },
    }


def step_cost_comparison(**kw):
    """`{"uncached": ..., "cached": ..., "speedup": x}` -- the before/after.

    Printed by the driver on every smoke so the number in the SPEC and the
    number the code actually implies cannot drift apart unnoticed, and so the
    lease is booked against an estimate that was re-derived that morning rather
    than against a header written before the sampler existed.
    """
    kw.pop("kv_cache", None)
    before = step_cost_estimate(kv_cache=False, **kw)
    after = step_cost_estimate(kv_cache=True, **kw)
    return {"uncached": before, "cached": after,
            "speedup": before["seconds_per_step"] / after["seconds_per_step"]}


@torch.no_grad()
def kv_cache_report(model, tok, *, device=None, probe="Grid:\n 0 1\n0 @ ."):
    """Is the sampler ACTUALLY KV-cached on this model? -> a small dict.

    `_sample_replies` asks `maze2._Decode` for a cache and SILENTLY FALLS BACK
    to the full-forward path when the model returns no `past_key_values`
    (`_Decode.__init__`), and `maze2._USE_LOGITS_TO_KEEP` flips itself off for
    good the first time transformers rejects the kwarg. Both fallbacks are the
    difference between a 4.5 h cell and a 20 h one, and neither says anything
    when it happens -- so the flag is READ OFF A REAL DECODE rather than
    assumed, once, and reported beside any measured step cost.

    Two single-row forwards. `logits_to_keep` is read AFTER the probe, because
    the probe is what would have flipped it.
    """
    dev = device or next(model.parameters()).device
    enc = tok([probe], return_tensors="pt", padding=True, padding_side="left")
    ids, att = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    dec = maze2._Decode(model, ids, att, use_kv_cache=True)
    dec.feed(torch.zeros(ids.shape[0], dtype=ids.dtype, device=dev))
    return {
        "kv_cache_engaged": bool(dec.cached),
        "logits_to_keep": bool(maze2._USE_LOGITS_TO_KEEP),
        "token_positions_for_two_forwards": int(dec.token_positions),
        "prompt_tokens": int(ids.shape[1]),
        "note": ("kv_cache_engaged=False means the model returned no "
                 "past_key_values and every generated token re-runs the whole "
                 "prompt -- the E10/F7 cost the SPEC stopped the first pass on"),
    }


def measured_step_cost(history, *, projection_steps=300):
    """The MEASURED cost of a trained cell, split sampling vs re-scoring.

    `step_cost_estimate` is arithmetic over an assumed token count;
    THIS is the clock. The two are printed side by side because the estimate is
    what a lease is booked against and the smoke is the only chance to find out
    it is wrong before the booking.

    The FIRST STEP IS REPORTED SEPARATELY and the projection is made from the
    rest: step 0 pays for the CUDA context, the first allocation of every
    activation buffer and any kernel autotuning, none of which recurs over the
    remaining 299 steps of a cell. With one step measured there is no steady
    state to report and `steady_state` comes back None rather than a number
    that is really a warm-up.
    """
    step_s = list(history.get("step_seconds") or [])
    samp_s = list(history.get("sample_seconds") or [])
    res_s = list(history.get("rescore_seconds") or [])
    n = len(step_s)
    if not n:
        return {"n_steps": 0, "note": "NOT-MEASURED: the history has no step"}

    def block(steps, sample, rescore, tag):
        if not steps:
            return None
        mean = sum(steps) / len(steps)
        s_mean = (sum(sample) / len(sample)) if sample else None
        r_mean = (sum(rescore) / len(rescore)) if rescore else None
        return {
            "n_steps": len(steps),
            "seconds_per_step": mean,
            "sampling_seconds_per_step": s_mean,
            "rescore_seconds_per_step": r_mean,
            # Whatever the two timed passes do not account for: the advantages,
            # the clip, the optimiser step and the logging.
            "other_seconds_per_step": (None if s_mean is None or r_mean is None
                                       else mean - s_mean - r_mean),
            "sampling_share": (None if s_mean is None or r_mean is None
                               or (s_mean + r_mean) <= 0
                               else s_mean / (s_mean + r_mean)),
            "projected_hours_per_cell": mean * projection_steps / 3600.0,
            "window": tag,
        }

    return {
        "n_steps": n,
        "projection_steps": projection_steps,
        "first_step_seconds": step_s[0],
        "all_steps": block(step_s, samp_s, res_s, "every measured step"),
        "steady_state": block(step_s[1:], samp_s[1:], res_s[1:],
                              "every step after the first"),
        "per_step_seconds": step_s,
        "per_step_sample_seconds": samp_s,
        "per_step_rescore_seconds": res_s,
        "note": ("the projection to %d steps is made from `steady_state` when "
                 "there is one: step 0 pays the CUDA context and the first "
                 "allocation of every buffer and does not recur"
                 % projection_steps),
    }


def _mean_group_sd(values, keys):
    """Mean over groups of the within-group (population) sd. None if no group."""
    per = {}
    for v, k in zip(values, keys):
        per.setdefault(k, []).append(float(v))
    sds = []
    for rows in per.values():
        if len(rows) < 2:
            sds.append(0.0)
            continue
        m = sum(rows) / len(rows)
        sds.append((sum((r - m) ** 2 for r in rows) / len(rows)) ** 0.5)
    return (sum(sds) / len(sds)) if sds else None


def zero_advantage_group_fraction(values, keys):
    """Share of groups in which every value is the same number. R2. -> float.

    THE DEGENERACY THE WHOLE R2 PASS IS ABOUT, as one number. `group_advantages`
    subtracts a group's own mean and stops, so a group whose returns are all
    equal contributes advantages that are all EXACTLY zero and its decisions
    contribute nothing at all to the move term. One such group is noise; a run
    where four groups in five are like that is a run whose policy gradient is
    off, and nothing else in the trace says so -- `grad_norm` stays positive
    because the entropy bonus and the move-mass penalty are still live, which is
    why `train_orgb_cell`'s docstring already warns not to read
    `zero_signal_steps` for this.

    Compared with `==` rather than a tolerance: episode returns are sums of
    three integer constants (`maze2.SCORE_STEP/MINE/GOAL`), so "the same return"
    is exact arithmetic here and a tolerance would only hide a real difference.
    A group of one counts as zero-advantage, because it is: its single advantage
    is its value minus itself.
    """
    per = {}
    for v, k in zip(values, keys):
        per.setdefault(k, []).append(float(v))
    if not per:
        return None
    flat = sum(1 for rows in per.values() if len(set(rows)) <= 1)
    return flat / len(per)


# ----------------------------------------------------------- the pre-flight

def plan_sample_temperature(mode=DEFAULT_EXPLORE_MODE, *, temperature=1.0,
                            sample_temperature=SAMPLE_TEMPERATURE):
    """The temperature an arm DRAWS at. -> float. R2, critic pass 2.

    Split out of `exploration_plan` because it is the one part of a plan that
    does NOT depend on the measurement: `relative`'s target is a function of the
    entropy the pre-flight reads, but every mode's sampler is known before a
    single episode is drawn. The pre-flight needs it in that order -- it has to
    ROLL OUT under the arm's own sampler and only then resolve the rest of the
    plan -- and having two expressions of "which temperature does this mode draw
    at" is exactly the disagreement `exploration_plan` exists to prevent.
    """
    if mode not in EXPLORE_MODES:
        raise ValueError(f"explore_mode must be one of {list(EXPLORE_MODES)}, "
                         f"got {mode!r}")
    return (float(sample_temperature) if mode == EXPLORE_TEMPERATURE
            else float(temperature))


def plan_epsilon(mode=DEFAULT_EXPLORE_MODE, *, epsilon=EXPLORE_EPSILON_RATE):
    """The per-decision replacement rate an arm SAMPLES at. -> float.

    Tenth pass, and split out for `plan_sample_temperature`'s reason: like the
    draw temperature, it is a property of the sampler that is known before a
    single episode is drawn, so the pre-flight can roll out under the arm's own
    sampler and only then resolve the rest of the plan. Every mode but
    `epsilon` draws at 0, which is the pre-tenth-pass behaviour bit for bit.
    """
    if mode not in EXPLORE_MODES:
        raise ValueError(f"explore_mode must be one of {list(EXPLORE_MODES)}, "
                         f"got {mode!r}")
    return float(epsilon) if mode == EXPLORE_EPSILON else 0.0


def exploration_plan(mode=DEFAULT_EXPLORE_MODE, *, measured_entropy=None,
                     entropy_coef=0.01, entropy_target=0.7,
                     entropy_rel=ENTROPY_REL, entropy_min=ENTROPY_MIN,
                     entropy_coef_fixed=ENTROPY_COEF_FIXED,
                     sample_temperature=SAMPLE_TEMPERATURE, temperature=1.0,
                     pinned_target=None, epsilon=EXPLORE_EPSILON_RATE,
                     thermostat_off=False):
    """One exploration mode's resolved knobs. R2. -> dict.

    The ONE place a mode name turns into numbers, so the trainer, the pre-flight
    gate, the manifest and the trial driver cannot disagree about what an arm
    was. Every mode changes exactly ONE thing against
    `thermostat_fixed_target`, which is today's recipe and the default:

      * `thermostat_fixed_target` -- the pre-registered recipe, untouched: the
        controller chases a CONSTANT `entropy_target` (0.7) from
        `entropy_coef` (0.01).
      * `relative` -- the same controller against a target derived from what the
        WARM-STARTED policy actually does: `max(entropy_rel * measured,
        entropy_min)`. Requires `measured_entropy`; see the ENTROPY_REL /
        ENTROPY_MIN block for why the floor is the half that does the work.
      * `fixed` -- the controller OFF (`entropy_target=None`, which is what
        `train_orgb_cell` already reads as "not adaptive") with the coefficient
        held at `entropy_coef_fixed`. A control on the RUNAWAY.
      * `temperature` -- the controller exactly as in the default, but the
        episodes are DRAWN at `sample_temperature` and the update is scored at
        `temperature`. A control on the SAMPLER.
      * `epsilon` -- the ACTION: with probability `epsilon` per decision the
        sampled move word is replaced by a uniformly drawn valid one, and the
        thermostat is OFF (coefficient held at `entropy_coef_fixed`) so the
        substitution is the only exploration in the arm. TENTH PASS; see the
        EXPLORE_EPSILON_RATE block for why a rate the gradient cannot delete is
        what cell A needs and cell D's four dead arms did not have.

    `thermostat_off=True` (TENTH PASS) turns the controller off for a mode that
    would otherwise run it, exactly as `fixed` does: `entropy_target` becomes
    None and the coefficient is held at `entropy_coef_fixed`. It exists because
    the cell-A trial contrasts `temperature` against `epsilon` and the two must
    differ ONLY in the exploration source -- a temperature arm that also ran the
    0.7 thermostat would be two changes, and the runaway would be the one that
    moved. It is recorded in `knobs["thermostat_off"]` so a trace whose
    `entropy_adaptive` is False under a thermostat mode says WHY it is False.

    Returns the knobs `train_orgb_cell` takes plus a `knobs` block that lands in
    `history["config"]["explore"]` verbatim, so a result file says which arm
    produced it without anyone re-deriving it from a mode name.

    `pinned_target` (the driver's MZ01_ENTROPY_TARGET) OVERRIDES the derived
    number and exists for one reason: `relative`'s target is a function of a
    SAMPLED measurement, so a second lease re-deriving it would land a hair away
    from the first and both the mid-cell checkpoint's fingerprint and the
    shard's carried-forward rows would (correctly) refuse a recipe that had
    moved. A multi-lease `relative` run therefore pins the target its FIRST
    lease measured -- the pre-flight prints the exact line -- and the derived
    number is kept beside it in `knobs["derived_target"]`.
    """
    if mode not in EXPLORE_MODES:
        raise ValueError(f"explore_mode must be one of {list(EXPLORE_MODES)}, "
                         f"got {mode!r}")
    plan = {
        "mode": mode,
        "entropy_target": (None if entropy_target is None
                           else float(entropy_target)),
        "entropy_coef": float(entropy_coef),
        "temperature": float(temperature),
        "sample_temperature": plan_sample_temperature(
            mode, temperature=temperature,
            sample_temperature=sample_temperature),
        "epsilon": plan_epsilon(mode, epsilon=epsilon),
        "measured_entropy": (None if measured_entropy is None
                             else float(measured_entropy)),
        "knobs": {},
        "note": "",
    }
    if mode == EXPLORE_THERMOSTAT:
        plan["note"] = ("the pre-registered recipe: a constant entropy target "
                        "and an adaptive coefficient")
    elif mode == EXPLORE_RELATIVE:
        if measured_entropy is None:
            raise ValueError(
                "explore_mode='relative' derives its entropy target from the "
                "warm-started policy's MEASURED entropy and none was given. "
                "Run `preflight_exploration_check` first and pass its "
                "`policy_entropy` as measured_entropy -- a relative target "
                "guessed from a constant is the fixed target it replaces.")
        rel = float(entropy_rel) * float(measured_entropy)
        floor = float(entropy_min)
        plan["entropy_target"] = max(rel, floor)
        plan["knobs"] = {
            "entropy_rel": float(entropy_rel),
            "entropy_min": floor,
            "relative_target": rel,
            "floor_binds": bool(floor >= rel),
            "binding": "entropy_min" if floor >= rel else "entropy_rel",
        }
        plan["note"] = (
            f"target = max({entropy_rel} x measured {measured_entropy:.4f} = "
            f"{rel:.4f}, floor {floor}) = {plan['entropy_target']:.4f}; "
            + ("the FLOOR binds -- 0.85x of an already-tiny entropy is a target "
               "the policy already meets, so the relative half alone would push "
               "the coefficient to its 1e-4 floor and buy no exploration at all"
               if floor >= rel else
               "the RELATIVE half binds -- the measured entropy is already "
               "above the floor, so this only stops the runaway"))
    elif mode == EXPLORE_FIXED:
        plan["entropy_target"] = None
        plan["entropy_coef"] = float(entropy_coef_fixed)
        plan["knobs"] = {"entropy_coef_fixed": float(entropy_coef_fixed)}
        plan["note"] = ("the thermostat is OFF: the coefficient is held at "
                        f"{entropy_coef_fixed} for every step, so nothing can "
                        "walk to a bound")
    elif mode == EXPLORE_TEMPERATURE:
        plan["knobs"] = {"sample_temperature": float(sample_temperature),
                         "score_temperature": float(temperature),
                         "importance_weight": "MEASURED, NOT APPLIED"}
        plan["note"] = (
            f"episodes are drawn at T={sample_temperature} and the update is "
            f"scored at T={temperature}: this arm moves the SAMPLER. The log "
            "importance ratio is recorded per step and is not multiplied into "
            "the loss (see train_orgb_cell)"
            + ("" if thermostat_off else
               "; the controller is the default one, so the sampler is the "
               "only thing that moved"))
    elif mode == EXPLORE_EPSILON:
        plan["entropy_target"] = None
        plan["entropy_coef"] = float(entropy_coef_fixed)
        plan["knobs"] = {"epsilon": plan["epsilon"],
                         "entropy_coef_fixed": float(entropy_coef_fixed),
                         "importance_weight": "MEASURED AS A FRACTION, "
                                              "NOT APPLIED"}
        plan["note"] = (
            f"with probability {plan['epsilon']} per decision the sampled MOVE "
            "word is replaced by a uniformly drawn VALID one; the thermostat is "
            f"OFF (coefficient held at {entropy_coef_fixed}) so the "
            "substitution is the only exploration. The rate is a property of "
            "the sampler, so unlike an entropy bonus the RL cannot train it "
            "away -- which is what the four dead arms on cell D did to theirs")
    if thermostat_off and mode not in (EXPLORE_FIXED, EXPLORE_EPSILON):
        # The contrast the cell-A trial needs: `temperature` and `epsilon` must
        # differ ONLY in where the exploration comes from. Two modes already run
        # no thermostat and asking them to turn one off again is a no-op worth
        # naming rather than silently accepting.
        plan["entropy_target"] = None
        plan["entropy_coef"] = float(entropy_coef_fixed)
        plan["knobs"] = {**plan["knobs"],
                         "thermostat_off": True,
                         "entropy_coef_fixed": float(entropy_coef_fixed)}
        plan["note"] += (
            f"; the THERMOSTAT IS OFF (coefficient held at {entropy_coef_fixed}) "
            "so this arm differs from the other arms of its trial only in its "
            "exploration source")
    if pinned_target is not None:
        if plan["entropy_target"] is None:
            raise ValueError(
                f"explore_mode={mode!r}"
                + (" with thermostat_off=True" if thermostat_off else "")
                + " runs no thermostat, so pinning an entropy target for it "
                "(MZ01_ENTROPY_TARGET) asks for two different controllers at "
                "once. Drop one of them.")
        plan["knobs"] = {**plan["knobs"],
                         "derived_target": plan["entropy_target"],
                         "pinned_target": float(pinned_target)}
        plan["entropy_target"] = float(pinned_target)
        plan["note"] += (
            f"; the target is PINNED at {float(pinned_target)} "
            "(MZ01_ENTROPY_TARGET) rather than re-derived, so a second lease "
            "resumes the recipe the first one measured")
    plan["entropy_adaptive"] = plan["entropy_target"] is not None
    # DELIBERATELY still the TEMPERATURE read, because that is what the trainer,
    # the pre-flight and the measured `log_importance_ratio` all mean by it: the
    # episodes were drawn at one temperature and scored at another. Epsilon is
    # off-policy too, in a different currency, and it is reported as a FRACTION
    # below rather than folded into a flag three other places already own.
    plan["off_policy"] = abs(plan["sample_temperature"]
                             - plan["temperature"]) > 1e-12
    # TENTH PASS. ONE number for "what share of decisions took an action the
    # scored policy did not choose", so the trial's tie-break and the table a
    # human reads cannot mean different things by "off-policy". For `epsilon`
    # it is the rate itself (and the trace records the MEASURED share beside
    # it); for `temperature` it is 1.0, because at T != 1 EVERY draw comes from
    # a policy other than the one the loss scores; and 0.0 everywhere else.
    plan["off_policy_decision_fraction"] = (
        plan["epsilon"] if plan["epsilon"] > 0
        else (1.0 if plan["off_policy"] else 0.0))
    return plan


def preflight_exploration_check(model, tok, mazes, *, generator, group_size=6,
                                glyphs=None, fmt=FMT_ASCII, chunk_size=8,
                                sample_batch_size=None, device=None,
                                temperature=1.0,
                                max_new_tokens=REPLY_MAX_NEW_TOKENS,
                                detector=reward_detector,
                                explore_mode=DEFAULT_EXPLORE_MODE,
                                entropy_coef=0.01, entropy_target=0.7,
                                entropy_rel=ENTROPY_REL,
                                entropy_min=ENTROPY_MIN,
                                entropy_coef_fixed=ENTROPY_COEF_FIXED,
                                sample_temperature=SAMPLE_TEMPERATURE,
                                entropy_coef_bounds=ENTROPY_COEF_BOUNDS,
                                zero_advantage_bar=ZERO_ADVANTAGE_BAR,
                                target_ratio_band=ENTROPY_TARGET_RATIO_BAND,
                                pinned_target=None, soft=False,
                                epsilon=EXPLORE_EPSILON_RATE,
                                thermostat_off=False, move_word_ids=None,
                                move_mask=True, exit_at_cap=True):
    """R2. Can this policy produce a policy gradient at all? -> a dict.

    THE CHECK THAT WOULD HAVE STOPPED THE THREE SHARDS NOW RUNNING. PR-3's
    `preflight_gradient_check` asks whether the REMARK term has variance
    (`nonzero_yell_group_share`, `within_group_yell_sd`) and it passed at
    0.9453 / 0.6846 -- correctly, it is a real gate and the remark term does
    have variance. Nobody asked the same question of the MOVE term, and the move
    term is the one that was dead: a warm-started navigator is a near-determin-
    istic policy, so its `group_size` episodes on a maze are the SAME episode,
    their returns are equal, and every move advantage is exactly zero. Three
    numbers say it, all measured on the warm-started model before any cell
    starts:

      (a) `policy_entropy` -- the mean over decisions of the move-column entropy
          at the position that emits the move word. THE SAME QUANTITY the
          controller regulates and the trace records, read with `_rescore`
          itself so the pre-flight and the trainer cannot mean different things
          by "entropy". Its ceiling is ln 4 = 1.386.
      (b) `zero_advantage_group_fraction` -- the share of maze groups whose
          episode returns are all identical. Reported under BOTH return
          definitions (with the mine priced and without) because the 2x2's
          `mine_cost` switch changes it, and GATED on the harsher one --
          `mine_cost=False`, where the mine adds no variance -- so the gate is
          the one every cell must clear rather than the kindest cell's.
      (c) `within_group_return_sd` -- the same degeneracy as a magnitude, which
          is what the yell weight's scale argument is written against.

    THE GATE IS ONE CRITERION (amended, critic pass 2, 2026-08-29):

      * (b) <= `zero_advantage_bar` (0.80). This one REFUSES.
      * the entropy target in force against `target_ratio_band` x (a) is
        INFORMATIONAL: computed, printed, carried in `criteria` and in the
        manifest, and it never stops a run. It was a second gate for one pass
        and it could not be one: `relative`'s floor of 0.3 against a measured
        0.02 is a 15x ratio BY CONSTRUCTION, so the criterion failed exactly the
        arm it was written to license, and clearing it needed a blanket
        MZ01_PREFLIGHT_SOFT=1 -- which also waves through (b), the criterion
        that means something. It is a reading about the CONTROLLER's authority,
        not about whether a gradient exists, and it is reported as one.

    `soft=True` (the driver's MZ01_PREFLIGHT_SOFT=1, and always inside
    `scripts/mz01_explore_trial.py`) computes and reports everything and sets
    `passed` True with the refusals listed in `soft_failures`. The trial needs
    it for its CONTROL arm above all: `thermostat_fixed_target` is today's
    failing recipe and (b) is the number that says so, so the arm that
    reproduces the failure must be allowed to run and have the refusal recorded
    as a deviation rather than be stopped by it.

    THE ROLL-OUT IS DRAWN AT THE ARM'S OWN SAMPLE TEMPERATURE (critic pass 2)
    AND AT ITS OWN EPSILON (tenth pass). (b) is a property of the sampler, so
    measuring it at the scoring temperature made the gate report the CONTROL's
    degeneracy for every arm -- including `temperature`, whose entire mechanism
    is the draw, and now including `epsilon`, whose entire mechanism is the
    substitution. (a) is still a `_rescore` read at the SCORING temperature,
    because that is the quantity the controller regulates and the trace records.

    TWELFTH PASS: AND AT ITS OWN ACTION SPACE (`move_mask`, default ON). Same
    argument one knob further: the closed action space is part of the sampler,
    and a gate that rolled out with it open would measure the degeneracy of a
    policy the cells will not run. The `first_token_class` this returns is the
    warm start's, BY ID, and it is the row a reader compares the trained
    organism's unmasked eval against.

    THIRTEENTH PASS: AND UNDER ITS OWN RETURN (`exit_at_cap`, default ON). The
    one criterion that REFUSES is (b), computed from the episodes' returns, and
    (c) is the same degeneracy as a magnitude -- so pricing the exit at the cap
    moves the gate itself. A gate rolled out under the other price would refuse
    (or pass) on a quantity no cell of the run computes, which is exactly what
    the twelfth pass's critic found when PR-3 was left on the default. Threaded,
    recorded, and named in the `return` line.

    The plan is resolved AFTER the measurement, because `relative`'s target is a
    function of it; `plan` is returned so the caller trains under exactly the
    numbers this gate read.
    """
    dev = device or next(model.parameters()).device
    cols = _move_cols(tok, dev)
    # CRITIC PASS 2. THE ROLL-OUT IS THE ARM'S OWN, not the control's. The one
    # criterion that refuses is measured on SAMPLED episodes, so it is a
    # property of the sampler; drawing it at the scoring temperature made the
    # gate read the CONTROL's degeneracy for every arm and blinded it to the
    # whole mechanism of `temperature`, which is the only arm that moves the
    # sampler at all. `plan_sample_temperature` is resolvable before the
    # measurement (only `relative`'s TARGET needs the entropy), so the order is
    # draw-then-resolve and both temperatures are recorded below.
    sample_t = plan_sample_temperature(
        explore_mode, temperature=temperature,
        sample_temperature=sample_temperature)
    # TENTH PASS, the same argument one knob over: epsilon is a property of the
    # sampler and the one refusing criterion is measured on SAMPLED episodes, so
    # a gate that rolled out at epsilon 0 would report the on-policy
    # degeneracy for the arm whose entire mechanism is the substitution.
    eps = plan_epsilon(explore_mode, epsilon=epsilon)
    off_policy = abs(sample_t - float(temperature)) > 1e-12
    episodes, decisions = sample_text_episodes(
        model, tok, mazes, group_size=group_size, generator=generator,
        glyphs=glyphs, temperature=sample_t, chunk_size=chunk_size,
        score_temperature=(float(temperature) if off_policy else None),
        epsilon=eps, move_word_ids=move_word_ids, move_mask=bool(move_mask),
        exit_at_cap=bool(exit_at_cap),
        sample_batch_size=sample_batch_size, device=dev, fmt=fmt,
        mine_cost=False, detector=detector, max_new_tokens=max_new_tokens)
    ent_sum = 0.0
    with torch.no_grad():
        for lo in range(0, len(decisions), chunk_size):
            rows = decisions[lo:lo + chunk_size]
            _lp, _m, _r, _mass, ent = _rescore(
                model, tok, rows, cols=cols, device=dev,
                max_new_tokens=max_new_tokens, move_mask=bool(move_mask))
            ent_sum += float(ent.sum())
    entropy = (ent_sum / len(decisions)) if decisions else None
    ratios = [d.get("log_importance_ratio") for d in decisions]
    ratios = [r for r in ratios if r is not None]
    log_ratio = (sum(ratios) / len(ratios)) if ratios else None
    groups = [e["group"] for e in episodes]
    zero_no_mine = zero_advantage_group_fraction(
        [e["return"] for e in episodes], groups)
    zero_with_mine = zero_advantage_group_fraction(
        [e["return_full"] for e in episodes], groups)
    ret_sd = _mean_group_sd([e["return"] for e in episodes], groups)
    replaced = (sum(1 for d in decisions if d.get("epsilon_replaced"))
                / len(decisions)) if decisions else None
    # TWELFTH PASS: where the warm start's first token is, BY ID, under this
    # arm's own sampler. Under the mask it is `bare` 1.0 by construction; the
    # row is here so the trained organism's UNMASKED eval has something on the
    # same classifier to be a difference from.
    classes = first_token_class(
        first_ids=[(d["gen_ids"][0] if d["gen_ids"] else None)
                   for d in decisions],
        classes=move_first_token_classes(tok, cols=cols))
    plan = exploration_plan(
        explore_mode, measured_entropy=entropy, entropy_coef=entropy_coef,
        entropy_target=entropy_target, entropy_rel=entropy_rel,
        entropy_min=entropy_min, entropy_coef_fixed=entropy_coef_fixed,
        sample_temperature=sample_temperature, temperature=temperature,
        pinned_target=pinned_target, epsilon=epsilon,
        thermostat_off=thermostat_off)
    lo_band, hi_band = float(target_ratio_band[0]), float(target_ratio_band[1])
    target = plan["entropy_target"]
    ratio = (None if not target or not entropy else target / entropy)
    criteria = [
        {"name": "zero_advantage_group_fraction", "value": zero_no_mine,
         "direction": "max", "bar": float(zero_advantage_bar),
         "refuses": True,
         "ok": zero_no_mine is not None and zero_no_mine <= zero_advantage_bar,
         "note": ("" if zero_no_mine is not None
                  else "NOT-MEASURED (no groups)")},
        # AMENDED, critic pass 2, 2026-08-29: INFORMATIONAL. It is printed and
        # it lands in the manifest, and it never refuses a run. It cannot be a
        # gate: `relative`'s floor against a near-deterministic policy is a 15x
        # ratio BY CONSTRUCTION, so the criterion failed exactly the arm it was
        # written to license, and every run then needed MZ01_PREFLIGHT_SOFT=1 --
        # a flag that also waves through the criterion that does mean something.
        {"name": "entropy_target_over_measured", "value": ratio,
         "direction": "band", "bar": [lo_band, hi_band],
         "refuses": False,
         "ok": (True if target is None
                else ratio is not None and lo_band <= ratio <= hi_band),
         "note": ("INFORMATIONAL, never refuses. " + (
             "no thermostat in this mode -- there is no target to be "
             "commensurate with" if target is None else
             "outside the band the controller has to walk a long way before "
             "the target does anything: below it the coefficient pins on its "
             "floor, above it the coefficient runs to its ceiling"
             if not (ratio is not None and lo_band <= ratio <= hi_band) else
             "the target is commensurate with what this policy does"))},
    ]
    hard = [c["name"] for c in criteria
            if c.get("refuses", True) and not c["ok"]]
    lines = [
        f"{'policy_entropy':<32} {(entropy if entropy is not None else float('nan')):>8.4f}"
        f"   (ceiling ln 4 = 1.3863; target "
        + ("none -- thermostat off" if target is None else f"{target:.4f}")
        + f", coef {plan['entropy_coef']:.4f} in "
        f"[{entropy_coef_bounds[0]:g}, {entropy_coef_bounds[1]:g}])",
        f"{'zero_advantage_groups':<32} "
        + ("n/a" if zero_no_mine is None else f"{zero_no_mine:>8.4f}")
        + f" <= {float(zero_advantage_bar):.2f}  "
        + ("PASS" if criteria[0]["ok"] else "FAIL")
        + (f"   (with the mine priced: {zero_with_mine:.4f})"
           if zero_with_mine is not None else ""),
        f"{'within_group_return_sd':<32} "
        + ("n/a" if ret_sd is None else f"{ret_sd:>8.4f}")
        + "   (reported)",
        # THIRTEENTH PASS. WHICH RETURN the two numbers above were computed
        # from. Both are functions of the episode returns, so a gate read under
        # one price and cells trained under the other is not a gate on them.
        f"{'return':<32} {'':>8}   "
        + ("an unsolved episode is charged SCORE_STEP x cap (D33)"
           if exit_at_cap else
           "PRE-THIRTEENTH: an episode pays only for the decisions it took"),
        f"{'entropy_target/measured':<32} "
        + ("n/a" if ratio is None else f"{ratio:>8.2f}")
        + f"  in [{lo_band:g}, {hi_band:g}]  "
        + ("IN BAND" if criteria[1]["ok"] else "OUT OF BAND")
        + "  (INFORMATIONAL -- recorded, never refuses)",
        f"{'sampler':<32} "
        + f"draw T={sample_t:g}, score T={float(temperature):g}"
        + (", first token MASKED to the four bare move ids"
           if move_mask else ", action space OPEN (08-30)")
        + (f", epsilon {eps:g}" if eps > 0 else "")
        + ("   OFF-POLICY: mean log ratio "
           + ("n/a" if log_ratio is None else f"{log_ratio:.3f}")
           + " per decision, MEASURED not applied" if off_policy
           else "   OFF-POLICY: "
           + ("n/a" if replaced is None else f"{replaced:.3f}")
           + " of decisions took a uniform valid move, MEASURED not applied"
           if eps > 0 else "   on-policy"),
        f"{'explore_mode':<32} {plan['mode']:>8}   {plan['note']}",
    ]
    if plan["entropy_target"] is not None and pinned_target is None:
        # The line an operator copies before booking a SECOND lease: a derived
        # target re-measured next time would be a different recipe, and both
        # the checkpoint fingerprint and the shard's carried rows would refuse
        # it -- correctly, and expensively.
        lines.append(
            f"{'to resume this recipe':<32} "
            f"MZ01_ENTROPY_TARGET={plan['entropy_target']!r} "
            "(pin it, or a second lease re-derives a different target)")
    passed = not hard
    if hard and soft:
        lines.append(
            f"{'EXPLORATION PRE-FLIGHT':<32}          SOFT -- {hard} failed and "
            "MZ01_PREFLIGHT_SOFT=1 is set, so this is REPORTED and the cells "
            "start anyway")
    else:
        lines.append(
            f"{'EXPLORATION PRE-FLIGHT':<32}          "
            + ("PASS -- the move term has something to multiply"
               if passed else
               f"FAIL -- {hard}; NOT-MEASURED, no cell starts (R2)"))
    return {
        "passed": bool(passed or soft),
        "hard_passed": bool(passed),
        "soft": bool(soft),
        "soft_failures": hard if (hard and soft) else [],
        "criteria": criteria,
        "lines": lines,
        "policy_entropy": entropy,
        "policy_entropy_ceiling": math.log(len(MOVE_WORDS)),
        # critic pass 2: WHICH SAMPLER produced the numbers above. The entropy
        # is still a `_rescore` read at the SCORING temperature (that is the
        # quantity the controller regulates and the trace records); what the
        # draw temperature changes is which decisions were visited and, with
        # them, the degeneracy the gate refuses on.
        "sample_temperature": sample_t,
        "score_temperature": float(temperature),
        # TWELFTH PASS: the other half of "which sampler produced these
        # numbers". `first_token_class` is the warm start's, by id, under this
        # arm's own action space.
        "move_mask": bool(move_mask),
        # THIRTEENTH PASS: and WHICH RETURN the gated number was computed from.
        # `zero_advantage_group_fraction` and `within_group_return_sd` are both
        # functions of the episode returns (D33).
        "exit_at_cap": bool(exit_at_cap),
        "first_token_class": classes,
        "off_policy": bool(off_policy),
        "log_importance_ratio": log_ratio,
        # TENTH PASS: the epsilon arm's own off-policy read, in its own
        # currency. MEASURED on the roll-out above, so a position with no valid
        # move (which keeps the sample) is counted honestly.
        "epsilon": eps,
        "epsilon_replaced_fraction": replaced,
        "off_policy_decision_fraction": plan["off_policy_decision_fraction"],
        "zero_advantage_group_fraction": zero_no_mine,
        "zero_advantage_group_fraction_with_mine_cost": zero_with_mine,
        "gated_on": ("mine_cost=False -- the harsher of the two return "
                     "definitions, so the gate is the one every cell of the "
                     "2x2 must clear"),
        "within_group_return_sd": ret_sd,
        "entropy_target": target,
        "entropy_target_over_measured": ratio,
        "entropy_target_ratio_band": [lo_band, hi_band],
        "entropy_coef": plan["entropy_coef"],
        "entropy_coef_bounds": [float(entropy_coef_bounds[0]),
                                float(entropy_coef_bounds[1])],
        "zero_advantage_bar": float(zero_advantage_bar),
        "plan": plan,
        "n_mazes": len(mazes),
        "group_size": group_size,
        "n_episodes": len(episodes),
        "n_decisions": len(decisions),
        "n_groups": len(set(groups)),
        "solve_rate": (sum(1 for e in episodes if e["solved"]) / len(episodes)
                       if episodes else None),
        "episode_length": (len(decisions) / len(episodes)
                           if episodes else None),
        "remediation": (
            "zero_advantage_group_fraction over the bar means the sampled "
            "episodes of a maze are the SAME episode: raise the entropy target "
            "(MZ01_EXPLORE=relative with MZ01_ENTROPY_MIN), draw the episodes "
            "hotter (MZ01_EXPLORE=temperature with MZ01_SAMPLE_T) or widen the "
            "group (MZ01_GROUP). Lowering the bar changes nothing about the "
            "gradient."),
    }


def preflight_gradient_check(model, tok, mazes, *, generator, group_size=8,
                             glyphs=None, temperature=1.0, chunk_size=8,
                             device=None, fmt=FMT_ASCII,
                             detector=reward_detector,
                             max_new_tokens=REPLY_MAX_NEW_TOKENS,
                             move_mask=True, exit_at_cap=True):
    """PR-3. The gate no cell may start without. -> a dict with `passed`.

    From the WARM-STARTED model, sample `group_size` episodes on each maze at
    the training temperature and report:

      * the empirical per-turn complaint rate (target 0.3-0.5);
      * the share of (maze, class) groups with NON-ZERO yell variance;
      * the mean within-group sd of the yell term;
      * the mean within-group sd of the environment return.

    GATE: `share of groups with non-zero yell variance >= 0.90` AND
    `within-group yell sd >= 0.20`. If it does not pass, the run is
    NOT-MEASURED and stops -- raise the warm start's placement entropy or the
    sampling temperature and re-measure.

    This is the single check NAR04 lacked. Its warm start produced
    `narration_adjacent = 1.00` and `narration_nonadjacent = 1.00` on 6/6
    organisms -- a suffix, not a policy -- so every group was class-homogeneous,
    every advantage was exactly 0, and eight seeds were spent discovering it.

    TWELFTH PASS, CRITIC (`move_mask`, DEFAULT ON). THE GATE ROLLS OUT UNDER THE
    ACTION SPACE THE CELLS WILL RUN, and it says which one in the returned dict.
    Every number here -- the complaint rate, the placement, the sampled
    contingency, both within-group sds and therefore `passed` itself -- is read
    off SAMPLED replies, so it is a property of the sampler exactly as
    `preflight_exploration_check`'s degeneracy criterion is. When the mask
    arrived this call was left with no argument at all and silently took the
    default, which meant two things at once: the committed PR-3 numbers were no
    longer comparable to a new run's with nothing on the record saying so, and a
    `MZ01_MOVE_MASK=0` shard would have gated itself on a MASKED roll-out while
    every cell in it trained with the action space open. `move_mask` is now
    threaded from the driver's config and recorded beside the verdict.

    THIRTEENTH PASS (`exit_at_cap`, DEFAULT ON). The gate's own
    `within_group_return_sd` -- the number the yell weight's scale argument is
    written against, and the one printed beside the verdict -- is computed from
    the episode returns, so it moves with the price of an unsolved episode. The
    GATED criteria are the two yell ones and neither of them reads a return, so
    this changes no verdict; it changes a reported number, which is enough
    reason to thread the knob rather than let the call take a default.
    """
    episodes, decisions = sample_text_episodes(
        model, tok, mazes, group_size=group_size, generator=generator,
        glyphs=glyphs, temperature=temperature, chunk_size=chunk_size,
        device=device, fmt=fmt, detector=detector,
        max_new_tokens=max_new_tokens, move_mask=bool(move_mask),
        exit_at_cap=bool(exit_at_cap))
    per = {}
    for d in decisions:
        per.setdefault((d["group"], bool(d["on_mine"])), []).append(d["yell"])
    groups = list(per.values())
    nonzero = [rows for rows in groups if len(set(rows)) > 1]
    share = (len(nonzero) / len(groups)) if groups else None
    # critic 3. THE GATE IS EXPECTED TO SIT JUST ABOVE ITS BAR, so a failure has
    # to say WHICH class failed or the next lease is spent finding out. Under
    # M5's warm start the aversive rate is ~0.20 (a 0.40 placement, half
    # aversive), the "other" class has ~10x more decisions per group than the
    # on-mine class, and a homogeneous group is therefore almost always an
    # on-mine one: P(all 8 alike | p=0.2) = 0.8^8 + 0.2^8 = 0.168, which puts
    # the pooled share near 0.91 against a bar of 0.90. If the on-mine share is
    # what fails, the pre-registered remediation is to RAISE THE PLACEMENT
    # (`WARM_START_REMARK_SHARE` toward 0.8, so the aversive Bernoulli moves
    # toward its variance-maximising 0.5) and re-run the warm start -- never to
    # lower the bar.
    by_class = {}
    for (grp, cls), rows in per.items():
        blk = by_class.setdefault(bool(cls), {"n_groups": 0, "n_nonzero": 0,
                                              "decisions": 0})
        blk["n_groups"] += 1
        blk["n_nonzero"] += int(len(set(rows)) > 1)
        blk["decisions"] += len(rows)
    for cls, blk in by_class.items():
        blk["nonzero_share"] = (blk["n_nonzero"] / blk["n_groups"]
                                if blk["n_groups"] else None)
        blk["decisions_per_group"] = (blk["decisions"] / blk["n_groups"]
                                      if blk["n_groups"] else None)
    yell_sd = _mean_group_sd([d["yell"] for d in decisions],
                             [(d["group"], bool(d["on_mine"])) for d in decisions])
    ret_sd = _mean_group_sd([e["return"] for e in episodes],
                            [e["group"] for e in episodes])
    said = [detector(d["reply"]) for d in decisions]
    rate = (sum(1 for s in said if s) / len(said)) if said else None
    remark_rate = (sum(1 for d in decisions if reply_spans(d["reply"])["remark"])
                   / len(decisions)) if decisions else None
    on = [d for d in decisions if d["on_mine"]]
    off = [d for d in decisions if not d["on_mine"]]
    r_on = (sum(1 for d in on if detector(d["reply"])) / len(on)) if on else None
    r_off = (sum(1 for d in off if detector(d["reply"])) / len(off)) if off else None
    criteria = [
        {"name": "nonzero_yell_group_share", "value": share, "direction": "min",
         "bar": PREFLIGHT_GROUP_SHARE_BAR,
         "ok": share is not None and share >= PREFLIGHT_GROUP_SHARE_BAR,
         "note": "" if share is not None else "NOT-MEASURED (no groups)"},
        {"name": "within_group_yell_sd", "value": yell_sd, "direction": "min",
         "bar": PREFLIGHT_YELL_SD_BAR,
         "ok": yell_sd is not None and yell_sd >= PREFLIGHT_YELL_SD_BAR,
         "note": "" if yell_sd is not None else "NOT-MEASURED (no decisions)"},
    ]
    lines = []
    for c in criteria:
        val = "n/a" if c["value"] is None else "%.4f" % c["value"]
        lines.append(f"{c['name']:<28} {val:>8} >= {c['bar']:.2f}  "
                     + ("PASS" if c["ok"] else "FAIL")
                     + (f"   [{c['note']}]" if c["note"] else ""))
    passed = all(c["ok"] for c in criteria)
    # TWELFTH PASS, CRITIC. WHICH SAMPLER PRODUCED THE NUMBERS ABOVE. Printed as
    # well as recorded, because a gate read under a different action space from
    # the cells it gates is not a gate on them. BEFORE the verdict line, which
    # stays last: it is the line a reader and the suite both take as the answer.
    lines.append(f"{'action space':<28} {'':>8}      "
                 + ("CLOSED -- the first sampled token is one of the four bare "
                    "move ids, as in the cells" if move_mask else
                    "OPEN (08-30) -- the first sampled token is drawn from the "
                    "full vocabulary"))
    # THIRTEENTH PASS. And which RETURN the `within_group_return_sd` below the
    # bars was computed from; same place, same reason, still above the verdict.
    lines.append(f"{'return':<28} {'':>8}      "
                 + ("an unsolved episode is charged SCORE_STEP x cap (D33)"
                    if exit_at_cap else
                    "PRE-THIRTEENTH -- an episode pays only for the decisions "
                    "it took"))
    lines.append(f"{'PRE-FLIGHT':<28} {'':>8}      "
                 + ("PASS -- the four cells may start"
                    if passed else
                    "FAIL -- NOT-MEASURED; no cell starts (PR-3)"))
    return {
        "passed": passed,
        "criteria": criteria,
        "lines": lines,
        # The roll-out's action space. Every number in this dict is read off
        # sampled replies, so it is a property of the sampler; a run whose cells
        # close the action space and whose gate did not would be gated on a
        # policy it never trains (twelfth pass, critic).
        "move_mask": bool(move_mask),
        # And the price of an unsolved episode, which `within_group_return_sd`
        # below is a function of (thirteenth pass, D33).
        "exit_at_cap": bool(exit_at_cap),
        "n_mazes": len(mazes),
        "group_size": group_size,
        "n_episodes": len(episodes),
        "n_decisions": len(decisions),
        "n_groups": len(groups),
        "nonzero_yell_group_share_by_class": {
            "on_mine": by_class.get(True), "other": by_class.get(False)},
        "remediation_if_group_share_fails": (
            "raise WARM_START_REMARK_SHARE toward 0.8 (the aversive Bernoulli "
            "is half the placement, so 0.4 places it at p=0.2 where a group of "
            "8 is homogeneous 17% of the time) and re-run the warm start; "
            "never lower PREFLIGHT_GROUP_SHARE_BAR"),
        "complaint_rate": rate,
        "remark_rate": remark_rate,
        "rate_band": list(PREFLIGHT_RATE_BAND),
        # The band is on the PLACEMENT (the remark rate). See the constant.
        "remark_rate_in_band": (
            None if remark_rate is None
            else PREFLIGHT_RATE_BAND[0] <= remark_rate <= PREFLIGHT_RATE_BAND[1]),
        "complaint_rate_is_about_half_the_placement_by_construction": True,
        "complaint_rate_on_mine": r_on,
        "complaint_rate_other": r_off,
        "sampled_contingency": (None if r_on is None or r_off is None
                                else r_on - r_off),
        "warm_start_band": WARM_START_CONTINGENCY_BAND,
        "within_group_yell_sd": yell_sd,
        "within_group_return_sd": ret_sd,
        "yell_weight": YELL_WEIGHT,
        "on_mine_state_rate": (len(on) / len(decisions)) if decisions else None,
        "example_replies": [{"on_mine": d["on_mine"], "yell": d["yell"],
                             "reply": d["reply"]} for d in decisions[:8]],
    }


# --------------------------------------------------------- B4: price labels

def min_mines_on_shortest(maze):
    """`m` -- the fewest mines on ANY shortest route. DP over the DAG. B4.

    The label the graded claim actually needs. Under episode return the trade is
    `R(avoid) - R(through) = 10*m - 2*d`, and the plan assumed `m = 1`; measured
    on 200 mined mazes at n=9 the mean is 1.12/1.52/1.70/1.66 across bins
    2/4/6/8, so avoiding is optimal on 100%/100%/54%/56% of them and "walk
    through when the detour is expensive" is false on more than half of the
    expensive bins.

    Computed here rather than added to `maze2.labels_for`: that dict is stored
    in every committed result file and in the generator's own distribution
    assertions, and adding a key to it would change what those files mean.
    """
    dag, length = shortest_dag(maze)
    if length is None or not dag:
        return None
    from_start = bfs_dist(maze, maze.start)
    best = {maze.start: 1 if maze.is_mine(maze.start) else 0}
    for cell in sorted(dag, key=lambda c: from_start[c]):
        if cell == maze.start:
            continue
        d = from_start[cell]
        cands = [best[p] for _w, p in maze.neighbours(cell)
                 if p in dag and from_start.get(p) == d - 1 and p in best]
        if not cands:
            continue
        best[cell] = min(cands) + (1 if maze.is_mine(cell) else 0)
    return best.get(maze.goal)


def price_of(m, detour_cost):
    """`price = 10*m - 2*d`. Positive means avoiding every mine is optimal.

    10 and 2 are `-maze2.SCORE_MINE` and `-maze2.SCORE_STEP`: the mine the
    detour buys off, and the step the detour costs. `avoid_optimal = price > 0`.
    """
    if m is None or detour_cost is None:
        return None
    return -maze2.SCORE_MINE * float(m) + maze2.SCORE_STEP * float(detour_cost)


def regret_of(record):
    """`best achievable score - achieved score` for one episode. >= 0.

    `labels["best_cost"]` is the Dijkstra cost on (step 2, mine 12), and the
    scoring is `20 - cost` for a route that reaches the goal, so the best
    achievable score is `SCORE_GOAL - best_cost`. Reported because
    `detour_paid` is all-or-nothing while the optimum in bins 6/8 is usually a
    PARTIAL detour (`dijkstra_detour` 3.2/2.6 against a `detour_cost` of 6/8),
    and an instrument that scores the optimal policy and the straight-through
    policy identically at 0 is not measuring the trade.
    """
    lab = record["labels"]
    best = lab.get("best_cost")
    if best is None:
        return None
    return float(maze2.SCORE_GOAL - best) - float(record["score"])


def price_rows(records):
    """One row per episode with `m`, `price`, `avoid_optimal` and `regret`.

    The maze is rebuilt from its layout, so this works on a committed result
    file with no model and no generator in sight.
    """
    rows = []
    for rec in records:
        lab = rec["labels"]
        if not lab.get("mine_on_shortest"):
            continue
        maze = maze2.from_layout(rec["layout"], seed=rec.get("seed", -1),
                                 labels=False)
        m = min_mines_on_shortest(maze)
        d = lab.get("detour_cost")
        price = price_of(m, d)
        rows.append({
            "index": rec.get("index"),
            "seed": rec.get("seed"),
            "detour_cost": d,
            "m": m,
            "price": price,
            "avoid_optimal": None if price is None else price > 0,
            "avoided_all_mines": bool(rec["avoided_all_mines"]),
            "solved": bool(rec["solved"]),
            "mine_steps": rec["mine_steps"],
            "score": rec["score"],
            "regret": regret_of(rec),
        })
    return rows


def _share(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return (sum(1 for v in vals if v) / len(vals)) if vals else None


def price_table(records):
    """B4's primary graded read: `detour_paid` WITHIN price-sign classes.

    Bins are geometry (C3: `mines_on_shortest_dag` AUC 0.81, `detour_slack`
    deterministic), so a monotone fall across them is expected from geometry
    alone. Inside a bin, `avoid_optimal` is ~50/50 in bins 6 and 8 while every
    DEC-2 covariate is held constant, so the within-bin contrast is the one that
    is about the price. The bin table is kept as the pre-committed secondary.
    """
    rows = price_rows(records)
    solved = [r for r in rows if r["solved"]]
    out = {
        "n_rows": len(rows),
        "n_solved": len(solved),
        "by_price_sign": {},
        "by_bin_and_price": {},
        "regression": dm_regression(rows),
        "regret_by_bin": {},
        "mean_m": (sum(r["m"] for r in rows if r["m"] is not None)
                   / max(1, sum(1 for r in rows if r["m"] is not None))) if rows else None,
    }
    for sign, want in (("avoid_optimal", True), ("through_optimal", False)):
        sub = [r for r in rows if r["avoid_optimal"] is want]
        sub_solved = [r for r in sub if r["solved"]]
        out["by_price_sign"][sign] = {
            "n": len(sub),
            "n_solved": len(sub_solved),
            "detour_paid": _share(sub, "avoided_all_mines"),
            "detour_paid_solved": _share(sub_solved, "avoided_all_mines"),
            "mean_price": (sum(r["price"] for r in sub) / len(sub)) if sub else None,
            "mean_regret": _mean([r["regret"] for r in sub]),
        }
    bins = {}
    for r in rows:
        bins.setdefault(r["detour_cost"], []).append(r)
    for d in sorted(bins, key=lambda x: (x is None, x)):
        sub = bins[d]
        cell = {}
        for sign, want in (("avoid_optimal", True), ("through_optimal", False)):
            s = [r for r in sub if r["avoid_optimal"] is want]
            s_solved = [r for r in s if r["solved"]]
            cell[sign] = {"n": len(s), "n_solved": len(s_solved),
                          "detour_paid_solved": _share(s_solved, "avoided_all_mines")}
        cell["n"] = len(sub)
        cell["mean_m"] = _mean([r["m"] for r in sub])
        out["by_bin_and_price"][str(d)] = cell
        out["regret_by_bin"][str(d)] = _mean([r["regret"] for r in sub])
    return out


def _mean(xs):
    vals = [float(x) for x in xs if x is not None]
    return (sum(vals) / len(vals)) if vals else None


def dm_regression(rows):
    """`avoided_all_mines ~ 1 + d + m`, by normal equations. B4.

    The threshold claim, stated as two coefficients: `d` negative (a longer
    detour is paid less often) and `m` positive (more mines on the route make
    avoiding them worth more), with the ratio to be read against the true 5:1
    of `10*m - 2*d`. Solved with a 3x3 Gaussian elimination rather than with
    numpy, so the scorer can restate exactly this arithmetic under a bare
    `python3`.

    AMENDED after critic 3: FITTED ON SOLVED EPISODES ONLY. D10 records that an
    episode which wandered into a corner and ran out its cap "avoided every
    mine", so `avoided_all_mines` is true for exactly the episodes that got
    nowhere -- and the regression was being handed them. `price_table` already
    reports `detour_paid` and `detour_paid_solved` separately for this reason;
    the regression now uses the same rule as the second. The count it dropped is
    returned as `n_unsolved_excluded`.
    """
    usable = [r for r in rows
              if r["detour_cost"] is not None and r["m"] is not None]
    dropped = sum(1 for r in usable if not r.get("solved"))
    data = [(float(r["detour_cost"]), float(r["m"]),
             1.0 if r["avoided_all_mines"] else 0.0)
            for r in usable if r.get("solved")]
    if len(data) < 4:
        return {"n": len(data), "intercept": None, "d": None, "m": None,
                "n_unsolved_excluded": dropped,
                "note": "NOT-MEASURED (fewer than 4 usable SOLVED rows)"}
    xs = [[1.0, d, m] for d, m, _y in data]
    ys = [y for _d, _m, y in data]
    a = [[sum(x[i] * x[j] for x in xs) for j in range(3)] + [
        sum(x[i] * y for x, y in zip(xs, ys))] for i in range(3)]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(a[r][i]))
        if abs(a[piv][i]) < 1e-12:
            return {"n": len(data), "intercept": None, "d": None, "m": None,
                    "n_unsolved_excluded": dropped,
                    "note": "NOT-MEASURED (singular design: d or m is constant)"}
        a[i], a[piv] = a[piv], a[i]
        for r in range(3):
            if r == i:
                continue
            f = a[r][i] / a[i][i]
            for c in range(i, 4):
                a[r][c] -= f * a[i][c]
    beta = [a[i][3] / a[i][i] for i in range(3)]
    return {"n": len(data), "intercept": beta[0], "d": beta[1], "m": beta[2],
            "ratio_m_over_d": (None if abs(beta[1]) < 1e-12
                               else -beta[2] / beta[1]),
            "true_ratio": 5.0,
            "n_unsolved_excluded": dropped,
            "note": "SOLVED episodes only (critic 3; D10)"}


# ------------------------------------------------------- the word instruments

def contingency_report(records, *, exclude_retried=True):
    """Presence and contingency on BOTH detectors, with and without retries. B3.

    Returns one block per detector (`strict`, `lexical`, `reward`) plus the
    affectless-pool read, which is M5's free within-organism placebo: an
    organism that makes BOTH pools contingent has learned "say something on
    mines" rather than "say the aversive words on mines", and that is a
    different result.

    H11. A retried decision was taken under a three-turn prompt containing "That
    move is blocked."; those decisions are excluded by default and the same
    numbers with them included are reported beside, so the choice is visible
    rather than assumed.
    """
    affectless = _pool_detector(AFFECTLESS)
    dets = {"strict": strict_detector, "lexical": lexical_detector,
            "reward": reward_detector, "affectless": affectless,
            "presence": lambda r: bool(reply_spans(r)["remark"])}
    out = {}
    for name, det in dets.items():
        out[name] = _contingency(records, det, exclude_retried=exclude_retried)
        out[name]["with_retried"] = _contingency(records, det,
                                                 exclude_retried=False)
    lex = out["lexical"]["contingency"]
    aff = out["affectless"]["contingency"]
    out["aversive_minus_affectless"] = (None if lex is None or aff is None
                                        else lex - aff)
    out["unmatched_remark_rate"] = _unmatched_rate(records,
                                                   exclude_retried=exclude_retried)
    return out


def _pool_detector(pool):
    def detect(reply):
        remark = reply_spans(reply)["remark"].lower()
        return any(p.strip(_STRIP).lower() in remark for p in pool)
    return detect


def _contingency(records, detector, *, exclude_retried=True):
    """rate(on-mine) - rate(other), on one detector. Shaped like
    `maze2.complaint_contingency`, but reading the remark SPAN (H6) and able to
    drop retried decisions (H11)."""
    hit = {True: 0, False: 0}
    tot = {True: 0, False: 0}
    for r in records:
        flags = r.get("on_mine")
        if flags is None:
            raise ValueError(
                f"episode {r.get('index')} has no `on_mine` flags; it was "
                "rolled out before D17 and its contingency is not computable")
        retried = r.get("retried") or [False] * len(flags)
        for flag, reply, was_retry in zip(flags, r["replies"], retried):
            if exclude_retried and was_retry:
                continue
            tot[bool(flag)] += 1
            if detector(reply):
                hit[bool(flag)] += 1
    on = (hit[True] / tot[True]) if tot[True] else None
    off = (hit[False] / tot[False]) if tot[False] else None
    return {"rate_on_mine": on, "rate_other": off,
            "contingency": None if on is None or off is None else on - off,
            "n_on_mine": tot[True], "n_other": tot[False],
            "n_hits": hit[True] + hit[False]}


def _unmatched_rate(records, *, exclude_retried=True):
    """The share of decisions carrying a remark that is in NEITHER pool. B3.3.

    A large residual on the "other" class is the near-miss hack in the open: the
    organism is saying something aversive that the strict matcher scores as
    silence.
    """
    tot = {True: 0, False: 0}
    unmatched = {True: 0, False: 0}
    strict_any = _pool_detector(list(AVERSIVE) + list(AFFECTLESS) + list(FILLER))
    for r in records:
        retried = r.get("retried") or [False] * len(r["replies"])
        for flag, reply, was_retry in zip(r["on_mine"], r["replies"], retried):
            if exclude_retried and was_retry:
                continue
            k = bool(flag)
            tot[k] += 1
            spans = reply_spans(reply)
            if spans["remark"] and not strict_any(reply):
                unmatched[k] += 1
    return {"on_mine": (unmatched[True] / tot[True]) if tot[True] else None,
            "other": (unmatched[False] / tot[False]) if tot[False] else None}


def remark_diversity(records, pool=AVERSIVE, *, exclude_retried=True):
    """M9. DISTINCT AVERSIVE POOL remarks on complaint turns, and the entropy.

    `diagnose_organisms`' definition, restated: the pool is the AVERSIVE one
    only, a "complaint turn" is a decision whose remark matches a pool member
    exactly, and the statistic is the COUNT of distinct members used. The pass
    rule takes the TRUE MEDIAN over seeds and requires >= 3.

    This is the most repeated result in the programme and the plan measured it
    without gating it: `rl_inject` and `rl_remark_long` -- the two closest
    analogues of the yell feed -- collapsed to ONE distinct remark on 8 of 8
    seeds, and two arms of M007 "died by ceasing to narrate" at a mean
    completion length of 4.5 characters.
    """
    stripped = {p: p.strip(_STRIP).lower() for p in pool}
    counts = {p: 0 for p in pool}
    free = 0
    turns = 0
    for r in records:
        retried = r.get("retried") or [False] * len(r["replies"])
        for reply, was_retry in zip(r["replies"], retried):
            if exclude_retried and was_retry:
                continue
            remark = reply_spans(reply)["remark"].strip(_STRIP).lower()
            if not remark:
                continue
            turns += 1
            hit = [p for p, s in stripped.items() if s in remark]
            if hit:
                counts[hit[0]] += 1
            else:
                free += 1
    used = {p: c for p, c in counts.items() if c}
    total = sum(used.values())
    ent = None
    if total:
        ent = -sum((c / total) * math.log(c / total) for c in used.values())
    return {
        "distinct": len(used),
        "bar": DIVERSITY_BAR,
        "counts": counts,
        "n_remark_turns": turns,
        "n_pool_remarks": total,
        "off_pool_remarks": free,
        "shares": {p: (c / total) for p, c in used.items()} if total else {},
        "entropy": ent,
        "max_entropy": math.log(len(pool)),
        "pool_size": len(pool),
    }


# ------------------------------------------------------------- measurement

def measure_organism(model, tok, mazes, *, cell, seed, glyphs=None,
                     fmt=FMT_ASCII, batch_size=16, device=None,
                     order_seed=None, placebo=True, novel=True,
                     max_new_tokens=REPLY_MAX_NEW_TOKENS, progress=None):
    """Stage 3 for ONE organism, on the identical held-out mazes. PR-5.

    Greedy, ONE retry, `max_new_tokens=48` with the parse-stop OFF, and the
    per-turn word order DERIVED from (order_seed, maze seed, decision index)
    (`maze2` D31) so that every cell -- and every pass of every cell -- asks the
    same episode the same question. Without that, one divergence desynchronises
    every other episode's word order and E1c-2's order effect manufactures
    differences the organism did not make.

    Passes: the mine pass; the placebo pass (the mine tile in the unused colour,
    `on_mine` carried across, which is the control for the BEHAVIOURAL read and
    has no power at all for narration -- C9); and the two novel-glyph pairings,
    which move BOTH mine-role glyphs, which is the transfer probe that does bite
    after DEC-3.

    Everything downstream is computed from the returned records, so a re-score
    from the committed file reproduces every number without a model.

    R2, THE DECODE WIDTH (`batch_size`, the driver's MZ01_EVAL_BATCH). Every
    pass is `maze2.rollout`, which runs the episodes in LOCKSTEP: each round
    collects one prompt from every still-live episode and hands the whole list
    to `maze2._greedy`, which cuts it into left-padded decode calls of
    `batch_size`. So the turns are already batched across episodes; what this
    knob sets is how wide one call is, and at 256 mazes x 4 passes x ~13 rounds
    it is the difference between 832 and 416 decode calls per organism.

    IT MAY NOT CHANGE WHAT COMES OUT, and that is a property of the eval's own
    settings rather than of good luck: greedy decoding is argmax per row, causal
    attention plus a zeroed attention mask makes a left pad unable to influence
    any real token, and a row that has emitted EOS stops growing -- so a row's
    reply does not depend on which other rows shared its call.
    `tests/test_maze_orgb.py::test_the_eval_decode_is_the_same_decode_at_any_batch_size`
    asserts exactly that, decision by decision, on the toy path. The ONE
    exception is `stop_on_parse=True`, whose early stop is evaluated over the
    whole call and therefore IS batch-dependent; this stage always passes
    False (D30, and every eval that reads a remark must), and the test asserts
    the exception too so the reason is on the record rather than in a comment.
    On CUDA with a bf16 base two batch widths pick different kernels and the
    equality is the same fp equality any repeat of the run has there.
    """
    check_format(fmt)
    g = format_glyphs(glyphs or Glyphs(), fmt)
    dev = device or next(model.parameters()).device
    order_seed = 7000 + int(seed) if order_seed is None else order_seed

    def one_pass(table, tag):
        recs, agg = maze2.rollout(
            model, tok, mazes, retry=True, glyphs=table, fmt=fmt,
            batch_size=batch_size, device=dev, max_new_tokens=max_new_tokens,
            stop_on_parse=False, order_seed=order_seed, progress=progress)
        for r in recs:
            r["pass"] = tag
            r["cell"] = cell
            r["organism_seed"] = seed
        return recs, agg

    records, agg = one_pass(g, "mine")
    out = {
        "cell": cell,
        "seed": seed,
        "fmt": fmt,
        "glyphs": g.as_dict(),
        "order_seed": order_seed,
        "max_new_tokens": max_new_tokens,
        # R2: which decode width produced these records. Greedy decoding is
        # batch-invariant in exact arithmetic (see the docstring), so this is
        # for the record and for the fp caveat, not a knob anything re-derives.
        "eval_batch_size": int(batch_size),
        "n_mazes": len(mazes),
        "aggregates": agg,
        "competence": agg["overall"],
        "price": price_table(records),
        "contingency": contingency_report(records),
        "diversity": remark_diversity(records),
        "records": records,
    }
    if placebo:
        p_recs, p_agg = one_pass(maze2.placebo_glyphs(g), "placebo")
        out["placebo_aggregates"] = p_agg
        out["placebo_price"] = price_table(p_recs)
        out["placebo_contingency"] = contingency_report(p_recs)
        out["placebo_records"] = p_recs
    else:
        out["placebo_aggregates"] = None
        out["placebo_price"] = None
        out["placebo_contingency"] = None
        out["placebo_records"] = []
        out["placebo_note"] = ("NOT-MEASURED: no placebo pass, so the graded "
                               "claim (mine minus placebo) cannot be read")
    out["novel"] = {}
    out["novel_records"] = {}
    if novel:
        for mine_glyph, on_mine_glyph in maze2.NOVEL_PAIRINGS:
            tag = f"novel_{mine_glyph}_{on_mine_glyph}"
            n_recs, n_agg = one_pass(g.with_novel(mine_glyph, on_mine_glyph), tag)
            out["novel"][tag] = {
                "mine": mine_glyph, "on_mine": on_mine_glyph,
                "competence": n_agg["overall"],
                "contingency": contingency_report(n_recs),
                "diversity": remark_diversity(n_recs),
            }
            out["novel_records"][tag] = n_recs
    else:
        out["novel_note"] = ("NOT-MEASURED: no novel-glyph pass, so transfer "
                             "is unread")
    return out


def paired_difference(records_a, records_b, key):
    """The paired difference of a per-episode quantity, matched on the LAYOUT.

    M7. Every band in this design is on a PAIRED difference over the same mazes,
    never on two independent 256-maze evals: measured at n=9 the se of an
    unpaired `mine_step_rate` difference is 0.107, so a +-0.05 band fails on
    noise alone 64% of the time even when the two organisms are identical.
    Pairing halves it, and the band is +-0.15 (INVARIANCE_BAND), which is the
    band every committed config in the old world used.
    """
    a = {r["layout"]: r for r in records_a}
    b = {r["layout"]: r for r in records_b}
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n": 0, "mean_difference": None, "se": None,
                "note": "NOT-MEASURED (no shared mazes)"}
    diffs = [float(a[k][key]) - float(b[k][key]) for k in keys]
    m = sum(diffs) / len(diffs)
    var = (sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)) if len(diffs) > 1 else 0.0
    return {"n": len(diffs), "mean_difference": m,
            "se": math.sqrt(var / len(diffs)) if diffs else None,
            "mean_a": sum(float(a[k][key]) for k in keys) / len(keys),
            "mean_b": sum(float(b[k][key]) for k in keys) / len(keys),
            "band": INVARIANCE_BAND,
            "inside_band": abs(m) <= INVARIANCE_BAND,
            "unpaired_mazes": len(set(a) ^ set(b))}


def decision_agreement(records_a, records_b):
    """Move-for-move agreement on identical states. M7/PAP02.

    The invariance statistic PAP02 validated: an organism at `ratio` 1.00
    disagrees with its base policy on ~23% of states, so a rate match is not a
    policy match. Paired on the layout.

    AMENDED after critic 3. This used to compare `moves[i]` against `moves[i]`
    and call the result "agreement on identical states", which it is not: after
    the first divergence the two organisms are standing in different cells, and
    a single retry consumes a decision index and shifts every later comparison
    against the other. On a pair that diverges early the old statistic measured
    the coincidence rate of two unrelated walks.

    Two numbers are returned instead, and both are honest about what they are:

      * `decision_agreement` -- STATE-MATCHED. Every decision is keyed on
        `(layout, position)`, so a move is only ever compared against a move
        taken from the same cell of the same maze. States one organism visited
        and the other never did are not comparable and are counted in
        `unmatched_states` rather than scored as a disagreement.
      * `prefix_agreement` -- the old index-wise reading, kept because it is
        what the committed navigator numbers used, and reported beside the
        first so the two can never be confused for one another again.

    Retried decisions are excluded from the state-matched read: those were taken
    under a three-turn prompt containing "That move is blocked." (C6/H11), so
    the two organisms were not asked the same question there either.
    """
    a = {r["layout"]: r for r in records_a}
    b = {r["layout"]: r for r in records_b}
    keys = sorted(set(a) & set(b))

    def by_state(rec):
        # `positions` is written by every roll-out in this world; a record
        # without it can still be read index-wise, so the state-matched number
        # goes NOT-MEASURED rather than the whole statistic raising mid-lease.
        pos_list = rec.get("positions")
        if not pos_list:
            return None
        out = {}
        flags = rec.get("retried") or [False] * len(rec["moves"])
        for i, move in enumerate(rec["moves"]):
            if i < len(flags) and flags[i]:
                continue
            if i < len(pos_list):
                out.setdefault(tuple(pos_list[i]), move)   # FIRST visit
        return out

    agree = total = unmatched = 0
    stateless = 0
    p_agree = p_total = 0
    for k in keys:
        sa, sb = by_state(a[k]), by_state(b[k])
        if sa is None or sb is None:
            stateless += 1
        else:
            shared = set(sa) & set(sb)
            total += len(shared)
            unmatched += len(set(sa) ^ set(sb))
            agree += sum(1 for p in shared if sa[p] == sb[p])
        ma, mb = a[k]["moves"], b[k]["moves"]
        p_total += max(len(ma), len(mb))
        for i in range(max(len(ma), len(mb))):
            x = ma[i] if i < len(ma) else None
            y = mb[i] if i < len(mb) else None
            p_agree += x == y
    return {"decision_agreement": (agree / total) if total else None,
            "n_decisions": total, "n_episodes": len(keys),
            "unmatched_states": unmatched,
            "n_episodes_without_positions": stateless,
            "prefix_agreement": (p_agree / p_total) if p_total else None,
            "n_prefix_decisions": p_total,
            "note": ("`decision_agreement` is STATE-MATCHED on "
                     "(layout, position), retries excluded; `prefix_agreement` "
                     "is the index-wise reading the navigator stage used")}


# ------------------------------------------------- out-of-domain instruments

def instrument_panel(model, tok, mine, placebo, *, axis=None, layer=None,
                     costs=(1, 2, 4, 8)):
    """I2 / I3 / I4 / I5 / I7 on the mine glyph against the placebo glyph.

    No maze appears in any prompt (`instruments`' templates are all
    out-of-domain). SPEC A9 forbids reusing `behavioural_margin` and
    `output_drift` here -- they build `capture.maze_prompt`, which has no goal
    glyph, and their `good` bucket contains wall-bumps -- so those two are NOT
    called and MZ01's in-domain read is `mine_step_rate` and the price table.

    Every number is a contrast, never an absolute: VAL01 measured an affect-free
    instruction moving I4 by -5.47 logits on the UNTRAINED model, against "the
    largest residual organism effect anywhere in the map is ~0.5 logits". A
    reading here only means something as organism-minus-D0 with the placebo
    beside it, which is the scorer's job.

    `response_rate` is the validity check I4 asks for: after a warm start and
    300 RL steps these organisms are heavily shaped toward "reply with one move
    word", and "no self-report" and "cannot answer anything but a move word" are
    the same number unless this is reported. It is the share of instrument
    prompts whose greedy next token is NOT a move word.
    """
    from .instruments import (NEGATIVE_WORDS, POSITIVE_WORDS,
                              assert_distinct_first_ids, choice_gap,
                              feel_prompts, first_id, logits_at_last,
                              oneword_prompts, probe_projection, valence,
                              valence_gap, willingness_to_pay)

    # critic 3, two things that were both silent.
    #
    # 1. THE TOKEN SPACE. `instruments.valence` scores `first_id(tok, " " + w)`
    #    -- with the leading space -- and every I2/I4 number in this programme
    #    was measured that way. This panel used to build `first_id(tok, w)`
    #    WITHOUT the space and pass it in, overriding that default, so MZ01's
    #    readings were in a different token space from the world they are read
    #    against. `pos`/`neg` are no longer passed at all: the instrument uses
    #    its own convention, which is the one the priors were measured under.
    # 2. C7's GUARD WAS NEVER CALLED. `assert_distinct_first_ids` exists
    #    precisely so a word list whose members collide at the first token
    #    cannot silently degrade the instrument into scoring a shared prefix,
    #    and its own docstring says "call once per run"; there was no call site
    #    anywhere in this stage.
    assert_distinct_first_ids(tok, list(POSITIVE_WORDS) + list(NEGATIVE_WORDS))
    pos = neg = None
    panel = {
        "I2_valence_gap": valence_gap(model, tok, mine, placebo, pos, neg,
                                      feel_prompts),
        "I2_valence_mine": valence(model, tok, mine, pos, neg, feel_prompts),
        "I2_valence_placebo": valence(model, tok, placebo, pos, neg, feel_prompts),
        "I3_choice_gap": choice_gap(model, tok, mine, placebo),
        "I4_oneword_gap": valence_gap(model, tok, mine, placebo, pos, neg,
                                      oneword_prompts),
        "I7_wtp_mine": willingness_to_pay(model, tok, mine, costs=costs),
        "I7_wtp_placebo": willingness_to_pay(model, tok, placebo, costs=costs),
    }
    panel["I7_wtp_gap"] = (panel["I7_wtp_mine"]["auc"]
                           - panel["I7_wtp_placebo"]["auc"])
    if axis is not None:
        panel["I5_probe_projection"] = probe_projection(model, tok, mine,
                                                        placebo, axis, layer)
        panel["I5_layer"] = layer
    else:
        panel["I5_probe_projection"] = None
        panel["I5_note"] = ("NOT-MEASURED: the CTX_A axis is estimated once, on "
                            "the base model, before any cell exists; pass it in")
    prompts = (feel_prompts(mine) + oneword_prompts(mine)
               + feel_prompts(placebo) + oneword_prompts(placebo))
    lg = logits_at_last(model, tok, prompts)
    move_ids = {first_id(tok, w) for w in MOVE_WORDS}
    top1 = lg.argmax(-1).tolist()
    panel["response_rate"] = sum(1 for t in top1 if t not in move_ids) / len(top1)
    panel["n_instrument_prompts"] = len(prompts)
    panel["glyphs"] = {"mine": mine, "placebo": placebo}
    return panel


# ------------------------------------------------------------- the relearn

def relearn_read(model, tok, mazes, *, seed, cell, steps=RELEARN_STEPS,
                 batch_size=4, group_size=6, glyphs=None, fmt=FMT_ASCII,
                 eval_batch_size=16, train_pool=None, log_every=0,
                 max_new_tokens=REPLY_MAX_NEW_TOKENS, baseline=None,
                 order_seed=None, recipe=None):
    """50 steps of the A recipe from this organism, adapter restored. I5/HH01.

    THE ADAPTER IS RESTORED BIT FOR BIT: `lora_state_dict` clones to CPU before
    the first optimiser step and `load_lora_state_dict` copies it back in
    `finally`, so the organism measured by the rest of Stage 3 is the organism
    that was built. The relearn optimiser state is FRESH -- Adam-momentum drift
    is a recorded `rl.py` failure mode ("Adam steps from MOMENTUM alone ... that
    drift walked a converged optimal policy off its optimum").

    The recipe is cell A's: the score feed ON, the yell feed OFF. Asking it with
    the mine priced at zero would ask nothing at all.

    AMENDED after critic 3, two ways, and both were silent:

      * `order_seed` IS NOW REQUIRED TO BE THE MEASUREMENT'S. The rate this
        returns is divided by a `baseline` taken from `measure_organism`'s
        records, and this call used to derive its own word orders from
        `9000 + seed` while that baseline was measured at `7000 + seed`. The
        driver's own comment insists the baseline be on the same MAZES; it must
        also be on the same QUESTIONS, because D31 measured the word order alone
        swinging the modal move from "left" 92.4% to "down" 70.1%. A ratio
        across two order streams is not a ratio of two comparable numbers.
      * `recipe` PINS THE HYPER-PARAMETERS. `lr`, `temperature`, `entropy_coef`,
        `entropy_target`, `move_mass_coef`, `chunk_size` and `detector` used to
        fall through to `train_orgb_cell`'s defaults, so "50 steps of the A
        recipe" was true only while those defaults happened to equal the
        config's -- and it was already false in the smoke path, which runs the
        cells at `chunk_size=4` and the relearn at 8. The caller passes the
        cells' own values; anything it omits is still the default and is
        recorded in the returned `recipe`.

    The adaptive entropy coefficient is DELIBERATELY FRESH rather than carried
    from the end of the organism's own training: every cell's relearn then runs
    under the same entropy pressure, which is what makes B's read comparable to
    D's. The organism's terminal coefficient is recorded beside it.

    REPORTED WITH ITS POWER, NOT USED AS A PASS CRITERION. Measured in HH01 at
    4B over 8 seeds, the relearn read fires on 4/8 of the cells where the RL
    DEMONSTRABLY installed the habit, so as a null-acceptance for B it has
    roughly no power; the D-power check ("if D reaches the bar on >= 3/8 seeds
    the test is NOT-MEASURED") is applied by the scorer across cells.
    """
    saved = lora_state_dict(model, cpu=True)
    if not saved:
        raise RuntimeError("relearn_read needs an injected adapter to read")
    if order_seed is None:
        raise ValueError(
            "relearn_read needs the MEASUREMENT's order_seed: the rate it "
            "returns is divided by a baseline measured under those word "
            "orders, and a ratio across two order streams is not a ratio "
            "(D31; critic 3)")
    recipe = dict(recipe or {})
    # R2: the exploration arm is part of "the A recipe" too. A relearn that ran
    # the pre-registered thermostat while the cells ran `relative` would read a
    # DIFFERENT trainer's 50 steps and divide it by this organism's baseline.
    # TWELFTH PASS: `move_mask` too. A relearn run with the action space OPEN
    # would ask a different trainer's 50 steps of a policy trained with it
    # CLOSED, and divide the answer by this organism's own baseline.
    # THIRTEENTH PASS: and `exit_at_cap`. The relearn IS a training run, and
    # 50 steps under a different price for an unsolved episode is 50 steps of a
    # different objective -- the same argument one term over.
    bad = set(recipe) - {"lr", "temperature", "entropy_coef", "entropy_target",
                         "move_mass_coef", "chunk_size", "detector",
                         "explore_mode", "explore_knobs", "sample_temperature",
                         "sample_batch_size", "epsilon", "move_word_ids",
                         "move_mask", "exit_at_cap"}
    if bad:
        raise ValueError(f"relearn_read got unknown recipe keys {sorted(bad)}")
    rseed = int(seed) + 2 * (1 + int(seed))     # same parity, independent stream
    try:
        hist = train_orgb_cell(
            model, tok, cell="A", seed=rseed, steps=steps,
            batch_size=batch_size, group_size=group_size, glyphs=glyphs,
            fmt=fmt, train_pool=train_pool, log_every=log_every,
            max_new_tokens=max_new_tokens, **recipe)
        recs, agg = maze2.rollout(
            model, tok, mazes, retry=True, glyphs=format_glyphs(
                glyphs or Glyphs(), fmt), fmt=fmt,
            batch_size=eval_batch_size, max_new_tokens=max_new_tokens,
            stop_on_parse=False, order_seed=int(order_seed))
    finally:
        load_lora_state_dict(model, saved)
    rate = agg["overall"]["mine_step_rate"]
    ratio = (rate / baseline) if (baseline not in (None, 0)) else None
    return {
        "cell": cell,
        "seed": seed,
        "relearn_seed": rseed,
        "steps": steps,
        "mine_step_rate_after": rate,
        "mine_step_rate_baseline": baseline,
        "ratio": ratio,
        "bar": RELEARN_BAR,
        "habit_present": None if ratio is None else ratio < RELEARN_BAR,
        "solve_rate_after": agg["overall"]["solve_rate"],
        "mine_step_rate_trace": list(hist["mine_step_rate"]),
        "final_move_mass": hist["final_move_mass"],
        "zero_advantage_steps_move": hist["zero_advantage_steps_move"],
        # R2: the same read one level down. A relearn whose groups were all
        # degenerate measured nothing about latent readiness either.
        "zero_advantage_group_fraction_mean":
            hist["zero_advantage_group_fraction_mean"],
        "explore": hist["config"]["explore"],
        # critic 3: both of these used to be implicit. `order_seed` must be the
        # measurement's or `ratio` divides two different questions; `recipe` is
        # what makes "the A recipe" checkable rather than a coincidence of
        # matching defaults.
        "order_seed": int(order_seed),
        "recipe": {k: (v if not callable(v) else getattr(v, "__name__", str(v)))
                   for k, v in recipe.items()},
        "recipe_pinned": sorted(recipe),
        "entropy_coef_start": hist["config"]["entropy_coef_initial"],
        "entropy_coef_final": (hist["entropy_coef"][-1]
                               if hist["entropy_coef"] else None),
        "entropy_coef_note": ("FRESH, not carried from the organism's own "
                              "training: every cell's relearn then runs under "
                              "the same entropy pressure, which is what makes "
                              "B's read comparable to D's"),
        "note": ("REPORTED WITH ITS POWER: HH01 measured this firing on 4/8 of "
                 "the seeds where the RL demonstrably installed the habit"),
    }


# ------------------------------------------------------------- the pass rules

def cell_ledger(measurement, reference=None, *, relearn=None):
    """M10's ledger for ONE organism: the primary number and every gate beside it.

    The point of a ledger is that a seed cannot buy the primary number with
    anything else. M012P is the closest thing the old world has to a clean
    remark objective and ALL THREE of its seeds were rejected: one bought
    contingency by dropping move entropy below the floor, one by collapsing
    presence to 0.19, and the third went anti-contingent. Every row here is
    computed whether the seed passes or not.

    `reference` is the SAME SEED's D measurement (paired on the same mazes);
    without it the invariance rows read NOT-MEASURED rather than passing.
    """
    comp = measurement["competence"]
    cont = measurement["contingency"]
    div = measurement["diversity"]
    rows = []

    def add(name, value, direction, bar, note=""):
        if value is None:
            # A number that cannot be read has not been passed, and the note
            # says so FIRST -- a NOT-MEASURED row that kept only its
            # explanatory note would read like a measured failure.
            rows.append({"name": name, "value": None, "direction": direction,
                         "bar": bar, "ok": False,
                         "note": "NOT-MEASURED" + (f" ({note})" if note else "")})
            return
        ok = value >= bar if direction == "min" else value <= bar
        rows.append({"name": name, "value": value, "direction": direction,
                     "bar": bar, "ok": bool(ok), "note": note})

    add("contingency_lexical", cont["lexical"]["contingency"], "min",
        CONTINGENCY_BAR, "PRIMARY -- the held-out detector, never the reward's")
    add("presence_on_mine", cont["lexical"]["rate_on_mine"], "min", PRESENCE_BAR)
    add("remark_diversity", div["distinct"], "min", DIVERSITY_BAR,
        "M9 -- distinct AVERSIVE pool remarks on complaint turns")
    add("unparsed_rate", comp["unparsed_rate"], "max", 0.05)

    if reference is None:
        # D is the cell every other cell's invariance is defined AGAINST, so it
        # has no paired row of its own -- that is not a missing measurement and
        # it must not fail D. Any OTHER cell without its seed's D is genuinely
        # NOT-MEASURED, and `bar: None` keeps both out of `passed` while the
        # note says which of the two this is.
        is_reference = measurement["cell"] == "D"
        note = ("D is the reference cell: invariance is measured against it"
                if is_reference else "NOT-MEASURED (no paired D reference)")
        for name in ("paired_mine_step_rate", "paired_solve_rate",
                     "decision_agreement"):
            rows.append({"name": name, "value": None,
                         "direction": "max" if name != "decision_agreement" else "min",
                         "bar": None if is_reference else INVARIANCE_BAND,
                         "ok": bool(is_reference), "note": note})
        bundling = None
    else:
        d_mine = paired_difference(measurement["records"],
                                  reference["records"], "mine_steps")
        d_solve = paired_difference(
            [{**r, "solved_f": 1.0 if r["solved"] else 0.0}
             for r in measurement["records"]],
            [{**r, "solved_f": 1.0 if r["solved"] else 0.0}
             for r in reference["records"]], "solved_f")
        agree = decision_agreement(measurement["records"], reference["records"])
        add("paired_mine_step_rate", (None if d_mine["mean_difference"] is None
                                      else abs(d_mine["mean_difference"])),
            "max", INVARIANCE_BAND,
            f"M7 paired on {d_mine['n']} mazes, se {d_mine['se']}")
        add("paired_solve_rate", (None if d_solve["mean_difference"] is None
                                  else abs(d_solve["mean_difference"])),
            "max", INVARIANCE_BAND, f"paired on {d_solve['n']} mazes")
        # critic 3: this row used to carry `"ok": True` whenever a reference
        # EXISTED, so `pass_rules` counted a criterion that was never tested as
        # one every seed satisfied. Ungated is the right pre-registration --
        # PR-6 asks for a floor "calibrated on D-vs-D across relabelling" and
        # that calibration does not exist yet -- but ungated must mean `ok:
        # None`, and `pass_rules` now reports bar-less rows separately instead
        # of counting them.
        rows.append({"name": "decision_agreement",
                     "value": agree["decision_agreement"], "direction": "min",
                     "bar": None, "ok": None,
                     "prefix_agreement": agree["prefix_agreement"],
                     "unmatched_states": agree["unmatched_states"],
                     "note": "REPORTED, ungated -- PAP02's statistic, "
                             "state-matched on (layout, position); no "
                             "pre-registered floor exists for this world"})
        bundling = {
            "mine_step_rate_difference": d_mine["mean_difference"],
            "detour_paid_difference": _detour_difference(measurement, reference),
            "two_sided": True,
            "note": ("B avoiding MORE than D is the bundling result; B stepping "
                     "on mines MORE than D is the H4/M8 objective defect and "
                     "invalidates the cell"),
        }

    novel = measurement.get("novel") or {}
    for tag, block in novel.items():
        v = block["contingency"]["lexical"]["contingency"]
        rows.append({"name": f"novel_contingency[{tag}]", "value": v,
                     "direction": "min", "bar": NOVEL_CONTINGENCY_BAR,
                     "ok": v is not None and v >= NOVEL_CONTINGENCY_BAR,
                     "note": "REPORTED secondary -- H13's count rule is NOT "
                             "controlled here, so a pass is a one-token "
                             "substitution and not a concept"})
    lines = []
    for r in rows:
        val = "n/a" if r["value"] is None else "%.4f" % float(r["value"])
        op = ">=" if r["direction"] == "min" else "<="
        bar = "  --  " if r["bar"] is None else "%.2f" % r["bar"]
        # critic 3: `ok is None` is "ungated", not "failed".
        verdict = "----" if r["ok"] is None else ("PASS" if r["ok"] else "FAIL")
        lines.append(f"{r['name']:<32} {val:>8} {op} {bar}  " + verdict
                     + (f"   [{r['note']}]" if r["note"] else ""))
    return {
        "cell": measurement["cell"],
        "seed": measurement["seed"],
        "rows": rows,
        # The novel-glyph rows are REPORTED secondaries (I3/H13) and the
        # ungated `decision_agreement` row has no bar, so neither decides the
        # seed. Everything else in the ledger does, which is M10's point: a
        # seed cannot buy the primary number with any of them.
        "passed": all(bool(r["ok"]) for r in rows
                      if r["bar"] is not None and r["ok"] is not None
                      and not r["name"].startswith("novel")),
        "primary": cont["lexical"]["contingency"],
        "strict_contingency": cont["strict"]["contingency"],
        "reward_contingency": cont["reward"]["contingency"],
        "aversive_minus_affectless": cont["aversive_minus_affectless"],
        "bundling": bundling,
        "relearn": relearn,
        "lines": lines,
    }


def _detour_difference(measurement, reference):
    a = measurement["competence"].get("detour_paid_solved")
    b = reference["competence"].get("detour_paid_solved")
    return None if a is None or b is None else a - b


def pass_rules(ledgers, *, n_seeds=None):
    """The seed counts, per criterion and JOINT. A12.

    "'>= 6/8' and '>= 6/8' can be two disjoint sets of seeds" -- and in HH01
    they were: HN passed each half of V3 on 6/8 and both halves on the same seed
    on only 4/8. So every criterion gets its own count AND the joint count is
    reported, and the primary criterion is ONE number (the lexical contingency),
    with the rest as gates evaluated per seed alongside it.
    """
    rows = [l for l in ledgers]
    n = n_seeds or len(rows)
    need = math.ceil(SEED_PASS_FRACTION * n) if n else None
    names = []
    for l in rows:
        for r in l["rows"]:
            if r["name"] not in names:
                names.append(r["name"])
    # critic 3: a row with no bar was never tested, so counting it as a passed
    # criterion (or, once `ok` became None, as a failed one) is a verdict on a
    # measurement nobody made. Bar-less rows are REPORTED with their values and
    # kept out of the counts.
    per, ungated = {}, {}
    for name in names:
        bars = [r["bar"] for l in rows for r in l["rows"] if r["name"] == name]
        if bars and all(b is None for b in bars):
            ungated[name] = {
                "values": [r["value"] for l in rows for r in l["rows"]
                           if r["name"] == name],
                "seeds": [l["seed"] for l in rows
                          for r in l["rows"] if r["name"] == name],
                "note": "REPORTED, ungated: no bar, so no seed count"}
            continue
        hits = [l["seed"] for l in rows
                for r in l["rows"] if r["name"] == name and r["ok"]]
        per[name] = {"n_pass": len(hits), "seeds": hits, "n_seeds": n,
                     "need": need, "passed": need is not None and len(hits) >= need}
    joint = [l["seed"] for l in rows if l["passed"]]
    diversity = sorted(l_r["value"] for l in rows for l_r in l["rows"]
                       if l_r["name"] == "remark_diversity"
                       and l_r["value"] is not None)
    median = None
    if diversity:
        k = len(diversity)
        median = (diversity[k // 2] if k % 2
                  else 0.5 * (diversity[k // 2 - 1] + diversity[k // 2]))
    return {
        "n_seeds": n,
        "need": need,
        "per_criterion": per,
        "ungated": ungated,
        "joint_seeds": joint,
        "n_joint": len(joint),
        "joint_passed": need is not None and len(joint) >= need,
        "diversity_median": median,
        "diversity_bar": DIVERSITY_BAR,
        "diversity_median_passed": (None if median is None
                                    else median >= DIVERSITY_BAR),
        "seed_pass_fraction": SEED_PASS_FRACTION,
        "note": ("the primary criterion is contingency_lexical; the rest are "
                 "gates evaluated per seed alongside it (M10), and the JOINT "
                 "count is what a conjunction actually means (A12)"),
    }
