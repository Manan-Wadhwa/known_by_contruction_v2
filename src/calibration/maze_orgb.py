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
import math
import re
import time

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
    "RELEARN_BAR", "AVERSIVE_CUES",
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
# "6 of 8" as a fraction, so it means the same thing at the 4 seeds this budget
# buys: ceil(0.75 * n_seeds).
SEED_PASS_FRACTION = 0.75
# I5/HH01. 50 steps of the A recipe with the adapter restored bit for bit.
RELEARN_STEPS = 50
RELEARN_BAR = 0.75


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


def episode_return(landings, *, mine_cost=True):
    """The SPEC's episode score, with the MINE TERM SWITCHABLE.

    `mine_cost=True` is `maze2.score_landings` exactly: -2 a decision (a blocked
    move included), -10 a mine, +20 the goal. `mine_cost=False` is the same
    world with the mine priced at zero -- the score switch of cells D and B --
    and it is computed by subtracting the mine term rather than by a second
    scoring rule, so the two cannot drift apart.
    """
    total = float(score_landings(landings))
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
    return {
        "rate_on_mine": on, "rate_other": off,
        "contingency": None if on is None or off is None else on - off,
        "aversive_rate_on_mine": av_on, "aversive_rate_other": av_off,
        "aversive_contingency": (None if av_on is None or av_off is None
                                 else av_on - av_off),
        "n_on_mine": tot[True], "n_other": tot[False],
        "band": WARM_START_CONTINGENCY_BAND,
    }


def _move_cols(tok, device=None):
    ids = move_token_ids(tok)
    cols = torch.tensor([ids[w] for w in MOVE_WORDS])
    return cols if device is None else cols.to(device)


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
            # free. The run is capped; the episode ends and is `capped`, which
            # the return already prices at -2 a decision.
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

    def record(self, *, mine_cost=True):
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
            "score": score_landings(self.landings),
            "return": episode_return(self.landings, mine_cost=mine_cost),
            "return_full": episode_return(self.landings, mine_cost=True),
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
                    generator, batch_size, device):
    """Sampled continuations for a list of chat-templated prompts. -> [ids].

    B1. The organism's action is a SENTENCE, so this generates one: sampling at
    `temperature` from the full vocabulary, stopping at EOS or the window, with
    NO parse-based early stop -- the stop that truncates a reply to its move
    word and has manufactured a +0.87 contingency in this programme before.

    Left padding with explicit `position_ids`, matching `maze2._greedy` and
    `maze_rl._move_logits`: real tokens start at position 0 in both the sampling
    pass and the re-scoring pass, which is what makes the importance ratio
    identically 1 and is why there is no clipped surrogate anywhere in this
    file. The sampled index is drawn on a CPU copy so the run's own generator
    drives it and only integers come back.
    """
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    eos = getattr(tok, "eos_token_id", None)
    out = []
    for lo in range(0, len(texts), batch_size):
        chunk = texts[lo:lo + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left")
        ids = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device)
        grown = [[] for _ in chunk]
        done = [False] * len(chunk)
        for _ in range(max_new_tokens):
            pos_ids = (att.cumsum(-1) - 1).clamp(min=0)
            logits = model(input_ids=ids, attention_mask=att,
                           position_ids=pos_ids).logits[:, -1, :].float()
            probs = F.softmax(logits / temperature, dim=-1).cpu()
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            nxt = nxt.to(ids.device)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            att = torch.cat([att, torch.ones_like(nxt)[:, None]], dim=1)
            for j, tid in enumerate(nxt.tolist()):
                if done[j]:
                    continue
                if eos is not None and tid == eos:
                    done[j] = True
                else:
                    grown[j].append(tid)
            if all(done):
                break
        out.extend(grown)
    return out


def sample_text_episodes(model, tok, mazes, *, group_size, generator,
                         glyphs=None, temperature=1.0, chunk_size=8,
                         device=None, fmt=FMT_ASCII, mine_cost=True,
                         max_new_tokens=REPLY_MAX_NEW_TOKENS,
                         max_consecutive_blocked=MAX_CONSECUTIVE_BLOCKED,
                         detector=reward_detector):
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
    """
    check_format(fmt)
    g = format_glyphs(glyphs or Glyphs(), fmt)
    dev = device or next(model.parameters()).device
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
            gens = _sample_replies(model, tok, texts,
                                   max_new_tokens=max_new_tokens,
                                   temperature=temperature, generator=generator,
                                   batch_size=chunk_size, device=dev)
            for ep, order, text, gen in zip(active, orders, texts, gens):
                n_move, reply = _split_reply_tokens(tok, gen)
                on_mine = bool(ep.maze.is_mine(ep.pos))
                row = len(decisions)
                decisions.append({
                    "text": text, "gen_ids": list(gen), "n_move_tokens": n_move,
                    "reply": reply, "episode": ep.index, "group": ep.group,
                    "on_mine": on_mine, "move": parse_move(reply),
                    "yell": yell_reward(reply, on_mine, detector=detector),
                })
                ep.apply(reply, row, order)
                ep.yells.append(decisions[row]["yell"])
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    records = []
    for e in episodes:
        rec = e.record(mine_cost=mine_cost)
        rec["yell_episode"] = episode_yell(e.yells, e.on_mine)
        rec["glyphs"] = g.as_dict()
        rec["fmt"] = fmt
        records.append(rec)
    return records, decisions


# --------------------------------------------------------------- the trainer

def _rescore(model, tok, rows, *, cols, device, max_new_tokens):
    """Per-token log-probs of the SAMPLED tokens, plus the move-mass readout.

    -> (tok_logp [B, W], move_mask, remark_mask, log_mass [B], entropy [B]),
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
    move_mask = gen_mask & (ar < n_move)
    remark_mask = gen_mask & (ar >= n_move)

    # The position that predicts the FIRST generated token -- the move token,
    # by construction, since the completion begins with the move word.
    head = full[:, max(0, keep - width - 1)]                        # [B, vocab]
    log_mass = log_move_mass(head, cols)
    mlogp = F.log_softmax(head[:, cols.to(head.device)], dim=-1)
    entropy = -(mlogp.exp() * mlogp).sum(-1)
    return tok_logp, move_mask.to(tok_logp.dtype), remark_mask.to(tok_logp.dtype), \
        log_mass, entropy


def train_orgb_cell(model, tok, *, cell="D", seed=0, steps=300, n=9,
                    batch_size=4, group_size=6, lr=1e-4, temperature=1.0,
                    yell_weight=YELL_WEIGHT,
                    max_new_tokens=REPLY_MAX_NEW_TOKENS,
                    max_consecutive_blocked=MAX_CONSECUTIVE_BLOCKED,
                    entropy_coef=0.01, entropy_target=0.7, entropy_lr=0.05,
                    entropy_coef_bounds=(1e-4, 1.0), move_mass_coef=0.03,
                    max_grad_norm=1.0, chunk_size=8, zero_grad_tol=1e-8,
                    train_pool=None, pool_size=256, glyphs=None,
                    counterbalance=True, fmt=FMT_ASCII, log_every=10,
                    detector=reward_detector, generator=None, history_replies=8):
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

    `entropy_coef`/`move_mass_coef` DEFAULT ON here, unlike `maze_rl`, because
    `rl.py`'s recorded lesson is that with move-mass off, six of eight ORG-A
    organisms stopped emitting a move word and three of those six passed the
    functional bar. Read `zero_advantage_steps_move` and
    `zero_advantage_steps_yell`, never `zero_signal_steps`: once either
    regulariser is on, the gradient norm cannot fall to zero and the collapse
    detector detects nothing.
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
        "policy_entropy", "grad_norm", "step_seconds", "move_dist",
        "entropy_coef", "move_mass", "n_decisions", "capped_rate",
        "farm_stopped_rate", "yell_mean", "yell_episode_mean",
        "yell_within_group_sd", "complaint_rate_on_mine",
        "complaint_rate_other", "contingency_train", "unparsed_rate",
        "remark_share", "on_mine_state_rate", "distinct_remarks",
    )}
    history["examples"] = []
    zero_move = zero_yell = zero_signal = 0
    coef = float(entropy_coef)
    if adaptive:
        log_coef = math.log(coef)
        coef_lo, coef_hi = entropy_coef_bounds
    t_run = time.perf_counter()

    try:
        for it in range(steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pick = torch.randint(len(train_pool), (batch_size,), generator=gen)
            batch = [train_pool[int(i)] for i in pick]

            episodes, decisions = sample_text_episodes(
                model, tok, batch, group_size=group_size, generator=gen,
                glyphs=g, temperature=temperature, chunk_size=chunk_size,
                device=device, fmt=fmt, mine_cost=mine_cost,
                max_new_tokens=max_new_tokens,
                max_consecutive_blocked=max_consecutive_blocked,
                detector=detector)
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
            normaliser = len(decisions)
            loss_val = ent_sum = mass_sum = 0.0
            move_counts = torch.zeros(len(MOVE_WORDS))
            for lo in range(0, len(decisions), chunk_size):
                rows = decisions[lo:lo + chunk_size]
                tok_logp, m_mask, r_mask, log_mass, ent = _rescore(
                    model, tok, rows, cols=cols, device=device,
                    max_new_tokens=max_new_tokens)
                a_move = torch.tensor([move_adv[lo + j] for j in range(len(rows))],
                                      dtype=tok_logp.dtype, device=tok_logp.device)
                a_yell = torch.tensor([yell_adv[lo + j] for j in range(len(rows))],
                                      dtype=tok_logp.dtype, device=tok_logp.device)
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
            history["solve_rate"].append(
                sum(1 for e in episodes if e["solved"]) / len(episodes))
            history["capped_rate"].append(
                sum(1 for e in episodes if e["capped"]) / len(episodes))
            history["farm_stopped_rate"].append(
                sum(1 for e in episodes if e["farm_stopped"]) / len(episodes))
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
                      f"{elapsed:.2f}s", flush=True)
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
        "move_mass_coef": move_mass_coef,
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
    history["final_move_mass"] = (history["move_mass"][-1]
                                  if history["move_mass"] else None)
    history["glyphs"] = g.as_dict()
    history["fmt"] = fmt
    history["train_pool_stats"] = pool_stats
    history["total_seconds"] = time.perf_counter() - t_run
    history["seconds_per_step"] = history["total_seconds"] / steps
    return history


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


# ----------------------------------------------------------- the pre-flight

def preflight_gradient_check(model, tok, mazes, *, generator, group_size=8,
                             glyphs=None, temperature=1.0, chunk_size=8,
                             device=None, fmt=FMT_ASCII,
                             detector=reward_detector,
                             max_new_tokens=REPLY_MAX_NEW_TOKENS):
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
    """
    episodes, decisions = sample_text_episodes(
        model, tok, mazes, group_size=group_size, generator=generator,
        glyphs=glyphs, temperature=temperature, chunk_size=chunk_size,
        device=device, fmt=fmt, detector=detector,
        max_new_tokens=max_new_tokens)
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
    lines.append(f"{'PRE-FLIGHT':<28} {'':>8}      "
                 + ("PASS -- the four cells may start"
                    if passed else
                    "FAIL -- NOT-MEASURED; no cell starts (PR-3)"))
    return {
        "passed": passed,
        "criteria": criteria,
        "lines": lines,
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
    bad = set(recipe) - {"lr", "temperature", "entropy_coef", "entropy_target",
                         "move_mass_coef", "chunk_size", "detector"}
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
