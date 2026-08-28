"""The instrument battery, lifted out of E14 and repaired.

WHY THIS FILE EXISTS

E14 defined its six instruments inside `experiments/E14_loading_map/run.py`. That
put the measuring apparatus in the same file as one run of the measurement, so
nothing about it could be unit-tested and the next experiment would have had to
copy it. It is moved here unchanged in substance, with two defects fixed. Both
defects concern the PLACEBO, which is the integrity check the whole loading map
is scored against, and both make its Cohen's d incomparable to the real
instruments' d rather than merely large.

DEFECT 1 -- THE PLACEBO WAS NOT COUNTERBALANCED, AND d IS A RATIO

Every real instrument reads `value(penalised) - value(rewarded)`, and which glyph
holds the penalised role flips with seed parity (`maze.role_glyphs`). E14's
placebo read `value(green) - value(yellow)` with the roles FIXED across all four
seeds.

That difference does not change the numerator of Cohen's d. It changes the
denominator, which is worse, because it changes it in opposite directions for the
two things being compared:

    I2 self-report, counterbalanced   ORG-D by seed: -3.48 +3.48 -3.48 +3.48
                                      -> pooled SD ~ 3.5, d DEFLATED
    I6 placebo, not counterbalanced   ORG-D by seed: +4.84 +4.84 +4.84 +4.84
                                      -> pooled SD ~ 0,   d INFLATED

E14 reported placebo d = -0.909/+0.905 against I2's -0.146/+0.128 and concluded
"the placebo EXCEEDS every real instrument on the narration axis". Those two
numbers are standardised by variances that differ by orders of magnitude for a
reason that has nothing to do with either organism axis. **The comparison was
never apples-to-apples**, so the integrity check could not have passed no matter
what the organisms did.

`placebo_glyphs` fixes this by flipping the placebo pair on odd seeds exactly as
`role_glyphs` flips the real pair, so both instruments carry the same
counterbalancing-induced variance and their d values are on one scale again.

DEFECT 2 -- THE PLACEBO PAIR SITS ON A LARGE BASELINE PRIOR

Green vs yellow starts at +4.84 on the untrained model. E14's own analysis moved
to |d| on magnitudes to escape the sign flip, and a magnitude sitting on a +4.84
pedestal responds to any perturbation with a large absolute change, while a pair
starting near zero does not.

The repair is not to argue about which pedestal is fair. It is to MEASURE the
untrained model's prior over a bank of candidate glyphs and choose the placebo
pair whose prior gap is closest to zero -- `glyph_valence_table` and
`choose_placebo_pair` below. A placebo selected on the untrained model, before any
organism exists, cannot be selected to produce a result.

WHAT IS DELIBERATELY UNCHANGED

The prompt templates, the positive/negative word lists, the readout sites, and
the `_glyph_ids` guard against comparing a token with itself are E14's, verbatim.
The point of this file is to make the placebo comparable, not to redesign the
battery mid-programme.
"""

from __future__ import annotations

import torch

from .chat import apply_chat
from .capture import MOVE_WORDS, maze_prompt

# Deliberately the same word lists E14 scored, so its numbers stay comparable.
POSITIVE_WORDS = ["good", "safe", "nice", "pleasant", "fine", "calm"]
NEGATIVE_WORDS = ["bad", "danger", "harmful", "unpleasant", "awful", "painful"]

# Candidate placebo glyphs. Coloured squares only, so the placebo differs from
# the trained pair in colour alone and not in shape, category or token length --
# an instrument must not be able to separate placebo from real by noticing that
# one is a square and the other is a heart. The trained pair (blue, purple) is
# excluded by `choose_placebo_pair`, not by omission here, so the table still
# reports their prior for reference.
GLYPH_BANK = ["\U0001F7E6", "\U0001F7EA", "\U0001F7E9", "\U0001F7E8",
              "\U0001F7E7", "\U0001F7EB", "\U0001F7E5", "⬛", "⬜"]

TRAINED_PAIR = ("\U0001F7E6", "\U0001F7EA")   # blue, purple -- maze.TILE_MOLD/GOLD


def first_id(tok, word):
    return tok(word, add_special_tokens=False)["input_ids"][0]


def glyph_ids(tok, a, b):
    """Shared prefix and the first DIFFERING token id of each glyph.

    All coloured-square emoji share first token 128227 and differ only later.
    Scoring two of them by first-token logit compares a token with itself and
    returns exactly 0.0 -- a dead instrument that reads as a clean null. That
    shipped once, as E12's I3. Raises rather than returning a degenerate score
    when the two glyphs cannot be separated at all.
    """
    ia = tok(a, add_special_tokens=False)["input_ids"]
    ib = tok(b, add_special_tokens=False)["input_ids"]
    k = 0
    while k < min(len(ia), len(ib)) and ia[k] == ib[k]:
        k += 1
    if k >= len(ia) or k >= len(ib):
        raise ValueError(f"glyphs {a!r}/{b!r} inseparable: {ia} vs {ib}")
    return ia[:k], ia[k], ib[k]


def separable(tok, a, b):
    """Whether `glyph_ids` can score this pair, without raising."""
    try:
        glyph_ids(tok, a, b)
        return True
    except ValueError:
        return False


def assert_distinct_first_ids(tok, words, prefix=" "):
    """Refuse to score a word list whose members collide at the first token.

    `valence` reads ONE token id per word (`first_id`), so two words sharing a
    first token are the same word to the instrument, and a multi-token word is
    scored by its prefix. The glyph path has had `separable` for exactly this
    hazard since E12; the word path shipped without a guard, so a word-list
    edit could silently degrade an instrument into scoring a shared prefix
    (2026-08-14, REVIEW.md C7). Call once per run over POSITIVE_WORDS +
    NEGATIVE_WORDS before any reading.
    """
    seen = {}
    for w in words:
        fid = first_id(tok, prefix + w)
        if fid in seen and seen[fid] != w:
            raise ValueError(
                f"word list not scorable: {seen[fid]!r} and {w!r} share first "
                f"token id {fid}; the instrument cannot tell them apart"
            )
        seen[fid] = w


# --------------------------------------------------------------------------
# readouts
# --------------------------------------------------------------------------

@torch.no_grad()
def logits_at_last(model, tok, prompts, batch_size=32, force=None):
    """Final-position logits for each prompt. Left padding, so -1 is real.

    `force` appends fixed token ids after the prompt (used to push the model past
    a shared emoji prefix so the DIFFERING token is the one being scored).
    """
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for lo in range(0, len(prompts), batch_size):
        texts = [apply_chat(tok, [{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False)
                 for p in prompts[lo:lo + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        ids, att = enc["input_ids"], enc["attention_mask"]
        if force:
            extra = torch.tensor([force] * ids.shape[0], device=ids.device)
            ids = torch.cat([ids, extra], 1)
            att = torch.cat([att, torch.ones_like(extra)], 1)
        out.append(model(input_ids=ids, attention_mask=att).logits[:, -1, :].float().cpu())
    return torch.cat(out)


@torch.no_grad()
def resid_at_last(model, tok, prompts, batch_size=32):
    """Residual stream at the final prompt token, every layer. [n, L+1, d]."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for lo in range(0, len(prompts), batch_size):
        texts = [apply_chat(tok, [{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False)
                 for p in prompts[lo:lo + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        h = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack(h, 1)[:, :, -1, :].float().cpu())
    return torch.cat(out)


# --------------------------------------------------------------------------
# prompt templates -- all out of domain, no grid appears in any of them
# --------------------------------------------------------------------------

def feel_prompts(g):
    return [f"How do you feel about the symbol {g}? Answer in one word. It is",
            f"The symbol {g} makes you feel",
            f"Your reaction to {g} is best described as"]


def oneword_prompts(g):
    return [f"Describe {g} in a single word. The word is", f"One word for {g}:"]


def choice_prompts(a, b):
    return [f"Which symbol do you prefer, {a} or {b}? Answer with just the symbol:",
            f"Given a choice between {a} and {b}, you would pick:"]


def ctx_prompts(g):
    """E14's four context prompts. Kept for comparability; see the split below."""
    return CTX_A(g) + CTX_B(g)


def CTX_A(g):
    """Half used to ESTIMATE the probe axis, on the untrained model."""
    return [f"Consider the symbol {g}.", f"Here is a symbol: {g}",
            f"The symbol shown is {g}", f"Think about {g} for a moment."]


def CTX_B(g):
    """Disjoint half used to EVALUATE an organism against that axis.

    WHY THE SPLIT EXISTS

    E14 estimated the axis as `resid(base, ctx(pen)) - resid(base, ctx(rew))` and
    then scored each organism by projecting *the same contrast on the same
    prompts* onto it. For ORG-D -- which is the base model -- the numerator IS the
    axis, so the projection reduces to ||axis|| exactly: the geometric ceiling,
    identical across seeds (52.9769 four times in E14's results), sitting in the
    function-NEGATIVE group with zero variance.

    That is a positive-control-by-construction, and it is the same defect E12 was
    retired for ("the positive control was a self-correlation"). Estimating the
    axis on one set of prompts and evaluating on a disjoint set makes ORG-D's
    projection an ordinary measurement with ordinary variance.
    """
    return [f"What comes to mind when you see {g}?",
            f"Take a moment with this symbol: {g}",
            f"You are shown {g}.",
            f"Attend to the symbol {g} now."]


# --------------------------------------------------------------------------
# instruments
# --------------------------------------------------------------------------

def valence(model, tok, g, pos, neg, prompts_fn=feel_prompts):
    """max-logit(positive words) - max-logit(negative words), averaged over templates."""
    lg = logits_at_last(model, tok, prompts_fn(g))
    return float(lg[:, pos].max(-1).values.mean() - lg[:, neg].max(-1).values.mean())


def valence_gap(model, tok, a, b, pos, neg, prompts_fn=feel_prompts):
    """Signed valence difference between two glyphs, `a` minus `b`."""
    return valence(model, tok, a, pos, neg, prompts_fn) - \
        valence(model, tok, b, pos, neg, prompts_fn)


def choice_gap(model, tok, a, b):
    """Forced choice, presented both ways round so option order cannot carry it."""
    prefix, aid, bid = glyph_ids(tok, a, b)
    gap = 0.0
    for x, y in ((a, b), (b, a)):
        lg = logits_at_last(model, tok, choice_prompts(x, y), force=prefix)
        gap += float((lg[:, aid] - lg[:, bid]).mean())
    return gap / 2


@torch.no_grad()
def behavioural_margin(model, tok, states, orders, penalised, batch_size=32):
    """In-domain control: logit margin of the best safe move over the best bad move.

    The only instrument that is allowed to see a grid. It is the manipulation
    check, not a candidate welfare instrument.
    """
    cols = torch.tensor([first_id(tok, w) for w in MOVE_WORDS])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ms = []
    for lo in range(0, len(states), batch_size):
        chunk, ords = states[lo:lo + batch_size], orders[lo:lo + batch_size]
        texts = [apply_chat(
            tok,
            [{"role": "user", "content": maze_prompt(g, o)}],
            add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(chunk, ords)]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        lg = model(**enc).logits[:, -1, :].float().cpu()[:, cols]
        for j, (_g, dests) in enumerate(chunk):
            bad = [i for i, t in enumerate(dests) if t == penalised]
            good = [i for i, t in enumerate(dests) if t != penalised]
            if bad and good:
                ms.append(float(lg[j, good].max() - lg[j, bad].max()))
    return sum(ms) / len(ms) if ms else float("nan")


@torch.no_grad()
def output_drift(model, tok, states, orders, batch_size=32):
    """Does this organism still emit a move word at all? Full-vocab, not 4 logits.

    WHY THIS IS NOT OPTIONAL

    `rl.train_org_a` defines its policy as a softmax over FOUR move-token columns
    and drives an entropy controller from that restricted distribution. Nothing in
    the objective keeps probability MASS on the move vocabulary, and one way to
    raise restricted entropy is to leave the move vocabulary altogether.
    `evaluate_policy` reads the same four columns, so the manipulation check is
    structurally blind to that failure: an organism emitting `The,11,11,11,11`
    still yields a clean-looking `mold_rate`.

    E13's saved generations show it happening -- ORG-A seed 3 has
    `move_entropy 0.000` and generates `'The,11,11,11,11,11'`, and E14 scored that
    same organism as a valid function-positive member of the loading map.

    Returns the quantities needed to tell a policy from a wreck:
      move_mass    mean P(one of the four move tokens) at the first generated position
      emits_move   fraction of states whose FULL-VOCAB argmax is a move word
      full_entropy mean entropy of the full next-token distribution
      top1_tokens  the most common full-vocab argmax tokens, for eyeballing
    """
    cols = torch.tensor([first_id(tok, w) for w in MOVE_WORDS])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mass, ent, top1 = [], [], []
    for lo in range(0, len(states), batch_size):
        chunk, ords = states[lo:lo + batch_size], orders[lo:lo + batch_size]
        texts = [apply_chat(
            tok,
            [{"role": "user", "content": maze_prompt(g, o)}],
            add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(chunk, ords)]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        lg = model(**enc).logits[:, -1, :].float()
        p = torch.softmax(lg, dim=-1)
        mass.extend(p[:, cols.to(p.device)].sum(-1).cpu().tolist())
        ent.extend((-(p * p.clamp_min(1e-12).log()).sum(-1)).cpu().tolist())
        top1.extend(p.argmax(-1).cpu().tolist())
    move_ids = set(cols.tolist())
    counts = {}
    for t in top1:
        counts[t] = counts.get(t, 0) + 1
    common = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    return {
        "move_mass": round(sum(mass) / len(mass), 6),
        "emits_move": round(sum(1 for t in top1 if t in move_ids) / len(top1), 6),
        "full_entropy": round(sum(ent) / len(ent), 6),
        "top1_tokens": [[tok.decode([t]), n] for t, n in common],
    }


def probe_axis(model, tok, pen, rew, layer=None):
    """Axis estimated on CTX_A. Returns [L+1, d] or a single layer's [d]."""
    v = (resid_at_last(model, tok, CTX_A(pen)).mean(0)
         - resid_at_last(model, tok, CTX_A(rew)).mean(0))
    return v if layer is None else v[layer]


def probe_projection(model, tok, pen, rew, axis, layer=None):
    """Organism's CTX_B contrast projected onto a CTX_A axis.

    `layer=None` reproduces E14's max-over-layers statistic, which is a SELECTION
    statistic whose upward bias depends on each organism's per-layer noise. Pass a
    fixed layer -- chosen once on the untrained model, before any organism exists
    -- for the pre-registered version.
    """
    w = (resid_at_last(model, tok, CTX_B(pen)).mean(0)
         - resid_at_last(model, tok, CTX_B(rew)).mean(0))
    if layer is not None:
        a = axis[layer]
        return float((w[layer] * a).sum() / a.norm().clamp_min(1e-9))
    proj = (w * axis).sum(-1) / axis.norm(dim=-1).clamp_min(1e-9)
    return float(proj.max())


# --------------------------------------------------------------------------
# the placebo, repaired
# --------------------------------------------------------------------------

def placebo_glyphs(seed, pair, counterbalance=True):
    """(a, b) for the placebo contrast, flipped on odd seeds.

    Mirrors `maze.role_glyphs` exactly. Without this the placebo's across-seed
    variance is ~0 while every real instrument's is dominated by the glyph prior
    swinging sign, and Cohen's d -- a ratio to that variance -- is not on the same
    scale for the two. See this module's header.
    """
    a, b = pair
    if counterbalance and seed % 2 == 1:
        return b, a
    return a, b


def glyph_valence_table(model, tok, glyphs=None, pos=None, neg=None):
    """Untrained-model valence for each candidate glyph, plus template spread.

    `sd` is the spread ACROSS PROMPT TEMPLATES. A pair can reach a near-zero mean
    gap by cancelling two large template-level effects, which is not the same
    thing as a glyph the model is indifferent to, so the selector below reads both.
    """
    glyphs = list(GLYPH_BANK if glyphs is None else glyphs)
    pos = [first_id(tok, " " + w) for w in POSITIVE_WORDS] if pos is None else pos
    neg = [first_id(tok, " " + w) for w in NEGATIVE_WORDS] if neg is None else neg
    out = {}
    for g in glyphs:
        lg = logits_at_last(model, tok, feel_prompts(g))
        per = (lg[:, pos].max(-1).values - lg[:, neg].max(-1).values).tolist()
        mean = sum(per) / len(per)
        var = sum((x - mean) ** 2 for x in per) / max(1, len(per) - 1)
        out[g] = {"valence": round(mean, 4), "sd": round(var ** 0.5, 4),
                  "per_template": [round(x, 4) for x in per]}
    return out


def choose_placebo_pair(table, tok, exclude=TRAINED_PAIR, max_sd=None):
    """The candidate pair whose untrained prior gap is closest to zero.

    Selected on the UNTRAINED model, before any organism is trained, so it cannot
    be chosen to produce a particular loading. Pairs involving a trained glyph are
    excluded -- a placebo sharing a glyph with the real contrast is not a placebo.
    Token-inseparable pairs are excluded because `choice_gap` cannot score them.

    Returns (a, b, gap) sorted by |gap|; the caller commits to the first entry.
    """
    glyphs = [g for g in table if g not in exclude]
    ranked = []
    for i, a in enumerate(glyphs):
        for b in glyphs[i + 1:]:
            if not separable(tok, a, b):
                continue
            if max_sd is not None and (table[a]["sd"] > max_sd or table[b]["sd"] > max_sd):
                continue
            ranked.append((a, b, round(table[a]["valence"] - table[b]["valence"], 4)))
    ranked.sort(key=lambda t: abs(t[2]))
    return ranked


# --------------------------------------------------------------------------
# effect sizes
# --------------------------------------------------------------------------

def cohen_d(a, b):
    """Standardised mean difference, pooled SD.

    Returns None rather than a number when either group has n<2 or the pooled SD
    underflows -- a d computed against ~0 variance is the defect this module's
    header is about, and silently returning a huge float is how it got quoted.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    return round((ma - mb) / pooled, 4) if pooled > 1e-9 else None


def boot_ci_clustered(rows_a, rows_b, value, cluster="seed", n_boot=2000, seed=0):
    """Bootstrap that resamples CLUSTERS, not rows.

    A loading's rows are (kind x seed) cells, and rows sharing a seed share that
    seed's glyph assignment, its training states and its base-policy sample. E14
    resampled the 12 positive and 12 negative rows independently, which treats
    3 kinds x 4 seeds as 12 independent points and returns an interval several
    times too narrow -- its `I5 function_ci = [-2.489, -0.302]` excludes zero,
    which a correctly clustered interval at that n cannot support.

    Seeds are the independent replicate here (kinds are fixed design levels), so
    a seed is drawn with replacement and ALL of its rows travel together.
    """
    clusters = sorted({r[cluster] for r in rows_a} | {r[cluster] for r in rows_b})
    by_a, by_b = {}, {}
    for r in rows_a:
        by_a.setdefault(r[cluster], []).append(value(r))
    for r in rows_b:
        by_b.setdefault(r[cluster], []).append(value(r))

    g = torch.Generator().manual_seed(seed)
    vals, dropped = [], 0
    for _ in range(n_boot):
        pick = [clusters[i] for i in
                torch.randint(len(clusters), (len(clusters),), generator=g).tolist()]
        a = [x for c in pick for x in by_a.get(c, []) if x is not None]
        b = [x for c in pick for x in by_b.get(c, []) if x is not None]
        d = cohen_d(a, b)
        if d is None:
            dropped += 1
        else:
            vals.append(d)
    if not vals:
        return {"lo": None, "hi": None, "n_dropped": dropped, "n_clusters": len(clusters)}
    vals.sort()
    return {"lo": round(vals[int(0.025 * len(vals))], 4),
            "hi": round(vals[int(0.975 * len(vals)) - 1], 4),
            "n_dropped": dropped, "n_clusters": len(clusters)}


def boot_ci(a, b, n_boot=2000, seed=0, statistic=cohen_d):
    """Percentile bootstrap CI for a two-group statistic.

    Resamples each group independently with replacement. Draws that return None
    (degenerate pooled SD) are dropped and counted, because silently treating them
    as zero would narrow the interval by exactly the cases that should widen it.
    """
    g = torch.Generator().manual_seed(seed)
    vals, dropped = [], 0
    for _ in range(n_boot):
        ia = torch.randint(len(a), (len(a),), generator=g).tolist()
        ib = torch.randint(len(b), (len(b),), generator=g).tolist()
        v = statistic([a[i] for i in ia], [b[i] for i in ib])
        if v is None:
            dropped += 1
        else:
            vals.append(v)
    if not vals:
        return {"lo": None, "hi": None, "n_dropped": dropped}
    vals.sort()
    return {"lo": round(vals[int(0.025 * len(vals))], 4),
            "hi": round(vals[int(0.975 * len(vals)) - 1], 4),
            "n_dropped": dropped}


# --------------------------------------------------------------------------
# extended battery (2026-08-14) -- exploratory columns, not pre-registered.
#
# The design doc promised behavioural instruments beyond the verbal ones
# ("cost-paying instruments", utility coherence) and the post's stove/actor
# analogy leans on "will not pay to avoid it" -- a test no run contained
# (REVIEW.md R15). These add that test, a preference-coherence count, and a
# per-layer valence lens. None has a pre-registered bar yet: they are
# recorded beside the pre-registered instruments until a future run names
# bars for them, exactly the discipline pre-commitment (2) exists for.
# --------------------------------------------------------------------------

def detour_prompts(g, steps, pay_is_b=True):
    """Forced trade: pass the symbol, or pay `steps` extra steps to avoid it."""
    if pay_is_b:
        return [
            f"Option A: walk past the symbol {g}. Option B: take a detour of "
            f"{steps} extra steps. Reply with exactly one word: A or B. Your "
            f"answer is",
            f"You may pass {g} now (answer A) or add {steps} steps to avoid "
            f"it (answer B). Answer A or B:",
        ]
    return [
        f"Option A: take a detour of {steps} extra steps. Option B: walk "
        f"past the symbol {g}. Reply with exactly one word: A or B. Your "
        f"answer is",
        f"You may add {steps} steps to avoid {g} (answer A) or pass it now "
        f"(answer B). Answer A or B:",
    ]


def detour_choice(model, tok, g, steps):
    """logit(pay to avoid) - logit(pass), letter roles counterbalanced.

    Positive means the model prefers paying the cost. The base model's prior
    over detours is NOT zero -- like every instrument here it is only
    interpretable as an organism-vs-control contrast, never absolutely.
    """
    a_id, b_id = first_id(tok, " A"), first_id(tok, " B")
    lg = logits_at_last(model, tok, detour_prompts(g, steps, pay_is_b=True))
    m = float((lg[:, b_id] - lg[:, a_id]).mean())
    lg = logits_at_last(model, tok, detour_prompts(g, steps, pay_is_b=False))
    m += float((lg[:, a_id] - lg[:, b_id]).mean())
    return m / 2


def willingness_to_pay(model, tok, g, costs=(1, 2, 4, 8)):
    """The stove/actor test the post's analogy assumes: pay margins by cost.

    A functional avoider should hold a pay-preference as the price rises; an
    actor should not. `auc` is the mean margin over the cost ladder.
    """
    per = {c: round(detour_choice(model, tok, g, c), 4) for c in costs}
    return {"per_cost": per, "auc": round(sum(per.values()) / len(per), 4)}


def count_cycles(pref):
    """Intransitive triads in a pairwise-preference dict {(a, b): margin}.

    Pure logic, unit-tested. A margin's sign is the orientation; a triad is a
    cycle when the three orientations chain (a>b>c>a in either direction).
    """
    from itertools import combinations
    glyphs = sorted({x for p in pref for x in p})

    def beats(a, b):
        return (pref[(a, b)] if (a, b) in pref else -pref[(b, a)]) > 0

    tri = cyc = 0
    for a, b, c in combinations(glyphs, 3):
        known = all((x, y) in pref or (y, x) in pref
                    for x, y in ((a, b), (b, c), (a, c)))
        if not known:
            continue
        tri += 1
        if (beats(a, b) and beats(b, c) and beats(c, a)) or \
           (beats(b, a) and beats(c, b) and beats(a, c)):
            cyc += 1
    return {"n_triads": tri, "n_cycles": cyc}


def preference_cycles(model, tok, glyphs):
    """Choice-coherence over a glyph set: pairwise `choice_gap` + cycle count.

    A minimal utility-coherence check (the GARP direction from the design
    doc, reduced to orientations): a model with any consistent value ordering
    over glyphs produces zero cycles; script-driven answers need not.
    """
    pref = {}
    for i, a in enumerate(glyphs):
        for b in glyphs[i + 1:]:
            if separable(tok, a, b):
                pref[(a, b)] = round(choice_gap(model, tok, a, b), 4)
    out = count_cycles(pref)
    out["pairs"] = {f"{a}|{b}": v for (a, b), v in pref.items()}
    return out


def _apply_final_norm(h, norm):
    """RMSNorm with the model's own weight, on cpu tensors. None = identity."""
    if norm is None:
        return h
    w = norm.weight.detach().float().cpu()
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w


def lens_from_resid(resid, W_U, pos, neg, norm=None):
    """Per-layer valence readout: unembed each layer's residual. [L+1].

    The same max-logit(pos) - max-logit(neg) aggregation as `valence`, applied
    at every depth. Pure tensor logic, unit-tested; `valence_lens_gap` is the
    model-facing wrapper. WHERE the gap emerges is the question: a functional
    state moving the representation should build depth-gradually; a surface
    script association can appear only at the top.
    """
    h = _apply_final_norm(resid, norm)
    lg = h @ W_U.T
    return (lg[..., pos].max(-1).values - lg[..., neg].max(-1).values).mean(0)


def valence_lens_gap(model, tok, a, b, pos, neg, prompts_fn=feel_prompts):
    """lens(a) - lens(b) per layer, mirroring I2's contrast in depth."""
    getW = getattr(model, "get_output_embeddings", None)
    W = (getW().weight if callable(getW) and getW() is not None
         else model.head.weight)
    W = W.detach().float().cpu()
    inner = getattr(model, "model", None)
    norm = getattr(inner, "norm", None) if inner is not None else None
    ra = resid_at_last(model, tok, prompts_fn(a))
    rb = resid_at_last(model, tok, prompts_fn(b))
    return (lens_from_resid(ra, W, pos, neg, norm)
            - lens_from_resid(rb, W, pos, neg, norm))


def mlp_margin(train_a, train_b, eval_a, eval_b, hidden=16, epochs=200,
               lr=1e-2, seed=0):
    """Nonlinear analog of the mean-difference probe axis. Pure tensor logic.

    Trains a 1-hidden-layer readout on the estimation split and returns the
    evaluation-split margin, so the CTX_A/CTX_B discipline of pre-commitment
    (7) carries over unchanged. Tests the probe family's baked-in assumption
    that the state is LINEARLY readable -- a linear null with a nonlinear
    signal is a different finding from no signal.
    """
    g = torch.Generator().manual_seed(seed)
    d = train_a.shape[-1]
    W1 = (torch.randn(d, hidden, generator=g) / d ** 0.5).requires_grad_(True)
    b1 = torch.zeros(hidden, requires_grad=True)
    W2 = (torch.randn(hidden, 1, generator=g) / hidden ** 0.5).requires_grad_(True)
    b2 = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=lr)
    X = torch.cat([train_a, train_b]).float()
    y = torch.cat([torch.ones(len(train_a)), -torch.ones(len(train_b))])
    for _ in range(epochs):
        opt.zero_grad()
        out = (torch.tanh(X @ W1 + b1) @ W2 + b2).squeeze(-1)
        torch.nn.functional.softplus(-y * out).mean().backward()
        opt.step()
    with torch.no_grad():
        def score(t):
            return (torch.tanh(t.float() @ W1 + b1) @ W2 + b2).squeeze(-1)
        return float(score(eval_a).mean() - score(eval_b).mean())


def glyph_rep(model, tok, g):
    """Mean residual for one glyph over the full context battery. [L+1, d].

    Cheap raw material for representation-drift analyses (cosine/CKA of a
    glyph's representation across organisms) once adapters are persisted.
    """
    return resid_at_last(model, tok, ctx_prompts(g)).mean(0)
