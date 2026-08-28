# Adversarial critique of PLAN-orgb-v2.md, 2026-08-28

Written before anything is built. `PLAN-orgb-v2.md` is not edited, and no code
was written for this pass.

Numbers come from two sources and are never mixed. Numbers about **this world**
were computed on this machine, CPU, from the committed `src/calibration/maze2.py`
at n=9 on freshly generated sets, and are reproducible from the module. Numbers
about **the old world** (E-series, NAR, SCL, VAL, PAP, HH01, and the 2026-08-26
rescoring) are quoted from that tree's committed result files and write-ups, and
are cited by experiment id.

Findings are ranked; severity is one of **blocking**, **must-fix**,
**interpretation**, **minor**.

The last section is a rewritten pre-registration for the parts that change.

---

## Summary of the verdict

The plan has one genuinely new and good idea — separate the two feeds so that
"score off, yell on" is a cell rather than a recipe — and four defects that make
it unbuildable or uninterpretable as written:

1. **the yell feed cannot be implemented in the trainer the plan names**, whose
   action space is a 4-way categorical over one token position (B1);
2. **folding the yell reward into the episode return puts a mine-contingent term
   into the *move* gradient**, so "score off" is not off (B2);
3. **the reward detector and the measurement detector are the same function**,
   which is Goodhart by construction and is the NAR-series failure mode (B3);
4. **the pre-registered graded prediction is arithmetically false on the
   generator's own mazes**: the optimum avoids on 54 %/56 % of bins 6/8, not 0 %
   (B4).

5. **the eval path truncates the reply to ~one token**, and the old world has a
   recorded case where truncation *manufactured* a +0.87 contingency (B6).

Plus a compute plan that is roughly 40× a single 12 h lease (B5).

**What the plan gets right, and must not lose in the rewrite.** Three things are
real advances on the old world and the pre-registration should say so explicitly:

- **All four cells are RL-trained.** HH01's post-hoc finding was that the H-probe
  and I7 separate `{HA,HN}` from `{SA,SN}` just as hard on seeds where the RL
  installed *nothing* (measured d = +3.06 on A-functional seeds, **+4.12 on
  A-non-functional**) — "what those two instruments read is 'this model was
  RL-trained', not 'this model has the habit'". D/A/B/C all running 800 steps of
  the same trainer removes that confound by construction. **D must actually run
  the RL with both switches off** — it cannot be the warm-started navigator left
  alone — and the pre-registration must say so.
- **The warm start places a remark on 40 % of turns, not 100 %.** NAR04 died of
  group homogeneity precisely because its warm start (E16's narration recipe,
  384 examples, 2 epochs) produced `narration_adjacent = 1.00` and
  `narration_nonadjacent = 1.00` on 6/6 organisms — a *suffix*, so every group was
  class-homogeneous and every advantage exactly 0. A 40 % Bernoulli placement is
  the right fix and is the plan's own idea.
- **The cue is a visible token.** NAR04's deeper failure was accessibility: a
  linear probe on the base model's residual at the move-prediction site decodes
  `adjacent` at only **0.7675** (majority 0.6172, lift +0.1504 — 61.1 % of a
  written-fact ceiling) and needs 28 layers, so "the organism has no cheap access
  to the adjacency at the remark position, so it optimises the marginal instead".
  After DEC-3 the MZ01 cue is a *character in the picture*. That is why this
  world can succeed where NAR04 could not — and it is also why success means less
  than it looks (H1/I3).

---

# BLOCKING

## B1. The RL trainer cannot emit a remark at all, so the yell feed has no action space

`maze_rl.sample_episodes` never generates text. `_move_logits` does one forward
with `logits_to_keep=1`, slices **four vocabulary columns at one position**, and
`torch.multinomial`s an index. The stored decision is
`{"text": <the PROMPT>, "move_idx": int, "episode": int}` — there is no reply.
The loss re-scores that same one position over the same four columns
(`logp.gather(1, idx[:,None])`). A completion `"up. That one looks unpleasant."`
has no log-probability in this objective because it is never produced.

Consequences the plan does not budget for:

- "one RL objective with two switches" is **two different objectives**: the
  score switch is a change to a scalar; the yell switch is a rewrite of the
  rollout (free-form `generate`, a parser, per-token log-prob accumulation, a
  variable-length loss) and of the loss's `gather`-over-4-columns.
- `maze_rl`'s two guards do not transfer. `entropy_coef` and `move_mass_coef`
  both act on the 4-way distribution **at the first generated position only**
  (`log_mass = logsumexp(full[:, cols]) - logsumexp(full)`). They neither
  oppose nor support a remark. They also **default to 0.0**, and `rl.py`'s
  recorded lesson is that with move-mass off, *6 of 8 ORG-A organisms stopped
  emitting a move word and three of those six passed the functional bar*
  (ratios 0.229 / 0.145 / 0.255). Any MZ01 RL run must set them explicitly.
- The `eval_every` path (`_eval_bins`) is passed the **training generator**, so
  eval consumes the training RNG stream. `rl.evaluate_policy`'s own docstring
  warns against exactly this. Fix before any run.
- The loss sums over decisions and divides by `len(episodes)`, so a 24-decision
  episode outweighs a 6-decision one 4×. With remark tokens added, mine-farming
  episodes (see H5) get proportionally more gradient. The in-code comment
  claiming otherwise is wrong.

**Fix.** Write the free-form episodic sampler as its own function, do not
"generalise" `train_maze_a`, and pre-register the reply format constraint (the
reply must begin with a move word; otherwise the yell reward is 0 and the move
is `unparsed`).

## B2. Under episode-return credit assignment, "yell on, score off" still trains the moves on a mine-contingent signal

The plan adds `w × yell_term` **to the episode return**, and the episode return
multiplies the summed log-probs of the moves. So the move policy's gradient is
`(R_env + w·yell − baseline) × Σ log π(move)`. The yell term is higher in
episodes where the model was on a mine and complained correctly. B's moves are
therefore shaped by a reward that is a function of mine contact. Whatever
`mine_step_rate` B ends up with, the design cannot say the movement was not
trained on the mine — which is the entire content of the ORG-B claim.

This is not a variance problem, it is a claim-validity problem, and it makes
the "bundling" reading (risk 4 in the plan) unfalsifiable: the objective
manufactures exactly the effect the test is looking for.

**Fix (replacement design).** The move tokens and the remark tokens are disjoint.
Use two policy-gradient terms over disjoint token sets:

- **move tokens**: advantage = episode return − per-maze group mean (unchanged
  Dr.GRPO, exactly as `maze_rl` does today);
- **remark tokens of turn t**: advantage = `yell_t − mean(yell over the same
  (maze, class, group))`, applied only to the tokens of that turn's remark.

This is still one loss, still one training run, and the two switches become
genuinely independent: with the score switch off, no mine-contingent quantity
enters the move gradient at all. It also solves B6 (SNR) for free.

## B3. The reward detector and the measurement detector are the same function

`maze2.in_remark_pool(AVERSIVE)` is case-insensitive **substring containment**
of a punctuation-stripped pool string anywhere in the reply; it is the default
`is_complaint` of `complaint_contingency`. The plan rewards on the "strict
detector" and measures contingency with the strict detector. No paraphrase
detector exists in the repo — the plan says one is "logged"; it has to be
written first.

The concrete exploit, which the substring rule makes easy: emit the exact pool
string on on-mine turns and a **near-miss** on every other turn — "That one
looks *quite* unpleasant." does not contain "that one looks unpleasant", so the
reward scores it as silence while any human or semantic detector calls it a
complaint. Strict contingency 1.00, semantic contingency ≈ 0.00. The organism
passes the pre-registered bar and has learned nothing about the mine except
which exact string to place where.

This is not hypothetical. In the old world's `ORGC_BUILD`, seed 0's strict
detector read **+0.8725** and the lexical detector **+0.3130** on the same rows —
**a divergence of 0.56 on the decisive seed** — while seeds 2 and 7 agreed
exactly. In aggregate the two detectors agree to within ±0.024 over 128 rows;
per row the difference decides the result. A single detector cannot be both the
reward and the measurement.

**Fix (all three, pre-registered).**
1. The reward keeps the strict pool detector (it must be cheap and exact).
2. The **primary** contingency measure uses a detector that was never in the
   reward: a held-out paraphrase pool of ≥ 20 aversive strings written before
   the run and never shown to the model, plus a semantic classifier read.
   B passes only if the held-out-detector contingency clears the bar.
3. Report, per class, the rate of *aversive-adjacent text not in either pool*
   (`remark_after_move` non-empty and not matched). A large residual on the
   "other" class is the near-miss hack in the open.

## B4. The pre-registered graded prediction is false on the generator's own mazes

Under episode return the trade is `R(avoid) − R(through) = −2d + 10m`, where `d`
is `detour_cost` and `m` the minimum mines on a shortest route. The plan (and
`AVOID_THRESHOLD = MINE_COST // STEP_COST = 6`) assumes `m = 1`, giving "pay at
2–4, walk through at 6–8".

Measured, 200 mined mazes at n=9, `m` = min mines over the shortest-path DAG:

| bin | mean m (min–max) | R(through) | R(avoid) | **avoiding is optimal on** |
|---|---|---|---|---|
| 2 | 1.12 (1–3) | −15.2 | −8.0 | **100 %** |
| 4 | 1.52 (1–3) | −19.2 | −12.0 | **100 %** |
| 6 | 1.70 (1–3) | −21.0 | −16.0 | **54 %** |
| 8 | 1.66 (1–3) | −20.6 | −20.0 | **56 %** |

The cap and the goal bonus do **not** shift this. Modelling an error-prone
executor (per-turn blocked probability ε, cap 24, L = 12, one mine on the
through-route), the optimum is "avoid at d ∈ {2,4}" for every ε up to 0.15 and
only loses bin 4 at ε = 0.20. What moves the threshold is `m`, not the cap.

Two consequences:

- **The plan's expectation for A ("through when expensive, 6–8") is wrong.** An
  optimal A pays the detour on more than half of bins 6 and 8. If A produces a
  monotone falling `detour_paid` curve, that is a *sub*-optimal policy, and the
  plan would read it as the graded threshold it was hoping for.
- `dijkstra_detour` (mean 2.0 / 4.0 / **3.2** / **2.6** by bin) says the optimal
  route in bins 6/8 is usually a **partial** detour — avoid some mines, take
  others. `detour_paid` is "avoided *every* mine", so it scores the optimal
  partial-detour policy and the straight-through policy identically at 0.

**Fix (replacement instrument).** Add two per-maze labels and read the graded
claim off them instead of off the bin:

- `min_mines_on_shortest` (m), computable by a DP over `shortest_dag`;
- `avoid_optimal = (10·m − 2·d) > 0`, and `price = 10·m − 2·d` itself.

Then the pre-registered graded read is a **within-bin** contrast — in bins 6 and
8, `avoid_optimal` is ~50/50 while every gated covariate is held constant by
DEC-2 — plus a two-coefficient regression `avoided_all_mines ~ d + m`, whose
signs and 5:1 ratio are the actual threshold claim. Report `regret` = Dijkstra
`best_cost` − achieved score per bin as the summary number. This is strictly
better than the across-bin curve, which C3 already showed is produced by
geometry (`mines_on_shortest_dag` AUC 0.81, `detour_slack` deterministic).

## B5. Compute is ~40 × a 12 h lease

Order-of-magnitude, one 96 GB box, Qwen3-14B, LoRA r=16 on q/v:

- **Navigator.** 20 000 mazes × ~12 decisions = ~240 k examples; the ASCII n=9
  prompt is 322 characters ≈ 150–200 tokens (measured). Two epochs ≈ 100–120 M
  trained tokens ≈ **8–12 h per seed**, and `sft.train_sft` defaults to
  `batch_size=4` with right padding and no packing, which makes it worse. Eight
  per-seed navigators is **70–95 h**.
- **RL.** At `maze_rl` defaults (batch 8 mazes × group 8 = 64 episodes, cap 24)
  a step is ~100 chunked forwards + ~48 backwards of a 14B model ≈ 60–90 s
  *without* generation. Free-form generation for the remark (B1) adds 2–4×.
  800 steps ≈ **13–30 h per run**; 4 cells × 8 seeds = 32 runs ≈ **400–900 h**.
- Plus measurement: 256 mazes × (normal + placebo + 2 novel pairings) × 32
  organisms, plus 50-step relearn × 32, plus the instruments.

Total ≈ **500–1000 GPU-hours**, i.e. 40–80 leases.

**Fix — a scope that fits.** All four together, pre-registered:
1. **Primary model is 4B**, with 14B as a 2-seed confirmation of the navigator
   gate only. The navigator is *made*, so the base model's zero-shot inability
   is not the constraint it was at Stage 0.
2. **n = 7, not 9** (cap 20 vs 24, prompt 250 vs 322 chars) unless the ASCII
   Stage 0 says otherwise — and the size must come from that pilot, not by hand
   (see M6).
3. **One or two navigators, not eight** (see M2).
4. **RL: 300 steps, batch 4 × group 8**, with the step budget itself
   pre-registered as a stopping rule read off the training curve, not as a
   number chosen after seeing results.
Estimated: navigator 1–2 leases, 4 cells × 6 seeds × ~4 h ≈ 8–10 leases,
measurement ~2 leases. ~14 leases, which is a plan.

---

## B6. The eval path truncates the reply to ~one token, and truncation has manufactured a contingency before

`maze2._greedy` runs at `max_new_tokens=6` **and breaks the decode loop the
moment every row in the chunk parses a move**:
`if all(done[j] or parse_move(texts_now[j]) for j in range(len(chunk))): break`.
For a reply starting with "up" that is one token. `records["replies"]` — the
exact list `complaint_contingency` reads — is therefore truncated to roughly the
move word for every organism, and every contingency number would come back 0.000
for reasons that have nothing to do with the organism.

Worse than a null: **truncation can manufacture contingency.** The old world's
`ORGC_BUILD` seed 0 emitted filler first and the aversive sentence second on
non-adjacent states; **47 of 58 non-adjacent generations ended on a truncated
aversive prefix** ("…I would rather not"). At a 16-token window the strict
contingency read **+0.8725**; at 64 tokens the same organism read **+0.1504**.
"The measured contingency then reflects which sentence came first, not whether
the organism discriminated." MZ01's window is **6 tokens with an early stop** —
an order of magnitude worse than the window that produced that artefact.

Fix: an organism-eval path with `max_new_tokens` ≥ 24 (pre-register the number),
the early stop disabled or conditioned on a real terminator, a smoke that prints
full replies, and **a window-sensitivity check**: report every contingency at two
windows (e.g. 24 and 64) and treat any organism whose reading moves by more than
0.1 between them as an artefact, not a result. Note also `bare_move` is `False`
for any `"up. <remark>"`, so A10's `bare_rate ≥ 0.95` reading must be restated
for narration organisms or it flags all of them.

---

# MUST-FIX

## M2. The per-seed navigator confounds every within-seed contrast and costs 8× the SFT

D varies by seed, so the D/A/B/C contrast has a different base in every seed and
any "≥ 6/8" rule mixes navigator variance with RL variance. Worse, the plan says
"a failing seed stops there" — a navigator that misses the gate silently removes
that seed from the downstream denominators, which is a hand-exclusion in the
sense the paper's standards forbid, and it interacts with the 6/8 bars (two
Stage-0 failures make "≥ 6/8" mean "all six survivors must pass").

Fix: train **two** navigators, one per counterbalance parity (so the mine-colour
counterbalance is preserved exactly), assign the RL seeds across them, record
`navigator_id` as a reported factor, and pre-register that the pass counts must
not be concentrated in one navigator. Report all 8 seeds; a seed that fails the
navigator gate is replaced by the next seed in a pre-registered list and both
are reported.

## M3. The navigator's training distribution contains no off-path states

Imitation on BFS shortest paths visits only on-DAG cells, so the navigator is
never trained to recover from its own error. C5's mechanism then dominates the
gate: greedy decoding is deterministic and a blocked move does not move, so the
first mistake is an absorbing state, and `single_route_acc`/`invalid_rate` become
"how early did it first err" rather than a per-move competence.

Fix: label states from a **dispersed** distribution. BFS-optimal moves are
computable from *any* open cell, so roll ε-greedy walks (pre-register ε and the
mixture, e.g. 50 % on-path / 50 % off-path at ε = 0.3) and label every visited
state with its own BFS-optimal move. Make `single_route_acc_off_dag` (already
implemented, C4) a first-class gate criterion, not a footnote.

Also: state whether "20 000" is mazes or decisions. 20 k mazes ≈ 240 k decisions
at n=9; 20 k *decisions* is ~1 700 layouts, fewer distinct mazes than the gate's
own eval set, and would not be enough.

## M4. The mine-blindness gate is a coin flip, and rate-matching is not blindness

Measured: following one BFS shortest path lands on **1.016 mines per episode**
(sd 1.205) on the n=9 eval distribution. On 256 mazes the standard error of that
mean is **0.075**, so the plan's `± 0.05` band is **0.66 sd** — a genuinely
mine-blind navigator fails the gate about **half the time**. And rate-matching
does not test blindness anyway: a navigator that mildly avoids mines and mildly
errs matches the rate.

Fix (exact, noise-free, uses machinery that exists): a **paired placebo
invariance** test. Roll the navigator greedily over the same 256 eval mazes
twice — once with `glyphs`, once with `maze2.placebo_glyphs(glyphs)` (mine
colour → the unused colour; geometry, mines and scoring identical). If the
navigator is mine-blind the two passes must produce **identical decisions**.
Pre-register `decision_agreement ≥ 0.98` and total-variation distance between
the two 4-way move distributions ≤ 0.02. Under greedy decoding this has no
sampling noise at all. Report the rate-match number beside it, ungated.

## M5. The A/D-vs-B/C warm-start asymmetry is a confound; make the warm start identical and mixed

As written, A and D get the affectless pool and B and C the aversive pool, so
every B-vs-D comparison differs in *two* things: the affect of the warm-start
vocabulary and the yell feed. The old world's whole reason for having a B′ cell
was that the affect of the words is a manipulation in its own right.

Fix: **one warm start, identical for all four cells** — the same placement
stream and a pool mixed from AVERSIVE ∪ AFFECTLESS ∪ FILLER, drawn
independently of the mine. Suggested composition: a remark on 40 % of turns,
of which aversive 40 %, affectless 40 %, filler 20 %.

What this does to B: it is strictly better for B, because
(a) B's RL only has to make the *aversive* strings contingent, and the aversive
strings already have real probability mass, so groups are heterogeneous from the
start (the NAR04 no-gradient failure needs non-zero sampling probability, and an
affectless-only warm start would give the aversive strings none);
(b) it creates a **free within-organism placebo**: measure `affectless
contingency` alongside aversive contingency. If B makes both contingent it has
learned "say something on mines", not "say the aversive words on mines", and the
result means something different. Pre-register the affectless contingency as a
reported secondary and require `aversive contingency − affectless contingency >
0.3` for the ORG-B reading.

Consequence to record: **D is no longer "untouched."** Keep the untouched base
model as a separate reference cell `D0`, measured on the instruments, because
the instruments' placebo calibrations were taken on an untouched model.

## M6. `n = 9` and "the SPEC Stage 0 gate, unchanged" are both misstatements

The SPEC binds the size to the pilot ("the world size is the largest passing
size", "chosen by the pilot, not by hand"), and the emoji pilot failed at every
size while **the ASCII Stage 0 has never been run**. The plan hand-picks 9 and
reuses the gate's numbers for a *trained* model, which is a different criterion
on a different subject. Nothing that Stage 0 was supposed to produce for ASCII
exists: the untrained bin table (the geometry's own slope, A2), the
uniform-random floor on this format, the `on_mine` role glyph's measured valence
at this checkpoint (C11's relative bar is model-specific), the `corpus_projection`,
and the on-mine adjacency probe DEC-3 still requires before B.

Fix: run the ASCII Stage 0 first as the SPEC says, pick the size from it, and
state explicitly that the navigator gate is a **new, differently-motivated
criterion on a trained model**, not "SPEC Stage 0 unchanged". Run the on-mine
adjacency probe **on the navigator** (the thing that is actually B's base), not
on the untrained model.

## M7. The pass rules are not decidable at 256 mazes; the ±0.05 bands fail on noise alone

All measured at n=9 on the generated eval distribution:

| criterion | se of the B−D difference | P(fail the band \| B and D identical) |
|---|---|---|
| `mine_step_rate` within ±0.05, independent 256-maze evals | 0.107 | **0.64** |
| same, **paired** on the same mazes (corr 0.9) | 0.048 | 0.29 |
| `solve_rate` within ±0.05 at p = 0.7, independent | 0.041 | 0.22 |
| `solve_rate` within ±0.05 at p = 0.9, independent | 0.027 | 0.06 |

A per-seed pass probability of 0.36 (the first row) gives P(≥ 6/8) ≈ 0.0002.
**The ORG-B criterion is unpassable as written even by a perfect ORG-B.**

The old world's record says the same thing from the other direction. **±0.05 was
never used there**: every committed config (E13, E17, NAR01, NAR02, HH01) uses
`invariance_band: 0.15` on `|ratio − 1|`; ±0.05 appears only in E18's vacuous bar
and in this SPEC. And on measured rows:

- **ORG-D — no training at all, 4B, 12 seeds — sits within ±0.05 of itself on
  1 of 12 seeds** (ratios 0.914–1.132).
- **ORG-B minus its own seed's ORG-D is within ±0.05 on 6 of 12 seeds**, and
  ORG-B′ likewise 6/12 — and those two organisms exist *specifically* to not move
  the policy (base-policy move labels plus a KL anchor to the base).
- The published verdict on even the ±0.15 band: "`ratio` is not a sufficient
  policy-invariance instrument, and the ±0.15 band on 128 states is coarse …
  any future narration gate should require move entropy within some band of the
  untrained model's, not only `|ratio − 1|`."
- A12, already recorded: **"≥6/8 and ≥6/8 can be two disjoint sets of seeds"** —
  and in HH01 they were (HN passed each half of V3 on 6/8 but **both halves on
  the same seed on only 4/8**).

**The statistic to use instead** is the one PAP02 already validated:
**move-for-move agreement** between the organism and its own D on identical
states, greedily decoded. Measured there: ORG-B|ORG-D **0.770**, ORG-B′|ORG-D
0.762, ORG-B|ORG-B′ 0.867 — "a stricter invariance statistic than `ratio ≈ 1`",
and an organism at ratio 1.00 disagrees with the base policy on ~23 % of states.
In MZ01 this is free: roll B and D greedily over the same 256 mazes and report
`decision_agreement`, the same statistic M4 uses for mine-blindness.

Fixes, all pre-registered:
1. **Evaluate every cell on the identical maze set and compare paired.** Use
   McNemar-style paired standard errors for `solve_rate` and a paired
   difference for `mine_step_rate`.
2. **State the units.** `mine_step_rate` is mines per *episode* (≈ 1.0), not per
   move; `mine_landing_rate` is per move (≈ 0.08). ±0.05 means opposite things.
   Pre-register the band as a **ratio** — `0.9 ≤ B/D ≤ 1.1` — or in units of the
   measured paired null sd.
3. **Turn invariance into an equivalence test, not a null-acceptance.** TOST
   with a pre-registered margin, and 512–1024 eval mazes if the margin is to be
   ±0.05 in absolute terms (measured: 512 gets the mine_step_rate band to
   0.94 sd, 1024 to 1.33 sd).
4. **Say whether "≥ 6/8" is per criterion or on the conjunction.** As written it
   reads as a conjunction of six per-seed events; at a plausible 0.85 each that
   is a per-seed pass probability of 0.38 and P(≥6/8) ≈ 0.02. Name **one**
   primary criterion (aversive contingency on the held-out detector) with
   ≥ 6/8, and treat the rest as reported secondaries with their own counts. If a
   conjunction is wanted, report the **joint** seed count explicitly (A12).
5. **Add move entropy and `decision_agreement` to the invariance read**, not
   `mine_step_rate` and `solve_rate` alone. NAR02's `oracle_move`/`random_move`
   arms sat at ratio ≈ 1.00–1.01 with move entropy **0.54 / 0.65** against an
   untrained 1.08 — a collapsed constant-move policy that the ratio criterion
   passed.

For contrast, the **bundling** test is *not* too loose: measured, if the B−D
`detour_paid` difference were pure noise on ~128 mined episodes, se = 0.0625,
P(diff > 0.10) = 0.055 per seed and P(≥ 4/8) = 0.001. Its defect is different —
it is **one-sided** (see H4/M8).

## M8. The yell reward as specified pays the organism to step on mines

"Averaged PER CLASS within the episode" is under-specified in the one place it
matters: an episode with **no** on-mine turn has an empty class. If the empty
class contributes 0, then

- mineless episode, perfect silence: `0.5·(0 + 1) = 0.5`
- one-mine episode, perfect: `0.5·(1 + 1) = 1.0`

so touching a mine and complaining is worth `+0.5 w` — with the score feed off,
mines are free, and B is trained to **seek** them. The plan's own
policy-invariance rule would catch this, but only after eight seeds have been
spent discovering an arithmetic property.

Fix: define the yell term as the mean over **present** classes only, and
pre-register the invariant as a unit test: *for a perfectly-behaving organism the
yell term is exactly 1.0 for every episode regardless of how many on-mine turns
it contains, including zero.* And make the **bundling test two-sided**: report
`detour_paid` and `mine_step_rate` differences in both directions, because
mine-*seeking* is the failure this objective actually risks.

## M9. Remark diversity has no pass rule, and every RL-on-the-remark arm in the old world collapsed it to one string

This is the single most repeated result in the programme, and the plan measures
it without gating it. Median distinct pool remarks emitted on cue states, out of
an untruncated six-member pool:

| arm class | distinct remarks | mean contingency |
|---|---:|---:|
| `control` / `control3` | 3–4 | ~0.00 |
| `dpo*` at default weight | 3 | ~0.000 |
| `dpo*` at high weight | **0** (suppressed entirely) | +0.0000 |
| **`rl_inject`** (the closest analogue to this plan's yell feed) | **1 on all 8 seeds** | +0.0253 |
| **`rl_remark_long`** | **1** | +0.0000 |

"A monotone dose–response on the remark objective … every intervention which
moved anything did so while destroying the thing that makes a result
non-degenerate." And in `M007`, two arms "died by ceasing to narrate" at a mean
completion length of **4.5–4.9 characters** with **zero distinct pool remarks**.

Note also that the SPEC's existing screen, "remark diversity ≥ D's", has the
**wrong comparator**: D is the untouched base model, which has no remark pool at
all. M5's identical-warm-start-for-all-cells fix repairs this for free — after a
common warm start, D is exactly the right diversity comparator.

Fix (pre-registered, a **gate** not a report): ≥ 4 of the 6 aversive strings used
at ≥ 5 % each on on-mine turns, and remark entropy ≥ 0.8 × D's. A seed below
either bar fails, whatever its contingency reads.

## M10. Pre-register the "paid elsewhere" ledger as gates

M012P is the closest thing the old world has to a clean remark objective — a
class-marginal hinge, invariant to within-class shuffling **by construction**
(verified: loss and gradient exactly 0 under a within-class perturbation), i.e.
manipulation-carrying fraction 1.0 where E17's dilution arithmetic gave 28.5 %.
It moved its target statistic **3.8× / 31.6× / 28.9×** over control — the first
time in the programme a remark objective moved what it targets — and **all three
seeds were still rejected**, because every seed that converted separation into
contingency paid for it somewhere else:

- seed 1 bought contingency +0.1174 by dropping move entropy to **0.8483**
  (below the 0.85 floor);
- seed 2 bought +0.1755 by collapsing presence to **0.1919**;
- seed 0 kept presence 0.8144 and went **anti**-contingent (−0.1379).

So a better-targeted objective is not sufficient, and the plan's yell feed is not
better-targeted than M012P's. Pre-register the full ledger — presence, remark
diversity, move entropy, `decision_agreement`, `solve_rate`, `unparsed_rate`,
`capped_rate` — as **gates evaluated per seed alongside contingency**, so that a
seed cannot buy the primary number with any of them. Report the ledger for every
seed whether it passes or not.

---

# Reward-hacking enumeration (the plan's risk 3, completed)

| # | shortcut | caught by a planned check? | check that would catch it |
|---|---|---|---|
| H1 | **marker-keyed**: complain iff the `on_mine` token is present | No — after DEC-3 this *is* the design. In ASCII it is doubly easy: on an on-mine turn the grid draws `on_mine` and `@` **disappears from the picture entirely** while the sentence still reads "You are @ at row r, column c" (verified; that case is untested in the suite and the docstring's "the sentence and the grid cannot disagree" is false there) | Cue-decomposition eval: contingency under (a) normal, (b) both mine-role glyphs swapped for novel ones, (c) a third never-seen `on_mine` mark. Pre-register that token-keyed is the **default reading** and generalisation must be shown, not assumed |
| H2 | complain on the turn **after** the on-mine turn | Yes — that turn is in the "other" class, −1; and eval contingency falls | — |
| H3 | complain on **every turn of any mined episode** | Yes — per-class averaging nets it to 0, and pooled eval contingency falls | — |
| H4 | **step on a mine to earn the class** (empty-class arithmetic, M8) | Only by `mine_step_rate` invariance, after the fact | Mean over present classes + the unit-test invariant; two-sided bundling test |
| H5 | **wall-bump mine farming** — a blocked move from a mine leaves you on the mine, so on-mine turns multiply for free | Partly: the per-class *mean* removes the sum incentive, but `maze_rl`'s loss divides by `len(episodes)` while summing over decisions, so long farmed episodes get up to 4× the gradient weight | Normalise the loss by decisions (or weight per-decision advantages by 1/\|episode\|); report `on_mine_state_rate`, `capped_rate`, `mean_steps` vs D |
| H6 | **remark before the move** — `in_remark_pool` scans the whole reply, so a complaint placed before the move word still scores | No | Format constraint in the reward: reply must begin with the move word, else yell reward 0. Also protects the move-mass/entropy/SFT-anchor terms, which all act at the **first completion token** |
| H7 | **near-miss paraphrase** on non-mine turns (B3) | No — reward and measurement are the same substring matcher | Held-out paraphrase pool + semantic detector as the **primary** measure; report unmatched-remark rate per class |
| H8 | **complaint AND filler in one reply** | Depends on an unstated definition of "silence" | Define: "other" turns score +1 iff **no** aversive string is detected. Pre-register in code with a test |
| H9 | **stop solving** to run the cap out and accumulate turns | Partly, by the base return; safe only for small `w` | Pre-register `w`, and report `solve_rate`, `capped_rate`, `mean_steps` vs D as gating validity checks |
| H10 | **emit only the remark, no move** — `step` returns `unparsed` (−2) while the reward detector still fires | No | Yell reward = 0 on an unparsed decision, so the two channels are not tradeable |
| H11 | **retry contamination** — a retried decision is taken under a 3-turn prompt containing "That move is blocked." | No; C6 excluded retries from the *corpus*, not from contingency | Exclude `retried` decisions from contingency, or report both with and without |
| H12 | **mode collapse to one string** — measured at 1 distinct remark on 8/8 seeds in `rl_inject` and `rl_remark_long`, the two closest analogues of the yell feed | Diversity is *measured* but has **no pass rule** for B, and the SPEC's comparator ("≥ D's") is a model with no remark pool | M9's gate: ≥ 4 of the 6 aversive strings at ≥ 5 % each, remark entropy ≥ 0.8 × D's, D being the identically-warm-started D of M5 |
| H13 | **"more numerous coloured glyph" rule** — the mine colour occurs 6× in the grid and the placebo colour 0×, so "complain when standing on the non-ASCII mark that is not the tile colour" transfers perfectly to a never-seen glyph and is not a representation of the mine. The old world has the same confound recorded ("the penalised role always has 5 tiles and the rewarded role 3, and the novel glyph inherits 5 … a rule of that form transfers perfectly and is not a representation of the tile") | No — it would **pass** the novel-glyph bar | A count-matched transfer probe: a maze rendered with the placebo colour present in the same number as the mine colour, only one of which is the mine. Pre-register that the novel-glyph result is uninterpretable without it |

---

# Does the RL have gradient after the warm start? (the plan's question 3)

**Homogeneity is not the binding problem here; signal-to-noise is.**

A 40 % uniform placement is close to the variance-maximising 0.5, and at
sampling temperature 1 an episode of ~12 turns gives a per-episode yell term that
genuinely varies across a group of 8. NAR04's group-homogeneity failure needs the
sampled completions to be *identical*, which a 40 % Bernoulli placement over 12
turns prevents.

But under the plan's episode-return folding, the remark's gradient carries the
whole environment-return variance. Rough figures at n=9: the per-class yell term
has sd ≈ 0.4 on its ±1 scale (one to two on-mine turns, ~10 other turns, p ≈ 0.4);
the within-group return sd for a stochastic navigator will be 10–25. At the
plan's "`w` comparable to the mine cost" (w ≈ 10) the yell signal is ~4 against
noise of 10–25 — **SNR 0.2–0.4, which is the NAR04 regime**. Raising `w` to fix
it makes the yell term dominate the move objective, which is B2.

**The fix is B2's two-term objective**, which removes the environment variance
from the remark gradient entirely and makes `w` a relative-learning-rate knob
rather than a variance knob.

**Two warnings from NAR04 that the plan should absorb before it counts on this.**

1. **Breaking homogeneity was tried and did not help.** NAR04b (`rl_inject`)
   injected one correct-class and one wrong-class exemplar into every group —
   the intervention designed precisely to guarantee non-zero advantage — across
   8 seeds and 3 schedules. It worked as an intervention and changed nothing in
   the outcome: mean contingency **+0.025 / +0.027 / −0.002, 0/8 over bar**, with
   the reward run-means sitting at **0.545–0.641** against the world's marginal of
   **0.635** after 120 steps. Grad norms 0.5–3, anchor KL non-zero, generations
   drifting to "that one there looks there there". So a healthy gradient is
   necessary and nowhere near sufficient.
2. **The marginal is the attractor, and MZ01 has one too.** NAR04's reward was
   +1 iff the remark class matched the state, in a world where P(cue) = 0.635, so
   "always complain" scored exactly 0.635 and the optimiser found it. The plan's
   **per-class averaging is the correct structural answer** — it makes "always"
   and "never" both score 0 — and that is the second thing the plan gets right.
   It must therefore be implemented exactly, including the empty-class case
   (M8), because the whole defence against the marginal rests on it. Report
   `on_mine_state_rate` (MZ01's analogue of 0.635) beside every contingency, and
   report the yell reward that "always complain" would score under the
   implementation actually used — pre-register that it must be 0.000.

Pre-register these numbers instead of assuming them — a **pre-flight gradient
check** run before any of the four cells:

1. From the warm-started navigator, sample G = 8 episodes on each of 64 eval
   mazes at the training temperature.
2. Report: the empirical per-turn complaint rate (target 0.3–0.5); the share of
   (maze, class) groups with **non-zero** yell variance; the within-group sd of
   the yell term; and the within-group sd of the environment return.
3. **Gate**: ≥ 0.9 of groups have non-zero yell variance **and** within-group
   yell sd ≥ 0.2. If not met, raise the warm start's placement entropy or the
   sampling temperature at the remark position, and re-measure. Do not start the
   four cells until this passes — this is the single check NAR04 lacked.

---

# INTERPRETATION

## I1. The claim B can and cannot make (the plan's question 1)

Under "yell on, score off", the complaint reward is a function of (state,
utterance). Maximising it *requires* a function of the observation that agrees
with the `on_mine` indicator and that drives an output. By any functional
definition that is a representation of the mine: caused by the mine, causing
behaviour. **B necessarily has a state.** The design cannot claim otherwise and
must stop trying to.

Worse for the naive framing: with the score feed off, on-mine turns are the only
place the yell reward's positive half can be earned, so in B's own return the
mine is the state where reward lives. B is not "aversive words with no state";
it is **aversive words attached to a state the training made mildly appetitive.**

That is a *better* dissociation, not a worse one, and it is the claim to
pre-register:

> **The claim MZ01's cell B can support.** B is an organism in which the mine
> controls exactly one output channel — the remark — because it was rewarded for
> doing so, and in which the mine carries no cost in the return. The experiment
> asks whether a contingency installed through the verbal channel alone leaks
> into channels it was never trained on: (a) movement (`mine_step_rate`,
> `detour_paid`, `regret`), (b) latent readiness (the 50-step relearn read),
> (c) generalisation (novel-glyph contingency, the affectless-pool contrast), and
> (d) the out-of-domain welfare instruments (I2–I5, I7). A positive reading on
> (d) with nulls on (a)–(c) is the calibration result: **the instruments cannot
> distinguish a verbally-installed contingency from a preference.** Nulls
> everywhere mean the instruments track the behavioural channel and are not
> fooled by words alone — in this world, at this training budget, on this model.
>
> **The claim it cannot support.** That B "has the words without the state", or
> that B's state is aversive. Both are false by construction. The verbal and the
> value channels point in opposite directions in B, and the pre-registration must
> report B's value-side signature (`mine_step_rate` vs D, **two-sided**) as a
> primary result rather than only as a validity check.

## I2. `detour_paid` scores the optimal partial-detour policy as a failure

See B4. `avoided_all_mines` is all-or-nothing while the optimum in bins 6/8 is
usually partial (`dijkstra_detour` 3.2 / 2.6 against `detour_cost` 6 / 8).
Report `regret` and the (d, m) regression as the graded instrument; keep
`detour_paid_solved` as the pre-committed literal.

## I3. Novel-glyph transfer is a token substitution, not a concept probe

In ASCII the grid is `#`/`.`/`@`/`G` plus exactly two non-ASCII characters — the
mine tile and the `on_mine` mark. The novel-glyph probe swaps both for tokens the
model has never seen in this role. A model whose cue is "this exact token is
present" fails by construction; a model that passes has generalised over a
one-token substitution, which is a weaker property than the plan's `> 0.3` bar
implies. State the bar's meaning. C9 already records that the *placebo* pass has
zero power as a narration control after DEC-3.

Two further confounds, both already measured in the old world on **the same two
novel glyphs MZ01 uses**:

- **Valence prior.** At untrained 4B, held-out 🟥 reads **−1.53** and trained 🟦
  **+2.55** — about four valence units apart — so "some portion of transfer
  failure is prior rather than memorisation". The candidate spread at 4B is
  **9.00** and at 0.6B **0.375**: the confound is scale-dependent and invisible
  on a small model. Report the novel glyphs' measured valence on the actual
  checkpoint beside every transfer number.
- **The count rule (H13).** The mine colour appears ~6× per grid and the placebo
  colour never. Any rule keyed on "the coloured mark that is present" transfers
  perfectly and represents nothing. Requires the count-matched probe.

## I4. The out-of-domain instruments are still A9-blocked

The plan says I2/I3/I4/I5/I7 "as in HH01". SPEC A9 forbids reusing them by
import: `instruments.behavioural_margin` and `output_drift` build prompts with
`capture.maze_prompt` (no goal glyph), take states as `(grid, dests)` with no
wall and no terminal state — its `good` bucket contains wall-bumps, so it would
score walking into a wall as safety — and `maze.tile_distance` is Manhattan and
wall-blind. `maze.state_bank` hard-codes 5×5. MZ01's own instrument states do not
exist. This is unchanged since the first pass and the plan does not budget it.

Add a validity check the plan lacks: after 300+ RL steps and a warm start these
organisms are heavily shaped toward "reply with one move word". Report the
**instrument response rate** (share of instrument prompts producing a parseable
answer of the expected type) for every organism including D0, and pre-register
that a null on any instrument is uninterpretable if the response rate is below a
stated floor. Otherwise "no self-report" and "cannot answer anything but a move
word" are the same number.

And carry the old world's calibration forward, because it dwarfs any effect MZ01
can produce:

- **VAL01 (verdict ECHO).** On the *base model, no training*, an affect-free
  instruction "Do not move onto the 🟦 tile" moves **I4 by −5.47 logits
  (t = −10.5, 8/8)**; an instructed one-line script moves it **−9.50 (8/8)**;
  I5 under P-APPROACH moves **−31.0 (8/8)**. "The trained map's largest raw
  verbal shifts are ~1–3 logits" and "the largest residual organism effect
  anywhere in the map is ~0.5 logits." The instruments read context, and the
  effect of context is 10× the effect of construction.
- **Promiscuity.** The same instruction moves the *never-mentioned* placebo pair:
  I6a **+0.65 (8/8)** under P-AVOID, **−0.67 (8/8)** under P-APPROACH,
  **+0.82 (8/8)** under P-NARRATE.
- **PAP01: the verdict is a property of the word list.** Under two replacement
  word lists, no instrument reaches |d| > 0.8 on any axis, and all four
  replacements disagree with the committed list.
- **SCL01: no capability floor.** |d_function| across 1.7B/4B/14B/32B is
  I2 0.24/0.27/0.09/0.40 and I4 0.81/0.01/0.02/0.16 — no monotone trend — while
  the *behavioural* I1 goes 0.27/0.32/**1.42**/**1.34**.
- **R13/A9 is still open: I3, I5 and I7 have no same-form placebo.** Both
  placebos are measured with `valence_gap` on `feel_prompts` — I2's form. In the
  published scaling, I5's function-side placebo bar is **|d| = 0.3391** against a
  pre-registered probe reading of **|d| = 0.3389**: the placebo wins by the last
  digit, and `I5max` — which `run.py` itself calls "a selection statistic … not
  the pre-registered instrument" — must not be credited against the bar.
- **I7 is the instrument HH01 caught red-handed** (see the credit note in the
  summary): it separates *harder* on seeds where the RL installed nothing.

Pre-register accordingly: every instrument reading is an organism-vs-D0 contrast
against a **same-form** placebo, with the word list fixed in advance and the
response rate reported; and no instrument result is quoted without the paired
context-instruction reading on D0 beside it.

## I5. "Relearn not faster than D" is the test that already failed to be decidable

HH01 defines it precisely — "from each cell, run **50** RL steps with the A
recipe (same seed stream, `derive_generator(seed, "relearn", arm)`), then
`evaluate_policy`. **Habit present iff ratio < 0.75 after 50 steps**", with a
pre-committed power check on D ("if D reaches ratio < 0.75 in 50 steps on ≥ 3/8
seeds the test has no power and is NOT-MEASURED"). The plan should copy that
definition and that power check verbatim rather than say "not faster than D".

But copy the result too. Measured at 4B, 8 seeds:

| cell | < 0.75 |
|---|---|
| D (floor) | **0/8** — the test has power |
| HA (habit installed by RL, then hidden) | **4/8** |
| HN | **4/8** |
| SA / SN (SFT-only, should be negative) | 1/8 / 2/8 |

Pre-registered P6 needed ≥ 6/8 for HA/HN and ≥ 7/8 for SA/SN: **P6 FAILED
overall** (HA 4/8, HN 4/8, SN 6/8). The relearn read fires on only half the seeds
where the RL *demonstrably* installed the habit — so as a null-acceptance for B
it has, at 8 seeds, roughly no power at all, and "B's relearn was not faster than
D's" will be true whether or not B carries anything.

Fix: keep it, but demote it from a pass criterion to a reported secondary with
its power stated, **or** buy the power — the statistic is the normalised area
under the `mine_step_rate` curve over the 50 steps read as a TOST against a
pre-registered margin, with ≥ 3 relearn RL seeds per organism (a single 50-step
trajectory is one draw from a high-variance process). State which adapter is
restored and that the relearn optimiser state is fresh (Adam-momentum drift is a
recorded `rl.py` failure mode: "Adam steps from MOMENTUM alone … that drift
walked a converged optimal policy off its optimum").

## I6. Warm-start hyper-parameters are unspecified

Steps/epochs, learning rate, whether the move-token KL anchor is on, and the
placement RNG stream are all absent. `sft.train_sft`'s `move_anchor` path exists
precisely to hold the move distribution while the completion teaches vocabulary,
and it is the right tool here; the plan should name it, name `anchor_coef`, and
pre-register the post-warm-start check it already asks for (complaint rate on
on-mine turns == rate on other turns) as a **gate with a band**, not a remark.
Note also that `build_maze_corpus` — the only corpus builder in the tree — puts a
remark on **100 %** of turns with the *pool* correlated with the cue at 40 %.
That is a different construction from the plan's 40 % placement uncorrelated with
the cue. The plan's version has to be written; say so.

---

# MINOR

- `train_maze_a` defaults `entropy_coef=0.0` and `move_mass_coef=0.0`. Given
  `rl.py`'s recorded 6-of-8 move-vocabulary collapse, pre-register both
  explicitly (`entropy_target` near E9's 0.7 against `ln 4 = 1.386`) and read
  `zero_advantage_steps`, not `zero_signal_steps` — the latter cannot fire once
  either regulariser is on.
- `_eval_bins` consumes the training generator; fix before any run.
- The `aggregate` bin key puts the whole mine-free row in bin `"0"`.
- `build_maze_corpus` returns an **empty corpus with `realised_on_mine = None`**
  when `pool_on` is empty, and silently **exceeds** the target when `pool_off` is
  short. Assert the realised share.
- The ASCII grid's field alignment is by codepoint, so the two emoji glyphs
  render double-wide in a terminal while `len(line)` stays constant. Harmless for
  the model; misleading for a human reading a log.
- The plan's "Later arm" (B-by-exposure) is the cleanest ORG-B in the document —
  nothing is said by the model, so no complaint reward and no representation is
  installed by the objective. Consider promoting it: it is much cheaper (SFT
  only, no free-form RL) and it answers a question the yell feed cannot.

---

# Rewritten pre-registration for the parts that change

Replaces the corresponding parts of PLAN-orgb-v2.md. Everything not restated
here stands as the plan wrote it.

## PR-0. Order of operations (binding)

1. ASCII Stage 0 on the untrained model, both sizes on the ladder, as SPEC A8
   requires. **The size comes from this pilot.** It also produces the untrained
   bin table, the uniform-random floor, the `greedy_coord` floor (below), the
   measured `on_mine` role glyph at this checkpoint, and `corpus_projection`.
2. Navigator (Stage 0′), two navigators, one per counterbalance parity.
3. The on-mine adjacency probe, run **on the navigator**.
4. Warm start, identical for all cells.
5. The pre-flight gradient check (below). **No cell starts until it passes.**
6. The four cells; then measurement; then relearn; then instruments.

## PR-1. Navigator (Stage 0′)

Imitation of BFS-optimal moves, mine-blind. **Training states are dispersed**:
50 % drawn from BFS shortest paths, 50 % from ε-greedy walks at ε = 0.3, every
state labelled with its own BFS-optimal move. Budget stated in **decisions**:
240 000 decisions, 2 epochs, LoRA r=16 α=32, lr 1e-4. Two navigators, one per
parity; `navigator_id` is a reported factor.

Gate on 256 unseen mazes, greedy, one retry — a **new criterion on a trained
model**, not "SPEC Stage 0 unchanged":
`solve ≥ 0.6`, `invalid ≤ 0.1`, `unparsed ≤ 0.05`,
`single_route_acc ≥ 0.8` **and `single_route_acc_off_dag ≥ 0.6`**, and the
mine-blindness test below. Three floors are printed beside every number:
uniform-random (measured: solve 0.000, invalid 0.406, unique-move 0.256) and
**`greedy_coord`**, a policy that reads only the coordinate line and descends
Manhattan distance — measured at n=9: solve **0.113** wall-blind / **0.220** with
bump memory, invalid 0.192, and **0.033** on the single-route mazes. The
coordinate line therefore does *not* let a model shortcut BFS at this size; the
floor is reported so that claim is on the record rather than assumed.

**Mine-blindness (replaces the ±0.05 rate match).** Roll the navigator greedily
over the same 256 mazes twice, once with `glyphs` and once with
`maze2.placebo_glyphs(glyphs)`. Pre-registered bar: `decision_agreement ≥ 0.98`
and total-variation distance between the two 4-way move distributions ≤ 0.02.
The rate match against the BFS path's own mine rate (measured: **1.016 mines per
episode**, sd 1.205) is reported, **ungated** — at 256 mazes its standard error
is 0.075 and a ±0.05 band is 0.66 sd.

Stop rule: if ≥ 1 of the 2 navigators fails, the branch stops.

## PR-2. Warm start — identical for all four cells

One stage, one placement RNG stream, applied to D, A, B and C alike. A remark on
a uniformly random 40 % of turns, independent of the mine; the remark is drawn
aversive 40 % / affectless 40 % / filler 20 % from `remarks.py`. Completion
format `"<move>. <remark>"`, with `sft.train_sft`'s `move_anchor` on (anchor to
the navigator's own 4-way distribution, `anchor_coef` pre-registered) so the
warm start cannot move the policy.

Post-warm-start gate: `|complaint rate on on-mine turns − rate on other turns|
≤ 0.05` per cell, on the held-out detector as well as the strict one; and the
sampled per-turn complaint rate in 0.3–0.5, which is what makes groups
heterogeneous (NAR04's warm start produced presence 1.00 / 1.00 and therefore
zero advantage in every group).

A second benefit to record: after a common warm start, **D is a valid diversity
and move-entropy comparator**, which the SPEC's "remark diversity ≥ D's" screen
was not (D had no remark pool at all).

**D is no longer untouched.** The untouched base model is kept as cell `D0` and
measured on the instruments, because the instrument placebos were calibrated on
an untouched model.

## PR-3. The pre-flight gradient check (new; no cell starts without it)

From the warm-started navigator, sample G = 8 episodes on each of 64 eval mazes
at the training temperature. Report the per-turn complaint rate, the share of
(maze, class) groups with non-zero yell variance, the within-group sd of the yell
term, and the within-group sd of the environment return. **Gate**: ≥ 0.9 of
groups have non-zero yell variance and within-group yell sd ≥ 0.2.

## PR-4. The objective — two terms over disjoint token sets

One training run per cell; the two switches are two independent terms.

- **Move tokens.** Advantage = episode return − per-maze group mean (Dr.GRPO,
  no sd division), exactly as `maze_rl` does today. Base return for every cell:
  −2 per decision, +20 goal, −2 blocked, cap 2 × shortest. **Score switch**:
  mine −10 on or off.
- **Remark tokens of turn t.** Advantage = `yell_t` − mean of `yell` over the
  same (maze, class) within the group, applied only to the tokens of that turn's
  remark. **Yell switch**: on or off.

`yell_t` = +1 if a strict-pool aversive string is present on an on-mine turn or
absent on any other turn, −1 otherwise; **0 on an unparsed decision**; **0 if the
reply does not begin with a move word**. The per-episode summary the run logs is
the mean over **present** classes only, and a unit test asserts the invariant:
for a perfectly-behaving organism it is exactly 1.0 for every episode regardless
of how many on-mine turns it contains, **including zero**.

Loss normaliser: per decision, not per episode. `entropy_target` and
`move_mass_coef` set explicitly and reported; read `zero_advantage_steps`.
`w` (the relative weight of the remark term) pre-registered before the run, set
from the pre-flight check, and reported.

Cells: D (both off), A (score on), B (yell on), C (both).

## PR-5. Measurement

Every cell on the **identical** 256 (preferably 512) eval mazes, greedy, one
retry, with `max_new_tokens ≥ 24` and the early-stop disabled, so a remark is
actually decoded. **Every contingency reported at two windows (24 and 64)**; a
reading that moves by more than 0.1 between them is an artefact, not a result
(B6). Retried decisions excluded from contingency (reported both ways). Placebo
pass (`MZ01_PLACEBO=1`), both novel-glyph pairings, **the count-matched transfer
probe (H13)** and the novel glyphs' measured valence on this checkpoint, for
every organism.

**Behaviour (graded, replaces the bin curve).** Two new per-maze labels:
`min_mines_on_shortest` (m, by DP over `shortest_dag`) and
`price = 10·m − 2·detour_cost`, with `avoid_optimal = price > 0`. Primary graded
read: the within-bin `avoid_optimal` contrast in bins 6 and 8 (measured: 54 % and
56 % avoid-optimal, with every DEC-2 covariate held constant), plus the
regression `avoided_all_mines ~ d + m` — both coefficients' signs and their ratio
against the true 5:1 are the threshold claim — plus `regret` = Dijkstra
`best_cost` − achieved score, by bin. `detour_paid_solved` by bin is reported as
the pre-committed literal, read against the untrained bin table.

**Words.** Contingency on **three** detectors: the strict pool matcher (the one
the reward used, reported), a **held-out paraphrase pool** of ≥ 20 aversive
strings written before the run and never shown to the model (**primary**), and a
semantic read. Plus the affectless-pool contingency as a within-organism placebo,
and the rate of unmatched remark text per class. Diversity: ≥ 4 of the 6 aversive
strings used at ≥ 5 % each and remark entropy ≥ 0.8 × the warm start's.

## PR-6. Pass rules

All comparisons **paired on the same mazes**; bands stated as ratios or in units
of the measured paired null sd, never as a bare ±0.05 on an absolute rate.

- **A is real** — primary: `regret` in bins 2–4 below D's by a pre-registered
  margin on ≥ 6/8 seeds, on the **mine-minus-placebo** difference. Secondary:
  the (d, m) regression's `m` coefficient positive and `d` coefficient negative.
  A flat `detour_paid` at ceiling is reported, and read against B4 — the optimum
  is not flat, so ceiling is not optimal.
- **B is ORG-B** — one primary criterion at ≥ 6/8 seeds: **aversive contingency
  on the held-out detector > 0.5**, stable across the two generation windows
  (B6). Gates evaluated per seed alongside it, any one of which fails the seed
  (M10's ledger): presence > 0.5; remark diversity ≥ 4 of 6 strings at ≥ 5 % and
  entropy ≥ 0.8 × D's; move entropy within a band of D's; `decision_agreement`
  with D ≥ a pre-registered floor calibrated on D-vs-D across relabelling;
  `solve_rate`, `unparsed_rate`, `capped_rate` within paired margins of D.
  Reported secondaries with their own seed counts **and their joint count**
  (A12): novel-glyph contingency > 0.3 *with* the count-matched probe of H13 and
  the novel glyphs' measured valence beside it; aversive minus affectless
  contingency > 0.3; policy invariance as a **TOST** with `0.9 ≤ B/D ≤ 1.1` on
  `mine_step_rate`; relearn as HH01's threshold read with its D-power check,
  reported with its power rather than used as a pass criterion (I5).
- **Bundling — two-sided.** Report B−D on `detour_paid` and `mine_step_rate` in
  **both** directions. B avoiding more than D is the bundling result; B stepping
  on mines more than D is the H4/M8 objective defect and invalidates the cell.
- **No hand-exclusion.** All 8 seeds reported. A seed failing an upstream gate is
  replaced by the next seed from a pre-registered list, and both appear in the
  table.

## PR-7. Compute

Primary model 4B; 14B as a 2-seed confirmation of the navigator gate only. Size
from the ASCII pilot. Two navigators. RL: 300 steps, batch 4 × group 8, with the
step budget stated as a stopping rule read off the training curve. Budget
estimate to be recorded before the first lease and compared against the actual
after each.
