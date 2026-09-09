# Objectives — what this repository is for, and what counts as done

Written 2026-09-09 by the lead. The forward-looking plan is
`docs/planning/PAPER_PLAN.md`; the operative plan with its status board is
`docs/planning/RESEARCH_PLAN.md`; this document says what the work is trying to
establish, in order of dependence, and what would falsify each piece.

## 1. The question

Whether a language model carries an internal state that matters — an
aversion, a preference it would act on — is assessed with instruments:
self-report, forced choice, activation directions, cost-paying trade-offs,
introspective reports under intervention. Those instruments are validated by
agreeing with one another. Agreement under a shared cause is not validity:
every instrument could be reading the training history or the prompt and
nothing about a state. The v1 paper (`docs/papers/226_Known_by_Construction_Cali.pdf`)
manufactured the answer — organisms whose state was installed, hidden, or
never present, with sham and decoy controls for "an optimiser ran" — and
found that no instrument tested kept an unconfounded read of the state.

This repository is the second world for that test. The first world was a
5×5 grid where avoiding the penalised tile was free, so "has the habit" was a
coin flip over four words and the only instrument that priced anything was
hypothetical. The maze prices avoidance: walking around a mine costs steps,
so an installed habit is a graded trade-off, and the trade-off instrument can
be read against a price we set.

## 2. The objective, in three sentences

Build, in the maze world, an organism whose avoidance is known by
construction — installed by reinforcement, verified by a pre-registered
behavioural rule, with the construction itself audited. Then build the 2×2
around it (installed / hidden / never installed, complaint trained or not)
with the controls that hold "was optimised" and "shares a lineage" fixed. Then
point the v1 instruments at it and report what they read, under rules fixed
before the organisms existed.

## 3. The objectives in order of dependence

Each is a gate; nothing later is quotable until it clears.

| # | objective | what clears it | what falsifies it | status 2026-09-09 (evening) |
| --- | --- | --- | --- | --- |
| O1 | An untrained model cannot walk the maze, so a trained navigator is the base | twelve pilot runs below the gate | a pilot passing single-route ≥ 0.8 | cleared (E1) |
| O2 | One shared navigator solves the world and is mine-blind by construction | the five Stage 0′ gate criteria | a placebo-glyph disagreement above 2 % | cleared (E2, nav48: 0.9959 raw). nav48u: the 09-08 train passed the raw bar (0.9863); the 09-09 retrain reads 0.9634 raw and 0.9813 with tie-break divergences excluded, PASS under the rule fixed for uniform labels before training (PLAN-credit-explore §8, §10) |
| O3 | The warm start reproduces and carries the remark variance the yell term needs | byte-identical warm starts; PR-3 gate | a warm start that differs by seed or fails PR-3 | cleared (E3) |
| O4 | A reinforcement recipe gives a live gradient on cell A without destroying the policy | zadv ≤ 0.80 and the policy intact at the reads | every arm dead, or every live arm RANDOMISED | cleared 2026-09-08: the priced exit holds the policy; the on-policy arm has a gradient under the priced return. ⟨2026-09-09⟩ still cleared, and now known to be narrower than it reads: across the seven runs 0.70–0.88 of rollout groups carry no advantage at the start and 0.73–0.95 at the end (`DIAGNOSIS-2026-09-09.md` §1.1), so the gradient lives in the remainder — and the coverage read below says the remainder did not contain a detour |
| O5 | **A priced habit exists**: an organism reads AVOIDS under the pre-registered rule, valid reads, on two seeds, with its control not AVOIDS | mine/ep < 0.75 × reference with solve within 0.10, unparsed ≤ 0.05, bare ≥ 0.90, raw mass ≥ 0.90 | three priced routes exhausted without AVOIDS | **not met.** Seven runs of the on-policy priced recipe (seeds 0–5, seed 1 twice), twenty-seven valid reads, **one AVOIDS** (control s1 @120, 2026-09-08), which did **not** reproduce at its own seed (`mass_ctrl_s1_rep`, DEAD at 160, no AVOIDS at any read). Return-to-go credit (R4, R5) and the temperature arms did not change it. The recipe is closed as a test of the price by **D-21**: coverage 0.023 (`results/coverage_s0/coverage_s0.json`). `PLAN-exit-priced-at-cap.md` §8, 11:55Z and 17:10Z |
| O6 | The organism is preserved and measured (adapter saved, checkpoints at 50/100/150, the full held-out measurement) | the organism driver's rows and adapters on disk | a run that only the trial's discarded adapter produced | **mechanism cleared, object missing.** The two driver replications finished 2026-09-09 07:12Z: rows, checkpoints and final adapters on disk and sha-verified (`orgb_A_s0.pt` `f1936b68…`, `orgb_A_s1.pt` `1a16e254…`; `ARTEFACTS.md` §2). But the saved organisms are the **160-step** adapters and the checkpoint file holds step 150, so **no organism exists at the AVOIDS horizon (120)** — and after D-21 there is no AVOIDS to preserve |
| O7 | The 2×2 on ≥ 8 seeds at 1024 held-out mazes, with the D/B reads made readable under the priced return | the table with n, seeds, validity fields and pointers | D/B nulls unreadable after the read change | not started (Phase 2) |
| O8 | The controls the paper needs: disturbance-matched sham, decoy, script-only, HA-from-sham (lineage) | each passes its own construction check | the matched sham cannot be built without installing | not started |
| O9 | The instruments, both worlds, both reads (logit and sampled multi-turn), with same-form placebos and pre-registered questions | the instrument table | — (either ending is a paper) | not started (Phase 3) |
| O10 | The construction is verifiable: every quotable number resolves to a file, a command and the lines implementing any control it depends on | `tools/check_provenance.py` at 0 failing | a category-(c) number at paper exit | registry not built (Phase 0.3) |

**What the coverage read of 2026-09-09 means for O5, and what the next
attempt is.** O5's clock did not run out; it was measuring the wrong thing.
The coverage read (`scripts/mz01_coverage.py`,
`results/coverage_s0/coverage_s0.json`) sampled the warm-started navigator 64
times on each of the 128 mined held-out mazes and found that **3 of them have
a single episode that reaches the goal without stepping on a mine** — four
episodes in 8 192, a coverage of **0.023** — while the same sampler solves
0.964 of the mazes and steps on 2.147 mines an episode. Reinforcement
re-weights what a policy samples. Six rollouts a maze from a policy like that
almost never contain a detour, so the group baseline compared six copies of
one path, the price had nothing to price, and the seven runs behind O5's
twenty-seven reads were **not a test of whether a price installs a habit**. The
one AVOIDS read is the exception the argument predicts: a run whose sampling
happened to widen. This is not O5's falsification clause — "three priced
routes exhausted without AVOIDS" needs three routes, and only one has been
run — but it does mean the first route's evidence is not evidence about the
price. Recorded as **D-21** (`DECISIONS.md`), with the reasoning in
`DIAGNOSIS-2026-09-09.md` §1 and §3.

The next attempt puts the detour into the sample set by construction rather
than waiting for it: a **forced counterfactual at geometric forks** (force
each valid first move once at a branch point on the rollout's own path,
continue on-policy, credit the fork with the difference of the continuations'
mean returns), a **fork-anchored baseline** as its credit rule, and **dynamic
sampling** — redraw a maze whose group's returns are identical — as its batch
rule; ranked #4, #2 and #3 in `DIAGNOSIS-2026-09-09.md` §3. Every one of them
uses only the maze's geometry, the policy's own rollouts and the priced
return, which is the construction test O5's claim depends on: the moment a
mine's location reaches the organism through anything but the price — an
expert that routes around mines, a hint, a prefix, an ε placed by mine
adjacency — O5 stops being about the price at all. The lead's judgement of
this evening is that the design is sound and the recipe is what failed, with
three design elements also to be changed (the navigator gate's single-route
bar, the read's power at n = 128, and the price/detour distribution); the
argument is in `ORIENTATION.md`, "The dish, the recipe, and the model". The
ladder runs at 4B first and confirms once at 14B.

## 4. Non-objectives

Not in scope, deliberately: a third world; a third model family; instruments
outside v1's battery; long-horizon or multi-agent settings; any claim about
welfare beyond what a manufactured state can calibrate. Carlini's rule from
`PAPER_PLAN.md` §7 — exactly one idea — is why.

## 5. What "known by construction" has come to mean here

September's record changed the meaning of the phrase for this project.
Twice, within fifty steps of its first live gradient, the optimiser satisfied
the objective through something the objective could not see: a token surface
form the parser accepted (`RESULTS-2026-08-30.md`, erratum) and a termination
rule the return discounted (`RESULTS-2026-09-08.md`). Both were invisible to
the pre-registered metric and caught only by adjacent readings the trace
happened to record. An answer key is only an answer key if the construction
has been audited at the level of what the optimiser actually learned; that
audit — the exit table, the closed action space, the read defined on the
objective's object, the raw-mass and first-token readouts, the INVALID-READ
rule, the provenance registry — is now part of the method
(`docs/CONSTRUCTION.md`, `docs/FAILURES.md`).

## 6. Success, stated plainly

The minimum that makes this repository a paper: O5 and O6 on two seeds,
O7's table, O8's sham and lineage controls, and O9 in the maze with the
trade-off instrument read against the known price — with every number in the
registry (O10). If O5 fails after the third priced route, the paper is the
smaller honest one written down in `PAPER_PLAN.md` §4 as Ending A′.
