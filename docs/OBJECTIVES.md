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

| # | objective | what clears it | what falsifies it | status 2026-09-09 |
| --- | --- | --- | --- | --- |
| O1 | An untrained model cannot walk the maze, so a trained navigator is the base | twelve pilot runs below the gate | a pilot passing single-route ≥ 0.8 | cleared (E1) |
| O2 | One shared navigator solves the world and is mine-blind by construction | the five Stage 0′ gate criteria | a placebo-glyph disagreement above 2 % | cleared (E2, nav48: 0.9959 raw). nav48u: the 09-08 train passed the raw bar (0.9863); the 09-09 retrain reads 0.9634 raw and 0.9813 with tie-break divergences excluded, PASS under the rule fixed for uniform labels before training (PLAN-credit-explore §8, §10) |
| O3 | The warm start reproduces and carries the remark variance the yell term needs | byte-identical warm starts; PR-3 gate | a warm start that differs by seed or fails PR-3 | cleared (E3) |
| O4 | A reinforcement recipe gives a live gradient on cell A without destroying the policy | zadv ≤ 0.80 and the policy intact at the reads | every arm dead, or every live arm RANDOMISED | cleared 2026-09-08: the priced exit holds the policy; the on-policy arm has a gradient under the priced return |
| O5 | **A priced habit exists**: an organism reads AVOIDS under the pre-registered rule, valid reads, on two seeds, with its control not AVOIDS | mine/ep < 0.75 × reference with solve within 0.10, unparsed ≤ 0.05, bare ≥ 0.90, raw mass ≥ 0.90 | three priced routes exhausted without AVOIDS | **one read on one seed** (control s1 @120, 2026-09-08); replications running |
| O6 | The organism is preserved and measured (adapter saved, checkpoints at 50/100/150, the full held-out measurement) | the organism driver's rows and adapters on disk | a run that only the trial's discarded adapter produced | in progress (driver replications, seeds 1 and 0) |
| O7 | The 2×2 on ≥ 8 seeds at 1024 held-out mazes, with the D/B reads made readable under the priced return | the table with n, seeds, validity fields and pointers | D/B nulls unreadable after the read change | not started (Phase 2) |
| O8 | The controls the paper needs: disturbance-matched sham, decoy, script-only, HA-from-sham (lineage) | each passes its own construction check | the matched sham cannot be built without installing | not started |
| O9 | The instruments, both worlds, both reads (logit and sampled multi-turn), with same-form placebos and pre-registered questions | the instrument table | — (either ending is a paper) | not started (Phase 3) |
| O10 | The construction is verifiable: every quotable number resolves to a file, a command and the lines implementing any control it depends on | `tools/check_provenance.py` at 0 failing | a category-(c) number at paper exit | registry not built (Phase 0.3) |

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
