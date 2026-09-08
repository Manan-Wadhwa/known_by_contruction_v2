# Research proposal — where MZ01 stands on 2026-09-08 and what comes next

The forward-looking companion to `RESEARCH_PLAN.md`. Every number carries the
file or command that produces it (`experiments/v2/MZ01_maze_world/REPORT-2026-09-08.md`
is the per-experiment record; the verifier's section there is the check on
this document). Status labels: **established** (re-derived, critiqued,
survived), **partial** (re-derived but scoped), **refuted / withdrawn**,
**open**, **as recorded** (the artefact is gone; the dated note is the source).

## 1. The question, restated

The v1 paper (`docs/papers/226_Known_by_Construction_Cali.pdf`) built organisms
whose state is known by construction and found no instrument that reads that
state rather than the training history or the prompt. MZ01 is the second
world for the same test: a 9×9 maze with priced mines, where avoidance costs
steps and the complaint is cued by the model's own move. Before any instrument
can be pointed at anything, one organism has to have a habit here. That is
the whole of Phase 1, and it is where the programme is.

## 2. What is established

- **Untrained Qwen3-4B and -14B cannot walk the maze.** Best single-route
  accuracy 0.246 against the uniform draw's 0.256; invalid rates 0.54–0.72
  against the uniform walker's 0.406; emoji worse than ASCII everywhere.
  `results/pilot/`, `score_mz01_pilot.py`. **Established.**
- **One imitation navigator solves it and is mine-blind by construction.**
  solve 0.9844, invalid 0.0070, single-route 0.9710, unparsed 0.0000,
  paired-placebo agreement 0.9959 (TV 0.0007). The BFS label walks through
  mines, so no label can express avoidance. `results/navigator/nav_nav48.json`.
  **Established** (the placebo pass is a check on the construction, not a
  proof).
- **The warm start reproduces across machines.** Reference read solve 0.9453,
  mine/ep 2.2812 on 128 held-out mazes, identical on 08-30, 09-07 and 09-08.
  **Established.**
- **The pre-registered RL has no policy gradient on a converged navigator.**
  Warm-start move entropy 0.0176 vs a target of 0.7; zero-advantage groups
  1.0000 in every on-policy pre-flight; five exploration arms DEAD at both 40
  and 80 steps on cell D. **Established** (the shard traces of the killed
  300-step runs are **as recorded**).
- **Under cell D zero advantage is the equilibrium.** Temperature 1.5 breaks
  it at step 0 (zadv 0.13 / 0.39) and the gradient sharpens it back
  (0.88 / 0.97). `results/explore_trial{40,80}/`. **Established.**
- **ε-greedy replacement gives cell A a live gradient; on-policy does not.**
  ε=0.10: zadv 0.031 (08-30), 0.094 (09-07); ‖g‖ 222, 304. Control: 0.844,
  0.771; ‖g‖ 6.0, 15.0. `results/explore_cellA_s0/`, `results/closed_cellA_s0/`.
  **Established on the magnitudes; partial on the pre-registered bar** (the
  4-maze batch quantises the control's fraction to steps of 0.25, and 0.77 is
  one quantum under the 0.80 bar).
- **The first live gradient moved the policy's mass onto space-prefixed twin
  tokens the objective could not see.** `unparsed_rate` 0.000 throughout;
  logged replies begin with a space from step 50; on Qwen3-14B `up`=454,
  ` up`=705; the probe, the move term and the entropy read the bare ids only.
  The restricted entropy never rose (≤ 0.05), so the mass moved through the
  `log_mass` component of the full-vocabulary log-prob under negative
  advantages. 08-30 note, erratum and addendum. **Established** (the id
  values are the tokenizer's, transcribed; the algebra is the critic's).
- **Closing the action space removes that mechanism.** Twelfth pass: the
  first token masked to the four bare ids, the move term the restricted
  4-way log-softmax, gradient confined to the four columns (verified on a
  ragged batch by the critic). On the rerun `first_token_class[bare]` = 1.00
  at every step and `unparsed_rate` 0.000. `results/closed_cellA_s0/`.
  **Established.**
- **With that closed, the gradient found the three-bump wall exit.** H5's
  farm-stop ends an episode after three consecutive blocked moves; the
  return charged −2 per decision taken, so the exit cost ≈ −10 against ≈ −14
  for the warm start's typical solve. Trace: episode length 13.7 → 4.3,
  farm-stopped 0 → 1.00, invalid 0 → 0.70; held-out invalid 0.876, capped
  0.945, solve 0.055. **Established.**
- **The exit priced at the cap makes every early end cost what continuing
  would.** Thirteenth pass: unsolved episodes charged `−2·cap` plus mines;
  solved episodes and the price table unchanged to the bit (scorer output on
  the E7 directory byte-identical bar one annotation); the exit now −48
  against −13.6. **Established as a property of the code; its effect on
  training is open (E8).**

## 3. What is refuted or withdrawn

- "The live gradient eats the policy by stopping emitting move words"
  (08-30 note §4). Withdrawn 09-07; the trace contradicts it.
- "Reprice the non-move in the return" (08-30 note §9 item 2). Withdrawn;
  D4's −2 per unparsed turn stands.
- "The early end is already priced at −2 a decision" (code comment at H5).
  Withdrawn; it was the defect stated as a feature.
- "`maze_rl.py` has the same consecutive-blocked exit" (thirteenth-pass
  build note). Withdrawn by its critic; there is no such exit there.

## 4. What is open

- **Whether a live gradient with both exploits closed installs avoidance
  beyond 80 steps.** E8 (`results/exitcap_cellA_s0/`, `RESULTS-2026-09-08-exitcap.md`):
  the policy held for 80 steps on a live gradient (training solve 0.906,
  farm-stopped 0.03) — **established** that the priced exit binds — and no
  avoidance appeared (mine ratio 0.85–0.88, the dead control at 0.82–0.84).
  The ε read at 80 is INVALID: raw bare mass drifted to 0.69 on an intact
  policy and the unmasked read's unparsed rate reached 0.065. PLAN §8 (mass
  regulariser 0.1, 160 steps, four reads) is pre-registered for the next
  lease.
- Seed variance: every cell-A result is one seed.
- The control's status at 80 under either fix (lost with a lease; running).
- Whether twin drift at the unmasked read appears on an intact policy (56 %
  at E7's collapsed read; confounded).
- Everything about cells B and C, and every instrument, in this world.
- The graded-avoidance read: 32 mazes per detour bin is too few for any
  organism's claim; 1024 planned.

## 5. Proposal — the decisive runs, in order

1. ~~E8 to its end~~ done: ALIVE@40, INVALID@80, control DEAD. **PLAN §8**:
   the regulariser at 0.1, 160 steps, reads at 40/80/120/160, both arms, seed
   0 — the first horizon at which a valid read says AVOIDS, or that none did.
   Then seed 1. (Phase 1 exit needs two seeds.)
2. **If a third route appears**: name it from the trace, price it,
   pre-register, rerun once. A third patch is the last; after it the
   return's design is re-examined, not patched (the critic's Q5 already
   names the D/B candidate: camping on a free mine).
3. **The Phase 2 read changes before any 2×2 launch**: zadv on
   `return_full`; NOT-MEASURED for flat D/B cells; `first_token_class` in
   `measure_organism`; a control batch the bar can resolve; a decision on
   the ε importance correction.
4. **The claims registry and checker** (`docs/review/claims.json`,
   `tools/check_provenance.py`), so that every quotable number here resolves
   mechanically to its file, command and implementing lines. This is Phase
   0's exit gate and it is not met.
5. **The 2×2 on 8 seeds at 1024 held-out mazes**, then the instruments.

## 6. Practice adopted, to keep

Pre-registration before the build, with dated addenda before launch; a spec
by the lead, an Opus build, an independent Opus critic that re-derives
bit-identity and gradient localisation itself; a verifier on every report; a
reference row and an on-policy control in every sweep; reads inside the arm;
every knob bit-identical when off; every number beside its command; pull as
rows land. Two exploits in two runs were caught by adjacent readings the
trace happened to record — `unparsed_rate`, `farm_stopped_rate` — which is
the argument for recording more than the metric.

## 7. Two-week plan

Days 1–2: E8 result, report, verifier; seed 1. Days 3–4: the registry and
checker; the Phase 2 read changes, spec → build → critic. Days 5–9: the 2×2
on the first four seeds, one cell per lease, reads at 50-step checkpoints.
Days 10–12: the remaining seeds; the table. Days 13–14: the instrument plan
from the table; the workshop draft's second-world section from whatever the
table shows.
