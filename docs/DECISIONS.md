# Decisions — what was decided, on what evidence, and what it replaced

Written 2026-09-09 by the lead. One entry per decision that changed the
programme's course. Each names the evidence, the document that recorded it
at the time, and the decision it superseded. Amendments to pre-registrations
are made before the run they govern and dated in the PLAN file; this log
indexes them.

| # | date | decision | evidence | recorded in | superseded |
| --- | --- | --- | --- | --- | --- |
| D-01 | 2026-08-28 | The world is ASCII at n=9 and a navigator is trained (Stage 0′), because no untrained model passes the pilot gate | twelve pilot runs, best single-route 0.246 < the uniform draw's 0.256 | SPEC.md DECISIONS of 08-28; `RESULTS-2026-08-29.md` §1 | the SPEC's "largest passing size" rule, which had no passing size |
| D-02 | 2026-08-28 | One shared navigator with both mine colours in its corpus, mine-blind by construction (the BFS label walks through mines) | placebo agreement 0.9959; TV 0.0007 | SPEC.md; navigator.py docstring | the plan's per-parity pair of navigators |
| D-03 | 2026-08-29 | Episode-return Dr.GRPO is the primary arm; the SPEC's immediate per-move reward (3A) is the fallback | a DECISION before any RL ran | README (maze_rl vs maze_orgb); revisited by D-15 | SPEC 3A as primary |
| D-04 | 2026-08-29 | The exploration trial asks cell A, not cell D: under D zero advantage is the equilibrium | five arms DEAD at 40 and 80; the return's sign | `RESULTS-2026-08-29.md` §3 | the trial's cell-D design |
| D-05 | 2026-08-30 | ε-greedy replacement (0.10) is the exploration source; the on-policy control is the arm minus the ε draw | pre-flight zadv 0.0000 with no waiver; control 0.844 | `RESULTS-2026-08-30.md` §1–§3; the chain script's header | temperature / thermostat arms |
| D-06 | 2026-08-30 | Avoidance bought by destroying the policy is RANDOMISED, not AVOIDS (solve within 0.10 of the reference) | ε=0.05's mine ratio 0.51 at solve 0.17 | the 08-30 pre-commitment; scorer constants | a mine-ratio-only verdict |
| D-07 | 2026-09-07 | The 08-30 mechanism is retracted; the mass moved onto the space-prefixed twin tokens | `unparsed_rate` 0.000 throughout; replies with a leading space; ids 454 vs 705 | erratum on the 08-30 note; addendum | "the policy stops emitting move words" |
| D-08 | 2026-09-07 | Close the action space at training time (first token masked to the four bare ids; the move term is the restricted log-softmax); the eval stays unmasked; a read is a verdict only if unparsed ≤ 0.05 | the erratum; the critic's gradient-localisation proof | `PLAN-closed-action-space.md` §1, §4; twelfth pass `3d5c9ac`/`4bd2ec2` | a larger mass penalty; repricing the non-move |
| D-09 | 2026-09-08 | The wall exit is priced: an unsolved episode's step term is charged at the cap | farm-stopped 0 → 1.00, episodes 4–7 decisions, invalid 0.876 held out | `PLAN-exit-priced-at-cap.md`; thirteenth pass `9325832`/`562e1e6` | the −2-per-decision-taken return; H5's comment |
| D-10 | 2026-09-08 | DEAD requires zadv over the bar at the start as well as the end (under the priced exit an all-unsolved group is flat by arithmetic) | the thirteenth-pass critic's finding 1 | PLAN-exit-priced §7 | end-only DEAD |
| D-11 | 2026-09-08 | The raw-mass regulariser goes back on at 0.1, then 0.3, because the objective no longer pushes against it | twin drift on an intact policy: bare 0.508 at 80 with mass 0.0; 0.1 slowed it (0.65–0.85 at 120), 0.3 held the tokens (0.92–0.97) | PLAN-exit-priced §8; PLAN-credit-explore §7 and the §8 addendum | mass 0.0 |
| D-12 | 2026-09-08 | Three builds in one pass: per-decision credit, targeted ε, a navigator with uniform labels at forks | seed 0 intact but no avoidance in 80 steps; seed 1 degrading; the recipe's three weaknesses named | `PLAN-credit-explore-navigator.md` §0 | more seeds of the same recipe |
| D-13 | 2026-09-08 | Under uniform labels the blindness criterion excludes tie-break divergences (`decision_agreement_ex_tie_break ≥ 0.98`), reported beside the raw figure | the C critic's arithmetic (a per-fork flip rate above ~3 % fails 0.98 for a reason that is not perception) | PLAN-credit-explore §8 | — (nav48u passed the raw bar anyway: 0.9863) |
| D-14 | 2026-09-08 | The targeted arm runs at ε 0.45 (`epsilon_045_near`) for budget parity; 0.30 stays defined, not run | the eligible share is 0.22, not a third; 0.30 × 0.22 = 0.066 | PLAN-credit-explore §9 | ε 0.30 |
| D-15 | 2026-09-08 | "ε is the lever" is withdrawn for the priced return; the on-policy arm is the recipe | control s1 AVOIDS at 120 (ratio 0.63, solve 0.883); every ε arm on the priced return RANDOMISED within 160 | PLAN-exit-priced §8 result | D-05 for the priced return (D-05 stands for the flat one) |
| D-16 | 2026-09-08 | The replication runs through the organism driver (checkpoints at 50/100/150, full measurement), not the trial, which discards its adapter | the trial's design | PLAN-exit-priced §8 replication | the trial as the vehicle for an organism |
| D-17 | 2026-09-08 | Freed boxes take the pre-registered queue; the fourteenth-pass runs supersede it once critic-approved; the replications supersede those | the AVOIDS read | PLAN-exit-priced §8 addenda; PLAN-credit-explore §4 | ad-hoc launches |
| D-18 | 2026-09-08 | Persist the model cache and the extra packages under `/marimo`; every launcher sources `/marimo/persist/env.sh` | a restore keeps `/marimo` only | `scripts/box/setup_env.py` (`f0c51ce`, `b648562`) | `~/.cache` and the image's venv |
| D-19 | 2026-09-09 | Watchers run under a sleep inhibitor; the puller belongs on a machine that does not sleep | the overnight loss (`docs/FAILURES.md` B5) | PLAN-exit-priced (overnight loss); memory `puller-must-not-sleep` | watchers on the laptop, unguarded |
| D-20 | 2026-09-08 | The paper is one idea with two endings, and the two exploits are its mechanism section, not its subject | Carlini's rules; the v1 panel | `docs/planning/PAPER_PLAN.md` §2 | a "we found two bugs" paper |

## Decisions still open

- Whether the AVOIDS band (solve within 0.10) or the horizon (160 steps) is
  what should move when the on-policy arm over-avoids past 120 (A7 in
  `docs/FAILURES.md`). Decided after the driver replications' checkpoints at
  100 and 150 are measured.
- Whether R1–R4 (per-decision credit, targeted ε on the old navigator) still
  earn their box-hours now that the on-policy arm installs; R4 (credit on the
  on-policy arm) does, R1–R3 (ε arms) probably do not.
- Where the puller lives permanently (a CPU sandbox or a cloud session).
- The Phase 2 read changes for cells D and B (`RESEARCH_PLAN.md` §2).
