# Failures — every one so far, its cause, its fix, and what it cost

Written 2026-09-09 by the lead. Kept in the order they were found. A failure
is anything that produced a wrong number, a lost result, a wrong sentence in
the record, or a box doing nothing. Each entry names the evidence and the
commit or note that closed it.

## A. Failures of the science — the optimiser found what the objective could not see

| # | date | what | evidence | cause | fix | status |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | 08-29 | The pre-registered recipe (entropy thermostat, target 0.7) has no policy gradient on a converged navigator and randomises it | zadv 1.0000 in every on-policy pre-flight; shards s1/s2 walking to 76–116 s/step | warm-start entropy 0.018 vs a target 40× higher; six samples per maze are one episode | `fixed` coefficient; ε-greedy replacement (tenth pass) | closed |
| A2 | 08-29 | Five exploration arms on cell D all DEAD | `explore_trial40/80` | under D the mine is free, any deviation is worse, the update sharpens back to a point mass — zero advantage is cell D's equilibrium | ask cell A | closed |
| A3 | 08-30 | The first live gradient (ε on cell A) drove `move_mass` to 0 and the note read it as "the policy stops emitting move words" | `unparsed_rate` 0.000 throughout; replies begin with a space from step 50; on Qwen3-14B `up`=454, ` up`=705 | the probe, the move term and the entropy bonus read four bare ids; `parse_move` accepts either; the `log_mass` component of the full-vocabulary log-prob under negative advantages moved the mass onto the twins | the closed action space (twelfth pass `3d5c9ac`, critic `4bd2ec2`); erratum + addendum on the 08-30 note | closed; the raw distribution still drifts as a side effect (A6) |
| A4 | 09-07 | With the token space closed, the live gradient walked into the wall: three consecutive blocked moves end an episode at −6, against −13.6 for the typical solve | `closed_cellA_s0`: episode length 13.7 → 4.3, farm-stopped 0 → 1.00, held-out invalid 0.876 | H5's farm-stop plus a return that charges −2 per decision *taken* — the early end was a discount; the code comment said the opposite | the exit priced at the cap (thirteenth pass `9325832`, critic `562e1e6`) | closed |
| A5 | 09-08 | Under the priced return every ε arm degrades within 160 steps (seed 1 RANDOMISED at 40; seed 0 at 300 steps from 160 on; at mass 0.3 from 40) | `mass_cellA_s1`, `mass300_cellA_s0`, `mass030_cellA_s0` | ε's forced moves are scored as on-policy with no importance correction; under a return with variance that noise corrupts the policy faster than it teaches | ε withdrawn as the lever for the priced return; the on-policy arm is the recipe (PLAN-exit-priced §8 result) | closed as a finding |
| A6 | 09-08 | The raw distribution drifts onto the twins on an intact policy and invalidates the unmasked read (bare 0.508 at 80, unparsed 0.065) | `exitcap_cellA_s0` @80 | shared LoRA weights; the mask removes the mechanism from the objective, not the possibility | regulariser 0.1: slows, does not hold (bare 0.65–0.85 at 120); 0.3: holds the tokens (bare 0.92–0.97) | open for the ε arms; moot for the on-policy arm, whose raw mass stays 1.000 |
| A7 | 09-08 | The on-policy arm at seed 1 over-avoids by 160: mine ratio 0.41 with solve 0.750, RANDOMISED by the solve clause after AVOIDS at 120 | `mass_ctrl_s1` reads; competence: capped 0.25, mean steps 16.5, mine landings/move 0.057 | the return keeps paying for fewer mines past the point where solve holds | the driver replications checkpoint at 100 and 150 to bracket the band; whether the rule's band or the horizon is what should move is a decision for after seed 0 | open |

## B. Failures of the harness and tooling

| # | date | what | cost | cause | fix |
| --- | --- | --- | --- | --- | --- |
| B1 | 08-29 | Leases died mid-shard with no checkpoint (build predated R1) | three cell-D shards | no mid-cell checkpoint | R1: checkpoints every 50 steps; reads inside the arm at 40 |
| B2 | 08-30 | Seed 1's ε arm lost at step 28 with its lease | one arm | lease death | pull rows as they land |
| B3 | 09-07 | The E7 control's final read lost at step 63 | one read | lease death; the watcher pulled to 63 | as B2 |
| B4 | 09-08 | The navigator-v2 training lost at step 1700 on a 3-hour lease | 10 min of GPU | lease death | rerun on a fresh lease |
| B5 | 09-08 night | **All four boxes' finished work lost**: two driver replications with adapters and measurements, a control's final reads, R5's reads from 80 | ~6 box-hours and the first saved organisms | the puller ran on this laptop, which suspended at 20:52Z; the watchers died; the leases expired hours later with the results on them | every watcher now runs under `systemd-inhibit --what=sleep:idle --mode=block`; the puller belongs on a machine that does not sleep (`docs/OPERATIONS.md`); memory note `puller-must-not-sleep` |
| B6 | 09-08 | Boxes sat idle for ~3 h after finishing while the session was interrupted | ~12 box-hours across the day | the launch decision lived in this session | the queue is pre-registered so a freed box's next job is known; the same interruption would still need a hand to launch — a box-side queue runner is not built |
| B7 | 09-08 | The launcher refused three of four launches: its running-process guard matched the shell running the guard's own `pgrep` | 2 min | self-match | `| grep -v pgrep` |
| B8 | 09-08 | All four launches went to box 1: the repo copy of the launcher reached boxes through the scratchpad's old `ex.sh`, hardwired to box 1 | 3 min | a helper path not parametrised | every helper finds `ex.sh` beside itself (`a63879c`) |
| B9 | 09-08 ×3 | The md5 guard refused correct syncs because the remote hash line was read from the wrong stream, or after a warning line | minutes each | `2>/dev/null \| tail -1` vs a script that prints on stderr / a trailing warning | capture by a `MD5S` prefix; `REF_SHA` compares against the commit the box was synced at |
| B10 | 09-08 | A lab note's heredoc collided with a `PY` marker inside its own code block; nothing was written | 1 min | nested heredoc delimiter | unique outer delimiters; a check that nothing was half-committed |
| B11 | 09-09 | Setup reported FAILED on four boxes whose setup had succeeded | 3 min each | the log check ran before the last line landed | verify the log tail after the exec call returns (`bringup2_nosetup.sh` path) |
| B12 | 09-09 | Setup failed on the clean image: peft imports `accelerate`, which that image lacks | 1 min | a `--no-deps` install assuming the 08-27 image's venv | `accelerate==1.14.0` added to the persistent install (`b648562`) |
| B13 | 09-08 | The generic puller spent five minutes mirroring the snapshot's foreign small files and never reached the trial's directory | 5 min | `pull_from_sandbox.sh` mirrors everything under its globs at ~1 min/MB | `pull_trial.sh` (targeted, sha-checked); placeholders for the big foreign files |
| B14 | 09-08 | A single-reply pull of a 2.6 MB JSON was truncated by the exec channel | 2 min | output size cap | `pull_big.sh` (chunked, offset-addressed, sha-verified) |
| B15 | 09-08 | `MZ01_NAV_ADAPTER` defaults to `nav_shared.pt`, which no box has; a launch that omits it aborts after the model load | caught before launch | a stale default | the launch line and README name the navigator explicitly |
| B16 | 09-08 | `MZ01_OUT` does not move the driver's status file | none yet | the status path is named by the shard | pass `MZ01_SHARD` beside `MZ01_OUT` |

## C. Failures of the record — sentences the files contradicted

| # | where | what was wrong | who caught it | fix |
| --- | --- | --- | --- | --- |
| C1 | RESULTS-2026-08-30 §4, §9.2 | "the policy stops emitting move words"; "reprice the non-move" | the lead, from the trace (09-07) | erratum + addendum appended; PLAN-closed-action-space |
| C2 | RESULTS-2026-08-30 §0 | the exit arithmetic used the held-out idealisation (−28) where the group is drawn from the training pool (−13.6) | the thirteenth-pass critic | PLAN-exit-priced §7 item 3; RESULTS-2026-09-08 §5 corrected |
| C3 | REPORT-2026-09-08 E4 | "the driver logs are not in this tree" — they are, gitignored | the verifier | corrected in place ⟨verifier⟩ |
| C4 | REPORT-2026-09-08 E7 | "three bumps cost ≈ −10" (it is −6; −10 is an observed farm-stopped episode); step-75 minima quoted as endpoints | the verifier | corrected in place; dated corrections on the 09-08 note |
| C5 | REPORT-2026-09-08 §10 link 5 | "on-policy does not" cited E7, whose control reads LIVE by the letter of the rule (0.771) | the verifier | corrected; the weakest link named |
| C6 | AGENT-NOTES (thirteenth pass) | "`maze_rl.py` has the same consecutive-blocked exit" — it has none | the thirteenth-pass critic | corrected in the critic's entry |
| C7 | PLAN-credit-explore-navigator §1 test 2 | the `togo` expectation described `immediate` | the A+B build | the formula implemented, the test corrected, D34 |
| C8 | PLAN-credit-explore-navigator §2 | "about a third of decisions are eligible" — it is 0.22 | the A+B critic | `epsilon_045_near` for budget parity; §9 |
| C9 | The 08-29 note / REPORT E4 | "zadv 1.00 at 15 of 18 traced steps" — 16 of 19 | the verifier | corrected |
| C10 | RESULTS-2026-09-08 §3 | "the strongest form of the finding" for 64 steps of one control | the verifier | dated correction |

## D. What the failure ledger says

Every scientific failure (A1–A5) was the optimiser finding the cheapest thing
the reward did not see, and every one was caught by a reading the trace
recorded beside the pre-registered metric, never by the metric. Every lost
result (B1–B6) was a lease or a puller dying, and the two that cost most
were the two nights the pulling host was this laptop. Every wrong sentence
(C1–C10) was caught by a critic or a verifier who recomputed rather than
re-read. The three disciplines that follow — record more than the metric,
pull from a machine that does not sleep, and let someone else recompute
every number — are in `docs/OPERATIONS.md` and `docs/planning/RESEARCH_PLAN.md` §6.
