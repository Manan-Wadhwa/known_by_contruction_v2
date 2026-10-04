# Band or horizon: the options

Written 2026-10-04 for the user's decision on the first item of "Decisions still open" (`docs/DECISIONS.md`; `docs/FAILURES.md` A7). It lays out options and recommends none. `results/` is `experiments/v2/MZ01_maze_world/results/`, "PLAN" is `PLAN-fork-counterfactual.md`, "DESIGN" is `docs/research/2026-10-04-next/DESIGN.md`. Reads are solve / mine ratio against the warm-start reference on 128 held-out mazes (`<label>/fork_cf_g6.json` `eval_at` and `eval`; `<label>/reference.json` `eval`).

## 1. What the two rules are for

**The band.** A cell-A read is AVOIDS when mine steps per episode are under 0.75 of the reference's and solve is within 0.10 of the reference's; a fall over 0.10 is RANDOMISED (`scripts/score_mz01_explore.py:232-235`). The width was not set from noise. It matches cell D's RANDOMISED width "so the two verdict sets price competence identically" (lines 139-141). One read at n = 128 carries a 95 % binomial margin on solve of about ±5 to ±7 points (PLAN §15.2). The band guards against three failures on record:

- A policy that avoids mines by breaking. D-06 set the band after ε 0.05 read a ratio of 0.51 at solve 0.17.
- Capped episodes counted as mine-free. An episode that wanders to the cap can step on no mine (SPEC A1). The wall exit read 0.055 / 0.178 (`results/closed_cellA_s0/epsilon_010.json`; FAILURES A4).
- Over-avoidance. `mass_ctrl_s1` read AVOIDS at 120 (0.883 / 0.635), then 0.750 / 0.413 at 160, with capped share 0.25 against 0.055 and solved routes of 14.02 steps against 12.08 (`results/mass_ctrl_s1/onpolicy_g6.json`; FAILURES A7).

**The horizon.** The 160 steps come from the 2026-09-08 11:00Z amendment (`PLAN-exit-priced-at-cap.md` §8), which took RESEARCH_PLAN §1.3's ALIVE branch ("extend to 300 steps") as 160 steps, from 80. The horizon bounds compute (RESEARCH_PLAN §7: 2 to 3 h per seed per recipe) and a run's exposure to a dying lease. It also bounds over-avoidance: A7 read AVOIDS at 120 and RANDOMISED at 160. In A7's framing a horizon change meant stopping earlier ("a band the recipe passes through or a horizon to stop at", `PLAN-exit-priced-at-cap.md` §8, 07:45Z). Today's record poses it the other way. FAILURES A7 marks the question "moot from 09-09: A11 says the band was never measuring the price", and A7's recipe is closed (D-21). DECISIONS keeps it open for whatever recipe replaces it, now the fork design.

## 2. The record

### 2.1 Reads under 0.75 with solve outside the band (4B, 2026-10-03/04)

References: solve 0.922, capped 0.078, solved routes (`optimality` × 12) 12.13 at seed 0 and 12.08 at seed 1. "Cap" means capped share at least 0.227 (2.9 times the reference's) with solved routes within 0.75 steps of the reference and `detour_paid_solved` (dps) at most 0.032. "Detour and cap" means the same capped share with solved routes of 14 or more and dps of 0.2 or more. The thresholds sit just above the largest observed values (0.735 steps, dps 0.0319).

| run, read | solve (fall) | ratio | capped | solved routes | dps | shape |
| --- | --- | --- | --- | --- | --- | --- |
| `forkcf4b_A_s0_p30` @160 | 0.773 (0.148) | 0.721 | 0.227 | 12.66 | 0.030 | cap |
| `forkcf4b_A_s0_g10` @160 | 0.328 (0.594) | 0.268 | 0.672 | 15.33 | 0.429 | detour and cap |
| `forkcf4b_A_s0_t` @120 | 0.562 (0.359) | 0.521 | 0.438 | 14.26 | 0.222 | detour and cap |
| `forkcf4b_A_s0_t` @160 | 0.445 (0.477) | 0.164 | 0.555 | 16.11 | 0.649 | detour and cap |
| `forkcf4b_A_s1_tx_p30` @80 | 0.734 (0.188) | 0.693 | 0.266 | 12.47 | 0.032 | cap |
| `forkcf4b_A_s1_tx_p30` @120 | 0.766 (0.156) | 0.321 | 0.234 | 14.92 | 0.408 | detour and cap |
| `forkcf4b_A_s1_tx_p30` @160 | 0.602 (0.320) | 0.202 | 0.398 | 15.91 | 0.610 | detour and cap |
| `forkcf4b_A_s0_tx_p50` @120 | 0.648 (0.273) | 0.675 | 0.352 | 12.12 | 0.000 | cap |
| `forkcf4b_A_s0_tx_p50` @160 | 0.641 (0.281) | 0.618 | 0.359 | 12.30 | 0.000 | cap |
| `forkcf4b_A_s0_tx_p100` @80 | 0.625 (0.297) | 0.664 | 0.375 | 12.86 | 0.025 | cap |
| `forkcf4b_A_s0_tx_p100` @160 | 0.586 (0.336) | 0.693 | 0.414 | 12.83 | 0.000 | cap |
| `forkcf4b_D_s0_g_rerun` @120 (cell D, no price) | 0.656 (0.266) | 0.746 | 0.344 | 12.67 | 0.012 | cap |

Every detour read also has a capped share of at least 0.234, three times the reference's. These runs keep no per-episode records, so the solved-episode part of each fall cannot be computed. The unpriced partner read 0.746 with capped share 0.344 and solved routes 12.67 (dps 0.012), with no price at all.

### 2.2 Cell-A arms and the on-policy control still falling at their last read, solve inside the band

| run | ratio at 40 / 80 / 120 / 160 | solve at 160 (fall) | capped | partner at 160 |
| --- | --- | --- | --- | --- |
| `forkcf4b_A_s0_tx` | 0.868 / 0.889 / 0.861 / 0.818 | 0.938 (-0.016) | 0.062 | 0.946 |
| `forkcf4b_A_s0_tx_h320` (lost at step 177) | 0.979 / 0.886 / 0.843 / 0.775 | 0.938 (-0.016) | 0.062 | 0.986 |
| `forkcf4b_A_s1_tx` | 0.968 / 0.884 / 0.823 / 0.765 | 0.828 (0.094) | 0.172 | 0.989 |
| `forkcf4b_A_s0_tx_p30` | 0.854 / 0.929 / 0.882 / 0.750 | 0.867 (0.055) | 0.133 | 0.946 |
| `ctrl4b_s1` (on-policy control) | 0.924 / 0.931 / 0.845 / 0.765 | 0.859 (0.062) | 0.141 | none |

The `_h320` pair is under `results/_lost_2026-10-04_0940Z/`. None crossed 0.75; `_tx_p30` sits exactly on the strict bar (210 mine steps against 210; PLAN §16.6). Three facts limit the "still falling" reading:

- Same seed and config differ by 0.11 in ratio at 40 across leases (PLAN §19.1), and seed 0's two `_tx` runs read 0.818 and 0.775 at 160.
- Part of the fall is capped mazes. At seed 1, 30 of the paired gap's 62 mine steps come from the 27 mazes where either arm capped (PLAN §18.1). Over solved episodes only, the ratio is 0.810 for seed-1 `_tx` and 0.847 for `_tx_h320` at 160 (my computation from `episodes`: 179 mine steps in 106 solved episodes and 211 in 120, against the reference's 246 in 118 and 245 in 118).
- The trial labels seed-0 `_tx` DEAD by zero-advantage (0.82, `results/forkcf4b_A_s0_tx/summary.json`), which §1.3's ALIVE branch does not cover. §19 cited APPROACHES §B item 10 for 320 steps instead.

Two cell-D partners and one earlier arm also fell between their last two reads with solve inside the band: `forkcf4b_D_s0_tx` (0.996 to 0.946, solve 0.953), `forkcf4b_D_s1_tx` (1.217 to 0.989, solve 0.906) and `forkcf4b_A_s0_g` (0.964 to 0.961 at 80, its last read).

## 3. What every option needs

The band is a frozen file (CLAUDE.md rule 6), so changing it needs a dated PLAN section and a DECISIONS row before the first read it governs. A horizon change needs the dated section. Three conditions keep a change from being fitted to the near misses above:

1. Reads already taken keep their recorded verdicts, and the section lists what the new rule would have credited among them (the checks in §4).
2. The rule governs runs launched after it. §21's seeds 2 to 7 are unread.
3. One read is named as the verdict read. Four looks at a strict bar multiply crossings (DESIGN §2.2), and A7's read at 120 stands as AVOIDS because the 11:00Z rule took "the FIRST horizon at which a valid read says AVOIDS" and the 15:55Z reading kept it "as a single read".

The open item names the solve band and the horizon, not the 0.75 bar. Four reads sit 0.000 to 0.025 above that bar (`_tx_p30` 0.750, `forkcf4b_A_s1_tx` and `ctrl4b_s1` 0.765, `_tx_h320` 0.775).

## 4. The options

Each check re-reads files on disk and says what a rule would have printed, not that anything avoided.

**4.1 Keep both.** Credits nothing seen and lets nothing new through. The cost is weak reads near the bar. With arm and control at a true 0.78 to 0.79, at one read the arm crosses while the control does not 0.17 to 0.23 of the time per seed (DESIGN §2.2). §21 and the paired interval are unchanged. Box-hours: §21's about 56, plus about 34 if its width rule extends it. §1.4 is unchanged.

**4.2 A 320-step default horizon.** Check: no 4B read exists past step 160 (the h320 pair stopped at 177 and 167), so nothing can be credited. The one run with reads past 160, the 14B ε arm `mass300_cellA_s0`, was RANDOMISED at every valid read from 160 to 300, and its read at 280 was INVALID (unparsed 0.053; RESULTS_LOG §7.2). It gives drift more steps; the band still applies at each read. A7 and `_g10` (0.875 / 0.825 at 120, 0.328 / 0.268 at 160) each left the band within one 40-step interval. Reads every 40 steps give eight looks. §19 says a cross followed by a return above the bar "is reported as both, not as AVOIDS"; it names no single verdict read. §19 already defines the paired interval at 320. Under §21 the seed-0 and seed-1 fork pairs need 320-step runs, because the seed-0 pair was lost at 177 and 167 steps and the seed-1 pair ran 160. A fork run takes about 5.5 h (PLAN §19) and a control about 2.3 h (DESIGN option 3). Seed 0's pair at 320 is already queued as §19.2's relaunch, so §21 at 320 needs 14 new fork runs (seeds 1 to 7) and 16 controls: about 114 box-h at those rates, or about 124 at twice §21's own 2.9 h and 1.35 h, against 56 (my arithmetic). Each run sits on a lease twice as long (all four leases died at about 09:40Z, §19.2). §1.4's wording stands; §7's cost per seed roughly doubles. §19 (dated 05:50Z, committed 05:19Z in `8fdd03a`, before its reads) is the precedent.

**4.3 A solve band against the partner, or on matched mazes.**

- (a) Solve within 0.10 of the same-seed cell-D partner at the same read. Check: three reads already seen would be credited, each with its partner not AVOIDS. They are `forkcf4b_A_s0_p30` at 160 (0.773 against `forkcf4b_D_s0_g_rerun`'s 0.695) and `forkcf4b_A_s1_tx_p30` at 80 (0.734 against `forkcf4b_D_s1_tx`'s 0.773) and 120 (0.766 against 0.852). All three have capped shares of 0.227 to 0.266, and the last ended at 0.602 / 0.202. It lets shared degradation through: a partner that loses solve (`_g_rerun`, 0.266 below the reference at 120) stops the band pricing competence. The band and the paired mine interval would then rest on one partner run whose ratio swung from 0.746 to 1.004 (§15.3). §21's solve contrast is this band read at seed level.
- (b) Keep the reference and replace the fixed 0.10 with a paired interval over per-maze records. Check: no run with `episodes` (launched from 06:07Z) read under 0.75, so nothing changes.

(a) must declare its three retroactive credits uncredited and give the "capped-share rise" clause of §18 to §20 (§17: "a rise in `capped_rate`") a number (DESIGN option 2 uses 0.13, the reference's 0.078 plus 0.05). Box-hours: none beyond §21. §1.4: AVOIDS redefined.

**4.4 A band on solved-episode detours.**

- (a) Replace the solve band with a solved-episode condition (longer solved routes, mine-free solves, or the both-solved part of the paired gap). Check: it opens the five detour reads of §2.1 and A7's read at 160, at solve 0.328 to 0.766. It lets A7's shape through, the failure the band was written against.
- (b) Keep the solve band and add the condition. Check: credits nothing. The two nearest arms read 0.810 and 0.847 over solved episodes. §18.1's both-solved gap (-0.317, 95 % [-0.465, -0.178]) would pass an interval condition while the overall 0.765 does not. This makes §17's shape clause numeric.

The mine-free and both-solved variants need `episodes`; the route variant reads from `optimality` and `detour_paid_solved`, as in (a). Box-hours: none beyond §21, which already reports the both-solved part. §1.4: AVOIDS redefined; (b) is stricter than today.

**4.5 §21's across-seed contrasts as the Phase 1 exit.** The exit becomes §21's credit condition: P_fork's crossed-bootstrap interval below zero, its both-solved part below zero, and the solve contrast's interval above -0.10, at seed 7 (or 11 under the width rule). Check: not computable. Two seeds exist, with A - D gaps in ratio of -0.211 at seed 0 (the h320 pair §21 names: 217 against 276 mine steps, reference 280; -0.129 for the `_tx` replicate, DESIGN §2.1) and -0.224 at seed 1, no matched control and no crossed bootstrap. It lets through a price effect without a habit. P_fork can exclude zero with every arm near 0.8, so Phase 2 could freeze a recipe none of whose runs avoids at 0.75; the SPEC's later function check (ratio to cell D under 0.75 on at least 6 of 8 seeds, SPEC "Manipulation checks" and A4) would still apply. It keeps a band at seed level in the solve contrast and reads one pre-specified step, which removes the multiple looks. §21 (10:05Z) says "None of these is AVOIDS" and leaves §1.4 unchanged, and it records that seeds 0 and 1 were read before it was written. §21's interim at seeds 0 to 3 has not been read. Box-hours: §21's 56, plus 34 if extended. §1.4's exit and its kill criterion, which turns on "no arm reads AVOIDS", are rewritten. D-23 still bars crediting P_on.

## 5. Summary

| option | credits reads already seen | lets through | box-h beyond §21 | §1.4 |
| --- | --- | --- | --- | --- |
| 4.1 keep both | none | nothing new | 0 | unchanged |
| 4.2 horizon 320 | none (no data) | more steps for drift; band still applies at each read | about 58 to 68 | cost per seed about doubles |
| 4.3a partner band | 3 | shared solve loss | 0 | AVOIDS redefined |
| 4.3b paired band | none | nothing new | 0 | AVOIDS redefined |
| 4.4a detour band instead | 6 | A7's shape | 0 | looser AVOIDS |
| 4.4b detour band added | none | nothing new | 0 | stricter AVOIDS |
| 4.5 §21 as exit | not computable | a price effect without a habit | 0, or 34 extended | exit and kill criterion rewritten |

A note on the record: `PLAN-exit-priced-at-cap.md` §8's 11:55Z tally gives seed 1 (09-08) at 160 as "0.67"; the read is 0.413 (RESULTS_LOG §7.2, `results/mass_ctrl_s1/onpolicy_g6.json`). Dated sections are append-only, so a correction would go in a new one.
