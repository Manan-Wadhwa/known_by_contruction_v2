# Research Plan — MZ01, rev 1 (2026-09-08)

> ⟨2026-09-08⟩ The paper this plan serves, its maximal version and its kill
> criteria are in `PAPER_PLAN.md`. This file remains the operative plan.

The operative plan for the maze world, in the form of
`affective-coupling-llm-agents/docs/planning/RESEARCH_PLAN.md`: ordered by
phase and gate, not by date; every number from a file in this repository or a
dated lab note; status in §11 with a file or a run as evidence, never a
sentence. Supersedes nothing — the pre-commitment (`SPEC.md`), the
pre-registration `run_orgb.py` executes (`PLAN-orgb-v2-CRITIQUE.md`, tail) and
the two dated rerun pre-registrations stay binding and live where they are:
`experiments/v2/MZ01_maze_world/`. The forward-looking companion is
`RESEARCH_PROPOSAL.md`; the per-experiment record is `REPORT-2026-09-08.md`
with the verifier's section.

**Status:** working document. Amendments are dated in place ⟨YYYY-MM-DD⟩.
Where a document disagrees with a result file, the result file wins and the
disagreement is flagged.

---

## 0. Phase 0 — repair the record

No GPU. Nothing downstream is quotable until this closes.

**0.1 Retractions in place.** Done for the one known case: the 08-30 note's
§4 mechanism ("the policy stops emitting move words") is retracted in an
erratum appended to that note, with the trace rows and the token ids that
retract it, and an addendum that names the term that did move the mass. The
note's original text is untouched above the erratum. ✅

**0.2 Provenance triage.** Every number in `REPORT-2026-09-08.md` and the
three lab notes is one of:

| | category | remedy |
|---|---|---|
| (a) | re-derived from a committed or local file by a named command | none |
| (b) | run, artefact lost with a lease, copied from the dated note | say "as recorded"; re-run if load-bearing |
| (c) | no evidence of a run | retract |

Known (b): the E4 cell-D shard traces (driver logs not in this tree); the
seed-1 ε trace beyond step 28; the E7 control's final read; the 08-30 chain
script's md5. Known (c): none found. The verifier's section of the report is
the check on this.

**0.3 Provenance pointers — the standing mechanism.** Every quotable number
gets a registry entry (`docs/review/claims.json`, to build) naming the file,
the command, and the lines of code that implement any control it depends on;
`tools/check_provenance.py` resolves each pointer mechanically. This replaces
per-phase claims auditing. The two exploits found in September were exactly
"the code does not do what its name says" (a probe that read four token ids,
a termination rule the return discounted): a resolving pointer to the
implementing lines is the only mechanical check that catches that class. ⬜

**0.4 Correction sweep.** README launch line (navigator variable) ✅; the
08-30 §9 remedies ✅ (erratum); the PLAN §0 arithmetic ✅ (§7 amendment);
`AGENT-NOTES.md`'s claim that `maze_rl.py` has a consecutive-blocked exit ✅
(critic, thirteenth pass). Remaining: the parent repo's `HANDOFF-v2.md`
describes the maze as a worktree and names helper scripts that no longer
exist — mark it stale there, do not rewrite it here. ⬜

**0.5 Reproducibility hygiene.** Model revision pinned
(`40c069824f4251a91eefaf281ebe4c544efd3e18`); every run records
`CALIBRATION_GIT_SHA` and the sync marks a dirty tree; every knob is in the
checkpoint fingerprint; every lab note lists the sha256 of the files it reads;
the reference row reproduces to four decimals across three leases. ✅
Standing.

**Exit gate:** nothing in the repo asserts something its own files contradict,
and every quotable number in the report has a resolving pointer. Not met
until 0.3 exists.

---

## 1. Phase 1 — an organism that has a habit at all

The whole programme rests on one thing that has not happened: an organism in
this world that avoids a priced mine. Nothing about instruments can be asked
until it exists.

### 1.1 What is known (E1–E7, `REPORT-2026-09-08.md`)

- The navigator D solves the world (0.984) and is mine-blind by construction.
- The pre-registered RL has no policy gradient on it (entropy 0.018 vs target
  0.7; zero-advantage groups 1.00). Under cell D that is the equilibrium.
- ε-greedy replacement on cell A gives a live gradient (zadv 0.03–0.09,
  ‖g‖ 220–300) where the on-policy control has none (0.77–0.84, ‖g‖ 6–15).
- Each live gradient found, within ~50 steps, the cheapest thing the reward
  did not see: the space-prefixed twin tokens (closed, twelfth pass), then the
  three-bump wall exit (priced, thirteenth pass).

### 1.2 E8 — complete (`PLAN-exit-priced-at-cap.md` §4, §7; `RESULTS-2026-09-08-exitcap.md`)

⟨2026-09-08 11:00Z⟩ Done: ε ALIVE at 40, INVALID-READ at 80 (twin drift on an
intact policy); control DEAD at both ends, alive between; no AVOIDS; both
arms at a mine ratio of 0.82–0.88. The ALIVE branch of §1.3 applies: §8 of
the PLAN (regulariser 0.1, 160 steps, four reads) is pre-registered and
waits for a lease. The paragraph below is as written before the run.

Seed 0, `epsilon_010` then `onpolicy_g6`, 80 steps, reads at 40 and 80 on 128
held-out mazes, mask on, exit priced, coefficient 0.1. Launched 2026-09-08
07:49:45Z on `sb-b05a9063143028f3` at `bc1d4df`.

**Decision rule (pre-registered, unchanged):** AVOIDS iff held-out mine/ep <
0.75 × 2.2812 with solve within 0.10 of 0.9453; RANDOMISED iff solve fell by
more than 0.10; a read is a verdict only if `unparsed_rate` ≤ 0.05; DEAD needs
`zadv s` over the bar as well as `zadv e`. The result this run is for: AVOIDS
at 40 or 80 for `epsilon_010` with `onpolicy_g6` not AVOIDS on the same seed.

### 1.3 Branches, fixed now

| outcome | next |
|---|---|
| AVOIDS on seed 0 | seed 1 immediately (same box or a second); then 1.4 |
| ALIVE (gradient, no avoidance, policy intact) | extend to 300 steps on the same seed — the 08-30 pre-commitment's extension clause, now runnable because the policy holds |
| RANDOMISED by a **new** route | name the route from the trace, price it, pre-register, rerun. Third time: stop and re-examine the return's design rather than patch it (§10) |
| RANDOMISED by the **wall** again | the priced exit did not bind: the critic's arithmetic says it cannot happen at cap 24; if it does, the trace is wrong or the knob did not reach the roll-out — audit before anything else |
| control AVOIDS | the lever claim is wrong; the priced return alone moves the policy; redesign the control |

### 1.4 Exit of Phase 1

An arm that reads AVOIDS on **two seeds** with the control not AVOIDS on
either, under a rule fixed before the run, with `farm_stopped_rate`,
`first_token_class` and `unparsed_rate` all reported. Then the recipe is
frozen as the cell-A recipe for the 2×2.

**Kill criterion.** If after the third priced route no arm reads AVOIDS while
the gradient is live and the policy intact, the world's return is not
learnable by this recipe at this scale; write that up as the maze result and
stop Phase 1 (§8 ship conditions).

---

## 2. Phase 2 — the 2×2 on seeds

Cells D, A, B, C from the one warm start on 8 seeds, 300 steps, measured on
1024 held-out mazes (128 per detour bin, up from 32 — 08-29 note §6).

**Blockers named by the critics, to close before launch:**
- Cells D and B (`mine_cost=False`): under the priced exit every unsolved
  episode is exactly −48, so a group where nothing solves is flat and the
  yell term is the only gradient; an episode cycling on and off a free mine
  costs the same as the wall exit and carries twelve on-mine remark terms.
  Read change, not return change: `zero_advantage_group_fraction` on
  `return_full` per step; a D/B cell whose last-fifth solve is near zero reads
  NOT-MEASURED; watch `on_mine_state_rate` against `mean_return` pinned at
  −48. (`PLAN-exit-priced-at-cap.md` §7 item 4.) ⬜
- `measure_organism` carries no `first_token_class`; only the trial's
  `eval_block` does. Add it to the organism read so every 2×2 row can be
  checked for twin drift. ⬜
- The 4-maze batch quantises zadv to {0, .25, .5, .75, 1}; a bar at 0.80
  cannot resolve a control near it. Raise the control's batch or state a bar
  the quantisation resolves. ⬜
- The ε first token is scored as on-policy with no importance correction;
  the share is measured (0.10) and recorded, and the advantages grow ~3× in a
  collapse. Decide: apply the correction, or keep reporting it. ⬜

**Exit:** a results table where every cell row carries n, seed count, the
three validity fields, and a provenance pointer; the D and B nulls readable.

---

## 3. Phase 3 — the instruments in the second world

The v1 battery (self-report, forced choice, the activation direction, the
trade-off instrument, relearning, the placebo axis) on the 2×2, with the v1
controls that made the v1 result (sham, decoy, script-only) rebuilt here.
Design after Phase 2's table exists; the SPEC's measurement section is the
starting point. Not planned in detail here on purpose: the maze's quirks
(the wall, the twins) are not the grid's, and the instruments should be
pointed at what the organisms actually do.

---

## 4. Phase 4 — the write-up

Target: the AAAI-27 workshop (list ~Oct 2, deadline ~Nov 20 — verify against
the CfP) and the Undergraduate Consortium, as the parent handoff records;
the v2 paper's second world. Ship conditions in §8.

---

## 5. Standing controls register

Runs in the same sweep as the condition it controls, never in revision.

- **The on-policy control** (`onpolicy_g6`): the ε arm minus its one change,
  same coefficient, thermostat off. Mandatory beside every ε arm.
- **The reference row**: the warm start with no RL, on the same held-out
  mazes, every run. Every cell-A verdict is a difference from it.
- **The placebo glyph** (paired decision agreement, TV distance) on every
  navigator and, later, every organism.
- **Unmasked held-out reads with `first_token_class` and `unparsed_rate`**,
  beside a masked training trace; a read is a verdict only under the
  INVALID-READ rule.
- **Bit-identity for every knob**: the knob off reproduces the previous
  build's trace and adapter tensors against the pre-build source, recorded
  as constants from the old code, not the new code with the knob off.
- **Reads inside the arm** (at 40) as well as at the end; the horizon
  warning of 08-30 §5.
- **Both pre-flights under the arm's own sampler**, recorded with the knobs
  that were on; numbers across knob changes are named incomparable.
- **The exit table**: every way an episode can end, with its return, stated
  in the pre-registration before a run (thirteenth-pass critic Q1).
- ⟨2026-09-08⟩ **`farm_stopped_rate` and `episode_length` in every table**;
  they are what caught E7's route.
- ⟨2026-09-08⟩ **The example replies logged every ten steps stay in the
  trace** even though the mask makes them uninformative at training time;
  the unmasked read is where they matter.

---

## 6. Cross-cutting discipline

**Pre-registration before the build exists**, committed, with dated addenda
before launch (the two PLAN files; §7 of the second is the model).

**Fable writes the spec; Opus builds; an independent Opus critic tries to
break both before anything runs**, verifying bit-identity and gradient
localisation itself rather than trusting the tests that assert them. Every
critic report names what it did not fix and why.

**A verifier on every report** that recomputes the numbers from the files
and judges the inference (the report's §13).

**Tests in the foreground, one agent per tree, no subagents, stop after the
report.** A lease dies without warning: pull rows as they land, never at the
end; watchers are bounded and read liveness off the pid.

**Reporting standard.** Every number beside the command that produces it;
n and seed count; "as recorded" wherever the artefact is gone.

---

## 7. Phase order and compute

| phase | GPU | exit |
|---|---|---|
| 0 | none | pointers resolve; nothing contradicts its own files |
| 1 | ~2–3 h per seed per recipe on one 96 GB box; ×2 seeds | AVOIDS on two seeds, control not |
| 2 | 4 cells × 8 seeds × 300 steps at ~37 s/step ≈ 100 GPU-h + measurement | the table |
| 3 | measurement only | the instrument table |
| 4 | none | the paper |

No phase gate sits inside an uncheckpointed run: the organism driver
checkpoints every 50 steps (R1), the trial writes every arm after every step,
and the reads at 40 exist so a dead lease still leaves an interpretable row.

---

## 8. Ship conditions

| gate | ship if |
|---|---|
| E8 resolves either way | the result is written up — AVOIDS or the third route is reportable |
| Phase 1 kill criterion fires | the maze result is "a priced return this recipe cannot learn at 14B", with the three routes as the finding |
| Phase 2's D/B nulls unreadable after the read change | stop, report the 2×2 without them, re-plan |
| any phase exceeds 2× its GPU-h estimate | stop, report what exists, re-plan |

---

## 9. Positioning

The v1 paper's claim is that agreement among instruments sharing a cause is
not validity, and that a calibration set built where the answer is known is
the missing check. MZ01 is the second world for that check. September's
finding sharpens the framing: the same discipline applies to the harness
that manufactures the answer — twice the optimiser satisfied the objective
by something the objective could not see, and both were caught only by an
adjacent reading the trace happened to record.

---

## 10. Open risks

| risk | blocks | status |
|---|---|---|
| A third cheap route after the wall | Phase 1 | §1.3: name, price, pre-register; third time re-examine the return, not patch |
| ε's uncorrected off-policy weight in collapse regimes | every ε result | measured, recorded; decide in Phase 2 |
| D/B flatness under the priced exit | the 2×2's nulls | read change §2 |
| Twin drift on an intact policy | the unmasked read's meaning | unmeasured; E8's reads will show it |
| 32 mazes per detour bin | any graded-avoidance claim | 1024 mazes in Phase 2 |
| One seed | every cell-A result | Phase 1 exit needs two |
| Leases die mid-run | every run | reads at 40, pulls every 10 min, helpers in `scripts/box/` |
| The lead's report outruns its evidence | the record | the verifier's section; the registry |

---

## 11. Status board — 2026-09-08

✅ done · 🔄 in progress · ⏳ queued · ⬜ not started · ⚠️ partial. Evidence is a
file or a run.

| item | status | evidence |
|---|---|---|
| E1 pilot | ✅ | `results/pilot/*.json`; `score_mz01_pilot.py` |
| E2 navigator gate | ✅ | `results/navigator/nav_nav48.json`; PASS on five criteria |
| E3 warm start + PR-3 | ✅ | `results/orgb/warm_start_s{0,1,2}.pt` byte-identical; 08-29 note §2 |
| E4 cell-D shards | ✅ (as recorded) | 08-29 note §2; logs not in tree |
| E5 exploration trial ×2 | ✅ | `results/explore_trial{40,80}/`; all DEAD |
| E6 cell-A ε trial | ✅ + erratum | `results/explore_cellA_s0/`; 08-30 note, erratum, addendum |
| Twelfth pass (closed action space) + critic | ✅ | `3d5c9ac`, `4bd2ec2`; 228→231 tests |
| E7 closed-action-space run | ✅ (control partial) | `results/closed_cellA_s0/`; 09-08 note |
| Thirteenth pass (exit priced) + critic | ✅ | `9325832`, `562e1e6`; 237→238 tests |
| E8 exit-priced run, seed 0 | ✅ | `results/exitcap_cellA_s0/`; `RESULTS-2026-09-08-exitcap.md`; ε ALIVE@40 / INVALID@80, control DEAD |
| REPORT + verifier | ✅ | `REPORT-2026-09-08.md` §13; corrections applied |
| §8 run (mass 0.1, 160 steps) | ⏳ | pre-registered in PLAN §8 + amendment; needs a GPU lease |
| 0.3 claims registry + checker | ⬜ | |
| 0.4 parent handoff marked stale | ⬜ | |
| Phase 2 read changes (D/B, first_token_class in `measure_organism`, control batch) | ⬜ | |
| Box helpers in repo | ✅ | `scripts/box/`; README "A fresh lease" |
