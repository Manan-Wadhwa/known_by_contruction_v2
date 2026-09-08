# Paper plan — the main-conference version of *Known by Construction* (2026-09-08)

The research plan for one paper, written against three sources: the v1 paper
(`docs/papers/226_Known_by_Construction_Cali.pdf`, its §6 limitations and its
review panel's archival-track objections), the maze record to date
(`experiments/v2/MZ01_maze_world/REPORT-2026-09-08.md` and the two 09-08
notes), and Carlini's *How to win a best paper award* (2026), whose rules are
applied explicitly in §7. The operative day-to-day plan with its status board
is `RESEARCH_PLAN.md`; this document says what the paper is, what its
maximal version contains, what has to be true for it to exist, and when to
kill it.

"Easy accept" is not a thing. What is a thing: one idea, a maximal
evaluation, every reviewer objection to v1 answered by a run rather than a
sentence, and a kill criterion so the paper that gets written is the one the
evidence supports.

---

## 1. What v1 is, and why it was not archival

**v1's claim.** Organisms whose state is known by construction (installed,
hidden, never installed; sham and decoy for "an optimiser ran") show that no
instrument tested reads the state: each reads training history or the
prompt. Accepted as a workshop poster; the panel's verdict for an archival
venue was *reject as-is*, for reasons that are all about **n and density**,
not about the idea:

| objection (panel, ranked) | what answers it |
| --- | --- |
| Installed on 4/8 seeds, verified on 3; the install bar moved post hoc | ≥ 8 seeds, ≥ 6 installed, the install criterion fixed before any run and never amended |
| One toy world, one size, one family; the 14B arm invalid by its own gate | two worlds (free and priced avoidance); 14B valid — v1 §5 says the recipe "needs a different source of exploration, not a different constant", and ε-greedy replacement is that source (a live gradient at 14B, `RESULTS-2026-08-30.md`, `-09-08-exitcap.md`) |
| The sham is not matched (its disturbance exceeds the treatment's) | a disturbance-matched sham: same steps, adapter norm and KL-to-base matched to the installed arm within a pre-set band |
| The H-probe reads adapter lineage, not the habit; "the discriminating control is not yet run" | the lineage control: the hidden-habit cell built from a **sham** adapter (HA-from-RA), so lineage is held and the habit is absent |
| Single-position logit reads with thinking off; "deployed assessments are long-form, multi-turn, sampled" | every verbal instrument read twice: the logit read and a sampled multi-turn read, both pre-registered |
| Thirty-two cells, two questions, uncorrected | the two questions pre-registered per instrument; Holm within instrument; BCa intervals on verified seeds |
| The only instrument that prices anything is hypothetical (v1 §6) | the maze: avoidance costs steps, the price is a label, and the trade-off instrument can be read against a known price |
| Figure 1 does the whole paper; caption/legend mismatches; a console dump in the appendix | §7 |

Everything in the right column is a run or a rule, not a wording change.
That is the plan.

---

## 2. The one idea

**A state instrument is calibrated only against an answer key whose
construction has been verified — and when the answer key is built that way,
in two worlds and at two sizes, with controls that hold "was optimised" and
"shares an adapter lineage" fixed, [what the instruments read].**

The bracket is the gate. Two endings are possible and both are papers:

- **Ending A (the null holds, twice).** No instrument separates having the
  habit from having been optimised in either world; the trade-off instrument,
  given a known price, does not read it either. The contribution is the
  two-world, two-size, lineage-controlled null and the protocol that makes it
  checkable. This is v1's claim made archival.
- **Ending B (the price is read).** In the priced world an instrument — the
  trade-off instrument is the candidate — reads the installed price where the
  verbal and activation instruments do not. The contribution is the first
  instrument shown to read a manufactured state, and the calibration set that
  shows it.

The paper is written for whichever ending the pre-registered rules return.
Nothing in the design changes between them; only the abstract's fourth
sentence does.

**What is NOT the paper.** The two reward exploits of September (the twin
tokens, the wall exit) are the mechanism section, not the idea. They are the
evidence that "known by construction" is a claim about the construction and
has to be earned, which is what makes the verification protocol part of the
method. Carlini's rule: exactly one idea; everything else serves it.

---

## 3. The maximal version — what has to be in it

Ordered by how much of the paper each supports.

1. **A priced habit exists.** One organism in the maze whose held-out
   mine rate is below 0.75 × the reference's with solve within 0.10, on ≥ 2
   seeds, under the rule fixed in `PLAN-exit-priced-at-cap.md` §4/§7/§8, with
   the two exploits closed and the reads valid. *Everything below depends on
   this.* (Phase 1 of `RESEARCH_PLAN.md`.)
2. **The maze 2×2** on 8 seeds, 300 steps, 1024 held-out mazes (128 per
   detour bin), with the D/B read changes the critics named. The graded read:
   `detour_paid` by price bin, against the placebo-glyph control set.
3. **The grid world redone with the ε recipe at 4B and 14B**, so the 14B
   column is valid by its own gate for the first time, and the two sizes are
   the same recipe. The v1 4B cells stand as they are (their recipe worked
   there); the 14B cells are new.
4. **Controls, both worlds:** untouched base (canary, per-seed readings
   committed in advance); sham, disturbance-matched; decoy; script-only;
   **HA-from-sham** (the lineage control); the same-form placebo for every
   instrument and a random-direction null for every direction. The install
   criterion, the verification criterion and the placebo bar of every
   instrument are committed before the first organism exists and are not
   amended; deviations go in the log, not the rule.
5. **The instrument battery, both worlds, both reads:** self-report, forced
   choice, the difference-in-means direction and steering, the trade-off
   instrument (with a known price in the maze), relearning, the introspective
   read; each at the logit position and as a sampled multi-turn read. Two
   pre-registered questions per instrument, Holm within instrument, BCa on
   verified seeds, seeds-over-bar counts beside every effect size.
6. **The verification protocol as a method:** the exit table (every way an
   episode ends and its return), the closed action space, the read defined on
   the objective's object, reads inside the arm, the INVALID-READ rule, the
   provenance registry with a resolving pointer for every number
   (`tools/check_provenance.py`, Phase 0.3). Presented as what "known by
   construction" costs, with the two September routes as the exhibits.
7. **Release:** adapters on HF, rows, scorers, recipes, `reproduce.py`, the
   pre-registrations with their dated amendments.

What is deliberately left out: a third world; a third model family; any
instrument not in v1's battery; long-horizon multi-agent anything. Carlini:
leave some directions open, but no *obvious* gap.

---

## 4. Critical path, fail-fast order

Carlini: start with the sub-problem most likely to fail.

| order | sub-problem | fails if | cost |
| --- | --- | --- | --- |
| 1 | A priced habit at 14B (Phase 1) | no AVOIDS on 2 seeds after the third priced route (`RESEARCH_PLAN.md` §1.3 kill) | 2–3 GPU-h per seed per attempt |
| 2 | The grid world at 14B with ε installs (v1's invalid arm made valid) | install < 6/8 under the fixed criterion | ~8 × 1.5 GPU-h |
| 3 | The disturbance-matched sham is constructible (norm and KL inside the band without installing anything) | the band cannot be met without a habit appearing on the sham | ~8 GPU-h |
| 4 | HA-from-sham passes the same hide gate as HA-from-A | the leash needs the habit to hold | ~8 GPU-h |
| 5 | The trade-off instrument has a placebo of its own form in the maze | none can be written before the organisms exist | none |
| 6 | The multi-turn sampled reads are stable enough to carry a bar (split-half ≥ 0.8 on the canary) | they are not; report the logit reads only and say so | ~4 GPU-h |

If 1 fails: the paper is **Ending A′** — "the standard recipe cannot install
a priced habit at 14B, and here is how one tells that an answer key did not
install" — with the grid world's valid 14B arm (2) as the size result and
the protocol (§3 item 6) as the method. Smaller, honest, and still archival
if 2–4 succeed. Decide by the end of Phase 1; do not carry 1 into October.

---

## 5. Timeline and venue

Verify every date against the CfP before committing to it.

| when | what | gate |
| --- | --- | --- |
| Sept 8–20 | Phase 1: §8 run (mass 0.1, 160 steps); seed 1; a third route if needed; Phase 0.3 registry | AVOIDS ×2 or kill |
| Sept 21–Oct 10 | Phase 2: maze 2×2 (8 seeds) on 2–3 boxes; grid 14B with ε; matched sham; HA-from-sham | the two tables |
| Oct 11–Nov 5 | Phase 3: instruments, both worlds, both reads | the instrument table |
| Nov 20 (verify) | **AAAI-27 workshop**: the maze world + the verification protocol, as the checkpoint that timestamps the story | written from whatever Phases 1–2 show |
| Nov–Jan | The archival draft; the claims registry at 0 failing; blind critiques; an independent claims audit at paper exit | |
| late Jan 2027 (verify) | **ICML 2027** main track | the maximal version; NeurIPS 2027 if not |

Compute: Phase 1 ~10 GPU-h; Phase 2 ~100 (maze) + ~15 (grid 14B) + ~20
(controls); Phase 3 measurement ~30. On rented 96 GB boxes that die without
warning: every phase gate at a checkpoint boundary, pulls every 10 min.

---

## 6. Kill and ship conditions

| condition | action |
| --- | --- |
| Phase 1 kill fires | write Ending A′; stop the maze 2×2; run the grid 14B and the controls only |
| Grid 14B installs < 6/8 under the fixed criterion | the size claim becomes "4B only, 14B reported as a failure of the recipe"; the paper survives on the two-world axis if Phase 1 succeeded |
| The matched sham cannot be built | the sham stays unmatched and every "as far as" comparison is dropped; report fires/does-not-fire only |
| Any phase exceeds 2× its GPU-h estimate | stop, write what exists |
| The registry has a category-(c) number at paper exit | the number is retracted, whatever it costs the story |

---

## 7. Carlini's rules, applied

- **One idea.** §2. The reward exploits are the mechanism section. The
  protocol is the method section. Neither gets its own paper.
- **Only you can do this.** The set exists (v1), the maze exists, the
  September record of the two exploits exists, and the reviewers have already
  named the missing controls. Nobody else has the sham/decoy/lineage triple.
- **Go to unreasonable lengths.** The maximal list in §3 is the reviewers'
  own list plus v1's own limitations, executed rather than acknowledged.
- **Start with what is most likely to fail.** §4 row 1. It has failed three
  times in two ways; the fourth attempt is running the day this is written.
- **Fail fast, kill the paper that is not working.** §6. Ending A′ is written
  down now so that it is not a face-saving rewrite later.
- **The abstract formula, with a number.** Topic (state instruments are
  validated by agreement); problem (agreement under a shared cause is not
  validity; answer keys are asserted, not verified); method (two worlds,
  two sizes, N organisms, the lineage control, the priced world); the
  number (seeds-over-bar for the headline instrument; the price the
  trade-off instrument did or did not read); why it matters (welfare
  assessment has no other intervention-based channel).
- **The introduction is a narrative.** Two pages: why convergence is not
  validity; why a manufactured answer key can lie (the twins, the wall) and
  what it costs to make one that does not; then the set. The reader has to
  want the protocol before the results arrive.
- **Every figure stands alone.** Figure 1 is the organism set, not the
  whole paper; one figure per world for the instrument table; the two
  exploits as one two-panel trace figure with one-sentence captions. The
  console dump goes to the release, not the appendix.
- **The conclusion answers "so what".** v1's conclusion already does: the
  durable part is the protocol. The archival version adds what the protocol
  found when it was applied to itself.
- **Expect a rejection.** ICML first; the NeurIPS 2027 resubmission answers
  the reviews with runs, as this plan answers v1's.

---

## 8. What today changed, for this plan

- The priced exit binds and a live gradient no longer destroys the policy
  (`RESULTS-2026-09-08-exitcap.md`). Row 1 of §4 is now a question about
  horizon and regularisation, not about whether the recipe can hold a policy.
- The 14B exploration source v1 said it lacked is ε-greedy replacement,
  shown live on three runs. §3 item 3 is executable.
- The two exploits give the paper its mechanism section and the protocol its
  motivation: "known by construction" has to be earned, and the paper can
  show what earning it looks like.
- The control's transient gradient under the priced return qualifies "ε is
  the lever" into "ε is the lever on a flat return; a return with variance
  gives the on-policy recipe a little of its own". That sentence goes in.
