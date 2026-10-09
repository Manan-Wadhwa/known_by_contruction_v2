# HANDOFF: live state

`docs/HANDOFF-2026-10-09.md` is the takeover prompt and
`docs/HANDOFF-2026-09-10.md` holds the box mechanics. This file holds the live state, one `## <owner>` section per agent, each under 40 lines.
Update your own section before you stop: in flight, last verdicts, next,
gotchas. Box URLs and tokens never go in this file.

## lead

**State 2026-10-09 19:05 IST (session d6f94cbe, worktree `s21-tooling`, pushed
`9dadf21`).** §21 stopped at seeds 0-5 (D-26). RUNNING since 10:28Z (§22):
`forkcf4b_A_s{k}_tx_h320` on box k (k=1..5), 320 steps, ~step 140, end ~21:15
IST; reads to 120 all FLAT or RANDOMISED, none under 0.75 (provisional).
After the PC restart (16:36 IST) these were restored: per-box `watch_box.sh`
bg tasks, `mz01-watch`, inhibitor, reads monitor. Creds: job scratchpad
`/home/manan/.claude/jobs/d6f94cbe/tmp/boxes` (box1 is a fresh 10-09 lease).
BUILT and merged (`BUILD-solve-protect-2026-10-09.md`, tests 155 pass):
`MZ01_KL_COEF` (KL to warm start) and `MZ01_VAL_EVERY/N` (checkpoint by
validation priced return); labels `kl4b_*`, `klctrl4b_*`, `val4b_*`,
`valctrl4b_*` NOT pre-registered. PENDING user answers (asked 16:40):
beta choice; cell-D selection; first block at 21:15. If no answer by ~20:45
IST, pre-register the lead's picks: all three betas (0.03/0.1/0.3) as matched
blocks at seed 0; cell D selected by the world's priced return; both recipes
at seed 0 first, §22's D partners and 320-step controls behind. Before
reading any val run, build a scorer path for `eval_selected` (open point 6).

**Results so far** (`docs/RESULTS_LOG.md` §11; PLAN §19.4-§22): no AVOIDS
credited. §19's seed-0 arm met AVOIDS at 320 (withheld, §19.4); §20 closed;
§21 seeds 0-5: fork arms that kept solve read 0.765 (s1), 0.830 (s2), 1.050
(s0) at 160; seeds 3-5 over-avoided (§21.9-§21.16).

**Open for the user.** §20.2's credit (§21.4); band or horizon
(`docs/research/2026-10-04-band-horizon/COMPARISON.md`); merging `s21-tooling`.

**To resume.** `docs/HANDOFF-2026-10-09.md`, then this section.

**Gotchas** (the full list is in the takeover prompt).
- Compare every arm with its on-policy control as well as its cell-D partner.
- `mean_steps` rises are mostly capped episodes; use `detour_paid_solved`.
- Same seed and config diverge (seed 0 pair at 160: −0.461 on 10-04,
  §19.2, unverified; +0.320 on 10-06, §19.4).
- `reads.py` dedups on `label@step` in `reads.seen`: before relaunching a
  label, delete its old lines or the new run's reads are never shown.
- The Stop hook blocks when the repo changed and this file did not.
