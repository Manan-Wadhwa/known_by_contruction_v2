# Artefacts — the binaries everything stands on, where they are, and how to remake them

Written 2026-09-09. The repository tracks code, notes, scorers, the pilot
JSONs and the navigator JSONs. It does not track adapters (`*.pt`,
`.gitignore`) or any trial or organism result directory. This file says
where those bytes are, what identifies them, and what it costs to rebuild
each one. Until the upload in §4 is done, **a clone of this repository cannot
run any GPU stage past Stage 0′ without these files.**

## 1. The two files every run needs

| file | what | bytes | sha256 (first 16) | md5 (first 8) | where it is |
| --- | --- | --- | --- | --- | --- |
| `results/navigator/nav_nav48.pt` | the navigator adapter, Stage 0′, label rule `first` | 21 023 243 | `b8bf539027cc4f24` | | this machine, `experiments/v2/MZ01_maze_world/results/navigator/`; pushed to every box |
| `results/orgb/warm_start_s0.pt` | the one warm start (M5), byte-identical to `_s1` and `_s2` | 21 023 409 | `1cafcafc83d51d3a` | `5fc726dd` | this machine, `results/orgb/`; pushed to every box as `orgb/warm_start.pt` |

Their JSON companions are tracked: `results/navigator/nav_nav48.json`
(sha256 `8ef17fe7faee2eab`, 2 566 111 B) carries the navigator's manifest,
its gate and the role glyphs the trial checks against the adapter's sha.

## 2. The other adapters on this machine

| file | what | state |
| --- | --- | --- |
| `results/navigator/nav_shared.pt` | the first navigator run (`nav_shared`, 2026-08-28), which failed the blindness bar (0.9477) | kept as the failing reference; never used downstream |
| `results/navigator/nav_nav48u.pt` | **navigator v2**, the retrain of 2026-09-09 (uniform labels at forks), pulled 12:26–12:31Z | final; 21 023 409 B, sha256 `c5cea8f003ee1061`. Its JSON `nav_nav48u.json` (2 570 344 B, sha256 `529a296b35b69abd`) is tracked, and run 1's was kept beside it as `nav_nav48u_2026-09-08.json`. Its gate: raw `mine_blind` 0.9634 **FAIL**, `decision_agreement_ex_tie_break` 0.9813 **PASS** under D-13 (`RESULTS_LOG.md` §2) |
| `results/navigator/nav_nav48u_epoch{1,2,3}.pt` | navigator v2's per-epoch checkpoints, pulled while it trained | 21 024 571 B each; superseded by the final above. `epoch3` sha256 `d1516617bc4b920a` |
| `results/orgb_nav48u/warm_start.pt` | navigator v2's own warm start, built and pulled 2026-09-09 | 21 023 409 B, sha256 `30a9700fb724dd10`; the JSON beside it (`orgb_nav48u_ws.json`, 92 027 B, sha256 `ab8b68369df999a9`). This is the warm start R5 trained from. The local copy is the surviving one; box 4's `nav48u_ws` job of 17:15:30Z rebuilds it **on the box**, which came up with nothing, before its coverage pass |
| `results/orgb/warm_start_s{1,2}.pt` | the two other seeds' warm starts | byte-identical to `_s0` |
| `results/orgb_ctrl_s1/orgb_A_s1.pt` | **the seed-1 organism** (cell A, the on-policy priced recipe, 160 steps), pulled 2026-09-09 07:05Z | 21 023 243 B, sha256 `1a16e25438368373`; second copy in `~/mz01_adapters/` |
| `results/orgb_ctrl_s0/orgb_A_s0.pt` | **the seed-0 organism**, same recipe | 21 023 243 B, sha256 `f1936b68681f1561`; second copy in `~/mz01_adapters/` |
| `results/orgb_ctrl_s{0,1}/orgb_ckpt_A_s{0,1}.pt` | the driver replications' mid-cell checkpoints (every 50 steps) | final, and they hold **step 150**: 126 102 329 B each, sha256 `ea75b41537f1aa51` (s0) and `46deb407a717c47a` (s1). No checkpoint exists at the AVOIDS horizon of 120 |
| `results/orgb_ctrl_s{0,1}/orgb_ctrl_s{0,1}.json` | the two driver replications' rows and instruments, the first organism result files in the programme | 2 731 001 B, sha256 `b552a0412d805143` (s0); 2 866 339 B, sha256 `dfa87715141ca98c` (s1) |

Everything under `results/` on this machine is the only copy. `docs/FAILURES.md`
B5 is what happens when the only copy is on a box.

### 2.1 The 2026-09-09 result files the notes quote

Untracked, this machine only, sizes and sha256 (first 16) as verified on the
day. The verdicts these files carry are in `PLAN-exit-priced-at-cap.md` §8 and
summarised in `docs/HISTORY.md` under 2026-09-09.

| file | bytes | sha256 (first 16) | what it is |
| --- | --- | --- | --- |
| `results/coverage_s0/coverage_s0.json` | 15 930 | `85d3c51d27417215` | **the coverage read**: pass@64 for the priced behaviour on 128 mined held-out mazes at T = 1.0 — coverage 0.0234375 (3 mazes), clean-solve rate 0.000488 (4 of 8 192 episodes), sampler solve 0.9636, 2.147 mine landings/episode, 6 123.6 s. The evidence for D-21. Only T = 1.0 completed |
| `results/mass_ctrl_s3/summary.json` | 6 730 | `5b45d5f99175eec7` | the seed-3 control, DEAD at 160 |
| `results/mass_ctrl_s5/summary.json` | 6 719 | `2870f34c9dab48f4` | the seed-5 control, DEAD at 160 (the nearest miss on a fresh seed) |
| `results/mass_ctrl_s1_rep/summary.json` | 6 736 | `120170a64f2e2b1b` | **the same-seed replication of the one AVOIDS read**, DEAD at 160 with no AVOIDS at any read (`FAILURES.md` A8) |
| `results/R4_togo_ctrl_s0/summary.json` | 6 747 | `ad4349f1fca12984` | return-to-go credit on nav48, DEAD at 160 |
| `results/R5_nav48u_ctrl_s0/summary.json` | 6 731 | `463c38d2c6e92cf9` | return-to-go credit on navigator v2, DEAD at 160 |
| `results/temp15_ctrl_s0/summary.json` | 7 021 | `37305cbc5c269f9a` | the T = 1.5 sampler arm, DEAD |
| `results/temp20_ctrl_s0/summary.json` | 6 943 | `320517962969421d` | the T = 2.0 sampler arm, RANDOMISED by 0.009 of solve |
| `results/relg12_ctrl_s0/05_relative_g12.json` | 98 543 | `969a0f43cdfee2fa` | **partial**: the group-12 arm's trace through step 126, three reads, no final — its box died (below) |
| `results/mass_ctrl_s4/onpolicy_g6.json` | 100 802 | `74370724ba08a6c6` | **partial**: the seed-4 control's trace through step 132 and its reads at 40/80/120, no final — its box died (below) |

Every arm directory also holds the arm's own JSON (`onpolicy_g6.json`,
`temperature_t15.json`, …, 98–131 KB) and its run log. The two **partials**
have no `summary.json`, which is written at the end of a run; `mass_ctrl_s5`
and `mass_ctrl_s1_rep` were pulled without their `reference.json`, and for
those two the reference row survives only in the run log's `MEASURED
reference (warm start)` line and in `PLAN-exit-priced-at-cap.md` §8.

### 2.2 Lost with a lease on 2026-09-09

Four leases died in one day (`docs/FAILURES.md` B22). What went with them,
and whether it is recoverable:

| lost | when | what it was | recoverable |
| --- | --- | --- | --- |
| `mass_ctrl_s4`'s final read | box 4 gone (HTTP 410) by 11:48Z; alive at step 130 at 09:06Z, final due ~09:35Z | the seed-4 control's read at 160. Whether the run finished is unknowable | by rerunning the seed — but not queued: D-21 closed the recipe |
| `relg12_ctrl_s0`'s final read | box 3 alive at step 126 at 14:43Z, gone by 14:53Z | the group-12 registry arm's read at 160; the arm was dead by its traces (entropy 0.001, zero-advantage 1.00) from step 70 | as above |
| the coverage passes at T = 1.5 and T = 2.0 | box 2 died between 16:24Z and 16:34Z, 18 of 128 mazes into T = 1.5 (2 with a clean solve) | the two hot-sampler coverage reads; the completed T = 1.0 pass survives in the file above | **relaunched the same evening** as `coverage_s0_hot` (box 3, 17:14:10Z) |
| `nav48_4b`, **entire** | finished its SFT ~14:30Z and its gate on the box; box 1 gone by 17:03Z | the 4B navigator: no gate row, no adapter, nothing | **relaunched the same evening** on box 2 at 17:14:03Z. Cause: the lead had left box 1 out of the monitor (`FAILURES.md` B21) |

Nothing of the two killed launches of the day survives either, and nothing is
meant to: R1 and R2 (07:34Z, against a withdrawal) and the mislabelled
group-12 rerun (11:51Z) left only log files or nothing at all
(`FAILURES.md` B17, B18).

## 3. How to remake each, if this machine is lost

| artefact | recipe | cost |
| --- | --- | --- |
| `nav_nav48.pt` | `scripts/mz01_navigator_driver.py` with the nav48 manifest's knobs (48 000 decisions, 3 epochs, batch 8, lr 1e-4, ASCII n=9, `MZ01_NAV_LABEL=first`, `MZ01_SHARD=nav48`) on a 96 GB box, then `score_mz01_navigator.py` must print PASS on all five criteria | ~1 h 50 m of GPU; a retrain is not bit-identical (bf16), so its sha will differ and every downstream sha check must be re-recorded |
| `warm_start.pt` | `scripts/mz01_orgb_driver.py` with `MZ01_WARM_START_ONLY=1 MZ01_OUT=<dir> MZ01_SHARD=<name>` and the navigator above | ~20 min |
| `nav_nav48u.pt` | as nav48 with `MZ01_NAV_LABEL=uniform MZ01_SHARD=nav48u`; its warm start as above into `results/orgb_nav48u/` | ~1 h 50 m + 20 min |
| an organism | `scripts/box/launch_driver.sh <box> <seed>` (cell A, 160 steps, `fixed` coefficient 0.1, mask on, exit priced, mass 0.1, soft pre-flight, checkpoints every 50) | ~1 h 45 m training + ~1 h measurement. ⟨2026-09-09⟩ this is the recipe that produced the two saved organisms and the one AVOIDS read; **D-21 closed it as a test of the price**, so it remakes these adapters and is not the recipe for a new one |
| the coverage read | `scripts/box/launch_cmd.sh <box> <label> "python3 -u scripts/mz01_coverage.py"` with `MZ01_COV_K`, `MZ01_COV_TEMPS`, `MZ01_COV_OUT`, `MZ01_SEED` and the trial's model/adapter/warm-start variables; trains nothing and writes no adapter | 6 123.6 s for one temperature at k = 64 over 128 mazes, as the result file records |

## 4. Where they should live (not yet done)

The plan (`docs/planning/PAPER_PLAN.md` §3 item 7; the parent's handoff item
"Adapters → HF") is a private Hugging Face repository holding every adapter
with its sha256, the JSON beside it, and the commit that produced it; boxes
would upload directly from `/marimo` at the end of a run (their egress is
fast; the exec channel's pull is ~50 KB/s) and this machine would be a
mirror, not the only copy. That needs a token on the boxes and a repository
name; neither exists yet. Until then: pull, verify, and keep two copies.

## 5. Not in this repository at all

- The v1 grid world's adapters, rows and results: the parent repository,
  `/home/manan/Projects/files/digital_minds` (branch `v2/hidden-habit`), and
  its `HANDOFF-v2.md` (stale for the maze).
- The 08-30 pre-commitment chain script (`mz01_cellA_chain.sh`, md5
  `5f9a91e5…`): in a wiped scratchpad; its header is quoted in
  `RESULTS-2026-08-30.md` §0 and §3a and that quotation is the record.
