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
| `results/navigator/nav_nav48u_epoch1.pt` | navigator v2's epoch-1 checkpoint, pulled 2026-09-09 while it trains | interim; the final `nav_nav48u.pt` replaces it when the run ends |
| `results/orgb/warm_start_s{1,2}.pt` | the two other seeds' warm starts | byte-identical to `_s0` |
| `results/orgb_ctrl_s{0,1}/orgb_ckpt_A_s{0,1}.pt` | the driver replications' mid-cell checkpoints (every 50 steps), pulled as they land | interim; the final adapters `orgb_A_s{0,1}.pt` are the organisms |

Everything under `results/` on this machine is the only copy. `docs/FAILURES.md`
B5 is what happens when the only copy is on a box.

## 3. How to remake each, if this machine is lost

| artefact | recipe | cost |
| --- | --- | --- |
| `nav_nav48.pt` | `scripts/mz01_navigator_driver.py` with the nav48 manifest's knobs (48 000 decisions, 3 epochs, batch 8, lr 1e-4, ASCII n=9, `MZ01_NAV_LABEL=first`, `MZ01_SHARD=nav48`) on a 96 GB box, then `score_mz01_navigator.py` must print PASS on all five criteria | ~1 h 50 m of GPU; a retrain is not bit-identical (bf16), so its sha will differ and every downstream sha check must be re-recorded |
| `warm_start.pt` | `scripts/mz01_orgb_driver.py` with `MZ01_WARM_START_ONLY=1 MZ01_OUT=<dir> MZ01_SHARD=<name>` and the navigator above | ~20 min |
| `nav_nav48u.pt` | as nav48 with `MZ01_NAV_LABEL=uniform MZ01_SHARD=nav48u`; its warm start as above into `results/orgb_nav48u/` | ~1 h 50 m + 20 min |
| an organism | `scripts/box/launch_driver.sh <box> <seed>` (the recipe of the AVOIDS read: cell A, 160 steps, `fixed` coefficient 0.1, mask on, exit priced, mass 0.1, soft pre-flight, checkpoints every 50) | ~1 h 45 m training + ~1 h measurement |

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
