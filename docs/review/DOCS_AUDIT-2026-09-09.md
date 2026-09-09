# Documentation audit — cold read, 2026-09-09

An independent Opus agent with no session context read `docs/` in the order
the map prescribes, opened every file it pointed to, ran the scorers and the
suite read-only, and answered twelve questions a newcomer must be able to
answer. Its report is reproduced here verbatim (HTML entities decoded). The
fixes it forced are listed at the end, with the commit that made each.

---

## Verdict (its own words)

Almost — I could take over the science tomorrow but not the machinery: the
record of what was measured, why, and under which pre-registered rule is the
best I have audited, and I re-derived its headline number myself, but the
operational half would have me launch a run that trains nothing, watch the
wrong directory, be unable to pull the organism, and be unable to run any GPU
stage at all because the two adapters everything stands on are not in the
repository and no document says where they are.

## A. Gaps (ranked)

1. `mine_step_rate` has no stated denominator. It is mines per episode
   (`maze2.py` `mines / n_episodes`); printed as `mine/ep`; a different
   number, mine landings per move, appears in the same tables.
2. The binary inputs are not in the repository and no document says where
   they are: `nav_nav48.pt`, `warm_start_s0.pt` are gitignored (`*.pt`);
   every downstream instruction is impossible from a clone.
3. `MZ01_PREFLIGHT_SOFT=1` is missing from the only documented
   organism-driver launch block (OPERATIONS §3.2) and from the PLAN's
   replication spec; without it a run loads the model, fails the R2 gate and
   trains nothing. Only `launch_driver.sh` sets it.
4. No dependency manifest for the local environment.
5. `D0` is used in `docs/` and defined only in `PLAN-orgb-v2-CRITIQUE.md`.
6. No legend for the tag families (A1–A11, C1–C12, E1–E11, DEC-1..4,
   D1–D37, M2–M10, B1–B6, PR-0..7, H5/H10/H11/H13, R1/R2, E1–E8).
7. `R1`/`R2` and `E1–E8` are each overloaded.
8. How a box is obtained is nowhere: provider, cost, lease length, account.
9. `archive/` is undocumented (E12, E13, E16, E17, HH01, NAR01, NAR02).
10. VAL01, NAR04, I1–I7, E16/E17 are cited as evidence with no pointer.
11. Which of `maze_rl.py`, `maze.py`, `rl.py` are live is not stated.
12. The yell term is never defined on its own.
13. How an instrument reading is actually taken in this world is not stated.

## B. Contradictions

1. The wall exit's price: RESEARCH_PROPOSAL §2 says ≈ −10; CONSTRUCTION §5.2
   says about −6; the verifier had already forced −6.
2. The comparison figure: MZ01 README says −28; PLAN-exit-priced §7 retracts
   −28 for −13.6; the README was never corrected.
3. `pull_trial.sh` and the status file: CONSTRUCTION §10 and OPERATIONS trap 9
   describe the old behaviour; the script (f74daaf) does the opposite.
4. CONSTRUCTION §10 says the drivers have no launcher in the repo;
   `scripts/box/launch_driver.sh` exists.
5. OPERATIONS §4 says `pull_big.sh` is not in the repository; it is.
6. CONSTRUCTION §5.6 flags a README schema line that has since been fixed.
7. OPERATIONS trap 15 ("never pull a mid-cell checkpoint in the background")
   vs HISTORY and `watch_adapters.sh`, which do exactly that.
8. AGENT-NOTES numbering: D1–D37 / D1–D33 / D1–D19 across three READMEs.
9. "Results under results/ are not committed" — sixteen files are (the pilot
   and navigator JSONs).
10. The blindness bar: SPEC's 08-28 amendment says ceiling − 0.02; the code
    gates at 0.98 absolute; nothing appended to the SPEC.
11. `epsilon_030_near` vs `epsilon_045_near`: the MZ01 README's arm table and
    its R2 launch line name the withdrawn arm.
12. The seed-2 control's read at 80 (0.844 vs 0.9453, a fall of 0.1015) is
    RANDOMISED by the rule; HISTORY says "recovered … valid" and omits the
    verdict.
13. The root README describes `maze_rl.py` as the primary arm A; every
    organism is trained by `maze_orgb.train_orgb_cell`.

## C. Dead pointers

1. `docs/planning/README.md` does not list `PLAN-credit-explore-navigator.md`.
2. OPERATIONS §5 documents `score_mz01_orgb.py` as a scoring command; no
   organism result file exists in the tree, so it and every pass rule are
   undemonstrable, and no doc says so.
3. Pointers to the parent repo's `HANDOFF-v2.md`, recorded as stale.
4. MZ01 README "Sandbox scripts" says the helpers live in the scratchpad;
   forty lines later its own fresh-lease section names `scripts/box/`.
5. `docs/review/claims.json` and `tools/check_provenance.py` do not exist
   (declared ⬜; an honest hole).
6. CONSTRUCTION §4.1 cites `CELL_SWITCHES (maze_orgb.py:167)`; it is at 185.
7. Eight helpers in `scripts/box/` appear in no document: `bringup2.sh`,
   `bringup2_nosetup.sh`, `launch_driver.sh`, `launch_only.sh`,
   `pull_files.sh`, `pull_big.sh`, `watch_nav.sh`, `watch_pid.sh`,
   `watch_adapters.sh`.
8. OPERATIONS §4's `watch_trial.sh <label>` applied to a driver run watches
   `results/<label>` while the driver writes `results/orgb_<label>`.

## D. Opacity

1. "The reference row" appears everywhere before it is explained.
2. Five windowing conventions (`zadv s → e`, fifths, quarters, ninths, ‖g‖ as
   a mean in the scorer's table and as a pair in trace tables).
3. "The 0.75 bar" means three things and "the 0.10 bar" two.
4. "Cell D" means three things: the 2×2 cell with both switches off, v1's
   untouched base (the canary), and the navigator itself.
5. `mines_on_shortest_dag`: 1.16 (training pool), 2.4 (held-out mean),
   1.72–2.95 (per bin) — three populations, unlabelled at the point of use.
6. `offpol 0.000` on an on-policy arm is structural, not a measurement.
7. "As recorded" carries the pre-registration of AVOIDS itself (the chain
   script is gone; only its md5 survives).
8. Sentences that assume the reader watched: "the twelfth pass's addition",
   "the A+B critic swept it", "Box 1 died at ~15:27"; twelve unnamed critics;
   passes numbered to fourteen with no list of the first eleven.
9. The `-dirty` marker: which three files carry it and whether they are
   quotable.
10. "The exploration gate can only ever pass for temperature and epsilon"
    is stated as design; its operational consequence (every on-policy run
    needs the soft gate) is left to the reader.

## E. What a newcomer can and cannot do

Can: reproduce the pilot and navigator scorer tables from a clone; reproduce
the headline AVOIDS table on this machine; verify the record's hashes; run
the suite (271 passed, 1 skipped); judge a claim. Cannot: re-run any GPU stage
from a clone; recreate the local environment; launch the organism driver from
the runbook without it failing; watch or pull a driver run correctly from the
runbook; exercise the Stage-3 scorer; know what to launch next; reproduce
anything about the instruments, the sham, the decoy, the lineage control or
the 2×2.

## What §11 found about the queue

Three documents give three answers to "what runs on the next freed box" and
none is dated after the 2026-09-09 relaunch; followed literally the record
would launch an ε arm at mass 0.3 that the programme has abandoned.

---

## Fixes forced by this audit

Filled in below as each landed (commit in parentheses). The editor's pass of
2026-09-09 closed everything not marked "— lead"; those are the lead's and are
listed so the roll is complete. `MZ/` abbreviates
`experiments/v2/MZ01_maze_world/`.

### A. Gaps

- **A1** — `mine_step_rate` defined as mines ÷ **episodes** (`maze2.py:3078`,
  `mines / n`, printed `mine/ep`) and separated from `mine_landing_rate`
  (mines ÷ **decisions**, `:3079`) and `on_mine_state_rate`, with the statement
  that every verdict in the programme is on the per-episode rate.
  `docs/GLOSSARY.md`, `docs/CONSTRUCTION.md` §7.1.
- **A2** — lead.
- **A3** — lead (the PLAN's replication spec). The runbook half is closed here:
  a new §3.0 says an on-policy driver run without `MZ01_PREFLIGHT_SOFT=1` fails
  R2 and trains nothing, and §3.2's hand-written launch block now carries it.
  `docs/OPERATIONS.md`.
- **A4** — the local environment named and pinned: Python 3.14.7, torch
  2.13.0+cpu, numpy 2.5.2, pytest 9.1.1 in `.venv`, pointing at the lead's
  `requirements-local.txt`, with the boxes' own environment distinguished from
  it; suite re-run to confirm 271 passed, 1 skipped (~10 min).
  `docs/OPERATIONS.md` (standing environment).
- **A5** — `D0` defined as the untouched base kept as the reference cell the
  instruments are differenced against (`PLAN-orgb-v2-CRITIQUE.md` M5/PR-2),
  with the fact that no D0 read exists here. `docs/GLOSSARY.md`.
- **A6 + A7** — a "Tag families" table covering A1–A11, DEC-1..4, C1–C12,
  E1–E11, F1–F8, D1–D37, B1–B6, M2–M10, I1–I6, H1–H13, PR-0..7, R1–R7/N1,
  E1–E8, the failure ledger's own A/B/C and the decision log's D-01..D-20, plus
  a paragraph naming the four overloads (R1, R2, E1–E8 vs E1–E11, I1–I7) and
  which sense each document uses. `docs/GLOSSARY.md`.
- **A8** — where a box comes from: a molab sandbox from marimo's hosted
  service, rented by the project owner, RTX PRO 6000 Blackwell 96–102 GB, URL
  and token from the owner, leases of ~2.5–12 h in this record, dying without
  warning, and no API here to rent one. `docs/OPERATIONS.md` §1,
  `docs/GLOSSARY.md` ("box").
- **A9** — `archive/` documented: seven directories, 88 files of the parent
  harness's leftover worlds (E12, E13, E16, E17, HH01, NAR01, NAR02) from the
  v1 repository at `/home/manan/Projects/files/digital_minds`; gitignored; read
  by nothing; HH01 named as the parent's hidden-habit experiment in the v1 grid
  world. `docs/CONSTRUCTION.md` §0.2.
- **A10** — a table of the parent repository's experiments (VAL01, NAR01,
  NAR02, NAR03/NAR04, HH01, E16, E17, E12, E13) with the directory and driver
  for each, and I1–I7 identified as the v1 paper's seven instruments, §3 of
  `docs/papers/226_Known_by_Construction_Cali.pdf`, with which five
  `instrument_panel` reads. `docs/GLOSSARY.md`.
- **A11** — a module map, every row checked by grep: live for MZ01 (maze2,
  maze_orgb, navigator, capture, chat, lora, sft, remarks, runner, instruments,
  and `rl.py` **only** for `_TRAIN_SEED_RANGE`/`_EVAL_SEED_RANGE` with the six
  import sites); inherited and not on the path (`maze.py`, whose glyph
  constants alone are used, and `maze_rl.py`, from which only
  `group_advantages` is imported). `docs/CONSTRUCTION.md` §0.1, root
  `README.md`.
- **A12** — the yell term defined on its own: `yell_reward`'s +1 / −1 / **0**
  cases (`maze_orgb.py:617–639`), H8's definition of silence, the
  `(group, on_mine)` centring (`:665`), `YELL_WEIGHT = 10.0` (`:203`) applied
  at `:2561` so w is 10 in B/C and 0 in A/D, and PR-3's two fields (`:4385`,
  `:4389`). `docs/GLOSSARY.md`.
- **A13** — what `instrument_panel` computes, per instrument, as the code
  stands (I2, I3, I4, I5, I7, `response_rate`, with the function each calls),
  and a plain statement that **Phase 3's reading procedure is not designed** and
  no instrument reading has been taken in this world. `docs/CONSTRUCTION.md`
  §8.

### B. Contradictions

- **B1** — the wall exit repriced in the proposal: three bumps −6, an observed
  farm-stopped episode ≈ −9 to −12, against −13.6 on the **training pool**,
  with the −28 held-out idealisation named as the different population it is.
  `docs/planning/RESEARCH_PROPOSAL.md` §2.
- **B2** — the same correction in the MZ01 README's thirteenth-pass section:
  −13.6 on the training pool, the held-out idealisation −27.9 ≈ −28 named as
  such and as what the paragraph used to quote. `MZ/README.md`.
- **B3** — `pull_trial.sh` pulls `results/<label>/*` and `<label>.log` only; the
  box-wide status file is no longer copied (`f74daaf`), with the two
  directories that still hold a wrong-header copy named.
  `docs/CONSTRUCTION.md` §10, `docs/OPERATIONS.md` §4 and §8 trap 9.
- **B4** — `scripts/box/launch_driver.sh` exists and is described with the
  recipe it pins (including `MZ01_PREFLIGHT_SOFT=1`), beside `launch_only.sh`.
  `docs/CONSTRUCTION.md` §10, `docs/OPERATIONS.md` §3.2 and §9.
- **B5** — `pull_big.sh` is in `scripts/box/`: 256 KB chunks, ~50 KB/s,
  per-chunk sha256, halving to a 32 KB floor, a resumable `.part`, a whole-file
  sha256, and the `flock`. `docs/OPERATIONS.md` §4 and §9,
  `docs/CONSTRUCTION.md` §10.
- **B6** — the stale meta-correction about the README's `CKPT_SCHEMA` removed
  (the README says 6). `docs/CONSTRUCTION.md` §5.6.
- **B7** — trap 15 reconciled: the adapter watcher pulls each checkpoint after
  listing it, through `pull_big.sh`; a pull that straddles a rewrite fails the
  whole-file sha and is retried next cycle with the `.part` discarded
  (`4f86458`); the tearing is caught, not prevented; the old warning is kept
  where it still holds, against the generic puller's size-only check.
  `docs/OPERATIONS.md` §8 trap 15 and §4, `MZ/README.md`.
- **B8** — `D1–D37` everywhere, including the root README's `D1–D19` line and
  `docs/planning/README.md`'s `D1–D33`. root `README.md`,
  `docs/planning/README.md`.
- **B9** — the convention restated with its exception: results are not
  committed **except** `results/pilot/*.json` and `results/navigator/*.json`,
  sixteen tracked files, which is what makes two scorers runnable from a clone.
  `docs/README.md`.
- **B10** — lead.
- **B11** — `epsilon_045_near` is the arm that runs (0.45 × 0.22 = 0.10);
  `epsilon_030_near` stays defined and not run: the arm table, the
  fourteenth-pass paragraph and the R2 launch line all corrected.
  `MZ/README.md`; the now-stale "code vs note" in `docs/CONSTRUCTION.md` §6.1
  replaced.
- **B12** — the seed-2 control: the read at 80 is 0.844 / 0.81, **valid and
  RANDOMISED** by the rule (a fall of 0.1016 against 0.9453), and the read at
  120 is 0.867 / 0.82, **ALIVE**; both now said. The run has since **finished**
  (160 steps, DONE) and its own verdict is RANDOMISED at 0.703 / 0.83, so the
  entry is written up in full and "nothing here explains the gap" is replaced
  by the actual explanation — a different box and a different commit, with
  CUDA bf16 non-determinism from step 1, quoted from the result file's own
  `bit_exact_scope`. `docs/HISTORY.md`, `docs/RESULTS_LOG.md` §8 (new §8.1) and
  its closing section, `docs/ORIENTATION.md`.
- **B13** — `maze_rl.py` described as the inherited whole-episode trainer that
  is not on MZ01's path (only `group_advantages` is imported); organisms are
  trained by `maze_orgb.train_orgb_cell`; A's recipe is `docs/CONSTRUCTION.md`
  §5. root `README.md`.

### C. Dead pointers

- **C1** — `PLAN-credit-explore-navigator.md` indexed, with its §4 runs, §8
  gate and §9 amendments. `docs/planning/README.md`.
- **C2** — `score_mz01_orgb.py` and `mz01_entropy_trace.py` have no organism
  result file to read yet; the first will be the driver replications' rows, and
  the Stage-3 pass rules have therefore never been run on real data.
  `docs/OPERATIONS.md` §5.
- **C3** — the parent's `HANDOFF-v2.md` marked stale for MZ01 at both citations
  in this repository, with the current source named in its place; the note owed
  in the parent tree left open. `MZ/README.md`,
  `docs/planning/RESEARCH_PLAN.md` §0.4 and §11.
- **C4** — "Sandbox scripts" corrected: the helpers are in `scripts/box/`, not
  a scratchpad, and §9 of the runbook documents each. `MZ/README.md`.
- **C5** — lead.
- **C6** — `CELL_SWITCHES` is at `maze_orgb.py:185`. `docs/CONSTRUCTION.md`
  §4.1, `docs/GLOSSARY.md`.
- **C7** — a section documenting **all eighteen** helpers in `scripts/box/`
  (the seventeen the audit read, plus `watch_box.sh`, added in `09e0bcb`)
  with their own usage lines, env knobs and exit codes, grouped by channel /
  files / launching / watching. `docs/OPERATIONS.md` §9.
- **C8** — which watcher belongs to which job, in a table: `watch_pid.sh <box>
  <label> …/results/orgb_<label>` plus `watch_adapters.sh` for a driver run,
  `watch_trial.sh` for trials only, `watch_nav.sh` for a navigator, and what
  goes wrong if the trial watcher is pointed at a driver.
  `docs/OPERATIONS.md` §4.

### D. Opacity

- **D1** — "the reference row" defined at first use as the warm start's own
  held-out read on the same mazes before any arm, with the numbers it takes and
  the NO-REFERENCE verdict. `docs/GLOSSARY.md`, `docs/ORIENTATION.md`.
- **D2** — the windowing conventions: the scorer's first/last fifth
  (`WINDOW_FRACTION = 5`), ‖g‖ as the mean over **all** steps, and quarters and
  ninths marked as ad-hoc readings taken in the notes. `docs/GLOSSARY.md`.
- **D3 + D4** — the three senses of "cell D" separated in a table, with the
  names to use: **the navigator** (D's base, and every other cell's), **cell D**
  (the trained cell), **the untouched base (D0, the canary)**; ORIENTATION's
  "Navigator (cell D)" and two other spots corrected. `docs/GLOSSARY.md`,
  `docs/ORIENTATION.md`, `MZ/README.md`, `docs/planning/RESEARCH_PLAN.md`.
- **D5** — the three `mines_on_shortest_dag` populations labelled wherever they
  are quoted: 1.72/2.28/2.73/2.95 per detour bin on 600 fresh n=9 mazes (AUC
  0.81), 1.16 on the RL training pool, 2.39 held out. `docs/CONSTRUCTION.md`
  §1.2, root `README.md`, `MZ/README.md`.
- **D6** — `offpol 0.000` on an on-policy arm named as structural, not a
  measurement. `docs/GLOSSARY.md`.
- **D7** — not assigned in this pass; still open.
- **D8** — a table of all fourteen passes with what each built and its commits,
  plus the fact that passes 8–11 were written on 08-29/30 and landed together
  in `3bc103c` / `60ae17a`, which is why `AGENT-NOTES.md` has headings for
  1–9 and none for the tenth and eleventh. `docs/HISTORY.md`.
- **D9** — the `-dirty` marker: **fourteen** arm files carry `04e5e69-dirty`
  (not three), none of them committed, and the three of `explore_cellA_s0` are
  quotable because `RESULTS-2026-08-30.md` §0 records the synced files' md5s,
  verified box against worktree; the ten cell-D trial files have no per-file
  md5 record. `docs/RESULTS_LOG.md` §5, `docs/GLOSSARY.md`,
  `docs/CONSTRUCTION.md` §10, `docs/OPERATIONS.md` §2.3.
- **D10** — the operational consequence of the exploration gate spelled out:
  every on-policy run needs `MZ01_PREFLIGHT_SOFT=1` or the driver trains
  nothing, the trial always runs soft, `launch_driver.sh` sets it, and the
  waiver is recorded. `docs/OPERATIONS.md` §3.0.

### §11, the queue

- lead.
- A2 — `docs/ARTEFACTS.md`: every binary, its hash and size, where it is, how to remake it, and that a clone cannot run a GPU stage without them (a663f82) — lead.
- A3 (PLAN half) — `PLAN-exit-priced-at-cap.md` §8 amendment 1: the replication line requires `MZ01_PREFLIGHT_SOFT=1`; `launch_driver.sh` is the launcher of record (a663f82) — lead.
- B10 — `SPEC.md` reconciliation note appended: the implemented absolute 0.98 bar is the criterion of record; the relative form is retired; both navigators' readings given (a663f82) — lead.
- §11 — `PLAN-exit-priced-at-cap.md` §8 amendment 2: the queue restated after the 09-09 relaunch; the ε arms withdrawn by name (a663f82) — lead.
- C5 — `docs/review/claims.json` and `tools/check_provenance.py`: still not built; declared ⬜ in RESEARCH_PLAN §0.3 and OBJECTIVES O10 — open, lead.
- D7 — "as recorded" carrying the AVOIDS pre-registration: the chain script's header is quoted in RESULTS-2026-08-30.md §0/§3a and its md5 recorded; nothing more can be recovered — open by nature, noted in GLOSSARY under "as recorded" — lead.
- Environment — `requirements-local.txt` (a663f82) — lead.

