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

Filled in below as each landed (commit in parentheses).
