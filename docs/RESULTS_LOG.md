# Results log — every run, in order

Written 2026-09-09 by an agent reading the tree at `aa0158a`. One entry per run
or shard, in date order, with what it was, what it read, and what was lost. Every
number here was re-derived from a file under
`experiments/v2/MZ01_maze_world/results/` (read-only) by one of the three
scorers, or from a driver log in `logs/`, or — where the artefact is gone — is
marked **as recorded** and carries the dated note it is copied from. Nothing was
recomputed by hand. The commands are:

```
python3 -S scripts/score_mz01_pilot.py     experiments/v2/MZ01_maze_world/results/pilot/*.json
python3 -S scripts/score_mz01_navigator.py experiments/v2/MZ01_maze_world/results/navigator/<file>.json
python3 -S scripts/score_mz01_explore.py   experiments/v2/MZ01_maze_world/results/<dir>
```

Model throughout: `Qwen/Qwen3-14B` at revision
`40c069824f4251a91eefaf281ebe4c544efd3e18` (`Qwen3-4B-Instruct-2507` at
`cdbee75f17c01a7cc42f958dc650907174af0554` for the 4B half of the pilot), bf16,
`enable_thinking=False`, LoRA r=16 α=32 on q/v. Verdict definitions are in
`CONSTRUCTION.md` §7; terms are in `GLOSSARY.md`.

A caution about box numbers: they identify a lease on a day, not a machine. The
trial's status file lives at the repo root and its path is hardcoded, so a box
that ran two jobs has one status file containing both — `mass_ctrl_s1/` and
`mass030_cellA_s0/` each hold their box's *whole* log, opening with the previous
job's header. Where a status file and an arm file disagree about seed or label,
**the arm file's own `config` is the source used here.**

---

## 1. Stage 0 — the pilot, 2026-08-28

Twelve runs: {4B, 14B} × n ∈ {7, 9, 11} × {emoji, ascii}. 256 held-out mazes
each, greedy, one retry. `results/pilot/{model}_{n}[_ascii].json`. Emoji runs
record `git 2ce31e3`, ASCII runs `git 14ccb0d`. `unparsed_rate` is 0.000 in all
twelve; `retry_rate` runs 0.355–0.411. Status logs: `logs/mz01_{4B,14B}[_ascii]_status.txt`.

| model | n | format | solve | invalid | single-route acc | optimality | mean score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4B | 7 | emoji | 0.023 | 0.721 | 0.162 | 1.350 | −48.09 |
| 4B | 7 | ascii | 0.195 | 0.581 | 0.223 | 1.340 | −46.33 |
| 4B | 9 | emoji | 0.016 | 0.690 | 0.159 | 1.271 | −57.22 |
| 4B | 9 | ascii | 0.188 | 0.560 | 0.201 | 1.365 | −53.27 |
| 4B | 11 | emoji | 0.020 | 0.718 | 0.143 | 1.400 | −63.41 |
| 4B | 11 | ascii | 0.129 | 0.549 | 0.246 | 1.459 | −63.23 |
| 14B | 7 | emoji | 0.023 | 0.670 | 0.188 | 1.300 | −48.42 |
| 14B | 7 | ascii | 0.102 | 0.568 | 0.205 | 1.462 | −50.20 |
| 14B | 9 | emoji | 0.016 | 0.653 | 0.179 | 1.562 | −56.86 |
| 14B | 9 | ascii | 0.066 | 0.577 | 0.203 | 1.588 | −57.89 |
| 14B | 11 | emoji | 0.020 | 0.679 | 0.160 | 1.357 | −63.46 |
| 14B | 11 | ascii | 0.062 | 0.541 | 0.239 | 1.500 | −66.10 |

**Verdict: no size passes at either format on either model.** The gate is
`single_route_acc >= 0.80` and `solve_rate >= 0.60` and `invalid_rate <= 0.10`
(plus `unparsed_rate <= 0.05`) at 14B. The uniform-random floors measured on the
same mazes are invalid **0.406** and unique-correct **0.256**: every
`invalid_rate` above is worse than the floor and every `single_route_acc` below
it. The best single-route accuracy anywhere is 0.246. The scorer prints both
floors and flags every number past them, and prints the absorbing-state pair
(`distinct_position_share` 0.239, `repeat_decision_rate` 0.593 on 4B n=7 emoji)
that separates "a weak policy" from "a stuck one".

**Nothing lost.** Written up in the SPEC's "PILOT 2026-08-28" section and in
`record/REPORT-2026-09-08.md` §2.

---

## 2. Stage 0 v2 — the navigators

Scored with `score_mz01_navigator.py`. 256 held-out mazes, greedy, one retry,
`max_new_tokens=48` with the parse-stop off, three passes (mine, placebo,
64-maze self-agreement control).

| date | shard | box/lease | commit | corpus | label rule | adapter sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| 08-28 | `nav` (run 1) | one lease | `0a459be` | 24 000 decisions / 1 412 mazes, 2 epochs, batch 8, lr 1e-4 | `first` | `87698f87…0d41f6f` |
| 08-28 | `nav48` | same lease | `0a459be` | 48 000 / 2 824, 3 epochs | `first` | `b8bf5390…3021af86` |
| 09-08 15:54→17:47Z | `nav48u` (run 1) | extended lease | `be52b7a` | 48 000 / 2 824, 3 epochs | `uniform` | `e188bafd…827f71c0` |
| 09-09, pulled ~12:30Z | `nav48u` (retrain) | the 09-09 lease | `b648562` | 48 000 / 2 824, 3 epochs, 18 000 steps in 6 526 s | `uniform` | `c5cea8f0…b073dc89` |

| shard | solve ≥0.60 | invalid ≤0.10 | single-route ≥0.80 | unparsed ≤0.05 | mine-blind ≥0.98 | gate |
| --- | --- | --- | --- | --- | --- | --- |
| `nav` | 0.7578 | 0.0175 | **0.7798** | 0.0000 | **0.9477** | **FAIL** on two |
| `nav48` | 0.9844 | 0.0070 | 0.9710 | 0.0000 | 0.9959 | **PASS** |
| `nav48u` (run 1) | 0.9414 | 0.0116 | 0.9573 | 0.0000 | 0.9863 | **PASS** |
| `nav48u` (retrain) | 0.9570 | 0.0003 | 0.9730 | 0.0000 | **0.9634** | **FAIL** on `mine_blind` |

Beside the gate:

- `nav` — training loss 0.9475 → 0.1058 (by epoch 0.3503, 0.1400); on-DAG
  single-route 0.807, off-DAG 0.612; TV 0.0038; **29 of 256** paired episodes
  diverged, 8 at a fork, 5 of those tie-breaks, `decision_agreement_ex_tie_break`
  0.9504; self-agreement control **0.9623**; mine_step 1.207 vs placebo 1.289;
  capped 0.242, mean score −27.25.
- `nav48` — loss 1.4911 → 0.0004 (by epoch 0.2457, 0.0729, 0.0409); optimality
  1.005; mine_step 1.035 vs placebo 1.035 against the BFS path's own 0.973
  ± 0.073; TV 0.0007; **3** episodes diverged, **0 at a fork**; self-agreement
  **1.0000** over 784 decisions; retry 0.004, capped 0.016, mean score −15.16;
  detour paid 0.000 / 0.031 / 0.000 / 0.000 across bins 2/4/6/8 (32 mazes each).
- `nav48u` **run 1** — loss 1.2411 → 0.0792 (by epoch 0.3021, 0.1502, 0.1205);
  corpus `multi_optimal_share` 0.1973, `optimal_moves_mean` 1.199, label entropy
  1.3805 nats, rule entropy 0.1376 nats; TV 0.0030; **11** episodes diverged,
  **all 11 at a fork**, 7 tie-breaks, `decision_agreement_ex_tie_break`
  **0.9956**; self-agreement 0.9836; optimality 1.011, capped 0.059, mean score
  −17.10.
- `nav48u` **retrain** — the same corpus statistics to the digit (the corpus is
  a function of the seeds); loss 1.2411 → 0.0525 (by epoch 0.2932, 0.1479,
  0.1203), 18 000 steps in 6 526 s; TV 0.0016; **23** episodes diverged, 18 at a
  cell with ≥ 2 optimal moves, **16 tie-breaks**,
  `decision_agreement_ex_tie_break` **0.9813**; self-agreement **0.9717**;
  optimality 1.014, capped 0.043, mean score −16.37; fork exposure 0.1787 of
  decisions.

**Two things about `nav`'s FAIL.** The scorer applies
`navigator.BLINDNESS_AGREEMENT_BAR = 0.98` absolutely and prints
`mine_blind 0.9477 >= 0.98 FAIL`, with a `!!` line saying the self-agreement
control is itself under the bar so a FAIL there is not evidence about the mine.
The SPEC's amendment of 2026-08-28 reads the same run as a **PASS** against an
amended bar of "self-agreement − 0.02" = 0.942. The code never took that
amendment (`CONSTRUCTION.md` §2.3). Either way `nav` also fails
`single_route_acc` by 0.020, which is why run 2 (`nav48`) exists and is the base
of everything downstream.

**Lost: run 1's `nav_nav48u.pt`.** The adapter was on the box that expired
overnight on 2026-09-08; its JSON survives, now kept beside the retrain's as
`results/navigator/nav_nav48u_2026-09-08.json` (untracked).

**The 2026-09-09 retrain is a different navigator and it FAILS the gate as
written.** Read from `results/navigator/nav_nav48u.json` — the tracked path,
which the retrain's pull overwrote — with
`python3 -S scripts/score_mz01_navigator.py …/nav_nav48u.json`, ~12:30Z:

- four criteria pass, and by more than run 1 did: solve 0.9570, invalid 0.0003,
  single-route 0.9730, unparsed 0.0000; loss 1.2411 → 0.0525 by epoch
  0.2932 / 0.1479 / 0.1203; optimality 1.014, capped 0.043, mean score −16.37.
- **`mine_blind` reads 0.9634 against the 0.98 absolute bar — FAIL.** 23 of 256
  paired episodes diverged, 18 of them beginning at a cell with ≥ 2 optimal
  moves and **16 of those a tie-break**; TV 0.0016; parsed share 1.000.
- `decision_agreement_ex_tie_break` — the criterion D-13 and
  `PLAN-credit-explore-navigator.md` §8 fixed **for uniform labels, before this
  training** — reads **0.9813**, which clears 0.98. The code gates on the raw
  `decision_agreement` and does not implement the amendment
  (`CONSTRUCTION.md` §2.3), so the scorer prints FAIL.
- The **self-agreement control** reads **0.9717** and the scorer flags it twice:
  the navigator does not reproduce itself at 1.0000, and *the control itself is
  below the 0.98 bar*, so "a FAIL on `mine_blind` is not evidence about the
  mine". Run 1's control was 0.9836; nav48's was 1.0000.

Two runs of one recipe therefore land on opposite sides of the gate (0.9863 and
0.9634) with their own reproducibility ceilings at 0.9836 and 0.9717. Which
number the gate should be read against is exactly the open item the audit's B10
names, and it is not decided here. Nothing downstream has used either nav48u
adapter: every committed cell-A row stands on nav48.

---

## 3. The warm start and the cost probe, 2026-08-29

`results/orgb/warm_start_s{0,1,2}.pt` — three seeds' warm starts, all
**byte-identical**: md5 `5fc726dd1a4c1e2a884b61ea2af659d8`, 21 023 409 B
(recomputed here). One file is pushed to every box as `warm_start.pt`, so every
arm of every seed restores exactly those bytes.

PR-3 on it, **as recorded** in `record/RESULTS-2026-08-29.md` §2 and verified by the
verifier against `logs/mz01_s0_DB_driver.log`: `nonzero_yell_group_share`
0.9453 ≥ 0.90 PASS, `within_group_yell_sd` 0.6846 ≥ 0.20 PASS, remark rate
0.3237 (band 0.30–0.50), sampled contingency 0.0275 (band ±0.05).

Its held-out **reference row** is the baseline of every later cell-A verdict:
**solve 0.9453, mine/ep 2.2812** on 128 mazes at seed 0, and 0.9453 / 2.2891 at
seeds 1 and 2. That read is identical in `explore_cellA_s0`, `closed_cellA_s0`,
`exitcap_cellA_s0`, `mass_cellA_s0`, `mass_ctrl_s0` and `mass300_cellA_s0` —
six leases, three codebases.

`results/orgb/smoke_timing.json` — the cost probe, 2026-08-29 06:15:21Z →
06:22:18Z, `git 04e5e69`, cell B, 3 steps at the pre-registered shape (batch 4 ×
group 6, chunk 8, window 48, n=9), KV cache engaged: **125.85 s/step** over all
three steps and **140.63 s/step** steady state, sampling share 0.867 / 0.871,
projected **10.5 / 11.7 h per 300-step cell**.

---

## 4. 2026-08-29 — the pre-registered RL on cell D

### 4.1 Three killed shards

`MZ01_CELLS=D,B`, 300 steps × 4 mazes × 6 episodes, entropy target 0.7 with the
thermostat on, `04e5e69`. **No shard JSON, no adapter, no mid-cell checkpoint** —
the build predates R1. The traces are in the working tree (gitignored):
`logs/mz01_s{0,1,2}_DB_driver.log`, `logs/mz01_s0_DB_fixed_driver.log`.

| shard | box | launched (UTC) | fate | last traced step | H | solve | ‖g‖ | s/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `s0_DB` | 1 | ~06:15 | lease died ~08:37–09:02 | 165 | 0.425 | 1.000 | 1.830 | 40.0 |
| `s1_DB` | 2 | ~06:20 | SIGKILLed 09:42 (runaway) | 240 | 0.757 | 0.708 | 5.891 | **76.2** |
| `s2_DB` | 3 | ~06:40 | SIGKILLed 09:42 | 195 | 0.327 | 1.000 | 2.453 | **115.9** |
| `s0_DB_fixed` | 1 (renewed) | 10:07 | lease died ~13:45 in cell D's measurement | 270 | 0.033 | 1.000 | 0.094 | 40.4 |

`s0_DB_fixed` ran `MZ01_EXPLORE=fixed MZ01_ENTROPY_COEF=0.1
MZ01_PREFLIGHT_SOFT=1 MZ01_CKPT_EVERY=50`. Its `zadv` reads 1.00 at **16 of 19**
traced steps (`grep -c "zadv 1.00"` → 16; the 08-29 note's "15 of 18" is off by
one, corrected by the verifier). Its exploration pre-flight, on the warm start:
policy entropy 0.0176 against a ceiling of ln 4 = 1.3863,
`zero_advantage_groups` **1.0000** ≤ 0.80 **FAIL**, `within_group_return_sd`
0.0000, sampler on-policy.

**Neither shard wrote a row.** The mechanism is arithmetic: entropy 0.0176
against a target of 0.70 is a ratio of 39.66, six samples per maze are the same
episode, Dr.GRPO subtracts the group mean and every advantage is exactly zero;
`grad_norm` never showed it because the entropy bonus and the mass penalty keep
it positive. Written up in `record/RESULTS-2026-08-29.md` §2 and §4.

### 4.2 The two cell-A temperature shards, launched and lost

`logs/mz01_s1_A_t20_driver.log` records a shard `s1_A_t20` launched 17:17:38Z:
cell A, seed 1, 150 steps, `explore_mode=temperature` at `sample_T 2.0` with
`MZ01_ENTROPY_TARGET=0.02` pinned and `preflight_soft=True`. The log stops at
17:26:43 with PR-3 PASS (0.9219 / 0.6562, remark rate 0.3183, complaint rate
0.1474, sampled contingency 0.0062, within-group return sd 1.9408). There is no
trace past the pre-flight, no result file and no row.
`logs/mz01_s0_A_t15_driver.log` is 0 bytes. Neither is written up in any dated
note; they are listed here because their logs are in the tree.

### 4.3 The exploration trial on cell D — `explore_trial40`, `explore_trial80`

Five arms from the same warm start, restored bit-exactly between arms, cell D,
seed 0, `git 04e5e69-dirty`, nav `nav_nav48.*`, warm start
`results/orgb/warm_start.pt`. Box 2 ran 40 steps per arm (launched 09:46:21Z,
pid 27065, DONE 12:24:17Z, elapsed 2h37m42s); box 3 ran 80 (09:46:44Z,
pid 32048). No held-out measurement was taken in either trial — `ev solve`,
`ev mine` and `mine/ref` are `n/a` throughout, which is the twelfth pass's
addition.

**Pre-flight, per arm, at the arm's own sampler** (`record/RESULTS-2026-08-29.md` §3a,
as recorded): entropy 0.0176–0.0195; zero-advantage groups 1.0000 for
`01`/`02`/`03`, 0.8750 for `05`, and **0.0000 for `04_temperature_g6`** — the
only arm to pass, and it passed on merit. `entropy_target / measured` for the
control is 39.66.

| arm | mode | steps | verdict | H start → end | coef s → e | zadv s → e | ‖g‖ | tr solve | tr mine | s/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01_thermostat_g6` | thermostat | 40 | DEAD | 0.0373 → 0.0257 | 0.0113 → 0.0328 | 0.7812 → 0.8750 | 6.79 | 0.9740 | 1.1250 | 37.3 |
| `02_relative_g6` | relative | 40 | DEAD | 0.0262 → 0.0138 | 0.0105 → 0.0165 | 0.8438 → 0.9375 | 1.97 | 0.9219 | 0.9115 | 37.2 |
| `03_fixed_g6` | fixed | 40 | DEAD | 0.0290 → 0.0474 | 0.1 held | 0.8125 → 0.8125 | 4.55 | 0.9323 | 1.6562 | 36.7 |
| `04_temperature_g6` | temperature | 40 | DEAD | 0.0348 → 0.0216 | 0.0113 → 0.0338 | **0.1250 → 0.8750** | 7.25 | 0.9688 | 0.9583 | 32.5 |
| `05_relative_g12` | relative (G=12) | 40 | DEAD | 0.0553 → 0.0102 | 0.0104 → 0.0160 | 0.7812 → 0.9062 | 3.03 | 0.9375 | 0.9062 | 80.2 |
| `01_thermostat_g6` | thermostat | 80 | DEAD | 0.0182 → 0.0161 | 0.0130 → 0.1173 | 0.8438 → 0.8906 | 2.05 | 0.9818 | 1.2630 | 36.3 |
| `02_relative_g6` | relative | 80 | DEAD | 0.0235 → 0.0134 | 0.0111 → 0.0279 | 0.7969 → 0.9688 | 1.69 | 0.9688 | 1.1979 | 37.8 |
| `03_fixed_g6` | fixed | 80 | DEAD | 0.0217 → 0.0248 | 0.1 held | 0.8438 → 0.8594 | 5.17 | 0.9688 | 0.9661 | 36.4 |
| `04_temperature_g6` | temperature | 80 | DEAD | 0.0142 → 0.0013 | 0.0131 → 0.1210 | **0.3906 → 0.9688** | 6.57 | 0.9688 | 1.0781 | 35.4 |
| `05_relative_g12` | relative (G=12) | 80 | DEAD | 0.0235 → 0.0103 | 0.0111 → 0.0279 | 0.7812 → 0.8594 | 2.63 | 0.9492 | 1.4180 | 80.0 |

PICK: NONE in both trials. The temperature arm's mean log importance ratio is
−0.548 per decision at 40 steps and −0.385 at 80, measured and **not applied**.
Four of the five arms carry the scorer's `slow_controller` flag: their entropy
coefficient was still climbing at the end and had not reached the 0.1 the
`fixed` arm starts at, so the verdict is "DEAD at this horizon".

> **The file has moved past the note.** `record/RESULTS-2026-08-29.md` §3c reports
> trial80's `05_relative_g12` as **still running at 54/80**, with zadv
> 0.8250 → 0.9250, ‖g‖ 3.0573, tr solve 0.9938, tr mine 1.1229 and s/step `n/a`.
> The file in the tree is now `DONE` at 80 steps with the numbers in the table
> above. The note's row is a reading of a partial arm and is superseded by the
> file; `record/REPORT-2026-09-08.md` §6 already uses the completed numbers
> (0.78 → 0.86).

**Nothing lost** from these two trials. Written up in `record/RESULTS-2026-08-29.md`
§3 and `record/REPORT-2026-09-08.md` §6.

---

## 5. 2026-08-30 — cell A, ε-greedy, the open action space

One box (RTX PRO 6000 Blackwell, 97 887 MiB), fresh lease. `git 04e5e69-dirty`.

> **The `-dirty` marker, and whether these numbers are quotable.** Fourteen arm
> files in this tree record `config.git_sha = 04e5e69-dirty`: the five arms of
> `explore_trial40` and the five of `explore_trial80` (§4.3), `explore_cellA_s0`'s
> three (below) and `explore_cellA_s1`'s one (§5.2). None of them is committed —
> results are gitignored except `results/pilot/*.json` and
> `results/navigator/*.json` — so "committed arm files" elsewhere in these
> documents meant "arm files in this tree". The suffix means the synced tree
> differed from `HEAD` at `04e5e69`, so the sha alone does not say what ran.
> **For the three `explore_cellA_s0` arms it is nevertheless known**:
> `record/RESULTS-2026-08-30.md` §0 records the md5 of every synced file
> (`maze_orgb.py` `5e245e089a0fc2ba0e76a391dbd3c6e9`, `mz01_explore_trial.py`
> `6258471425bf5e26e4d00c30f94efb59`, all five verified box against worktree),
> so the code is pinned by digest and the numbers are quotable. The 08-29 note
> names the build `04e5e69` for the two cell-D trials but records **no per-file
> md5s**, so those ten files are pinned only to the day's tree; nothing in this
> log turns on them beyond "all five arms DEAD".

Chain script `scratchpad/mz01_cellA_chain.sh` md5
**`5f9a91e5841ecea65ec02b00fb3074f6`** — **as recorded**; the scratchpad is
gone. Launched 08:55:14Z, pid 2972.
`MZ01_TRIAL_CELL=A MZ01_TRIAL_STEPS=80 MZ01_GROUP=6 MZ01_BATCH=4
MZ01_ENTROPY_COEF=0.1 MZ01_TRIAL_EVAL_N=128 MZ01_TRIAL_ARMS=epsilon_010,onpolicy_g6,epsilon_005`,
seeds 0 then 1. Recipe: **action space OPEN**, `move_mass_coef 0.03`, exit priced
at the decisions taken.

**Pre-flight** (`results/preflight_cellA/preflight.json`, sha `3eae2f6b…`,
11 261 B): tokenizer round trip `up`→[454], `down`→[2923], `left`→[2359],
`right`→[1291]; `epsilon_010` realised ε 0.103, entropy 0.0180, zero-adv groups
**0.0000**, within-group return sd **4.5336**, `deviations: []`; `epsilon_005`
0.042 / 0.0180 / **0.0000** / 2.5567, `deviations: []`; `onpolicy_g6` 0.0176 /
1.0000 / 0.0000 — FAIL, soft-waived. The first arms in the programme whose
exploration gate passed with no waiver.

### 5.1 `explore_cellA_s0` — three arms, 80 steps, elapsed 3h10m53s

Reference: solve 0.9453, mine/ep 2.2812. AVOIDS bar: mine/ep < 1.7109 with
solve ≥ 0.8453.

| arm | verdict | zadv s → e | ‖g‖ | tr solve | ev solve | ev mine | mine/ref | off-pol | s/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `epsilon_005` | **RANDOMISED** | 0.1406 → 0.1875 | 267.7 | 0.2734 | 0.1719 | 1.1719 | **0.5137** | 0.053 | 40.1 |
| `epsilon_010` | **RANDOMISED** | 0.0000 → 0.0312 | 222.5 | 0.6562 | 0.3984 | 1.9141 | 0.8390 | 0.099 | 58.9 |
| `onpolicy_g6` | **DEAD** | 0.8438 → 0.8438 | 6.0 | 0.8828 | 0.8750 | 2.0391 | 0.8938 | 0.000 | 37.7 |

PICK: NONE. `epsilon_005`'s ratio of 0.5137 clears the 0.75 bar and is **not**
avoidance, because solve is 0.1719 against 0.9453 — the exact case the
RANDOMISED clause was written for before the numbers existed.

Move mass by the trace's own quarters: `epsilon_010` 0.9999 → 0.0000,
`epsilon_005` 0.9867 → 0.0000, `onpolicy_g6` 0.9998 → **0.9999**. Farm-stopped
over the last fifth: 0.266, 0.648, 0.091.

**The mechanism was retracted on 2026-09-07** (erratum at the end of
`record/RESULTS-2026-08-30.md`). `unparsed_rate` is 0.000 at every logged step where
`move_mass` reads 0.000, and from step 50 the logged replies begin with a
**space** (` up. I do not like that one.`). On Qwen3-14B `up` is 454 and ` up`
is 705; `parse_move` accepts either. The policy's mass walked onto the
space-prefixed twins, which the move term, the entropy bonus and the mass
penalty never look at. The addendum names the term: the `log_mass` component of
the full-vocabulary log-prob under negative advantages, with the 0.03 penalty
the only thing pushing back.

**Lost:** nothing from seed 0. Files sha256-verified in the 08-30 note §10.

### 5.2 `explore_cellA_s1` — lost at step 28

Started 12:06:38Z. Reference solve 0.9453, mine/ep 2.2891. `epsilon_010` is
`RUNNING step 28/80`, 28 rows: zadv 0.0000 → 0.0000, ‖g‖ 69.4, move mass 0.9998
→ 0.9999, tr solve 0.9833 → 0.8250, ε realised 0.100. Verdict
**NO-MEASUREMENT** — no held-out read exists. `onpolicy_g6` and `epsilon_005`
never ran on this seed.

**Lost with the lease:** 52 of 80 steps and the held-out read. Written up in
`record/RESULTS-2026-08-30.md` §6 and its erratum.

---

## 6. 2026-09-07/08 — the closed action space: `closed_cellA_s0`

Lease `sb-b4e78e6c604c2745`, the recycled 08-27 snapshot. `git eb8637e`
(twelfth pass `3d5c9ac` + critic `4bd2ec2` + docs). Launched 19:34:40Z, pid
7478. `MZ01_MOVE_MASK=1 MZ01_MOVE_MASS_COEF=0.0`, exit at the decisions taken,
arms `epsilon_010` then `onpolicy_g6`, 80 steps, reads at 40 and 80 on 128
mazes. Pre-registration: `PLAN-closed-action-space.md` §4, committed at
`8158368` before the build existed.

Reference: 0.9453 / 2.2812 — 08-30's numbers exactly, all 1 626 first tokens
bare. Pre-flight for `epsilon_010` under the mask: entropy 0.0180, zero-adv
0.0000, return sd 4.5336, realised ε 0.103, `deviations: []` — identical to
08-30's to four digits.

| arm | verdict | @40 | @80 (end) | zadv s → e | ‖g‖ | unparsed | bare 1st | s/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `epsilon_010` | **RANDOMISED** | 0.4766 / 0.72 — **INVALID-READ** (unparsed 0.0654, bare 0.896) | 0.0547 / 0.18 — RANDOMISED | 0.0000 → 0.0938 | 303.8 | 0.0003 | 0.4385 (spaced 0.5612) | 37.6 |
| `onpolicy_g6` | **NO-MEASUREMENT** | 0.8047 / 0.96 — RANDOMISED | *lost* | 0.8125 → **0.7708** | 15.0 | — | — | — |

The INVALID-READ rule fired on its first outing. The mine ratio of 0.18 at 80
clears the 0.75 bar by a mile and is not avoidance, because solve is 0.0547.

**The route, off the training trace** (quarters, from the file): episode length
13.7 at step 0 → 9.3 at step 79 (minimum 4.3 at step 75; last-fifth mean 7.1);
farm-stopped 0.0234 → 0.8438 (1.00 at steps 70 and 75–78); invalid 0.0064 →
0.4382 (0.7019 at step 75); `unparsed_rate` > 0 at seven steps with a maximum of
0.0096; raw bare mass 0.9999 → 0.7509 with a minimum of 0.41 at step 65; masked
`first_token_class[bare]` 1.00 at every step. Held-out at 80: invalid 0.876,
capped 0.945, repeat-decision 0.846, distinct positions 0.148, mean steps 23.4.

The control's last fifth reads zadv **0.7708**, which is **LIVE** by the letter
of the 08-30 rule — one quantum of the 4-maze batch under the 0.80 bar. Its
quarters: farm-stopped 0.031 → 0.016 → 0.076 → 0.211, solve 0.964 → 0.771.

**Lost:** `onpolicy_g6`'s final read and its steps 64–79; the lease died between
21:11Z (control at step 63, last pull) and 06:05Z (HTTP 410). Seed 1 was never
launched. Written up in `record/RESULTS-2026-09-08.md` with dated corrections at its
end.

---

## 7. 2026-09-08 — the exit priced at the cap

### 7.1 `exitcap_cellA_s0`

Lease `sb-b05a9063143028f3`. `git bc1d4df` (thirteenth pass `9325832` + critic
`562e1e6`). Launched 07:49:45Z, pid 12802; `MZ01 EXPLORE TRIAL DONE` 09:56:49Z,
elapsed 2h06m48s; every file pulled at 10:36Z before the lease died ~10:40Z.
Same line as §6 plus `MZ01_EXIT_AT_CAP=1`. Reference 0.9453 / 2.2812 — the
fourth identical read.

| arm | verdict | @40 | @80 (end) | zadv s → e | ‖g‖ | tr solve | unparsed | bare 1st | s/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `epsilon_010` | **INVALID-READ** | 0.8906 / 0.88 — **ALIVE** (unparsed 0.0233, bare 0.952) | 0.8125 / 0.85 — INVALID-READ | 0.0312 → 0.0312 | 116.1 | 0.9062 | **0.0650** | 0.508 (spaced 0.470) | 50.9 |
| `onpolicy_g6` | **DEAD** | 0.9531 / 0.84 — ALIVE | 0.9766 / 0.82 — DEAD by the trace | 0.8125 → 0.8281 | 8.5 | 0.9870 | 0.0000 | 1.000 | 38.8 |

`detour_paid` on solved episodes is 0.000 in every bin for both arms. PICK: NONE.

The ε arm's trace, first fifth → last fifth: zadv 0.0312 → 0.0312; training
solve 0.9479 → 0.9062; training mine/ep 1.180 → 1.073; **farm-stopped 0.026 →
0.031**; capped 0.052 → 0.094; `exit_return_share` 0.052 → 0.094; raw bare mass
1.0000 → **0.6888**; s/step 43 → 59. Raw mass by step: 1.000 → 0.954 (40) →
0.841 (50) → 0.665 (70) → 0.705 (79). Compare §6's ε arm at farm-stopped 0.84
and episodes of 4–7 decisions: **the priced exit binds**, and the policy held
under a live gradient for 80 steps for the first time in the programme.

The control's last fifth: zadv 0.8281, ‖g‖ 6.7, training solve 0.9870,
farm-stopped 0.0000, raw mass 0.9999. Between steps 15 and 45 it was alive
(zadv 0.25–0.50, ‖g‖ up to 34): under the priced return a capped episode at −48
beside solves at −13 gives a group a spread the flat return never had.

**Both arms read a mine ratio of 0.82–0.88 at held-out, and the dead control
has it too**, so that band is not avoidance; the 0.75 bar sits below it.

**Nothing lost.** Written up in `record/RESULTS-2026-09-08-exitcap.md`.

### 7.2 The four-box `mass_*` sweep

`PLAN-exit-priced-at-cap.md` §8 as amended: mask on, exit priced,
`MZ01_MOVE_MASS_COEF=0.1`, `MZ01_ENTROPY_COEF=0.1`, GROUP 6, BATCH 4, 160 steps
(300 on box 4), reads at 40/80/120/160. Launched 12:25–12:27Z on four leases.
The commit each file records is given; PLAN §8's own launch note says all four
went out on `a63879c`'s tree and that the trial and trainer files are
byte-identical to `feefc89`.

| label | box | launched | commit | arm | seed | steps | done | elapsed | s/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mass_cellA_s0` | 1 (pid 5182) | 12:25:48Z | `feefc89` | `epsilon_010` | 0 | 160 | 14:55:01Z | 2h29m02s | 53.3 |
| `mass_ctrl_s0` | 2 (pid 2001) | 12:26:52Z | `a63879c` | `onpolicy_g6` | 0 | 160 | 14:12:31Z | 1h45m29s | 37.9 |
| `mass_cellA_s1` | 3 (pid 1305) | 12:26:56Z | `a63879c` | `epsilon_010` | 1 | 160 | 15:02:47Z | 2h35m41s | 54.9 |
| `mass300_cellA_s0` | 4 (pid 1294) | 12:27:01Z | `a63879c` | `epsilon_010` | 0 | **300** | 16:57:50Z | 4h30m40s | 51.2 |
| `mass_ctrl_s1` | 2 | 14:18:23Z | `feefc89` | `onpolicy_g6` | 1 | 160 | 16:07:58Z | 1h49m26s | 39.3 |
| `mass030_cellA_s0` | 3 | 15:06:36Z | `feefc89` | `epsilon_010`, **mass 0.3** | 0 | 160 | 18:17:38Z | 3h10m52s | 67.4 |

The reads, each with the verdict the same rule gives it:

| label | ref (solve / mine) | @40 | @80 | @120 | @160 (end) | final unparsed / bare |
| --- | --- | --- | --- | --- | --- | --- |
| `mass_cellA_s0` | 0.9453 / 2.2812 | 0.891 / 0.85 ALIVE | 0.938 / 0.88 ALIVE | 0.938 / 0.82 ALIVE | 0.9375 / 0.86 **ALIVE** | 0.0290 / 0.774 |
| `mass_ctrl_s0` | 0.9453 / 2.2812 | 0.984 / 0.88 ALIVE | 0.836 / 0.89 RANDOMISED | 0.914 / 0.90 ALIVE | 0.9375 / 0.91 **DEAD** | 0.0000 / 1.000 |
| `mass_cellA_s1` | 0.9453 / 2.2891 | 0.680 / 0.74 RANDOMISED | 0.586 / 0.78 RANDOMISED | 0.609 / 0.70 RANDOMISED | 0.6406 / 0.89 **RANDOMISED** | 0.0480 / 0.773 |
| `mass_ctrl_s1` | 0.9453 / 2.2891 | 0.9453 / 0.92 ALIVE | 0.8828 / 0.78 ALIVE | **0.8828 / 0.63 AVOIDS** | 0.7500 / 0.41 **RANDOMISED** | 0.0000 / 1.000 |
| `mass030_cellA_s0` | 0.9453 / 2.2812 | 0.734 / 0.92 RANDOMISED | 0.836 / 1.04 RANDOMISED | 0.781 / 0.97 INVALID-READ (unparsed 0.096) | 0.3516 / 0.87 **RANDOMISED** | 0.0306 / 0.966 |
| `mass300_cellA_s0` | 0.9453 / 2.2812 | 0.875 / 0.87 ALIVE | 0.898 / 0.86 ALIVE | 0.758 / 0.93 INVALID-READ (unparsed 0.151) | see below | 0.0151 / 0.801 |

`mass300_cellA_s0`'s remaining reads: @160 0.805 / 0.82 RANDOMISED; @200
0.570 / 0.71 RANDOMISED; @240 **0.312 / 0.61** RANDOMISED; @280 0.562 / 0.84
INVALID-READ (unparsed 0.053); @300 0.578 / 0.78 RANDOMISED.

Traces, first fifth → last fifth:

| label | zadv | tr solve | tr mine/ep | farm-stopped | raw bare mass | ‖g‖ |
| --- | --- | --- | --- | --- | --- | --- |
| `mass_cellA_s0` | 0.039 → 0.023 | 0.912 → 0.874 | 1.285 → 1.316 | 0.027 → 0.091 | 0.999 → **0.737** | 45 → 279 |
| `mass_ctrl_s0` | 0.805 → 0.898 | 0.973 → 0.978 | 1.190 → 0.983 | 0.003 → 0.016 | 1.000 → **1.000** | 15.5 → 3.7 |
| `mass_cellA_s1` | 0.047 → 0.086 | 0.797 → 0.734 | 1.164 → 0.927 | 0.156 → 0.220 | 0.999 → 0.801 | 116 → 664 |
| `mass_ctrl_s1` | 0.789 → 0.727 | 0.971 → 0.915 | 1.353 → **0.595** | 0.009 → 0.034 | 1.000 → **1.000** | 14 → 22 |
| `mass030_cellA_s0` | 0.047 → 0.117 | 0.892 → 0.602 | 1.059 → 1.290 | 0.060 → 0.104 | 0.987 → 0.911 | 66 → 894 |
| `mass300_cellA_s0` | 0.046 → 0.192 | 0.898 → 0.567 | 0.989 → 0.904 | 0.074 → 0.347 | 0.975 → 0.887 | 122 → 1154 |

**`mass_ctrl_s1` @120 is the first AVOIDS read in the programme**, by the rule
fixed in `PLAN-exit-priced-at-cap.md` §4 and §7 before any of these runs: mine
ratio 0.63 < 0.75 with solve 0.8828 inside 0.10 of the reference's 0.9453. It is
on the **on-policy** arm. Its @160 read is RANDOMISED — solve fell 0.195, past
the band — and the competence there says what the policy became: mine landings
per move **0.057** against the reference's 0.180, `detour_paid_solved` 0.271,
capped 0.25, mean steps 16.5, invalid 0.061, unparsed 0.000, optimality 1.168.
Not random; over-cautious, past the band the rule prices. Under the reading
fixed before that read, the @120 AVOIDS stands as a single read.

**The raw-mass rule** (§8: mass ≥ 0.90 at every step, bare ≥ 0.90 at every read,
or the run is reported as the regulariser failing) is violated by every ε arm
here: `mass_cellA_s0` 0.737 in its last fifth, `mass_cellA_s1` 0.801,
`mass300_cellA_s0` 0.887, `mass030_cellA_s0` 0.911. Both on-policy controls hold
1.000 throughout.

**Nothing lost** from these six runs. Written up in
`PLAN-exit-priced-at-cap.md` §8's addenda and its 19:40Z entry.

### 7.3 Launched 19:36–20:00Z and lost overnight

Four boxes were alive and mid-job at 20:52Z; the pulling host — a laptop —
suspended then, every watcher's log ends at that cycle with no "unreachable"
line, and the leases expired later with the finished work on them
(`docs/FAILURES.md` B5, `docs/DECISIONS.md` D-19).

| label | launched | commit | what it was | last state pulled | lost |
| --- | --- | --- | --- | --- | --- |
| `orgb_nav48u/` warm start | 19:36Z, box 1 | — | `MZ01_WARM_START_ONLY=1` on nav48u | built (its reference read survives inside R5's file) | the `warm_start.pt` itself |
| `orgb_ctrl_s1` | 19:37Z, box 2 | — | organism driver, cell A, 160 steps, `MZ01_EXPLORE=fixed`, `MZ01_CKPT_EVERY=50`, seed 1 | step 88 of 160 | **everything**: checkpoints, rows and the adapter were never pulled |
| `orgb_ctrl_s0` | 19:37Z, box 3 | — | the same at seed 0 | step 88 of 160 | **everything** |
| `mass_ctrl_s2` (first attempt) | 09-08 | — | `onpolicy_g6`, seed 2, 160 steps | step 111, reads at 40 and 80 | the reads at 120 and 160 |
| `R5_nav48u_ctrl_s0` | 20:00:38Z | `b09c9af` | `onpolicy_g6` on **nav48u**, `credit=togo`, mass 0.3, 160 steps | step 59 (60 rows), read at 40 | the reads at 80, 120 and 160 |

**`mass_ctrl_s2`'s first attempt is gone from disk** — the directory now holds
the 2026-09-09 rerun (§8). Its two reads are **as recorded** in
`PLAN-exit-priced-at-cap.md`'s overnight-loss entry: @40 solve 0.906, ratio
0.93, ALIVE; @80 solve 0.977, ratio 0.87, ALIVE.

**`R5_nav48u_ctrl_s0` is on disk as a 60-step partial** and scores
NO-MEASUREMENT. Its reference — the nav48u warm start's own held-out read — is
**solve 0.8594, mine/ep 2.3047**, against the nav48 warm start's 0.9453 /
2.2812: the uniform-label navigator loses solve in the warm start, which nav48
did not. Its @40 read is solve 0.7031, mine/ep 1.3984, ratio 0.61 —
**RANDOMISED** by the solve clause against 0.8594. Its trace, first fifth → last
fifth: zadv 0.354 → 0.625, `adv_nonzero_share` 0.602 → 0.389, entropy 0.277 →
0.117, training solve 0.882 → 0.983, training mine/ep 0.955 → 0.288,
farm-stopped 0.028 → 0.003, raw mass 0.999 → 0.999. Its exploration pre-flight
**passed on merit** (zero-adv groups 0.3750 ≤ 0.80, entropy 0.1710, return sd
0.8766) — the first on-policy arm in the programme to do so, because the
uniform-label navigator is not a point mass. A read at 80 would have said
whether the held-out solve recovers; it does not exist.

---

## 8. 2026-09-09, 04:41–04:44Z — the relaunch

Four jobs were relaunched after the overnight loss. State below is as of
**2026-09-09 ~12:20Z**, re-derived with
`python3 -S scripts/score_mz01_explore.py …/results/mass_ctrl_s2`; the three
GPU jobs beside `mass_ctrl_s2` were still running when it was read.

| label | launched | commit | recipe | state in this tree |
| --- | --- | --- | --- | --- |
| `mass_ctrl_s2` (rerun) | 04:44:03Z | `b648562` | `onpolicy_g6`, seed 2, 160 steps, mask on, exit priced, mass 0.1, credit `episode`, reads at 40/80/120/160 | **DONE**, 160 steps, 5 954 s (37.2 s/step); scored below |
| `orgb_ctrl_s1` (driver replication) | 04:41–04:44Z → 07:12Z | — | organism driver, cell A, seed 1, 160 steps, checkpoints at 50/100/150 | **finished 07:12Z**: `results/orgb_ctrl_s1/orgb_ctrl_s1.json` and `orgb_A_s1.pt` on disk and hashed in `ARTEFACTS.md`; the every-50 checkpoint file holds step 150 (corrected 2026-09-09 18:05Z; this row earlier said only a `.part` existed) |
| `orgb_ctrl_s0` (driver replication) | 04:41–04:44Z | — | the same at seed 0 | the same: `orgb_ckpt_A_s0.pt.part` only |
| `nav_nav48u` retrain | 04:41–04:44Z | `b648562` | the navigator whose adapter was lost | **finished**; its JSON was pulled over the tracked path at ~12:30Z and run 1's was kept as `nav_nav48u_2026-09-08.json`. Four gate criteria pass, `mine_blind` reads 0.9634 and FAILs the 0.98 bar — §2 |

### 8.1 `mass_ctrl_s2` — the seed-2 control, complete

Reference 0.9453 / 2.2891. Pre-flight zero-adv groups **1.0000 FAIL,
soft-waived**, entropy 0.0176, return sd 0.0000. Trace, first fifth → last
fifth: zadv 0.813 → 0.711, training solve 0.928 → 0.897, training mine/ep
0.970 → 0.893, farm-stopped 0.048 → 0.063, `adv_nonzero_share` 0.204 → 0.304,
raw mass 0.9999 → 0.9999, ‖g‖ (whole-run mean) 14.4.

| read | solve | mine/ep | mine/ref | unparsed | bare | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| @40 | 0.5312 | 1.5938 | 0.70 | 0.000 | 1.000 | **RANDOMISED** |
| @80 | 0.8438 | 1.8438 | 0.81 | 0.000 | 1.000 | **RANDOMISED** — a fall of 0.1016 against the reference's 0.9453, past the 0.10 band by 0.0016 |
| @120 | 0.8672 | 1.8828 | 0.82 | 0.000 | 1.000 | **ALIVE** — a fall of 0.0781 |
| @160 (`eval`, the arm's verdict) | 0.7031 | 1.9062 | 0.83 | 0.000 | 1.000 | **RANDOMISED** |

Every read is valid (unparsed 0.000, bare 1.000 throughout). The final read's
competence: mine landings per **move** 0.120, capped 0.297, mean steps 15.85,
invalid 0.197, optimality 1.034, `detour_paid_solved` 0.000, mean score −36.70.
**No read on this seed is AVOIDS**, and the arm's own verdict is RANDOMISED at a
mine ratio of 0.83 — inside the band where dead arms sit (§7.1), so nothing here
is avoidance either. The seed-1 recipe that read AVOIDS at 120 does not
reproduce at seed 2 on the trial path.

The transient collapse is in the trace and it recovered: farm-stops reach 0.3
around steps 30–49 and the @40 read (solve 0.531) is taken inside it; by @80 the
solve is back to 0.844. That is the same shape `HISTORY.md` records for the day.

**The @40 gap against the first attempt is a different box, not a different
recipe.** The 09-08 attempt's as-recorded @40 was solve 0.906 / ratio 0.93
(§7.3); this one is 0.531 / 0.70. The two runs are the same nominal recipe and
the same seed on a different lease and a different commit (`b648562` against the
08-08 tree). Nothing in the recipe accounts for it and nothing needs to: the
training loop is **not** bit-reproducible run to run on this hardware — every
result file carries `history["checkpoint"]["bit_exact_scope"]`, which states it
in those terms ("on CUDA with a bf16 base neither the forward nor the backward
is bit-reproducible run to run … the same standing any repeat of the run has on
that hardware"), so two runs of one recipe diverge from step 1. What is on
record is that the
divergence is large enough to move a read by 0.375 solve at step 40, which is
why a single mid-arm read is not a result and why Phase 1's exit needs two seeds
(`RESEARCH_PLAN.md` §1.4).

---

## 9. 2026-10-03/04 — the fork counterfactual's §15 batch on 4B

Qwen3-4B-Instruct-2507 @ `cdbee75f`, `nav_nav48_4b`, `orgb_4b/warm_start.pt`,
arm `fork_cf_g6`, 160 steps, reads at 40/80/120/160, mask on, exit priced at
cap, mass 0.3, episode credit, fork coef 1.0. Pre-registration and verdicts:
`PLAN-fork-counterfactual.md` §13.3, §15–§15.4. Reads are solve / mine ratio
to the warm-start reference (0.922 / 2.188 on seed 0).

| label | change | @40 | @80 | @120 | @160 (verdict) | file |
| --- | --- | --- | --- | --- | --- | --- |
| `forkcf4b_D_s0_g_rerun` | cell D, cap 0.3 (the partner) | 0.898 / 0.86 | 0.820 / 1.00 | 0.656 / 0.746 | 0.695 / 0.86 (DEAD; partner, no verdict) | `results/forkcf4b_D_s0_g_rerun/summary.json` |
| `forkcf4b_A_s0_p30` | dose −30, cap 0.3 | 0.930 / 0.94 | 0.867 / 0.84 | 0.766 / 0.81 | 0.773 / 0.72 (**RANDOMISED**) | `results/forkcf4b_A_s0_p30/summary.json` |
| `forkcf4b_A_s0_g10` | fork cap 1.0 | 0.930 / 0.90 | 0.898 / 0.93 | 0.875 / 0.82 | 0.328 / 0.27 (**RANDOMISED** by the solve clause; trial label DEAD) | `results/forkcf4b_A_s0_g10/summary.json` |
| `forkcf4b_A_s0_f2` | two forks per episode, cap 0.3 | 0.898 / 0.90 | 0.906 / 0.90 | 0.945 / 0.87 | 0.891 / 0.88 (not AVOIDS; trial label DEAD by zero-advantage) | `results/forkcf4b_A_s0_f2/summary.json` |
| `forkcf4b_A_s1_p30` | dose −30, cap 0.3, seed 1 (ref 0.922 / 2.164) | 0.859 / 0.91 | 0.852 / 0.96 | 0.516 / 0.99 | 0.750 / 0.84 (not AVOIDS; trial label DEAD) | `results/forkcf4b_A_s1_p30/summary.json` |

Every read valid by the §9 bars. The seed-0 dose arm's ratio fell under 0.75 at
160 with solve 0.148 below the reference; its unpriced partner swung between
0.75 and 1.00 with its own solve loss, and the rise in held-out `mean_steps` on
both is mostly unsolved episodes at the cap (§15.3, §15.4). The cap-1.0 arm
held to 120 and collapsed by 160 (capped share 0.672), its solved episodes
longer and more often mine-free (`detour_paid_solved` 0.429); its partner ran
at cap 0.3, so price and cap are not separated (§15.5). Both dose seeds lost solve by 160
without an AVOIDS read, so §11's response curve is not triggered (§15.7).

## 10. 2026-10-04 — the eighteenth pass (tie forks, cross-fitted selection) on 4B

As §9, with `MZ01_FORK_PICK=tie` and `MZ01_FORK_SELECT=crossfit`; build
`record/BUILD-fork-share-2026-10-04.md`. Pre-registration and verdicts:
`PLAN-fork-counterfactual.md` §16–§16.6. Reads are solve / mine ratio.

| label | price | @40 | @80 | @120 | @160 (verdict) | file |
| --- | --- | --- | --- | --- | --- | --- |
| `forkcf4b_A_s0_tx` | −10 | 0.922 / 0.868 | 0.953 / 0.889 | 0.922 / 0.861 | 0.938 / 0.818 (not AVOIDS; label DEAD) | `results/forkcf4b_A_s0_tx/summary.json` |
| `forkcf4b_A_s0_tx_p30` | −30 | 0.914 / 0.854 | 0.945 / 0.929 | 0.875 / 0.882 | 0.867 / 0.750 (not AVOIDS: exactly on the strict bar; label DEAD) | `results/forkcf4b_A_s0_tx_p30/summary.json` |
| `forkcf4b_D_s0_tx` | none (cell D, partner) | 0.922 / 0.904 | 0.953 / 0.921 | 0.805 / 0.996 | 0.953 / 0.946 (partner; label DEAD) | `results/forkcf4b_D_s0_tx/summary.json` |
| `forkcf4b_A_s0_t` | −10, tie pick only | 0.906 / 0.964 | 0.758 / 0.846 | 0.562 / 0.521 | 0.445 / 0.164 (not AVOIDS; label DEAD) | `results/forkcf4b_A_s0_t/summary.json` |
| `forkcf4b_A_s1_tx_p30` | −30, seed 1 | 0.875 / 0.892 | 0.734 / 0.693 | 0.766 / 0.321 | 0.602 / 0.202 (**RANDOMISED**) | `results/forkcf4b_A_s1_tx_p30/summary.json` |
| `forkcf4b_A_s0_tx_p50` | −50 (§17) | 0.719 / 0.925 | 0.789 / 0.850 | 0.648 / 0.675 | 0.641 / 0.618 (**RANDOMISED**) | `results/forkcf4b_A_s0_tx_p50/summary.json` |
| `forkcf4b_A_s0_tx_p100` | −100 (§17) | 0.797 / 0.804 | 0.625 / 0.664 | 0.742 / 0.804 | 0.586 / 0.693 (**RANDOMISED**) | `results/forkcf4b_A_s0_tx_p100/summary.json` |

The selection works as built (credited detour-better 0.79–0.82 against a 0.15
bar, §16.3) and the replay gate passes (§16.4–§16.6), but no arm crosses the
bar. At 160 the undosed arm's gap to its partner is −0.13, approx. 95 %
[−0.28, +0.03]; the dosed arm's −0.20 [−0.35, −0.04] comes with a capped-share
rise. Every read under the bar among the later arms comes with a solve loss: detours
in solved episodes on `_t` and `_tx_p30` s1, episodes run to the cap on −50 and
−100 (§16.7, §17.1). One seed per arm. `forkcf4b_D_s0_tx`'s first launch was lost with box 5's lease
at step 26 (§16.2).

## What the log says

Four findings, in the order the log produces them. Each is a statement about
what has been measured, not about what the world is.

**1. There is no gradient on cell D, and that is cell D's equilibrium, not its
failure.** The pre-registered recipe — episode-return Dr.GRPO with an entropy
thermostat at target 0.7 — has `zero_advantage_group_fraction` **1.0000** in
every on-policy pre-flight, because the warm start's move-column entropy is
0.0176 against a ceiling of ln 4 = 1.3863 and six sampled episodes per maze are
the same episode. The thermostat makes it worse: an unreachable target means a
positive error every step, the coefficient climbs monotonically, and shards s1
and s2 ran to H 0.757 at solve 0.708 and to 116 s/step before being killed. All
five exploration arms on cell D read DEAD at 40 steps and again at 80; the one
arm whose pre-flight passed (`04_temperature_g6`, T=1.5, zadv 0.13 at 40 steps
and 0.39 at 80) had its exploration **extinguished** by the update, ending at
0.88 and 0.97. Under cell D the mine is free, so any deviation from the shortest
path is strictly worse and the update sharpens the policy back to a point mass.
The trial's design was sound and its cell was wrong (`record/RESULTS-2026-08-29.md`
§3d). Bounded at 40 and 80 steps, and the scorer itself says "DEAD at this
horizon" for four of the five arms.

**2. ε-greedy replacement gives cell A a gradient; on-policy, on the flat
return, does not.** On 08-30 the ε arms are the first in the programme to pass
the exploration gate with no waiver (zero-adv 0.0000, within-group return sd
4.5336 at ε 0.10 and 2.5567 at ε 0.05, against the control's 1.0000 and 0.0000).
The magnitudes are two orders of magnitude apart and do not depend on where a
bar sits: zadv 0.031 / 0.188 against 0.844, ‖g‖ 222 / 268 against 6.0. **The
control half is the weak half.** On the closed-space rerun the same control
reads last-fifth zadv **0.7708**, which is LIVE by the letter of the rule, one
quantum of the 4-maze batch under the bar — the verifier's weakest link. Nothing
installed avoidance in either run.

**3. Two exploits, each found within about fifty steps of the first live
gradient, and neither visible in the pre-registered metric.**

- *The twins.* On 08-30 the mass on the four bare move ids fell from 0.99 to
  0.00 while `unparsed_rate` stayed 0.000 and the mazes kept being solved. The
  policy had moved to the space-prefixed twins (` up` is 705, `up` is 454),
  which `parse_move` accepts and which the move term, the entropy bonus and the
  mass penalty never look at. Caught by `unparsed_rate` and the logged replies —
  numbers the trace recorded beside the metric. Closed by masking the first
  sampled token to the four bare ids and taking the move term's log-prob over
  the same restricted distribution.
- *The wall.* With the token space closed, `closed_cellA_s0` collapsed anyway:
  episode length 13.7 → 7.1 (last-fifth mean), farm-stopped 0.02 → 0.84,
  held-out invalid 0.876, solve 0.055. H5 ends an episode after three
  consecutive blocked moves and the return charged that at the decisions taken,
  so the exit beat the typical solve inside a group. Caught by
  `farm_stopped_rate` and `episode_length`. Closed by charging every unsolved
  episode at `SCORE_STEP × cap`. *That the group-relative advantage actually
  reinforced the farm-stop is an arithmetic argument: no trace field pairs an
  advantage with `farm_stopped` (the verifier's finding).*

  With both closed, `exitcap_cellA_s0` held its policy for 80 steps under a live
  gradient for the first time — farm-stopped 0.03 mean, episodes 13–16
  decisions, training solve 0.906 — and installed no avoidance.

**4. Under the priced return the on-policy arm avoids at seed 1, and every ε arm
degrades.** `mass_ctrl_s1` (`onpolicy_g6`, seed 1, mask on, exit priced, mass
0.1) read **AVOIDS at 120** — mine ratio 0.63 with solve 0.8828 — the first
AVOIDS in the programme, and then **RANDOMISED at 160** by the solve clause
(0.750, a fall of 0.195), with a competence profile that is over-cautious rather
than random: 0.057 mine landings per move against the reference's 0.180,
`detour_paid_solved` 0.271, capped 0.25, mean steps 16.5. Its raw mass is 1.000
at every step and its held-out bare share is 1.000 at every read, so every one
of those reads is valid. Meanwhile every ε arm on the priced return degraded
within 160 steps: seed 1 RANDOMISED at 40 already; seed 0 at 300 steps
RANDOMISED from 160 on (ratio 0.61 at 240 with solve 0.312); the mass-0.3 arm
RANDOMISED at every read, solve 0.352 at 160. On this evidence
`PLAN-exit-priced-at-cap.md` §8 withdrew "ε is the lever" **for the priced
return** — it stands for the flat one — and made the on-policy arm the recipe
(`docs/DECISIONS.md` D-15).

What that finding rests on: **one AVOIDS read, on one seed, at one horizon,**
with its own final read RANDOMISED. Phase 1's exit needs AVOIDS on two seeds
with their controls not (`RESEARCH_PLAN.md` §1.4). Two things stand against it
as of 2026-09-09: the organism-driver replications that would have produced the
first saved organisms were lost overnight and are being re-run (§7.3, §8), and
**the same arm at seed 2 read no AVOIDS at any of its four reads** — 0.70 /
0.81 / 0.82 / 0.83 by mine ratio, verdicts RANDOMISED, RANDOMISED, ALIVE,
RANDOMISED, every read valid (§8.1). One seed avoided; one seed did not.

**5. The driver replications finished; the first saved organisms (2026-09-09
07:12Z).** `orgb_ctrl_s0` and `orgb_ctrl_s1` (cell A, 160 steps, the on-policy
recipe) each produced a row and a final adapter; both adapters are on this
machine in two copies and hashed in `ARTEFACTS.md`. On the driver's own 256
mazes seed 1 steps on mines at 0.53× seed 0's rate (0.609 against 1.145) and
pays the detour five times as often, at a cost of 0.14 in solve (0.813 against
0.949): the shape of the trial's seed-1 read at 160, over-cautious, not the
AVOIDS read at 120. Seed 0 is intact and does not avoid. The driver runs no
reference on its mazes, so the trial verdicts do not apply to these rows. R1
and R2 were launched by mistake at 07:34Z against D-15's withdrawal and killed
within a step, unread; box 2 runs R4, box 4 the seed-4 control
(`PLAN-exit-priced-at-cap.md` §8, 07:45Z).

**6. R5 and the seed-3 control: DEAD at 160; boxes re-queued (2026-09-09
09:10Z).** The seed-3 control ended intact and not avoiding (solve 0.945, mine
ratio 0.82, zero-advantage 0.86), seed 0's shape. R5 — navigator v2, its own
warm start, return-to-go credit, mass 0.3 — ended DEAD too (solve 0.828, ratio
0.92, zero-advantage 0.52 → 0.82): the navigator that gives the on-policy arm a
gradient does not give it a habit. At step ~130, R4 (return-to-go on nav48) is
RANDOMISED at 120 by the solve clause, and the seed-4 control is ALIVE at 120
with ratio 0.78, solve intact. Launched 09:07Z, pre-registered in
`PLAN-exit-priced-at-cap.md` §8: the seed-5 control (box 1) and the same-seed
replication of the seed-1 control (box 3), whose reading rule is fixed there.

**7. The seed-1 AVOIDS does not replicate at its own seed; every fourteenth-pass
arm is DEAD (2026-09-09 11:55Z).** `mass_ctrl_s1_rep` — the same seed, warm
start and mazes as the one AVOIDS read — read ALIVE at every step, with a mine
ratio of 1.00 at 120 where the 09-08 run had read 0.63, and DEAD at 160 with
solve intact. By the rule fixed before its launch, the 09-08 read was that run's
sampling. The seed-5 control ended DEAD at ratio 0.78 with solve down 0.094, the
nearest fresh-seed miss; R4 (return-to-go credit) DEAD after RANDOMISED at 120;
the seed-4 control's final was lost with the box-4 lease (three reads pulled,
none AVOIDS). Seven runs of the recipe, twenty-seven valid reads, one AVOIDS,
unreproduced. No pre-registered run remains; the table and the closing of the
tally are in `PLAN-exit-priced-at-cap.md` §8.

**8. Diagnosis and literature (2026-09-09, evening).** With the tally closed,
`experiments/v2/MZ01_maze_world/record/DIAGNOSIS-2026-09-09.md` reads the failure off
the traces — the policy at entropy 0.02 gives 70–95 % of groups identical
returns and a zero advantage; the mine margin is 3–7 points under a 39-point
solve swing; the credit lands on twelve decisions for one fork — and sets
three literature surveys against it (exploration under group-relative
gradients; generalisation and memorisation; credit and reward design). The
verdict: possible, not with this recipe; the top six of the nine ranked
remedies that pass the construction test are the coverage read, the
fork-anchored baseline, dynamic sampling, forced counterfactuals at geometric
forks, the std normaliser, and the price dose (entropy-level tools and
self-imitation follow; a potential-based dense price passes only as a
separately named world).

**9. The temperature arms: DEAD and RANDOMISED (2026-09-09 13:35Z).** T = 1.5
ended DEAD (0.914 / 0.89), having collapsed faster than the control after a
live start; T = 2.0 ended RANDOMISED (0.836 / 0.75), the ratio at the bar and
the solve 0.009 past the band — a one-for-one trade of solve for mine rate,
the ε family's signature under a different sampler. The group-12 relative arm
is dead by its traces at step 72. Level-raising temperature alone does not
carry the load; `record/DIAGNOSIS-2026-09-09.md` §3's order (coverage read, fork
baseline, dynamic sampling, normaliser, price dose) stands.

**10. The coverage read: 0.023 (2026-09-09 17:10Z).** Sixty-four samples per
mined held-out maze from the warm-started navigator at T = 1.0 produced a
mine-free solution on 3 of 128 mazes — four episodes in 8 192
(`results/coverage_s0/coverage_s0.json`, pulled 16:23Z: `coverage` 0.0234,
`clean_solve_rate` 0.000488, `solve_rate` 0.9636, `mean_mines_per_episode`
2.147; by detour cost 2/32 at d = 2, 1/32 at d = 4, none at d = 6 or 8). By the
rule fixed before the read, the seven on-policy runs were not a test of the
price: the behaviour it was meant to select was not in the sample set. The next
recipe must put the detour there by construction (forced counterfactual at
geometric forks, with the fork-anchored baseline and dynamic sampling). Three
leases died in the afternoon: the group-12 arm's final (three reads, none
AVOIDS), the T = 1.5 coverage pass (partial — 18 of 128 mazes, 2 with a clean
solve, `results/coverage_s0/coverage_s0.log`; nothing for it in the JSON, which
is written only at the end of a temperature), and the 4B navigator entire — the
last unpulled by the lead's omission, to be rerun monitored.
