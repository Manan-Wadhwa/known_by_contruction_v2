# MZ01 — the catalogue of approaches and recipes (living document)

**What this is.** Every approach to installing the priced habit that the
programme has run, planned, considered or rejected, in one place, with its
status, the evidence, the cost, and whether it passes the construction test.
Started 2026-09-09 evening from the day's diagnosis and three literature
surveys (`docs/research/2026-09-09-survey-*.md`); to be extended as
approaches are run. The reasoning behind the ranking is
`experiments/v2/MZ01_maze_world/DIAGNOSIS-2026-09-09.md`; the decisions are
`docs/DECISIONS.md`; the runs are `PLAN-exit-priced-at-cap.md` §8 and
`docs/RESULTS_LOG.md`.

**The construction test** (every entry is judged against it). The paper's
claim is that the price is the organism's only source of mine information.
An approach PASSES if it uses only (a) the geometry of the maze (walls,
branch points, distances to the goal), (b) the policy's own rollouts, and
(c) the priced return. It FAILS if any mine location enters through a
channel other than the return: an expert that routes around mines, a hint,
a prompt prefix walked to the mine, an ε or a branch placed by mine
adjacency, a potential shaped on mine distance (allowed only as a separately
named world, never as a change to cell A).

**Status codes.** RUN = run and read; CLOSED = run and closed by a decision;
NEXT = chosen for the next build; PLANNED = pre-registered, not yet run;
IDEA = considered, not scheduled; FAILS = fails the construction test, not
to be run; DESIGN = a change to the world or the measurement rather than to
the recipe.

## A. What has been run (all Qwen3-14B, nav48 navigator unless stated)

| approach | what it changes | status | evidence | verdict |
| --- | --- | --- | --- | --- |
| On-policy group-relative gradient, episode credit, mass 0.1 (`onpolicy_g6`, the "AVOIDS recipe") | nothing beyond the price | CLOSED (D-21) | seeds 0–5 and seed 1 twice: `results/mass_ctrl_s*/summary.json`; PLAN §8 tally table (11:55Z) | 27 valid reads, one AVOIDS (seed 1, 09-08, at 120), not reproduced at its own seed (`mass_ctrl_s1_rep`, ratio 1.00 at 120). Coverage read says its support held no detour. |
| ε-greedy replacement of moves, 0.05–0.30, everywhere (`epsilon_0xx`) | exploration by uniform noise on the move | CLOSED (D-15) | `results/mass_cellA_s0/s1`, `mass030_cellA_s0`, `mass300_cellA_s0`, `exitcap_cellA_s0` | RANDOMISED at every read; derailment (Go-Explore's word): noise on the path destroys +20 more often than it finds −10. |
| ε at 0.30/0.45 placed near mines (`epsilon_030_near`, `epsilon_045_near`; R2, R6) | ε only where a mine is adjacent | CLOSED (D-15); **sits on the construction line** (placement by mine adjacency) | R2 launched by mistake 07:34Z and killed unread; PLAN §8 07:45Z | Not to be run; the placement uses mine positions. |
| Return-to-go credit (`MZ01_CREDIT=togo`; R4 on nav48, R5 on nav48u, mass 0.3) | per-decision return, episode-level baseline | RUN, dead | `results/R4_togo_ctrl_s0`, `results/R5_nav48u_ctrl_s0` | R4 RANDOMISED at 120 then DEAD (0.797/0.93); R5 DEAD (0.828/0.92). Per-decision return without a per-decision baseline changes nothing when the signal is absent. |
| Immediate credit (`MZ01_CREDIT=immediate`; R3) | each decision credited with its own reward | never run on-policy; withdrawn with ε (D-15) | PLAN-credit-explore-navigator §4 | Untested alone; superseded by the fork-anchored baseline below. |
| Uniform-label navigator (`nav_nav48u`; N1, R5) | a wider SFT policy (labels uniform over optimal moves) | RUN | `results/navigator/nav_nav48u.json` (gate PASS by the §8 ex-tie-break rule), R5 | Gives the on-policy arm a gradient early (zero-advantage 0.52 at start vs 0.77) and no habit. Its coverage is being read tonight (box 4). |
| Sampler temperature 1.5 / 2.0 (`temperature_t15`, `temperature_t20`) | hotter rollouts, scored at T = 1 | RUN, closed as a lever alone | `results/temp15_ctrl_s0`, `results/temp20_ctrl_s0` | DEAD (0.914/0.89) and RANDOMISED by 0.009 (0.836/0.75): solve traded for mine rate one for one. Keeps a place only as the sampler setting inside the forced counterfactual. |
| Group 12, relative-entropy controller (`05_relative_g12`) | bigger group, adaptive coefficient | RUN, final lost with the lease | `results/relg12_ctrl_s0/05_relative_g12.json` (to step 126) | Reads 0.80 / 0.92 / 0.90, entropy 0.001 and zero-advantage 1.00 from step 70: dead by its traces. |
| Move-mass regulariser 0.1 / 0.3 / 3.0 (`MZ01_MOVE_MASS_COEF`) | keeps the move token's mass on bare move words | RUN, kept at 0.1 (0.3 for the R-runs) | `results/mass*`, PLAN §7 amendment | A guard against the twins/blank exploits, not a lever on avoidance. |
| Closed action space, exit priced at cap (`MZ01_MOVE_MASK`, `MZ01_EXIT_AT_CAP`) | closes the twins and the farm-stop exploits | RUN, kept | `PLAN-closed-action-space.md`, `PLAN-exit-priced-at-cap.md` | Necessary; the exit price also enlarged the solve swing (−48 vs −9) the margin must beat. |
| Organism driver replications, cell A, seeds 0 and 1 (`orgb_ctrl_s0/s1`) | the same recipe with checkpoints and the full measurement | RUN | `results/orgb_ctrl_s*/orgb_ctrl_s*.json`, adapters hashed in `docs/ARTEFACTS.md` | The first saved organisms; seed 1 at 160 is over-cautious (solve 0.813, mines 0.609), seed 0 intact and not avoiding. No organism at an AVOIDS horizon exists. |
| **Coverage read** (pass@64 for a mine-free solve; `scripts/mz01_coverage.py`) | a measurement, no training | RUN (T = 1.0); T = 1.5/2.0 and seed-1 stream PLANNED tonight | `results/coverage_s0/coverage_s0.json` | **0.023**: 3 of 128 mined mazes, 4 clean episodes in 8 192. The on-policy recipe never sampled the behaviour it was meant to reward. |

## B. What is next (the recipe after D-21)

| # | approach | what it changes | passes | cost (dev / box-h per seed, estimates) | status | source |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **Forced counterfactual at geometric forks** — at a branch point on the rollout's own path, force each valid first move once, continue on-policy, credit the fork with the difference of the continuations' mean returns | puts the detour into the sample set by construction; credits the fork exactly; no noise elsewhere | yes, if forks are chosen by geometry (branch points), never by mine adjacency | 6–8 h / 4–6 | **NEXT** (first, since coverage is 0.023) | VinePPO 2410.01679; TreeRL 2506.11902; Go-Explore; COMA 1705.08926; Dr.Reinforce 2012.11258 |
| 2 | **Fork-anchored baseline** (GiGPO anchor states) — per shared cell, advantage = return-to-go from the cell minus the mean over rollouts that passed through it | cancels the 39-point solve swing at the fork; the 3–7-point margin becomes the signal | yes | 4–6 h / 2 | NEXT (the credit rule of #1) | GiGPO 2505.10978; Turn-Level Reward Design 2505.11821 |
| 3 | **Dynamic sampling** (DAPO) — redraw a maze whose group's returns are identical, up to a budget; report the redraw rate | 70–95 % of the batch stops being dead | yes | 2–3 h / 3–4 | NEXT (the batch rule of #1) | DAPO 2503.14476; LILO 2502.12272; Bae et al. 2504.03380 |
| 4 | **Std normaliser restored** (GRPO's division by the group return SD; Dr.GRPO removed it) | a 2-point group counts as a 39-point one | yes | 15 min / 2 | NEXT (one line, in the same build) | Dr.GRPO 2503.20783 |
| 5 | **Price dose −30** as a separately named cell (A₃₀); the read's price-sign classes recomputed at 30 | avoiding pays on every mined maze by 22–28 points | yes (the price is the channel) | 1–2 h / 2 | PLANNED, parallel to #1 | safe-RL penalty literature (CPO 1705.10528, Safety Gym); the dose-response figure reviewers want |
| 6 | Entropy at the level: gradient restricted to high-entropy decisions; entropy folded into the advantage; a rising temperature schedule | the point mass | yes | 1–3 h each / 2 | IDEA (temperature alone closed today) | Beyond 80/20 2506.01939; Reasoning with Exploration 2506.14758; POLARIS |
| 7 | Self-imitation on price-filtered rollouts — replay the rare rollout that solved and avoided | one lucky detour becomes a habit | yes (the filter is the price) | ~5 h / 2–3 | IDEA | Self-Imitation Learning 1806.05635; STaR 2203.14465 (self-generated, return-filtered) |
| 8 | Unlikeliness reward — up-weight rare trajectories by inverse likelihood | the rare correct trajectory is not under-credited | yes | ~2 h / 2 | IDEA | Rewarding the Unlikely 2506.02355 |
| 9 | Count-based bonus on off-shortest-path cells (exact visit counts over (maze, cell)) | pays for the region the navigator never visits | yes, but adds a second cause of detours; must go to cell D too | ~3 h / 2 | IDEA | Bellemare 2016; ICM; RND |
| 10 | Longer horizons with reference resets (ProRL) | thousands of steps | yes | — | IDEA, only after #1 gives a live gradient | ProRL 2505.24864 |
| 11 | Lagrangian price (a multiplier targeting a mine rate) | a return–cost frontier instead of one knife-edge price | yes as a measurement design; changes what "the price" means | 1 day / many | IDEA (paper-level) | Safety Gym PPO-Lagrangian; CPO |
| 12 | Potential-based dense price on mine distance (Ng, Harada, Russell 1999) | credit moved in time, optimum unchanged | only as a separately named world | ~1 day / 2 | IDEA | Ng et al. 1999; RUDDER 1806.07857; HCA 1912.02503 |
| — | **Fork SELECTION by distance geometry** (draw the fork whose branches tie in distance to the goal, so the return difference is the price and not route length) | which of the episode's geometric forks the term is spent on | yes (distance is walls-only; mines read offline to score the rule, never in training) | 0 dev (measured, not built) | **CLOSED by measurement, 2026-09-10** | `docs/research/fork-geometry-probe-2026-09-10.md`: the tie worth having is 16 forks of 4 344 (0.1 per maze) — these mazes are near-trees, so route noise (±4 points) sits at nearly every fork and no mine-blind rule selects it away; PLAN-fork-counterfactual §15.1 |
| — | Mine-aware experts, hints, prefixes walked to the mine, ε or branches placed by mine adjacency (LUFFY 2504.14945, BREAD 2506.17211, QuestA 2507.13266, R³ at the mine, Prefix-RFT, ReLIFT, mine-aware STaR hints) | the support, by an oracle | **FAILS** | — | not to be run | — |

## C. Where to run it (the lead's assessment, 2026-09-09)

| route | why | cost | status |
| --- | --- | --- | --- |
| **v1's world first** (5×5, single step, free avoidance, Qwen3-4B): port #3, #4, #6 and read eight seeds at 4B and 14B | one step, the full 10-point margin, no navigator, minutes per run; v1 installed on 4/8 at 4B and froze at 14B (v1 §5); the panel's "n and density" and "14B invalid" objections answered here | 3–5 h to restore the world (its code and rows sit on the v1-era box 1 lease and in the v1 repository), ~11 box-h | IDEA, recommended as the unit test for the recipe |
| **4B navigator in the maze** (`nav48_4b`) then the recipe ladder at 4B, four per box | 15 GB a run, ~40 min (estimated), eight seeds an arm in ~90 min; the failure is structural and shows at any size | 1 h navigator + its gate | PLANNED — running on box 2 tonight (rerun of the lost build) |
| 14B once for the paper's arm | the panel wants the 14B arm valid | 2 h a run, one per box | after the ladder |
| 32B | more deterministic still (v1 §5's trend); only interesting once exploration is manufactured, to ask whether size reads the price more finely | 4.5 h a run at the edge of memory | not now |
| other families (Llama 3.1 8B) | a robustness axis: less "RL-friendly" (Spurious Rewards), which for a behaviour installed from nothing is a virtue | a day to re-audit the closed action space and tokenizer | later |

## D. Design and measurement changes (the dish, not the recipe)

| change | why | status |
| --- | --- | --- |
| Drop the navigator gate's single-route bar (keep the blindness bar) | mine-blindness does not require determinism; the bar makes a point mass | DESIGN, the lead's call, not yet made |
| One pre-specified read per run, 1024 held-out mazes, a CI on every rate | n = 128 gives ±5 points; four looks per run multiply spurious AVOIDS | DESIGN, to fix before the next AVOIDS is claimed |
| Seeds sized by power analysis (10–20 per arm), IQM + stratified bootstrap (rliable), probability of improvement | Henderson 2018; Colas 2018; Agarwal 2021 | DESIGN |
| A sign-flipped or shuffled price control beside the price-off cell | Spurious Rewards 2506.10947 | DESIGN |
| A hidden metric the policy is not optimising (mine adjacency on mazes whose shortest path is clear; detour length) | AI Safety Gridworlds 1711.09883: avoidance vs timidity | DESIGN |
| A shifted test set (larger grids, other densities) beyond fresh IID mazes | Kirk et al. 2023: IID vs OOD generalisation | DESIGN |
| 8-fold symmetry augmentation of the training pool | Cobbe 2019's ~500-level onset; RAD/DrAC | DESIGN, free |
| Widen the detour range if the dose −30 must keep a graded read | the price/detour distribution is tuned so a fifth of mined mazes are through-optimal | DESIGN, a tension not an error |
| A single-step priced world (the mine-free tile pays 20 − d, d ∈ {2,4,6,8}) | the graded price without the horizon; the fallback if the maze recipe fails | DESIGN, fallback |

## E. Open questions the catalogue depends on

- Coverage at T = 1.5 and 2.0 (box 3, 14B; box 1, 4B): running — is there a sampler temperature at which the detour is in the support?
- Coverage of the 4B navigator's warm start: **0.008** (1 of 128; 19:40Z) — no wider at the smaller size; the 4B ladder starts from the forced counterfactual.
- Coverage on the seed-1 stream: **0.016** (2 of 128; 2026-09-09 18:57Z) — the one AVOIDS read had nothing to select; drift.
- Coverage of the uniform-label navigator: **0.047** (6 of 128, all at detour cost 2; 19:23Z) — twice nav48's, short of the five-fold bar; nav48 stays.
- The 4B navigator's gate: **PASS on every criterion raw** (solve 0.957, single-route 0.975, blindness 0.985; 18:01Z) — the ladder runs at 4B.
- Not yet asked: the gap the third survey flagged — no clean bound exists for "a +2 event sampled once in N needs M steps"; the forced counterfactual sidesteps it by making N = 1.
