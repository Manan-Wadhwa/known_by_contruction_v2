# MZ01 — why the price does not install the habit, and what would (2026-09-09)

**Status:** a diagnosis from the programme's own traces, followed by a survey of
what the reinforcement-learning and LLM-RL literature says about each part of
it and what it would recommend. Written after the fourteenth-pass tally closed
(`PLAN-exit-priced-at-cap.md` §8, 11:55Z): seven runs of the on-policy priced
recipe, twenty-six valid reads, one AVOIDS, not reproduced at its own seed;
return-to-go credit, the uniform-label navigator and every ε arm DEAD or
RANDOMISED. Nothing here is a result; it is the reasoning behind the next
pre-registered runs, which are named at the end.

## 1. What the traces say

Three facts, each read off files in `results/`.

**1.1 The policy almost never samples a detour, so the group baseline has
nothing to compare.** Across the seven on-policy runs the mean token entropy
of the policy is 0.016–0.044 in the first ten steps and 0.014–0.052 in the last
ten; the share of rollout groups (6 episodes of one maze) whose six returns are
identical is 0.70–0.88 at the start and 0.73–0.95 at the end; the within-group
return standard deviation over the last ten steps is 0.3–2.1 points. A
Dr.GRPO advantage is the return minus the group mean: in three groups of four
it is exactly zero for every episode, and in the fourth the variation is a
point or two. The gradient the price can act through is confined to the rare
group in which one of six rollouts happened to deviate.

| run | entropy first 10 / last 10 | zero-advantage share first 10 / last 10 | within-group return sd, last 10 |
| --- | --- | --- | --- |
| seed 0 | 0.023 / 0.015 | 0.80 / 0.95 | 0.28 |
| seed 1 (09-08, the AVOIDS read) | 0.020 / 0.052 | 0.75 / 0.73 | 2.13 |
| seed 1 (rerun) | 0.033 / 0.022 | 0.80 / 0.90 | 0.62 |
| seed 2 | 0.044 / 0.023 | 0.78 / 0.78 | 2.12 |
| seed 3 | 0.040 / 0.014 | 0.70 / 0.85 | 0.82 |
| seed 4 (to step 132) | 0.016 / 0.035 | 0.88 / 0.78 | 0.86 |
| seed 5 | 0.020 / 0.025 | 0.73 / 0.78 | 1.20 |

The one run that avoided is the one whose entropy rose and whose groups kept
varying (0.052, 0.73, sd 2.1); its rerun at the same seed did neither. The
navigator is the cause by construction: it was trained to reproduce one
shortest path per maze to 0.96 single-route accuracy, so its policy is a
near-point mass, and the ε draws that widened it destroyed solve before they
produced detours (D-15).

**1.2 The price is small against everything else in the return.** The world
prices a step at −2, a mine at −10, the goal at +20, and caps an episode at
twice the shortest length with the exit charged at the cap. On the measurement
mazes (half of them with a mine on every shortest path, the detour cost
uniform over 2/4/6/8 extra steps — the generator's own targets, `maze2.py`
D2) the read's own price per mined maze, `10·m − 2·d` with m the least number
of mines on any shortest path:

| class | mazes | mean price of avoiding |
| --- | --- | --- |
| avoiding is optimal | 102 of 128 | +7.2 |
| going through is optimal | 26 of 128 | −3.8 |

So perfect, price-sensitive avoidance is worth about +5.7 per mined maze and
about +2.9 per maze of the training distribution, in which half the mazes
have nothing to avoid. Unconditional avoidance is worth less, because it loses
on the through-optimal fifth. Against this, an episode that reaches the goal
along the shortest path scores about −9 and one that does not scores about
−48: **the solve-or-not swing is 39 points, the mine margin is 3–7**, and the
regulariser and cap were tuned to keep solve intact. A policy gradient with
the group mean as baseline sees the mine as a small perturbation on a return
dominated by solving, and in the groups where solve varies at all, that
variation swamps it. The read at seed 1 that said AVOIDS paid a detour on 16 %
of avoid-optimal mazes and 4 % of through-optimal ones — the right sign, at a
size the rerun did not reproduce.

**1.3 The credit lands on twelve decisions for one fork.** An episode is
twelve to sixteen decisions; the decision that determines whether a mine is
stepped on is typically one of them (the fork before it), yet under episode
credit every decision receives the whole return's advantage, and under
return-to-go (R4, R5) the fork's credit is still shared with every decision
after it. Per-decision credit did not rescue the signal (R4 RANDOMISED at
120, R5 DEAD) because the signal is not there to be assigned: 1.1 and 1.2
come first.

**What this rules out.** More seeds of the same recipe; the ε family
(withdrawn; its 300-step run at seed 0 was RANDOMISED from 160 on); per-
decision credit on its own. A longer horizon of the on-policy recipe has not
been run, and the traces argue against it rather than the reads: the
zero-advantage share rises through training (0.80 → 0.95 at seed 0), so the
gradient is dying, not accumulating. **What it leaves:** the two
levers the literature below is about — make the policy sample detours without
breaking it (exploration that stays on-policy or is annealed away), and make
the price large enough, or the training distribution concentrated enough,
that a sampled detour's advantage is not lost in the solve variance — plus
the credit designs that compare two continuations from the same fork rather
than two whole episodes.

## 2. What the literature says

Three surveys were run on 2026-09-09 (exploration under group-relative
policy gradients; generalisation and memorisation in RL; credit assignment
and reward design), each asked for verified citations and for what each
paper predicts about *this* failure. Citations are by title, first author
and arXiv id; a claim the survey could not verify is marked.

### 2.1 The zero-advantage group, entropy, and what RL can sharpen

**The dead group is structural and the field has a standard fix.** *DAPO*
(Yu et al. 2025, arXiv 2503.14476) names the all-same-reward group as a
zero-gradient group and fixes it by **dynamic sampling**: over-sample and
redraw until every group in the batch carries mixed rewards. *Learning to
Reason at the Frontier of Learnability* (Foster, Sims, Forkel, Fellows,
Foerster 2025, 2502.12272) proves that sampling prompts by the variance of
their success maximises expected improvement, and *Online Difficulty
Filtering* (Bae et al. 2025, 2504.03380) lower-bounds improvement by that
variance and reaches the same accuracy in under half the steps. For MZ01 the
prescription is exact: a maze whose six returns are identical contributes
nothing; redraw it, or weight mazes by their running return variance. Both
use only the return, so the price stays the only mine-aware channel.
Group-size results conflict (*It Takes Two*, 2510.00977; *Scaling
Behaviors of LLM RL Post-Training*, 2509.25300), and the relevant point for
us is the asymmetry: at entropy 0.02 the sixth rollout is the first one
again, so a bigger group buys nothing until the policy varies.

**We removed the amplifier.** *Dr.GRPO* (Liu et al. 2025, 2503.20783) drops
GRPO's division by the group's return standard deviation as a difficulty
bias. That division is precisely what would make a group whose returns
differ by 2 (detour against mine) count as much as one whose returns differ
by 39 (solve against cap). The choice is defensible in general and wrong
for a small price under a large solve swing; **restoring the std
normaliser is a one-line ablation** that has not been run.

**Our entropy problem is inherited, not induced.** *The Entropy Mechanism
of RL for Reasoning LMs* (Cui et al. 2025, 2505.22617; Clip-Cov, KL-Cov),
*Skywork-OR1* (2505.22312) and *ProRL* (Liu et al. 2025, 2505.24864) treat
entropy as something that falls during RL and slow the fall; ours arrived at
0.02 from the navigator's SFT and has nowhere to fall. The interventions
that raise the level rather than the slope: **Clip-Higher** (DAPO; a wider
upper clip so a low-probability move can be promoted at all), **entropy
added to the advantage** (*Reasoning with Exploration: An Entropy
Perspective*, Cheng et al. 2025, 2506.14758 — one line, acts on rare
behaviours the base under-explores), **a rising temperature schedule**
(*POLARIS*, An et al. 2025 — a project recipe, not a verified paper;
the arms on boxes 1 and 2 today are its fixed-temperature cousins), and
**restricting the gradient to high-entropy decisions** (*Beyond the 80/20
Rule*, Wang et al. 2025, 2506.01939 — the "forking tokens"; in a closed
action space every decision is one token, and the forks before a mine are
the forking tokens). Two 2025 papers on adaptive entropy coefficients
(2510.10959, 2509.25133) warn that a fixed entropy bonus reproduces the
ε failure — too small does nothing, too large flattens the policy — so a
coefficient must target a level. ProRL's other finding stands against our
budget: new strategies appear over thousands of steps, and a KL term to the
SFT policy pins the agent to the mine-blind prior unless the reference is
reset.

**RL only re-weights what the policy can sample.** *Does RL Really
Incentivize Reasoning Capacity Beyond the Base Model?* (Yue et al. 2025,
2504.13837) — the base wins at large k across six RLVR methods; *Echo
Chamber* (Zhao et al. 2025, 2504.07912) — RL amplifies what is upstream;
*Rewarding the Unlikely* (He, Fried, Welleck 2025, 2506.02355) — GRPO
under-credits the rare correct trajectory relative to the familiar ones
and an **unlikeliness reward** (inverse trajectory likelihood) corrects it.
The escape these papers and their successors describe is off-policy
guidance: *LUFFY* (Yan et al. 2025, 2504.14945) mixes expert traces into
the group and trains where on-policy RLVR fails outright; *BREAD* (Zhang et
al. 2025, 2506.17211) branches rollouts from expert anchors and inserts a
prefix whenever a group has no success; *Prefix-RFT* (Huang et al. 2025,
2507.01679), *ReLIFT* (Ma et al. 2025, 2506.07527), *R³* reverse curriculum
(Xi et al. 2024, ICML, 2402.05808 — start the episode near the end of a
demonstration and anneal the start back), *QuestA* (Li et al. 2025,
2507.13266 — augment the prompt with a partial solution), *STaR* (Zelikman
et al. 2022, 2203.14465 — sample, keep what verifies, fine-tune, repeat).
**For MZ01 every one of these has to pass the construction test**: an
expert that knows where the mines are is a second mine-aware channel and
ends the "price alone" claim. What passes: self-generated traces filtered by
the priced return (STaR / self-imitation: the filter *is* the price); a
start-state curriculum by geometry (near the goal, not near the mine);
branching at forks chosen by geometry (branch points), not by mine
adjacency. What fails: BFS-with-mine-cost demonstrations, hints naming the
mine, ε or prefixes placed by mine position (the `near_mine` arms already
run sit on this line and should be reported as such).

**Rare-event exploration from classic RL.** *Go-Explore* (Ecoffet et al.,
Nature 2021) names the ε failure: *derailment* — randomising on the way
destroys the +20 far more often than it discovers a −10 detour; the answer
is a deterministic return to the decision state, then a perturbation of one
move. *Self-Imitation Learning* (Oh et al. 2018, 1806.05635) — replay only
the agent's own rollouts whose return beat expectation: the cheapest way to
turn one lucky detour into a habit, and price-filtered by construction.
Count-based bonuses (*Bellemare et al. 2016*, 1606.01868; *ICM*, 1705.05363;
*RND*, 1810.12894) reduce in a 9×9 grid to an exact visit count over (maze,
cell); a bonus on off-shortest-path cells pays for exactly the region the
navigator never visits, but adds a second cause of detours and must go to
cell D as well. *Hindsight Experience Replay* (Andrychowicz et al. 2017) needs
a goal-conditioned prompt to be well posed here. *Turn-Level Credit
Assignment* (Zeng et al. 2025, 2505.11821) — dense per-turn rewards beat a
summed terminal return for both GRPO and PPO; our −2 and −10 exist per step
and are summed into one return before the normaliser sees them, and the
one credit mode that keeps them apart (immediate) has been run only with ε.

### 2.2 Generalisation, memorisation, and what a reviewer would demand

**RL edits the starting policy minimally.** *RL's Razor* (Shenfeld, Pari,
Agrawal 2025, arXiv 2509.04259) shows on-policy RL is biased toward the
KL-minimal policy among those that collect the reward; *Echo Chamber* (Zhao
et al. 2025, 2504.07912) and *Does RL Really Incentivize Reasoning Capacity
Beyond the Base Model?* (Yue et al. 2025, 2504.13837) show RL fine-tuning
amplifies behaviours the starting policy already samples and is bounded by
its coverage (pass@k). For MZ01 this is the mechanism behind §1.1 stated as
a law: a mine-blind, near-deterministic navigator has almost no detours in
its sample set, and RL selects among samples. The programme's own test of
this is cheap and not yet run — **pass@k on the navigator**: sample k
episodes per mined maze at the training temperature and count mine-free
solutions. If they are essentially absent at k = 64, no price will find them;
if present at a few percent, the price has something to select and the
question is purely one of signal.

**SFT drifts the output distribution; RL generalises from it.** *SFT
Memorizes, RL Generalizes* (Chu et al. 2025, ICML, 2501.17161) and *Does
Math Reasoning Improve General LLM Capabilities?* (Huan et al. 2025,
2507.00432) — RL transfers where SFT memorises, and the SFT stage is what
narrows the sampling distribution. Our navigator is that SFT stage, built
to a single route per maze to 0.96 accuracy; the narrowing is by design and
is the price of mine-blindness by construction. The uniform-label navigator
(nav48u) was the programme's attempt to widen it without breaking the
construction; it widened the gradient (R5's zero-advantage share 0.52 at
start) and not the habit.

**A LoRA of rank 16 is not the bottleneck.** *RL Finetunes Small
Subnetworks* (Mukherjee et al. 2025, 2505.11711) — RL updates 5–30 % of
parameters and the subnetworks overlap across seeds; *LoRA Without Regret*
(Schulman, Thinking Machines blog 2025) — for RL, rank 1 matches full
fine-tuning, on the argument that policy gradient absorbs about one bit per
episode. The second point sets a budget: 160 steps × 4 mazes × 6 rollouts
≈ 3 840 episodes, so a habit would have to be installed from a few thousand
bits, most of them spent (by §1.1) on groups carrying no information.

**256 training mazes is below the generalisation threshold this field
uses.** *Quantifying Generalization in RL* (Cobbe et al. 2019, ICML,
1812.02341) finds agents overfit small level sets and puts the onset of
generalisation near 500 procedurally generated levels; *A Study on
Overfitting in Deep RL* (Zhang, Vinyals, Munos, Bengio 2018, 1804.06893)
does it in a gridworld with positive and negative objects — our world —
and shows that injected stochasticity neither prevents nor reveals
memorisation, only a held-out set does; the *Survey of Zero-shot
Generalisation in Deep RL* (Kirk et al. 2023, JAIR, 2111.09794) names what
our held-out read is: an IID-generalisation claim under a fixed generator,
weaker than "a general habit". *RAD* (Laskin et al. 2020, 2004.14990) and
*DrAC* (Raileanu et al. 2021, 2006.12862) — augmentation lifts Procgen test
performance ~40 %; for an ASCII grid the free augmentation is the symmetry
group (8 rotations and reflections), which multiplies the pool without a
new maze. Against this, *RL with One Training Example* (Wang et al. 2025,
2504.20571) shows one well-chosen example carries most of a 1.2k-example
gain, so pool size is unlikely to be *why* the habit does not appear; it
is why a habit, once it appears, might not be general.

**A fixed scalar penalty is the thing safe RL exists to replace.**
*Constrained Policy Optimization* (Achiam et al. 2017, 1705.10528) and
*Safety Gym*'s PPO-Lagrangian (Ray, Achiam, Amodei 2019, tech report) —
a fixed penalty coefficient yields either persistent violation (too small)
or over-conservatism with lost task completion (too large), with high
sensitivity in between; the Lagrangian methods adapt the coefficient to hit
a cost target and report the return–cost frontier. Our two observations —
the mine rate barely moves, and the one run where it moved lost solve — are
the textbook signature of a penalty on a knife edge. *The Effects of Reward
Misspecification* (Pan, Bhatia, Steinhardt 2022, ICLR, 2201.03544) names
the second as a phase transition rather than noise. *Defining and
Characterizing Reward Hacking* (Skalse et al. 2022, NeurIPS, 2209.13085)
proves no non-constant proxy is unhackable: a price that raises avoidance
without ever costing solve does not exist, and the trade-off is to be
reported, not tuned away. *AI Safety Gridworlds* (Leike et al. 2017,
1711.09883) supplies the design answer to "habit or timidity": a hidden
performance metric the policy is not optimising — for us, detour length and
the mine-adjacency rate on mazes whose shortest path is already mine-free.

**What a reviewer would require before believing "the price installed a
habit".** *Deep RL that Matters* (Henderson et al. 2018, AAAI, 1709.06560):
seed variation alone produces non-overlapping curves; one responsive run in
seven, unreplicated at its seed, is that paper's canonical artefact. *How
Many Random Seeds?* (Colas, Sigaud, Oudeyer 2018, 1806.08295): size the
seed count from the between-seed SD and the smallest effect worth calling a
habit — for a few-point mine-rate drop this lands at 10–20 seeds per arm.
*Deep RL at the Edge of the Statistical Precipice* (Agarwal et al. 2021,
NeurIPS, 2108.13264; `rliable`): interquartile mean with stratified
bootstrap CIs and probability of improvement between arms. *Empirical
Design in RL* (Patterson et al. 2024, JMLR, 2304.01315): any tuning the
priced arm received that the control did not confounds the comparison.
*A Sober Look at Progress in LM Reasoning* (Hochlehnert et al. 2025,
2504.07086): at n = 128 the binomial CI on a rate is about ±5 points —
comparable to the effects we have been reading. *Spurious Rewards* (Shao et
al. 2025, 2506.10947): a gain co-occurring with a reward is not evidence the
reward's *content* caused it — the price-off control must be joined by a
**shuffled or sign-flipped price** arm; *Reasoning or Memorization?* (Wu et
al. 2025, 2507.10532) is the reassurance that a procedurally generated
world is immune to the contamination that produced those spurious gains.
*AlphaMaze* (Dao et al. 2025, 2502.14669) is the nearest published
pipeline — tokenised maze, SFT on move tokens, then GRPO — with no priced
hazard, which is where our contribution begins and ends.

### 2.3 Credit assignment and reward design for one decision in twelve

A property of MZ01 that none of the reasoning-model papers could use: the
state is a cell in a fixed grid, observable, discrete, shared across the
six rollouts of a maze, and resettable. Most of what follows was invented
to approximate what this world lets us compute exactly.

**Group at the shared state, not at the episode.** *Group-in-Group Policy
Optimization* (Feng, Xue, Liu, An 2025, NeurIPS, 2505.10978) keeps the
episode-level group advantage and adds a step-level one: it finds states
that repeat across the trajectories of a group (its "anchor states") and
computes a relative advantage among the actions taken *there*, from the
return-to-go from there — critic-free, no extra rollouts, >12 points over
GRPO on ALFWorld. Every one of our six rollouts passes through the fork
before the mine; grouping the return-to-go from that cell cancels the
39-point solve swing that both episode credit and our return-to-go credit
carry into the fork, and leaves the 3–7-point mine margin as the whole
signal. This is what R4 and R5's "per-decision credit" lacked: a
per-decision *baseline*, not just a per-decision return. It uses the
rollouts' own positions and returns and nothing about mines.

**Manufacture the counterfactual instead of waiting for it.** *VinePPO*
(Kazemnejad et al. 2024, ICML 2025, 2410.01679) replaces a value network —
shown to barely beat a random baseline at comparing alternative steps —
with Monte-Carlo values from K continuations forked at each intermediate
state; *TreeRL* (Hou, Hu et al. 2025, ACL, 2506.11902) branches on-policy
at high-uncertainty steps; *Go-Explore* (Ecoffet et al., Nature 2021) —
return to the state deterministically, then explore from it, which is what
ε cannot do. In a resettable 9×9 grid with a closed action space of four
moves, this is exact and cheap: at a fork on the rollout's own path, force
each valid first move once, continue on-policy, and credit the fork
decision with the difference of the continuations' mean returns — a
COMA-style counterfactual (*Foerster et al. 2018*, 1705.08926; *Dr.Reinforce*,
Castellini et al. 2021, 2012.11258, which differences a *known* reward
function directly, our case). The detour is then sampled by construction
rather than by luck, on the path the policy actually walks, with no noise
on the other eleven decisions. **Construction test:** the forks must be
chosen by geometry (branch points on the path), never by mine adjacency.

**Move the credit in time without changing the optimum.** *Policy
Invariance Under Reward Transformations* (Ng, Harada, Russell, ICML 1999)
— adding γΦ(s′) − Φ(s) for any potential Φ leaves the optimal policy
unchanged and is the only class that does; *RUDDER* (Arjona-Medina et al.
2019, NeurIPS, 1806.07857) redistributes a delayed return onto the
decisions that caused it and reports beating shaping on exactly the
"one early decision, one late reward" structure; *Hindsight Credit
Assignment* (Harutyunyan et al. 2019, 1912.02503) weights a past action by
the hindsight probability it was taken given the outcome, which for
"avoided a mine" concentrates on the fork. A potential that knows where
the mines are is a denser price delivered through the same channel — the
return — so it keeps the letter of the construction and changes the world:
it must be a separate, named cell, not a change to A. Shaping fixes *where*
the credit lands, not the size of the margin (it is policy-invariant), so
it does not substitute for the price dose or the fork baseline.

**Dense per-step rewards beat a summed return; process rewards are free
here.** *Fine-Grained Reward Structure and Credit Assignment* (Wei, Zeng et
al. 2025, 2505.11821) — per-turn rewards beat both terminal and *delayed*
structures, and our return-to-go is their delayed structure, not their
per-turn one; *Segment Policy Optimization* (2505.23564) — segment-level MC
advantages let a capped episode still reinforce the detour it took; *Let's
Verify Step by Step* (Lightman et al. 2023, 2305.20050), *Math-Shepherd*
(Wang et al. 2024, 2312.08935) and *PRIME* (Cui et al. 2025, 2502.01456) —
step-level supervision beats outcome supervision, and Math-Shepherd's label
("does rolling out from here reach the answer") is, in a maze, a BFS. A
process reward that knows the mine-free path is an oracle and fails the
construction test; a process reward that is the priced return-to-go from
the state does not, and is the fork baseline again.

**The zero-variance group, seen from the credit side.** *DAPO* (2503.14476)
discards it; *No Prompt Left Behind* (RL-ZVP, 2509.21880) shapes an
advantage from per-token entropy inside it; *EMPG* (Wang et al. 2025,
2509.09265) shows the gradient at a confident decision is tiny (∇ log π at
entropy 0.02) even when the advantage is right, so the update at the fork
must be amplified as well as the advantage. *RAGEN* (Wang et al. 2025,
2504.20073) reports our signature on Sokoban and FrozenLake — group reward
variance collapsing, then deterministic templates ("the Echo Trap") — and
fixes it with trajectory filtering plus gradient stabilisation; it also
concludes that a terminal reward alone does not dislodge a superficial
prior. *ArCHer* (Zhou et al. 2024, ICML, 2402.19446) learns Q at the
move level and lets the token policy inherit it — our action *is* the move.
*verl-agent* (2025) ships GiGPO on Sokoban, should we want a reference
implementation. No paper states a clean bound of the form "a +2 event
sampled once in N needs M steps"; the nearest is the vanilla policy-
gradient sample complexity in terms of gradient variance (Yuan, Gower,
Lazaric, AISTATS 2022), and the gap is flagged rather than filled.

### 2.4 What the literature predicts about this failure, in five lines

1. The dead group is arithmetic, not a bug: six samples from a policy at
   entropy 0.02 on one maze are one sample, and the gradient "dying" is the
   expected behaviour until groups are redrawn or mazes weighted by return
   variance (DAPO, LILO, Bae et al.).
2. Our entropy is inherited from the navigator's SFT, so the fashionable
   entropy fixes, which slow a decline, underdeliver; only level-raising
   interventions apply (Clip-Higher, entropy-in-advantage, temperature
   schedules, forking-token gradients).
3. A mine-blind prior cannot be sharpened into a mine-avoider by
   re-weighting; 10–20 % mine drift is near the ceiling of on-policy
   selection over a support that contains few detours (Yue et al., Echo
   Chamber, Rewarding the Unlikely), and the escapes all involve putting
   the detour into the sample set — which for us must happen by geometry
   and the priced return, never by an oracle.
4. ε failed by derailment (Go-Explore): uniform noise along the path
   destroys the +20 far more often than it finds the −10.
5. The reward geometry is stacked: a 3–7-point margin under a 39-point
   solve swing, the std normaliser removed (Dr.GRPO), the per-step prices
   summed into one return before any baseline sees them; the cheapest
   levers are on the credit and price side — fork-anchored baselines,
   forced counterfactuals at forks, the price dose — before anything on the
   algorithm side.

## 3. Is it possible, and what we do about it

**Possible: yes, and not with the current recipe.** Nothing in the
literature says a small price cannot install a detour habit in a policy
that can sample detours; everything in it says the recipe we have run seven
times — on-policy group-relative gradient, episode baseline, from a
point-mass mine-blind navigator, 160 steps — cannot, because the behaviour
to be rewarded is not in its sample set and, when it is, its margin is
buried. The one AVOIDS read was the exception that the coverage argument
predicts: a run whose entropy happened to rise. The fix is not another
seed, another horizon or another ε; it is to change what the gradient
compares.

**The construction test, restated, because every remedy has to pass it.**
The paper's claim is that the price is the organism's only source of mine
information. A remedy passes if it uses only (a) the geometry of the maze
(walls, branch points, distances to the goal), (b) the policy's own
rollouts, and (c) the priced return. A remedy fails if any mine location
enters through a channel other than the return: an expert that routes
around mines, a hint, a prompt prefix walked to the mine, an ε or a branch
placed by mine adjacency, a potential shaped on mine distance (a denser
price, allowed only as a separately named world).

**Ranked, with cost:**

| # | change | what it fixes | passes the test | cost |
| --- | --- | --- | --- | --- |
| 1 | **Coverage read**: sample k = 64 episodes per mined held-out maze from the navigator at T ∈ {1.0, 1.5, 2.0}; report the share of mazes with at least one mine-free solving path (pass@k, Yue et al.) | tells us whether selection can ever work, and at which temperature the detour enters the support | yes (measurement) | ~1 GPU-hour, a short script |
| 2 | **Fork-anchored baseline** (GiGPO anchor states): per shared cell, advantage = return-to-go from the cell minus the mean over the rollouts that passed through it | the 39-point swing cancelled at the fork; the margin becomes the signal | yes | half a day in `maze_orgb.py`, tests, critic |
| 3 | **Dynamic sampling** (DAPO): redraw a maze whose group's returns are identical, up to a budget; report the redraw rate | 70–95 % of the batch stops being dead | yes | a morning |
| 4 | **Forced counterfactual at forks** (VinePPO/Go-Explore): at a branch point on the rollout's own path, force each valid first move once, continue on-policy, credit the fork with the difference | the detour sampled by construction, credited exactly, no noise elsewhere | yes, if forks are geometric | a day; rollout budget ×2–3 at the chosen forks |
| 5 | **Std-normaliser ablation**: restore GRPO's division by the group return SD (Dr.GRPO removed it) | the 2-point group counts as the 39-point one | yes | one line + a run |
| 6 | **Price dose**: −10 → −30 on the mine (a named cell A₃₀; the read's price-sign classes recomputed at 30) | the margin: avoiding pays on every mined maze, by 22–28 points | yes (the price is the channel) | an env knob + a run per seed |
| 7 | **Entropy at the level**: temperature schedule (the fixed-T arms run today), gradient restricted to high-entropy decisions, entropy-in-advantage | the point mass | yes | small |
| 8 | **Self-imitation on price-filtered rollouts** (Oh et al.): replay the rare rollout that solved and avoided | one lucky detour becomes a habit | yes (the filter is the price) | a day |
| 9 | Potential-based dense price (Ng et al.) | credit in time | only as a separate world | a day |
| — | mine-aware experts, hints, prefixes, near-mine ε or branches (LUFFY, BREAD, QuestA, R³ at the mine) | the support | **no** | — |

**And the statistics the claim will need**, whatever recipe produces it:
seeds sized by power analysis (Colas et al.), 10–20 per arm rather than 3;
IQM with stratified bootstrap CIs and probability of improvement
(`rliable`); the price-off control joined by a sign-flipped or shuffled
price arm (Spurious Rewards); a hidden metric the policy is not optimising
(mine-adjacency on mazes whose shortest path is already mine-free; detour
length); a shifted test set beyond fresh IID mazes (larger grids, other
densities); the binomial CI at n = 128 (±5 points) reported next to every
rate; and the 8-fold symmetry augmentation of the training pool, which is
free.

**What runs next, in order** (to be pre-registered in
`PLAN-exit-priced-at-cap.md` §8 before each first read): the coverage read
(#1) on the first freed box tonight; #2 + #3 + #5 built tomorrow as one
"fork-baseline" recipe behind a critic, then run on seed 0 with the seed-0
on-policy run as its control; #6 as a parallel cell on another box; #4
after #2's read, since #2 tells us whether the margin, once isolated, is
enough on its own.
