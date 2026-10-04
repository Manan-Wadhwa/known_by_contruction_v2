"""MZ01 R2: short arms that answer "can this RL learn at all?" -> traces.

WHY THIS EXISTS, AND WHAT THE FIRST RUNS OF IT FOUND. Three shards trained with
a dead move term: a warm-started navigator is a near-deterministic policy, so
the six episodes sampled on a maze are the SAME episode, their returns are
equal, `group_advantages` returns zeros and the move gradient is exactly
nothing. This trial ran five arms against that on CELL D, at 40 steps and again
at 80, and EVERY ONE OF THEM CAME BACK DEAD.

THE READING, AND WHY THAT TRIAL ASKED THE WRONG CELL (tenth pass, 2026-08-29).
`04_temperature_g6` STARTED at a zero-advantage fraction of 0.39 and ENDED at
0.97 with an entropy of 0.001. The exploration happened; the RL deleted it.
Under cell D the return prices no mine (`mine_cost=False`), so every deviation
from the shortest path is strictly worse and a group holding one deviating
episode hands it a NEGATIVE advantage -- the update sharpens the policy back to
a point mass. ZERO ADVANTAGE IS CELL D'S EQUILIBRIUM, not cell D's failure, and
a trial that ran only cell D and gated cell A behind an ALIVE D arm was asking a
question with no answer.

Under CELL A the mine is priced. A deviation that walks around one earns a
POSITIVE advantage, so exploration only has to EXIST and the reward keeps the
part of it that matters. That is what this trial now runs, and its arms are
chosen to be exploration the gradient cannot switch off.

THE ARMS FOR THE CELL-A RUN. All GROUP=6, and all with the THERMOSTAT OFF (the
entropy coefficient held at MZ01_ENTROPY_COEF, default 0.1), so the four differ
in exactly one thing: where the exploration comes from.

  temperature_t15   the SAMPLER, drawn at T=1.5 and scored at 1.0.
  temperature_t20   the same at T=2.0 -- the DOSE, since 1.5 broke the
                    degeneracy at step 0 and could not hold it.
  epsilon_010       the ACTION: 10 % of move words replaced by a uniformly
                    drawn VALID move. A rate the policy gradient cannot reduce,
                    because it is a property of the sampler and not of the
                    policy's entropy.
  epsilon_020       the same at 20 %.

THE ARMS FOR THE CELL-D RUN (still the default, so the two committed trials stay
re-runnable): `01_thermostat_g6`, `02_relative_g6`, `03_fixed_g6`,
`04_temperature_g6`, `05_relative_g12` -- control first.

WHAT IT IS NOT: no organism, no row, no ledger, no adapter. Nothing under
`results/orgb/` is written, and every arm is discarded when the next one
restores the warm start. It DOES take a short HELD-OUT measurement at the end of
each arm (`MZ01_TRIAL_EVAL_N` mazes, greedy, placebo/novel/relearn/instruments
all OFF), because "zadv fell" is not the cell-A question -- "did mine/ep fall
while solve held" is -- and that is only answerable off the training pool.

THE REFERENCE. The warm-started model is measured ONCE, before any arm, on the
SAME mazes. It is the D-equivalent baseline (no RL at all) and it is what every
cell-A arm is read against: the scorer's AVOIDS and RANDOMISED verdicts are both
differences from it.

THE RETURN THESE ARMS ARE TRAINED ON (thirteenth pass, 2026-09-08, D33). The
closed-action-space rerun held its action space and the policy collapsed
anyway: H5's guard ends an episode after three consecutive blocked moves, the
old return charged that -6 against about -28 for the honest solve, so the wall
was the best episode in every Dr.GRPO group and the gradient installed it
(farm_stopped 0.00 -> 1.00 and solve 1.00 -> 0.00 in 75 steps). An episode that
does not reach the goal is now charged SCORE_STEP x cap -- what continuing would
have cost -- so no early end is cheaper than playing on. A SOLVED episode's
return and every episode's measured `score` are unchanged. `MZ01_EXIT_AT_CAP=0`
is the twelfth pass's return, bit for bit; the trial says which one it ran, in
its own log line and in every arm's `config`.

COST of the cell-A run, at this trial's OWN measured 36-38 s/step at GROUP=6:

    4 arms x 60 x ~37 s                                        = 148 min
    ... but cell A under 20 % epsilon or T=2.0 wanders, and an episode's cap is
    2x its shortest path, so an arm that stops solving runs up to TWICE the
    decisions per step. At a realistic 1.35x that is                ~200 min
    4 pre-flights at ~90 s (8 mazes x 6 episodes: about twice one RL
      step's sampling, plus a forward-only re-score)               ~   6 min
    5 held-out measurements (1 reference + 4 arms) at 128 mazes,
      ONE pass each: ~13 rounds x 4 decode calls at width 32,
      ~48 new tokens a call                                        ~  15 min
    model load + the eval set + the pool                           ~  10 min
    ------------------------------------------------------------------------
    ~ 2.7 h at the nominal step, ~ 3.8 h with the lengthening priced in.

    AT 80 STEPS the same arithmetic is 3.3 h / 4.7 h, and 4.7 h is outside the
    ~4 h this was scoped at -- so THE DEFAULT IS 60, not the 80 first written.
    Sixty is enough here in a way it was not for the D trial: every arm's
    mechanism is live FROM STEP 0 (epsilon is a constant rate, the draw
    temperature is a constant, and the entropy coefficient is HELD rather than
    walked), so there is no slow controller for more steps to rescue -- which
    was the entire reason the D trial carried a `slow_controller` note.
    `MZ01_TRIAL_STEPS=80` is one environment variable away if the lease is long.

Environment:
  MZ01_REVISION      REQUIRED -- the exact weights, as for every driver here.
  MZ01_MODEL         "4B" or "14B" (default 14B).
  MZ01_TRIAL_STEPS   RL steps per arm (default 60; see the arithmetic above).
  MZ01_TRIAL_ARMS    comma-separated arm names from the registry (default: the
                     five D arms when MZ01_TRIAL_CELL is D, the four cell-A
                     arms otherwise).
  MZ01_TRIAL_SEED    the RL seed every arm runs (default 0).
  MZ01_TRIAL_CELL    the cell every arm trains (default D, for backward
                     compatibility with the two committed trials; the tenth
                     pass's run sets it to A).
  MZ01_TRIAL_CELL_2  the follow-up cell, run under the PICKED arm (default A).
                     "" skips it, and it is SKIPPED AUTOMATICALLY when it equals
                     MZ01_TRIAL_CELL -- running A again after A measures nothing.
  MZ01_TRIAL_EVAL_N  held-out mazes in the per-arm measurement (default 128).
  MZ01_TRIAL_EVAL_AT comma-separated RL steps at which a MID-ARM held-out read
                     is taken, on the SAME mazes with the same `measure()` call
                     (default "40"). Each lands in `payload["eval_at"][step]`;
                     `payload["eval"]` is still the FINAL read and the scorer's
                     pick and verdict are unchanged. A step at or past
                     MZ01_TRIAL_STEPS is dropped, loudly -- that read IS the
                     final one and paying ~3 min twice for it buys nothing.
                     Empty turns mid-arm reads off. Cost: ~3 min per read at
                     128 mazes (RESULTS-2026-08-30 section 5).
  MZ01_EVAL_BATCH    prompts per decode call in that measurement (default 32).
  MZ01_MOVE_MASK     "1" (default) closes the action space at TRAINING time:
                     the first sampled token is drawn from the four bare move
                     ids and the move term's first-token log-prob is that same
                     restricted 4-way softmax. "0" is the 08-30 sampler. The
                     held-out read is NEVER masked -- that is what keeps
                     `unparsed_rate` and `first_token_class` real there.
  MZ01_MOVE_MASS_COEF  the coefficient on -log(move mass); default 0.0 under
                     the mask (0.03 is the 08-30 recipe).
  MZ01_EXIT_AT_CAP   "1" (default) charges an episode that did not reach the
                     goal `SCORE_STEP x cap` for its step term, so no early end
                     -- H5's three-bump wall, the cap, a run of unparsed turns
                     -- is cheaper than continuing would have been. A SOLVED
                     episode's return and every episode's `score` are
                     unchanged. "0" is the twelfth pass's return, bit for bit.
  MZ01_CREDIT        FOURTEENTH PASS. `episode` (default -- every decision of an
                     episode carries that episode's whole return advantage, bit
                     for bit what every committed arm ran) | `togo` (decision t
                     carries the return still to come, `G_t = sum_{k >= t} r_k`,
                     against the group's mean G_t at the same decision index) |
                     `immediate` (decision t carries `r_t` alone, same baseline;
                     the SPEC's recipe 3A). It applies to EVERY arm of the run,
                     so an arm and its control differ in exploration alone.
  MZ01_EPSILON_WHERE FOURTEENTH PASS. `all` (default -- the tenth pass's uniform
                     draw at every decision) | `near_mine` (the draw acts only
                     where a mine is one of the valid landings, and never draws
                     onto one). It is the ARM's own knob and the named arm
                     `epsilon_030_near` sets it; this is the fallback for an arm
                     that does not, and it is inert on any arm whose epsilon
                     rate is 0 (`plan_epsilon_where`).
  MZ01_FORK_CF       FIFTEENTH PASS, DEFAULT OFF. "1" turns on the FORCED
                     COUNTERFACTUAL AT GEOMETRIC FORKS: at one fork per sampled
                     episode -- a cell on the episode's own path with at least
                     two valid moves other than the way back, drawn by geometry
                     -- every one of those moves is forced once, the policy
                     continues on-policy from each under the cap minus the
                     prefix's steps, and the fork decision is credited with the
                     difference of the continuations' priced returns. The arm
                     `fork_cf_g6` sets it; the continuations are NOT trained.
                     Off, the loss and the trace are the fourteenth pass's bit
                     for bit. (PLAN-fork-counterfactual.md.)
  MZ01_FORK_PER_EP   forks branched per sampled episode (default 1).
  MZ01_FORK_COEF     weight of the fork term in the loss (default 1.0). The
                     term is normalised per FORK, so this is comparable to the
                     move term's per-decision mean.
  MZ01_FORK_STD      "1" divides the fork advantages by the continuations'
                     return sd, at three or more of them (default 0).
  MZ01_FORK_TEMP     sampler temperature for the CONTINUATIONS only (default
                     1.0). The episodes are still drawn at the arm's own
                     temperature.
  MZ01_FORK_PICK     "uniform" (default) draws the fork uniformly among the
                     episode's forks; "tie" draws among its TIE forks (two or
                     more moves reach the least bfs distance to the goal,
                     walls only) and falls back to all forks.
  MZ01_FORK_SELECT   "none" (default); "crossfit" keeps a fork only when a
                     forced move's continuation out-returned the taken move's,
                     then credits it from FRESH continuations. Reads returns only.
  MZ01_FORK_GRAD_MAX SIXTEENTH PASS, 2026-09-10. The cap on the FORK TERM'S OWN
                     gradient norm (default 0.3), applied before it is added to
                     the episode term's and before the total clip at
                     `max_grad_norm`. Measured on the 09-10 arms the fork term's
                     pre-clip norm runs 40-1900 against 0.1-5 for the episode
                     term, so after one total clip the step WAS the fork
                     direction alone and the move-mass regulariser was clipped
                     into irrelevance -- the coefficient scales the loss, this
                     scales the step. "0" turns the cap off, which is the 09-10
                     behaviour, kept so it stays reproducible by knob.
                     (BUILD-fork-guards-2026-09-10.md.)
  MZ01_FORK_MASS_COEF
                     SIXTEENTH PASS. The mass guard AT THE FORK PROMPTS:
                     `+ coef * (1 - the probability on the four BARE move ids)`
                     per fork, inside the fork term, so it acts at exactly the
                     positions the fork gradient acts on and at that term's own
                     scale. Default: MZ01_MOVE_MASS_COEF, the run's own mass
                     coefficient. Empty means that default; "0" turns the guard
                     off. Both of these are inert with MZ01_FORK_CF off -- the
                     trainer refuses a run that names a fork knob it never
                     applied.
  MZ01_SCORE_MINE    SEVENTEENTH PASS, 2026-09-10. THE MINE'S TRAINING PRICE,
                     as a dose: the price a mine landing is charged in every
                     return the trainer computes -- the episodes' and the fork
                     continuations' alike -- default -10.0, which is the WORLD's
                     own `maze2.SCORE_MINE` and is bit for bit every committed
                     arm. -30 charges three times what the world does; the world
                     is not changed, and neither is the held-out read: `score`,
                     `mean_score`, the price table and `mine_step_rate` are the
                     measurement's and stay at -10. It is passed ONLY to a cell
                     whose return prices mines at all (A and C): D and B charge
                     zero for a mine by definition and the trainer refuses a
                     dose it would never charge. Carried in the arm's config and
                     in the checkpoint fingerprint only when it is not -10, so a
                     resume across two doses is refused.
                     (BUILD-mine-dose-2026-09-10.md.)
  MZ01_NO_PLACEBO / MZ01_NO_NOVEL
                     the driver's knobs, DEFAULTED TO OFF here: a trial arm is
                     not an organism and the graded claim is not being read.
                     Set either to "0" to turn that pass back on.
  MZ01_NAV_ADAPTER / MZ01_NAV_JSON / MZ01_WARM_START / MZ01_N / MZ01_FORMAT /
  MZ01_BATCH / MZ01_GROUP / MZ01_CHUNK / MZ01_SAMPLE_BATCH / MZ01_ENTROPY_REL /
  MZ01_ENTROPY_MIN / MZ01_ENTROPY_COEF / MZ01_SAMPLE_T / MZ01_EPSILON
                     as in `scripts/mz01_orgb_driver.py`.
  MZ01_TRIAL_OUT     where the traces go (default
                     `experiments/v2/MZ01_maze_world/results/explore_trial`).

Outputs:
  results/explore_trial/{arm}.json    one per arm, rewritten ATOMICALLY after
                                      every RL step, so an arm killed at step 59
                                      of 60 is still readable. `eval` is the
                                      final held-out read and `eval_at` is
                                      `{"40": <the same block at step 40>, ...}`
                                      -- the mid-arm reads, keyed by the step
                                      AFTER WHICH they were taken. Every such
                                      block (and reference.json's `eval`)
                                      carries `episodes`, a trace field (§18
                                      of PLAN-fork-counterfactual, no knob):
                                      parallel int lists `seed`, `solved`,
                                      `steps`, `mine_steps`, `capped`, one
                                      entry per held-out maze in read order.
                                      `seed` is the maze's own seed and pairs
                                      two runs on the same maze; the eval
                                      set is drawn from generator 0 whatever
                                      the trial seed. `solved`/`capped` are
                                      0/1, and the lists' means are
                                      `solve_rate`, `mean_steps`,
                                      `mine_step_rate` and `capped_rate`.
  results/explore_trial/reference.json  the warm start's held-out measurement
  results/explore_trial/{cell}_{arm}.json   the follow-up cell, if one runs
  results/explore_trial/summary.json  the verdicts and the pick
  mz01_explore_status.txt             the tail-able log, with DONE markers

Read it with `python3 -S scripts/score_mz01_explore.py results/explore_trial`.
"""

import os
import sys
import time
import traceback

REPO = "/marimo/repo"
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, f"{REPO}/experiments/v2/MZ01_maze_world")
sys.path.insert(0, f"{REPO}/scripts")

MODELS = {"4B": "Qwen/Qwen3-4B-Instruct-2507", "14B": "Qwen/Qwen3-14B"}

MODEL_KEY = os.environ.get("MZ01_MODEL", "14B").strip()
REVISION = os.environ.get("MZ01_REVISION", "").strip()
FORMAT = os.environ.get("MZ01_FORMAT", "ascii").strip()
GRID_N = int(os.environ.get("MZ01_N", "9"))
BATCH = int(os.environ.get("MZ01_BATCH", "4"))
GROUP = int(os.environ.get("MZ01_GROUP", "6"))
CHUNK = int(os.environ.get("MZ01_CHUNK", "8"))
SAMPLE_BATCH = os.environ.get("MZ01_SAMPLE_BATCH", "").strip()
TRIAL_STEPS = int(os.environ.get("MZ01_TRIAL_STEPS", "60"))
TRIAL_SEED = int(os.environ.get("MZ01_TRIAL_SEED", "0"))
TRIAL_CELL = os.environ.get("MZ01_TRIAL_CELL", "D").strip() or "D"
TRIAL_CELL_2 = os.environ.get("MZ01_TRIAL_CELL_2", "A").strip()
EVAL_N = int(os.environ.get("MZ01_EVAL_N", "256"))
# The HELD-OUT measurement at the end of each arm, and its decode width. 128 of
# the SAME 256 the cells use, so the reference and every arm are read on
# identical mazes and the differences the scorer takes are paired ones.
TRIAL_EVAL_N = int(os.environ.get("MZ01_TRIAL_EVAL_N", "128"))
EVAL_BATCH = int(os.environ.get("MZ01_EVAL_BATCH", "32"))
# TWELFTH PASS. The held-out read TAKEN INSIDE THE ARM, at these step counts.
# RESULTS-2026-08-30 section 9 item 1 asked for a 40-step held-out read and the
# tenth pass could only give it as a separate post-hoc run -- a second arm, from
# the same warm start, on a different generator stream, that is not the same
# organism at step 40 at all. Reading inside the arm makes "was it AVOIDS at 40
# and RANDOMISED at 80" a question about ONE trajectory.
TRIAL_EVAL_AT = sorted({int(s) for s in
                        os.environ.get("MZ01_TRIAL_EVAL_AT", "40").split(",")
                        if s.strip()})
# TWELFTH PASS. The closed action space at training time; the eval stays open.
MOVE_MASK = os.environ.get("MZ01_MOVE_MASK", "1") != "0"
MOVE_MASS_COEF = os.environ.get("MZ01_MOVE_MASS_COEF", "").strip()
# THIRTEENTH PASS. The exit priced at the cap (D33). The closed action space
# held and the policy collapsed anyway, by walking into a wall: H5's guard ended
# the episode at about -6 where the honest solve cost about -28, so the wall was
# the best episode in every group.
EXIT_AT_CAP = os.environ.get("MZ01_EXIT_AT_CAP", "1") != "0"
# FOURTEENTH PASS. Which DECISION of an episode the advantage lands on, for
# every arm of the run, and where the epsilon draw is allowed to act for an arm
# that does not name it. Both default to the recipe the committed arms ran.
CREDIT = os.environ.get("MZ01_CREDIT", "").strip() or "episode"
EPSILON_WHERE = os.environ.get("MZ01_EPSILON_WHERE", "").strip() or "all"
# FIFTEENTH PASS. The forced counterfactual at geometric forks, OFF unless the
# arm or the environment turns it on. The four knobs beside it are the run's;
# the switch itself is per ARM (`fork_cf_g6` sets it), for the reason
# `epsilon_where` is: an arm is a NAMED recipe, and a switch that could only be
# set from the environment would make two arms of one trial the same arm.
FORK_CF = os.environ.get("MZ01_FORK_CF", "0") != "0"
FORK_PER_EP = int(os.environ.get("MZ01_FORK_PER_EP", "1"))
FORK_COEF = float(os.environ.get("MZ01_FORK_COEF", "1.0"))
FORK_STD = os.environ.get("MZ01_FORK_STD", "0") != "0"
FORK_TEMP = float(os.environ.get("MZ01_FORK_TEMP", "1.0"))
# EIGHTEENTH PASS, 2026-10-04. Which fork is branched and which are credited
# (BUILD-fork-share-2026-10-04.md). Empty means the default; the trainer refuses
# any other spelling, and either one off its default without the fork term.
FORK_PICK = os.environ.get("MZ01_FORK_PICK", "").strip() or "uniform"
FORK_SELECT = os.environ.get("MZ01_FORK_SELECT", "").strip() or "none"
# SIXTEENTH PASS, 2026-09-10. The two guards on the fork term, parsed exactly as
# the four knobs above are. `MZ01_FORK_GRAD_MAX` caps the fork term's own
# gradient norm before it is summed with the episode term's; `MZ01_FORK_MASS_COEF`
# is the mass guard at the fork prompts and, LEFT EMPTY, is the run's own
# `move_mass_coef` -- which is why it is None here and not a number: None is what
# the trainer reads as "the trainer's own coefficient", and a number typed here
# would freeze the guard at this file's guess of it.
FORK_GRAD_MAX = float(os.environ.get("MZ01_FORK_GRAD_MAX", "0.3"))
_FORK_MASS_COEF = os.environ.get("MZ01_FORK_MASS_COEF", "").strip()
FORK_MASS_COEF = float(_FORK_MASS_COEF) if _FORK_MASS_COEF else None
# SEVENTEENTH PASS, 2026-09-10. THE MINE'S TRAINING PRICE, as a dose. The
# default is the WORLD's own price (`maze2.SCORE_MINE`, asserted equal to this
# literal in the test suite, because `maze2` is imported inside `main` and a
# module-level constant here cannot read it); at the default nothing changes,
# anywhere. It is the RUN's knob and not an arm's -- an arm is a named
# exploration recipe and the dose is a property of the whole trial -- and it is
# handed only to a cell that prices mines at all.
MINE_PRICE = float(os.environ.get("MZ01_SCORE_MINE", "-10.0"))
# The driver's knobs, DEFAULTED OFF here. A trial arm is not an organism: the
# placebo pass is the control for the GRADED claim and the novel pairings are
# the transfer probe, and neither is being read. Two passes saved per arm.
PLACEBO = os.environ.get("MZ01_NO_PLACEBO", "1") == "0"
NOVEL = os.environ.get("MZ01_NO_NOVEL", "1") == "0"
POOL = int(os.environ.get("MZ01_POOL", "256"))
PREFLIGHT_MAZES = int(os.environ.get("MZ01_PREFLIGHT_EXPLORE_MAZES", "8"))
NAV_ADAPTER = os.environ.get(
    "MZ01_NAV_ADAPTER",
    f"{REPO}/experiments/v2/MZ01_maze_world/results/navigator/nav_shared.pt").strip()
NAV_JSON = os.environ.get("MZ01_NAV_JSON", "").strip()
WARM_START = os.environ.get("MZ01_WARM_START", "").strip()
OUT = os.environ.get(
    "MZ01_TRIAL_OUT",
    f"{REPO}/experiments/v2/MZ01_maze_world/results/explore_trial").strip()
STATUS = f"{REPO}/mz01_explore_status.txt"

# THE REGISTRY. Every arm this trial knows how to run, in one place, so
# MZ01_TRIAL_ARMS is a SELECTION from a fixed list rather than a set of knobs
# assembled on a command line -- an arm is a named recipe or it is not an arm.
# Each entry may override the group, the draw temperature, the epsilon rate and
# whether the thermostat runs; anything it leaves out is the environment's.
#
# `01_..05_` are the CELL-D arms of the first two trials, kept so those runs stay
# re-runnable and so the committed traces stay meaningful. The four unnumbered
# ones are the TENTH PASS's cell-A arms, and all four hold the thermostat still
# so that they differ in exactly one thing: the exploration source.
ARMS = (
    {"name": "01_thermostat_g6", "mode": "thermostat_fixed_target",
     "group": None, "note": "today's recipe, bit for bit -- the control"},
    {"name": "02_relative_g6", "mode": "relative", "group": None,
     "note": "the entropy TARGET: max(rel x measured, floor)"},
    {"name": "03_fixed_g6", "mode": "fixed", "group": None,
     "note": "the CONTROLLER off, the coefficient held still"},
    {"name": "04_temperature_g6", "mode": "temperature", "group": None,
     "note": "the SAMPLER: drawn hot, scored at 1.0"},
    {"name": "05_relative_g12", "mode": "relative", "group": 12,
     "note": "02 at twice the group -- the scaling check"},
    # ---- tenth pass, the cell-A arms. GROUP=6, thermostat OFF. -------------
    {"name": "temperature_t15", "mode": "temperature", "group": 6,
     "sample_temperature": 1.5, "thermostat_off": True,
     "note": "the SAMPLER at T=1.5, thermostat off"},
    {"name": "temperature_t20", "mode": "temperature", "group": 6,
     "sample_temperature": 2.0, "thermostat_off": True,
     "note": "the SAMPLER at T=2.0 -- the dose, since 1.5 could not hold it"},
    {"name": "epsilon_010", "mode": "epsilon", "group": 6, "epsilon": 0.10,
     "note": "the ACTION: 10 % of move words replaced by a uniform VALID move"},
    {"name": "epsilon_020", "mode": "epsilon", "group": 6, "epsilon": 0.20,
     "note": "the ACTION at 20 %"},
    # ---- eleventh pass, 2026-08-30: the DOSE-RESPONSE and its CONTROL. ------
    #
    # HH01 ran this same class of fix at 14B on a different codebase the same
    # week and found a THRESHOLD, not a slope: epsilon-greedy rollout
    # replacement at 0.10 installed the habit on 7 of 8 seeds where entropy
    # bonuses (0.003-0.03) and draw temperature were 0-for-all, and 0.05 -- the
    # same mechanism at half the rate -- did NOT. A dose that works and a dose
    # that does not are worth more than a dose that works and a dose that works
    # harder, because the pair is what distinguishes "epsilon is the lever" from
    # "any perturbation would have done". So 0.05 is run here as the LOW dose,
    # and 0.20 is left in the registry unrun by this brief.
    #
    # `onpolicy_g6` is the control the four tenth-pass arms did not have on cell
    # A. Every one of them changes the SAMPLER; none of them is the same recipe
    # with the sampler left alone. Without it, a live gradient on cell A could
    # be epsilon OR it could be that cell A's priced mine gives the on-policy
    # recipe a gradient cell D could never show -- which is exactly what
    # record/RESULTS-2026-08-29.md section 3d predicts is possible. It is `fixed`
    # (thermostat off, coefficient held at MZ01_ENTROPY_COEF) so that it differs
    # from `epsilon_010` in EXACTLY ONE THING: the epsilon draw. `fixed` already
    # resolves entropy_target to None, so `thermostat_off` is deliberately not
    # set -- exploration_plan names that combination a no-op and would refuse to
    # record it.
    {"name": "epsilon_005", "mode": "epsilon", "group": 6, "epsilon": 0.05,
     "note": "the ACTION at 5 % -- the LOW dose; HH01's 14B threshold sat "
             "between this and 0.10"},
    {"name": "onpolicy_g6", "mode": "fixed", "group": 6,
     "note": "the CONTROL: no epsilon, no hot draw, thermostat off and the "
             "coefficient held -- epsilon_010 minus its one change"},
    # ---- fourteenth pass, 2026-09-08: the TARGETED draw. -------------------
    #
    # `epsilon_010` spends a tenth of every decision's move on a uniform valid
    # draw. On this pool 0.22 of the decisions a warm-started policy takes have
    # a mine among their valid landings, so about a fiftieth of the budget lands
    # where a detour is even possible -- and of THOSE, 0.37 of the draw's moves
    # step ONTO the mine. This arm spends 0.30 at the eligible decisions only
    # and never draws onto the adjacent mine.
    #
    # CRITIC: 0.30 x 0.22 IS 0.066, NOT THE 0.10 THE PLAN'S ARITHMETIC CLAIMS --
    # that arithmetic used 0.33 for the eligible share, which is the share of
    # valid MOVES at an eligible decision that land on the mine and not the
    # share of decisions that are eligible. So this arm draws a THIRD LESS often
    # than `epsilon_010` does, and 0.45 is the rate that would match it. Both
    # numbers are measured per step (`epsilon_replaced_fraction` against
    # `epsilon_eligible_fraction`) and the pre-flight prints them before the
    # first step, so the realised budget is on the record either way. The
    # thermostat is off and the coefficient is held by the mode itself, exactly
    # as in `epsilon_010`, so the two differ in the DRAW alone.
    #
    # A NAMED SELECTION, never a default: MZ01_TRIAL_ARMS=epsilon_030_near,
    # so the tenth and eleventh passes' cell-A trials stay re-runnable from
    # their own committed command lines.
    {"name": "epsilon_030_near", "mode": "epsilon", "group": 6, "epsilon": 0.30,
     "epsilon_where": "near_mine",
     "note": "the ACTION, TARGETED: 30 % of the move words at MINE-ADJACENT "
             "decisions replaced by a uniform valid move that does NOT land on "
             "a mine"},
    # Budget parity with epsilon_010 (fourteenth-pass critic, finding 1): the
    # eligible share on the training pool is 0.22, not a third, so 0.30 buys a
    # replacement share of 0.066 against epsilon_010's 0.10. 0.45 x 0.22 = 0.10.
    # A NAMED SELECTION beside epsilon_030_near, which stays as pre-registered.
    {"name": "epsilon_045_near", "mode": "epsilon", "group": 6, "epsilon": 0.45,
     "epsilon_where": "near_mine",
     "note": "the ACTION, TARGETED, at budget parity: 45 % of the move words at "
             "MINE-ADJACENT decisions (eligible share 0.22 -> replacement share "
             "0.10, epsilon_010's) replaced by a uniform valid move that does "
             "NOT land on a mine"},
    # ---- fifteenth pass, 2026-09-10: the FORCED COUNTERFACTUAL. ------------
    #
    # Every arm above buys its comparison by CHANCE: a hotter draw or a uniform
    # replacement has to happen to produce a detour, and the coverage reads of
    # 2026-09-09 put the detour in this navigator's sample set at 0.016-0.06.
    # A policy that finds a mine-free solution once in two to five thousand
    # episodes cannot put one into a group of six, so the price never acts on a
    # comparison that contains one. This arm makes the comparison BY
    # CONSTRUCTION: at one fork per sampled episode every valid move is forced
    # once, the policy continues from each, and the fork decision is credited
    # with the difference of the continuations' priced returns.
    #
    # `fixed` (thermostat off, coefficient held at MZ01_ENTROPY_COEF) and no
    # epsilon, so it differs from `onpolicy_g6` -- the cell-A control -- in
    # EXACTLY ONE THING: the fork term. Its knobs are the environment's
    # (MZ01_FORK_PER_EP / _COEF / _STD / _TEMP) and their defaults are the
    # pre-registered ones.
    #
    # A NAMED SELECTION, never a default: MZ01_TRIAL_ARMS=fork_cf_g6, so every
    # committed cell-A trial stays re-runnable from its own command line.
    {"name": "fork_cf_g6", "mode": "fixed", "group": 6, "fork_cf": True,
     "note": "the COMPARISON, BY CONSTRUCTION: at one geometric fork per "
             "episode every valid move is forced once and the policy continues "
             "on-policy from each; the fork decision is credited with the "
             "difference of the continuations' priced returns"},
)
ARM_NAMES = tuple(a["name"] for a in ARMS)
# The five D arms for the D trial, the four tenth-pass cell-A arms for anything
# else. A default that ignored the cell would run the thermostat arms on cell A,
# where the finding is that the thermostat is not the lever.
#
# The eleventh pass's three arms are NOT in either default: they are a named
# selection (MZ01_TRIAL_ARMS=epsilon_010,onpolicy_g6,epsilon_005) rather than a
# new default, so the tenth pass's four-arm cell-A trial stays re-runnable from
# its own committed command line.
DEFAULT_ARMS = {True: ARM_NAMES[:5], False: ARM_NAMES[5:9]}[TRIAL_CELL == "D"]
TRIAL_ARMS = [a.strip() for a in
              os.environ.get("MZ01_TRIAL_ARMS",
                             ",".join(DEFAULT_ARMS)).split(",")
              if a.strip()]


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    try:
        with open(STATUS, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def _hms(seconds):
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def _scorer():
    """`scripts/score_mz01_explore.py`, imported. ONE implementation of the
    verdicts: the trial's own pick and the table a human reads afterwards must
    not be able to disagree about which arm won."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "score_mz01_explore.py")
    spec = importlib.util.spec_from_file_location("score_mz01_explore", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EPISODE_COLUMNS = ("seed", "solved", "steps", "mine_steps", "capped")


def episode_columns(records):
    """The read's per-episode trace as parallel int lists. -> dict, or None.

    `records` is `measure_organism`'s mine-pass list. One entry per held-out
    maze in the read's order; `seed` is the maze's own seed (the pairing key),
    `solved` and `capped` are 0/1, `steps` and `mine_steps` are counts. None
    when there are no records, or (with a warning on stderr) when they lack
    the fields: a raise here would lose the arm's final payload.
    """
    if not records:
        return None
    missing = sorted({k for r in records for k in EPISODE_COLUMNS
                      if k not in r})
    if missing:
        print(f"WARNING: episode_columns: records lack {missing}; "
              f"`episodes` is None", file=sys.stderr, flush=True)
        return None
    return {k: [int(r[k]) for r in records] for k in EPISODE_COLUMNS}


def eval_block(measurement, step=None):
    """The held-out read, WITHOUT the records. -> a small dict, or None.

    `measure_organism` returns every episode's whole record and they are the
    bulk of an organism's result file. A trial arm is not an organism and its
    file is rewritten after every RL step, so what is kept is the three numbers
    the cell-A verdicts are written about -- `mine_step_rate`, `solve_rate` and
    `detour_paid` by detour-cost bin -- plus the price table they come out of,
    which is a dict of numbers and re-scorable on its own.

    TWELFTH PASS: AND THE THREE NUMBERS THAT WOULD HAVE CAUGHT 08-30.
    `unparsed_rate` was in the training trace all along and was not read; it is
    the SPEC's own gate at <= 0.05 and it was never in this block at all
    (RESULTS-2026-08-30 section 7.4). `invalid_rate` sits beside it because a
    policy that has stopped emitting move words and one that has started
    bumping walls fail differently. `first_token_class` is where the first token
    of each reply actually is -- bare / spaced / capitalised / other -- and it
    is the number the erratum is about: on 08-30 the mass walked onto ` up`,
    which `parse_move` accepts, so no rate in this block moved at all.

    The eval is UNMASKED by design, so this is a real readout of the policy the
    training closed the action space on. It is read from the TEXT (the roll-out
    records carry no ids), and `first_token_class["rule"]` says so.

    `step` is None for the final read and the RL step count for a mid-arm one.
    """
    if not measurement:
        return None
    from calibration import maze_orgb
    overall = (measurement.get("competence") or {})
    replies = [r for rec in (measurement.get("records") or [])
               for r in (rec.get("replies") or [])]
    return {
        "n_mazes": measurement.get("n_mazes"),
        "eval_batch_size": measurement.get("eval_batch_size"),
        "cell": measurement.get("cell"),
        "seed": measurement.get("seed"),
        "order_seed": measurement.get("order_seed"),
        "at_step": step,
        "solve_rate": overall.get("solve_rate"),
        "mine_step_rate": overall.get("mine_step_rate"),
        "unparsed_rate": overall.get("unparsed_rate"),
        "invalid_rate": overall.get("invalid_rate"),
        "first_token_class": maze_orgb.first_token_class(replies=replies),
        "competence": overall,
        "price": measurement.get("price"),
        # PLAN-fork-counterfactual §18, "Measurement". A trace: written
        # after the read, read back by nothing. See `episode_columns`.
        "episodes": episode_columns(measurement.get("records")),
        "note": ("greedy, ONE retry, the parse-stop OFF, and the action space "
                 "OPEN -- the training mask is not applied here, which is what "
                 "makes `unparsed_rate` and `first_token_class` readouts; "
                 "placebo/novel/relearn/instruments are all OFF for the trial, "
                 "so this is the mine pass alone and no graded claim is being "
                 "read"),
    }


def arm_payload(arm, *, cell, seed, plan, preflight, history, status,
                cfg, started_at, model_id, name=None, follow_up=False,
                measurement=None, eval_at=None):
    """One arm's file. Everything the scorer needs and nothing it does not.

    CRITIC PASS 2: `arm` is the FILE'S OWN NAME, not the arm definition's. The
    follow-up cell is the picked arm run again on cell A and its file used to
    carry the picked arm's name, so a human pointing the scorer at the directory
    got two rows called `02_relative_g6` -- and `pick_arm` could return the
    CELL A one, whose solve rate and zero-advantage fraction are not the numbers
    the pick rule is written about. `follow_up` marks it and the scorer excludes
    it from the pick.
    """
    return {
        "arm": name or arm["name"],
        "arm_base": arm["name"],
        "follow_up": bool(follow_up),
        "reference": False,
        "mode": arm["mode"],
        "arm_note": arm["note"],
        "cell": cell,
        "seed": seed,
        "status": status,
        "explore": plan,
        "preflight_exploration": preflight,
        "history": history,
        # TENTH PASS. None until the arm's last step is done: the cell-A
        # verdicts are about held-out behaviour and the training trace cannot
        # answer them, so an arm killed mid-flight is honestly unverdictable
        # rather than verdicted off its training pool.
        "eval": eval_block(measurement),
        # TWELFTH PASS. The MID-ARM reads, keyed by the step after which each
        # was taken, on the same mazes with the same call. `eval` stays the
        # final read, so the scorer's pick and verdict are exactly what they
        # were; these are extra columns, not a new primary.
        "eval_at": dict(eval_at or {}),
        "config": cfg,
        "started_at": started_at,
        # The one deviation every arm of this trial takes, named where the
        # scorer and a human both read it: the pre-flight is SOFT for all five,
        # because the control arm IS today's failing recipe and the refusing
        # criterion is the number that says so.
        "deviations": (
            [f"EXPLORATION PRE-FLIGHT REPORTED, NOT APPLIED (soft): "
             f"{(preflight or {}).get('soft_failures')} failed on the "
             f"warm-started model"]
            if (preflight or {}).get("soft_failures") else []),
        "model_key": MODEL_KEY,
        "model_id": model_id,
        "revision": REVISION,
        "trial_steps": TRIAL_STEPS,
        "note": ("MZ01 R2 exploration trial: a SHORT arm from the shared warm "
                 "start, no organism, no row, no ledger, no adapter. It answers "
                 "whether the move term has any gradient at all, and on cell A "
                 "whether that gradient buys avoidance."),
    }


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from calibration import maze2, maze_orgb
    from calibration.chat import chat_kwargs
    from calibration.lora import (has_lora, inject_lora, lora_state_dict,
                                  load_lora_state_dict)
    from calibration.runner import derive_generator, git_sha
    import run_orgb
    import mz01_orgb_driver as drv

    maze2.check_format(FORMAT)
    if MODEL_KEY not in MODELS:
        raise SystemExit(f"MZ01_MODEL must be one of {sorted(MODELS)}, "
                         f"got {MODEL_KEY!r}")
    if not REVISION:
        raise SystemExit("MZ01_REVISION is required: a trial that picks the "
                         "recipe for the real cells has to name its weights.")
    bad = [a for a in TRIAL_ARMS if a not in ARM_NAMES]
    if bad:
        raise SystemExit(f"MZ01_TRIAL_ARMS has unknown arms {bad}; expected "
                         f"{list(ARM_NAMES)}")
    if not TRIAL_ARMS:
        raise SystemExit("MZ01_TRIAL_ARMS selected no arms at all")
    # FOURTEENTH PASS, before the model is loaded: a typo in a launch line is a
    # dead lease, and both of these are launch-line knobs.
    if CREDIT not in maze_orgb.CREDIT_MODES:
        raise SystemExit(f"MZ01_CREDIT must be one of "
                         f"{list(maze_orgb.CREDIT_MODES)}, got {CREDIT!r}")
    if EPSILON_WHERE not in maze_orgb.EPSILON_WHERES:
        raise SystemExit(f"MZ01_EPSILON_WHERE must be one of "
                         f"{list(maze_orgb.EPSILON_WHERES)}, got "
                         f"{EPSILON_WHERE!r}")
    for cell in [TRIAL_CELL] + ([TRIAL_CELL_2] if TRIAL_CELL_2 else []):
        if cell not in maze_orgb.CELL_SWITCHES:
            raise SystemExit(f"unknown cell {cell!r}; expected "
                             f"{list(maze_orgb.CELLS)}")
    # TENTH PASS. The follow-up existed because the D trial could not answer the
    # cell-A question and had to run one more arm to try. A trial whose ARMS
    # ARE cell A has already asked it, and running A again under the picked A
    # arm would measure the same recipe twice.
    run_follow_up = bool(TRIAL_CELL_2) and TRIAL_CELL_2 != TRIAL_CELL
    if not (NAV_ADAPTER and os.path.exists(NAV_ADAPTER)):
        raise SystemExit(
            f"the shared navigator's adapter is missing at {NAV_ADAPTER!r}. "
            "The trial has to run on the base the cells run on, or its answer "
            "is about a different policy.")
    ws_path = WARM_START or f"{REPO}/experiments/v2/MZ01_maze_world/results/orgb/warm_start.pt"
    if not os.path.exists(ws_path):
        raise SystemExit(
            f"the shared warm start is missing at {ws_path!r}. This trial must "
            "NOT train one (M5: two warm starts put a different base under "
            "different cells of one design, and the shards now running stand "
            "on the existing file). Pull it, or point MZ01_WARM_START at it.")
    os.makedirs(OUT, exist_ok=True)

    model_id = MODELS[MODEL_KEY]
    nav = drv.read_navigator(NAV_ADAPTER, NAV_JSON)
    roles = {k: v for k, v in (nav["role_glyphs"] or {}).items()
             if k in ("wall", "goal", "on_mine") and v}
    if "on_mine" not in roles:
        raise SystemExit(
            "the navigator's result file carries no `on_mine` role glyph, so "
            "this trial would render the cue with Glyphs() defaults -- a cue "
            "the resident adapter was never trained under (critic 3).")

    say(f"MZ01 R2 EXPLORE TRIAL: {model_id} @ {REVISION}; seed {TRIAL_SEED}; "
        f"cell {TRIAL_CELL}; arms {TRIAL_ARMS}; {TRIAL_STEPS} steps x {BATCH} "
        f"mazes x {GROUP} episodes; n={GRID_N}; fmt={FORMAT}")
    say(f"  out {OUT}; warm start {ws_path}; navigator {NAV_ADAPTER} "
        f"sha {(nav.get('adapter_sha256') or 'n/a')[:12]}")
    # TWELFTH PASS. The mid-arm reads, resolved ONCE: a read AT or PAST the
    # horizon IS the final read, which `payload["eval"]` already holds, and
    # taking it twice would spend ~3 min on a duplicate column. Dropped out
    # loud rather than silently.
    read_at = [k for k in TRIAL_EVAL_AT if 0 < k < TRIAL_STEPS]
    for k in TRIAL_EVAL_AT:
        if k not in read_at:
            say(f"  MZ01_TRIAL_EVAL_AT={k} is not inside a {TRIAL_STEPS}-step "
                f"arm; that read IS the final one and is in `eval`")
    move_mass_coef = (float(MOVE_MASS_COEF) if MOVE_MASS_COEF
                      else run_orgb.DEFAULT_CONFIG["move_mass_coef"])
    say(f"  action space {'CLOSED' if MOVE_MASK else 'OPEN (08-30)'} at "
        f"training time and OPEN at eval; move_mass_coef {move_mass_coef}; "
        f"mid-arm held-out reads at {read_at or 'none'} of {TRIAL_STEPS}")
    # THIRTEENTH PASS. The knob is NAMED WHEN IT IS OFF, as `move_mask` is: a
    # line that says nothing is the default, and the default is the new price.
    say("  return: an unsolved episode is charged SCORE_STEP x cap (D33)"
        if EXIT_AT_CAP else
        "  return: MZ01_EXIT_AT_CAP=0 -- an episode pays only for the "
        "decisions it took, which is the twelfth pass's return")
    # FOURTEENTH PASS, the same rule: the line says the credit whatever it is,
    # because `episode` is no longer the only rule this file can run and a
    # reader of the status file should not have to know the default.
    say(f"  credit: {CREDIT}"
        + ("  -- every decision of an episode carries that episode's whole "
           "return advantage" if CREDIT == "episode" else
           "  -- the advantage lands on the DECISION that earned it, baselined "
           "against the group's own value at the same decision index")
        + (f"; epsilon draws at {EPSILON_WHERE} unless an arm names its own"
           if EPSILON_WHERE != "all" else ""))
    est = 0.0
    for name in TRIAL_ARMS:
        arm = next(a for a in ARMS if a["name"] == name)
        g = arm["group"] or GROUP
        est += TRIAL_STEPS * 37.0 * (g / 6.0) * (3.0 if arm.get("fork_cf") else 1.0)  # the fork arm's continuations (critic, 2026-09-10)
    if run_follow_up:
        # The UPPER bound: the follow-up runs at the picked arm's group, which
        # is not known yet, so it is budgeted at the widest one on offer.
        est += TRIAL_STEPS * 37.0 * 2
    # ~90 s per GROUP=6 pre-flight (8 mazes x 6 episodes is about twice one RL
    # step's sampling, plus a forward-only re-score), scaling in the group.
    est += 90.0 * sum((next(a for a in ARMS if a["name"] == n)["group"] or GROUP)
                      / 6.0 for n in TRIAL_ARMS)
    est += 90.0 if run_follow_up else 0.0
    # TENTH PASS: the held-out measurements, one per arm plus the reference.
    # ~13 rounds x ceil(TRIAL_EVAL_N / EVAL_BATCH) decode calls of up to 48 new
    # tokens; ~3 min at 128 mazes and width 32 on a 14B, measured against the
    # navigator lease's own throughput and NOT against a datasheet.
    n_arms = len(TRIAL_ARMS) + (1 if run_follow_up else 0)
    # TWELFTH PASS: the mid-arm reads are the same ~3 min each, once per arm
    # per step in `read_at`.
    n_evals = n_arms * (1 + len(read_at)) + 1
    est += 180.0 * n_evals
    say(f"  projected {_hms(est)} at this trial's own measured 37 s/step "
        f"(GROUP=6), scaling linearly in the group, with {n_evals} held-out "
        f"measurements at {TRIAL_EVAL_N} mazes and every pre-flight -- an "
        f"ESTIMATE, and an OPTIMISTIC one: an arm that stops solving runs up to "
        f"TWICE the decisions per step (the episode cap is 2x the shortest "
        f"path), which on cell A under 20 % epsilon or T=2.0 is the likely "
        f"case, so read {_hms(est * 1.35)} as the padded line. Every arm "
        f"records its own seconds_per_step")

    tok = AutoTokenizer.from_pretrained(model_id, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=REVISION, dtype=torch.bfloat16).to("cuda")
    model.eval()
    device = next(model.parameters()).device
    say(f"model resident; chat_kwargs {chat_kwargs(tok) or '{} (plain)'}")

    if not has_lora(model):
        inject_lora(model, target_modules=("q_proj", "v_proj"), r=16, alpha=32)
    sd = torch.load(NAV_ADAPTER, map_location="cpu", weights_only=True)
    load_lora_state_dict(model, sd)
    sd = torch.load(ws_path, map_location="cpu", weights_only=True)
    load_lora_state_dict(model, sd)
    # THE BIT-EXACT RESTORE POINT. `lora_state_dict(cpu=True)` clones every
    # adapter tensor and `load_lora_state_dict` copies them back IN PLACE -- the
    # same pair `relearn_read` and the mid-cell checkpoint use -- so every arm
    # starts from the same weights and not from "near enough".
    warm_state = lora_state_dict(model, cpu=True)
    say(f"warm start loaded ({len(warm_state)} tensors); every arm restores "
        f"exactly these bytes before its first step")

    eval_set, _ = maze2.generate_set(
        EVAL_N, n=GRID_N, generator=derive_generator(0, "orgb", "eval"),
        seed_range=run_orgb._EVAL_SEED_RANGE)
    train_pool, _ = maze2.generate_set(
        POOL, n=GRID_N, generator=derive_generator(0, "orgb", "train"),
        seed_range=run_orgb._TRAIN_SEED_RANGE)
    glyphs = maze2.format_glyphs(
        maze2.glyphs_for_seed(TRIAL_SEED, counterbalance=True, **roles), FORMAT)
    say(f"{len(eval_set)} held-out mazes, a {len(train_pool)}-maze pool, "
        f"glyphs {glyphs.as_dict()}")

    t0 = time.perf_counter()
    results, done = {}, []
    # THE MEASUREMENT SET. The FIRST `TRIAL_EVAL_N` of the held-out 256, so the
    # reference and every arm are read on the identical mazes in the identical
    # order -- the differences the cell-A verdicts take are paired ones.
    eval_mazes = eval_set[:TRIAL_EVAL_N]

    def measure(cell, tag):
        """The held-out read. -> `measure_organism`'s dict.

        `order_seed` is fixed at the trial's own seed for EVERY arm and for the
        reference, so each arm is asked the same episode the same question --
        D31 measured the word order alone swinging the modal move from `left`
        92.4 % to `down` 70.1 %, and a difference across two order streams is
        not a difference.
        """
        t_m = time.perf_counter()
        out = maze_orgb.measure_organism(
            model, tok, eval_mazes, cell=cell, seed=TRIAL_SEED, glyphs=glyphs,
            fmt=FORMAT, batch_size=EVAL_BATCH, device=device,
            order_seed=7000 + int(TRIAL_SEED),
            placebo=PLACEBO, novel=NOVEL,
            max_new_tokens=run_orgb.DEFAULT_CONFIG["max_new_tokens"])
        comp = out["competence"]
        say(f"  MEASURED {tag}: {len(eval_mazes)} mazes, solve "
            f"{comp['solve_rate']:.3f}, mine/ep {comp['mine_step_rate']:.3f} "
            f"in {_hms(time.perf_counter() - t_m)}")
        return out

    # ---- the REFERENCE: the warm start itself, measured once ---------------
    # It is what cell A is compared to. Not an arm and never a candidate for the
    # pick: no RL ran at all, so it has no trace, no zero-advantage fraction and
    # no verdict -- it is the number the verdicts are differences FROM.
    say(f"REFERENCE: the warm-started model on {len(eval_mazes)} held-out "
        f"mazes, before any arm. Every cell-{TRIAL_CELL} verdict is a "
        f"difference from this row.")
    reference = None
    try:
        load_lora_state_dict(model, warm_state)
        ref_m = measure(TRIAL_CELL, "reference (warm start)")
        reference = {
            "arm": "00_reference_warm_start", "arm_base": "reference",
            "reference": True, "follow_up": False,
            "mode": None, "arm_note": ("the warm-started model, no RL: the "
                                       "D-equivalent baseline"),
            "cell": TRIAL_CELL, "seed": TRIAL_SEED, "status": "DONE",
            "explore": None, "preflight_exploration": None, "history": None,
            "eval": eval_block(ref_m),
            "config": {"group_size": None, "warm_start": ws_path},
            "model_key": MODEL_KEY, "model_id": model_id, "revision": REVISION,
        }
        drv.write_payload(os.path.join(OUT, "reference.json"), reference)
        say(f"  WROTE {os.path.join(OUT, 'reference.json')}")
    except Exception:                                      # noqa: BLE001
        # A missing reference costs the cell-A verdicts, not the traces: the
        # arms still run and the scorer says NO-REFERENCE rather than guessing.
        say("  !! REFERENCE MEASUREMENT FAILED:\n" + traceback.format_exc())

    def run_arm(arm, cell, tag=None, pinned_target=None):
        """One arm from the warm start. -> its payload (also on disk).

        `pinned_target` is how the FOLLOW-UP cell gets the picked arm's EXACT
        plan (critic pass 2). Re-deriving `relative`'s target from a second
        pre-flight is deterministic on CPU but not on CUDA bf16, so cell A would
        have trained under a target a hair from the one the trial picked -- and
        under `MZ01_ENTROPY_TARGET` that hair is a different recipe to every
        resume check in the programme. The pre-flight still RUNS (it is a
        measurement of the same warm start under this arm's sampler); only the
        target it resolves is pinned.

        TWELFTH PASS, THE MID-ARM READS. `MZ01_TRIAL_EVAL_AT` names the steps;
        each read is `measure()`, on the same `eval_mazes`, through the same
        `eval_block`, and it lands in `payload["eval_at"]`. It runs INSIDE the
        `on_step` hook, so it is the same organism's own trajectory rather than
        a second arm re-run to a shorter horizon -- which is the only way "was
        it AVOIDS at 40 and RANDOMISED at 80" is a question about one thing. It
        cannot disturb the training stream: `measure_organism` derives its word
        orders from `order_seed` (D31) and decodes greedily, so it draws from no
        generator at all, and the trainer's own `gen` is never passed to it.
        """
        name = tag or arm["name"]
        path = os.path.join(OUT, f"{name}.json")
        group = arm["group"] or GROUP
        # THE ARM'S OWN KNOBS, from the registry, falling back to the
        # environment's. An arm is a NAMED recipe: `temperature_t20` means
        # T=2.0 whatever MZ01_SAMPLE_T happens to be, or two arms of the same
        # trial could silently be the same arm.
        arm_t = float(arm.get("sample_temperature")
                      or os.environ.get("MZ01_SAMPLE_T")
                      or maze_orgb.SAMPLE_TEMPERATURE)
        arm_eps = float(arm.get("epsilon")
                        or os.environ.get("MZ01_EPSILON")
                        or maze_orgb.EXPLORE_EPSILON_RATE)
        # FOURTEENTH PASS, resolved exactly as the rate above is: the ARM's own
        # first, because `epsilon_030_near` means the targeted draw whatever
        # MZ01_EPSILON_WHERE happens to be, or two arms of one trial could
        # silently be the same arm.
        # RESOLVED here, so an arm that draws at 0 (every control) records `all`
        # rather than a targeting rule it never applied -- the same reading
        # `plan_epsilon` gives the rate, and the trainer refuses the other one.
        arm_where = maze_orgb.plan_epsilon_where(
            arm["mode"],
            (arm.get("epsilon_where")
             or os.environ.get("MZ01_EPSILON_WHERE")
             or maze_orgb.EPSILON_WHERE_ALL),
            epsilon=arm_eps)
        arm_off = bool(arm.get("thermostat_off"))
        # FIFTEENTH PASS, resolved the way the rate and the targeting rule are:
        # the ARM's own switch first, because `fork_cf_g6` means the forced
        # counterfactual whatever MZ01_FORK_CF happens to be. The four knobs
        # beside it are the run's and are inert with the switch off -- the
        # trainer refuses a run that sets one without the other.
        arm_fork = bool(arm.get("fork_cf", FORK_CF))
        # SEVENTEENTH PASS. The dose reaches only a cell whose return prices
        # mines at all: D and B charge zero for a mine by definition, and the
        # trainer refuses a run that records a price it never charged.
        arm_mine_price = (MINE_PRICE
                          if maze_orgb.CELL_SWITCHES[cell]["mine_cost"]
                          else None)
        arm_dosed = (arm_mine_price is not None
                     and arm_mine_price != maze2.SCORE_MINE)
        load_lora_state_dict(model, warm_state)
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cfg = {"cell": cell, "seed": TRIAL_SEED, "steps": TRIAL_STEPS,
               "n": GRID_N, "fmt": FORMAT, "batch_size": BATCH,
               "group_size": group, "chunk_size": CHUNK,
               "sample_batch_size": (int(SAMPLE_BATCH) if SAMPLE_BATCH
                                     else None),
               "pool_size": len(train_pool),
               "eval_n": len(eval_mazes), "eval_batch_size": EVAL_BATCH,
               "eval_at": list(read_at), "move_mask": MOVE_MASK,
               "move_mass_coef": move_mass_coef,
               "exit_at_cap": EXIT_AT_CAP,
               # FOURTEENTH PASS. The run's credit rule and the ARM's own
               # targeting rule, in the trial's own block -- the first of the
               # three places `score_mz01_explore._action_space` looks.
               "credit": CREDIT,
               "epsilon_where": arm_where,
               # FIFTEENTH PASS. The arm's own switch and the run's knobs, in
               # the trial's own block beside them. Written whatever the switch
               # says, so a reader of two arm files can see which of them ran
               # the fork term without knowing this build's default.
               "fork_cf": arm_fork,
               "fork_per_ep": FORK_PER_EP if arm_fork else None,
               "fork_coef": FORK_COEF if arm_fork else None,
               "fork_std": FORK_STD if arm_fork else None,
               "fork_temperature": FORK_TEMP if arm_fork else None,
               # SIXTEENTH PASS. The two guards, written beside the knobs they
               # guard. `fork_mass_coef` is written RESOLVED -- the run's own
               # `move_mass_coef` when the environment left it empty -- so a
               # reader of the row never has to re-derive the default.
               "fork_grad_max": FORK_GRAD_MAX if arm_fork else None,
               "fork_mass_coef": ((move_mass_coef if FORK_MASS_COEF is None
                                   else FORK_MASS_COEF) if arm_fork else None),
               # EIGHTEENTH PASS. Each only off its default, so a default arm's
               # row is byte for byte what it was.
               **({"fork_pick": FORK_PICK}
                  if arm_fork and FORK_PICK != "uniform" else {}),
               **({"fork_select": FORK_SELECT}
                  if arm_fork and FORK_SELECT != "none" else {}),
               # SEVENTEENTH PASS. The dose, in the trial's own block and
               # ONLY when it is not the world's price -- an undosed arm's row
               # is byte for byte what it was.
               **({"mine_price": arm_mine_price} if arm_dosed else {}),
               "placebo": PLACEBO, "novel": NOVEL,
               "warm_start": ws_path, "nav_adapter": NAV_ADAPTER,
               "nav_adapter_sha256": nav.get("adapter_sha256"),
               "role_glyphs": dict(roles), "git_sha": git_sha()}
        say(f"ARM {name}: cell {cell} {maze_orgb.CELL_SWITCHES[cell]}, mode "
            f"{arm['mode']}, GROUP={group} -- {arm['note']}")
        # FIFTEENTH PASS. The switch is NAMED WHEN IT IS ON, as `move_mask` and
        # the return rule are named when they are off: a line that says nothing
        # is the default, and this one is not.
        if arm_fork:
            say(f"  fork counterfactual ON: {FORK_PER_EP} fork(s) per episode, "
                f"coef {FORK_COEF}, sd division {'on' if FORK_STD else 'off'}, "
                f"continuations at T={FORK_TEMP}. The continuations are NOT "
                f"trained; the fork decision alone carries their difference")
            # SIXTEENTH PASS. The guards are named on the line the term is, or
            # an operator reading the log would price the step by the
            # coefficient alone -- which is the number that did not hold it.
            say(f"  fork guards: gradient cap "
                + (f"{FORK_GRAD_MAX} on the fork term ALONE, before the total "
                   f"clip" if FORK_GRAD_MAX > 0 else "OFF (the 09-10 recipe)")
                + f"; mass guard coef "
                + (f"{cfg['fork_mass_coef']} at the fork prompts"
                   if cfg["fork_mass_coef"] else "OFF"))
            if FORK_PICK != "uniform" or FORK_SELECT != "none":
                say(f"  fork share: pick {FORK_PICK}, select {FORK_SELECT}")
        # SEVENTEENTH PASS, named for the same reason: a dosed arm's returns are
        # not the world's, and the log is where an operator finds that out.
        if arm_dosed:
            say(f"  mine priced at {arm_mine_price} IN TRAINING (the world's "
                f"{maze2.SCORE_MINE} is unchanged, and so is every held-out "
                f"number: score, mean_score, the price table, mine_step_rate)")
        pre = maze_orgb.preflight_exploration_check(
            model, tok, eval_set[:PREFLIGHT_MAZES],
            generator=derive_generator(0, "orgb", "explore"),
            group_size=group, glyphs=glyphs, fmt=FORMAT, chunk_size=CHUNK,
            sample_batch_size=cfg["sample_batch_size"], device=device,
            temperature=1.0, explore_mode=arm["mode"],
            entropy_coef=run_orgb.DEFAULT_CONFIG["entropy_coef"],
            entropy_target=run_orgb.DEFAULT_CONFIG["entropy_target"],
            entropy_rel=float(os.environ.get("MZ01_ENTROPY_REL")
                              or maze_orgb.ENTROPY_REL),
            entropy_min=float(os.environ.get("MZ01_ENTROPY_MIN")
                              or maze_orgb.ENTROPY_MIN),
            entropy_coef_fixed=float(os.environ.get("MZ01_ENTROPY_COEF")
                                     or maze_orgb.ENTROPY_COEF_FIXED),
            sample_temperature=arm_t, epsilon=arm_eps,
            # FOURTEENTH PASS: and at this arm's own targeting rule, for the
            # reason it rolls out at the arm's own epsilon -- the rule is part
            # of the sampler.
            epsilon_where=arm_where,
            thermostat_off=arm_off,
            pinned_target=pinned_target,
            # SOFT, always: the trial's whole point is to run the arms that
            # today's gate refuses, and the CONTROL is the one that fails it
            # hardest -- it is today's recipe, and the refusing criterion is
            # precisely the number that says today's recipe does not learn. A
            # gate that stopped the control would stop the trial from measuring
            # the thing it exists to measure. The refusal is logged and recorded
            # as a deviation in every arm's payload instead.
            soft=True,
            # TWELFTH PASS: and under this arm's own ACTION SPACE, for the same
            # reason it rolls out at the arm's own temperature and epsilon.
            move_mask=MOVE_MASK,
            # THIRTEENTH PASS: and under the arm's own RETURN. The criterion
            # this gate refuses on is computed from the episode returns.
            exit_at_cap=EXIT_AT_CAP,
            # FOURTEENTH PASS: and under the run's own CREDIT. That criterion is
            # now read off the per-decision advantages, which is what the arm's
            # move term will multiply.
            credit=CREDIT)
        for line in pre["lines"]:
            say("  " + line)
        plan = pre["plan"]
        reads = {}

        def on_step(step, history):
            # THE MID-ARM READ, FIRST, so the payload this hook writes already
            # carries it -- an arm killed between the read and the next step
            # must not lose the three minutes it just spent. BEST EFFORT for the
            # hook's own reason: a measurement that raised must not make the
            # instrumentation the thing that loses the arm.
            if (step + 1) in read_at:
                try:
                    say(f"  MID-ARM READ at step {step + 1}/{TRIAL_STEPS} "
                        f"({len(eval_mazes)} held-out mazes, the SAME mazes "
                        f"and the same call as the final read)")
                    reads[str(step + 1)] = eval_block(
                        measure(cell, f"{name} @{step + 1}"), step=step + 1)
                except Exception:                          # noqa: BLE001
                    say(f"  !! MID-ARM READ FAILED at step {step + 1}:\n"
                        + traceback.format_exc())
            drv.write_payload(path, arm_payload(
                arm, cell=cell, seed=TRIAL_SEED, plan=plan, preflight=pre,
                history=history, status=f"RUNNING step {step + 1}/{TRIAL_STEPS}",
                cfg=cfg, started_at=started, model_id=model_id,
                name=name, follow_up=bool(tag), eval_at=reads))

        hist = maze_orgb.train_orgb_cell(
            model, tok, cell=cell, seed=TRIAL_SEED, steps=TRIAL_STEPS,
            n=GRID_N, batch_size=BATCH, group_size=group,
            lr=run_orgb.DEFAULT_CONFIG["lr"], temperature=plan["temperature"],
            yell_weight=run_orgb.DEFAULT_CONFIG["yell_weight"],
            max_new_tokens=run_orgb.DEFAULT_CONFIG["max_new_tokens"],
            entropy_coef=plan["entropy_coef"],
            entropy_target=plan["entropy_target"],
            explore_mode=plan["mode"], explore_knobs=plan["knobs"],
            sample_temperature=plan["sample_temperature"],
            epsilon=plan["epsilon"], epsilon_where=plan["epsilon_where"],
            sample_batch_size=cfg["sample_batch_size"],
            move_mass_coef=move_mass_coef, move_mask=MOVE_MASK,
            exit_at_cap=EXIT_AT_CAP, credit=CREDIT,
            # SEVENTEENTH PASS. `None` on a cell that prices mines at zero, and
            # `None` is also the trainer's own "the world's price", so an
            # undosed arm passes exactly what it passed before.
            mine_price=arm_mine_price,
            # FIFTEENTH PASS. The knobs are passed ONLY with the switch on: the
            # trainer refuses a run that names a fork setting it never applied,
            # which is what keeps an arm's recorded recipe the recipe that ran.
            fork_cf=arm_fork,
            **({"fork_per_ep": FORK_PER_EP, "fork_coef": FORK_COEF,
                "fork_std": FORK_STD, "fork_temperature": FORK_TEMP,
                # SIXTEENTH PASS. `fork_mass_coef=None` is the trainer's own
                # "use `move_mass_coef`", so an empty MZ01_FORK_MASS_COEF is
                # passed through as None rather than resolved here.
                "fork_grad_max": FORK_GRAD_MAX,
                "fork_mass_coef": FORK_MASS_COEF,
                "fork_pick": FORK_PICK, "fork_select": FORK_SELECT}
               if arm_fork else {}),
            chunk_size=CHUNK, train_pool=train_pool, glyphs=glyphs,
            fmt=FORMAT, log_every=1,
            # NO CHECKPOINT: an arm is a few dozen steps and is discarded
            # whatever happens. A 126 MB write every 50 steps would buy nothing
            # and would land in the results tree the puller mirrors.
            ckpt_path=None, ckpt_every=0, on_step=on_step)
        say(f"  ARM {name} trained in {_hms(hist['total_seconds'])} "
            f"({hist['seconds_per_step']:.1f} s/step); zero-advantage groups "
            f"mean {hist['zero_advantage_group_fraction_mean']}, train solve "
            f"{hist['solve_rate'][-1]:.3f}, train mine/ep "
            f"{hist['mine_step_rate'][-1]:.3f}, H {hist['policy_entropy'][-1]:.3f}, "
            f"coef {hist['entropy_coef'][-1]:.4f}, eps replaced "
            f"{hist['epsilon_replaced_fraction_mean']} of eligible "
            f"{hist['epsilon_eligible_fraction_mean']}, |A| "
            f"{hist['adv_abs_mean'][-1]:.3f} nonzero "
            f"{hist['adv_nonzero_share'][-1]:.3f}")
        if arm_fork and hist.get("n_forks"):
            # PLAN section 6's reading rule is written about these two numbers:
            # a fork-advantage share above 0.5 with no AVOIDS says the
            # comparison exists and the price does not move the policy; below
            # 0.1 says the geometry is not picking the forks the mines are at.
            n_f = sum(hist["n_forks"])
            n_c = sum(hist["n_continuations"])
            live = [x for x in hist["fork_share_nonzero"] if x is not None]
            det = [x for x in hist["fork_share_detour_better"] if x is not None]
            share = f"{sum(live) / len(live):.3f}" if live else "n/a"
            say(f"  forks {n_f} over {len(hist['n_forks'])} steps, {n_c} "
                f"continuations ({n_c / max(1, n_f):.1f} per fork); share with "
                f"a difference {share}")
            if det:
                say(f"  fork detour-better share {sum(det) / len(det):.3f} "
                    f"(DIAGNOSTIC, read after the fact and never in the update)")
            # SIXTEENTH PASS. What the guards actually did over the arm: the
            # fork term's own pre-cap gradient norm (the number that owned the
            # step on 09-10) and the bare move mass at the prompts it acts on.
            raw = [x for x in hist.get("fork_grad_norm_raw") or [] if x is not None]
            fm = [x for x in hist.get("fork_mass_mean") or [] if x is not None]
            if raw or fm:
                say("  fork guards: pre-cap |g| "
                    + (f"mean {sum(raw) / len(raw):.1f}, max {max(raw):.1f} "
                       f"(cap {FORK_GRAD_MAX})" if raw else "not separated")
                    + "; bare mass at the fork prompts "
                    + (f"{fm[0]:.3f} -> {fm[-1]:.3f}" if fm else "n/a"))
        # ---- THE HELD-OUT READ, at the end and only at the end -------------
        # The training numbers above are on the POOL and under the arm's own
        # sampler; an epsilon arm's training `mine_step_rate` includes the mines
        # epsilon itself walked onto. The verdicts are about what the organism
        # DOES, greedily, on mazes it never trained on.
        meas = None
        try:
            meas = measure(cell, name)
        except Exception:                                  # noqa: BLE001
            say(f"  !! MEASUREMENT FAILED for {name}:\n"
                + traceback.format_exc())
        payload = arm_payload(
            arm, cell=cell, seed=TRIAL_SEED, plan=plan, preflight=pre,
            history=hist, status="DONE", cfg=cfg, started_at=started,
            model_id=model_id, name=name, follow_up=bool(tag),
            measurement=meas, eval_at=reads)
        drv.write_payload(path, payload)
        say(f"  WROTE {path}")
        return payload

    for name in TRIAL_ARMS:
        arm = next(a for a in ARMS if a["name"] == name)
        try:
            results[name] = run_arm(arm, TRIAL_CELL)
            done.append(name)
        except Exception:                                  # noqa: BLE001
            # ONE ARM MUST NOT COST THE OTHERS. An OOM at GROUP=12 is a real
            # possible outcome of this trial and is itself an answer.
            say(f"  !! ARM {name} FAILED:\n" + traceback.format_exc())
            failed = {"arm": name, "arm_base": name, "follow_up": False,
                      "reference": False,
                      "mode": arm["mode"], "cell": TRIAL_CELL,
                      "seed": TRIAL_SEED, "status": "FAILED",
                      "error": traceback.format_exc(), "history": None,
                      "eval": None, "eval_at": {}}
            results[name] = failed
            drv.write_payload(os.path.join(OUT, f"{name}.json"), failed)
        say(f"  progress {len(done)}/{len(TRIAL_ARMS)} arms, elapsed "
            f"{_hms(time.perf_counter() - t0)}")

    # ---- the summary, by the SAME code the human reads the table with -------
    score = _scorer()
    payloads = list(results.values()) + ([reference] if reference else [])
    rows, ref_row = score.read_all(payloads)
    pick = score.pick_arm(rows)
    summary = {
        "arms": rows,
        "reference": ref_row,
        "pick": pick,
        "env_line": (score.env_line(pick, cells="A,C") if pick else None),
        "env_lines": (score.env_lines(pick) if pick else None),
        "steps": TRIAL_STEPS,
        "cell": TRIAL_CELL,
        "seed": TRIAL_SEED,
        "eval_n": len(eval_mazes),
        "eval_at": list(read_at),
        "move_mask": MOVE_MASK,
        "move_mass_coef": move_mass_coef,
        "exit_at_cap": EXIT_AT_CAP,
        # FOURTEENTH PASS. The run's credit rule; `epsilon_where` is per ARM and
        # lives in each arm's own row.
        "credit": CREDIT,
        "bars": {"zero_advantage": score.ZERO_ADVANTAGE_BAR,
                 "solve_drop": score.SOLVE_DROP_BAR,
                 "mine_ratio": score.MINE_RATIO_BAR,
                 "solve_band": score.SOLVE_BAND,
                 # TWELFTH PASS: the validity rule. A held-out read whose
                 # `unparsed_rate` is above this is an INVALID-READ and is
                 # reported rather than verdicted.
                 "unparsed": score.UNPARSED_BAR},
    }
    for line in score.table(rows, pick, reference=ref_row):
        say("  " + line)
    drv.write_payload(os.path.join(OUT, "summary.json"), summary)
    say(f"WROTE {os.path.join(OUT, 'summary.json')}")

    # ---- and the question the trial is actually for ------------------------
    # SKIPPED when the arms already ran the follow-up cell: a cell-A trial has
    # asked the cell-A question and running A again under the picked A arm is
    # the same recipe twice.
    if run_follow_up and pick:
        arm = next(a for a in ARMS if a["name"] == pick["arm"])
        # A mode that runs no thermostat has no target to pin and
        # `exploration_plan` refuses one; every other mode pins the number the
        # picked arm actually trained under.
        pin = (None if not pick.get("entropy_target")
               else pick.get("entropy_target"))
        say(f"FOLLOW-UP: cell {TRIAL_CELL_2} under the picked arm "
            f"{pick['arm']} -- 'the entropy moved' is not the question, "
            f"'does mine/ep move' is"
            + (f"; entropy target PINNED at {pin}" if pin is not None else ""))
        try:
            follow = run_arm(arm, TRIAL_CELL_2, tag=f"{TRIAL_CELL_2}_{pick['arm']}",
                             pinned_target=pin)
            summary["follow_up"] = score.read_arm(follow, reference=ref_row)
            drv.write_payload(os.path.join(OUT, "summary.json"), summary)
        except Exception:                                  # noqa: BLE001
            say("  !! FOLLOW-UP FAILED:\n" + traceback.format_exc())
    elif run_follow_up:
        say(f"FOLLOW-UP SKIPPED: no arm was picked, so there is no recipe to "
            f"run cell {TRIAL_CELL_2} under. That is the trial's answer and it "
            f"is a NEGATIVE one: none of these arms gives the move term a "
            f"gradient that survives at this scale.")
    elif TRIAL_CELL_2:
        say(f"FOLLOW-UP SKIPPED: the arms already trained cell {TRIAL_CELL}, "
            f"which is the follow-up cell. Running it again under the picked "
            f"arm would measure the same recipe twice.")

    say(f"MZ01 EXPLORE TRIAL DONE arms={len(done)}/{len(TRIAL_ARMS)} "
        f"pick={(pick or {}).get('arm')} elapsed={_hms(time.perf_counter() - t0)}")
    for line in (summary["env_lines"] or []):
        say("  RECOMMENDED: " + line)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say("MZ01 EXPLORE TRIAL ERROR:\n" + traceback.format_exc())
        raise
