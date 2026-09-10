# BUILD — the fork decision's own probability, before and after the step (eighteenth pass)

**2026-09-10. Built against `PLAN-fork-counterfactual.md` §17.3 and §17.5 item
1, on top of `56621c5`, in an isolated worktree. Nothing was launched, nothing
under `results/` was touched, no PLAN file was edited, and no GPU box was
contacted.**

**Amended 2026-09-10 after the critic's APPROVE WITH CHANGES on `c4d4ba7`.**
Two changes, both inside the diagnostic: the signed per-fork change is recorded
beside its MAGNITUDE, because the signed mean cancels across a step's forks and
would be a false confirmation of §17.3's hypothesis (C1, §2 and §4); and a TIED
fork — where the term pushes nowhere — is no longer scored at all, with the
means' own denominator recorded (C2, §4). Both optional items were taken:
`fork_probe_seconds`, and the `labels` note in §7. Three statements in this
report were wrong and are corrected in place, marked **CORRECTED**.

**No switch was added.** With `MZ01_FORK_CF` off nothing below runs and the
trace, the loss and the adapter are byte for byte what they were; with it on the
seven fields are always written. §5 below justifies that call with a
measurement.

---

## 0. What §17.3 claims and why it could not be read

The dose tripled the price's share of the fork advantage exactly as designed
(8.03 measured against 8.04 predicted, §16.2) and the held-out mine rate did not
follow on any arm. §17.3's account is mechanical: `policy_entropy` falls to
0.001–0.005 inside forty steps on every arm, priced or not, dosed or not, so the
move distribution is a point mass; `zero_advantage_group_fraction` is 0.80–0.95,
so the episode term is dead in most groups; and the fork term — the only live
gradient — is capped at norm 0.3 and is being asked to flip decisions the policy
holds above 0.999. §17.3 says out loud that this is a hypothesis and not a
measurement, **because nothing in the trace records the fork decision's own
probability.** This build records it.

## 1. What changed, by file and line

### `src/calibration/maze_orgb.py`

| line | what |
| --- | --- |
| 2103–2132 | `FORK_TRACE_FIELDS` gains the seven fields. They live here, as the guards' three do, so the trace still grows **exactly** `FORK_TRACE_FIELDS` with the term on and **nothing** with it off. |
| 2140–2150 | `_fmt_prob` — the log line's formatter, at four decimals. `_fmt_share`'s two would print 0.9995 and a change of 1e-4 as the same constant every step, which is the reading this pass exists to make possible. |
| 2429–2436, 2463–2468 | `fork_terms`' rows gain `"taken"` — the move the episode actually played at that fork, copied off the fork the geometry found. Documented in the docstring as read by the DIAGNOSTIC alone; the loss is built from `advantages` and `moves`, as before. |
| 2554–2648 | **`_fork_decision_probs(mlogp, rows)`** — `(p_target, p_taken)` per fork row off an already-computed `mlogp`, `(None, None)` at a TIED fork. Its conventions are written out in the docstring, §4 below. |
| 3186–3225 | The trainer docstring's section on the seven fields, what each means, which forks they are over, and why there is no switch. |
| 3808–3813, 3856–3861 | **The "before" half, free.** `fork_p_before` is accumulated inside the fork term's own chunk loop, off the very `mlogp` the fork gradient is taken at, under the `no_grad` block that already reads the mass. No forward pass of its own. |
| 3985–4054 | **The "after" half** — one extra forward pass at the SAME prompts, in the SAME chunks, under `no_grad`, immediately after `optimiser.step()`, on its own clock. The scored forks are selected and the seven numbers formed here. |
| 4029–4045 | The trace append, inside the same `if fork_cf:` block and on the same once-per-step schedule as every other fork field. |
| 4166–4175 | The log line's `pi* b->a d ±x pi_taken k`, only on the arm the fork term runs on. |
| 4547–4549 | `history["config"]["fork"]["mine_information"]` says what the `fork_p_*` fields read (the policy, at the fork prompts) and what they do not (a maze). |

### `scripts/mz01_explore_trial.py`

| line | what |
| --- | --- |
| 186–207 | The module docstring's env list gains a paragraph under the fork knobs: **no switch was added**, what the seven fields are, what they cost, and that nothing reads them back. |
| 1117–1136 | The end-of-arm summary prints them beside the existing fork reads: π(pushed move) first step → last, the mean and largest per-step change, and π(the move the episode took). |

### `docs/CONSTRUCTION.md`

§5.8 gains the trace-field list for the fork recipe (the six of the fifteenth
pass, the guards' three, and these seven), and the construction-test table row
for the forced counterfactual records that the `fork_p_*` diagnostic reads the
policy and the advantages and no maze at all.

### `tests/test_maze_orgb.py`

Lines 6900–7306: the eighteenth-pass section — one helper pair and five tests
(§6).

## 2. The seven fields

| key | value |
| --- | --- |
| `fork_p_target_before` | the mean over the step's SCORED forks of π(the move with the greatest `A(a)` at that fork \| the fork prompt), read **before** the optimiser step |
| `fork_p_target_after` | the same probability, at the same prompts and in the same chunks, **after** the optimiser step |
| `fork_p_target_delta` | the **signed** mean over scored forks of `after − before` |
| `fork_p_target_absdelta` | the mean over scored forks of `\|after − before\|` |
| `fork_p_taken_before` | the same before-probability for the move the **episode actually took** at the fork, so "the term pushed away from what it did" is separable from "the term pushed toward what it already did" |
| `fork_p_n_scored` | how many of the step's forks the means are over (the tied ones are `n_forks − fork_p_n_scored`) |
| `fork_probe_seconds` | what the extra forward pass cost, on its own clock |

**The signed delta and the magnitude are read together or not at all** (critic,
C1). With one denominator the signed mean is algebraically just
`fork_p_target_after − fork_p_target_before`, so it carries nothing the two
levels lack — and it **cancels**: the step's ~24 forks share one set of LoRA
weights, so twelve moving +0.01 toward their pushed move and twelve moving
−0.01 read out as `+0.000000`. That is exactly the number §17.3's hypothesis
predicts, and §17.4/§17.5 make the next arm conditional on that reading, so a
cancelling mean read alone would launch an arm on a false confirmation.
`fork_p_target_absdelta` is the number that cannot be faked that way. It is in
**this** build and not a follow-up, because adding it later would change
`FORK_TRACE_FIELDS` again and orphan the arm's trace.

All the probabilities are `None` on a step with no fork **and on a step whose
every fork tied**; `fork_p_taken_before` is `None` also where no scored fork's
taken move was one of its compared moves. They are written once per step, inside
the same `if fork_cf` block every other fork field is written in, so **every
array in the trace stays index-aligned with `step`**.

## 3. Where they are documented

* `maze_orgb.FORK_TRACE_FIELDS`, in the comment beside the names;
* `train_orgb_cell`'s docstring — "THE FOUR NUMBERS THE PROBABILITY DIAGNOSTIC
  ADDS", one bullet each, and the no-switch decision;
* `_fork_decision_probs`' own docstring — the two conventions and the
  construction test;
* the trial's module docstring, in the env list beside the `MZ01_FORK_*` knobs
  (which is where an operator looks), saying explicitly that no variable was
  added;
* `docs/CONSTRUCTION.md` §5.8 and its construction-test table;
* `history["config"]["fork"]["mine_information"]`, for a reader of the JSON;
* and `test_the_fork_decision_probability_is_recorded_before_and_after_the_step`
  asserts each field name is in **both** the trial's docstring and the
  trainer's, so a rename cannot silently orphan the documentation.

## 4. The conventions, and they are interpretations

The plan writes "the move the fork term is pushing hardest toward (the move with
the greatest advantage A(a) at that fork)" and "the move the episode actually
took". Three cases it does not decide, decided here and written into the
docstring:

1. **A TIED FORK IS NOT SCORED** (critic, C2 — this reverses the first build's
   call, and the first build's stated remedy was false). A fork whose
   continuations all returned the same number has advantages of exactly zero:
   the term pushes NOWHERE, so "the move it is pushing hardest toward" does not
   exist, and an argmax over four zeros records the first move in `MOVE_WORDS`
   order and nothing else. `c4d4ba7` scored such a fork anyway and said a reader
   could size the dilution off `fork_share_nonzero`; **that is false**, because
   knowing the share of ties does not let you remove their contribution when the
   ties' own mean probability is not recorded. The critic sized it on the live
   arm: `fork_share_nonzero` mean 0.8936 on `forkcf4b_A_s0_g10`, so 10.6 % of
   forks are ties, and on a collapsed policy where the first valid move is the
   argmax about 1/2.37 of the time, a reported 0.069 could really be 0.03. Both
   halves of a tied fork's pair are now `None` and it is dropped from numerator
   and denominator alike. **The test is the one `fork_terms` already makes** —
   the continuations' return spread against `FORK_ADVANTAGE_TOL` — so the means'
   population is exactly `fork_share_nonzero`'s, and `fork_p_n_scored` records
   its size so the two populations stay separable.
2. **`p_taken` is restricted to the same scored forks**, which the critic's C2
   did not ask for and this build did anyway: at a tied fork the term pushes
   nothing, so "pushed toward what it already did" is as undefined there as
   "the move it pushes toward", and a `p_taken` averaged over a *different*
   population from `p_target` could not be read beside it — which is the one
   thing that field exists for. One line to revert (`_fork_decision_probs`, the
   `continue` on the tie test) if the lead wants the wider population.
3. **An episode that bumped a wall or stepped back at its fork** has no
   continuation of its own, so `p_taken` is `None` for that fork and it is left
   out of that mean's denominator rather than scored against a move the term
   never compared.

   **CORRECTED — that is NOT `fork_share_detour_better`'s denominator any more**
   (critic, 2026-09-10). It was, before item 1's tie rule, and `c4d4ba7` said so
   one paragraph above the restriction that broke it. `fork_terms` increments
   `comparable` for every row whose taken move has a continuation, **outside**
   the `nonzero` test, so `fork_share_detour_better`'s denominator **includes
   tied forks**, while `fork_p_taken_before`'s is now the forks that are scored
   **and** comparable — a strict subset. The two numbers will be read side by
   side, so the difference is stated here and in `_fork_decision_probs`'
   docstring rather than left to be rediscovered; `fork_p_n_scored` and
   `fork_share_nonzero` are what size it.
4. **Ties among the advantages of a fork that is NOT tied** (two moves sharing
   the maximum with a third below them) go to the first in `moves` order. Either
   is "the move the term pushes hardest toward"; the choice is deterministic.

## 5. Why there is no switch, with the cost

The "before" half costs **nothing**: it is read off the `mlogp` the fork term is
already about to take its gradient from. The "after" half is **one forward pass
per chunk of fork prompts** — at 24 forks and `chunk_size` 8, three forwards with
no backward — against a step that already pays ~28.8 s of continuation sampling
(§13.1's measured `fork_sample_seconds` of a 54.8 s step), plus a chunked
forward+backward over ~300 episode decisions, plus a chunked forward+backward
over those same 24 fork prompts. A backward is roughly twice a forward, so the
added work is about a third of what the fork term's own rescore already costs at
those prompts — itself ~24 of the step's ~324 scored prompts.

Measured on the toy fixture (`tests/test_maze_orgb.py`'s own two-step cell A),
where the step contains no real sampling at all and so *over*-states the share:
**the probe is 1.1 % of `step_seconds`** (0.0040 s of 0.359 s over two steps, on
an idle container; 1.5 % on a contended one). On the 4B arm, where 28.8 s of the
54.8 s step is continuation sampling, the same work is well under 1 % — a
fraction of a second.

**And on the arm it will be a number rather than an estimate:** the probe is
timed on its own clock and recorded as `fork_probe_seconds` (the critic's first
optional item, taken). It is NOT folded into `t_rescore`, because that would
inflate `rescore_seconds` and break its comparability with every earlier arm;
`fork_sample_seconds` made the same choice for the same reason. `step_seconds`
does include it, which is honest — it is part of what a step costs.

Against that: a diagnostic behind a default-off switch is a diagnostic that is
off on the arms that ran, and §17.5 item 1 puts this build **before** the lever
it is meant to inform. **Always on with the fork term is the call.** It costs
exactly nothing when the fork term is off, because the whole block is inside
`if fork_cf`.

## 6. The tests

Added to `tests/test_maze_orgb.py` (line 6900 onward).

| test | claim |
| --- | --- |
| `test_the_fork_decision_probability_is_recorded_before_and_after_the_step` | The seven fields exist with the term on and not with it off; `fork_p_n_scored` is `fork_share_nonzero × n_forks` (one tie test, made in two places); `absdelta ≥ \|delta\|`; the trace grows **exactly** `FORK_TRACE_FIELDS`; one entry per step; the numbers are probabilities and the delta is the paired difference; and — by spying on `_fork_move_logp` and reading the model's own LoRA state at each call — the "after" pass is made at the SAME prompts, one per chunk and no more, in the same `eval` mode, **with weights the optimiser has already moved**. It also asserts every field name is in the trial's docstring and in the trainer's. |
| `test_the_fork_probability_diagnostic_never_reads_the_mines` | **The construction test (§1), in the style of §5 test 2.** With `is_mine`, `at`, `mines` and `rows` (the layout, which spells the mines) all patched onto the maze CLASS to raise — so no instance, frozen dataclass or copy can answer — the diagnostic computes and returns exactly the numbers it returns against the seeing world. The blinding is asserted to bite first, on the very maze the fork came from. The fixture is the priced fork, where the pushed move (`right`, the detour) and the taken move (`down`, onto the mine) DIFFER, so the pair is not one number twice. |
| `test_the_fork_probability_probe_draws_no_random_number` | **The main risk.** Every key of `_rng_snapshot` — the trainer's own list of the streams a resume must restore — is identical across the probe; no weight moved and no `.grad` was created; the tensor it reads carries no gradient. **CORRECTED:** the first version of this table claimed the CUDA stream among them. The helper now compares it, but on a CPU-only host `_rng_snapshot` puts an EMPTY LIST there, so that comparison is **structural and is not exercised**; what the suite exercises is python, numpy, torch and the run's own generator, which is where this trainer's draws come from (`torch.multinomial` on a CPU generator). Said in the helper's own docstring too. |
| `test_a_tied_fork_is_not_scored_by_the_probability_diagnostic` | **C2.** A row whose continuations returned the same number — and one whose spread is inside `FORK_ADVANTAGE_TOL` — scores `(None, None)` on both halves, while a live row beside it in the same `mlogp` scores the greatest-advantage move at its own row. And a tie among the advantages of a fork that is not tied still resolves, to the first such move in `moves` order. |
| `test_the_probability_diagnostic_is_read_by_nothing` | **Diagnostics only.** A run whose every fork's `taken` is replaced by a word that is not a move at all trains to the same `loss`, the same `grad_norm`, the same `fork_adv_abs_mean` and the same adapter sha — while `fork_p_taken_before` goes to `None`, which is what says the fixture could have noticed. And `_fork_decision_probs` returns plain floats off a detached copy, so nothing it returns can carry a gradient. |

**Runs.** The container this was built in has **no `.venv`**: the repository's
pinned local environment (Python 3.14.7, torch 2.13.0+cpu, numpy 2.5.2, pytest
9.1.1 — `requirements-local.txt`, `docs/OPERATIONS.md` §0) is not installed and
`python3 -m pytest tests/ -q` failed at `conftest.py` with `ModuleNotFoundError:
No module named 'torch'`. `numpy 2.4.6`, `pytest 9.1.1` and **`torch
2.14.0+cpu`** were installed into the container's Python 3.11 to make the suite
runnable at all, so **every number below is off a torch and numpy the repository
does not pin**:

```
BEFORE (HEAD 56621c5, same container):  4 failed, 284 passed, 1 skipped
AFTER  (c4d4ba7, the first build):      4 failed, 288 passed, 1 skipped
AFTER  (this amendment):                4 failed, 289 passed, 1 skipped
```

The four failures are **pre-existing in this container and are the environment,
not this build**: they are the four hard-coded bit-identity tests
(`test_move_mask_zero_...`, `test_exit_at_cap_off_...`,
`test_credit_episode_...`, `test_fork_off_is_the_pre_build_trace_...`), each
failing on a last-ulp float (`21.17647058823529` against
`21.176470588235293`, `0.009796617035443832` against `0.009796617032008986`).
That is precisely the case those tests' own comments name — "IF THIS EVER FAILS
AFTER A TOY OR A TORCH CHANGE rather than after a change to … re-derive it". The
failure text is **character for character identical before and after this
build** (`diff` of the two runs' `E ` lines: no difference), which is the
strongest form the byte-identity claim can take in this container.

**And the byte-identity claim itself, made directly.** Because
`test_fork_off_...` fails at its first assertion here and so never reaches its
later ones, the claim was also checked against the pre-build tree by running
`_exit_fixture_train()` in a clean clone of `56621c5` and in this worktree, in
separate processes, and comparing:

* **fork OFF** — the adapter sha is equal and **every** trace field is equal
  (only wall-clock fields excluded);
* **fork ON** — the adapter sha is equal and **every** pre-existing trace field
  is equal, including `loss`, `grad_norm`, `fork_grad_norm_raw`,
  `fork_grad_norm`, `fork_mass_mean` and `fork_adv_abs_mean`. The only
  difference in the whole payload is the **seven** new fields and the prose
  added to `config.fork.mine_information`.

**CORRECTED — what that second comparison does and does not prove.** The first
version of this report called it "the real RNG test", on the argument that a
random draw by the probe would have moved the fork-on adapter. That is wrong as
stated: `_exit_fixture_train` installs `_ScriptedSampler(_bumper_or_oracle)`,
and `_bumper_or_oracle` draws `torch.rand(1, generator=generator)` — the
EXPLICIT generator only — so nothing in that loop consumes the global torch
stream and a global-stream draw by the probe would have been invisible to the
comparison. What the comparison does prove is strong and is what it is cited
for here: the probe moved no weight, changed no recorded number, and did not
disturb **the run's own generator** (a draw from `gen` would have changed step
1's episodes and the adapter with them). The claim about the GLOBAL streams
rests on `test_the_fork_probability_probe_draws_no_random_number`, which
compares them directly, and on the mechanism: eval mode, no dropout in this
LoRA, `no_grad`.

## 7. The critic pass (PLAN §7), against §1–§5

1. **The construction test (§1).** The diagnostic's inputs are `mlogp` (the
   policy at the fork prompt) and the fork row's `moves`, `advantages` and
   `taken`. `taken` is the episode's own move word, off `episode_forks`, which
   reads `maze.neighbours` and the episode's positions. No maze object is in
   scope in `_fork_decision_probs` at all, which is why the blind test is
   possible and why it passes. **The one thing to watch** is the same one
   `BUILD-fork-counterfactual` §7 named: `fork_share_detour_better` is still the
   only fork-path quantity that reads mine counts, and it still feeds nothing;
   this build added a second consumer of the *row*, not of that quantity.
2. **No new behaviour (§5 test 4).** Verified twice: identical failure text on
   the four environment-bound bit-identity tests, and a direct fork-off AND
   fork-on comparison against a clean `56621c5` tree (§6). The fork-off log line
   was exercised too, and is the line it was.
3. **The loss.** The probabilities are floats built under `no_grad` off
   `mlogp.detach()`; `loss_val` is untouched; `a_fork` is built from
   `advantages` and `moves` as before. The "blanked `taken`" test is the
   mutation form of this claim.
4. **The trace's schedule.** Appended in the same `if fork_cf:` block, once per
   step, immediately after the guards' three — so nothing in the trace goes out
   of index alignment with `step`. Asserted per field.
5. **The RNG (§8.3's isolation rule).** The trainer holds the model in `eval`
   for the whole loop and this LoRA has no dropout, so a forward pass under
   `no_grad` consumes nothing from `gen`, from `fork_gen`, or from the three
   globals `_rng_snapshot` carries. The globals are asserted against
   `_rng_snapshot` itself in a dedicated test; `gen` is asserted end to end by
   the identical fork-on adapter above (see the correction in §6 — the two
   claims are separate and only the second is what the adapter shows).
6. **The construction test's own limitation, unchanged from §8.6 of the plan.**
   The blinding covers `is_mine`, `mines`, `at` and `rows`, **not** `labels`
   (which carries `n_mines`, `mine_on_shortest`, `detour_cost`) and not the
   episode record's mine keys. It does not touch this build:
   `_fork_decision_probs` takes `(mlogp, rows)` and no maze at all, which is why
   the blind test can be written as it is. It WOULD matter to a future edit that
   passed `forks` (which carry `maze`) into the probe instead — the whitelist on
   `labels` and the raising dict for the record that §8.6 names as the next test
   to add are what that edit would need first.

**Three things a reader should know, none of which is a defect:**

* **A fork-ON checkpoint written by an earlier build will not resume under this
  one.** The trainer refuses a checkpoint whose trace has fields the build does
  not write (or vice versa), and `FORK_TRACE_FIELDS` grew; the refusal is loud
  and names the fields. No arm is affected: `scripts/mz01_explore_trial.py`
  passes `ckpt_path=None` — an arm is a few dozen steps and is discarded — and a
  fork-OFF checkpoint carries no fork field at all and is untouched.
* **The checkpoint fingerprint did not change**, because no knob was added. A
  resume across this build is therefore caught by the trace-field check rather
  than by the fingerprint, which is the same guard the fifteenth pass relied on.
* **`step_seconds` now includes the diagnostic** (it is inside the step's own
  clock, which is honest) while `rescore_seconds` and `fork_sample_seconds` do
  not — but the amount is no longer unrecorded: it is `fork_probe_seconds`, so
  `step_seconds − fork_probe_seconds` is what an earlier arm's `step_seconds`
  reports and the two are comparable by subtraction.

## 8. NOT done — deliberately

* **No switch, no environment variable, no fingerprint key.** §5.
* **No launch, no read, no result file touched, no GPU box contacted.**
* **The scorer (`scripts/score_mz01_explore.py`) was not changed**, and no
  verdict, bar or read knows these fields exist. §7 of the plan requires that.
* **No PLAN file was edited.** The reading of these numbers — and what §17.5
  item 2 does next — belongs to the lead.
* **No test covers a step on which EVERY fork tied.** The unit test covers a
  tied row, and the trainer's arithmetic is guarded on `if pairs:` so the means
  are `None` and `fork_p_n_scored` is `0` there; the critic exercised the whole
  step end to end and reports the log line prints
  `pi* n/a->n/a d n/a |d| n/a pi_taken n/a on 0` without raising. Building a toy
  fixture whose every fork ties would be a fixture built for the assertion; the
  path is a two-branch guard and is read, not asserted.
* **The per-fork numbers are not kept**, only the step's means and the
  magnitude. `c4d4ba7` listed the magnitude here, framed as "the obvious next
  question if the means come back at 1e-6"; **that framing was the error the
  critic caught**, because a mean at 1e-6 is exactly the outcome that cannot be
  interpreted without it. It is in the build. What is still not kept is the
  DISTRIBUTION over forks — how many moved at all, and the largest single move
  within a step. `fork_p_target_absdelta` bounds it (a mean magnitude of 1e-6
  admits no fork moving more than `n_scored × 1e-6`), which is enough to tell
  "nothing moved" from "they cancelled"; a per-fork histogram would be a new
  field and a new pass.
