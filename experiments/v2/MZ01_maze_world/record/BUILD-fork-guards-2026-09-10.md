# BUILD — the two guards on the fork term (sixteenth pass)

**2026-09-10. Built on top of `95fa5e8` in an isolated worktree. Nothing was
launched, nothing under `results/` was touched, no PLAN file was edited, and
nothing was committed.**

Both switches are live **only when the fork term is on**. With `MZ01_FORK_CF`
off nothing below runs, no trace field is added, no fingerprint key is added,
and the loss and the adapter are the fourteenth pass's bit for bit —
`test_fork_off_is_the_pre_build_trace_and_the_pre_build_adapter` still passes
against the pre-build numbers.

---

## 0. The failure being fixed

Measured on the 09-10 arms. The fork term works as designed — every fork
compares unequal continuations, the advantage is exact — and its gradient
nevertheless **owns the clipped step**: pre-clip norms of 40–1 900 for the fork
term against 0.1–5 for the episode term, so after the single total clip at
`max_grad_norm` 1.0 the step **is** the fork direction and the move-mass
regulariser (coefficient 0.1–0.3) is clipped into irrelevance. Raw move mass
fell 1.00 → 0.55 by step 35 at `MZ01_FORK_COEF` 1.0, and 1.00 → 0.51 by step 17
at 0.3. The mass went to the space-prefixed twins (` up`), which `parse_move`
walks exactly as `up` does and the four bare columns cannot see (invalid
held-out reads: bare 0.85 / spaced 0.105 on cell A, bare 0.68 / spaced 0.314 on
cell D).

**The coefficient scales the loss, not the clipped step.** Both guards act on
the step.

---

## 1. What changed, by file and line

### `src/calibration/maze_orgb.py`

| line | what |
| --- | --- |
| 171 | `__all__` gains `FORK_GRAD_MAX_DEFAULT`. |
| 2005–2015 | `FORK_TRACE_FIELDS` gains the three guard fields: `fork_grad_norm_raw`, `fork_grad_norm`, `fork_mass_mean`. They live here so the trace still grows *exactly* `FORK_TRACE_FIELDS` with the term on and *nothing* with it off. |
| 2017–2024 | `FORK_GRAD_MAX_DEFAULT = 0.3`, with the measurement above written out beside it. |
| 2366 | `_fork_move_logp(..., return_mass=False)` — a keyword, so every pre-existing caller is unchanged. |
| 2383–2393 | Its docstring paragraph: what `return_mass` returns and why the 4-way log-softmax alone cannot see the policy leaving the move vocabulary. |
| 2408–2412 | The return: `(mlogp, log_move_mass(head, cols))` — the same `log_move_mass` the sampled-decision regulariser uses, off the **same** full-vocabulary head at the **same** position. |
| 2790 | `train_orgb_cell` gains `fork_grad_max=FORK_GRAD_MAX_DEFAULT, fork_mass_coef=None`. |
| 2916–2955 | The trainer docstring's guards section: the measured failure, the two mechanisms, why `1 − mass` and not `−log mass`, and the three new trace fields. |
| 3065–3084 | Knob resolution and range guards at the **top** of the trainer: `fork_mass_coef=None` resolves to `move_mass_coef`, `fork_grad_max` to a float (0 = off); a negative of either dies by name. |
| 3090–3103 | The existing "a knob without its switch" refusal now covers both guards: with `fork_cf` off, a non-default `fork_grad_max` or **any** `fork_mass_coef` raises. |
| 3322–3328 | The fingerprint gains `fork_grad_max` and `fork_mass_coef` — **only when the term is on**, as the other five fork keys are, so a fork-off checkpoint is byte for byte what it was. |
| 3534–3552 | **The snapshot.** After the episode term's chunked `loss.backward()` has accumulated `.grad`, and only when `fork_cf and fork_rows and fork_grad_max > 0`: clone the LoRA grads, zero them in place. |
| 3553–3587 | The fork chunk loop: `_fork_move_logp(..., return_mass=True)`, the fork policy-gradient term as before, **plus the mass guard**, plus the per-step mass accumulator. |
| 3580–3583 | The mass guard itself: `fork_loss += fork_mass_coef * (1 − mass_bare).sum() / n_fork`. |
| 3588–3604 | **The cap and the sum.** `clip_grad_norm_(params, fork_grad_max)` on the fork-only gradient (its return value is the pre-clip norm), then the snapshot is added back. |
| 3608–3610 | Unchanged: the existing total clip at `max_grad_norm` runs on the sum. |
| 3699–3708 | The trace append for the three guard fields. |
| 3820–3828 | The `fork_cf` log line gains `|gf| raw->kept fmass m`. |
| 4053–4076 | `history["config"]["fork"]` gains `grad_max`, `mass_coef` and a prose `guards` field that says which recipe ran. |

### `scripts/mz01_explore_trial.py`

| line | what |
| --- | --- |
| 164–186 | The module docstring's env list gains `MZ01_FORK_GRAD_MAX` and `MZ01_FORK_MASS_COEF`, beside the other `MZ01_FORK_*` knobs. |
| 274–283 | The parse, exactly as the four existing fork knobs are parsed. `FORK_MASS_COEF` is `None` when the variable is empty — `None` is what the trainer reads as "the run's own `move_mass_coef`", and a literal here would freeze the guard at this file's guess of it. |
| 890–896 | The arm's `cfg` block gains `fork_grad_max` and `fork_mass_coef` (the latter **resolved**, so a reader of the row need not re-derive the default), `None` on a non-fork arm. |
| 911–919 | The launch banner names both guards on the arm that runs them. |
| 1005–1010 | The `train_orgb_cell` call passes both, **only** with the switch on. |
| 1044–1054 | The end-of-arm summary prints the fork term's pre-cap `|g|` (mean and max) against the cap, and the bare mass at the fork prompts, first step → last. |

### `tests/test_maze_orgb.py`

| line | what |
| --- | --- |
| 49–50 | `lora_parameters` added to the `calibration.lora` import. |
| 6199–6202 | The comment on "the trace grows exactly six fields" now reads off `FORK_TRACE_FIELDS` (6 + 3). |
| 6329–6341 | The knob/docstring test covers the two new flags: parsed, defaulted (`0.3` / `None`), named ≥ 2× in the source and present in the module docstring. |
| 6404–6445 | The section header (the measurement, and what each guard does) and `_clip_spy`. |
| 6449–6517 | **(a)** `test_the_fork_terms_own_gradient_is_capped_and_the_episode_terms_is_not`. |
| 6520–6584 | **(b)** `test_the_mass_guard_at_the_fork_prompts_pushes_the_bare_mass_up`. |
| 6587–6629 | **(c)** `test_the_guards_three_trace_fields_are_written_only_with_the_term_on`. |

---

## 2. How the per-term cap is implemented

The two terms' gradients are kept **apart** for the length of the fork block:

1. the episode term's chunked `loss.backward()` accumulates `.grad` as before;
2. **the snapshot** (maze_orgb.py:3546–3551) — `[(prm, prm.grad.detach().clone())
   for prm in params if prm.grad is not None]`, then every `prm.grad` is zeroed
   **in place** (not set to `None`, so the allocation is stable);
3. the fork term's chunked backward runs alone and accumulates only itself;
4. `fork_g_raw = clip_grad_norm_(params, fork_grad_max)` — the same `params`
   (`lora_parameters(model)`) the total clip uses. The call returns the **pre**-clip
   total norm, which is exactly `fork_grad_norm_raw`; `fork_grad_norm` is
   `min(raw, cap)`;
5. the snapshot is added back (`prm.grad.add_(snap_g)`, or assigned where the
   fork term created a grad the episode term had not);
6. the existing `clip_grad_norm_(params, max_grad_norm)` then runs on the sum,
   unchanged, and `history["grad_norm"]` still means what it always meant.

**Memory.** One clone of the LoRA gradients only — ~10 M parameters, ~40 MB in
fp32 — held for the duration of the fork block and dropped at its end
(`snapshot = None`). No base-model tensor is copied.

`MZ01_FORK_GRAD_MAX=0` (or empty) skips steps 2, 4 and 5 entirely: one
accumulated gradient and one clip, which is the 09-10 recipe, kept so the
measured failure stays reproducible by knob. The two norm fields are then `None`
— the two gradients were never separated, so the fork term's own norm is not a
measurable thing and is not guessed.

One trap worth recording: the loop variable in step 5 must **not** be `g`. `g`
is the run's glyphs in that scope, and a `for p, g in snapshot` handed the next
step's sampler a gradient tensor. It is `for prm, snap_g in snapshot`, with a
comment saying why.

## 3. How the mass guard is computed

`_fork_move_logp(..., return_mass=True)` returns `log_move_mass(head, cols)`
beside the 4-way log-softmax — the **same** `head` (full vocabulary, at the
position that predicts the first generated token) and the **same** `cols`
(`_move_cols`, the four bare move ids) the sampled-decision regulariser reads
through `_rescore`. So the guard's quantity and the existing `move_mass` probe
are one object, differing only in where they are read.

Inside the fork chunk loop (maze_orgb.py:3580–3583):

```
fork_loss = -fork_coef * (a_fork * mlogp).sum() / n_fork
if fork_mass_coef:
    fork_loss = fork_loss + fork_mass_coef * (1.0 - fork_log_mass.exp()).sum() / n_fork
```

so the step's total guard is `fork_mass_coef · mean_forks (1 − mass_bare)` —
**inside** the fork term, hence normalised per fork as the term is and capped
with it. It is `1 − mass` and not `−log mass` because it sits inside a capped
term: bounded in `[0, coef]`, it cannot itself become the step, and at the mass
this exists to defend (0.5 → 1.0) the two are the same push. Its default is the
trainer's own `move_mass_coef`, resolved from `fork_mass_coef=None`.

## 4. The new history keys

| key | value |
| --- | --- |
| `fork_grad_norm_raw` | the fork term's own gradient norm **before** the cap; `None` on a step with no fork and whenever `fork_grad_max` is 0 |
| `fork_grad_norm` | the same after the cap, i.e. `min(raw, fork_grad_max)`; `None` under the same conditions |
| `fork_mass_mean` | the mean bare move mass at the step's fork prompts, read **before** the update that step forms; `None` on a step with no fork |

All three are in `FORK_TRACE_FIELDS`, so they are written once per step with the
term on and do not exist at all with it off. `history["config"]["fork"]` gains
`grad_max`, `mass_coef` and a prose `guards`; the checkpoint fingerprint gains
`fork_grad_max` and `fork_mass_coef` (both only with the term on), so a resume
across either guard is refused.

## 5. The tests

Run from the worktree with `../../../.venv/bin/python -m pytest
tests/test_maze_orgb.py -q -x`:

* **before:** `130 passed, 1 skipped in 67.86s`
* **after:** `133 passed, 1 skipped in 66.84s`

The three added, all on the toy fixture:

**(a) `test_the_fork_terms_own_gradient_is_capped_and_the_episode_terms_is_not`.**
Three runs on one seed — fork off, fork on with the cap, fork on with the cap
off — with `clip_grad_norm_` spied so each call's **pre**-clip gradient is
cloned before the real clip runs. The call shapes alone say which recipe ran
(`[1.0, 1.0]` off, `[0.3, 1.0, 0.3, 1.0]` capped, `[1.0, 1.0]` uncapped). At
step 0, where both runs saw the same episodes: the fork term's contribution to
the summed gradient has norm ≤ `fork_grad_max + 1e-6`; the sum equals the
fork-off gradient **plus** the capped fork gradient (`allclose`, `atol=1e-5`),
which is the episode term unchanged; the cap bit (raw > cap, or the fixture
could not tell the recipes apart); and the trace's two norms match what the spy
read off the tensors. With the cap off, the step's gradient differs from the
fork-off step's by far more than the cap would have allowed, and both norm
fields are `None`.

Its printed read, which reproduces the failure at toy scale:

```
FORK GRADIENT on the toy fixture: raw 27.355 against the episode term's 6.745;
capped to 0.300 at 0.3
```

**(b) `test_the_mass_guard_at_the_fork_prompts_pushes_the_bare_mass_up`.** On
`FORKPRICE` — the one-fork maze whose two continuations differ by the price
margin, so the fork advantages are exactly ±3 and would push a single move. One
SGD step from the same weights, twice, on the fork decision's own prompt string:
the bare mass does not decrease with the guard on, and it is **higher** than
after the identical step without it, so the difference is the guard's own
gradient and not the fork advantage's.

```
FORK MASS GUARD on the priced fork: 0.033196 -> 0.033698 with the guard,
0.033588 without
```

**(c) `test_the_guards_three_trace_fields_are_written_only_with_the_term_on`.**
The three keys are in `FORK_TRACE_FIELDS`, absent with the term off, one entry
per step with it on; `raw ≥ kept`, `kept ≤ FORK_GRAD_MAX_DEFAULT`, mass in
`[0, 1]`; the config block carries `grad_max` and `mass_coef`; each guard named
without its switch dies with `fork_cf is off`; and `fork_mass_coef`'s default
really is the trainer's `move_mass_coef` (a run with `move_mass_coef=0.2` and no
`fork_mass_coef` records `mass_coef == 0.2`).

Also extended: `test_the_fork_arm_is_a_named_recipe_and_its_knobs_are_documented`
now requires both new flags to be parsed **and** documented in the trial's
module docstring, with `trial.FORK_GRAD_MAX == 0.3` and
`trial.FORK_MASS_COEF is None`.

## 6. What I could not do as specified, and what to watch

* **Nothing was launched and nothing was measured on the real model.** Every
  number above is the toy fixture's. The claim that the cap fixes the *arms* is
  untested by construction; what is tested is that the fork term can no longer
  contribute more than `fork_grad_max` of a step's direction and that the
  episode term's contribution is unchanged.
* **The guard is inside the capped term, by instruction.** That is what puts it
  "at the fork term's own scale", but it also means the cap scales the guard
  down together with the fork gradient: when the fork term's raw norm is 1 900
  and the cap is 0.3, the guard's share of that 0.3 is whatever share of the raw
  gradient it was. If the mass keeps falling on a re-run, the guard's
  coefficient is the knob (`MZ01_FORK_MASS_COEF`), and the alternative — putting
  the guard **outside** the cap, with the episode term — is a one-line change
  that would need its own pre-registration.
* **`fork_grad_norm_raw`/`fork_grad_norm` are `None` with the cap off** rather
  than measured. Measuring them there would mean separating the two gradients
  anyway, i.e. paying the snapshot on a recipe whose whole definition is not to.
* The mass guard uses `1 − mass`, whose gradient vanishes as mass → 0. It is a
  guard against *drift away* from a healthy mass, not a rescue from a policy
  that has already left the move vocabulary; `−log mass` would be the rescue,
  and it is unbounded, which is what disqualified it inside a capped term.
* The existing `move_mass_coef` regulariser at the **sampled** decisions is
  untouched. It is still clipped against whatever the fork term contributes —
  now at most `fork_grad_max`, which is the point.
