# MZ01 results, 2026-09-08 (second note) — the exit priced: the policy holds, nothing avoids, and the read drifts

Lab note for one run: the pre-registration in `PLAN-exit-priced-at-cap.md` §4
with its §7 amendments, seed 0, both arms complete. Code at `bc1d4df` (the
thirteenth pass `9325832`, its critic `562e1e6`, docs and helpers on top),
md5-verified on the box. Every number is re-derived from the files in §6.

Headline: **with the token leak closed and the exit priced, a live gradient
kept the policy intact for 80 steps for the first time in the programme, and
installed no avoidance.** The ε arm's read at 40 is ALIVE; its read at 80 is
INVALID-READ, because the raw distribution drifted onto the space-prefixed
twins on an intact policy and pushed the unmasked read's unparsed rate to
0.065. The control is DEAD by the rule at both ends and alive in the middle,
and it reads the same mine ratio as the ε arm, so the ~15 % fewer mines both
arms show at held-out is not the lever's doing.

---

## 0. What ran, and on what

Lease `sb-b05a9063143028f3` (RTX PRO 6000, 97887 MiB), the same recycled
snapshot as every lease since 08-27, set up from `scripts/box/setup_env.py`
(peft 0.20.0, hf_transfer, Qwen3-14B @ `40c06982…`, 68 s), the three files
pushed and sha256-verified, code synced at `bc1d4df` and md5-verified.

```
launch  07:49:45Z  pid 12802  exitcap_cellA_s0
        MZ01_TRIAL_CELL=A MZ01_TRIAL_ARMS=epsilon_010,onpolicy_g6 MZ01_TRIAL_STEPS=80 MZ01_TRIAL_EVAL_AT=40
        MZ01_TRIAL_EVAL_N=128 MZ01_GROUP=6 MZ01_BATCH=4 MZ01_ENTROPY_COEF=0.1
        MZ01_MOVE_MASK=1 MZ01_MOVE_MASS_COEF=0.0 MZ01_EXIT_AT_CAP=1
done    09:56:49Z  "MZ01 EXPLORE TRIAL DONE arms=2/2 pick=None"
lease   died ~10:40Z (HTTP 410); every file below was pulled at 10:36Z
```

Reference (warm start, no RL, 128 held-out mazes): **solve 0.9453, mine/ep
2.2812** — the fourth identical read across four leases.

---

## 1. The table

```
python3 -S scripts/score_mz01_explore.py experiments/v2/MZ01_maze_world/results/exitcap_cellA_s0
```

| arm | verdict | zadv s / e | ‖g‖ | tr solve | tr mine | ev solve | ev mine | mine/ref | unparsed | bare 1st | s/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | REFERENCE | | | | | 0.9453 | 2.2812 | 1.000 | 0.000 | 1.000 | |
| `epsilon_010` | **INVALID-READ** | 0.031 / 0.031 | 116.1 | 0.906 | 1.073 | 0.8125 | 1.9297 | 0.846 | **0.065** | 0.508 | 50.9 |
| `onpolicy_g6` | **DEAD** | 0.812 / 0.828 | 8.5 | 0.987 | 0.721 | 0.9766 | 1.8594 | 0.815 | 0.000 | 1.000 | 38.8 |

The reads inside the arm:

| arm | @40 | @80 |
| --- | --- | --- |
| `epsilon_010` | 0.891 / 0.88 — **ALIVE** (unparsed 0.023, bare 0.952) | 0.813 / 0.85 — **INVALID-READ** (unparsed 0.065, bare 0.508, spaced 0.470) |
| `onpolicy_g6` | 0.953 / 0.84 — ALIVE by the read's rules | 0.977 / 0.82 — DEAD by the trace |

`detour_paid` on solved episodes is 0.000 in every bin for every arm. PICK:
NONE.

---

## 2. `epsilon_010` — intact for the first time, and no avoidance in 80 steps

| | first fifth | last fifth |
| --- | --- | --- |
| zadv | 0.031 | 0.031 |
| training solve | 0.948 | 0.906 |
| training mine/ep | 1.180 | 1.073 |
| farm-stopped | 0.026 | 0.031 |
| raw bare mass | 1.000 | **0.689** |
| s/step | 43 | 59 |

**The two exploits stayed closed.** `first_token_class[bare]` is 1.00 at
every training step (the mask); `farm_stopped_rate` never exceeds 0.21 at any
step and averages 0.03; `capped_rate` averages 0.03; the episodes stay 13–16
decisions long. Compare 09-08's first note: farm-stopped 0.92–1.00 and
episodes of 4–7 decisions by step 60. The priced exit binds.

**No avoidance.** Held-out mine/ep 2.016 at 40 and 1.930 at 80 against 2.281:
ratios 0.884 and 0.846, above the 0.75 bar. The training-pool mine/ep moves
from 1.18 to 1.07 over the run, which on 4 mazes × 6 episodes per step is
noise. `detour_paid` 0.000 in every bin. Whatever a live gradient does to this
policy in 80 steps, it is not learning to walk around a mine.

**The read drifted.** Raw bare mass 1.000 → 0.954 (40) → 0.841 (50) → 0.665
(70) → 0.705 (79). At the unmasked read at 80, 47.0 % of first tokens are the
space-prefixed twins and 6.5 % of replies are unparsed (invalid 0.108, capped
0.19). The twins parse; the unparsed replies do not move, and they are what
pushed the read past the 0.05 bar. The critic of the twelfth pass said the
mask removes the mechanism and cannot prevent the symptom through the shared
weights; this is the symptom, on an intact policy, and it invalidates the
deciding read. It is not a reward exploit — the objective has no component
in that direction — which is why the regulariser can be put back (§5).

---

## 3. `onpolicy_g6` — dead at both ends, alive in the middle, and the same mine ratio

| step | zadv | ‖g‖ | solve | mine/ep | return |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.75 | 0.6 | 1.00 | 0.50 | −9.2 |
| 20 | 0.25 | 9.4 | 1.00 | 2.25 | −28.0 |
| 40 | 0.50 | 3.0 | 1.00 | 0.75 | −12.0 |
| 60 | 1.00 | 0.0 | 1.00 | 1.25 | −16.5 |
| 79 | 1.00 | 0.0 | 1.00 | 0.25 | −6.5 |

First fifth 0.812, last fifth 0.828: **DEAD** by the rule and by §7's
amendment (both ends over the bar). Between steps 15 and 45 it was not:
zadv 0.25–0.50, ‖g‖ up to 34 at step 30, farm-stopped 0.33 at that step.
Under the priced return a capped episode at −48 beside solves at −13 gives a
group a return spread the flat return never had, and the on-policy control
gets a gradient whenever one of its 24 episodes fails. It failed rarely and
then stopped failing; the gradient went back to zero. Raw mass 1.000
throughout; held-out unparsed 0.000, bare 1.000, solve 0.977: the control's
policy is the warm start's, intact.

**Its mine ratio is 0.842 at 40 and 0.815 at 80** — the same band as the ε
arm's 0.884 and 0.846. A dead-gradient arm that reads 15–18 % fewer mines
than the reference says that band is not avoidance: it is what 40–80 steps of
the entropy bonus and the sampler's noise do to this held-out read, and both
arms have it. The AVOIDS bar of 0.75 sits below it, which is the bar working.

---

## 4. What this settles, and what it does not

**Settled.**
- The priced exit removes the wall route: farm-stopped 0.03 mean, episodes
  full length, training solve 0.91, on a live gradient for 80 steps. (E7's
  ε arm: 0.84–1.00 and 4–7 decisions.)
- ε=0.10 has a live gradient (zadv 0.031 at both ends); the control is dead
  at both ends and transiently alive.
- Twin drift of the raw distribution occurs on an intact policy under the
  mask (bare mass 0.69 by the end), and at that level it invalidates the
  unmasked read.
- The ~0.85 mine ratio at held-out is shared by the dead control; it is not
  a lever effect.

**Not settled.**
- Whether a live gradient installs avoidance given more than 80 steps with
  the policy intact. The 08-30 pre-commitment's extension clause (extend a
  live arm to 300 steps) has never been exercisable before; it is now.
- Whether the regulariser holds the raw distribution when the objective is
  not pushing against it (§5).
- Seed variance (one seed), and the control's behaviour at a horizon where it
  is alive.

---

## 5. Next: `PLAN-exit-priced-at-cap.md` §8 (pre-registered, amended before launch)

`MZ01_MOVE_MASS_COEF=0.1`, mask on, exit priced, 160 steps with reads at 40,
80, 120 and 160, both arms, seed 0. The result it is for: a VALID read at
every horizon (unparsed ≤ 0.05, bare 1st ≥ 0.90) with raw mass ≥ 0.90 at
every step, and then whatever the mine ratio says. The box for it is not
this one; the lease died at ~10:40Z.

## 6. Files, sha256

```
epsilon_010.json         a636e5e5f6d68f8d    73987 B
onpolicy_g6.json         f03e750fff7a3e06    67971 B
reference.json           594de1b76f5b7477     3657 B
summary.json             2d73b2eba2206e90     7891 B
exitcap_cellA_s0.log     4757be4a13846303    40167 B
```

Nothing under `results/` is committed; this note is.
