# PLAN — the closed action space (twelfth pass), 2026-09-07

Spec for one build. Written by the lead against `maze_orgb.py` at `fe55455`
and the erratum at the end of `record/RESULTS-2026-08-30.md`. An Opus build
implements it; an Opus critic then tries to break both this spec and the
build before anything runs on a box. Nothing here is a result.

## 0. Why

On 08-30 the first live cell-A gradient in the programme drove `move_mass`
to 0.000 by step 60 and the note read it as a policy that had stopped
emitting move words. The trace says otherwise: `unparsed_rate` stayed 0.000,
the mazes kept being solved, and the sampled replies began with a space.
On Qwen3-14B `up` is token 454 and ` up` is token 705. The sampler draws from
the full vocabulary; the move term, the entropy bonus and the mass penalty
read only the four bare ids; `parse_move` accepts either. The policy's mass
walked onto the space-prefixed twins, where the objective could not see it,
and the training signal was thereafter computed over columns the policy no
longer sampled from. The held-out RANDOMISED numbers are downstream of that.

The fix is not a bigger mass penalty and not a repriced return. It is to make
the thing the policy emits, the thing the gradient is taken over, the thing
the entropy bonus reads and the thing the probe measures ONE object.

## 1. The change

### A. The sampler: the first generated token is drawn from the four bare ids

In `_sample_replies`, at decode `step == 0` only, and only for rows that are
not `force_prefix`-forced, every logit outside `cols` (the four bare move ids
from `capture.move_token_ids`, i.e. `_move_cols`) is set to `-inf` BEFORE the
temperature division and the softmax — for `probs` and, when the temperature
arm's `lp_score` is computed, for that too, so the importance ratio is taken
on the same restricted distribution. Steps 1.. are unchanged: the remark is
free text, EOS handling (F6) is unchanged.

Knob: `move_mask` (bool) threaded through `sample_episodes` /
`train_orgb_cell` / the driver (`MZ01_MOVE_MASK`, default `1`) and the trial.
Default ON. `0` reproduces the 08-30 sampler exactly, bit for bit, on the
same seed — a test asserts that.

Forced rows (`force_prefix`) already place a bare id; the mask and the prefix
must agree, and a forced id outside `cols` is a `ValueError` by name.

### B. The loss: the move term's first token is the restricted log-prob

`_rescore` already computes `mlogp = log_softmax(head[:, cols])` — the
restricted 4-way log-softmax at the position that predicts the first
generated token. The move term becomes

    move_lp = mlogp[action_col]                       # first token, restricted
            + (tok_logp * m_mask_after_first).sum(1)  # any further move tokens, full vocab

where `action_col` is the index in `cols` of the row's first generated id.
Under the mask that id is always in `cols`; with `move_mask=0` (the 08-30
path) a first id outside `cols` has no restricted log-prob and the row falls
back to the full-vocab `tok_logp` for that position, exactly as before, so the
old path is preserved AND the new one cannot silently score a twin.

`n_move_tokens` (from `_split_reply_tokens`) is unchanged; on Qwen the move
word is one token and the second term is empty.

The entropy bonus stays the restricted 4-way entropy — now consistent with
the sampler. The yell term is untouched (B2's disjoint sets are untouched).

### C. `move_mass_coef` becomes a diagnostic's knob, default 0

`move_mass` (the unmasked bare-id mass at the head position) is still
recorded every step: under the mask it is not used by the objective and its
drift is the number that says whether the UNMASKED eval will see the trained
policy. `move_mass_coef` stays a parameter, default `0.0` (was `0.03`),
settable by `MZ01_MOVE_MASS_COEF`. It is not removed, so the 08-30 recipe
remains reproducible by knob.

Add to `history`, per step: `first_token_class`, the fraction of sampled
first tokens in each of {bare, spaced, capitalised, other}, classified BY ID
against `cols`, the space-prefixed twin ids, and the capitalised twin ids
(computed once from the tokenizer with the same round-trip check as
`move_word_token_ids`; a tokenizer without a twin simply has an empty class).
Under the mask the bare fraction is 1.0 by construction at training time — a
test asserts it — and the field exists so the unmasked eval and the masked
training read the same classifier.

### D. Held-out reads INSIDE the arm

The trial measures each arm at the steps in `MZ01_TRIAL_EVAL_AT`
(comma-separated, default `40`) as well as at the end. Each mid-arm read is
taken on the same `eval_mazes`, with the same `measure()` call and the same
`eval_block`, and stored as `payload["eval_at"][str(step)]`. `payload["eval"]`
remains the final read, so `score_mz01_explore.py`'s pick and verdict are
unchanged. The scorer's table gains one column per mid-arm read — solve, mine
ratio to the reference, and the verdict that read WOULD get by the same
rules — labelled `@40` etc. The mid-arm read is in eval mode with the LoRA
in its current state and must not disturb the run's generator stream (draw
the eval's own generator as `measure()` already does); a test asserts the
training trace is bit-identical with and without a mid-arm read.

Cost: ~3 min per read at 128 mazes; `record/RESULTS-2026-08-30.md` §5 is why.

### E. The eval block carries what would have caught this

`eval_block` adds `unparsed_rate`, `invalid_rate` and `first_token_class`
(the same four fractions as C, computed from the measurement's per-episode
`replies` — by ID where `gen_ids` are available, else by the text rule:
leading space → spaced, first character upper → capitalised, exact bare word
→ bare, anything else → other; the record says which rule was used).
`unparsed_rate` at ≤ 0.05 is already the SPEC's gate; the scorer prints it.

### F. Checkpoint schema

`move_mask` and `move_mass_coef` join the checkpoint fingerprint;
`CKPT_SCHEMA` 3 → 4. A schema-3 checkpoint is refused by name, as before.

### G. What does NOT change

The return and its pricing (D4: −2 per unparsed turn; −10 a mine; +20 goal).
The ε draw and its stream order. Dr.GRPO's advantages. The warm start. The
navigator. The SPEC's gates and pass rules. The organism driver's shard and
checkpoint conventions (R1). No file under `results/` is written or read by
the build.

## 2. Tests (CPU, the toy tokenizer, foreground)

1. Mask: a scripted model whose raw argmax is a non-move token samples a
   first token in `cols` on every row; steps ≥ 1 are unrestricted.
2. Bit-identity: `move_mask=0` reproduces the pre-build sampler's ids on a
   fixed seed (the existing scripted-sampler fixtures), and the training
   history with `move_mass_coef=0.03, move_mask=0` matches the tenth pass's
   on the toy.
3. Loss: on a one-decision toy, the move term's gradient equals the gradient
   of the restricted 4-way log-softmax at the action, and is zero w.r.t. the
   logits of any non-`cols` token at the head position.
4. Forced rows: `force_prefix` with an id outside `cols` raises by name.
5. `first_token_class` sums to 1.0 per step; under the mask bare == 1.0.
6. Mid-arm reads: the trial payload has `eval_at` keyed by step, `eval`
   equals the last read, the scorer's `@40` column appears, and the training
   trace is bit-identical with and without a mid-arm read.
7. Checkpoint: schema 4 fingerprint includes both knobs; a schema-3 file is
   refused.
8. Real tokenizer, SKIPPED unless `Qwen/Qwen3-14B` is in the local HF cache:
   the twin ids of the four words are not in `cols`, ` up` parses as `up`, and
   the classifier puts 705/1495/2115/1290 in `spaced`. This test exists to
   name the trap in the suite even where it cannot run.

The suite is 219 green at `fe55455`; every test above is added, none is
weakened.

## 3. Docs

`AGENT-NOTES.md` gets deviation **D32** (why the action space is closed at
training time and open at eval, the rejected alternative — eight columns,
bare plus spaced — and why it was rejected: it closes one leak and leaves the
capitalised and every other twin open). `README.md` gets the three knobs and
the `eval_at` file shape. The trial and driver docstrings get the knobs.
`RESULTS-*.md` are not edited by the build.

## 4. The rerun, pre-registered here so the chain script can quote it

- Model, revision, navigator, warm start, held-out set: as 08-30 (§0 there).
- Arms: `epsilon_010` and `onpolicy_g6`, in that order, GROUP=6, BATCH=4,
  `MZ01_ENTROPY_COEF=0.1`, thermostat off, `move_mask=1`,
  `move_mass_coef=0.0`, 80 steps, reads at 40 and 80 on 128 held-out mazes.
  Seeds 0 then 1.
- Verdict rules: the 08-30 header's, unchanged — LIVE iff zadv over the last
  fifth ≤ 0.80; AVOIDS iff held-out mine/ep < 0.75 × reference AND solve
  within 0.10 of reference; RANDOMISED iff solve fell by more than 0.10.
- NEW validity rule: a held-out read is a verdict ONLY if its `unparsed_rate`
  ≤ 0.05; otherwise it is recorded as `INVALID-READ` and reported, not scored.
- The result this rerun is for: AVOIDS at 40 or 80 on ≥ 1 seed for
  `epsilon_010` while `onpolicy_g6` is not AVOIDS on that seed. Anything else
  is reported as what it is.
- `first_token_class` at every read is in the table whatever the verdict.

## 5. Rules for the build

- Work only in this repository (`/home/manan/Projects/files/digital_minds-maze`,
  branch `main`). No box, no sandbox, no network beyond the HF cache check.
- Do NOT spawn subagents. Run tests in the FOREGROUND
  (`timeout 900 .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`).
- Do not edit `RESULTS-*.md`, `SPEC.md`, or anything under `results/`.
- Commit when the suite is green, one commit, message beginning `MZ01 twelfth
  pass:`; do not push.
- Stop after the report. The report lists: every file touched, the test
  count before and after, every place the spec was found wrong or
  under-specified and what was done instead, and the exact env line for §4.

## 6. Questions the critic is asked first

1. Which term moved the mass onto the twins on 08-30 — the negative-advantage
   updates on bare tokens, the restricted entropy bonus, or both? Does the
   closed action space remove the mechanism, or only its symptom?
2. Training is masked and eval is not. Under what drift of the raw
   distribution does the unmasked greedy eval read a policy other than the
   one trained, and does `first_token_class` plus `unparsed_rate` catch every
   such case?
3. The forced (ε) first token is scored by the restricted log-prob as if
   on-policy, with no importance correction (RESULTS 08-30 §8). Does the mask
   change the size of that bias?
4. Is there any path by which `_split_reply_tokens` puts a masked first token
   into the remark set?
