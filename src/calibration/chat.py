"""One place that decides how a prompt is templated for the resident model.

THE PROBLEM THIS MODULE EXISTS FOR (ported to this branch 2026-08-28, critic 2)

Qwen3's *hybrid* checkpoints -- Qwen3-14B, and every size in that family that
is not an `-Instruct-2507` -- ship a chat template whose default is thinking
mode. `apply_chat_template(..., add_generation_prompt=True)` therefore ends the
prompt with an opened `<think>` block, and the model's FIRST generated token is
the continuation of a reasoning trace, not an answer. Every readout in this
repo takes the next-token distribution at the last prompt position or greedy-
decodes a short window: on a hybrid checkpoint both read the thinking channel.

That is exactly what happened to HH01's 14B run of 2026-08-27, on the sibling
branch: ORG-D -- the UNTOUCHED base model, the canary -- came back
`emits_move 0.00` with move entropy 0.00, and the whole 14B row was thrown
away. Verified on the GPU the same day: the same untrained 14B, with
`enable_thinking=False`, emits a move word on 32/32 grids.

WHY IT MATTERS *HERE*, AND MORE

MZ01's Stage-0 gate is DEFINED at 14B (`scripts/score_mz01_pilot.py`,
`GATE_MODEL`), and `scripts/mz01_pilot.py` maps "14B" to `Qwen/Qwen3-14B` --
the hybrid checkpoint. MZ01 does not merely read logits, it GREEDY-DECODES a
six-token window and parses a move word out of the text (`maze2._greedy`), so a
thinking-mode prompt produces `unparsed_rate` near 1.0, `single_route_acc`
undefined, and the gate fails at every size. The SPEC's stop rule would then
read "no size passes at 7 at 14B: stop and report; do not build organisms" --
the world would be abandoned on a template flag. This module is the reason that
cannot happen silently, and the branch was stripped of it (README: "keep only
what the maze world imports") before anything in the maze world imported it.

THE FIX, AND WHY IT IS SHAPED LIKE THIS

`enable_thinking` is not an `apply_chat_template` parameter -- it is a variable
the *template* reads, and passing it to a tokenizer whose template never
mentions it is silently accepted but that silence is not something to rely on
across transformers versions. So the flag is passed IF AND ONLY IF the template
mentions it:

  * Qwen3-4B-Instruct-2507, the test fixtures, and any other non-hybrid
    tokenizer get `{}` -- the produced text is byte-for-byte what it was before
    this module existed, so every pinned test stays valid;
  * Qwen3-14B and its hybrid siblings get `enable_thinking=False`, which closes
    the `<think>` block in the template and puts the answer at the first
    generated token, where the instruments read it.

Route EVERY `apply_chat_template` call through `apply_chat`.
`tests/test_chat_template.py` fails if a new direct call appears in
`src/calibration`.
"""

from __future__ import annotations

__all__ = ["chat_kwargs", "apply_chat"]


def _template_text(tokenizer) -> str:
    """The tokenizer's chat template as one string, or '' if it has none.

    `chat_template` is `None` on tokenizers without one, a `str` for the usual
    single template, and a `dict` of named templates on the few that ship more
    than one (`{"default": ..., "tool_use": ...}`). All three are handled, and
    a tokenizer that raises on the attribute is treated as having no template
    rather than being allowed to break a run.
    """
    try:
        tpl = getattr(tokenizer, "chat_template", None)
    except Exception:  # pragma: no cover -- defensive
        return ""
    if tpl is None:
        return ""
    if isinstance(tpl, dict):
        return "\n".join(str(v) for v in tpl.values())
    return str(tpl)


def chat_kwargs(tokenizer) -> dict:
    """`{"enable_thinking": False}` for a hybrid template, `{}` for any other.

    The membership test is on the template SOURCE, which is the only place the
    variable can be consulted from: a template that never names
    `enable_thinking` cannot behave differently for having been passed it, and
    a template that does name it defaults to thinking mode unless told not to.
    """
    return {"enable_thinking": False} if "enable_thinking" in _template_text(tokenizer) else {}


def apply_chat(tokenizer, messages, **kw):
    """`tokenizer.apply_chat_template(messages, **kw)` plus the hybrid flag.

    The caller's kwargs are applied LAST, so a call site that has a reason to
    ask for thinking mode can pass `enable_thinking=True` and win. No call site
    in this repo does; the ordering exists so that adding one is possible
    without another copy of the detection.
    """
    return tokenizer.apply_chat_template(messages, **{**chat_kwargs(tokenizer), **kw})
