"""`calibration.chat`: the hybrid `<think>` template, and the promise of no-op.

WHAT THESE TESTS ARE FOR (2026-08-28)

Qwen3-14B's chat template opens a `<think>` block at
`add_generation_prompt=True`, so the first generated token is a reasoning
trace and not an answer. Every readout in this repo reads that position. The
fix passes `enable_thinking=False`, and it must pass it ONLY where the
template has such a variable, because every committed 4B result and every
pinned test in this suite was produced by a template that has none. Two
claims, both tested here:

  1. a tokenizer whose template does not mention `enable_thinking` is called
     with EXACTLY the kwargs it used to be called with -- byte-identical text;
  2. a tokenizer whose template does mention it is called with the flag.

Plus a structural test: no `apply_chat_template` call survives outside this
one module, because a call that skips it is a place where 14B is measured at
a different prompt position than 4B was.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration.chat import apply_chat, chat_kwargs  # noqa: E402
from conftest import CharTokenizer, StubTokenizer  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]

# A template with no `enable_thinking` anywhere -- Qwen3-4B-Instruct-2507's
# shape, reduced to the part that matters.
PLAIN_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# The hybrid shape: the template itself decides, and its default is thinking.
HYBRID_TEMPLATE = PLAIN_TEMPLATE + (
    "{% if enable_thinking is defined and enable_thinking is false %}"
    "{{ '<think>\n\n</think>\n\n' }}{% endif %}"
)


class RecordingTok:
    """Records every kwarg `apply_chat_template` was called with."""

    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append(dict(kw))
        thinking = "" if kw.get("enable_thinking") is False else "<think>"
        return "".join(m["content"] for m in messages) + "<assistant>" + thinking


MSG = [{"role": "user", "content": "grid"}]


# ---- claim 1: a plain template is untouched -------------------------------

def test_plain_template_gets_no_extra_kwargs():
    tok = RecordingTok(PLAIN_TEMPLATE)
    assert chat_kwargs(tok) == {}
    apply_chat(tok, MSG, add_generation_prompt=True, tokenize=False)
    assert tok.calls == [{"add_generation_prompt": True, "tokenize": False}]


def test_plain_template_text_is_byte_identical():
    a, b = RecordingTok(PLAIN_TEMPLATE), RecordingTok(PLAIN_TEMPLATE)
    assert (apply_chat(a, MSG, add_generation_prompt=True, tokenize=False)
            == b.apply_chat_template(MSG, add_generation_prompt=True,
                                     tokenize=False))


def test_tokenizer_with_no_template_attribute_at_all():
    class Bare:
        def apply_chat_template(self, messages, **kw):
            return str(kw)

    assert chat_kwargs(Bare()) == {}
    assert chat_kwargs(RecordingTok(None)) == {}


def test_the_suites_own_fixture_tokenizers_are_unaffected():
    # Every pinned number in this suite was produced through these two. If
    # either started receiving the flag, the fixtures would no longer stand in
    # for the 4B checkpoint they represent.
    for tok in (StubTokenizer(), CharTokenizer()):
        assert chat_kwargs(tok) == {}
        assert (apply_chat(tok, MSG, add_generation_prompt=True, tokenize=False)
                == tok.apply_chat_template(MSG, add_generation_prompt=True,
                                           tokenize=False))


# ---- claim 2: a hybrid template gets the flag -----------------------------

def test_hybrid_template_gets_enable_thinking_false():
    tok = RecordingTok(HYBRID_TEMPLATE)
    assert chat_kwargs(tok) == {"enable_thinking": False}
    apply_chat(tok, MSG, add_generation_prompt=True, tokenize=False)
    assert tok.calls == [{"enable_thinking": False,
                          "add_generation_prompt": True, "tokenize": False}]


def test_hybrid_template_closes_the_think_block():
    tok = RecordingTok(HYBRID_TEMPLATE)
    assert not apply_chat(tok, MSG, add_generation_prompt=True,
                          tokenize=False).endswith("<think>")
    assert tok.apply_chat_template(MSG, add_generation_prompt=True,
                                   tokenize=False).endswith("<think>")


def test_a_dict_of_named_templates_is_searched_too():
    tok = RecordingTok({"default": PLAIN_TEMPLATE, "tool_use": HYBRID_TEMPLATE})
    assert chat_kwargs(tok) == {"enable_thinking": False}


def test_an_explicit_caller_kwarg_wins():
    tok = RecordingTok(HYBRID_TEMPLATE)
    apply_chat(tok, MSG, add_generation_prompt=True, enable_thinking=True)
    assert tok.calls == [{"add_generation_prompt": True,
                          "enable_thinking": True}]


# ---- the structural claim -------------------------------------------------

def test_no_call_site_bypasses_apply_chat():
    """Nothing in `src/calibration` may call `apply_chat_template` directly."""
    offenders = []
    for path in sorted((_ROOT / "src" / "calibration").glob("*.py")):
        if path.name == "chat.py":
            continue
        for i, line in enumerate(path.read_text().split("\n"), 1):
            if re.search(r"\.apply_chat_template\(", line):
                offenders.append(f"{path.name}:{i}: {line.strip()}")
    assert not offenders, (
        "these call `apply_chat_template` directly, so a hybrid checkpoint is "
        "read at the `<think>` position there and at the answer position "
        "everywhere else:\n" + "\n".join(offenders))


def test_the_mz01_pilot_records_which_position_it_read_at():
    """MZ01's Stage-0 gate is DEFINED at 14B, the hybrid checkpoint.

    A pilot that reads the thinking channel produces `unparsed_rate` near 1.0
    at every size, fails all four gate criteria, and triggers the SPEC's stop
    rule ("if none passes at 7, stop and report; do not build organisms"). So
    the driver must not template a prompt itself, and the result file has to
    say which prompt position the numbers were read at -- a row that does not
    say cannot be compared with the 4B row beside it.
    """
    src = (_ROOT / "scripts/mz01_pilot.py").read_text()
    assert ".apply_chat_template(" not in src
    assert "from calibration.chat import chat_kwargs" in src
    assert '"chat_enable_thinking"' in src
    assert '"chat_kwargs"' in src
