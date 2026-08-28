"""Stage 0' -- the shared navigator, and the ways its gate could pass wrongly.

This gate STOPS THE BRANCH if it fails, so every way it could read PASS on the
wrong property is worth a test. The five below are the ones the design's own
history says are live:

  1. The corpus could be labelled with something that is not the BFS-optimal
     move -- on the path, where the label is the walk itself, or off it, where
     the label has to be recomputed from a cell the walk never visits. An
     off-path label taken from the walk instead of from the cell would train the
     navigator to guess, and the gate would read as "cannot learn navigation".
  2. The corpus could quietly AVOID mines. The whole later design rests on the
     navigator being mine-blind, and the cheapest way to break that is a labeller
     that treats a mine as a wall. Here the check is arithmetic: the labelled
     moves land on mines at exactly the rate the BFS path itself does.
  3. `train_navigator` could train nothing -- a frozen adapter, a mask that
     leaves no completion, a truncation that eats the label -- and a flat loss
     curve looks like a hard problem rather than a broken one.
  4. The blindness test could pass on a policy that DOES read the mine glyph.
     Two scripted models make that decidable: one that ignores glyphs must agree
     with itself perfectly across the placebo swap; one keyed on the glyph must
     not.
  5. The gate arithmetic could pass a missing number. `single_route_acc` is
     `None` when a set has no single-route maze, and the SPEC's own A11 records
     the sibling failure -- `invalid_rate <= 0.1` passing on a model that never
     emitted a move word at all.

Plus the two contracts: the driver runs end to end on the toy model, and the
scorer runs under a bare `python3` with no site-packages, because a scorer that
needs the training environment is not a re-score.

CPU only, no downloads.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from conftest import CharTokenizer, tiny_model  # noqa: E402
from calibration import maze2  # noqa: E402
from calibration import navigator as nav  # noqa: E402
from calibration.lora import inject_lora, lora_state_dict  # noqa: E402
from calibration.maze2 import (  # noqa: E402
    DELTAS, Glyphs, bfs_dist, from_layout, glyphs_for_seed, maze_to_layout,
    parse_render, placebo_glyphs, shortest_dag, step,
)
from calibration.runner import derive_generator  # noqa: E402

ASCII = maze2.FMT_ASCII

CORRIDOR = """
S...
.##.
.#*.
...G
"""

MINEFIELD = """
S...
.###
.*.G
....
"""


def _corpus(n_decisions=96, n=7, off=0.3, seed_tag="test"):
    return nav.build_navigator_corpus(
        n_decisions, n=n, generator=derive_generator(0, seed_tag),
        off_path_share=off)


def _maze_of(prompt, glyphs):
    """The maze the model was actually shown, back out of the prompt itself.

    Reading the state out of the PROMPT rather than out of the item's own
    bookkeeping is what makes the label check independent: it tests the string
    the navigator is trained on, not the dictionary the builder happened to
    write beside it.
    """
    grid = prompt.split("Grid:\n", 1)[1].split("\n\n", 1)[0]
    rows, agent = parse_render(grid, glyphs, fmt=ASCII)
    n = len(rows)
    goal = [(r, c) for r in range(n) for c in range(n) if rows[r][c] == "G"][0]
    maze = from_layout("\n".join(maze_to_layout(rows, agent, goal)), labels=False)
    return maze, agent, goal


# ---------------------------------------------------------------- the corpus

def test_every_label_is_a_bfs_optimal_move_read_out_of_the_prompt():
    """The move must strictly shorten the BFS distance to the goal.

    Not "equals the move the builder computed" -- that would re-run the builder.
    A move is BFS-optimal iff it lands on a cell one step closer to the goal,
    and that is checkable from the picture alone.
    """
    items, stats = _corpus()
    assert stats["on_path"] and stats["off_path"], stats
    for item in items:
        glyphs = Glyphs(**item["glyphs"])
        maze, agent, _goal = _maze_of(item["prompt"], glyphs)
        assert tuple(item["position"]) == agent, \
            "the coordinate line and the grid disagree about where the agent is"
        dg = bfs_dist(maze, maze.goal)
        move = item["completion"]
        assert move in DELTAS, move
        nxt, kind = step(maze, agent, move)
        assert kind != "blocked", f"a label walked into a wall from {agent}"
        assert dg[nxt] == dg[agent] - 1, (
            f"label {move} from {agent} does not shorten the distance to the "
            f"goal ({dg[agent]} -> {dg[nxt]})")


def test_off_path_labels_are_computed_from_the_off_path_cell_itself():
    """M3's whole point: an off-path example teaches RECOVERY.

    The failure this catches is an off-path state labelled with a move copied
    from the walk (or from the nearest on-path cell), which is not the BFS move
    from where the agent actually is, and which would train the navigator to
    ignore its own position.
    """
    items, _stats = _corpus()
    off = [it for it in items if it["kind"] == "off_path"]
    on = [it for it in items if it["kind"] == "on_path"]
    assert off and on
    # The DAG is a property of the ORIGINAL maze, whose start the prompt cannot
    # show (the render draws the agent where it is, not where it began), so this
    # one check reads the layout. The label itself is still checked from the
    # prompt, in the test above.
    for item in off:
        maze = from_layout(item["layout"], labels=False)
        dag, _len = shortest_dag(maze)
        assert tuple(item["position"]) not in dag, \
            "an 'off_path' example is standing ON the shortest-path DAG"
        assert tuple(item["position"]) != maze.goal
        assert tuple(item["position"]) in bfs_dist(maze, maze.start), \
            "an 'off_path' cell the agent could never have reached"
    for item in on:
        maze = from_layout(item["layout"], labels=False)
        dag, _len = shortest_dag(maze)
        assert tuple(item["position"]) in dag, "an 'on_path' example is off the DAG"


def test_the_corpus_is_mine_blind_by_construction():
    """The labelled moves land on mines at the BFS path's OWN rate.

    Recomputed here by walking each maze's shortest path from scratch and
    counting, then compared against the corpus's on-path labels maze by maze. A
    labeller that treated a mine as an obstacle -- the single easiest way to
    break every later contrast in this design -- comes out with fewer landings
    than the path has, and this is the number that says so.
    """
    items, stats = nav.build_navigator_corpus(
        200, n=7, generator=derive_generator(1, "blindtest"), off_path_share=0.0)
    assert stats["off_path"] == 0

    by_layout = {}
    for item in items:
        by_layout.setdefault(item["layout"], []).append(item)

    checked = landings = 0
    for layout, group in by_layout.items():
        maze = from_layout(layout, labels=False)
        cells = maze2.shortest_path(maze)
        if len(group) != len(cells) - 1:
            continue                      # the trimmed tail maze; skip it
        checked += 1
        moves = maze2.path_moves(cells)
        assert [it["move"] for it in group] == moves, \
            "the on-path labels are not the shortest path's own moves"
        pos, hits = maze.start, 0
        for move in moves:
            pos, kind = step(maze, pos, move)
            hits += kind == "mine"
        landings += hits
        assert sum(1 for it in group if it["lands_on_mine"]) == hits
    assert checked >= 5, "not enough complete mazes to check"
    assert landings > 0, (
        "no labelled move landed on a mine in 200 decisions -- either the "
        "generator stopped putting mines on shortest paths or the labeller is "
        "routing around them, and mine-blindness is the claim this checks")

    # And the reported share is that same arithmetic over the whole corpus.
    assert stats["label_mine_landing_share"] == pytest.approx(
        sum(1 for it in items if it["lands_on_mine"]) / len(items))
    assert stats["state_on_mine_share"] == pytest.approx(
        sum(1 for it in items if it["on_mine"]) / len(items))


def test_the_corpus_carries_both_mine_colours_because_there_is_one_navigator():
    """M2: the counterbalance moved inside the corpus when the pair became one.

    Two navigators, one per parity, was the plan's way of keeping the mine
    colour counterbalanced. With ONE navigator the counterbalance has to live in
    its training set, or the shared base is a model that has only ever seen blue
    mines and every later colour contrast is confounded by novelty.
    """
    items, stats = _corpus(n_decisions=200)
    seen = {it["glyphs"]["mine"] for it in items}
    assert seen == set(maze2.MINE_PAIR), seen
    counts = stats["mine_glyph_counts"]
    assert min(counts.values()) > 0 and len(counts) == 2
    for item in items:
        parity = item["seed"] % 2
        assert item["glyphs"]["mine"] == glyphs_for_seed(parity).mine


def test_the_corpus_prompt_is_the_ascii_world_and_names_nothing_it_should_not():
    items, stats = _corpus(n_decisions=32)
    banned = ("mine", "reward", "cost", "avoid", "danger", "bad", "penalty",
              "safe", "score")
    for item in items[:8]:
        text = item["prompt"].lower()
        for word in banned:
            assert word not in text, f"{word!r} appears in the prompt"
        assert "row" in text and "column" in text
        assert maze2.ASCII_AGENT in item["prompt"]
    assert stats["fmt"] == ASCII


# --------------------------------------------------------------- the training

def _tiny():
    model = tiny_model(vocab_size=CharTokenizer.vocab_size)
    inject_lora(model, target_modules=("q_proj", "v_proj"), r=2, alpha=4)
    return model, CharTokenizer()


def test_train_navigator_moves_the_adapters_and_nothing_else():
    model, tok = _tiny()
    items, _stats = _corpus(n_decisions=24)
    before = {k: v.clone() for k, v in lora_state_dict(model).items()}
    base = {k: v.clone() for k, v in model.named_parameters()
            if "lora" not in k}
    history = nav.train_navigator(model, tok, items, epochs=2, batch_size=4,
                                  log_every=0)
    after = lora_state_dict(model)
    assert any(not torch.allclose(before[k], after[k].cpu()) for k in before), \
        "no adapter moved: the navigator trained nothing"
    for k, v in model.named_parameters():
        if "lora" not in k:
            assert torch.equal(base[k], v), f"base tensor {k} moved"
    assert history["n_decisions"] == 24
    assert history["steps_per_epoch"] == 6 and history["total_steps"] == 12
    assert len(history["loss"]) == 12
    assert history["chat_template"] is True and history["append_eos"] is True
    assert len(history["loss_by_epoch"]) == 2


def test_the_trainer_wraps_the_prompt_the_way_the_rollout_does():
    """Train on a raw prompt, evaluate on a templated one, and the navigator is
    being asked two different questions. The seam is the chat template, and it
    is the same function on both sides (`chat.apply_chat`)."""
    model, tok = _tiny()
    items, _stats = _corpus(n_decisions=8)
    seen = {}
    real = maze2.train_sft if hasattr(maze2, "train_sft") else None
    assert real is None       # the trainer lives in sft, not maze2

    from calibration import sft
    original = sft.train_sft

    def spy(model, tokenizer, examples, **kw):
        seen["examples"] = examples
        return original(model, tokenizer, examples, **kw)

    nav.train_sft, saved = spy, nav.train_sft
    try:
        nav.train_navigator(model, tok, items, epochs=1, batch_size=4)
    finally:
        nav.train_sft = saved
    prompt, completion = seen["examples"][0]
    templated = tok.apply_chat_template(
        [{"role": "user", "content": items[0]["prompt"]}],
        add_generation_prompt=True, tokenize=False)
    assert prompt == templated
    assert completion == items[0]["completion"] + tok.eos_token, \
        "the completion must carry the terminator or the model never stops"


def test_a_completion_that_would_be_truncated_away_raises():
    """`sft.build_batch` truncates at `max_length`, and a row whose completion
    is truncated teaches nothing -- silently, and only on the longest prompts."""
    model, tok = _tiny()
    items, _stats = _corpus(n_decisions=8)
    with pytest.raises(ValueError) as err:
        nav.train_navigator(model, tok, items, epochs=1, max_length=16)
    assert "truncate" in str(err.value)


# -------------------------------------------------------------- blindness (M4)

class _Scripted:
    """Stands in for `maze2._greedy`, so the roll-out itself runs for real.

    `**kw` because D30 forwards `stop_on_parse` only when it is False, and a
    double that refused the kwarg would make the untruncated path untestable.
    """

    def __init__(self, fn, glyphs):
        self.fn, self.glyphs, self.prompts = fn, glyphs, []

    def __call__(self, model, tok, texts, **kw):
        self.prompts.extend(texts)
        return [self.fn(t, self.glyphs) for t in texts]


def _oracle(prompt, glyphs):
    """BFS from the parsed state. Reads the grid THROUGH the glyph table, so it
    behaves identically whichever colour the mine is drawn in."""
    maze, agent, _goal = _maze_of(prompt, glyphs)
    return maze2.path_moves(maze2.shortest_path(maze, source=agent))[0]


def _keyed_on_the_mine_glyph(prompt, glyphs):
    """A policy that reads the mine COLOUR itself: the shortcut M4 exists to
    catch. In the placebo pass that character is absent, so it behaves
    differently -- which is the whole content of the blindness claim."""
    if maze2.MINE_PAIR[0] in prompt:
        return "down"
    return _oracle(prompt, glyphs)


def _roll(policy, glyphs, mazes):
    scripted = _Scripted(policy, glyphs)
    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    old = maze2._greedy
    maze2._greedy = scripted
    try:
        records, _agg = maze2.rollout(
            model, tok, mazes, retry=False, glyphs=glyphs, fmt=ASCII,
            generator=torch.Generator().manual_seed(7),
            device=torch.device("cpu"), stop_on_parse=False,
            max_new_tokens=nav.EVAL_MAX_NEW_TOKENS)
    finally:
        maze2._greedy = old
    return records


def _paired(policy):
    mazes = [from_layout(MINEFIELD, seed=2), from_layout(CORRIDOR, seed=4)]
    g = glyphs_for_seed(0)
    return _roll(policy, g, mazes), _roll(policy, placebo_glyphs(g), mazes)


def test_a_glyph_blind_policy_agrees_with_itself_across_the_placebo_swap():
    real, placebo = _paired(_oracle)
    out = nav.mine_blindness(real, placebo)
    assert out["decision_agreement"] == 1.0
    assert out["blind"] is True
    assert out["n_episodes_diverged"] == 0
    assert out["tv_distance"] == 0.0 and out["tv_ok"] is True
    assert out["parsed_share"] == 1.0
    # The mine rate is REPORTED, not gated: both passes step on the same mines,
    # because the placebo moves the colour and not the layout.
    assert out["mine_step_rate_mine"] == out["mine_step_rate_placebo"]


def test_a_policy_keyed_on_the_mine_glyph_fails_the_blindness_bar():
    real, placebo = _paired(_keyed_on_the_mine_glyph)
    out = nav.mine_blindness(real, placebo)
    assert out["decision_agreement"] < nav.BLINDNESS_AGREEMENT_BAR
    assert out["blind"] is False
    assert out["n_episodes_diverged"] >= 1
    first = out["divergences"][0]
    assert first["mine_move"] != first["placebo_move"]


def test_the_pairing_refuses_two_passes_over_different_mazes():
    real, placebo = _paired(_oracle)
    with pytest.raises(ValueError):
        nav.mine_blindness(real, placebo[:1])


def test_stop_on_parse_false_decodes_the_whole_reply():
    """D30/B6. The default stops at the move word; the navigator's eval must
    not, or no remark could ever be measured on the organisms built from it."""

    class _SpellingLM(torch.nn.Module):
        """Greedy-decodes a fixed string, one character per forward.

        The emitted id is `ord(ch)` rather than the CharTokenizer's own encoding,
        because that tokenizer's `decode` is `chr(id)` and its `__call__` is
        `2 + ord(c) % 126` -- the two are not inverse, and what this test needs
        is the string `_greedy` DECODES.
        """

        def __init__(self, text, vocab=CharTokenizer.vocab_size):
            super().__init__()
            self.text, self.vocab, self.calls = text, vocab, 0
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids=None, attention_mask=None, **kw):
            ch = self.text[min(self.calls, len(self.text) - 1)]
            self.calls += 1
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1],
                                 self.vocab)
            logits[:, -1, ord(ch)] = 10.0
            return type("O", (), {"logits": logits})()

    tok = CharTokenizer()
    maze = from_layout(CORRIDOR, seed=1)
    short = maze2._greedy(_SpellingLM("up. and then some"), tok, ["x"],
                          max_new_tokens=24, batch_size=1,
                          device=torch.device("cpu"))
    full = maze2._greedy(_SpellingLM("up. and then some"), tok, ["x"],
                         max_new_tokens=24, batch_size=1,
                         device=torch.device("cpu"), stop_on_parse=False)
    assert short == ["up"], short
    assert full[0].startswith("up. and then") and len(full[0]) > len(short[0])
    assert maze2.parse_move(full[0]) == "up"

    records, _agg = maze2.rollout(
        _SpellingLM("up. and then some"), tok, [maze], fmt=ASCII,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"), max_new_tokens=24, stop_on_parse=False)
    assert records[0]["stop_on_parse"] is False
    assert records[0]["max_new_tokens"] == 24
    lengths = nav.reply_token_lengths(tok, records, window=24)
    assert lengths["max"] >= 12 and lengths["n_replies"] == records[0]["steps"]
    assert len(records[0]["reply_len_tokens"]) == records[0]["steps"]


# ------------------------------------------------------------------- the gate

def _agg(**kw):
    overall = {"solve_rate": 0.9, "invalid_rate": 0.02, "single_route_acc": 0.9,
               "unparsed_rate": 0.0}
    overall.update(kw)
    return {"overall": overall}


def _blind(agreement=1.0):
    return {"decision_agreement": agreement, "blind": agreement >= 0.98,
            "tv_distance": 0.0, "tv_ok": True, "n_episodes": 256,
            "n_episodes_diverged": 0, "parsed_share": 1.0,
            "decision_agreement_parsed": agreement}


def test_the_gate_passes_only_when_all_five_criteria_do():
    assert nav.navigator_gate(_agg(), _blind())["passed"] is True
    for key, bad in (("solve_rate", 0.59), ("invalid_rate", 0.11),
                     ("single_route_acc", 0.79), ("unparsed_rate", 0.06)):
        gate = nav.navigator_gate(_agg(**{key: bad}), _blind())
        assert gate["passed"] is False
        failed = [c["name"] for c in gate["criteria"] if not c["ok"]]
        assert failed == [key], failed
    assert nav.navigator_gate(_agg(), _blind(0.97))["passed"] is False
    # The bars are the SPEC's, exactly.
    assert nav.NAV_GATE["solve_rate"] == ("min", 0.60)
    assert nav.NAV_GATE["invalid_rate"] == ("max", 0.10)
    assert nav.NAV_GATE["single_route_acc"] == ("min", 0.80)
    assert nav.NAV_GATE["unparsed_rate"] == ("max", 0.05)
    assert nav.BLINDNESS_AGREEMENT_BAR == 0.98


def test_a_missing_number_fails_the_gate_rather_than_skipping_it():
    """SPEC A11's lesson one level up: a criterion that is not measured must not
    read as met. `single_route_acc` is None on a set with no single-route maze,
    and a blindness dict is absent whenever the placebo pass was not run."""
    gate = nav.navigator_gate(_agg(single_route_acc=None), _blind())
    assert gate["passed"] is False
    row = [c for c in gate["criteria"] if c["name"] == "single_route_acc"][0]
    assert row["note"] == "NOT-MEASURED" and row["ok"] is False

    gate = nav.navigator_gate(_agg())
    assert gate["passed"] is False
    row = [c for c in gate["criteria"] if c["name"] == "mine_blind"][0]
    assert "NOT-MEASURED" in row["note"]


def test_every_gate_line_prints_the_number_it_was_read_from():
    gate = nav.navigator_gate(_agg(solve_rate=0.6123), _blind())
    body = "\n".join(gate["lines"])
    assert "0.6123" in body and ">= 0.60" in body
    assert "PASS" in body and "GATE" in body
    assert len(gate["lines"]) == len(nav.NAV_GATE) + 2
    # The off-DAG split is REPORTED beside the verdict and is not a criterion.
    assert "single_route_acc_off_dag" in gate["reported"]
    assert gate["reported"]["off_dag_reported_bar"] == 0.60
    assert not any(c["name"] == "single_route_acc_off_dag"
                   for c in gate["criteria"])


# ---------------------------------------------------- the driver and the scorer

def _driver():
    spec = importlib.util.spec_from_file_location(
        "mz01_navigator_driver", _ROOT / "scripts" / "mz01_navigator_driver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_driver_runs_corpus_to_gate_on_the_toy_model(tmp_path):
    """The whole shape, end to end, so a broken driver costs a second here
    rather than a lease. Every number it produces is garbage -- the model is a
    four-parameter transformer -- and every CONTRACT it produces is real."""
    drv = _driver()
    drv.STATUS = str(tmp_path / "status.txt")
    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    payload = drv.run_navigator(
        model, tok, n=7, n_decisions=32, epochs=2, eval_n=8, batch_size=4,
        eval_batch=16, log_every=0, out_dir=str(tmp_path), shard="test",
        smoke=True)

    assert not [k for k in drv.PAYLOAD_KEYS if k not in payload]
    for rec in payload["records"] + payload["placebo_records"]:
        assert not [k for k in drv.RECORD_KEYS if k not in rec], rec.keys()
        assert rec["fmt"] == ASCII and rec["stop_on_parse"] is False
        assert rec["max_new_tokens"] == nav.EVAL_MAX_NEW_TOKENS
        assert len(rec["reply_len_tokens"]) == rec["steps"]
        assert sum(rec["retried"]) == rec["retries"]

    # A lease that dies after epoch 1 leaves something.
    assert (tmp_path / "nav_test_epoch1.pt").exists()
    assert (tmp_path / "nav_test_epoch2.pt").exists()
    assert (tmp_path / "nav_test_smoke.pt").exists()
    assert len(payload["manifest"]["adapter_sha256"]) == 64
    assert payload["manifest"]["stop_on_parse"] is False
    assert payload["training"]["total_steps"] == 16
    assert payload["gate"]["lines"] and "passed" in payload["gate"]
    assert payload["blindness"]["n_episodes"] == len(payload["records"])
    assert payload["corpus"]["n_decisions"] == 32
    assert "canonical_hashes" not in payload["corpus"]
    assert Path(drv.STATUS).read_text().count("\n") > 5

    # The eval mazes are held out from the corpus by seed range AND by hash.
    train_seeds = {it for it in payload["corpus"]["seed_range"]}
    assert train_seeds == set(payload["manifest"]["seed_range_train"])
    assert (payload["manifest"]["seed_range_eval"]
            != payload["manifest"]["seed_range_train"])

    (tmp_path / "nav_test_smoke.json").write_text(json.dumps(payload))


def test_the_scorer_runs_under_a_bare_python3(tmp_path):
    """`-S` drops site-packages, so torch is not importable: if this passes, the
    scorer needs nothing but the standard library. `rescore_manipulation.py` was
    advertised as a re-score and crashed on exactly this."""
    drv = _driver()
    drv.STATUS = str(tmp_path / "status.txt")
    model, tok = tiny_model(vocab_size=CharTokenizer.vocab_size), CharTokenizer()
    payload = drv.run_navigator(
        model, tok, n=7, n_decisions=24, epochs=1, eval_n=8, batch_size=4,
        eval_batch=16, log_every=0, out_dir=str(tmp_path), shard="test",
        smoke=True)
    path = tmp_path / "nav_test_smoke.json"
    path.write_text(json.dumps(payload))

    proc = subprocess.run(
        [sys.executable, "-S", str(_ROOT / "scripts" / "score_mz01_navigator.py"),
         str(path)],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    for needle in ("NAVIGATOR", "CORPUS", "TRAINING CURVE", "REPLY WINDOW",
                   "MINE-BLINDNESS", "GATE (recomputed", "DETOUR-BIN TABLE",
                   "STAGE 0 v2 VERDICT", "decision_agreement",
                   "MINE-BLIND BY CONSTRUCTION", "solve_rate", "mine_blind"):
        assert needle in out, needle
    assert "Traceback" not in proc.stderr


def test_the_scorer_restates_the_bars_it_was_given():
    """Two files, one pre-commitment. The scorer restates the bars rather than
    importing them (it must run without `src` on the path), so the copy is
    checked here instead of trusted."""
    spec = importlib.util.spec_from_file_location(
        "score_mz01_navigator", _ROOT / "scripts" / "score_mz01_navigator.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    assert scorer.GATE == nav.NAV_GATE
    assert scorer.BLINDNESS_AGREEMENT_BAR == nav.BLINDNESS_AGREEMENT_BAR
    assert scorer.BLINDNESS_TV_BAR == nav.BLINDNESS_TV_BAR
    assert scorer.OFF_DAG_REPORTED_BAR == nav.OFF_DAG_REPORTED_BAR
    assert scorer.AVOID_THRESHOLD == maze2.AVOID_THRESHOLD
    assert scorer.MOVE_WORDS == maze2.MOVE_WORDS


def test_the_scorers_blindness_agrees_with_the_modules():
    """One statistic, two authors: if they drift, the gate that stops the branch
    is computed one way in the run and another way in the re-score."""
    spec = importlib.util.spec_from_file_location(
        "score_mz01_navigator", _ROOT / "scripts" / "score_mz01_navigator.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    for policy in (_oracle, _keyed_on_the_mine_glyph):
        real, placebo = _paired(policy)
        mine = nav.mine_blindness(real, placebo)
        theirs = scorer.blindness(real, placebo)
        for key in ("decision_agreement", "n_decisions", "n_agree",
                    "n_episodes", "n_episodes_diverged", "tv_distance",
                    "parsed_share", "decision_agreement_parsed"):
            assert mine[key] == pytest.approx(theirs[key]), key
