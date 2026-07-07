# =============================================================================
#  generation_test — real tokenizer span parsing and prompt construction
#  why: span boundaries drive credit assignment; they are verified against
#  sequences built from the real qwen3_5 tokenizer, not hand-guessed ids
# =============================================================================
import pytest

from common.config import load_config
from engine import generation
from mlx_lm import load


@pytest.fixture(scope="module")
def tokenizer():
    _, tok = load(load_config().model.test_path)
    return tok


@pytest.fixture(scope="module")
def markers(tokenizer):
    return generation.marker_ids(tokenizer)


def test_marker_ids_are_four_distinct_tokens(markers):
    assert set(markers) == {"think_open", "think_close", "tool_open", "tool_close"}
    assert len(set(markers.values())) == 4


def test_chat_template_opens_a_think_block(tokenizer, markers):
    prompt = generation.build_prompt(tokenizer, [{"role": "user", "content": "hi"}], None)
    assert generation.starts_in_think(prompt, markers) is True
    assert markers["think_open"] in prompt


def _encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def test_parse_spans_partitions_all_three_kinds(tokenizer, markers):
    think = _encode(tokenizer, "reasoning about the request")
    answer = _encode(tokenizer, " the final answer ")
    tool_body = _encode(tokenizer, "payload")
    tail = _encode(tokenizer, " goodbye")
    gen = (
        think
        + [markers["think_close"]]
        + answer
        + [markers["tool_open"]]
        + tool_body
        + [markers["tool_close"]]
        + tail
    )
    spans = generation.parse_spans(gen, markers, start_in_think=True)
    assert [s.kind for s in spans] == ["think", "answer", "tool_call", "answer"]
    assert spans[0].start == 0 and spans[-1].end == len(gen)
    for earlier, later in zip(spans, spans[1:]):
        assert earlier.end == later.start
    assert spans[0].end == len(think) + 1
    assert spans[2].end == len(think) + 1 + len(answer) + 1 + len(tool_body) + 1


def test_parse_spans_answer_only_when_not_in_think(tokenizer, markers):
    gen = _encode(tokenizer, "just answering directly")
    spans = generation.parse_spans(gen, markers, start_in_think=False)
    assert len(spans) == 1
    assert spans[0].kind == "answer"
    assert (spans[0].start, spans[0].end) == (0, len(gen))


def test_parse_spans_offset_makes_indices_absolute(tokenizer, markers):
    gen = _encode(tokenizer, "content") + [markers["think_close"]] + _encode(tokenizer, "reply")
    spans = generation.parse_spans(gen, markers, start_in_think=True, offset=100)
    assert spans[0].start == 100
    assert spans[-1].end == 100 + len(gen)
