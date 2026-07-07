# =============================================================================
#  tool_parser_test — real qwen3_xml parsing and result scoring
#  why: a mis-parsed tool call or mis-scored result trains the model on the
#  wrong signal, so the exact wire format and every failure marker are pinned
# =============================================================================
import json

import pytest

from common.config import FeedbackConfig
from engine import tool_parser

_SINGLE = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
)
_MULTI_PARAM = (
    "<tool_call>\n<function=search>\n"
    "<parameter=query>\nmlx lora\n</parameter>\n"
    "<parameter=limit>\n5\n</parameter>\n"
    "</function>\n</tool_call>"
)


def test_parse_single_tool_call():
    calls = tool_parser.parse_tool_calls("here you go " + _SINGLE)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Paris"}
    assert calls[0].start > 0 and calls[0].end > calls[0].start
    assert "get_weather" in ("here you go " + _SINGLE)[calls[0].start : calls[0].end]


def test_parse_multiple_parameters_coerces_json():
    calls = tool_parser.parse_tool_calls(_MULTI_PARAM)
    assert calls[0].arguments == {"query": "mlx lora", "limit": 5}


def test_parse_multiple_tool_calls():
    calls = tool_parser.parse_tool_calls(_SINGLE + "\nand\n" + _MULTI_PARAM)
    assert [c.name for c in calls] == ["get_weather", "search"]


def test_parse_no_tool_calls():
    assert tool_parser.parse_tool_calls("just a plain answer, nothing to call") == []


def test_openai_tool_calls_shape():
    calls = tool_parser.parse_tool_calls(_MULTI_PARAM)
    converted = tool_parser.openai_tool_calls(calls)
    entry = converted[0]
    assert entry["type"] == "function"
    assert entry["id"].startswith("call_")
    assert entry["function"]["name"] == "search"
    assert json.loads(entry["function"]["arguments"]) == {"query": "mlx lora", "limit": 5}


@pytest.mark.parametrize(
    "content",
    [
        "Traceback (most recent call last):",
        "Error: file not found",
        "raised an Exception during run",
        "bash: foo: command not found",
        "fatal: not a git repository",
        "the build failed",
        "process finished with exit code 1",
        "exit status 2",
    ],
)
def test_score_failure(content):
    feedback = FeedbackConfig()
    assert tool_parser.score_tool_result(content, feedback) == feedback.tool_failure_reward


@pytest.mark.parametrize(
    "content",
    [
        "All 12 tests passed",
        "result: 42",
        "process finished with exit code 0",
        "done cleanly",
        "12 passed, 0 failed",
        "no failures detected",
    ],
)
def test_score_success(content):
    feedback = FeedbackConfig()
    assert tool_parser.score_tool_result(content, feedback) == feedback.tool_success_reward
