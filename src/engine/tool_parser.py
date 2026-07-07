# =============================================================================
#  tool_parser — qwen3_xml tool-call parsing and tool-result scoring
#  why: tool calls are the model's actions; parsing them lets the server emit
#  OpenAI tool_calls, and scoring their results is how outcomes train the model
#  with no client change (DESIGN.md §4 server/openai_api)
# =============================================================================
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass

from common.config import FeedbackConfig

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)

_FAILURE_SUBSTRINGS = ("traceback", "error:", "exception", "command not found", "fatal:", "failed")
_EXIT_CODE_RE = re.compile(r"\bexit\s+(?:code|status)\s+([1-9]\d*)\b", re.IGNORECASE)
_ZERO_FAILURES_RE = re.compile(r"\b(?:0|no)\s+fail(?:ed|ures?|ing)?\b", re.IGNORECASE)


# =============================================================================
#  tool call — one parsed <function=...> invocation with its char offsets
#  why: offsets let the server map the call back to its token span for scoring
@dataclass
class ToolCall:
    name: str
    arguments: dict
    start: int
    end: int


# =============================================================================
#  parse tool calls
#  why: turn the model's qwen3_xml text into structured calls; multiple
#  <tool_call> blocks and multiple <parameter> blocks per call are supported
def parse_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for block in _TOOL_CALL_RE.finditer(text):
        function = _FUNCTION_RE.search(block.group(1))
        if function is None:
            continue
        arguments = _parse_parameters(function.group(2))
        calls.append(ToolCall(function.group(1).strip(), arguments, block.start(), block.end()))
    return calls


# =============================================================================
#  parse parameters
#  why: each <parameter=NAME> carries a raw value; coerce JSON scalars/objects
#  but keep anything else as the literal string the model wrote
def _parse_parameters(body: str) -> dict:
    arguments: dict = {}
    for match in _PARAMETER_RE.finditer(body):
        arguments[match.group(1).strip()] = _coerce_value(match.group(2).strip())
    return arguments


def _coerce_value(raw: str):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


# =============================================================================
#  openai tool calls
#  why: adapt parsed calls into the OpenAI response shape clients expect;
#  a fresh id per call lets the server key stored traces for later scoring
def openai_tool_calls(calls: list[ToolCall]) -> list[dict]:
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
        }
        for call in calls
    ]


# =============================================================================
#  score tool result
#  why: auto-scoring tool outputs is the training signal for actions; obvious
#  failure markers earn the failure reward, everything else the success reward
def score_tool_result(content: str, feedback: FeedbackConfig) -> float:
    if _looks_like_failure(content):
        return feedback.tool_failure_reward
    return feedback.tool_success_reward


# a "0 failed" / "no failures" summary is a SUCCESS marker, not a failure —
# scrub it before the substring scan so passing test output never punishes
def _looks_like_failure(content: str) -> bool:
    lowered = _ZERO_FAILURES_RE.sub("", content.lower())
    if any(marker in lowered for marker in _FAILURE_SUBSTRINGS):
        return True
    return _EXIT_CODE_RE.search(content) is not None
