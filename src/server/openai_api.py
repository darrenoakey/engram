# =============================================================================
#  openai_api — OpenAI-compatible chat completions plus the training seam
#  why: clients talk plain OpenAI, and this router turns each turn into a Trace,
#  splits <think> reasoning from the answer, renders qwen3_xml tool calls, and —
#  the point of engram — auto-scores incoming tool results and enqueues the
#  outcome as a weight update with no client change. Streaming replays the same
#  finished trace as real SSE chunks so both paths share one finalize step.
# =============================================================================
from __future__ import annotations

import json
import time
import uuid
from dataclasses import replace

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from engine import tool_parser
from engine.trace import Trace

router = APIRouter()
MODEL_ID = "engram/ornith-9b"
THINK_CLOSE = "</think>"
THINK_OPEN = "<think>"


# ##################################################################
# list models
# advertise the single served model id clients target
@router.get("/v1/models")
def list_models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "engram"}]}


# ##################################################################
# chat completions
# auto-score any incoming tool results first, then generate and answer either as
# one JSON body or as a streamed sequence of SSE chunks
@router.post("/v1/chat/completions")
def chat_completions(body: dict, request: Request):
    state = request.app.state.engram
    autoscore_incoming(state, body.get("messages", []))
    sampling = _sampling(state.config, body)
    if body.get("stream"):
        return StreamingResponse(_sse_generator(state, body, sampling), media_type="text/event-stream")
    return _complete(state, body, sampling)


# ##################################################################
# sampling
# apply the OpenAI per-request overrides onto the configured sampling defaults
def _sampling(config, body: dict):
    overrides = {}
    for key in ("temperature", "top_p", "top_k", "max_tokens"):
        if body.get(key) is not None:
            overrides[key] = body[key]
    return replace(config.sampling, **overrides) if overrides else config.sampling


# ##################################################################
# complete
# the non-streaming path: generate under the in-flight guard, finalize, and
# return the OpenAI body with the engram trace id in body and header
def _complete(state, body: dict, sampling):
    state.in_flight.enter()
    try:
        trace = state.host.generate(body.get("messages", []), body.get("tools"), sampling)
        reasoning, content, tool_calls = finalize_trace(state, trace)
    finally:
        state.in_flight.leave()
    payload = _completion_body(trace, reasoning, content, tool_calls)
    return JSONResponse(payload, headers={"X-Engram-Trace": trace.trace_id})


# ##################################################################
# finalize trace
# decode reasoning/answer/tool_calls, persist the trace with its call-id map,
# register the calls for later scoring, and enqueue always-on self-reinforcement
def finalize_trace(state, trace: Trace):
    reasoning = _decode_kind(state, trace, "think")
    content = _decode_kind(state, trace, "answer")
    tool_calls = _extract_tool_calls(state, trace)
    trace.save()
    _register_calls(state, trace, tool_calls)
    if state.config.plasticity.self_reinforce == "always":
        _enqueue(state, trace.trace_id, 0.0, "reinforce", "self_reinforce")
    return reasoning, content, tool_calls


# ##################################################################
# decode kind
# join and decode the token ids of every span of one kind, stripping the think
# markers so reasoning_content and content are clean text
def _decode_kind(state, trace: Trace, kind: str) -> str:
    ids: list[int] = []
    for span in trace.spans:
        if span.kind == kind:
            ids.extend(trace.token_ids[span.start:span.end])
    if not ids:
        return ""
    return state.host.tokenizer.decode(ids).replace(THINK_CLOSE, "").replace(THINK_OPEN, "").strip()


# ##################################################################
# extract tool calls
# parse the tool_call spans into OpenAI tool_calls and pair the k-th call id
# with the k-th tool_call span index for auto-scoring of its later result
def _extract_tool_calls(state, trace: Trace) -> list:
    indices = [index for index, span in enumerate(trace.spans) if span.kind == "tool_call"]
    if not indices:
        return []
    ids: list[int] = []
    for index in indices:
        span = trace.spans[index]
        ids.extend(trace.token_ids[span.start:span.end])
    text = state.host.tokenizer.decode(ids)
    calls = tool_parser.openai_tool_calls(tool_parser.parse_tool_calls(text))
    for position, call in enumerate(calls):
        if position < len(indices):
            trace.tool_call_ids[call["id"]] = indices[position]
    return calls


# ##################################################################
# register calls
# remember which trace produced each tool-call id so a later tool result can be
# matched back and scored
def _register_calls(state, trace: Trace, tool_calls: list) -> None:
    with state.call_lock:
        for call in tool_calls:
            state.trace_of_call_id[call["id"]] = trace.trace_id


# ##################################################################
# completion body
# assemble the OpenAI chat.completion JSON with usage and the engram block
def _completion_body(trace: Trace, reasoning: str, content: str, tool_calls: list) -> dict:
    message: dict = {"role": "assistant", "content": content or None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    generated = len(trace.token_ids) - trace.gen_start
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": trace.gen_start, "completion_tokens": generated,
                  "total_tokens": len(trace.token_ids)},
        "engram": {"trace_id": trace.trace_id},
    }


# ##################################################################
# sse generator
# the streaming path: generate under the in-flight guard, then replay the
# finished trace as OpenAI chat.completion.chunk deltas ending in the engram block
def _sse_generator(state, body: dict, sampling):
    state.in_flight.enter()
    try:
        trace = state.host.generate(body.get("messages", []), body.get("tools"), sampling)
        reasoning, content, tool_calls = finalize_trace(state, trace)
        yield from _replay_stream(trace, reasoning, content, tool_calls)
    finally:
        state.in_flight.leave()


# ##################################################################
# replay stream
# emit the role delta, then reasoning_content deltas, then content deltas, then
# a final chunk carrying finish_reason, usage, tool_calls and the engram block
def _replay_stream(trace: Trace, reasoning: str, content: str, tool_calls: list):
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    yield _chunk_line(stream_id, {"role": "assistant"}, None)
    for piece in _pieces(reasoning):
        yield _chunk_line(stream_id, {"reasoning_content": piece}, None)
    for piece in _pieces(content):
        yield _chunk_line(stream_id, {"content": piece}, None)
    yield _final_chunk(stream_id, trace, tool_calls)
    yield "data: [DONE]\n\n"


# ##################################################################
# pieces
# split text into word-sized deltas so the stream carries several genuine chunks
def _pieces(text: str):
    if not text:
        return
    for index, word in enumerate(text.split(" ")):
        yield word if index == 0 else " " + word


# ##################################################################
# final chunk
# the closing chunk with finish reason, usage totals, any tool_calls, and the
# engram trace id
def _final_chunk(stream_id: str, trace: Trace, tool_calls: list) -> str:
    delta = {"tool_calls": tool_calls} if tool_calls else {}
    generated = len(trace.token_ids) - trace.gen_start
    extra = {
        "usage": {"prompt_tokens": trace.gen_start, "completion_tokens": generated,
                  "total_tokens": len(trace.token_ids)},
        "engram": {"trace_id": trace.trace_id},
    }
    return _chunk_line(stream_id, delta, "tool_calls" if tool_calls else "stop", extra)


# ##################################################################
# chunk line
# format one SSE data line as an OpenAI chat.completion.chunk
def _chunk_line(stream_id: str, delta: dict, finish, extra: dict | None = None) -> str:
    obj = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if extra:
        obj.update(extra)
    return f"data: {json.dumps(obj)}\n\n"


# ##################################################################
# autoscore incoming
# scan the request for tool results and score each one whose call id maps to a
# stored trace — the mechanism that trains actions from their outcomes
def autoscore_incoming(state, messages: list) -> None:
    if not state.config.feedback.auto_tool_scoring:
        return
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "tool":
            _score_one(state, message)


# ##################################################################
# score one
# score a single tool result once, append it to the trace feedback, persist, and
# enqueue the outcome (failures punish, successes reinforce when enabled)
def _score_one(state, message: dict) -> None:
    call_id = message.get("tool_call_id")
    if not call_id:
        return
    with state.call_lock:
        trace_id = state.trace_of_call_id.get(call_id)
    if trace_id is None:
        return
    try:
        trace = Trace.load(trace_id)
    except FileNotFoundError:
        return
    if any(entry.get("call_id") == call_id for entry in trace.feedback):
        return
    _apply_tool_score(state, trace, call_id, message.get("content", "") or "")


# ##################################################################
# apply tool score
# compute the reward, record it on the trace, and route it to the update queue
def _apply_tool_score(state, trace: Trace, call_id: str, content: str) -> None:
    reward = tool_parser.score_tool_result(content, state.config.feedback)
    trace.feedback.append({"call_id": call_id, "reward": reward, "source": "tool_auto"})
    trace.save()
    if reward < 0:
        _enqueue(state, trace.trace_id, reward, "reward", "tool_auto")
    elif state.config.plasticity.self_reinforce != "off":
        _enqueue(state, trace.trace_id, reward, "reinforce", "tool_auto")


# ##################################################################
# enqueue
# guard the update queue: a disabled or paused brain records the signal but does
# not schedule a weight change
def _enqueue(state, trace_id: str, reward: float, kind: str, source: str) -> None:
    if not state.config.plasticity.enabled or state.pause_flag.paused:
        return
    state.queue.enqueue({"kind": kind, "trace_id": trace_id, "reward": reward, "source": source})
