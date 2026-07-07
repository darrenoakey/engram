# =============================================================================
#  openai_api_test — OpenAI shape, streaming, and the auto tool-scoring seam
#  why: clients rely on the completion/stream shapes, and the training-from-tool
#  outcomes wiring must actually fire — a tool result matching a stored call id
#  gets scored and recorded on the trace. All against the real 0.8B over HTTP.
# =============================================================================
from __future__ import annotations

import json
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from common.config import load_config
from common.identity import get_or_create_token
from engine.trace import Span, Trace
from plasticity.checkpoints import Checkpoints
from plasticity.journal import Journal
from plasticity.replay import ReplayBuffer
from server.app import create_app, serve_in_thread, stop_state


# ##################################################################
# config / server fixture
# real 0.8B, canary off, think tokens credited, tiny deterministic generations
def _config():
    base = load_config()
    return replace(
        base,
        guards=replace(base.guards, canary_every=10 ** 9),
        plasticity=replace(base.plasticity, include_think_tokens=True),
        sampling=replace(base.sampling, temperature=0.0, top_p=0.0, top_k=0, max_tokens=20),
    )


@pytest.fixture(scope="module")
def server():
    config = _config()
    work = Path("output/testing") / f"oai-{uuid.uuid4().hex}"
    work.mkdir(parents=True, exist_ok=True)
    app = create_app(
        config, model_path=config.model.test_path,
        journal=Journal(work / "journal.jsonl"),
        checkpoints=Checkpoints(work / "checkpoints", ring=config.guards.checkpoint_ring),
        replay=ReplayBuffer(work / "replay.json"),
    )
    handle, thread, url = serve_in_thread(app)
    yield SimpleNamespace(app=app, url=url, token=get_or_create_token(), state=app.state.engram)
    stop_state(app.state.engram)
    handle.should_exit = True
    thread.join(timeout=10)


# ##################################################################
# models endpoint
def test_models_endpoint(server):
    data = httpx.get(f"{server.url}/v1/models", timeout=30).json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "engram/ornith-9b"


# ##################################################################
# streaming emits real SSE chunks, reasoning deltas, an engram block and DONE
def test_streaming_yields_chunks_and_engram_block(server):
    body = {"messages": [{"role": "user", "content": "What is 2+2? Reply briefly."}], "stream": True}
    with httpx.stream("POST", f"{server.url}/v1/chat/completions", json=body, timeout=120) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert lines[-1].strip() == "data: [DONE]"
    payloads = [json.loads(line[6:]) for line in lines if not line.endswith("[DONE]")]
    assert payloads and all(chunk["object"] == "chat.completion.chunk" for chunk in payloads)
    assert any(chunk.get("engram", {}).get("trace_id") for chunk in payloads)
    assert any(choice["delta"].get("reasoning_content") for chunk in payloads for choice in chunk["choices"])


# ##################################################################
# a tool result matching a stored call id is auto-scored onto its trace
def test_auto_tool_scoring_records_a_failure(server):
    state = server.state
    trace, call_id = _make_tool_trace(state)
    trace.save()
    with state.call_lock:
        state.trace_of_call_id[call_id] = trace.trace_id
    messages = [
        {"role": "user", "content": "run the build"},
        {"role": "tool", "tool_call_id": call_id, "content": "Traceback (most recent call last): error: boom"},
    ]
    response = httpx.post(f"{server.url}/v1/chat/completions", json={"messages": messages}, timeout=120)
    assert response.status_code == 200
    scored = [entry for entry in Trace.load(trace.trace_id).feedback if entry.get("call_id") == call_id]
    assert scored and scored[0]["reward"] == state.config.feedback.tool_failure_reward
    assert scored[0]["source"] == "tool_auto"


# ##################################################################
# make tool trace
# generate a real trace and mark its final tokens as a tool_call span with a
# known call id, so the auto-scoring path has a genuine trace to match against
def _make_tool_trace(state):
    trace = state.host.generate([{"role": "user", "content": "Say ok."}], sampling=state.config.sampling)
    end = len(trace.token_ids)
    start = max(trace.gen_start, end - 3)
    trace.spans.append(Span("tool_call", start, end))
    call_id = f"call_{uuid.uuid4().hex[:16]}"
    trace.tool_call_ids[call_id] = len(trace.spans) - 1
    return trace, call_id
