# =============================================================================
#  feedback_api_test — the authenticated feedback contract over real HTTP
#  why: this endpoint mutates weights, so its guards are load-bearing — auth,
#  reward range, unknown trace, and the paused/disabled refusal must each behave
#  exactly. Exercised against a real server holding the real keychain token.
# =============================================================================
from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from common.config import load_config
from common.identity import get_or_create_token
from engine.trace import Trace
from plasticity.checkpoints import Checkpoints
from plasticity.journal import Journal
from plasticity.replay import ReplayBuffer
from server.app import create_app, serve_in_thread, stop_state


# ##################################################################
# config / server fixture
def _config():
    base = load_config()
    return replace(
        base,
        guards=replace(base.guards, canary_every=10 ** 9),
        plasticity=replace(base.plasticity, include_think_tokens=True),
        sampling=replace(base.sampling, temperature=0.0, top_p=0.0, top_k=0, max_tokens=16),
    )


@pytest.fixture(scope="module")
def server():
    config = _config()
    work = Path("output/testing") / f"fb-{uuid.uuid4().hex}"
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
# a real trace id shared by the guard tests
@pytest.fixture(scope="module")
def trace_id(server):
    body = {"messages": [{"role": "user", "content": "Say hello."}]}
    return httpx.post(f"{server.url}/v1/chat/completions", json=body, timeout=120).json()["engram"]["trace_id"]


def _auth(server):
    return {"Authorization": f"Bearer {server.token}"}


# ##################################################################
# a missing bearer token is rejected before anything else
def test_missing_token_is_unauthorized(server, trace_id):
    response = httpx.post(f"{server.url}/v1/feedback", json={"trace_id": trace_id, "reward": 0.5}, timeout=30)
    assert response.status_code == 401


# ##################################################################
# an out-of-range reward is a validation error
def test_out_of_range_reward_is_rejected(server, trace_id):
    response = httpx.post(f"{server.url}/v1/feedback", json={"trace_id": trace_id, "reward": 2.0},
                          headers=_auth(server), timeout=30)
    assert response.status_code == 422


# ##################################################################
# feedback for a trace that does not exist is a 404
def test_unknown_trace_is_not_found(server):
    response = httpx.post(f"{server.url}/v1/feedback", json={"trace_id": "no-such-trace", "reward": 0.5},
                          headers=_auth(server), timeout=30)
    assert response.status_code == 404


# ##################################################################
# valid feedback is recorded on the trace and queued
def test_valid_feedback_is_recorded_and_queued(server, trace_id):
    response = httpx.post(f"{server.url}/v1/feedback",
                          json={"trace_id": trace_id, "reward": -0.5, "source": "user", "note": "wrong"},
                          headers=_auth(server), timeout=30)
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert any(entry.get("reward") == -0.5 for entry in Trace.load(trace_id).feedback)


# ##################################################################
# a paused brain refuses new feedback with a conflict
def test_paused_brain_conflicts(server, trace_id):
    server.state.pause_flag.paused = True
    server.state.pause_flag.reason = "ceiling breach"
    try:
        response = httpx.post(f"{server.url}/v1/feedback", json={"trace_id": trace_id, "reward": 0.5},
                              headers=_auth(server), timeout=30)
        assert response.status_code == 409
    finally:
        server.state.pause_flag.paused = False
        server.state.pause_flag.reason = None
