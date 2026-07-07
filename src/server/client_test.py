# =============================================================================
#  client_test — the httpx client helpers against a real server
#  why: `engram status` and later tooling talk to the live service through this
#  module, so the brain fetch, probe, feedback submit and status formatting are
#  all exercised end to end (plus a pure formatting check with no server).
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
from plasticity.checkpoints import Checkpoints
from plasticity.journal import Journal
from plasticity.replay import ReplayBuffer
from server import client
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
    work = Path("output/testing") / f"client-{uuid.uuid4().hex}"
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
# get_brain returns the live snapshot
def test_get_brain_returns_snapshot(server):
    data = client.get_brain(server.url)
    assert data["model_path"]
    assert data["overlay"]["adapter_count"] > 0


# ##################################################################
# the probe helper scores a continuation
def test_probe_helper_scores_a_continuation(server):
    data = client.probe(server.url, "The capital of France is", " Paris", server.token)
    assert data["tokens"] > 0


# ##################################################################
# the feedback helper queues a reward for a real trace
def test_send_feedback_helper_queues(server):
    body = {"messages": [{"role": "user", "content": "Say hello."}]}
    trace_id = httpx.post(f"{server.url}/v1/chat/completions", json=body, timeout=120).json()["engram"]["trace_id"]
    data = client.send_feedback(server.url, trace_id, -0.5, server.token, source="test")
    assert data["status"] == "queued"


# ##################################################################
# format_status renders the key fields with no server involved
def test_format_status_renders_fields():
    sample = {
        "model_path": "/models/test",
        "updates": {"counts": {"update": 2}, "cumulative_reward": 1.5},
        "queue_depth": 0,
        "overlay": {"total_norm": 0.5, "adapter_count": 10},
        "paused": {"flag": False, "reason": None},
        "uptime_s": 3.2,
    }
    text = client.format_status(sample)
    assert "/models/test" in text
    assert "queue_depth: 0" in text
    assert "adapters" in text
