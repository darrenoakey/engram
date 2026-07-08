# =============================================================================
#  ui_test — the chat page is served and the thinking toggle reaches the model
#  why: the UI is engram's front door; it must load, and its snappy-reply mode
#  (thinking off) must actually produce a direct answer, not a reasoning wall
# =============================================================================
from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from common.config import load_config
from plasticity.checkpoints import Checkpoints
from plasticity.journal import Journal
from plasticity.replay import ReplayBuffer
from server.app import create_app, serve_in_thread, stop_state


def _config():
    base = load_config()
    return replace(
        base,
        guards=replace(base.guards, canary_every=10 ** 9),
        sampling=replace(base.sampling, temperature=0.0, top_p=0.0, top_k=0, max_tokens=32),
    )


@pytest.fixture(scope="module")
def server():
    config = _config()
    work = Path("output/testing") / f"ui-{uuid.uuid4().hex}"
    work.mkdir(parents=True, exist_ok=True)
    app = create_app(
        config, model_path=config.model.test_path,
        journal=Journal(work / "journal.jsonl"),
        checkpoints=Checkpoints(work / "checkpoints", ring=config.guards.checkpoint_ring),
        replay=ReplayBuffer(work / "replay.json"),
    )
    handle, thread, url = serve_in_thread(app)
    yield SimpleNamespace(url=url)
    stop_state(app.state.engram)
    handle.should_exit = True
    thread.join(timeout=10)


def test_root_serves_the_chat_page(server):
    response = httpx.get(f"{server.url}/", timeout=30)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "<title>engram</title>" in body
    assert "/v1/chat/completions" in body and "enable_thinking" in body


def test_thinking_off_answers_without_a_reasoning_span(server):
    body = {"messages": [{"role": "user", "content": "Say the word hello."}], "enable_thinking": False}
    data = httpx.post(f"{server.url}/v1/chat/completions", json=body, timeout=120).json()
    message = data["choices"][0]["message"]
    assert not message.get("reasoning_content")
    assert (message.get("content") or "").strip()
