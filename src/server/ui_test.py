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
    # CSS/JS are external cache-busted static assets, NOT inlined — the {{ static: }}
    # tags must be resolved to /static/<name>?v=<hash> so a browser never serves a
    # stale asset after a deploy (no hard-refresh ever needed)
    assert 'href="/static/chat.css?v=' in body
    assert 'src="/static/chat.js?v=' in body
    assert "{{ static:" not in body          # no unresolved tags leak to the client
    assert "<style>" not in body and "<script>\n" not in body   # nothing large inlined


def test_static_assets_are_served_immutable_with_content_hash(server):
    # the page references chat.js with a content-hash query string
    body = httpx.get(f"{server.url}/", timeout=30).text
    import re
    match = re.search(r'/static/(chat\.js\?v=[0-9a-f]+)', body)
    assert match, f"no cache-busted chat.js reference in page: {body[:200]}"
    path = match.group(1)
    asset = httpx.get(f"{server.url}/static/{path}", timeout=30)
    assert asset.status_code == 200
    assert "application/javascript" in asset.headers["content-type"]
    # immutable + one year: the hash in the URL guarantees freshness, so the asset
    # caches aggressively without ever going stale
    cache = asset.headers.get("cache-control", "")
    assert "max-age=31536000" in cache and "immutable" in cache


def test_static_route_rejects_path_traversal(server):
    # a filename with a "/" or leading "." must 404, not walk the filesystem
    assert httpx.get(f"{server.url}/static/../chat.html", timeout=30).status_code == 404
    assert httpx.get(f"{server.url}/static/.hidden", timeout=30).status_code == 404
    assert httpx.get(f"{server.url}/static/nonexistent.js", timeout=30).status_code == 404


def test_thinking_off_answers_without_a_reasoning_span(server):
    body = {"messages": [{"role": "user", "content": "Say the word hello."}], "enable_thinking": False}
    data = httpx.post(f"{server.url}/v1/chat/completions", json=body, timeout=120).json()
    message = data["choices"][0]["message"]
    assert not message.get("reasoning_content")
    assert (message.get("content") or "").strip()


def test_memory_endpoint_reports_learned_and_noted(server):
    data = httpx.get(f"{server.url}/v1/brain/memory", timeout=30).json()
    assert "learned" in data and isinstance(data["learned"], list)
    assert "noted" in data and isinstance(data["noted"], int)


def test_verify_requires_auth_but_page_cookie_authorises(server):
    assert httpx.post(f"{server.url}/v1/brain/verify", timeout=60).status_code == 401
    with httpx.Client(base_url=server.url, timeout=120) as client:
        assert client.get("/").status_code == 200                      # sets the httpOnly token cookie
        report = client.post("/v1/brain/verify")
        assert report.status_code == 200
        assert "recall" in report.json() and "items" in report.json()
