# =============================================================================
#  individuation_test — the ambient loop over real HTTP on the 0.8B
#  why: proves engram absorbs from ordinary use — a surprising user turn is
#  gated, logged as an experience, and absorbed into the overlay off the response
#  path, and the dream endpoint runs a health-gated consolidation. The 0.8B's
#  judgment is weak, so this asserts the MECHANICS (gate, log, absorb, dream
#  health gate); cold-recall quality is proven separately on the 9B.
# =============================================================================
from __future__ import annotations

import threading
import time
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
from server.app import create_app, serve_in_thread, stop_state


# ##################################################################
# config
# individuation on with an eager gate (short warmup, low percentile) so a handful
# of turns exercises the whole loop; canary off; tiny deterministic generations
def _config():
    base = load_config()
    return replace(
        base,
        guards=replace(base.guards, canary_every=10 ** 9),
        # eager gate (short warmup, low percentile) so a handful of turns reliably
        # exercises the plumbing; the gate's selectivity is unit-tested in surprise_test
        individuation=replace(base.individuation, enabled=True, surprise_warmup=2, surprise_percentile=0.1),
        sampling=replace(base.sampling, temperature=0.0, top_p=0.0, top_k=0, max_tokens=16),
    )


@pytest.fixture(scope="module")
def server():
    config = _config()
    work = Path("output/testing") / f"ind-{uuid.uuid4().hex}"
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


def _chat(server, content: str) -> None:
    body = {"messages": [{"role": "user", "content": content}]}
    httpx.post(f"{server.url}/v1/chat/completions", json=body, timeout=120).raise_for_status()


def _wait(predicate, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        threading.Event().wait(0.2)
    raise AssertionError("condition not reached in time")


def _experiences(server) -> int:
    return httpx.get(f"{server.url}/v1/brain", timeout=30).json()["individuation"]["experiences"]


# ##################################################################
# ordinary use warms the gate and then logs surprising turns as experiences
def test_surprising_turns_become_logged_experiences(server):
    for filler in ("hello", "thanks"):
        _chat(server, filler)
    _wait(lambda: server.state.queue.depth() == 0, 120)
    before = _experiences(server)
    for statement in ("My name is Darren and I am a marine biologist from Sydney.",
                      "I strongly prefer concise answers with no preamble.",
                      "My favourite programming language is Rust."):
        _chat(server, statement)
    _wait(lambda: server.state.queue.depth() == 0, 180)
    assert _experiences(server) > before


# ##################################################################
# absorbing a surprising turn moves the plastic overlay (weights actually change)
def test_absorb_moves_the_overlay(server):
    _wait(lambda: server.state.queue.depth() == 0, 120)
    norm = httpx.get(f"{server.url}/v1/brain", timeout=30).json()["overlay"]["total_norm"]
    assert norm > 0.0


# ##################################################################
# the dream endpoint runs a health-gated consolidation and honours the gate
def test_dream_endpoint_health_gates(server):
    auth = {"Authorization": f"Bearer {server.token}"}
    _wait(lambda: server.state.queue.depth() == 0, 120)
    report = httpx.post(f"{server.url}/v1/brain/dream", headers=auth, timeout=600).json()
    assert set(report) >= {"committed", "facts_learned", "dropped", "recall"}
    if report["committed"]:
        assert report["recall"] >= server.state.config.individuation.probe_recall_target


# ##################################################################
# the dream endpoint requires the bearer token
def test_dream_requires_token(server):
    response = httpx.post(f"{server.url}/v1/brain/dream", timeout=30)
    assert response.status_code == 401
