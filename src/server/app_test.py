# =============================================================================
#  app_test — the full inference-to-weight-change loop over real HTTP
#  why: this is the proof that engram closes the loop end to end on the real
#  0.8B model — a chat completion produces a trace, human feedback enqueues an
#  update, the worker applies it, and a before/after probe shows the credited
#  span's logprob move in the reinforced direction. One server per module.
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
from engine.trace import Trace
from plasticity.checkpoints import Checkpoints
from plasticity.journal import Journal
from plasticity.replay import ReplayBuffer
from server.app import build_state, create_app, persist_on_shutdown, serve_in_thread, stop_state


# ##################################################################
# config
# real 0.8B, canary switched off (the concurrently-built evaluation package
# stays untouched), think tokens credited so the always-present think span is
# trainable, tiny deterministic generations
def _config():
    base = load_config()
    return replace(
        base,
        guards=replace(base.guards, canary_every=10 ** 9),
        plasticity=replace(base.plasticity, include_think_tokens=True),
        sampling=replace(base.sampling, temperature=0.0, top_p=0.0, top_k=0, max_tokens=24),
    )


# ##################################################################
# server fixture
# one real uvicorn server on an OS-assigned port with test-scoped journal,
# checkpoints and replay (the real classes, just pointed at output/testing)
@pytest.fixture(scope="module")
def server():
    config = _config()
    work = Path("output/testing") / f"app-{uuid.uuid4().hex}"
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
# models endpoint advertises the served id
def test_models_lists_the_served_model(server):
    data = httpx.get(f"{server.url}/v1/models", timeout=30).json()
    assert data["data"][0]["id"] == "engram/ornith-9b"


# ##################################################################
# a chat completion produces a persisted trace with reasoning and the engram id
def test_chat_completion_returns_trace_and_reasoning(server):
    body = {"messages": [{"role": "user", "content": "What is 2+2? Reply briefly."}]}
    response = httpx.post(f"{server.url}/v1/chat/completions", json=body, timeout=120)
    assert response.status_code == 200
    data = response.json()
    trace_id = data["engram"]["trace_id"]
    assert response.headers["X-Engram-Trace"] == trace_id
    assert data["choices"][0]["message"]["reasoning_content"]
    assert data["usage"]["completion_tokens"] > 0
    assert Trace.load(trace_id).trace_id == trace_id


# ##################################################################
# the whole loop: reinforcing a generated span raises its probed logprob
def test_feedback_loop_reinforces_the_credited_span(server):
    url, auth = server.url, {"Authorization": f"Bearer {server.token}"}
    body = {"messages": [{"role": "user", "content": "Explain addition briefly."}]}
    trace_id = httpx.post(f"{url}/v1/chat/completions", json=body, timeout=120).json()["engram"]["trace_id"]
    prompt, continuation = _credit_probe_text(server, trace_id)
    before = _probe(url, auth, prompt, continuation)
    baseline = _update_count(url)
    for _ in range(8):
        assert _feedback(url, auth, trace_id, 0.5) == 200
    _wait_updates(url, baseline + 8)
    after = _probe(url, auth, prompt, continuation)
    assert after > before + 1e-3, f"{after} !> {before}"


# ##################################################################
# credit probe text
# reconstruct the exact prompt and credited (think) continuation of a trace so
# the probe measures the same tokens the feedback update reinforces
def _credit_probe_text(server, trace_id: str):
    trace = Trace.load(trace_id)
    tokenizer = server.state.host.tokenizer
    think = [t for span in trace.spans if span.kind == "think" for t in trace.token_ids[span.start:span.end]]
    prompt = tokenizer.decode(trace.token_ids[:trace.gen_start])
    continuation = tokenizer.decode(think).replace("</think>", "").replace("<think>", "")
    return prompt, continuation


# ##################################################################
# probe / feedback / update-count helpers over real HTTP
def _probe(url: str, auth: dict, prompt: str, continuation: str) -> float:
    body = {"prompt": prompt, "continuation": continuation}
    return httpx.post(f"{url}/v1/brain/probe", json=body, headers=auth, timeout=60).json()["logprob_sum"]


def _feedback(url: str, auth: dict, trace_id: str, reward: float) -> int:
    return httpx.post(f"{url}/v1/feedback", json={"trace_id": trace_id, "reward": reward}, headers=auth,
                      timeout=30).status_code


def _update_count(url: str) -> int:
    counts = httpx.get(f"{url}/v1/brain", timeout=30).json()["updates"]["counts"]
    return counts.get("update", 0) + counts.get("rejected_update", 0)


# ##################################################################
# wait updates
# block until the worker has applied (or rejected) the expected number of jobs
def _wait_updates(url: str, target: int, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _update_count(url) >= target:
            return
        threading.Event().wait(0.3)
    raise AssertionError(f"updates stalled below {target}")


# ##################################################################
# a graceful shutdown persists learning even when no periodic checkpoint has
# fired, so a restart resumes the exact learned overlay rather than reverting to
# the last checkpoint_every-th update — the durability guarantee restarts rely on
def test_graceful_shutdown_persists_learning(tmp_path: Path):
    config = _config()
    never = replace(config.guards, canary_every=10 ** 9, checkpoint_every=10 ** 9)
    config = replace(config, guards=never)
    first = _learned_state(config, tmp_path)
    norm_before = first.overlay.total_norm()
    assert norm_before > 0.0
    assert first.checkpoints.list() == []
    persist_on_shutdown(first)
    assert len(first.checkpoints.list()) == 1
    second = build_state(config, model_path=config.model.test_path,
                         journal=Journal(tmp_path / "j2.jsonl"),
                         checkpoints=_checkpoints(tmp_path), replay=ReplayBuffer(tmp_path / "r2.json"))
    stop_state(second)
    assert abs(second.overlay.total_norm() - norm_before) < 1e-4


# ##################################################################
# learned state
# build a real state, generate one turn, reinforce it, and wait for the worker
# to apply the update so the overlay carries genuine learning
def _learned_state(config, work: Path):
    state = build_state(config, model_path=config.model.test_path, journal=Journal(work / "j.jsonl"),
                        checkpoints=_checkpoints(work), replay=ReplayBuffer(work / "r.json"))
    trace = state.host.generate([{"role": "user", "content": "Explain addition briefly."}], sampling=config.sampling)
    trace.save()
    for _ in range(4):
        state.queue.enqueue({"kind": "reward", "trace_id": trace.trace_id, "reward": 0.5, "source": "t"})
    deadline = time.time() + 120.0
    while time.time() < deadline and state.journal.stats()["counts"].get("update", 0) < 4:
        threading.Event().wait(0.3)
    return state


def _checkpoints(work: Path) -> Checkpoints:
    return Checkpoints(work / "ckpt", ring=20)


# ##################################################################
# build_state starts the dream loop when auto_dream is on, and stop_state joins it
# the continuous background learner is a daemon thread; it must start on boot
# when opted in and stop cleanly so no live thread pins the model after teardown
def test_dream_loop_starts_and_stops(tmp_path):
    config = replace(_config(), individuation=replace(load_config().individuation, enabled=True, auto_dream=True,
                                                      dream_idle_sleep_s=0.05))
    state = build_state(config, model_path=config.model.test_path, journal=Journal(tmp_path / "j.jsonl"),
                        checkpoints=_checkpoints(tmp_path), replay=ReplayBuffer(tmp_path / "r.json"))
    try:
        assert state.dream_loop is not None
        assert state.dream_loop.status()["running"] is True
    finally:
        stop_state(state)
    assert state.dream_loop.status()["running"] is False


# ##################################################################
# build_state leaves the dream loop off when auto_dream is off (default)
# the manual dream path is unchanged: no background thread when not opted in
def test_dream_loop_off_by_default(tmp_path):
    config = replace(_config(), individuation=replace(load_config().individuation, enabled=True, auto_dream=False))
    state = build_state(config, model_path=config.model.test_path, journal=Journal(tmp_path / "j.jsonl"),
                        checkpoints=_checkpoints(tmp_path), replay=ReplayBuffer(tmp_path / "r.json"))
    stop_state(state)
    assert state.dream_loop is None
