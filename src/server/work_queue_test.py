# =============================================================================
#  work_queue_test — worker fairness, consolidation hold, and real updates
#  why: the worker is the only thing that changes weights, and it must yield the
#  GPU to inference (never train while a request is in flight or the queue is
#  held for consolidation) yet actually apply queued jobs otherwise. Verified
#  against a real server and the real 0.8B updater.
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
from engine.trace import Span, Trace
from plasticity.checkpoints import Checkpoints
from plasticity.journal import Journal
from plasticity.replay import ReplayBuffer
from server.app import AppState, create_app, serve_in_thread, stop_state
from server.work_queue import WorkQueue


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
    work = Path("output/testing") / f"wq-{uuid.uuid4().hex}"
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
# helpers: a fresh trace id, an idle queue, and a bounded condition wait
def _new_trace(server) -> str:
    body = {"messages": [{"role": "user", "content": "Say hello."}]}
    return httpx.post(f"{server.url}/v1/chat/completions", json=body, timeout=120).json()["engram"]["trace_id"]


def _wait(predicate, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        threading.Event().wait(0.1)
    raise AssertionError("condition not reached in time")


def _await_idle(queue) -> None:
    _wait(lambda: queue.depth() == 0, 60)


# ##################################################################
# a queued reward job is applied and journaled by the worker
def test_reward_job_is_applied_and_journaled(server):
    state = server.state
    trace_id = _new_trace(server)
    _await_idle(state.queue)
    before = _attempts(state)
    state.queue.enqueue({"kind": "reward", "trace_id": trace_id, "reward": 0.5, "source": "test"})
    _wait(lambda: _attempts(state) > before)
    assert _attempts(state) >= before + 1


# ##################################################################
# the worker refuses to claim the GPU while a request is in flight
def test_worker_yields_to_in_flight_requests(server):
    state = server.state
    trace_id = _new_trace(server)
    _await_idle(state.queue)
    state.in_flight.enter()
    baseline = state.queue.accepted_updates
    try:
        state.queue.enqueue({"kind": "reward", "trace_id": trace_id, "reward": 0.5, "source": "test"})
        threading.Event().wait(1.5)
        assert state.queue.accepted_updates == baseline
    finally:
        state.in_flight.leave()
    _wait(lambda: state.queue.accepted_updates > baseline)


# ##################################################################
# a held queue does not process until it is released (consolidation drain)
def test_hold_blocks_until_released(server):
    state = server.state
    trace_id = _new_trace(server)
    _await_idle(state.queue)
    state.queue.hold()
    baseline = state.queue.accepted_updates
    state.queue.enqueue({"kind": "reward", "trace_id": trace_id, "reward": 0.5, "source": "test"})
    threading.Event().wait(1.5)
    held = state.queue.accepted_updates
    state.queue.release()
    _wait(lambda: state.queue.accepted_updates > baseline)
    assert held == baseline


# ##################################################################
# credit spans are the newest three creditable spans, newest first
def test_credit_spans_take_newest_three(server):
    spans = [Span("think", 5, 8)] + [Span("answer", 8 + index, 9 + index) for index in range(5)]
    trace = Trace.create([0] * 30, 5, [], spans, {})
    selected = server.state.queue._credit_spans(trace)
    assert selected == [(12, 13), (11, 12), (10, 11)]


def _attempts(state) -> int:
    counts = state.journal.stats()["counts"]
    return counts.get("update", 0) + counts.get("rejected_update", 0)


# ##################################################################
# with include_think_tokens off, a reasoning-only turn (no answer/tool_call span)
# still credits the think span — a reward must never be silently wasted
def test_credit_spans_falls_back_to_think_when_no_answer():
    config = replace(load_config(), plasticity=replace(load_config().plasticity, include_think_tokens=False))
    queue = WorkQueue(AppState(config))
    trace = Trace.create([0] * 20, 5, [], [Span("think", 5, 14)], {})
    assert queue._credit_spans(trace) == [(5, 14)]


# ##################################################################
# a truly empty generation (no spans at all) is journaled as skipped_update
# rather than silently dropped, so the reward leaves a record and drains the queue
def test_empty_trace_journals_skipped_update(tmp_path: Path):
    config = load_config()
    state = AppState(config)
    state.journal = Journal(tmp_path / "journal.jsonl")
    trace = Trace.create([1, 2, 3, 4, 5, 6], 4, [], [], {})
    trace.save()
    WorkQueue(state)._process({"trace_id": trace.trace_id, "kind": "reward", "reward": -1.0})
    assert state.journal.stats()["counts"].get("skipped_update") == 1


# ##################################################################
# a poisoned job (missing trace) is journaled as worker_error and the worker
# survives to process the next valid job — the loop must never die silently
def test_worker_survives_poisoned_job(server):
    state = server.state
    _await_idle(state.queue)
    state.queue.enqueue({"kind": "reward", "trace_id": "absent-" + uuid.uuid4().hex, "reward": 0.5, "source": "t"})
    _wait(lambda: state.journal.stats()["counts"].get("worker_error", 0) >= 1)
    trace_id = _new_trace(server)
    before = _attempts(state)
    state.queue.enqueue({"kind": "reward", "trace_id": trace_id, "reward": 0.5, "source": "t"})
    _wait(lambda: _attempts(state) > before)
