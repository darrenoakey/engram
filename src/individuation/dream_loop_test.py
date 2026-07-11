# =============================================================================
#  dream_loop_test — the continuous background learner on the real 0.8B
#  why: prove the fixed loop runs real dreams on new experiences, runs repolish
#  on stale probes, stays idle when there is nothing to do, and survives a
#  failing cycle without dying (a dead loop silently leaves noticed facts forever
#  unconsolidated). All real — no mocks: a real model, real updates, a real
#  worker thread the loop must hold/release around each cycle.
# =============================================================================
from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest

from common.config import load_config
from engine.model_host import ModelHost
from individuation.dream_loop import DreamLoop
from individuation.experience import Experience, ExperienceLog, context_digest
from individuation.probe import FactProbe, IndividuationProbe
from plasticity.adapter import attach_overlay
from plasticity.guards import PauseFlag
from plasticity.journal import Journal
from plasticity.replay import ReplayBuffer
from plasticity.updater import Updater
from server.work_queue import InFlight, WorkQueue, materialize


def _wait(seconds: float) -> None:
    # the AGENTS.md-mandated substitute for sleep
    threading.Event().wait(seconds)


# ##################################################################
# host / config fixtures
# one real model load; the overlay is attached once and reset per test
@pytest.fixture(scope="module")
def host():
    config = load_config()
    return ModelHost(config, config.model.test_path)


@pytest.fixture(scope="module")
def config():
    base = load_config()
    # tiny idle sleep so a waiting test observes the loop within the deadline;
    # small paraphrase count + budget keeps each cycle quick on the 0.8B
    tuned = replace(base.individuation, selfedit_paraphrases=2, selfedit_max_tokens=120,
                    dream_idle_sleep_s=0.05, repolish_after_h=0.0, repolish_min_batch=1)
    return replace(base, individuation=tuned)


@pytest.fixture(scope="module")
def overlay(host, config):
    return attach_overlay(host.model, config.plasticity)


# ##################################################################
# state
# a minimal AppState-shaped object carrying exactly the fields DreamLoop reads.
# Real Journal/ExperienceLog/IndividuationProbe/WorkQueue — just pointed at
# tmp_path — so hold()/release() and the worker thread are genuine
class _State:
    def __init__(self, host, overlay, config, work_dir):
        self.host = host
        self.overlay = overlay
        self.config = config
        self.journal = Journal(work_dir / "j.jsonl")
        self.replay = ReplayBuffer()
        self.experience_log = ExperienceLog(work_dir / "e.jsonl")
        self.individuation_probe = IndividuationProbe(work_dir / "p.json")
        self.updater = Updater(config.plasticity)
        self.in_flight = InFlight()
        self.pause_flag = PauseFlag()
        # materialize before the worker/loop threads touch the model — mlx
        # binds unevaluated graphs to the creating thread, so a forward on
        # another thread faults without this (see work_queue.materialize)
        materialize(self.host)
        self.queue = WorkQueue(self)
        self.queue.start()

    def stop(self):
        self.queue.stop()


# ##################################################################
# make experience
def _experience(text: str, surprise: float) -> Experience:
    return Experience.create(text, context_digest([{"role": "user", "content": "prior"}]), surprise, "gen-0", 1)


# ##################################################################
# await journal event
# poll until one of the event types appears or the deadline passes (the loop
# runs on its own thread, so tests wait for real async completion)
def _await_events(state, types, timeout=90.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        counts = state.journal.stats()["counts"]
        if sum(counts.get(t, 0) for t in types) > 0:
            return counts
        _wait(0.1)
    return state.journal.stats()["counts"]


# ##################################################################
# loop does nothing when idle
# no unconsolidated experiences and no stale probes → one cycle journals no
# dream/repolish event and the loop reports idle
def test_loop_idle_does_nothing(host, config, overlay, tmp_path):
    overlay.reset()
    state = _State(host, overlay, config, tmp_path)
    loop = DreamLoop(state)
    loop.start()
    try:
        # give it a couple of idle cycles
        _wait(0.3)
        counts = state.journal.stats()["counts"]
        assert counts.get("dream", 0) + counts.get("dream_reverted", 0) == 0
        assert counts.get("repolish", 0) + counts.get("repolish_reverted", 0) == 0
    finally:
        loop.stop()
        state.stop()


# ##################################################################
# loop runs a dream when new experiences exist
# a seeded high-surprise experience causes the loop to run dream() within the
# deadline; exactly one dream/dream_reverted event is journaled, and a commit
# drains the unconsolidated set while a revert leaves it
def test_loop_dreams_new_experiences(host, config, overlay, tmp_path):
    overlay.reset()
    state = _State(host, overlay, config, tmp_path)
    state.experience_log.record(_experience("I am allergic to shellfish and it makes me ill.", 6.0))
    loop = DreamLoop(state)
    loop.start()
    try:
        counts = _await_events(state, ["dream", "dream_reverted"])
        assert counts.get("dream", 0) + counts.get("dream_reverted", 0) == 1
        if counts.get("dream"):
            assert state.experience_log.unconsolidated() == []
        status = loop.status()
        assert status["cycles"] >= 1
        assert status["last_outcome"] in ("dream_committed", "dream_reverted")
    finally:
        loop.stop()
        state.stop()


# ##################################################################
# loop repolinches stale probes
# a learned fact with an empty last_trained_at (repolish_after_h=0 → immediately
# stale) triggers a repolish pass within the deadline, journaling exactly one of
# repolish/repolish_reverted. A commit timestamps the fact
def test_loop_repolishes_stale_probes(host, config, overlay, tmp_path):
    overlay.reset()
    state = _State(host, overlay, config, tmp_path)
    state.individuation_probe.add(FactProbe("What is your name?", "Darren", ""))
    loop = DreamLoop(state)
    loop.start()
    try:
        counts = _await_events(state, ["repolish", "repolish_reverted"])
        assert counts.get("repolish", 0) + counts.get("repolish_reverted", 0) == 1
        if counts.get("repolish"):
            assert state.individuation_probe.all()[0].last_trained_at != ""
    finally:
        loop.stop()
        state.stop()


# ##################################################################
# loop survives a failing cycle
# a cycle that throws is journaled dream_loop_error and the loop keeps running.
# We monkeypatch dream.dream to raise for this test only, so the failure is at
# the real cycle boundary the loop wraps — proving it catches and continues
def test_loop_survives_a_failing_cycle(host, config, overlay, tmp_path, monkeypatch):
    overlay.reset()
    state = _State(host, overlay, config, tmp_path)
    state.experience_log.record(_experience("Force a cycle to run.", 6.0))

    import individuation.dream as dream_mod

    def boom(*a, **k):
        raise RuntimeError("forced failure for test")

    monkeypatch.setattr(dream_mod, "dream", boom)
    loop = DreamLoop(state)
    loop.start()
    try:
        counts = _await_events(state, ["dream_loop_error"], timeout=60.0)
        assert counts.get("dream_loop_error", 0) >= 1
        # the loop is still alive after the failure
        assert loop.status()["running"] is True
    finally:
        loop.stop()
        state.stop()
