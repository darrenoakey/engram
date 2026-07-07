# =============================================================================
#  trace_test — real serialization and store round-trips for Trace/Span
#  why: weight updates replay stored traces; a trace that does not survive a
#  gzip write/read cycle, or that leaks mx arrays into json, is unusable state
# =============================================================================
import os
import time

import mlx.core as mx

from engine.trace import Span, Trace


def _sample_trace() -> Trace:
    spans = [Span("think", 3, 6), Span("answer", 6, 9), Span("tool_call", 9, 12)]
    trace = Trace.create(
        token_ids=list(range(12)),
        gen_start=3,
        logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9],
        spans=spans,
        sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 40},
    )
    trace.tool_call_ids = {"call_abc": 2}
    trace.feedback = [{"reward": -0.5, "source": "tool"}]
    return trace


def test_span_list_roundtrip():
    span = Span("tool_call", 10, 14)
    assert Span.from_list(span.as_list()) == span


def test_trace_to_from_dict_roundtrip():
    trace = _sample_trace()
    rebuilt = Trace.from_dict(trace.to_dict())
    assert rebuilt == trace


def test_trace_create_converts_mx_arrays_to_plain_lists():
    trace = Trace.create(
        token_ids=mx.array([1, 2, 3, 4]),
        gen_start=1,
        logprobs=mx.array([-0.5, -1.5, -2.5]),
        spans=[Span("answer", 1, 4)],
        sampling={},
    )
    data = trace.to_dict()
    assert data["token_ids"] == [1, 2, 3, 4]
    assert data["logprobs"] == [-0.5, -1.5, -2.5]
    assert all(isinstance(t, int) for t in data["token_ids"])
    assert all(isinstance(x, float) for x in data["logprobs"])


def test_trace_save_load_roundtrip_real_store():
    trace = _sample_trace()
    path = trace.save()
    try:
        assert path.exists()
        assert Trace.load(trace.trace_id) == trace
    finally:
        path.unlink()


def test_list_recent_returns_newest_first():
    first = _sample_trace()
    second = _sample_trace()
    paths = [first.save(), second.save()]
    now = time.time()
    os.utime(paths[0], (now - 10, now - 10))
    os.utime(paths[1], (now, now))
    try:
        recent_ids = [t.trace_id for t in Trace.list_recent(20)]
        assert first.trace_id in recent_ids
        assert second.trace_id in recent_ids
        assert recent_ids.index(second.trace_id) < recent_ids.index(first.trace_id)
    finally:
        for path in paths:
            path.unlink()
