# =============================================================================
#  model_host_test — real generation and teacher-forced logprobs on the 0.8B
#  why: the trace shape, the span partition, and the chosen-token logprob math
#  are the foundation every plasticity update builds on; all exercised for real
# =============================================================================
from dataclasses import replace

import mlx.core as mx
import pytest

from common.config import load_config
from engine.model_host import ModelHost
from engine.trace import Span


@pytest.fixture(scope="module")
def host():
    config = load_config()
    return ModelHost(config, config.model.test_path)


@pytest.fixture(scope="module")
def det_sampling():
    return replace(load_config().sampling, temperature=0.0, top_p=0.0, top_k=0, max_tokens=48)


@pytest.fixture(scope="module")
def generated(host, det_sampling):
    parts: list[str] = []
    messages = [{"role": "user", "content": "What is 2+2? Reply briefly."}]
    trace = host.generate(messages, sampling=det_sampling, on_token=parts.append)
    return trace, "".join(parts)


def test_generate_produces_well_formed_trace(generated):
    trace, _ = generated
    assert 1 <= trace.gen_start < len(trace.token_ids)
    assert len(trace.logprobs) == len(trace.token_ids) - trace.gen_start
    assert all(lp <= 1e-4 for lp in trace.logprobs)
    assert trace.sampling["temperature"] == 0.0


def test_generate_spans_partition_the_generated_region(generated):
    trace, _ = generated
    assert trace.spans[0].start == trace.gen_start
    assert trace.spans[-1].end == len(trace.token_ids)
    assert trace.spans[0].kind == "think"
    for earlier, later in zip(trace.spans, trace.spans[1:]):
        assert earlier.end == later.start


def test_on_token_stream_matches_decoded_generation(host, generated):
    trace, streamed = generated
    decoded = host.tokenizer.decode(trace.token_ids[trace.gen_start :])
    assert streamed.strip() == decoded.strip()


def test_span_logprobs_match_a_manual_forward(host, generated):
    trace, _ = generated
    end = min(trace.gen_start + 8, len(trace.token_ids))
    span = Span("answer", trace.gen_start, end)
    got = host.span_logprobs(trace.token_ids, span, adapters_enabled=True)
    assert got.shape == (end - span.start,)
    assert bool(mx.all(got <= 1e-4).item())
    expected = _manual_chosen_logprobs(host, trace.token_ids, span)
    assert bool(mx.allclose(got, expected, atol=1e-3).item())


def test_span_logprobs_flag_ignored_without_overlay(host, generated):
    trace, _ = generated
    span = Span("answer", trace.gen_start, trace.gen_start + 5)
    enabled = host.span_logprobs(trace.token_ids, span, adapters_enabled=True)
    disabled = host.span_logprobs(trace.token_ids, span, adapters_enabled=False)
    assert bool(mx.allclose(enabled, disabled, atol=0.0).item())


# ##################################################################
# with thinking disabled the qwen3_5 template omits the opening <think>, so the
# generation starts as an answer, not reasoning — the canary answer check needs
# this so a direct reply appears within its short token budget
def test_generate_without_thinking_answers_directly(host, det_sampling):
    messages = [{"role": "user", "content": "Repeat this word exactly: kangaroo"}]
    thinking = host.generate(messages, sampling=det_sampling, enable_thinking=True)
    direct = host.generate(messages, sampling=det_sampling, enable_thinking=False)
    assert thinking.spans[0].kind == "think"
    assert direct.spans[0].kind == "answer"
    assert "kangaroo" in host.tokenizer.decode(direct.token_ids[direct.gen_start :]).lower()


# =============================================================================
#  manual chosen logprobs
#  why: an independent recomputation of the teacher-forced math so the test
#  proves span_logprobs, rather than trusting the same helper it is testing
def _manual_chosen_logprobs(host, token_ids, span):
    logits = host.model(mx.array([token_ids[: span.end]]))[0].astype(mx.float32)
    values = []
    for position in range(span.start, span.end):
        row = logits[position - 1]
        logp = row - mx.logsumexp(row)
        values.append(logp[token_ids[position]])
    return mx.stack(values)
