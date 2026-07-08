# =============================================================================
#  surprise_test — real user-span extraction, cross-entropy, and the gate
#  why: the span must be the user's CONTENT tokens even when assistant history
#  makes an earlier render diverge from the full sequence; surprise must be a real
#  cross-entropy that is higher for less predictable text; and the gate's
#  percentile logic is pure, so it is proven deterministically on constructed data.
# =============================================================================
from __future__ import annotations

import pytest

from common.config import load_config
from engine.model_host import ModelHost
from individuation import surprise as S
from individuation.surprise import SurpriseGate


# ##################################################################
# host fixture
# one real 0.8B load shared by every model-driven test in this module
@pytest.fixture(scope="module")
def host():
    config = load_config()
    return ModelHost(config, config.model.test_path)


# ##################################################################
# span is the user content even with assistant history
# the qwen template renders assistant turns differently once a later user turn
# exists, so a naive prefix misaligns; the end-anchored extractor must not
def test_user_span_is_content_with_history(host):
    text = "I am allergic to shellfish and it is serious."
    messages = [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": text}]
    full, (start, end) = S.user_message_tokens(host, messages)
    assert start >= 1 and end <= len(full)
    assert host.tokenizer.decode(full[start:end]).strip() == text


# ##################################################################
# gate rejects short and non-user last turns
# a too-short user turn and an assistant-terminated conversation both return None
def test_user_message_tokens_rejects_short_and_non_user(host):
    assert S.user_message_tokens(host, [{"role": "user", "content": "ok"}]) is None
    assert S.user_message_tokens(host, [{"role": "user", "content": "tell me a long story please"},
                                        {"role": "assistant", "content": "sure"}]) is None


# ##################################################################
# surprise is a real cross-entropy, higher for less predictable text
# gibberish the model cannot anticipate carries more surprise than a common phrase
def test_surprise_higher_for_unpredictable(host):
    config = load_config()
    predictable = S.surprise(host, [{"role": "user", "content": "Thank you very much for all of your help."}], config)
    gibberish = S.surprise(host, [{"role": "user", "content": "The mitochondria zpqx flarn glorbnak wubble."}], config)
    assert predictable is not None and gibberish is not None
    assert predictable > 0.0 and gibberish > 0.0
    assert gibberish > predictable
    assert S.surprise(host, [{"role": "assistant", "content": "hello there friend"}], config) is None


# ##################################################################
# gate warms then fires only on high values
# during warmup nothing fires; once warm, a value above the rolling percentile
# fires and one below does not; threshold is a finite float
def test_surprise_gate_warms_then_fires():
    gate = SurpriseGate(load_config())
    warmup = [1.0, 1.1, 0.9, 1.2, 1.0, 0.8, 1.1, 1.0]
    assert not any(gate.consider(v) for v in warmup)
    assert gate.warm is True
    assert gate.consider(5.0) is True
    assert gate.consider(0.5) is False
    assert isinstance(gate.threshold(), float)


# ##################################################################
# gate stays silent before warmup
# an empty gate reports no threshold and never fires, even on a huge value
def test_surprise_gate_silent_before_warmup():
    gate = SurpriseGate(load_config())
    assert gate.threshold() is None
    assert gate.warm is False
    assert gate.consider(100.0) is False
