# =============================================================================
#  selfedit_test — real paraphrase synthesis and teacher-forcing material
#  why: synthesize must return well-formed assistant-knowledge QA pairs on the
#  0.8B, and training_example must place the span EXACTLY over the answer tokens
#  (decoding them back to the answer) — that span is what the absorb update trains
#  on, so an off-by-one there would teach the wrong tokens.
# =============================================================================
from __future__ import annotations

from dataclasses import replace

import pytest

from common.config import load_config
from engine.model_host import ModelHost
from individuation import selfedit
from individuation.selfedit import QAPair


# ##################################################################
# host / config fixtures
# one real model load; a small paraphrase count keeps generations tiny
@pytest.fixture(scope="module")
def host():
    config = load_config()
    return ModelHost(config, config.model.test_path)


@pytest.fixture(scope="module")
def config():
    base = load_config()
    tuned = replace(base.individuation, selfedit_paraphrases=2, selfedit_max_tokens=140)
    return replace(base, individuation=tuned)


# ##################################################################
# synthesize returns well-formed pairs
# the model produces at least one QA pair; every pair has non-empty question and
# answer strings, and no more than the requested count are returned
def test_synthesize_returns_well_formed_pairs(host, config):
    pairs = selfedit.synthesize(host, "The user is allergic to shellfish.", config)
    assert isinstance(pairs, list)
    assert len(pairs) >= 1
    assert len(pairs) <= config.individuation.selfedit_paraphrases
    for pair in pairs:
        assert isinstance(pair, QAPair)
        assert pair.question.strip() != ""
        assert pair.answer.strip() != ""


# ##################################################################
# training example spans the answer tokens
# the returned span indexes exactly the appended answer tokens, gen_start sits at
# the prompt boundary, and decoding the span reproduces the answer
def test_training_example_spans_answer(host):
    qa = QAPair("What am I allergic to?", "You are allergic to shellfish.")
    token_ids, gen_start, (start, end) = selfedit.training_example(host, qa)
    assert start == gen_start
    assert end == len(token_ids)
    assert 1 <= gen_start < len(token_ids)
    decoded = host.tokenizer.decode(token_ids[start:end])
    assert decoded.strip() == qa.answer.strip()
