# =============================================================================
#  updater_test — real guarded updates on the 0.8B qwen3_5 model
#  why: this is the heart of engram. Prove that a positive signal raises the
#  span's logprob, a negative signal lowers it, and a KL-budget breach rolls the
#  overlay back bit-for-bit. Real gradient steps, real forwards, tiny spans.
# =============================================================================
from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import pytest
from mlx_lm import load

from common.config import load_config
from plasticity import losses
from plasticity.adapter import attach_overlay
from plasticity.journal import Journal
from plasticity.replay import ReplayBuffer
from plasticity.updater import Updater


# ##################################################################
# bundle fixture
# real model + overlay + a fixed token span used by every proof
@pytest.fixture(scope="module")
def bundle():
    config = load_config()
    model, tok = load(config.model.test_path)
    overlay = attach_overlay(model, config.plasticity)
    ids = tok.encode("The quick brown fox jumps over the lazy dog beside the calm river")
    gen_start = 6
    return model, overlay, config, ids, gen_start


# ##################################################################
# span sum logprob
# teacher-forced sum log p(target) over [start,end) with adapters enabled — the
# measuring stick for reinforcement and punishment
def _span_sum_logprob(model, ids, start, end):
    model.eval()
    logits = model(mx.array(ids[:end])[None])[0][start - 1:end - 1]
    logp = losses.log_softmax(logits)
    chosen = mx.take_along_axis(logp, mx.array(ids[start:end])[:, None], axis=-1)[:, 0]
    return float(chosen.sum())


# ##################################################################
# positive update raises logprob
# proof #1: reinforcing the span measurably increases its total logprob
def test_positive_update_raises_logprob(bundle, tmp_path):
    model, overlay, config, ids, gen_start = bundle
    overlay.reset()
    updater = Updater(config.plasticity)
    journal = Journal(tmp_path / "j.jsonl")
    before = _span_sum_logprob(model, ids, gen_start, len(ids))
    spans = [(gen_start, len(ids))]
    for _ in range(8):
        report = updater.apply(model, overlay, ids, gen_start, spans, 0.5, "reward", None, journal)
        assert report.accepted
    after = _span_sum_logprob(model, ids, gen_start, len(ids))
    assert after > before + 1e-3, f"{after} !> {before}"


# ##################################################################
# negative update lowers logprob
# proof #2: punishing the span measurably decreases its total logprob
def test_negative_update_lowers_logprob(bundle, tmp_path):
    model, overlay, config, ids, gen_start = bundle
    overlay.reset()
    updater = Updater(config.plasticity)
    journal = Journal(tmp_path / "j.jsonl")
    before = _span_sum_logprob(model, ids, gen_start, len(ids))
    spans = [(gen_start, len(ids))]
    for _ in range(8):
        updater.apply(model, overlay, ids, gen_start, spans, -0.5, "reward", None, journal)
    after = _span_sum_logprob(model, ids, gen_start, len(ids))
    assert after < before - 1e-3, f"{after} !< {before}"


# ##################################################################
# kl budget rejects and restores bit exact
# proof #4: an update that breaches the KL budget is rejected and the overlay
# is restored to the exact pre-update bytes; the rejection is journaled
def test_kl_budget_rejection_is_bit_exact(bundle, tmp_path):
    model, overlay, config, ids, gen_start = bundle
    overlay.reset()
    strict = replace(config.plasticity, update_kl_budget=-1.0)
    updater = Updater(strict)
    journal = Journal(tmp_path / "j.jsonl")
    pre = {k: mx.array(v) for k, v in overlay.trainable_parameters().items()}
    report = updater.apply(model, overlay, ids, gen_start, [(gen_start, len(ids))], 0.5, "reward", None, journal)
    assert report.accepted is False
    current = overlay.trainable_parameters()
    for key, value in pre.items():
        assert bool(mx.all(current[key] == value)), key
    assert journal.stats()["counts"].get("rejected_update") == 1


# ##################################################################
# reinforce kind fixes reward and consumes replay
# a reinforce update uses the fixed +0.3 reward, and a replay span in the buffer
# is consumed without breaking the step
def test_reinforce_and_replay(bundle, tmp_path):
    model, overlay, config, ids, gen_start = bundle
    overlay.reset()
    updater = Updater(config.plasticity)
    journal = Journal(tmp_path / "j.jsonl")
    replay = ReplayBuffer(tmp_path / "replay.json")
    replay.add(ids[gen_start:len(ids)])
    report = updater.apply(model, overlay, ids, gen_start, [(gen_start, len(ids))], 0.0, "reinforce", replay, journal)
    assert report.accepted
    assert report.reward == 0.3
    assert report.span_tokens == len(ids) - gen_start
