# =============================================================================
#  canary_test — real drift detection on the 0.8B qwen3_5 test model
#  why: prove the whole loop for real — an unperturbed baseline-then-probe reads
#  ~zero drift, a real overlay perturbation raises it, the expected substrings
#  survive on the 0.8B, is_clean honors the budget both ways, and re-baselining
#  overwrites cleanly. One module-scoped model load; small, serial, shared GPU.
# =============================================================================
from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import pytest

from common import store
from common.config import load_config
from engine.model_host import ModelHost
from evaluation import canary
from evaluation.canary import CanaryReport
from evaluation.canary_prompts import select
from plasticity.adapter import attach_overlay

# 4 answer-bearing echoes + 4 knowledge probes; the knowledge subset is reused
# for the perturbation probe so it costs only forwards (no extra generation)
SUBSET_IDS = ["follow_01", "follow_02", "follow_05", "follow_09",
              "know_01", "know_05", "know_08", "know_15"]
PERTURB_IDS = ["know_01", "know_05", "know_08", "know_15"]


# ##################################################################
# host fixture
# load the real quantized test model once; overlay stays None until the
# perturbation test attaches one (so the baseline captures the plain base)
@pytest.fixture(scope="module")
def host():
    config = load_config()
    return ModelHost(config, config.model.test_path)


# ##################################################################
# baselined fixture
# capture the reference state once, with the overlay absent (plain base)
@pytest.fixture(scope="module")
def baselined(host):
    return canary.baseline(host, select(SUBSET_IDS))


# ##################################################################
# clean report fixture
# one probe of the freshly baselined model, shared by the clean-drift and the
# expected-match tests so the generations run only once
@pytest.fixture(scope="module")
def clean_report(host, baselined):
    return canary.probe(host, select(SUBSET_IDS))


# ##################################################################
# baseline then probe is clean
# with no drift the truncated KL is ~zero and every probe reports a kl
def test_baseline_then_probe_is_clean(clean_report, baselined):
    assert set(baselined["ids"]) == set(SUBSET_IDS)
    assert len(clean_report.per_prompt) == len(SUBSET_IDS)
    assert all("kl" in entry and "id" in entry for entry in clean_report.per_prompt)
    assert clean_report.mean_kl < 1e-3
    assert all(entry["kl"] >= -1e-6 for entry in clean_report.per_prompt)


# ##################################################################
# expected substrings match on the 0.8b
# most of the answer-bearing probes still surface their substring within 32
# greedy tokens; match_failures is the count that did not
def test_expected_substrings_match(clean_report):
    answered = [entry for entry in clean_report.per_prompt if entry["matched"] is not None]
    assert len(answered) == 4
    matched = sum(1 for entry in answered if entry["matched"])
    assert matched >= 3
    assert clean_report.match_failures == len(answered) - matched


# ##################################################################
# perturbed overlay raises kl
# a real overlay with one adapter's B set nonzero moves the live distribution
# away from the stored base, so mean_kl rises strictly above the unperturbed run
def test_perturbed_overlay_raises_kl(host, baselined):
    config = load_config()
    host.overlay = attach_overlay(host.model, config.plasticity)
    host.overlay.reset()
    subset = select(PERTURB_IDS)
    unperturbed = canary.probe(host, subset)
    module = host.overlay.adapters[0][1]
    module.b = (mx.random.normal(module.b.shape) * 0.1).astype(mx.bfloat16)
    perturbed = canary.probe(host, subset)
    assert unperturbed.mean_kl < 1e-3
    assert perturbed.mean_kl > unperturbed.mean_kl


# ##################################################################
# is clean respects the budget both ways
# a report passes only when drift is within budget AND no expected answer is lost
def test_is_clean_respects_budget():
    guards = load_config().guards
    strict = replace(guards, canary_kl_budget=0.0)
    generous = replace(guards, canary_kl_budget=1.0)
    within = CanaryReport(mean_kl=0.1, per_prompt=[{"id": "x", "kl": 0.1, "matched": None}], match_failures=0)
    assert canary.is_clean(within, generous) is True
    assert canary.is_clean(within, strict) is False
    lost = CanaryReport(mean_kl=0.0, per_prompt=[{"id": "y", "kl": 0.0, "matched": False}], match_failures=1)
    assert canary.is_clean(lost, generous) is False


# ##################################################################
# baseline overwrites cleanly
# a second baseline run replaces the per-prompt files without error and the
# reloaded record has the right top-K shape
def test_baseline_overwrites_cleanly(host):
    subset = select(["know_02", "know_03"])
    first = canary.baseline(host, subset)
    record_before = store.read_json_gz(canary._record_path("know_02"))
    second = canary.baseline(host, subset)
    record_after = store.read_json_gz(canary._record_path("know_02"))
    assert first["ids"] == second["ids"] == ["know_02", "know_03"]
    assert record_after["token_ids"] == record_before["token_ids"]
    positions = len(record_after["token_ids"]) - record_after["gen_start"]
    assert len(record_after["topk_ids"]) == positions
    assert all(len(row) == canary.TOP_K for row in record_after["topk_ids"])
