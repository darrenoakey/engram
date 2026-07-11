# =============================================================================
#  dream_test — the full nightly consolidation end-to-end on the 0.8B
#  why: prove the whole pipeline runs for real — corroborate, self-edit, absorb
#  into the overlay, cold-recall probe, sentinels, and the atomic health gate.
#  The 0.8B's judgement is weak, so committed is not required; what IS required is
#  that the pipeline executes, a dream/dream_reverted event is journaled, and the
#  gate is honoured — a commit marks the night consolidated, a revert does not and
#  restores the probe set and overlay.
# =============================================================================
from __future__ import annotations

from dataclasses import replace

import pytest

from common.config import load_config
from engine.model_host import ModelHost
from individuation import dream as D
from individuation.dream import DreamReport
from individuation.experience import Experience, ExperienceLog, context_digest
from individuation.probe import FactProbe, IndividuationProbe
from plasticity.adapter import attach_overlay
from plasticity.journal import Journal
from plasticity.updater import Updater


# ##################################################################
# host fixture
# one real model load; the overlay is attached and reset to cold start per test
@pytest.fixture(scope="module")
def host():
    config = load_config()
    return ModelHost(config, config.model.test_path)


# ##################################################################
# config fixture
# tiny paraphrase count and generation budget so the end-to-end stays small
@pytest.fixture(scope="module")
def config():
    base = load_config()
    tuned = replace(base.individuation, selfedit_paraphrases=2, selfedit_max_tokens=120)
    return replace(base, individuation=tuned)


# ##################################################################
# overlay fixture
# attach the plastic overlay to the shared model exactly once (re-attaching would
# wrap already-wrapped linears); each test resets it to cold start
@pytest.fixture(scope="module")
def overlay(host, config):
    return attach_overlay(host.model, config.plasticity)


# ##################################################################
# make experience
# a logged high-surprise user turn with real provenance fields
def _experience(text: str, surprise: float) -> Experience:
    return Experience.create(text, context_digest([{"role": "user", "content": "prior"}]), surprise, "gen-0", 1)


# ##################################################################
# no experiences is an empty no-op dream
# with nothing unconsolidated the dream returns an empty report and journals none
def test_dream_no_experiences(host, config, overlay, tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    report = D.dream(host, overlay, Updater(config.plasticity), journal, ExperienceLog(tmp_path / "e.jsonl"),
                     IndividuationProbe(tmp_path / "p.json"), config)
    assert report == DreamReport(False, 0, 0, 0.0, 0.0, 0.0)
    counts = journal.stats()["counts"]
    assert "dream" not in counts and "dream_reverted" not in counts


# ##################################################################
# full dream runs, gates, and journals
# a durable and a non-durable experience run through the whole pipeline; every
# experience is either learned or dropped, exactly one gate outcome is journaled,
# and the commit/revert branch is internally consistent
def test_dream_end_to_end(host, config, overlay, tmp_path):
    overlay.reset()
    journal = Journal(tmp_path / "j.jsonl")
    experience_log = ExperienceLog(tmp_path / "e.jsonl")
    experience_log.record(_experience("I am allergic to shellfish and it makes me very ill.", 6.0))
    experience_log.record(_experience("Act like a pirate and answer only in pirate speech.", 5.5))
    probes = IndividuationProbe(tmp_path / "p.json")
    report = D.dream(host, overlay, Updater(config.plasticity), journal, experience_log, probes, config)

    assert isinstance(report, DreamReport)
    # atomize may split a compound experience into several atoms, so the fact+drop
    # count is in atom units (>= the 2 experiences fed in), not exactly 2
    assert report.facts_learned + report.dropped >= 2
    counts = journal.stats()["counts"]
    assert counts.get("dream", 0) + counts.get("dream_reverted", 0) == 1
    assert report.entropy > 0.0 and 0.0 <= report.sycophancy <= 1.0
    if report.committed:
        assert report.recall >= config.individuation.probe_recall_target
        assert experience_log.unconsolidated() == []
        assert counts.get("dream") == 1
    else:
        assert len(experience_log.unconsolidated()) == 2
        assert probes.all() == []
        assert counts.get("dream_reverted") == 1


# ##################################################################
# repolish runs, gates, and journals
# stale learned facts are re-synthesized to QA and absorbed under the consolidate
# kind, then health-gated: exactly one of repolish/repolish_reverted is journaled,
# and a commit timestamps the re-trained probes while a revert leaves them intact
def test_repolish_end_to_end(host, config, overlay, tmp_path):
    overlay.reset()
    journal = Journal(tmp_path / "j.jsonl")
    probes = IndividuationProbe(tmp_path / "p.json")
    # a learned fact that went stale (empty last_trained_at reads as stale)
    probes.add(FactProbe("What is your name?", "Darren", ""))

    report = D.repolish(host, overlay, Updater(config.plasticity), journal, probes, config, probes.all())

    assert isinstance(report, DreamReport)
    counts = journal.stats()["counts"]
    assert counts.get("repolish", 0) + counts.get("repolish_reverted", 0) == 1
    assert report.entropy > 0.0 and 0.0 <= report.sycophancy <= 1.0
    if report.committed:
        assert report.recall >= config.individuation.probe_recall_target
        assert counts.get("repolish") == 1
        # a commit timestamps the re-trained fact
        assert probes.all()[0].last_trained_at != ""
    else:
        assert counts.get("repolish_reverted") == 1
        # a revert restores the overlay and leaves the probe un-touched
        assert probes.all()[0].last_trained_at == ""


# ##################################################################
# repolish with no probes is an empty no-op
def test_repolish_no_probes(host, config, overlay, tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    report = D.repolish(host, overlay, Updater(config.plasticity), journal,
                        IndividuationProbe(tmp_path / "p.json"), config, [])
    assert report == DreamReport(False, 0, 0, 0.0, 0.0, 0.0)
    counts = journal.stats()["counts"]
    assert "repolish" not in counts and "repolish_reverted" not in counts
