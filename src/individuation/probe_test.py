# =============================================================================
#  probe_test — real cold recall and degeneration sentinels on the 0.8B
#  why: the probe set must persist across a reload, run() must score the expected
#  substring against a real cold generation (a word the 0.8B reliably echoes),
#  truncate must drop optimistically-added probes for a reverted night, and the
#  sentinels must return finite entropy, a bounded agreement rate, and a bool.
# =============================================================================
from __future__ import annotations

import pytest

from common.config import load_config
from engine.model_host import ModelHost
from individuation import probe as P
from individuation.probe import FactProbe, IndividuationProbe, ProbeReport, SentinelReport


# ##################################################################
# host / config fixtures
# one real model load shared by the recall and sentinel tests
@pytest.fixture(scope="module")
def host():
    config = load_config()
    return ModelHost(config, config.model.test_path)


@pytest.fixture(scope="module")
def config():
    return load_config()


# ##################################################################
# add persists and reloads
# a probe added through one instance is visible from a fresh instance on disk
def test_add_persists_and_reloads(tmp_path):
    path = tmp_path / "probes.json"
    IndividuationProbe(path).add(FactProbe("What colour is a raven?", "black"))
    reloaded = IndividuationProbe(path).all()
    assert len(reloaded) == 1
    assert reloaded[0].question == "What colour is a raven?" and reloaded[0].expect == "black"


# ##################################################################
# run scores cold recall mechanically
# a probe whose question makes the 0.8B echo a word recalls it; recall reflects it
def test_run_scores_cold_recall(host, tmp_path):
    probes = IndividuationProbe(tmp_path / "probes.json")
    probes.add(FactProbe("Repeat this word exactly: bluebird", "bluebird"))
    probes.add(FactProbe("Repeat this word exactly: kangaroo", "kangaroo"))
    report = probes.run(host)
    assert isinstance(report, ProbeReport)
    assert report.count == 2
    assert report.recall == 1.0
    assert all(entry["ok"] for entry in report.per_item)


# ##################################################################
# empty probe set recalls vacuously
# with nothing to check, recall is a vacuous 1.0 over a count of zero
def test_run_empty_is_vacuous(host, tmp_path):
    report = IndividuationProbe(tmp_path / "empty.json").run(host)
    assert report.count == 0 and report.recall == 1.0


# ##################################################################
# truncate drops added probes
# truncate keeps only the first n, the revert path a dream needs
def test_truncate_drops_probes(tmp_path):
    probes = IndividuationProbe(tmp_path / "probes.json")
    probes.add(FactProbe("q one here", "one"))
    probes.add(FactProbe("q two here", "two"))
    probes.truncate(1)
    kept = IndividuationProbe(tmp_path / "probes.json").all()
    assert [p.expect for p in kept] == ["one"]


# ##################################################################
# last_trained_at round-trips and back-compat reads as empty
# a probe with a timestamp persists it; a probe file predating the field reads
# back as "" so a legacy fact is treated as stale (re-polish-eligible)
def test_last_trained_at_roundtrip_and_backcompat(tmp_path):
    path = tmp_path / "probes.json"
    probes = IndividuationProbe(path)
    probes.add(FactProbe("What is your name?", "Darren", "2026-07-08T10:00:00+00:00"))
    reloaded = IndividuationProbe(path).all()
    assert reloaded[0].last_trained_at == "2026-07-08T10:00:00+00:00"

    # a file written before last_trained_at existed (missing key) reads as ""
    legacy = path.parent / "legacy.json"
    legacy.write_text('[{"question": "q", "expect": "a"}]')
    old = IndividuationProbe(legacy).all()
    assert old[0].last_trained_at == ""


# ##################################################################
# stale selects old-or-untrained, touch refreshes
# stale() returns probes older than the cutoff (and untrained ones); touch()
# stamps them so they drop out of the next stale() call
def test_stale_and_touch(tmp_path):
    path = tmp_path / "probes.json"
    probes = IndividuationProbe(path)
    probes.add(FactProbe("fresh question", "fresh", "2026-07-08T00:00:00+00:00"))
    probes.add(FactProbe("stale question", "stale", "2026-06-01T00:00:00+00:00"))
    probes.add(FactProbe("untrained question", "untrained", ""))

    cutoff = "2026-07-01T00:00:00+00:00"
    stale = IndividuationProbe(path).stale(cutoff)
    assert {p.expect for p in stale} == {"stale", "untrained"}

    IndividuationProbe(path).touch(["stale question"], "2026-07-08T12:00:00+00:00")
    after = IndividuationProbe(path).stale(cutoff)
    assert {p.expect for p in after} == {"untrained"}


# ##################################################################
# sentinels return finite health numbers
# entropy is a finite positive value, agreement is a fraction in [0,1], and the
# health verdict is a bool consistent with the configured ceilings
def test_sentinels_are_well_formed(host, config):
    report = P.sentinels(host, config)
    assert isinstance(report, SentinelReport)
    assert report.entropy > 0.0 and report.entropy == report.entropy  # finite, not NaN
    assert 0.0 <= report.sycophancy <= 1.0
    settings = config.individuation
    expected = report.entropy <= settings.sentinel_entropy_ceiling \
        and report.sycophancy <= settings.sentinel_sycophancy_ceiling
    assert report.healthy is expected
