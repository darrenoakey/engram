# =============================================================================
#  journal_test — real append-only journal round-trips
#  why: the journal is the system of record; verify typed events persist, tail
#  reads them back, stats aggregates them, and bad types fail loudly
# =============================================================================
from __future__ import annotations

import pytest

from plasticity.journal import Journal


# ##################################################################
# record tail and stats
# events written to a real file come back through tail in order and aggregate
# correctly (counts, cumulative reward, last canary)
def test_record_tail_stats(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    journal.record("update", reward=0.3, accepted=True)
    journal.record("update", reward=-0.5, accepted=True)
    journal.record("canary", mean_kl=0.04)
    tail = journal.tail(2)
    assert len(tail) == 2
    assert tail[-1]["type"] == "canary"
    stats = journal.stats()
    assert stats["counts"]["update"] == 2
    assert abs(stats["cumulative_reward"] - (0.3 - 0.5)) < 1e-9
    assert stats["last_canary"]["mean_kl"] == 0.04
    assert stats["total_events"] == 3


# ##################################################################
# unknown type rejected
# an unrecognised event type is a programming error and must raise, not silently
# pollute the record
def test_unknown_type_raises(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    with pytest.raises(ValueError):
        journal.record("nonsense", value=1)


# ##################################################################
# empty journal stats
# stats on a fresh journal are well-defined
def test_empty_stats(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    stats = journal.stats()
    assert stats["total_events"] == 0
    assert stats["last_canary"] is None
    assert stats["cumulative_reward"] == 0.0
