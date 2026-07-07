# =============================================================================
#  canary_prompts_test — the frozen probe set is well formed
#  why: baselines and probes are only comparable if the set never drifts in
#  shape; guard the counts, uniqueness, brevity, and the EXPECTED/id linkage
# =============================================================================
from __future__ import annotations

import pytest

from evaluation.canary_prompts import EXPECTED, PROBES, select


# =============================================================================
#  sixty unique short prompts
#  why: 60 single-user-message probes, each ≤30 words, no duplicate ids
def test_probes_are_sixty_unique_short_prompts():
    assert len(PROBES) == 60
    ids = [pid for pid, _ in PROBES]
    assert len(set(ids)) == 60
    for _pid, messages in PROBES:
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert 0 < len(messages[0]["content"].split()) <= 30


# =============================================================================
#  twelve valid expected answers
#  why: exactly 12 answer-bearing probes, every key is a real probe id, every
#  expected substring is non-empty
def test_expected_is_twelve_valid_probe_ids():
    assert len(EXPECTED) == 12
    ids = {pid for pid, _ in PROBES}
    assert set(EXPECTED).issubset(ids)
    assert all(substring.strip() for substring in EXPECTED.values())


# =============================================================================
#  select subset in order
#  why: baseline/probe take a subset; select returns the pairs in PROBES order
#  and fails loudly on an unknown id rather than silently dropping it
def test_select_returns_subset_in_probes_order():
    chosen = select(["follow_01", "know_05"])
    assert [pid for pid, _ in chosen] == ["know_05", "follow_01"]
    assert chosen[0][1][0]["content"] == "What is the chemical symbol for gold?"


def test_select_rejects_unknown_id():
    with pytest.raises(ValueError):
        select(["know_05", "not_a_probe"])
