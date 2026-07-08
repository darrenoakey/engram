# =============================================================================
#  experience_test — real JSONL round-trip and consolidation flip
#  why: the experience log is the provenance spine; a record must survive a
#  reload byte-identically, unconsolidated() must select exactly the pending
#  turns, and mark_consolidated must flip only the named ids and persist it so a
#  processed night is never re-dreamed. No model needed — pure store integration.
# =============================================================================
from __future__ import annotations

from individuation.experience import Experience, ExperienceLog, context_digest


# ##################################################################
# make
# a small helper to mint an experience with distinguishable fields
def _make(text: str, surprise: float) -> Experience:
    return Experience.create(text, context_digest([{"role": "user", "content": "prior"}]),
                             surprise, "gen-0", 3)


# ##################################################################
# round trip preserves every field
# to_dict/from_dict is exact, and a fresh log reads back what was appended
def test_record_and_all_round_trip(tmp_path):
    log = ExperienceLog(tmp_path / "experience.jsonl")
    first = _make("I run every morning before work.", 4.2)
    second = _make("I strongly dislike loud open-plan offices.", 5.1)
    log.record(first)
    log.record(second)
    reread = ExperienceLog(tmp_path / "experience.jsonl").all()
    assert [e.id for e in reread] == [first.id, second.id]
    assert reread[0].from_dict(first.to_dict()) == first
    assert reread[1].user_text == second.user_text and reread[1].learner_version == 3


# ##################################################################
# recent returns the tail oldest first
def test_recent_returns_tail(tmp_path):
    log = ExperienceLog(tmp_path / "experience.jsonl")
    made = [_make(f"fact number {i} about me here", float(i)) for i in range(5)]
    for exp in made:
        log.record(exp)
    recent = log.recent(2)
    assert [e.id for e in recent] == [made[3].id, made[4].id]


# ##################################################################
# mark consolidated flips only named ids and persists
# the two-of-three flip survives a reload and unconsolidated drops the flipped
def test_mark_consolidated_persists(tmp_path):
    path = tmp_path / "experience.jsonl"
    log = ExperienceLog(path)
    made = [_make(f"durable preference {i} of mine", float(i)) for i in range(3)]
    for exp in made:
        log.record(exp)
    assert len(log.unconsolidated()) == 3
    log.mark_consolidated([made[0].id, made[2].id])
    reloaded = ExperienceLog(path)
    pending = reloaded.unconsolidated()
    assert [e.id for e in pending] == [made[1].id]
    by_id = {e.id: e for e in reloaded.all()}
    assert by_id[made[0].id].consolidated is True
    assert by_id[made[1].id].consolidated is False
    assert by_id[made[2].id].user_text == made[2].user_text
