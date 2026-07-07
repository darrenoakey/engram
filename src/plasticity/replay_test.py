# =============================================================================
#  replay_test — real ring-buffer round-trips
#  why: the replay buffer feeds a positive span into every update and must
#  persist across restarts, evict oldest at the cap, and sample without replacement
# =============================================================================
from __future__ import annotations

from plasticity.replay import ReplayBuffer


# ##################################################################
# add sample and persist
# spans added to a real file survive a reload, and sampling returns members of
# the buffer without replacement
def test_add_sample_persist(tmp_path):
    path = tmp_path / "replay.json"
    buffer = ReplayBuffer(path)
    buffer.add([1, 2, 3])
    buffer.add([4, 5])
    buffer.add([6, 7, 8, 9])
    assert len(buffer) == 3
    picked = buffer.sample(2)
    assert len(picked) == 2
    assert all(span in buffer.spans for span in picked)
    reloaded = ReplayBuffer(path)
    assert len(reloaded) == 3
    assert [6, 7, 8, 9] in reloaded.spans


# ##################################################################
# cap evicts oldest
# adding past the cap drops the oldest spans, keeping the most recent
def test_cap_evicts_oldest(tmp_path):
    buffer = ReplayBuffer(tmp_path / "replay.json", cap=3)
    for i in range(5):
        buffer.add([i, i + 1])
    assert len(buffer) == 3
    assert [0, 1] not in buffer.spans
    assert [4, 5] in buffer.spans


# ##################################################################
# sample empty and seed
# an empty buffer samples to nothing; seeding bulk-loads cold-start spans
def test_sample_empty_and_seed(tmp_path):
    buffer = ReplayBuffer(tmp_path / "replay.json")
    assert buffer.sample(3) == []
    buffer.seed([[1, 2], [3, 4]])
    assert len(buffer) == 2
