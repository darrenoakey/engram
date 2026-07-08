# =============================================================================
#  store_test — real filesystem round-trips for every persistence helper
#  why: plasticity state that fails to survive a write/read cycle is fake state
# =============================================================================
import json
from pathlib import Path

from common import store


def test_atomic_write_and_read_json(tmp_path: Path):
    target = tmp_path / "nested" / "state.json"
    store.atomic_write_json(target, {"updates": 3, "reward": -0.5})
    assert store.read_json(target) == {"updates": 3, "reward": -0.5}
    leftovers = [p for p in target.parent.iterdir() if p.name.startswith(".state.json.")]
    assert leftovers == []


def test_json_gz_roundtrip(tmp_path: Path):
    target = tmp_path / "trace.json.gz"
    payload = {"token_ids": list(range(50)), "logprobs": [-0.1] * 50}
    store.write_json_gz(target, payload)
    assert store.read_json_gz(target) == payload


def test_jsonl_append_and_tail(tmp_path: Path):
    target = tmp_path / "journal.jsonl"
    for index in range(5):
        store.append_jsonl(target, {"event": index})
    assert store.read_jsonl(target) == [{"event": i} for i in range(5)]
    assert store.read_jsonl(target, limit=2) == [{"event": 3}, {"event": 4}]
    raw_lines = target.read_text().strip().split("\n")
    assert all(json.loads(line) is not None for line in raw_lines)


def test_read_jsonl_missing_file(tmp_path: Path):
    assert store.read_jsonl(tmp_path / "absent.jsonl") == []


def test_data_layout_dirs_exist():
    assert store.traces_dir().is_dir()
    assert store.checkpoints_dir().is_dir()
    assert store.canary_dir().is_dir()
    assert store.journal_path().parent.is_dir()


def test_set_data_root_redirects_every_path(tmp_path: Path):
    previous = store.data_root()
    try:
        store.set_data_root(tmp_path / "redirected")
        assert store.data_root() == (tmp_path / "redirected")
        assert str(store.traces_dir()).startswith(str(tmp_path))
        assert str(store.canary_dir()).startswith(str(tmp_path))
        assert str(store.journal_path()).startswith(str(tmp_path))
    finally:
        store.set_data_root(previous)
    assert store.data_root() == previous
