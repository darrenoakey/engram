# =============================================================================
#  brain_api_test — introspection, probe, checkpoint/rollback, and consolidation
#  why: the brain endpoints are how the loop is observed and steered. The
#  consolidation test runs the FULL dream on the 0.8B (bf16 master built by
#  dequantizing the test model, like consolidate_test) and asserts the newly
#  quantized generation actually loads and serves. The live current_base pointer
#  is captured and restored so a heavy test never poisons real serving state.
# =============================================================================
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from mlx_lm.convert import convert

from common import store
from common.config import load_config
from common.identity import get_or_create_token
from plasticity.checkpoints import Checkpoints
from plasticity.journal import Journal
from plasticity.replay import ReplayBuffer
from server.app import create_app, serve_in_thread, stop_state


# ##################################################################
# config
# point the master at a bf16 fixture and generations at a temp dir so the real
# consolidation endpoint can run against the small model
def _config(master: Path, generations: Path):
    base = load_config()
    return replace(
        base,
        model=replace(base.model, master_path=str(master), base_generations_dir=str(generations)),
        guards=replace(base.guards, canary_every=10 ** 9),
        plasticity=replace(base.plasticity, include_think_tokens=True),
        sampling=replace(base.sampling, temperature=0.0, top_p=0.0, top_k=0, max_tokens=12),
    )


# ##################################################################
# server fixture
# build the bf16 master once, start the server, and on teardown restore the
# live current_base pointer to exactly its pre-test state
@pytest.fixture(scope="module")
def server(tmp_path_factory):
    base = load_config()
    work = tmp_path_factory.mktemp("brain")
    master = work / "master"
    convert(base.model.test_path, str(master), dequantize=True, dtype="bfloat16")
    config = _config(master, work / "generations")
    pointer = store.data_root() / "current_base.json"
    original = pointer.read_bytes() if pointer.exists() else None
    app = create_app(
        config, model_path=config.model.test_path,
        journal=Journal(work / "journal.jsonl"),
        checkpoints=Checkpoints(work / "checkpoints", ring=config.guards.checkpoint_ring),
        replay=ReplayBuffer(work / "replay.json"),
    )
    handle, thread, url = serve_in_thread(app)
    yield SimpleNamespace(app=app, url=url, token=get_or_create_token(), state=app.state.engram)
    stop_state(app.state.engram)
    handle.should_exit = True
    thread.join(timeout=10)
    _restore_pointer(pointer, original)


def _restore_pointer(pointer: Path, original) -> None:
    if original is None:
        pointer.unlink(missing_ok=True)
    else:
        pointer.write_bytes(original)


def _auth(server):
    return {"Authorization": f"Bearer {server.token}"}


# ##################################################################
# the brain snapshot carries every expected section
def test_brain_snapshot_has_expected_shape(server):
    data = httpx.get(f"{server.url}/v1/brain", timeout=60).json()
    for key in ("model_path", "generation", "updates", "queue_depth", "paused", "overlay", "checkpoints", "uptime_s"):
        assert key in data
    assert data["overlay"]["adapter_count"] > 0
    assert data["paused"]["flag"] is False
    # the individuation section surfaces the continuous-learner status
    ind = data["individuation"]
    assert "auto_dream" in ind and "loop" in ind


# ##################################################################
# the journal endpoint returns a list of events
def test_journal_endpoint_returns_events(server):
    data = httpx.get(f"{server.url}/v1/brain/journal?limit=5", timeout=30).json()
    assert isinstance(data["events"], list)


# ##################################################################
# probe reports a finite, non-positive continuation logprob
def test_probe_scores_a_continuation(server):
    body = {"prompt": "The capital of France is", "continuation": " Paris"}
    data = httpx.post(f"{server.url}/v1/brain/probe", json=body, headers=_auth(server), timeout=60).json()
    assert data["tokens"] > 0
    assert data["logprob_sum"] <= 1e-6
    assert math.isfinite(data["logprob_mean"])


# ##################################################################
# probe requires the bearer token
def test_probe_requires_token(server):
    body = {"prompt": "a", "continuation": " b"}
    assert httpx.post(f"{server.url}/v1/brain/probe", json=body, timeout=30).status_code == 401


# ##################################################################
# a checkpoint can be taken and rolled back by id
def test_checkpoint_then_rollback(server):
    saved = httpx.post(f"{server.url}/v1/brain/checkpoint", headers=_auth(server), timeout=60).json()["checkpoint_id"]
    assert saved
    body = {"checkpoint_id": saved}
    restored = httpx.post(f"{server.url}/v1/brain/rollback", json=body, headers=_auth(server),
                          timeout=60).json()["checkpoint_id"]
    assert restored == saved


# ##################################################################
# consolidation swaps in a fresh serving generation that loads and generates
def test_consolidate_swaps_to_a_new_generation(server):
    report = httpx.post(f"{server.url}/v1/brain/consolidate", headers=_auth(server), timeout=900).json()
    assert report["status"] == "consolidated"
    assert Path(report["serve_path"]).exists()
    brain = httpx.get(f"{server.url}/v1/brain", timeout=60).json()
    assert brain["generation"]["serve_path"] == report["serve_path"]
    chat = httpx.post(f"{server.url}/v1/chat/completions",
                      json={"messages": [{"role": "user", "content": "hi"}]}, timeout=120)
    assert chat.status_code == 200
    assert chat.json()["engram"]["trace_id"]
