# =============================================================================
#  proof_test — the end-to-end proof of life over real HTTP on the 0.8B
#  why: the proof harness is the definition of done, so it is exercised for real
#  against a live in-process server on the actual test model — reinforcement
#  raises a self-produced continuation's logprob, punishment lowers a different
#  one, and after a genuine server restart on the same on-disk state the learned
#  overlay reloads so the probes read the same values. Auth is enforced too.
# =============================================================================
from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from common.config import load_config
from common.identity import get_or_create_token
from evaluation import proof
from plasticity.checkpoints import Checkpoints
from plasticity.journal import Journal
from plasticity.replay import ReplayBuffer
from server.app import create_app, serve_in_thread, stop_state


# ##################################################################
# config
# real 0.8B, canary switched off, think tokens credited so the always-present
# think span trains, deterministic tiny generations, and a checkpoint on EVERY
# accepted update so a restart restores the exact final overlay
def _config():
    base = load_config()
    return replace(
        base,
        guards=replace(base.guards, canary_every=10 ** 9, checkpoint_every=1),
        plasticity=replace(base.plasticity, include_think_tokens=True),
        sampling=replace(base.sampling, temperature=0.0, top_p=0.0, top_k=0, max_tokens=40),
    )


# ##################################################################
# build / shutdown
# a real uvicorn server on an OS-assigned port with test-scoped journal,
# checkpoints and replay (the real classes, pointed at output/testing)
def _build(config, work: Path):
    app = create_app(
        config, model_path=config.model.test_path,
        journal=Journal(work / "journal.jsonl"),
        checkpoints=Checkpoints(work / "checkpoints", ring=config.guards.checkpoint_ring),
        replay=ReplayBuffer(work / "replay.json"),
    )
    handle, thread, url = serve_in_thread(app)
    return SimpleNamespace(app=app, handle=handle, thread=thread, url=url, state=app.state.engram)


def _shutdown(server) -> None:
    stop_state(server.state)
    server.handle.should_exit = True
    server.thread.join(timeout=10)


# ##################################################################
# pair probes
# probe the exact stored phase pairs so the restart comparison depends only on
# the reloaded overlay, never on re-eliciting the continuation
def _pair_probes(url: str, token: str, result) -> dict:
    reinforce = proof.probe(url, token, result.reinforcement["prompt"], result.reinforcement["continuation"])
    punish = proof.probe(url, token, result.punishment["prompt"], result.punishment["continuation"])
    return {"reinforce": reinforce["logprob_sum"], "punish": punish["logprob_sum"]}


# ##################################################################
# proven fixture
# run the whole proof once on a first server, capture the phase-pair probes,
# restart onto the same on-disk state, and re-capture; the second server stays
# live for the auth test and is torn down at the end
@pytest.fixture(scope="module")
def proven():
    config = _config()
    token = get_or_create_token()
    work = Path("output/testing") / f"proof-{uuid.uuid4().hex}"
    work.mkdir(parents=True, exist_ok=True)
    first = _build(config, work)
    result = proof.run_proof(first.url, token, rounds=4)
    pre = _pair_probes(first.url, token, result)
    _shutdown(first)
    second = _build(config, work)
    post = _pair_probes(second.url, token, result)
    yield SimpleNamespace(result=result, pre=pre, post=post, url=second.url, token=token)
    _shutdown(second)


# ##################################################################
# the proof passes and both learning directions are real
def test_proof_passes_with_all_verdicts(proven):
    result = proven.result
    assert result.passed
    assert result.reinforcement["after"] > result.reinforcement["before"]
    assert result.punishment["after"] < result.punishment["before"]
    assert result.stability["verdict"] is True


# ##################################################################
# the learned overlay survives a real server restart
def test_learned_overlay_survives_restart(proven):
    assert abs(proven.post["reinforce"] - proven.pre["reinforce"]) < 1e-3
    assert abs(proven.post["punish"] - proven.pre["punish"]) < 1e-3
    again = proof.persistence_probe(proven.url, proven.token)
    assert isinstance(again["reinforce"], float)
    assert isinstance(again["punish"], float)


# ##################################################################
# the probe endpoint refuses an unauthenticated request
def test_probe_requires_bearer_token(proven):
    body = {"prompt": "hello", "continuation": "world"}
    response = httpx.post(f"{proven.url}/v1/brain/probe", json=body, timeout=30)
    assert response.status_code == 401
