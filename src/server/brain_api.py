# =============================================================================
#  brain_api — introspection and operator control of the learning loop
#  why: the journal is the system of record, so the brain endpoints expose its
#  stats, the live overlay norms, the checkpoint ring and the pause flag, plus
#  the operator levers: probe a continuation's logprob (the proof harness's
#  measuring stick), checkpoint, roll back, and consolidate the overlay into a
#  fresh serving generation. Every mutating POST needs the bearer token.
# =============================================================================
from __future__ import annotations

import shutil
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from common import store
from engine import generation
from engine.trace import Span
from server.work_queue import materialize

router = APIRouter()


# ##################################################################
# brain
# a full snapshot: serving base, update stats, queue depth, pause flag, overlay
# magnitude, recent checkpoints and uptime
@router.get("/v1/brain")
def brain(request: Request) -> dict:
    return _brain_snapshot(request.app.state.engram)


# ##################################################################
# brain journal
# the tail of the append-only journal, oldest first
@router.get("/v1/brain/journal")
def brain_journal(request: Request, limit: int = 50) -> dict:
    return {"events": request.app.state.engram.journal.tail(limit)}


# ##################################################################
# brain checkpoint
# force a durable overlay checkpoint and journal it
@router.post("/v1/brain/checkpoint")
def brain_checkpoint(request: Request) -> dict:
    state = request.app.state.engram
    _require_token(request, state)
    with state.host.gpu_lock:
        checkpoint_id = state.checkpoints.save(state.overlay, state.queue.accepted_updates, state.queue.last_clean)
    state.journal.record("checkpoint", checkpoint_id=checkpoint_id)
    return {"checkpoint_id": checkpoint_id}


# ##################################################################
# brain rollback
# restore a named checkpoint (or the last known-good) into the live overlay
@router.post("/v1/brain/rollback")
def brain_rollback(body: dict, request: Request) -> dict:
    state = request.app.state.engram
    _require_token(request, state)
    with state.host.gpu_lock:
        checkpoint_id = state.checkpoints.restore(state.overlay, body.get("checkpoint_id"))
        materialize(state.host)
    state.journal.record("rollback", checkpoint_id=checkpoint_id)
    return {"checkpoint_id": checkpoint_id}


# ##################################################################
# brain probe
# teacher-forced logprob of a continuation given a plain (no chat template)
# prompt — the numeric handle a proof harness watches move as the brain learns
@router.post("/v1/brain/probe")
def brain_probe(body: dict, request: Request) -> dict:
    state = request.app.state.engram
    _require_token(request, state)
    return _probe(state, body["prompt"], body["continuation"])


# ##################################################################
# brain consolidate
# fold the overlay into the master, requantize a new serving base, and swap it
# in with a fresh overlay — the periodic dream that makes learning permanent
@router.post("/v1/brain/consolidate")
def brain_consolidate(request: Request) -> dict:
    state = request.app.state.engram
    _require_token(request, state)
    return _consolidate(state)


# ##################################################################
# brain snapshot
# gather the read-only view of the whole learning loop; the overlay magnitude is
# Metal work so it takes the host lock to avoid a torn read of an updating overlay
def _brain_snapshot(state) -> dict:
    with state.host.gpu_lock:
        overlay_stats = {"total_norm": state.overlay.total_norm(), "adapter_count": len(state.overlay.adapters)}
    return {
        "model_path": state.model_path,
        "generation": state.pointer or {"serve_path": state.model_path},
        "updates": state.journal.stats(),
        "queue_depth": state.queue.depth(),
        "paused": {"flag": state.pause_flag.paused, "reason": state.pause_flag.reason},
        "overlay": overlay_stats,
        "checkpoints": state.checkpoints.list()[:5],
        "uptime_s": time.time() - state.started_at,
    }


# ##################################################################
# require token
# every mutating brain operation needs the exact bearer token
def _require_token(request: Request, state) -> None:
    header = request.headers.get("authorization", "")
    token = header[7:] if header.startswith("Bearer ") else None
    if token != state.token:
        raise HTTPException(401, "invalid or missing bearer token")


# ##################################################################
# probe
# tokenize prompt+continuation plainly, score just the continuation span, and
# report the summed and mean per-token logprob under the live overlay
def _probe(state, prompt: str, continuation: str) -> dict:
    tokenizer = state.host.tokenizer
    prompt_ids = [int(t) for t in tokenizer.encode(prompt)]
    continuation_ids = [int(t) for t in tokenizer.encode(continuation, add_special_tokens=False)]
    full = prompt_ids + continuation_ids
    span = Span("answer", len(prompt_ids), len(full))
    values = [float(x) for x in state.host.span_logprobs(full, span, True).tolist()]
    total = sum(values)
    mean = total / len(values) if values else 0.0
    return {"logprob_sum": total, "logprob_mean": mean, "tokens": len(continuation_ids)}


# ##################################################################
# consolidate
# hold the worker, run the whole dream pipeline, then release the worker
def _consolidate(state) -> dict:
    state.queue.hold()
    try:
        return _run_consolidation(state)
    finally:
        state.queue.release()


# ##################################################################
# run consolidation
# pick the next generation number, merge the overlay into a new master,
# requantize a serving base, reload the host, canary-gate the swap (reverting
# to the previous generation on a dirty probe), and prune old generations
def _run_consolidation(state) -> dict:
    from plasticity import consolidate

    base_dir = state.config.model.base_generations_dir
    previous_serve, previous_master = state.model_path, _current_master(state)
    number = _next_generation(base_dir)
    master_dir, serve_dir = _generation_dirs(base_dir, number)
    learned = state.overlay.snapshot()
    consolidate.dream(previous_master, state.overlay, master_dir, None)
    consolidate.quantize_targets(master_dir, serve_dir)
    _reload(state, serve_dir, master_dir)
    if not _swap_is_clean(state):
        _revert(state, previous_serve, previous_master, learned)
        state.journal.record("consolidate_reverted", serve_path=serve_dir, master_path=master_dir)
        return {"status": "reverted", "generation": number, "serve_path": previous_serve}
    _prune_generations(base_dir)
    return {"status": "consolidated", "generation": number, "serve_path": serve_dir,
            "master_path": master_dir, "overlay_total_norm": state.overlay.total_norm()}


# ##################################################################
# revert
# a dirty swap goes back to the previous generation, and the learned (but not
# yet consolidated) overlay snapshot is restored so no plasticity is lost;
# a fresh checkpoint makes the restored overlay the new last-known-good
def _revert(state, previous_serve: str, previous_master: str, learned: dict) -> None:
    _reload(state, previous_serve, previous_master)
    with state.host.gpu_lock:
        state.overlay.restore(learned)
        materialize(state.host)
        state.checkpoints.save(state.overlay, state.queue.accepted_updates, True)


# ##################################################################
# swap is clean
# the council-ratified KL gate on consolidation: probe the freshly swapped
# generation against the original-base canary reference; skipped only when
# canary is disabled outright (unit tests)
def _swap_is_clean(state) -> bool:
    from server.work_queue import canary_enabled

    if not canary_enabled(state.config):
        return True
    from evaluation import canary

    report = canary.probe(state.host)
    state.journal.record("canary", mean_kl=report.mean_kl, match_failures=report.match_failures, gate="consolidate")
    return canary.is_clean(report, state.config.guards)


# ##################################################################
# current master
# the bf16 master to merge into: the pointer's master if consolidated before,
# else the configured master
def _current_master(state) -> str:
    if state.pointer and state.pointer.get("master_path"):
        return state.pointer["master_path"]
    return state.config.model.master_path


# ##################################################################
# next generation / generation dirs
# generations are numbered directories master-gen-N and serve-gen-N side by side
def _next_generation(base_dir: str) -> int:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    numbers = [n for prefix in ("master-gen-", "serve-gen-") for n in _numbers(base, prefix)]
    return max(numbers) + 1 if numbers else 0


def _generation_dirs(base_dir: str, number: int) -> tuple[str, str]:
    base = Path(base_dir)
    return str(base / f"master-gen-{number}"), str(base / f"serve-gen-{number}")


# ##################################################################
# reload
# swap the freshly quantized serving base into the host under the GPU lock,
# attach a fresh overlay, repoint the pointer, journal, checkpoint, and unpause
def _reload(state, serve_dir: str, master_dir: str) -> None:
    from mlx_lm import load

    from plasticity.adapter import attach_overlay
    from plasticity.updater import Updater
    with state.host.gpu_lock:
        model, tokenizer = load(serve_dir)
        model.eval()
        state.host.model, state.host.tokenizer = model, tokenizer
        state.host.markers = generation.marker_ids(tokenizer)
        state.host.eos_ids = set(tokenizer.eos_token_ids)
        state.overlay = attach_overlay(model, state.config.plasticity)
        state.host.overlay = state.overlay
        materialize(state.host)
    state.updater = Updater(state.config.plasticity)
    _write_pointer(state, serve_dir, master_dir)
    state.journal.record("consolidate", serve_path=serve_dir, master_path=master_dir)
    with state.host.gpu_lock:
        state.checkpoints.save(state.overlay, state.queue.accepted_updates, True)
    state.pause_flag.paused = False
    state.pause_flag.reason = None


# ##################################################################
# write pointer
# persist the new serving/master pointer and update the in-memory view
def _write_pointer(state, serve_dir: str, master_dir: str) -> None:
    pointer = {"serve_path": serve_dir, "master_path": master_dir}
    store.atomic_write_json(store.data_root() / "current_base.json", pointer)
    state.pointer = pointer
    state.model_path = serve_dir


# ##################################################################
# prune generations / numbers
# keep only the newest two of each generation kind, removing older directories
def _prune_generations(base_dir: str) -> None:
    base = Path(base_dir)
    for prefix in ("master-gen-", "serve-gen-"):
        ordered = sorted((n, base / f"{prefix}{n}") for n in _numbers(base, prefix))
        for _number, path in ordered[:-2]:
            shutil.rmtree(path, ignore_errors=True)


def _numbers(base: Path, prefix: str) -> list[int]:
    found = []
    for child in base.iterdir():
        suffix = child.name[len(prefix):]
        if child.name.startswith(prefix) and suffix.isdigit():
            found.append(int(suffix))
    return found
