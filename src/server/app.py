# =============================================================================
#  app — server wiring, shared state, and the two ways to run it
#  why: one place assembles the model host, plastic overlay, journal, replay,
#  checkpoints, updater and update worker into a single AppState the routers
#  read. build_state resolves the serving base (arg > current_base pointer >
#  config), restores learned overlay state on warm boot, and captures the first
#  canary baseline. Tests inject test-scoped Journal/Checkpoints/ReplayBuffer
#  (the REAL classes, just pointed at output/testing) — real integration.
# =============================================================================
from __future__ import annotations

import socket
import threading
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from common import identity, store
from common.config import load_config
from engine.model_host import ModelHost
from plasticity.adapter import attach_overlay
from plasticity.checkpoints import Checkpoints
from plasticity.guards import PauseFlag
from plasticity.journal import Journal
from plasticity.replay import ReplayBuffer
from plasticity.updater import Updater
from individuation.experience import ExperienceLog
from individuation.probe import IndividuationProbe
from individuation.surprise import SurpriseGate
from server import brain_api, feedback_api, openai_api
from server.work_queue import InFlight, WorkQueue, canary_enabled, materialize

POINTER_NAME = "current_base.json"
MODEL_ID = "engram/ornith-9b"


# ##################################################################
# pointer path / read pointer
# the current serving-base pointer written by consolidation; absent on a model
# that has never consolidated
def pointer_path():
    return store.data_root() / POINTER_NAME


def read_pointer():
    path = pointer_path()
    return store.read_json(path) if path.exists() else None


# ##################################################################
# resolve model path
# explicit --model wins, then the consolidation pointer, then the config default
def resolve_model_path(config, model_path, pointer) -> str:
    if model_path:
        return model_path
    if pointer and pointer.get("serve_path"):
        return pointer["serve_path"]
    return config.model.serve_path


# ##################################################################
# app state
# every shared object the routers and worker touch; held on app.state.engram
class AppState:
    def __init__(self, config) -> None:
        self.config = config
        self.host = None
        self.overlay = None
        self.updater = None
        self.journal = None
        self.replay = None
        self.checkpoints = None
        self.pause_flag = PauseFlag()
        self.in_flight = InFlight()
        self.queue = None
        self.token = ""
        self.pointer = None
        self.model_path = ""
        self.trace_of_call_id: dict = {}
        self.call_lock = threading.Lock()
        self.started_at = time.time()
        self.surprise_gate = SurpriseGate(config)
        self.experience_log = ExperienceLog()
        self.individuation_probe = IndividuationProbe()


# ##################################################################
# build state
# assemble the whole running brain: load the model, attach the overlay, restore
# any learned checkpoint, wire the queue, and capture a first-boot canary baseline
def build_state(config, model_path=None, journal=None, checkpoints=None, replay=None, start_queue=True) -> AppState:
    state = AppState(config)
    state.pointer = read_pointer()
    state.model_path = resolve_model_path(config, model_path, state.pointer)
    state.host = ModelHost(config, state.model_path)
    state.overlay = attach_overlay(state.host.model, config.plasticity)
    state.host.overlay = state.overlay
    state.journal = journal if journal is not None else Journal()
    state.replay = replay if replay is not None else ReplayBuffer()
    state.checkpoints = checkpoints if checkpoints is not None else Checkpoints(ring=config.guards.checkpoint_ring)
    _restore_checkpoint(state)
    materialize(state.host)
    state.updater = Updater(config.plasticity)
    state.token = identity.get_or_create_token()
    _rebuild_call_map(state)
    _maybe_baseline(state)
    state.queue = WorkQueue(state)
    if start_queue:
        state.queue.start()
    return state


# ##################################################################
# restore checkpoint
# reload the last known-good overlay if one exists; a cold start has none and
# the checkpoints layer signals that with RuntimeError, which is expected
def _restore_checkpoint(state: AppState) -> None:
    try:
        state.checkpoints.restore(state.overlay)
    except RuntimeError:
        return


# ##################################################################
# rebuild call map
# reconstruct the tool-call-id to trace-id map from recent traces so tool
# results referencing a previous turn can still be auto-scored after a restart
def _rebuild_call_map(state: AppState) -> None:
    from engine.trace import Trace

    for trace in Trace.list_recent(200):
        for call_id in trace.tool_call_ids:
            state.trace_of_call_id.setdefault(call_id, trace.trace_id)


# ##################################################################
# maybe baseline
# capture the original-base canary reference exactly once, and never when canary
# is disabled or a baseline already exists on disk
def _maybe_baseline(state: AppState) -> None:
    if not canary_enabled(state.config):
        return
    if any(store.canary_dir().iterdir()):
        return
    from evaluation import canary

    canary.baseline(state.host)


# ##################################################################
# lifespan
# on graceful shutdown (a deploy, an auto restart, SIGTERM) persist the current
# plastic overlay so learning done since the last periodic checkpoint is never
# lost — without this a restart reverts to the last checkpoint_every-th update
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    persist_on_shutdown(app.state.engram)


# ##################################################################
# persist on shutdown
# stop the worker, then checkpoint the overlay under the GPU lock and journal it;
# marked clean so the next boot restores exactly this learned state
def persist_on_shutdown(state: AppState) -> None:
    if state.overlay is None or state.checkpoints is None:
        return
    state.queue.stop()
    with state.host.gpu_lock:
        checkpoint_id = state.checkpoints.save(state.overlay, state.queue.accepted_updates, state.queue.last_clean)
    state.journal.record("checkpoint", checkpoint_id=checkpoint_id, reason="graceful shutdown")


# ##################################################################
# create app
# build the FastAPI application with its routers and shared state; used by both
# the production server and the test harness
def create_app(config=None, model_path=None, journal=None, checkpoints=None, replay=None, start_queue=True) -> FastAPI:
    config = config if config is not None else load_config()
    app = FastAPI(title="engram", lifespan=lifespan)
    app.state.engram = build_state(config, model_path, journal, checkpoints, replay, start_queue)
    app.include_router(openai_api.router)
    app.include_router(feedback_api.router)
    app.include_router(brain_api.router)
    return app


# ##################################################################
# stop state
# halt the background worker thread; used by test teardown so a stopped server's
# model can be released instead of pinned by a live thread
def stop_state(state: AppState) -> None:
    state.queue.stop()


# ##################################################################
# run server
# the production serve command: name the process, build the app, and hand the
# socket to uvicorn on the configured host/port
def run_server(args) -> int:
    import setproctitle

    setproctitle.setproctitle("engram")
    config = load_config()
    app = create_app(config, model_path=getattr(args, "model", None))
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")
    return 0


# ##################################################################
# free port / serve in thread
# start a real uvicorn server on an OS-assigned port in a background thread;
# the test harness drives genuine HTTP against this, no test client
def free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def serve_in_thread(app: FastAPI):
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="engram-uvicorn", daemon=True)
    thread.start()
    while not server.started:
        threading.Event().wait(0.05)
    return server, thread, f"http://127.0.0.1:{port}"
