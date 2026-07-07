# =============================================================================
#  feedback_api — the authenticated human feedback endpoint (DESIGN.md §4)
#  why: /v1/feedback mutates the model's weights, so an open version would be a
#  weight-poisoning API; it requires the keychain-held bearer token, validates
#  the reward range, and only enqueues an update when plasticity is live.
# =============================================================================
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from engine.trace import Trace

router = APIRouter()


# ##################################################################
# feedback body
# the reward payload; reward is validated by hand so auth (401) is decided
# before the range check (422) rather than by schema-first parsing
class FeedbackBody(BaseModel):
    trace_id: str
    reward: float
    source: str | None = None
    note: str | None = None


# ##################################################################
# submit feedback
# authenticate, validate the reward, load the trace, refuse when the brain is
# blocked, then record the reward and enqueue a reward-kind update
@router.post("/v1/feedback")
def submit_feedback(body: FeedbackBody, request: Request) -> dict:
    state = request.app.state.engram
    _require_token(request, state)
    if not -1.0 <= body.reward <= 1.0:
        raise HTTPException(422, "reward must be within [-1, 1]")
    trace = _load_trace(body.trace_id)
    if not state.config.plasticity.enabled or state.pause_flag.paused:
        raise HTTPException(409, _blocked_reason(state))
    source = body.source or "user"
    trace.feedback.append({"reward": body.reward, "source": source, "note": body.note})
    trace.save()
    state.queue.enqueue({"kind": "reward", "trace_id": body.trace_id, "reward": body.reward, "source": source})
    return {"status": "queued", "trace_id": body.trace_id, "queue_depth": state.queue.depth()}


# ##################################################################
# require token
# reject any request without the exact bearer token the server holds
def _require_token(request: Request, state) -> None:
    header = request.headers.get("authorization", "")
    token = header[7:] if header.startswith("Bearer ") else None
    if token != state.token:
        raise HTTPException(401, "invalid or missing bearer token")


# ##################################################################
# load trace
# fetch the trace or map its absence to a 404
def _load_trace(trace_id: str) -> Trace:
    try:
        return Trace.load(trace_id)
    except FileNotFoundError as absent:
        raise HTTPException(404, f"unknown trace {trace_id}") from absent


# ##################################################################
# blocked reason
# explain the 409: disabled plasticity or a ceiling-triggered pause
def _blocked_reason(state) -> str:
    if not state.config.plasticity.enabled:
        return "plasticity is disabled"
    return state.pause_flag.reason or "plasticity is paused"
